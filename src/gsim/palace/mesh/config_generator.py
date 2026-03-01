"""Palace configuration file generation.

This module handles generating Palace config.json and collecting mesh statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.models import DrivenConfig
    from gsim.palace.ports.config import PalacePort


def generate_palace_config(
    groups: dict,
    ports: list[PalacePort],
    port_info: list,
    stack: LayerStack,
    output_path: Path,
    model_name: str,
    fmax: float,
    driven_config: DrivenConfig | None = None,
) -> Path:
    """Generate Palace config.json file.

    Args:
        groups: Physical group information from mesh generation
        ports: List of PalacePort objects
        port_info: Port metadata list
        stack: Layer stack for material properties
        output_path: Output directory path
        model_name: Base name for output files
        fmax: Maximum frequency (Hz) - used as fallback if driven_config not provided
        driven_config: Optional DrivenConfig for frequency sweep settings

    Returns:
        Path to the generated config.json
    """
    from gsim.palace.ports.config import PortGeometry

    # Use driven_config if provided, otherwise fall back to legacy parameters
    if driven_config is not None:
        solver_driven = driven_config.to_palace_config()
    else:
        # Legacy behavior - compute from fmax
        freq_step = fmax / 40e9
        solver_driven = {
            "Samples": [
                {
                    "Type": "Linear",
                    "MinFreq": 1.0,  # 1 GHz
                    "MaxFreq": fmax / 1e9,
                    "FreqStep": freq_step,
                    "SaveStep": 0,
                }
            ],
            "AdaptiveTol": 0.02,
        }

    config: dict[str, object] = {
        "Problem": {
            "Type": "Driven",
            "Verbose": 3,
            "Output": f"output/{model_name}",
        },
        "Model": {
            "Mesh": f"{model_name}.msh",
            "L0": 1e-6,  # um
            "Refinement": {
                "UniformLevels": 0,
                "Tol": 1e-2,
                "MaxIts": 0,
            },
        },
        "Solver": {
            "Linear": {
                "Type": "Default",
                "KSPType": "GMRES",
                "Tol": 1e-6,
                "MaxIts": 400,
            },
            "Order": 2,
            "Device": "CPU",
            "Driven": solver_driven,
        },
    }

    # Build domains section
    materials: list[dict[str, object]] = []
    for material_name, info in groups["volumes"].items():
        mat_props = stack.materials.get(material_name, {})
        mat_entry: dict[str, object] = {"Attributes": [info["phys_group"]]}

        if material_name == "airbox":
            mat_entry["Permittivity"] = 1.0
            mat_entry["LossTan"] = 0.0
        else:
            mat_entry["Permittivity"] = mat_props.get("permittivity", 1.0)
            sigma = mat_props.get("conductivity", 0.0)
            if sigma > 0:
                mat_entry["Conductivity"] = sigma
            else:
                mat_entry["LossTan"] = mat_props.get("loss_tangent", 0.0)

        materials.append(mat_entry)

    config["Domains"] = {
        "Materials": materials,
        "Postprocessing": {"Energy": [], "Probe": []},
    }

    # Build boundaries section
    conductors: list[dict[str, object]] = []

    # Handle finite conductivity surfaces (volumetric conductors)
    for name, info in groups["conductor_surfaces"].items():
        # Extract layer name from "layer_xy" or "layer_z"
        layer_name = name.rsplit("_", 1)[0]
        layer = stack.layers.get(layer_name)
        if layer:
            mat_props = stack.materials.get(layer.material, {})
            conductors.append(
                {
                    "Attributes": [info["phys_group"]],
                    "Conductivity": mat_props.get("conductivity", 5.8e7),
                    "Thickness": layer.zmax - layer.zmin,
                }
            )

    # Handle PEC surfaces (planar conductors)
    pec_attrs = [info["phys_group"] for info in groups.get("pec_surfaces", {}).values()]

    lumped_ports: list[dict[str, object]] = []
    port_idx = 1

    for port in ports:
        port_key = f"P{port_idx}"
        if port_key in groups["port_surfaces"]:
            port_group = groups["port_surfaces"][port_key]

            if port.multi_element:
                # Multi-element port (CPW)
                if port_group.get("type") == "cpw":
                    elements = [
                        {
                            "Attributes": [elem["phys_group"]],
                            "Direction": elem["direction"],
                        }
                        for elem in port_group["elements"]
                    ]

                    lumped_ports.append(
                        {
                            "Index": port_idx,
                            "R": port.impedance,
                            "Excitation": port_idx if port.excited else False,
                            "Elements": elements,
                        }
                    )
            else:
                # Single-element port
                direction = (
                    "Z" if port.geometry == PortGeometry.VIA else port.direction.upper()
                )
                lumped_ports.append(
                    {
                        "Index": port_idx,
                        "R": port.impedance,
                        "Direction": direction,
                        "Excitation": port_idx if port.excited else False,
                        "Attributes": [port_group["phys_group"]],
                    }
                )
        port_idx += 1

    boundaries: dict[str, object] = {
        "Conductivity": conductors,
        "LumpedPort": lumped_ports,
    }

    # Add PEC boundaries if any exist
    if pec_attrs:
        boundaries["PEC"] = {"Attributes": pec_attrs}

    if "absorbing" in groups["boundary_surfaces"]:
        boundaries["Absorbing"] = {
            "Attributes": [groups["boundary_surfaces"]["absorbing"]["phys_group"]],
            "Order": 2,
        }

    config["Boundaries"] = boundaries

    # Write config file
    config_path = output_path / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=4)

    # Write port information file
    port_info_path = output_path / "port_information.json"
    port_info_struct = {"ports": port_info, "unit": 1e-6, "name": model_name}
    with port_info_path.open("w") as f:
        json.dump(port_info_struct, f, indent=4)

    return config_path


def collect_mesh_stats() -> dict:
    """Collect mesh statistics from gmsh after mesh generation.

    Must be called while gmsh is initialized and the mesh is generated.

    Returns:
        Dict with mesh statistics including:
        - bbox: Bounding box coordinates
        - nodes: Number of nodes
        - elements: Total element count
        - tetrahedra: Tet count
        - quality: Shape quality metrics (gamma)
        - sicn: Signed Inverse Condition Number
        - edge_length: Min/max edge lengths
        - groups: Physical group info
    """
    stats = {}

    # Get bounding box
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        stats["bbox"] = {
            "xmin": xmin,
            "ymin": ymin,
            "zmin": zmin,
            "xmax": xmax,
            "ymax": ymax,
            "zmax": zmax,
        }
    except Exception:
        pass

    # Get node count
    try:
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        stats["nodes"] = len(node_tags)
    except Exception:
        pass

    # Get element counts and collect tet tags for quality
    tet_tags = []
    try:
        element_types, element_tags, _ = gmsh.model.mesh.getElements()
        total_elements = sum(len(tags) for tags in element_tags)
        stats["elements"] = total_elements

        # Count tetrahedra (type 4) and save tags
        for etype, tags in zip(element_types, element_tags, strict=False):
            if etype == 4:  # 4-node tetrahedron
                stats["tetrahedra"] = len(tags)
                tet_tags = list(tags)
    except Exception:
        pass

    # Get mesh quality for tetrahedra
    if tet_tags:
        # Gamma: inscribed/circumscribed radius ratio (shape quality)
        try:
            qualities = gmsh.model.mesh.getElementQualities(tet_tags, "gamma")
            if len(qualities) > 0:
                stats["quality"] = {
                    "min": round(min(qualities), 3),
                    "max": round(max(qualities), 3),
                    "mean": round(sum(qualities) / len(qualities), 3),
                }
        except Exception:
            pass

        # SICN: Signed Inverse Condition Number (negative = invalid element)
        try:
            sicn = gmsh.model.mesh.getElementQualities(tet_tags, "minSICN")
            if len(sicn) > 0:
                sicn_min = min(sicn)
                invalid_count = sum(1 for s in sicn if s < 0)
                stats["sicn"] = {
                    "min": round(sicn_min, 3),
                    "mean": round(sum(sicn) / len(sicn), 3),
                    "invalid": invalid_count,
                }
        except Exception:
            pass

        # Edge lengths
        try:
            min_edges = gmsh.model.mesh.getElementQualities(tet_tags, "minEdge")
            max_edges = gmsh.model.mesh.getElementQualities(tet_tags, "maxEdge")
            if len(min_edges) > 0 and len(max_edges) > 0:
                stats["edge_length"] = {
                    "min": round(min(min_edges), 3),
                    "max": round(max(max_edges), 3),
                }
        except Exception:
            pass

    # Get physical groups with tags
    try:
        groups = {"volumes": [], "surfaces": []}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            entry = {"name": name, "tag": tag}
            if dim == 3:
                groups["volumes"].append(entry)
            elif dim == 2:
                groups["surfaces"].append(entry)
        stats["groups"] = groups
    except Exception:
        pass

    return stats


def write_config(
    mesh_result,
    stack: LayerStack,
    ports: list[PalacePort],
    driven_config: DrivenConfig | None = None,
) -> Path:
    """Write Palace config.json from a MeshResult.

    Use this to generate config separately after mesh().

    Args:
        mesh_result: Result from generate_mesh(write_config=False)
        stack: LayerStack for material properties
        ports: List of PalacePort objects
        driven_config: Optional DrivenConfig for frequency sweep settings

    Returns:
        Path to the generated config.json

    Raises:
        ValueError: If mesh_result has no groups data

    Example:
        >>> result = sim.mesh(output_dir, write_config=False)
        >>> config_path = write_config(result, stack, ports, driven_config)
    """
    if not mesh_result.groups:
        raise ValueError(
            "MeshResult has no groups data. Was it generated with write_config=False?"
        )

    config_path = generate_palace_config(
        groups=mesh_result.groups,
        ports=ports,
        port_info=mesh_result.port_info,
        stack=stack,
        output_path=mesh_result.output_dir,
        model_name=mesh_result.model_name,
        fmax=mesh_result.fmax,
        driven_config=driven_config,
    )

    # Update the mesh_result with the config path
    mesh_result.config_path = config_path

    return config_path


__all__ = [
    "collect_mesh_stats",
    "generate_palace_config",
    "write_config",
]
