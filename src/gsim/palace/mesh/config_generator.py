"""Palace configuration file generation.

This module handles generating Palace config.json and collecting mesh statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import gmsh

from gsim.palace.ports.config import PortType

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.models import (
        DrivenConfig,
        EigenmodeConfig,
        ElectrostaticConfig,
        NumericalConfig,
    )
    from gsim.palace.models.ports import TerminalConfig
    from gsim.palace.ports.config import PalacePort


def generate_palace_config(
    groups: dict,
    ports: list[PalacePort],
    port_info: list,
    stack: LayerStack,
    output_path: Path,
    model_name: str,
    fmax: float,
    simulation_type: str = "driven",
    driven_config: DrivenConfig | None = None,
    eigenmode_config: EigenmodeConfig | None = None,
    numerical_config: NumericalConfig | None = None,
    absorbing_boundary: bool = True,
    periodic_axis: str | None = None,
    hints: dict[str, Any] | None = None,
    electrostatic_config: ElectrostaticConfig | None = None,
    terminals: list[TerminalConfig] | None = None,
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
        driven_config: Optional DrivenConfig for frequency sweep settings.
            When provided, material dispersion is evaluated at the center
            frequency ``(fmin + fmax) / 2`` of the sweep band.
        eigenmode_config: Optional EigenmodeConfig for eigenproblems settings
        absorbing_boundary: Whether to add absorbing (PML) boundary
        periodic_axis: Optional periodic axis identifier
        hints: Additional config hints merged into the JSON

    Returns:
        Path to the generated config.json
    """
    from gsim.palace.ports.config import PortGeometry

    if simulation_type not in (
        "driven",
        "eigenmode",
        "electrostatic",
        "electrostatics",
    ):
        raise ValueError(f"Unsupported simulation type: {simulation_type}")

    # Use driven_config if provided, otherwise fall back to legacy parameters
    if driven_config is not None:
        solver_driven = driven_config.to_palace_config()
    else:
        # Legacy behavior - compute from fmax
        freq_step = fmax / 40e9
        solver_driven = {
            "Samples": [
                {
                    "Type": "Driven",
                    "MinFreq": 1.0,  # 1 GHz
                    "MaxFreq": fmax / 1e9,
                    "FreqStep": freq_step,
                    "SaveStep": 0,
                }
            ],
            "AdaptiveTol": 0.02,
        }

    if eigenmode_config is not None:
        solver_eigenmode = eigenmode_config.to_palace_config()
    else:
        # Legacy behavior - compute from fmax
        solver_eigenmode = (
            {
                "N": 10,
                "Tol": 1.0e-6,
                "Target": fmax,
            },
        )

    solver_conf: dict[str, object]
    if numerical_config is not None:
        solver_conf = dict(numerical_config.to_solver_config())
    else:
        # Backward-compatible defaults for direct generate_palace_config() calls
        # that do not provide a NumericalConfig.
        solver_conf = {
            "Linear": {
                "Type": "Default",
                "KSPType": "GMRES",
                "Tol": 1e-6,
                "MaxIts": 1000,
            },
            "Order": 2,
            "Device": "CPU",
        }

    if simulation_type == "driven":
        solver_conf["Driven"] = solver_driven
    elif simulation_type == "eigenmode":
        solver_conf["Eigenmode"] = solver_eigenmode
    elif simulation_type in ("electrostatic", "electrostatics"):
        if electrostatic_config is not None:
            solver_conf["Electrostatic"] = electrostatic_config.to_palace_config()
        else:
            solver_conf["Electrostatic"] = {"Save": 0}

    config: dict[str, object] = {
        "Problem": {
            "Type": simulation_type.capitalize(),
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
        "Solver": solver_conf,
    }

    # Build domains section
    # Evaluate dispersion models at the center frequency of the sweep band
    stack_materials = stack.materials
    if driven_config is not None:
        from gsim.palace.materials import resolve_palace_materials_at_frequency

        stack_materials = resolve_palace_materials_at_frequency(
            stack.materials, driven_config.center_frequency
        )

    materials: list[dict[str, object]] = []
    for volume_name, info in groups["volumes"].items():
        material_name = volume_name
        is_via = info.get("is_via", False)
        is_shaped_dielectric = info.get("is_shaped_dielectric", False)

        if is_via or is_shaped_dielectric:
            layer = stack.layers.get(material_name)
            if layer is None:
                continue
            mat_props = stack_materials.get(layer.material, {})
        else:
            mat_props = stack_materials.get(material_name, {})

        mat_entry: dict[str, object] = {"Attributes": [info["phys_group"]]}

        if volume_name in {"airbox", "air"}:
            mat_entry["Permittivity"] = 1.0
            mat_entry["LossTan"] = 0.0
        elif is_via:
            sigma = mat_props.get("conductivity", 0.0)
            mat_entry["Permittivity"] = 1.0
            if isinstance(sigma, (int, float)) and sigma > 0:
                mat_entry["Conductivity"] = sigma
        elif is_shaped_dielectric:
            perm = mat_props.get("permittivity", 1.0)
            mat_entry["Permittivity"] = perm

            lt = mat_props.get("loss_tangent", 0.0)
            if isinstance(lt, list) or (isinstance(lt, (int, float)) and lt > 0):
                mat_entry["LossTan"] = lt
            else:
                mat_entry["LossTan"] = 0.0

            if "permeability" in mat_props:
                mat_entry["Permeability"] = mat_props["permeability"]

            if "material_axes" in mat_props:
                mat_entry["MaterialAxes"] = mat_props["material_axes"]
        else:
            perm = mat_props.get("permittivity", 1.0)
            mat_entry["Permittivity"] = perm

            if "permeability" in mat_props:
                mat_entry["Permeability"] = mat_props["permeability"]

            sigma = mat_props.get("conductivity", 0.0)
            lt = mat_props.get("loss_tangent", 0.0)
            if (isinstance(sigma, (int, float)) and sigma > 0) or isinstance(
                sigma, list
            ):
                mat_entry["Conductivity"] = sigma
            elif isinstance(lt, list) or (isinstance(lt, (int, float)) and lt > 0):
                mat_entry["LossTan"] = lt
            else:
                mat_entry["LossTan"] = 0.0

            if "material_axes" in mat_props:
                mat_entry["MaterialAxes"] = mat_props["material_axes"]

        materials.append(mat_entry)

    config["Domains"] = {
        "Materials": materials,
        "Postprocessing": {"Energy": [], "Probe": []},
    }

    # Build boundaries section
    conductors: list[dict[str, object]] = []

    for name, info in groups["conductor_surfaces"].items():
        # Extract layer name from "layer_xy" or "layer_z"
        layer_name = name.rsplit("_", 1)[0]
        layer = stack.layers.get(layer_name)
        if layer:
            mat_props = stack_materials.get(layer.material, {})
            conductors.append(
                {
                    "Attributes": [info["phys_group"]],
                    "Conductivity": mat_props.get("conductivity", 5.8e7),
                    "Thickness": layer.zmax - layer.zmin,
                }
            )

    # Handle PEC surfaces (planar conductors + PEC blocks)
    pec_attrs: list[int] = [
        info["phys_group"] for info in groups.get("pec_surfaces", {}).values()
    ]

    is_electrostatic = simulation_type in ("electrostatic", "electrostatics")

    if is_electrostatic and terminals:
        terminal_layer_names: set[str] = {t.layer for t in terminals}
        via_boundary = groups.get("via_boundary_surfaces", {})

        def _via_touches(via_name: str, conductor_layer_name: str) -> bool:
            """Z-range overlap (or touching) between a via and a conductor."""
            via = stack.layers.get(via_name)
            cond = stack.layers.get(conductor_layer_name)
            if via is None or cond is None:
                return False
            return via.zmin <= cond.zmax and via.zmax >= cond.zmin

        terminal_entries: list[dict[str, object]] = []
        assigned_pgs: set[int] = set()

        # Track which vias were attached to a terminal so we don't also
        # send their surfaces to ground.
        vias_on_terminal: set[str] = set()

        pec_surfaces = groups.get("pec_surfaces", {})

        for idx, terminal in enumerate(terminals, start=1):
            attrs: list[int] = []
            # Thick conductor shells (named "<layer>_xy" / "<layer>_z")
            for surf_name, surf_info in groups["conductor_surfaces"].items():
                surf_layer = surf_name.rsplit("_", 1)[0]
                if surf_layer == terminal.layer:
                    attrs.append(surf_info["phys_group"])
            # Planar (thin) conductor surfaces (keyed by layer name)
            if terminal.layer in pec_surfaces:
                attrs.append(pec_surfaces[terminal.layer]["phys_group"])
            # Vias touching this terminal's layer
            for via_name, via_pgs in via_boundary.items():
                if _via_touches(via_name, terminal.layer):
                    attrs.extend(via_pgs)
                    vias_on_terminal.add(via_name)

            assigned_pgs.update(attrs)
            terminal_entries.append(
                {
                    "Index": idx,
                    "Attributes": sorted(attrs),
                }
            )

        # Anything that wasn't assigned to a terminal becomes Ground.
        ground_attrs: list[int] = []
        for surf_info in groups["conductor_surfaces"].values():
            pg = surf_info["phys_group"]
            if pg not in assigned_pgs:
                ground_attrs.append(pg)
        for surf_info in pec_surfaces.values():
            pg = surf_info["phys_group"]
            if pg not in assigned_pgs:
                ground_attrs.append(pg)

        # Vias that touch a non-terminal conductor -> tie to ground so they
        # don't float (Palace's solver has no current-flow through volumes).
        for via_name, via_pgs in via_boundary.items():
            if via_name in vias_on_terminal:
                continue
            for cond_name in stack.layers or {}:
                if cond_name in terminal_layer_names:
                    continue
                cond = stack.layers.get(cond_name)
                if cond is None or cond.layer_type != "conductor":
                    continue
                if _via_touches(via_name, cond_name):
                    ground_attrs.extend(via_pgs)
                    break

        boundaries: dict[str, object] = {
            "Terminal": terminal_entries,
        }
        if ground_attrs:
            boundaries["Ground"] = {"Attributes": sorted(set(ground_attrs))}

    else:
        lumped_ports: list[dict[str, object]] = []
        wave_ports: list[dict[str, object]] = []
        port_idx = 1
        # Passive reactive ports are appended after all primary ports are assigned
        # indices so that their synthetic indices never clash.  We collect them here
        # and append them to lumped_ports once the primary loop finishes.
        passive_reactive_ports: list[dict[str, object]] = []

        for port in ports:
            if simulation_type != "driven":
                port.excited = False
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
                    if port.port_type == PortType.LUMPED:
                        direction = (
                            "Z"
                            if port.geometry == PortGeometry.VIA
                            else port.direction.upper()
                        )

                        has_reactive = (
                            port.resistance is not None
                            or (port.inductance is not None and port.inductance > 0)
                            or (port.capacitance is not None and port.capacitance > 0)
                        )

                        if simulation_type == "driven" and has_reactive:
                            # Driven simulations forbid L/C on excited ports.
                            # Emit the excited port with only R=impedance, then a
                            # separate passive (Active: false) lumped port carrying
                            # the reactive parameters on the same boundary surface.
                            # Palace allows shared attributes when Active is false.
                            port_entry: dict[str, object] = {
                                "Index": port_idx,
                                "R": port.impedance,
                                "Direction": direction,
                                "Excitation": port_idx if port.excited else False,
                                "Attributes": [port_group["phys_group"]],
                            }
                            lumped_ports.append(port_entry)

                            reactive_entry: dict[str, object] = {
                                # Placeholder index - will be replaced after the
                                # primary loop assigns all port_idx values.
                                "Index": None,
                                "Direction": direction,
                                "Attributes": [port_group["phys_group"]],
                                "Active": False,
                            }
                            if port.resistance is not None:
                                reactive_entry["R"] = port.resistance
                            if port.inductance is not None and port.inductance > 0:
                                reactive_entry["L"] = port.inductance
                            if port.capacitance is not None and port.capacitance > 0:
                                reactive_entry["C"] = port.capacitance
                            passive_reactive_ports.append(reactive_entry)
                        else:
                            # Eigenmode (or non-excited driven port): R/L/C on the
                            # same port entry is supported.
                            eigenmode_entry: dict[str, object] = {
                                "Index": port_idx,
                                "Direction": direction,
                                "Excitation": port_idx if port.excited else False,
                                "Attributes": [port_group["phys_group"]],
                            }
                            if port.impedance:
                                eigenmode_entry["R"] = port.impedance
                            if port.resistance is not None:
                                eigenmode_entry["R"] = port.resistance
                            if port.inductance is not None and port.inductance > 0:
                                eigenmode_entry["L"] = port.inductance
                            if port.capacitance is not None and port.capacitance > 0:
                                eigenmode_entry["C"] = port.capacitance
                            lumped_ports.append(eigenmode_entry)

                    elif port.port_type == PortType.WAVEPORT:
                        wave_ports.append(
                            {
                                "Index": port_idx,
                                "Mode": port.mode,
                                "Offset": port.offset,
                                "Excitation": port_idx if port.excited else False,
                                "Attributes": [port_group["phys_group"]],
                            }
                        )
            port_idx += 1

        # Assign unique indices to passive reactive ports now that all primary
        # indices are consumed (port_idx is one past the last primary index).
        synthetic_idx = port_idx
        for entry in passive_reactive_ports:
            entry["Index"] = synthetic_idx
            synthetic_idx += 1
        lumped_ports.extend(passive_reactive_ports)

        boundaries: dict[str, object] = {
            "Conductivity": conductors,
            "LumpedPort": lumped_ports,
            "WavePort": wave_ports,
        }

        # Add PEC boundaries if any exist
        if pec_attrs:
            boundaries["PEC"] = {"Attributes": pec_attrs}

    if "absorbing" in groups["boundary_surfaces"] and absorbing_boundary:
        absorbing_pg = groups["boundary_surfaces"]["absorbing"]["phys_group"]
        # phys_group may be a list (multiple __None groups) or a single int
        attrs = absorbing_pg if isinstance(absorbing_pg, list) else [absorbing_pg]
        boundaries["Absorbing"] = {
            "Attributes": attrs,
            "Order": 2,
        }

    if (
        simulation_type == "eigenmode"
        and eigenmode_config is not None
        and eigenmode_config.floquet
    ):
        axis = (periodic_axis or "").lower()
        if axis not in {"x", "y"}:
            raise ValueError(
                "Floquet eigenmode requires a periodic axis set in mesh(). "
                "Use mesh(periodic_axis='x') or mesh(periodic_axis='y')."
            )

        donor_info = groups["boundary_surfaces"].get("periodic_donor")
        receiver_info = groups["boundary_surfaces"].get("periodic_receiver")
        if donor_info is None or receiver_info is None:
            raise ValueError(
                "Floquet enabled but periodic donor/receiver boundaries were not "
                "found in the generated mesh."
            )

        donor_pg = donor_info.get("phys_group")
        receiver_pg = receiver_info.get("phys_group")
        periodic_donor_attrs = donor_pg if isinstance(donor_pg, list) else [donor_pg]
        periodic_receiver_attrs = (
            receiver_pg if isinstance(receiver_pg, list) else [receiver_pg]
        )

        if not periodic_donor_attrs or not periodic_receiver_attrs:
            raise ValueError("Floquet periodic boundary attributes are empty.")

        axis_lit: Literal["x", "y"] = "x" if axis == "x" else "y"

        floquet_vector = eigenmode_config.compute_floquet_wave_vector(
            periodic_axis=axis_lit,
            l0=float(config["Model"]["L0"]),
        )

        boundaries["Periodic"] = {
            "FloquetWaveVector": floquet_vector,
            "BoundaryPairs": [
                {
                    "DonorAttributes": sorted(periodic_donor_attrs),
                    "ReceiverAttributes": sorted(periodic_receiver_attrs),
                }
            ],
        }

    config["Boundaries"] = boundaries

    # Merge any extra hints into the config
    if hints:
        config.update(hints)

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
    simulation_type: str = "driven",
    driven_config: DrivenConfig | None = None,
    eigenmode_config: EigenmodeConfig | None = None,
    numerical_config: NumericalConfig | None = None,
    absorbing_boundary: bool = True,
    hints: dict[str, Any] | None = None,
    electrostatic_config: ElectrostaticConfig | None = None,
    terminals: list[TerminalConfig] | None = None,
) -> Path:
    """Write Palace config.json from a MeshResult.

    Use this to generate config separately after mesh().

    Args:
        mesh_result: Result from generate_mesh(write_config=False)
        stack: LayerStack for material properties
        ports: List of PalacePort objects
        driven_config: Optional DrivenConfig for frequency sweep settings.
            When provided, material dispersion is evaluated at the center
            frequency of the sweep band.
        eigenmode_config: Optional EigenmodeConfig for eigenproblems settings
        absorbing_boundary: Whether to add absorbing (PML) boundary
        hints: Additional config hints merged into the JSON

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
        simulation_type=simulation_type,
        driven_config=driven_config,
        eigenmode_config=eigenmode_config,
        numerical_config=numerical_config,
        absorbing_boundary=absorbing_boundary,
        periodic_axis=mesh_result.periodic_axis,
        hints=hints,
        electrostatic_config=electrostatic_config,
        terminals=terminals,
    )

    # Update the mesh_result with the config path
    mesh_result.config_path = config_path

    return config_path


__all__ = [
    "collect_mesh_stats",
    "generate_palace_config",
    "write_config",
]
