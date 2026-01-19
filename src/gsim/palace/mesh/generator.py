"""Mesh generator for Palace EM simulation.

This module generates meshes directly from palace-api data structures,
replacing the gds2palace backend with a cleaner implementation.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh

from . import gmsh_utils

if TYPE_CHECKING:
    from gsim.palace.ports.config import PalacePort
    from gsim.palace.stack import LayerStack

from gsim.palace.ports.config import PortGeometry

logger = logging.getLogger(__name__)


@dataclass
class GeometryData:
    """Container for geometry data extracted from component."""

    polygons: list  # List of (layer_num, pts_x, pts_y) tuples
    bbox: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    layer_bboxes: dict  # layer_num -> (xmin, ymin, xmax, ymax)


@dataclass
class MeshResult:
    """Result from mesh generation."""

    mesh_path: Path
    config_path: Path | None = None
    port_info: list = field(default_factory=list)


def extract_geometry(component, stack: LayerStack) -> GeometryData:
    """Extract polygon geometry from a gdsfactory component.

    Args:
        component: gdsfactory Component
        stack: LayerStack for layer mapping

    Returns:
        GeometryData with polygons and bounding boxes
    """
    polygons = []
    global_bbox = [math.inf, math.inf, -math.inf, -math.inf]
    layer_bboxes = {}

    # Get polygons from component
    polygons_by_index = component.get_polygons()

    # Build layer_index -> GDS tuple mapping
    layout = component.kcl.layout
    index_to_gds = {}
    for layer_index in range(layout.layers()):
        if layout.is_valid_layer(layer_index):
            info = layout.get_info(layer_index)
            index_to_gds[layer_index] = (info.layer, info.datatype)

    # Build GDS tuple -> layer number mapping
    gds_to_layernum = {}
    for layer_data in stack.layers.values():
        gds_tuple = tuple(layer_data.gds_layer)
        gds_to_layernum[gds_tuple] = gds_tuple[0]

    # Convert polygons
    for layer_index, polys in polygons_by_index.items():
        gds_tuple = index_to_gds.get(layer_index)
        if gds_tuple is None:
            continue

        layernum = gds_to_layernum.get(gds_tuple)
        if layernum is None:
            continue

        for poly in polys:
            # Convert klayout polygon to lists (nm -> um)
            points = list(poly.each_point_hull())
            if len(points) < 3:
                continue

            pts_x = [pt.x / 1000.0 for pt in points]
            pts_y = [pt.y / 1000.0 for pt in points]

            polygons.append((layernum, pts_x, pts_y))

            # Update bounding boxes
            xmin, xmax = min(pts_x), max(pts_x)
            ymin, ymax = min(pts_y), max(pts_y)

            global_bbox[0] = min(global_bbox[0], xmin)
            global_bbox[1] = min(global_bbox[1], ymin)
            global_bbox[2] = max(global_bbox[2], xmax)
            global_bbox[3] = max(global_bbox[3], ymax)

            if layernum not in layer_bboxes:
                layer_bboxes[layernum] = [xmin, ymin, xmax, ymax]
            else:
                bbox = layer_bboxes[layernum]
                bbox[0] = min(bbox[0], xmin)
                bbox[1] = min(bbox[1], ymin)
                bbox[2] = max(bbox[2], xmax)
                bbox[3] = max(bbox[3], ymax)

    return GeometryData(
        polygons=polygons,
        bbox=(global_bbox[0], global_bbox[1], global_bbox[2], global_bbox[3]),
        layer_bboxes=layer_bboxes,
    )


def _get_layer_info(stack: LayerStack, gds_layer: int) -> dict | None:
    """Get layer info from stack by GDS layer number."""
    for name, layer in stack.layers.items():
        if layer.gds_layer[0] == gds_layer:
            return {
                "name": name,
                "zmin": layer.zmin,
                "zmax": layer.zmax,
                "thickness": layer.zmax - layer.zmin,
                "material": layer.material,
                "type": layer.layer_type,
            }
    return None


def _add_metals(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
) -> dict:
    """Add metal and via geometries to gmsh.

    Creates extruded volumes for vias and shells (surfaces) for conductors.

    Returns:
        Dict with layer_name -> list of (surface_tags_xy, surface_tags_z) for
        conductors, or volume_tags for vias.
    """
    # layer_name -> {"volumes": [], "surfaces_xy": [], "surfaces_z": []}
    metal_tags = {}

    # Group polygons by layer
    polygons_by_layer = {}
    for layernum, pts_x, pts_y in geometry.polygons:
        if layernum not in polygons_by_layer:
            polygons_by_layer[layernum] = []
        polygons_by_layer[layernum].append((pts_x, pts_y))

    # Process each layer
    for layernum, polys in polygons_by_layer.items():
        layer_info = _get_layer_info(stack, layernum)
        if layer_info is None:
            continue

        layer_name = layer_info["name"]
        layer_type = layer_info["type"]
        zmin = layer_info["zmin"]
        thickness = layer_info["thickness"]

        if layer_type not in ("conductor", "via"):
            continue

        if layer_name not in metal_tags:
            metal_tags[layer_name] = {
                "volumes": [],
                "surfaces_xy": [],
                "surfaces_z": [],
            }

        for pts_x, pts_y in polys:
            # Create extruded polygon
            surfacetag = gmsh_utils.create_polygon_surface(kernel, pts_x, pts_y, zmin)
            if surfacetag is None:
                continue

            if thickness > 0:
                result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
                volumetag = result[1][1]

                if layer_type == "via":
                    # Keep vias as volumes
                    metal_tags[layer_name]["volumes"].append(volumetag)
                else:
                    # For conductors, get shell surfaces and remove volume
                    _, surfaceloops = kernel.getSurfaceLoops(volumetag)
                    if surfaceloops:
                        metal_tags[layer_name]["volumes"].append(
                            (volumetag, surfaceloops[0])
                        )
                    kernel.remove([(3, volumetag)])

    kernel.removeAllDuplicates()
    kernel.synchronize()

    return metal_tags


def _add_dielectrics(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    margin: float,
    air_margin: float,
) -> dict:
    """Add dielectric boxes and airbox to gmsh.

    Returns:
        Dict with material_name -> list of volume_tags
    """
    dielectric_tags = {}

    # Get overall geometry bounds
    xmin, ymin, xmax, ymax = geometry.bbox
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin

    # Track overall z range
    z_min_all = math.inf
    z_max_all = -math.inf

    # Sort dielectrics by z (top to bottom for correct layering)
    sorted_dielectrics = sorted(
        stack.dielectrics, key=lambda d: d["zmax"], reverse=True
    )

    # Add dielectric boxes
    offset = 0
    offset_delta = margin / 20

    for dielectric in sorted_dielectrics:
        material = dielectric["material"]
        d_zmin = dielectric["zmin"]
        d_zmax = dielectric["zmax"]

        z_min_all = min(z_min_all, d_zmin)
        z_max_all = max(z_max_all, d_zmax)

        if material not in dielectric_tags:
            dielectric_tags[material] = []

        # Create box with slight offset to avoid mesh issues
        box_tag = gmsh_utils.create_box(
            kernel,
            xmin - offset,
            ymin - offset,
            d_zmin,
            xmax + offset,
            ymax + offset,
            d_zmax,
        )
        dielectric_tags[material].append(box_tag)

        # Alternate offset to avoid coincident faces
        offset = offset_delta if offset == 0 else 0

    # Add surrounding airbox
    air_xmin = xmin - air_margin
    air_ymin = ymin - air_margin
    air_xmax = xmax + air_margin
    air_ymax = ymax + air_margin
    air_zmin = z_min_all - air_margin
    air_zmax = z_max_all + air_margin

    airbox_tag = kernel.addBox(
        air_xmin,
        air_ymin,
        air_zmin,
        air_xmax - air_xmin,
        air_ymax - air_ymin,
        air_zmax - air_zmin,
    )
    dielectric_tags["airbox"] = [airbox_tag]

    kernel.synchronize()

    return dielectric_tags


def _add_ports(
    kernel,
    ports: list[PalacePort],
    stack: LayerStack,
) -> tuple[dict, list]:
    """Add port surfaces to gmsh.

    Args:
        kernel: gmsh kernel
        ports: List of PalacePort objects (single or multi-element)
        stack: Layer stack

    Returns:
        (port_tags dict, port_info list)

    For single-element ports: port_tags["P{num}"] = [surface_tag]
    For multi-element ports: port_tags["P{num}"] = [surface_tag, surface_tag, ...]
    """
    port_tags = {}  # "P{num}" -> [surface_tag(s)]
    port_info = []
    port_num = 1

    for port in ports:
        if port.multi_element:
            # Multi-element port (CPW)
            if port.layer is None or port.centers is None or port.directions is None:
                continue
            target_layer = stack.layers.get(port.layer)
            if target_layer is None:
                continue

            z = target_layer.zmin
            hw = port.width / 2
            hl = (port.length or port.width) / 2

            # Determine axis from orientation
            angle = port.orientation % 360
            is_y_axis = 45 <= angle < 135 or 225 <= angle < 315

            surfaces = []
            for cx, cy in port.centers:
                if is_y_axis:
                    surf = gmsh_utils.create_port_rectangle(
                        kernel, cx - hw, cy - hl, z, cx + hw, cy + hl, z
                    )
                else:
                    surf = gmsh_utils.create_port_rectangle(
                        kernel, cx - hl, cy - hw, z, cx + hl, cy + hw, z
                    )
                surfaces.append(surf)

            port_tags[f"P{port_num}"] = surfaces

            port_info.append(
                {
                    "portnumber": port_num,
                    "Z0": port.impedance,
                    "type": "cpw",
                    "elements": [
                        {"surface_idx": i, "direction": port.directions[i]}
                        for i in range(len(port.centers))
                    ],
                    "width": port.width,
                    "length": port.length or port.width,
                    "zmin": z,
                    "zmax": z,
                }
            )

        elif port.geometry == PortGeometry.VIA:
            # Via port: vertical between two layers
            if port.from_layer is None or port.to_layer is None:
                continue
            from_layer = stack.layers.get(port.from_layer)
            to_layer = stack.layers.get(port.to_layer)
            if from_layer is None or to_layer is None:
                continue

            x, y = port.center
            hw = port.width / 2

            if from_layer.zmin < to_layer.zmin:
                zmin = from_layer.zmax
                zmax = to_layer.zmin
            else:
                zmin = to_layer.zmax
                zmax = from_layer.zmin

            # Create vertical port surface
            if port.direction in ("x", "-x"):
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, x, y - hw, zmin, x, y + hw, zmax
                )
            else:
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, x - hw, y, zmin, x + hw, y, zmax
                )

            port_tags[f"P{port_num}"] = [surfacetag]
            port_info.append(
                {
                    "portnumber": port_num,
                    "Z0": port.impedance,
                    "type": "via",
                    "direction": "Z",
                    "length": zmax - zmin,
                    "width": port.width,
                    "xmin": x - hw if port.direction in ("y", "-y") else x,
                    "xmax": x + hw if port.direction in ("y", "-y") else x,
                    "ymin": y - hw if port.direction in ("x", "-x") else y,
                    "ymax": y + hw if port.direction in ("x", "-x") else y,
                    "zmin": zmin,
                    "zmax": zmax,
                }
            )

        else:
            # Inplane port: horizontal on single layer
            if port.layer is None:
                continue
            target_layer = stack.layers.get(port.layer)
            if target_layer is None:
                continue

            x, y = port.center
            hw = port.width / 2
            z = target_layer.zmin

            hl = (port.length or port.width) / 2
            if port.direction in ("x", "-x"):
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, x - hl, y - hw, z, x + hl, y + hw, z
                )
                length = 2 * hl
                width = port.width
            else:
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, x - hw, y - hl, z, x + hw, y + hl, z
                )
                length = port.width
                width = 2 * hl

            port_tags[f"P{port_num}"] = [surfacetag]
            port_info.append(
                {
                    "portnumber": port_num,
                    "Z0": port.impedance,
                    "type": "lumped",
                    "direction": port.direction.upper(),
                    "length": length,
                    "width": width,
                    "xmin": x - hl if port.direction in ("x", "-x") else x - hw,
                    "xmax": x + hl if port.direction in ("x", "-x") else x + hw,
                    "ymin": y - hw if port.direction in ("x", "-x") else y - hl,
                    "ymax": y + hw if port.direction in ("x", "-x") else y + hl,
                    "zmin": z,
                    "zmax": z,
                }
            )

        port_num += 1

    kernel.synchronize()

    return port_tags, port_info


def _assign_physical_groups(
    kernel,
    metal_tags: dict,
    dielectric_tags: dict,
    port_tags: dict,
    port_info: list,
    geom_dimtags: list,
    geom_map: list,
    _stack: LayerStack,
) -> dict:
    """Assign physical groups after fragmenting.

    Args:
        kernel: gmsh kernel
        metal_tags: Metal layer tags
        dielectric_tags: Dielectric material tags
        port_tags: Port surface tags (may have multiple surfaces for CPW)
        port_info: Port metadata including type info
        geom_dimtags: Dimension tags from fragmentation
        geom_map: Geometry map from fragmentation
        _stack: Layer stack (unused; reserved for future material metadata)

    Returns:
        Dict with group info for config file generation
    """
    groups = {
        "volumes": {},
        "conductor_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }

    # Assign volume groups for dielectrics
    for material_name, tags in dielectric_tags.items():
        new_tags = gmsh_utils.get_tags_after_fragment(
            tags, geom_dimtags, geom_map, dimension=3
        )
        if new_tags:
            # Only take first N tags (same as original count)
            new_tags = new_tags[: len(tags)]
            phys_group = gmsh_utils.assign_physical_group(3, new_tags, material_name)
            groups["volumes"][material_name] = {
                "phys_group": phys_group,
                "tags": new_tags,
            }

    # Assign surface groups for conductors
    for layer_name, tag_info in metal_tags.items():
        if tag_info["volumes"]:
            all_xy_tags = []
            all_z_tags = []

            for item in tag_info["volumes"]:
                if isinstance(item, tuple):
                    _volumetag, surface_tags = item
                    # Get updated surface tags after fragment
                    new_surface_tags = gmsh_utils.get_tags_after_fragment(
                        surface_tags, geom_dimtags, geom_map, dimension=2
                    )

                    # Separate xy and z surfaces
                    for tag in new_surface_tags:
                        if gmsh_utils.is_vertical_surface(tag):
                            all_z_tags.append(tag)
                        else:
                            all_xy_tags.append(tag)

            if all_xy_tags:
                phys_group = gmsh_utils.assign_physical_group(
                    2, all_xy_tags, f"{layer_name}_xy"
                )
                groups["conductor_surfaces"][f"{layer_name}_xy"] = {
                    "phys_group": phys_group,
                    "tags": all_xy_tags,
                }

            if all_z_tags:
                phys_group = gmsh_utils.assign_physical_group(
                    2, all_z_tags, f"{layer_name}_z"
                )
                groups["conductor_surfaces"][f"{layer_name}_z"] = {
                    "phys_group": phys_group,
                    "tags": all_z_tags,
                }

    # Assign port surface groups
    for port_name, tags in port_tags.items():
        # Find corresponding port_info entry
        port_num = int(port_name[1:])  # "P1" -> 1
        info = next((p for p in port_info if p["portnumber"] == port_num), None)

        if info and info.get("type") == "cpw":
            # CPW port: create separate physical group for each element
            element_phys_groups = []
            for i, tag in enumerate(tags):
                new_tag_list = gmsh_utils.get_tags_after_fragment(
                    [tag], geom_dimtags, geom_map, dimension=2
                )
                if new_tag_list:
                    elem_name = f"{port_name}_E{i}"
                    phys_group = gmsh_utils.assign_physical_group(
                        2, new_tag_list, elem_name
                    )
                    element_phys_groups.append(
                        {
                            "phys_group": phys_group,
                            "tags": new_tag_list,
                            "direction": info["elements"][i]["direction"],
                        }
                    )

            groups["port_surfaces"][port_name] = {
                "type": "cpw",
                "elements": element_phys_groups,
            }
        else:
            # Regular single-element port
            new_tags = gmsh_utils.get_tags_after_fragment(
                tags, geom_dimtags, geom_map, dimension=2
            )
            if new_tags:
                phys_group = gmsh_utils.assign_physical_group(2, new_tags, port_name)
                groups["port_surfaces"][port_name] = {
                    "phys_group": phys_group,
                    "tags": new_tags,
                }

    # Assign boundary surfaces (from airbox)
    if "airbox" in groups["volumes"]:
        airbox_tags = groups["volumes"]["airbox"]["tags"]
        if airbox_tags:
            _, simulation_boundary = kernel.getSurfaceLoops(airbox_tags[0])
            if simulation_boundary:
                boundary_tags = list(next(iter(simulation_boundary)))
                phys_group = gmsh_utils.assign_physical_group(
                    2, boundary_tags, "Absorbing_boundary"
                )
                groups["boundary_surfaces"]["absorbing"] = {
                    "phys_group": phys_group,
                    "tags": boundary_tags,
                }

    kernel.synchronize()

    return groups


def _setup_mesh_fields(
    kernel,
    groups: dict,
    geometry: GeometryData,
    stack: LayerStack,
    refined_cellsize: float,
    max_cellsize: float,
) -> None:
    """Set up mesh refinement fields."""
    # Collect boundary lines from conductor surfaces
    boundary_lines = []
    for surface_info in groups["conductor_surfaces"].values():
        for tag in surface_info["tags"]:
            lines = gmsh_utils.get_boundary_lines(tag, kernel)
            boundary_lines.extend(lines)

    # Add port boundaries
    for surface_info in groups["port_surfaces"].values():
        if surface_info.get("type") == "cpw":
            # CPW port: get tags from each element
            for elem in surface_info["elements"]:
                for tag in elem["tags"]:
                    lines = gmsh_utils.get_boundary_lines(tag, kernel)
                    boundary_lines.extend(lines)
        else:
            # Regular port
            for tag in surface_info["tags"]:
                lines = gmsh_utils.get_boundary_lines(tag, kernel)
                boundary_lines.extend(lines)

    # Setup main refinement field
    field_ids = []
    if boundary_lines:
        field_id = gmsh_utils.setup_mesh_refinement(
            boundary_lines, refined_cellsize, max_cellsize
        )
        field_ids.append(field_id)

    # Add box refinement for dielectrics based on permittivity
    xmin, ymin, xmax, ymax = geometry.bbox
    field_counter = 10

    for dielectric in stack.dielectrics:
        material_name = dielectric["material"]
        material_props = stack.materials.get(material_name, {})
        permittivity = material_props.get("permittivity", 1.0)

        if permittivity > 1:
            local_max = max_cellsize / math.sqrt(permittivity)
            gmsh_utils.setup_box_refinement(
                field_counter,
                xmin,
                ymin,
                dielectric["zmin"],
                xmax,
                ymax,
                dielectric["zmax"],
                local_max,
                max_cellsize,
            )
            field_ids.append(field_counter)
            field_counter += 1

    if field_ids:
        gmsh_utils.finalize_mesh_fields(field_ids)


def _generate_palace_config(
    groups: dict,
    ports: list[PalacePort],
    port_info: list,
    stack: LayerStack,
    output_path: Path,
    model_name: str,
    fmax: float,
) -> Path:
    """Generate Palace config.json file."""
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
            "Driven": {
                "Samples": [
                    {
                        "Type": "Linear",
                        "MinFreq": 1e9 / 1e9,
                        "MaxFreq": fmax / 1e9,
                        "FreqStep": fmax / 40e9,
                        "SaveStep": 0,
                    }
                ],
                "AdaptiveTol": 2e-2,
            },
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


def generate_mesh(
    component,
    stack: LayerStack,
    ports: list[PalacePort],
    output_dir: str | Path,
    model_name: str = "palace",
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 300.0,
    margin: float = 50.0,
    air_margin: float = 50.0,
    fmax: float = 100e9,
    show_gui: bool = False,
) -> MeshResult:
    """Generate mesh for Palace EM simulation.

    Args:
        component: gdsfactory Component
        stack: LayerStack from palace-api
        ports: List of PalacePort objects (single and multi-element)
        output_dir: Directory for output files
        model_name: Base name for output files
        refined_mesh_size: Mesh size near conductors (um)
        max_mesh_size: Max mesh size in air/dielectric (um)
        margin: XY margin around design (um)
        air_margin: Air box margin (um)
        fmax: Max frequency for config (Hz)
        show_gui: Show gmsh GUI during meshing

    Returns:
        MeshResult with paths and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msh_path = output_dir / f"{model_name}.msh"

    # Extract geometry
    logger.info("Extracting geometry...")
    geometry = extract_geometry(component, stack)
    logger.info("  Polygons: %s", len(geometry.polygons))
    logger.info("  Bbox: %s", geometry.bbox)

    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)

    if "palace_mesh" in gmsh.model.list():
        gmsh.model.setCurrent("palace_mesh")
        gmsh.model.remove()
    gmsh.model.add("palace_mesh")

    kernel = gmsh.model.occ
    config_path: Path | None = None
    port_info: list = []

    try:
        # Add geometry
        logger.info("Adding metals...")
        metal_tags = _add_metals(kernel, geometry, stack)

        logger.info("Adding ports...")
        port_tags, port_info = _add_ports(kernel, ports, stack)

        logger.info("Adding dielectrics...")
        dielectric_tags = _add_dielectrics(kernel, geometry, stack, margin, air_margin)

        # Fragment geometry
        logger.info("Fragmenting geometry...")
        geom_dimtags, geom_map = gmsh_utils.fragment_all(kernel)

        # Assign physical groups
        logger.info("Assigning physical groups...")
        groups = _assign_physical_groups(
            kernel,
            metal_tags,
            dielectric_tags,
            port_tags,
            port_info,
            geom_dimtags,
            geom_map,
            stack,
        )

        # Setup mesh fields
        logger.info("Setting up mesh refinement...")
        _setup_mesh_fields(
            kernel, groups, geometry, stack, refined_mesh_size, max_mesh_size
        )

        # Show GUI if requested
        if show_gui:
            gmsh.fltk.run()

        # Generate mesh
        logger.info("Generating mesh...")
        gmsh.model.mesh.generate(3)

        # Save mesh
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_path))

        logger.info("Mesh saved: %s", msh_path)

        # Generate config
        logger.info("Generating Palace config...")
        config_path = _generate_palace_config(
            groups, ports, port_info, stack, output_dir, model_name, fmax
        )

    finally:
        gmsh.clear()
        gmsh.finalize()

    # Build result
    result = MeshResult(
        mesh_path=msh_path,
        config_path=config_path,
        port_info=port_info,
    )

    return result
