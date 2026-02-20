"""Physical group assignment for Palace mesh generation.

This module handles assigning gmsh physical groups after geometry fragmentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import gmsh_utils

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack


def assign_physical_groups(
    kernel,
    metal_tags: dict,
    dielectric_tags: dict,
    port_tags: dict,
    port_info: list,
    geom_dimtags: list,
    geom_map: list,
    _stack: LayerStack,
    planar_conductors: bool = False,
) -> dict:
    """Assign physical groups after fragmenting.

    Args:
        kernel: gmsh OCC kernel
        metal_tags: Metal layer tags from add_metals()
        dielectric_tags: Dielectric material tags from add_dielectrics()
        port_tags: Port surface tags (may have multiple surfaces for CPW)
        port_info: Port metadata including type info
        geom_dimtags: Dimension tags from fragmentation
        geom_map: Geometry map from fragmentation
        _stack: Layer stack (unused; reserved for future material metadata)
        planar_conductors: If True, conductors are 2D PEC surfaces

    Returns:
        Dict with group info for config file generation:
        {
            "volumes": {material_name: {"phys_group": int, "tags": [int]}},
            "conductor_surfaces": {layer_name: {"phys_group": int, "tags": [int]}},
            "pec_surfaces": {layer_name: {"phys_group": int, "tags": [int]}},
            "port_surfaces": {port_name: {"phys_group": int, "tags": [int]} or
                            {"type": "cpw", "elements": [...]}},
            "boundary_surfaces": {"absorbing": {"phys_group": int, "tags": [int]}}
        }
    """
    groups = {
        "volumes": {},
        "conductor_surfaces": {},
        "pec_surfaces": {},
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
        # Handle planar conductors (2D PEC surfaces)
        if planar_conductors and tag_info["surfaces_xy"]:
            new_surface_tags = gmsh_utils.get_tags_after_fragment(
                tag_info["surfaces_xy"], geom_dimtags, geom_map, dimension=2
            )
            if new_surface_tags:
                phys_group = gmsh_utils.assign_physical_group(
                    2, new_surface_tags, f"{layer_name}_pec"
                )
                groups["pec_surfaces"][layer_name] = {
                    "phys_group": phys_group,
                    "tags": new_surface_tags,
                }
        
        # Handle volumetric conductors (finite conductivity)
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


__all__ = ["assign_physical_groups"]
