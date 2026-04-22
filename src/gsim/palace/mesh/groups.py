"""Physical group assignment for Palace mesh generation.

This module builds the ``groups`` dict consumed by the config generator
from the ``pg_map`` produced by ``run_boolean_pipeline``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import gmsh_utils

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack

logger = logging.getLogger(__name__)


def assign_physical_groups(
    kernel,
    metal_tags: dict,
    dielectric_tags: dict,
    port_tags: dict,
    port_info: list,
    entities: list[gmsh_utils.Entity],
    pg_map: dict[str, int],
    _stack: LayerStack,
    pec_block_tags: dict | None = None,
) -> dict:
    """Build the ``groups`` dict from the boolean-pipeline result.

    Args:
        kernel: gmsh OCC kernel
        metal_tags: Metal layer tags from add_metals()
        dielectric_tags: Dielectric material tags from add_dielectrics()
        port_tags: Port surface tags (may have multiple surfaces for CPW)
        port_info: Port metadata including type info
        entities: Entity list used in run_boolean_pipeline
        pg_map: name -> physical-group tag returned by run_boolean_pipeline
        _stack: Layer stack used to identify via layers

    Returns:
        Dict with the same schema as before::

            {
                "volumes": {name: {"phys_group": int, "tags": [int]}},
                "conductor_surfaces": {name: {"phys_group": int, "tags": [int]}},
                "pec_surfaces": {name: {"phys_group": int, "tags": [int]}},
                "port_surfaces": {name: ...},
                "boundary_surfaces": {name: {"phys_group": int, "tags": [int]}},
            }
    """
    groups: dict[str, dict] = {
        "volumes": {},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }

    # Helper: entity name -> (phys_group, surface_tags)
    entity_by_name: dict[str, gmsh_utils.Entity] = {e.name: e for e in entities}

    # Build set of via layer names
    via_layers: set[str] = set()
    if _stack:
        via_layers = {
            n for n, layer in _stack.layers.items() if layer.layer_type == "via"
        }

    # --- Volumes (dielectrics + airbox) ---
    for material in dielectric_tags:
        entity = entity_by_name.get(material)
        pg = pg_map.get(material)
        if entity and pg is not None:
            vol_tags = [t for d, t in entity.dimtags if d == 3]
            if vol_tags:
                groups["volumes"][material] = {
                    "phys_group": pg,
                    "tags": vol_tags,
                }

    # --- Via volumes (3D material regions with conductivity) ---
    for layer_name in via_layers:
        entity = entity_by_name.get(layer_name)
        pg = pg_map.get(layer_name)
        if entity and pg is not None:
            vol_tags = [t for d, t in entity.dimtags if d == 3]
            if vol_tags:
                groups["volumes"][layer_name] = {
                    "phys_group": pg,
                    "tags": vol_tags,
                    "is_via": True,
                }

    # --- PEC surfaces (planar conductors) ---
    for layer_name, tag_info in metal_tags.items():
        if tag_info["surfaces_xy"]:
            pec_name = f"{layer_name}_pec"
            entity = entity_by_name.get(pec_name)
            pg = pg_map.get(pec_name)
            if entity and pg is not None:
                surf_tags = [t for d, t in entity.dimtags if d == 2]
                if surf_tags:
                    groups["pec_surfaces"][layer_name] = {
                        "phys_group": pg,
                        "tags": surf_tags,
                    }

    # --- PEC block surfaces ---
    if pec_block_tags:
        for block_name in pec_block_tags:
            for suffix in ("_xy", "_z"):
                name = f"{block_name}{suffix}"
                entity = entity_by_name.get(name)
                pg = pg_map.get(name)
                if entity and pg is not None:
                    surf_tags = [t for d, t in entity.dimtags if d == 2]
                    if surf_tags:
                        groups["pec_surfaces"][name] = {
                            "phys_group": pg,
                            "tags": surf_tags,
                        }

    # --- Volumetric conductor surfaces (finite thickness) ---
    for layer_name, tag_info in metal_tags.items():
        if tag_info["volumes"]:
            for suffix in ("_xy", "_z"):
                name = f"{layer_name}{suffix}"
                entity = entity_by_name.get(name)
                pg = pg_map.get(name)
                if entity and pg is not None:
                    surf_tags = [t for d, t in entity.dimtags if d == 2]
                    if surf_tags:
                        groups["conductor_surfaces"][name] = {
                            "phys_group": pg,
                            "tags": surf_tags,
                        }

    # --- Port surfaces ---
    for port_name, tags in port_tags.items():
        port_num = int(port_name[1:])
        info = next(
            (p for p in port_info if p["portnumber"] == port_num),
            None,
        )

        if info and info.get("type") == "cpw":
            element_phys_groups = []
            for i in range(len(tags)):
                elem_name = f"{port_name}_E{i}"
                entity = entity_by_name.get(elem_name)
                pg = pg_map.get(elem_name)
                if entity and pg is not None:
                    surf_tags = [t for d, t in entity.dimtags if d == 2]
                    if surf_tags:
                        element_phys_groups.append(
                            {
                                "phys_group": pg,
                                "tags": surf_tags,
                                "direction": info["elements"][i].get("direction"),
                            }
                        )
            groups["port_surfaces"][port_name] = {
                "type": "cpw",
                "elements": element_phys_groups,
            }
        else:
            pg = pg_map.get(port_name)
            entity = entity_by_name.get(port_name)
            if entity and pg is not None:
                surf_tags = [t for d, t in entity.dimtags if d == 2]
                if surf_tags:
                    groups["port_surfaces"][port_name] = {
                        "phys_group": pg,
                        "tags": surf_tags,
                    }

    # --- Boundary surfaces (outer faces labelled *__None by the pipeline) ---
    boundary_pgs: list[int] = [
        pg for name, pg in pg_map.items() if name.endswith("__None")
    ]
    if boundary_pgs:
        groups["boundary_surfaces"]["absorbing"] = {
            "phys_group": boundary_pgs,
            "tags": [],  # tags not needed; pg_map is authoritative
        }

    kernel.synchronize()
    return groups


__all__ = ["assign_physical_groups"]
