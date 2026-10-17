"""Materialize simulation layers without GDS target-layer aliasing.

GDS stores concrete mask polygons, while a GDSFactory ``LayerStack`` can
describe physical layers with Boolean expressions.  Multiple physical levels
may share the same declared output tuple, so materializing them onto those
targets loses their identities.  This module evaluates each level separately
and writes it to a deterministic, simulation-only GDS tuple.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhysicalLayerExport:
    """A component and stack pair with one GDS tuple per physical level."""

    component: Any
    stack: Any
    gdsfactory_stack: Any
    layer_map: dict[str, tuple[int, int]]


def _temporary_layer_candidates() -> Iterator[tuple[int, int]]:
    """Yield valid GDS tuples in a stable order from the top of the range."""
    for datatype in range(65_536):
        for layer_number in range(65_535, -1, -1):
            yield layer_number, datatype


def allocate_physical_layers(
    level_names: list[str],
    used_layers: set[tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    """Allocate deterministic tuples that do not collide with source masks."""
    candidates = _temporary_layer_candidates()
    unavailable = set(used_layers)
    allocated: dict[str, tuple[int, int]] = {}

    for level_name in level_names:
        target = next(
            candidate for candidate in candidates if candidate not in unavailable
        )
        allocated[level_name] = target
        unavailable.add(target)

    return allocated


def _active_gdsfactory_stack(stack: Any) -> Any | None:
    """Return the matching active-PDK stack, if *stack* came from that PDK."""
    import gdsfactory as gf

    try:
        pdk = gf.get_active_pdk()
    except ValueError:
        return None

    if getattr(stack, "pdk_name", None) != getattr(pdk, "name", None):
        return None
    return getattr(pdk, "layer_stack", None)


def _matching_layer_level(
    layer_name: str,
    layer: Any,
    gdsfactory_stack: Any | None,
) -> Any | None:
    """Find the authoritative GDSFactory level corresponding to a gsim layer."""
    if gdsfactory_stack is None:
        return None

    level = gdsfactory_stack.layers.get(layer_name)
    if level is None:
        return None

    from gsim.common.stack._layer_utils import get_gds_layer_tuple

    if get_gds_layer_tuple(level) != tuple(layer.gds_layer):
        return None
    return level


def _evaluate_level_region(component: Any, level: Any) -> Any:
    """Evaluate one GDSFactory physical level from its source expression."""
    from gsim.common.polygon import layer_level_region

    return layer_level_region(component, level)


def _source_layer_region(component: Any, gds_layer: tuple[int, int]) -> Any:
    """Read one concrete source-mask layer recursively from *component*."""
    from kfactory import kdb

    layer_index = component.kcl.layer(*gds_layer)
    return kdb.Region(component.kdb_cell.begin_shapes_rec(layer_index))


def _build_gdsfactory_stack(
    stack: Any,
    layer_map: dict[str, tuple[int, int]],
    source_stack: Any | None,
) -> Any:
    """Create a direct-layer GDSFactory stack for the materialized component."""
    from gdsfactory.technology import LayerLevel, LayerStack

    levels = {}
    for layer_name, layer in stack.layers.items():
        source_level = (
            source_stack.layers.get(layer_name) if source_stack is not None else None
        )
        mesh_order = getattr(source_level, "mesh_order", 3)
        levels[layer_name] = LayerLevel(
            layer=layer_map[layer_name],
            thickness=layer.thickness,
            zmin=layer.zmin,
            material=layer.material,
            sidewall_angle=layer.sidewall_angle,
            mesh_order=mesh_order,
        )
    return LayerStack(layers=levels)


def materialize_physical_layers(component: Any, stack: Any) -> PhysicalLayerExport:
    """Evaluate every simulation layer onto its own native GDS tuple.

    Layers originating in the active PDK are evaluated from their authoritative
    ``LogicalLayer`` or ``DerivedLayer`` expression.  Custom/YAML stack entries
    fall back to reading their concrete ``gds_layer`` directly.  The returned
    component and both returned stacks must be consumed together.
    """
    import gdsfactory as gf

    source_stack = _active_gdsfactory_stack(stack)
    used_layers = {tuple(layer) for layer in component.layers}
    layer_map = allocate_physical_layers(list(stack.layers), used_layers)

    materialized = gf.Component()
    remapped_stack = stack.model_copy(deep=True)

    for layer_name, layer in stack.layers.items():
        source_level = _matching_layer_level(layer_name, layer, source_stack)
        if source_level is None:
            region = _source_layer_region(component, tuple(layer.gds_layer))
        else:
            region = _evaluate_level_region(component, source_level)

        target = layer_map[layer_name]
        target_index = materialized.kcl.layer(*target)
        materialized.shapes(target_index).insert(region)
        remapped_stack.layers[layer_name].gds_layer = target

    materialized.add_ports(component.ports)
    materialized.copy_child_info(component)

    physical_gdsfactory_stack = _build_gdsfactory_stack(
        remapped_stack,
        layer_map,
        source_stack,
    )
    return PhysicalLayerExport(
        component=materialized,
        stack=remapped_stack,
        gdsfactory_stack=physical_gdsfactory_stack,
        layer_map=layer_map,
    )


__all__ = [
    "PhysicalLayerExport",
    "allocate_physical_layers",
    "materialize_physical_layers",
]
