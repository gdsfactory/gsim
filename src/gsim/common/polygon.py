"""Polygon extraction and processing for layered GDS components.

Ported from gplugins.common.base_models.component to avoid the gplugins
dependency. Uses gdsfactory's DerivedLayer/LogicalLayer `.get_shapes()`
to resolve boolean layer operations (e.g., WG - DEEP_ETCH) and returns
merged Shapely polygons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import shapely
import shapely.geometry
import shapely.ops

if TYPE_CHECKING:
    from gdsfactory.technology import LayerStack

    from gsim.common.types import AnyShapelyPolygon, GFComponent


def round_coordinates(geom: AnyShapelyPolygon, ndigits: int = 4) -> AnyShapelyPolygon:
    """Round polygon coordinates to eliminate floating point noise."""

    def _round_coords(x: float, y: float, z: float | None = None) -> list[float]:
        x = round(x, ndigits)
        y = round(y, ndigits)
        if z is not None:
            z = round(z, ndigits)
        return [c for c in (x, y, z) if c is not None]

    return shapely.ops.transform(_round_coords, geom)


def fuse_polygons(
    component: GFComponent,
    layer,
    round_tol: int = 4,
    simplify_tol: float = 1e-4,
) -> AnyShapelyPolygon:
    """Extract and merge all polygons for a layer from a component.

    Calls ``layer.get_shapes(component)`` which handles DerivedLayer boolean
    operations (KLayout Region). Converts the resulting KLayout polygons to
    Shapely, including holes, then merges and simplifies.

    Args:
        component: gdsfactory Component
        layer: gdsfactory LayerLevel (has ``.get_shapes()`` via its
            ``.layer`` attribute which is a LogicalLayer or DerivedLayer)
        round_tol: decimal places for coordinate rounding
        simplify_tol: tolerance for polygon simplification

    Returns:
        Merged Shapely Polygon or MultiPolygon
    """
    layer_region = layer.get_shapes(component)

    shapely_polygons = []
    for klayout_polygon in layer_region.each_merged():
        exterior_points = [
            (point.x / 1000, point.y / 1000)
            for point in klayout_polygon.each_point_hull()
        ]
        interior_points = []
        for hole_index in range(klayout_polygon.holes()):
            hole_points = [
                (point.x / 1000, point.y / 1000)
                for point in klayout_polygon.each_point_hole(hole_index)
            ]
            interior_points.append(hole_points)

        shapely_polygons.append(
            round_coordinates(
                shapely.geometry.Polygon(shell=exterior_points, holes=interior_points),
                round_tol,
            )
        )

    return shapely.ops.unary_union(shapely_polygons).simplify(
        simplify_tol, preserve_topology=False
    )


def cleanup_component(
    component: GFComponent,
    layer_stack: LayerStack,
    round_tol: int = 2,
    simplify_tol: float = 1e-2,
) -> dict[str, AnyShapelyPolygon]:
    """Extract and fuse polygons for every layer in a stack.

    Args:
        component: gdsfactory Component
        layer_stack: gdsfactory LayerStack
        round_tol: decimal places for coordinate rounding
        simplify_tol: polygon simplification tolerance

    Returns:
        Dict mapping layer name to merged Shapely polygon
    """
    layer_stack_dict = layer_stack.to_dict()

    return {
        layername: fuse_polygons(
            component,
            layer["layer"],
            round_tol=round_tol,
            simplify_tol=simplify_tol,
        )
        for layername, layer in layer_stack_dict.items()
        if layer["layer"] is not None
    }
