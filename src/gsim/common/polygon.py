"""Polygon extraction and processing for layered GDS components.

Ported from gplugins.common.base_models.component to avoid the gplugins
dependency. Uses gdsfactory's DerivedLayer/LogicalLayer `.get_shapes()`
to resolve boolean layer operations (e.g., WG - DEEP_ETCH) and returns
merged Shapely polygons.
"""

from __future__ import annotations

from math import radians, tan
from typing import TYPE_CHECKING, Any

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


def _background_exclusions(component: GFComponent, layer) -> Any:
    """Collect the source masks removed from a blanket film."""
    import gdsfactory as gf
    from kfactory import kdb

    excluded = kdb.Region()
    for excluded_layer in getattr(layer, "background_exclude_layers", ()):
        layer_index = component.kcl.layer(*gf.get_layer_tuple(excluded_layer))
        excluded += kdb.Region(component.kdb_cell.begin_shapes_rec(layer_index))
    return excluded


def layer_level_region(component: GFComponent, layer) -> Any:
    """Evaluate the KLayout region a layer or LayerLevel covers on a component.

    A ``LayerLevel`` with ``background=True`` fills the component bounding box,
    whether or not its source layer has polygons, minus the polygons on its
    ``background_exclude_layers``. Any other level, or a bare layer, evaluates
    its source expression with ``get_shapes()``, which resolves DerivedLayer
    booleans.

    ``background`` was added to ``LayerLevel`` in gdsfactory 9.4x; on older
    versions the attribute is absent and every level evaluates its source layer.
    """
    if getattr(layer, "background", False):
        from kfactory import kdb

        return kdb.Region(component.kdb_cell.bbox()) - _background_exclusions(
            component, layer
        )

    source_layer = getattr(layer, "layer", layer)
    return source_layer.get_shapes(component)


def fuse_polygons(
    component: GFComponent,
    layer,
    round_tol: int = 4,
    simplify_tol: float = 1e-4,
    *,
    background_bounds: tuple[float, float, float, float] | None = None,
    background_overhang_axes: tuple[bool, bool] = (True, True),
) -> AnyShapelyPolygon:
    """Extract and merge all polygons for a layer from a component.

    Evaluates the layer with :func:`layer_level_region`, which handles
    DerivedLayer boolean operations and background levels. Converts the
    resulting KLayout polygons to Shapely, including holes, then merges and
    simplifies. Corner contacts are split into separate polygons so their
    contours form valid Shapely rings.

    Args:
        component: gdsfactory Component
        layer: gdsfactory LayerLevel (has ``.get_shapes()`` via its
            ``.layer`` attribute which is a LogicalLayer or DerivedLayer)
        round_tol: decimal places for coordinate rounding
        simplify_tol: tolerance for polygon simplification
        background_bounds: Optional XY bounds in micrometers for blanket films.
            Patterned masks are unchanged; background exclusion masks still apply.
        background_overhang_axes: Add sidewall overhang only on resolved domain
            axes. Automatic axes retain their layout reference until domain sizing.

    Returns:
        Merged Shapely Polygon or MultiPolygon
    """
    expand_background = background_bounds is not None and getattr(
        layer, "background", False
    )
    layer_region = (
        _background_exclusions(component, layer)
        if expand_background
        else layer_level_region(component, layer)
    ).dup()
    # KLayout's default merge can produce self-touching rings, invalid in GEOS.
    layer_region.min_coherence = True

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

    geometry = shapely.ops.unary_union(shapely_polygons)
    if expand_background and background_bounds is not None:
        slope = abs(float(layer.thickness)) * tan(radians(float(layer.sidewall_angle)))
        # Keep artificial outer edges outside the domain at every Z section.
        # Exclusion-mask contours stay at their original reference position.
        overhang = max(0.0, -layer.width_to_z * slope, (1 - layer.width_to_z) * slope)
        x_overhang = overhang if background_overhang_axes[0] else 0.0
        y_overhang = overhang if background_overhang_axes[1] else 0.0
        xmin, ymin, xmax, ymax = background_bounds
        footprint = shapely.geometry.box(
            xmin - x_overhang,
            ymin - y_overhang,
            xmax + x_overhang,
            ymax + y_overhang,
        )
        geometry = footprint.difference(geometry)
    return geometry.simplify(simplify_tol, preserve_topology=False)


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
    return {
        layername: fuse_polygons(
            component,
            layer,
            round_tol=round_tol,
            simplify_tol=simplify_tol,
        )
        for layername, layer in layer_stack.layers.items()
        if layer.layer is not None
    }
