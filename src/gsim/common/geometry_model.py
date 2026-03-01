"""Solver-agnostic geometry model for 3D visualization.

Provides generic Prism and GeometryModel dataclasses that represent
extruded 2D polygons without any solver dependency (no meep, no tidy3d).
These are suitable for client-side visualization and geometry inspection.

Classes:
    Prism: A 2D polygon extruded along z.
    GeometryModel: Collection of prisms organized by layer.

Functions:
    extract_geometry_model: Convert a LayeredComponentBase into a GeometryModel.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point

if TYPE_CHECKING:
    from shapely import Polygon

    from gsim.common.layered_component import LayeredComponentBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prism
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prism:
    """A 2D polygon extruded in z. No solver dependency.

    Attributes:
        vertices: (N, 2) numpy array of xy coordinates defining the polygon.
        z_base: Bottom z coordinate of the extrusion.
        z_top: Top z coordinate of the extrusion.
        layer_name: Name of the layer this prism belongs to.
        material: Name of the material assigned to this prism.
        sidewall_angle: Sidewall taper angle in degrees (gdsfactory convention).
        original_polygon: Optional reference to the source Shapely polygon.
    """

    vertices: np.ndarray  # (N, 2) xy coords
    z_base: float
    z_top: float
    layer_name: str = ""
    material: str = ""
    sidewall_angle: float = 0.0
    original_polygon: Any = field(default=None, repr=False, compare=False)

    @property
    def height(self) -> float:
        """Extrusion height (z_top - z_base)."""
        return self.z_top - self.z_base

    @property
    def z_center(self) -> float:
        """Center z coordinate."""
        return (self.z_base + self.z_top) / 2.0


# ---------------------------------------------------------------------------
# GeometryModel
# ---------------------------------------------------------------------------


@dataclass
class GeometryModel:
    """Complete 3D geometry: layers of prisms, ready for visualization.

    Attributes:
        prisms: Mapping from layer name to list of Prism objects.
        bbox: Axis-aligned 3D bounding box as ((xmin, ymin, zmin), (xmax, ymax, zmax)).
        layer_bboxes: Optional per-layer bounding boxes for 2D slice logic.
        layer_mesh_orders: Optional mapping of layer_name -> mesh_order for
            z-ordering in 2D plots.
    """

    prisms: dict[str, list[Prism]]
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]]
    layer_bboxes: dict[
        str,
        tuple[tuple[float, float, float], tuple[float, float, float]],
    ] = field(default_factory=dict)
    layer_mesh_orders: dict[str, int] = field(default_factory=dict)

    @property
    def all_prisms(self) -> list[Prism]:
        """Flat list of all prisms across all layers."""
        return [p for layer_prisms in self.prisms.values() for p in layer_prisms]

    @property
    def layer_names(self) -> list[str]:
        """Ordered list of layer names that contain prisms."""
        return list(self.prisms.keys())

    @property
    def size(self) -> tuple[float, float, float]:
        """(dx, dy, dz) extent of the bounding box."""
        mn, mx = self.bbox
        return (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])

    def get_layer_center(self, layer_name: str) -> tuple[float, float, float]:
        """Return the center of a layer's bounding box.

        Falls back to the geometry-wide center if no per-layer bbox is stored.
        """
        bb = self.get_layer_bbox(layer_name)
        mn, mx = bb
        return (
            (mn[0] + mx[0]) / 2.0,
            (mn[1] + mx[1]) / 2.0,
            (mn[2] + mx[2]) / 2.0,
        )

    def get_layer_bbox(
        self,
        layer_name: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return the bounding box for a specific layer.

        Falls back to the geometry-wide bbox if no per-layer bbox is stored.
        """
        return self.layer_bboxes.get(layer_name, self.bbox)


# ---------------------------------------------------------------------------
# Triangulation helpers (ported from fdtd/geometry/core.py, no meep)
# ---------------------------------------------------------------------------


def _triangulate_polygon_with_holes(
    polygon: Polygon,
    z_base: float,
    z_top: float,
    sidewall_angle: float,
    layer_name: str,
    material: str,
) -> list[Prism]:
    """Create triangular Prisms from a polygon with holes via Delaunay.

    Collects all boundary points (exterior + interiors), runs Delaunay
    triangulation, and keeps only triangles whose centroid lies inside
    the original polygon.
    """
    all_points: list[tuple[float, float]] = []
    all_points.extend(list(polygon.exterior.coords[:-1]))
    for interior in polygon.interiors:
        all_points.extend(list(interior.coords[:-1]))

    if len(all_points) < 3:
        # Degenerate case -- fall back to exterior-only prism
        return _prism_from_exterior(
            polygon,
            z_base,
            z_top,
            sidewall_angle,
            layer_name,
            material,
        )

    # Optimised ring triangulation for single-hole, similarly-sized rings
    if (
        len(polygon.interiors) == 1
        and len(all_points) < 200
        and abs(
            len(list(polygon.exterior.coords)) - len(list(polygon.interiors[0].coords)),
        )
        < 10
    ):
        return _ring_triangulation(
            polygon,
            z_base,
            z_top,
            sidewall_angle,
            layer_name,
            material,
        )

    points_2d = np.array(all_points)
    tri = Delaunay(points_2d)

    triangular_prisms: list[Prism] = []
    for simplex in tri.simplices:
        triangle_pts = points_2d[simplex]
        centroid = np.mean(triangle_pts, axis=0)
        if polygon.contains(Point(centroid[0], centroid[1])):
            triangular_prisms.append(
                Prism(
                    vertices=triangle_pts.copy(),
                    z_base=z_base,
                    z_top=z_top,
                    layer_name=layer_name,
                    material=material,
                    sidewall_angle=sidewall_angle,
                    original_polygon=polygon,
                ),
            )

    if not triangular_prisms:
        warnings.warn(
            "No valid triangles found for holed polygon; "
            "falling back to exterior-only prism.",
            stacklevel=2,
        )
        return _prism_from_exterior(
            polygon,
            z_base,
            z_top,
            sidewall_angle,
            layer_name,
            material,
        )

    logger.debug(
        "Created %d triangular prisms for polygon with %d holes",
        len(triangular_prisms),
        len(polygon.interiors),
    )
    return triangular_prisms


def _ring_triangulation(
    polygon: Polygon,
    z_base: float,
    z_top: float,
    sidewall_angle: float,
    layer_name: str,
    material: str,
) -> list[Prism]:
    """Efficient fan triangulation for a polygon with a single hole.

    Pairs exterior vertices with interior vertices proportionally and
    creates two triangles per exterior edge, covering the ring area.
    """
    exterior_coords = list(polygon.exterior.coords[:-1])
    interior_coords = list(polygon.interiors[0].coords[:-1])

    n_outer = len(exterior_coords)
    n_inner = len(interior_coords)

    triangular_prisms: list[Prism] = []

    for i in range(n_outer):
        next_i = (i + 1) % n_outer
        inner_i = int(i * n_inner / n_outer) % n_inner
        inner_next = int(next_i * n_inner / n_outer) % n_inner

        tri1 = np.array(
            [exterior_coords[i], exterior_coords[next_i], interior_coords[inner_i]],
        )
        triangular_prisms.append(
            Prism(
                vertices=tri1,
                z_base=z_base,
                z_top=z_top,
                layer_name=layer_name,
                material=material,
                sidewall_angle=sidewall_angle,
                original_polygon=polygon,
            ),
        )

        tri2 = np.array(
            [
                exterior_coords[next_i],
                interior_coords[inner_next],
                interior_coords[inner_i],
            ],
        )
        triangular_prisms.append(
            Prism(
                vertices=tri2,
                z_base=z_base,
                z_top=z_top,
                layer_name=layer_name,
                material=material,
                sidewall_angle=sidewall_angle,
                original_polygon=polygon,
            ),
        )

    logger.debug(
        "Ring triangulation: %d prisms (from %d boundary points)",
        len(triangular_prisms),
        n_outer + n_inner,
    )
    return triangular_prisms


def _prism_from_exterior(
    polygon: Polygon,
    z_base: float,
    z_top: float,
    sidewall_angle: float,
    layer_name: str,
    material: str,
) -> list[Prism]:
    """Create a single Prism from a polygon's exterior ring (ignoring holes)."""
    coords = np.array(polygon.exterior.coords[:-1])
    return [
        Prism(
            vertices=coords,
            z_base=z_base,
            z_top=z_top,
            layer_name=layer_name,
            material=material,
            sidewall_angle=sidewall_angle,
            original_polygon=polygon,
        ),
    ]


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_geometry_model(
    layered_component: LayeredComponentBase,
) -> GeometryModel:
    """Convert a LayeredComponentBase into a GeometryModel with generic Prisms.

    For each geometry layer (sorted by mesh_order):
      1. Retrieve the merged Shapely polygon from *layered_component.polygons*.
      2. Compute z_base / z_top from *get_layer_bbox*.
      3. Iterate sub-polygons for MultiPolygon geometries.
      4. Handle polygons with holes via Delaunay triangulation.
      5. Produce Prism objects with (N, 2) numpy vertex arrays.

    Args:
        layered_component: A LayeredComponentBase (or subclass) instance
            that provides polygons, geometry_layers, and get_layer_bbox.

    Returns:
        A GeometryModel containing all extracted prisms and the overall
        3D bounding box.
    """
    all_prisms: dict[str, list[Prism]] = {}
    layer_bboxes: dict[
        str,
        tuple[tuple[float, float, float], tuple[float, float, float]],
    ] = {}
    layer_mesh_orders: dict[str, int] = {}

    # Sort layers by mesh_order (ascending) for consistent rendering order
    sorted_layers = sorted(
        layered_component.geometry_layers.items(),
        key=lambda item: item[1].mesh_order,
    )

    for name, layer in sorted_layers:
        shape = layered_component.polygons[name]
        bbox = layered_component.get_layer_bbox(name)
        layer_bboxes[name] = bbox
        layer_mesh_orders[name] = (
            layer.mesh_order if layer.mesh_order is not None else 0
        )
        z_base = bbox[0][2]
        z_top = bbox[1][2]

        sidewall_angle = layer.sidewall_angle or 0.0
        material_name = str(layer.material) if layer.material else ""

        # Normalise to a list of individual Polygon objects
        if hasattr(shape, "geoms"):
            polygons: list[Polygon] = list(shape.geoms)
        else:
            polygons = [shape]

        layer_prisms: list[Prism] = []

        for polygon in polygons:
            if polygon.is_empty or not polygon.is_valid:
                continue

            if hasattr(polygon, "interiors") and polygon.interiors:
                layer_prisms.extend(
                    _triangulate_polygon_with_holes(
                        polygon,
                        z_base=z_base,
                        z_top=z_top,
                        sidewall_angle=sidewall_angle,
                        layer_name=name,
                        material=material_name,
                    ),
                )
            else:
                coords = np.array(polygon.exterior.coords[:-1])
                layer_prisms.append(
                    Prism(
                        vertices=coords,
                        z_base=z_base,
                        z_top=z_top,
                        layer_name=name,
                        material=material_name,
                        sidewall_angle=sidewall_angle,
                        original_polygon=polygon,
                    ),
                )

        all_prisms[name] = layer_prisms

    return GeometryModel(
        prisms=all_prisms,
        bbox=layered_component.bbox,
        layer_bboxes=layer_bboxes,
        layer_mesh_orders=layer_mesh_orders,
    )
