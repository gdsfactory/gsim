"""Geometry module for 3D component modeling in FDTD simulations.

This module contains the Geometry class which is used to model 3D components
in the Tidy3D simulation environment.

Classes:
    Geometry: Represents a 3D component in the Tidy3D simulation environment.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
from gplugins.common.base_models.component import LayeredComponentBase
from pydantic import NonNegativeFloat
from tidy3d.components.geometry.base import from_shapely

from gsim.fdtd.geometry.render2d import plot_prism_slices
from gsim.fdtd.geometry.render3d import (
    create_web_export,
    export_3d_mesh,
    plot_prisms_3d,
    plot_prisms_3d_open3d,
    serve_threejs_visualization,
)
from gsim.fdtd.util import sort_layers

if TYPE_CHECKING:
    pass


class Geometry(LayeredComponentBase):
    """Represents a 3D component in the Tidy3D simulation environment.

    Attributes:
        component: GDS component (can be None for initialization)
        layer_stack: LayerStack (can be None for initialization)
        extend_ports (NonNegativeFloat): The extension length for ports.
        port_offset (float): The offset for ports.
        pad_xy_inner (NonNegativeFloat): The inner padding in the xy-plane.
        pad_xy_outer (NonNegativeFloat): The outer padding in the xy-plane.
        pad_z_inner (float): The inner padding in the z-direction.
        pad_z_outer (NonNegativeFloat): The outer padding in the z-direction.
        dilation (float): Dilation of the polygon in the base by shifting each edge along its
            normal outwards direction by a distance;
            a negative value corresponds to erosion. Defaults to zero.
       reference_plane (Literal["bottom", "middle", "top"]): the reference plane
           used by tidy3d's PolySlab when applying sidewall_angle to a layer
    """

    extend_ports: NonNegativeFloat = 0.5
    port_offset: float = 0.2
    pad_xy_inner: NonNegativeFloat = 3.0
    pad_xy_outer: NonNegativeFloat = 3.0
    pad_z_inner: float = 3.0
    pad_z_outer: NonNegativeFloat = 3.0
    dilation: float = 0.0
    reference_plane: Literal["bottom", "middle", "top"] = "middle"

    @cached_property
    def bbox(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Override bbox to use core layer bounds plus padding."""
        try:
            core_bbox = self.get_layer_bbox("core")

            xmin = core_bbox[0][0] - self.pad_xy_outer
            ymin = core_bbox[0][1] - self.pad_xy_outer
            zmin = core_bbox[0][2] - self.pad_z_outer

            xmax = core_bbox[1][0] + self.pad_xy_outer
            ymax = core_bbox[1][1] + self.pad_xy_outer
            zmax = core_bbox[1][2] + self.pad_z_outer

            return ((xmin, ymin, zmin), (xmax, ymax, zmax))
        except KeyError:
            return super().bbox

    @cached_property
    def polyslabs(self) -> dict[str, tuple[td.Geometry, ...]]:
        """Returns a dictionary of PolySlab instances for each layer in the component.

        Returns:
            dict[str, tuple[td.PolySlab, ...]]: A dictionary mapping layer names
                to tuples of PolySlab instances.
        """
        slabs = {}
        layers = sort_layers(self.geometry_layers, sort_by="mesh_order", reverse=True)
        for name, layer in layers.items():
            bbox = self.get_layer_bbox(name)
            shape = self.polygons[name].buffer(distance=0.0, join_style="mitre")
            geom = from_shapely(
                shape,
                axis=2,
                slab_bounds=(bbox[0][2], bbox[1][2]),
                dilation=self.dilation,
                sidewall_angle=np.deg2rad(layer.sidewall_angle),
                reference_plane=self.reference_plane,
            )
            slabs[name] = geom

        return slabs

    @cached_property
    def meep_prisms(self) -> dict[str, list]:
        """Returns MEEP Prism instances for each layer.

        Alternative to Tidy3D PolySlabs for MEEP-based simulations.

        Returns:
            dict[str, list[mp.Prism]]: Layer names mapped to lists of MEEP Prism objects.
        """
        import meep as mp

        prisms = {}
        layers = sort_layers(self.geometry_layers, sort_by="mesh_order", reverse=True)

        for name, layer in layers.items():
            bbox = self.get_layer_bbox(name)
            zmin = bbox[0][2]
            height = bbox[1][2] - bbox[0][2]

            shape = self.polygons[name]

            layer_prisms = []

            if hasattr(shape, "geoms"):
                polygons = shape.geoms
            else:
                polygons = [shape]

            for polygon in polygons:
                if polygon.is_empty or not polygon.is_valid:
                    continue

                if hasattr(polygon, "interiors") and polygon.interiors:
                    triangular_prisms = self._create_triangulated_prisms(
                        polygon, height, zmin, layer.sidewall_angle
                    )
                    layer_prisms.extend(triangular_prisms)
                else:
                    vertices = [
                        mp.Vector3(p[0], p[1], zmin)
                        for p in polygon.exterior.coords[:-1]
                    ]

                    prism = mp.Prism(
                        vertices=vertices,
                        height=height,
                        sidewall_angle=(
                            np.deg2rad(layer.sidewall_angle)
                            if layer.sidewall_angle
                            else 0
                        ),
                    )

                    prism._original_polygon = polygon
                    layer_prisms.append(prism)

            prisms[name] = layer_prisms

        return prisms

    def _create_triangulated_prisms(
        self, polygon, height: float, zmin: float, sidewall_angle: float = 0
    ):
        """Create multiple triangular MEEP prisms from a polygon with holes."""
        import meep as mp

        try:
            import shapely.geometry as sg
            from scipy.spatial import Delaunay
        except ImportError:
            print(
                "Warning: scipy not available, falling back to exterior-only prism"
            )
            vertices = [
                mp.Vector3(p[0], p[1], zmin) for p in polygon.exterior.coords[:-1]
            ]
            prism = mp.Prism(
                vertices=vertices,
                height=height,
                sidewall_angle=np.deg2rad(sidewall_angle) if sidewall_angle else 0,
            )
            prism._original_polygon = polygon
            return [prism]

        all_points = []
        all_points.extend(list(polygon.exterior.coords[:-1]))

        for interior in polygon.interiors:
            all_points.extend(list(interior.coords[:-1]))

        if len(all_points) < 3:
            vertices = [
                mp.Vector3(p[0], p[1], zmin) for p in polygon.exterior.coords[:-1]
            ]
            prism = mp.Prism(
                vertices=vertices,
                height=height,
                sidewall_angle=np.deg2rad(sidewall_angle) if sidewall_angle else 0,
            )
            prism._original_polygon = polygon
            return [prism]

        triangular_prisms = []

        if (
            len(polygon.interiors) == 1
            and len(all_points) < 200
            and abs(
                len(list(polygon.exterior.coords))
                - len(list(polygon.interiors[0].coords))
            )
            < 10
        ):
            triangular_prisms = self._create_ring_triangulation(
                polygon, height, zmin, sidewall_angle
            )

        else:
            points_2d = np.array(all_points)
            tri = Delaunay(points_2d)

            for triangle_indices in tri.simplices:
                triangle_points = points_2d[triangle_indices]
                centroid = np.mean(triangle_points, axis=0)
                centroid_point = sg.Point(centroid[0], centroid[1])

                if polygon.contains(centroid_point):
                    triangle_vertices = [
                        mp.Vector3(triangle_points[0][0], triangle_points[0][1], zmin),
                        mp.Vector3(triangle_points[1][0], triangle_points[1][1], zmin),
                        mp.Vector3(triangle_points[2][0], triangle_points[2][1], zmin),
                    ]

                    triangle_prism = mp.Prism(
                        vertices=triangle_vertices,
                        height=height,
                        sidewall_angle=(
                            np.deg2rad(sidewall_angle) if sidewall_angle else 0
                        ),
                    )

                    triangle_prism._original_polygon = polygon
                    triangular_prisms.append(triangle_prism)

        if not triangular_prisms:
            print(
                "Warning: No valid triangles found, falling back to exterior-only prism"
            )
            vertices = [
                mp.Vector3(p[0], p[1], zmin) for p in polygon.exterior.coords[:-1]
            ]
            prism = mp.Prism(
                vertices=vertices,
                height=height,
                sidewall_angle=np.deg2rad(sidewall_angle) if sidewall_angle else 0,
            )
            prism._original_polygon = polygon
            return [prism]

        print(
            f"Created {len(triangular_prisms)} triangular prisms for polygon "
            f"with {len(polygon.interiors)} holes"
        )
        return triangular_prisms

    def _create_ring_triangulation(
        self, polygon, height: float, zmin: float, sidewall_angle: float = 0
    ):
        """Create efficient triangulation for simple polygons with one hole."""
        import meep as mp

        exterior_coords = list(polygon.exterior.coords[:-1])
        interior_coords = list(polygon.interiors[0].coords[:-1])

        n_outer = len(exterior_coords)
        n_inner = len(interior_coords)

        triangular_prisms = []

        for i in range(n_outer):
            next_i = (i + 1) % n_outer

            inner_i = int(i * n_inner / n_outer) % n_inner
            inner_next = int(next_i * n_inner / n_outer) % n_inner

            triangle1_vertices = [
                mp.Vector3(exterior_coords[i][0], exterior_coords[i][1], zmin),
                mp.Vector3(exterior_coords[next_i][0], exterior_coords[next_i][1], zmin),
                mp.Vector3(interior_coords[inner_i][0], interior_coords[inner_i][1], zmin),
            ]

            triangle1_prism = mp.Prism(
                vertices=triangle1_vertices,
                height=height,
                sidewall_angle=np.deg2rad(sidewall_angle) if sidewall_angle else 0,
            )
            triangle1_prism._original_polygon = polygon
            triangular_prisms.append(triangle1_prism)

            triangle2_vertices = [
                mp.Vector3(exterior_coords[next_i][0], exterior_coords[next_i][1], zmin),
                mp.Vector3(
                    interior_coords[inner_next][0], interior_coords[inner_next][1], zmin
                ),
                mp.Vector3(interior_coords[inner_i][0], interior_coords[inner_i][1], zmin),
            ]

            triangle2_prism = mp.Prism(
                vertices=triangle2_vertices,
                height=height,
                sidewall_angle=np.deg2rad(sidewall_angle) if sidewall_angle else 0,
            )
            triangle2_prism._original_polygon = polygon
            triangular_prisms.append(triangle2_prism)

        print(
            f"Efficient polygon triangulation: {len(triangular_prisms)} triangular prisms "
            f"(was {n_outer + n_inner} boundary points)"
        )
        return triangular_prisms

    def plot_prism(
        self,
        x: float | str | None = None,
        y: float | str | None = None,
        z: float | str = "core",
        ax: plt.Axes | None = None,
        legend: bool = True,
        slices: str = "z",
    ) -> plt.Axes | None:
        """Plot cross sections of MEEP prisms with multi-view support.

        Args:
            x: The x-coordinate for the cross section. If str, uses layer name.
            y: The y-coordinate for the cross section. If str, uses layer name.
            z: The z-coordinate for the cross section. If str, uses layer name.
            ax: The Axes instance to plot on. If None, creates new figure.
            legend: Whether to include a legend in the plot.
            slices: Which slice(s) to plot ("x", "y", "z", "xy", "xz", "yz", "xyz").

        Returns:
            plt.Axes or None: Returns None when creating new figure,
                returns Axes if ax was provided.
        """
        return plot_prism_slices(self, x, y, z, ax, legend, slices)

    def plot_3d(self, backend: str = "open3d", **kwargs) -> Any:
        """Create interactive 3D visualization of the geometry.

        Args:
            backend: Rendering backend ("open3d" for Jupyter/VS Code,
                "pyvista" for desktop)
            **kwargs: Additional arguments passed to the backend renderer
        """
        if backend == "pyvista":
            return plot_prisms_3d(self, **kwargs)
        elif backend == "open3d":
            return plot_prisms_3d_open3d(self, **kwargs)
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Use 'open3d' or 'pyvista'"
            )

    def export_3d(self, filename: str, format: str = "auto") -> None:
        """Export 3D geometry to mesh file."""
        return export_3d_mesh(self, filename, format)

    def serve_3d(self, port: int = 8000, auto_open: bool = True, **kwargs) -> str:
        """Start FastAPI server to display Three.js visualization in browser.

        Args:
            port: Port to serve on (default 8000)
            auto_open: Whether to automatically open browser
            **kwargs: Additional Three.js options

        Returns:
            URL of the running server
        """
        return serve_threejs_visualization(
            self, port=port, auto_open=auto_open, **kwargs
        )

    def export_web_3d(
        self,
        filename: str = "geometry_3d.html",
        title: str = "3D Geometry Visualization",
    ) -> str:
        """Export 3D visualization as standalone HTML file."""
        return create_web_export(self, filename, title)

    @td.components.viz.add_ax_if_none
    def plot_slice(
        self,
        x: float | str | None = None,
        y: float | str | None = None,
        z: float | str | None = None,
        offset: float = 0.0,
        ax: plt.Axes | None = None,
        legend: bool = False,
    ) -> plt.Axes:
        """Plots a cross section of the component at a specified position.

        Args:
            x: The x-coordinate for the cross section.
            y: The y-coordinate for the cross section.
            z: The z-coordinate for the cross section.
            offset: The offset for the cross section.
            ax: The Axes instance to plot on.
            legend: Whether to include a legend in the plot.

        Returns:
            plt.Axes: The Axes instance with the plot.
        """
        x, y, z = (
            self.get_layer_center(c)[i] if isinstance(c, str) else c
            for i, c in enumerate((x, y, z))
        )
        x, y, z = (c if c is None else c + offset for c in (x, y, z))

        colors = dict(
            zip(
                self.polyslabs.keys(),
                plt.colormaps.get_cmap("Spectral")(
                    np.linspace(0, 1, len(self.polyslabs))
                ),
            )
        )

        layers = sort_layers(self.geometry_layers, sort_by="zmin", reverse=True)
        meshorders = np.unique([v.mesh_order for v in layers.values()])
        order_map = dict(zip(meshorders, range(0, -len(meshorders), -1)))
        xmin, xmax = np.inf, -np.inf
        ymin, ymax = np.inf, -np.inf

        for name, layer in layers.items():
            if name not in self.polyslabs:
                continue
            poly = self.polyslabs[name]

            axis, position = poly.parse_xyz_kwargs(x=x, y=y, z=z)
            xlim, ylim = poly._get_plot_limits(axis=axis, buffer=0)
            xmin, xmax = min(xmin, xlim[0]), max(xmax, xlim[1])
            ymin, ymax = min(ymin, ylim[0]), max(ymax, ylim[1])
            for idx, shape in enumerate(poly.intersections_plane(x=x, y=y, z=z)):
                _shape = td.Geometry.evaluate_inf_shape(shape)
                patch = td.components.viz.polygon_patch(
                    _shape,
                    facecolor=colors[name],
                    edgecolor="k",
                    linewidth=0.5,
                    label=name if idx == 0 else None,
                    zorder=order_map[layer.mesh_order],
                )
                ax.add_artist(patch)

        size = list(self.size)
        cmin = list(self.bbox[0])
        size.pop(axis)
        cmin.pop(axis)

        sim_roi = plt.Rectangle(
            cmin,
            *size,
            facecolor="none",
            edgecolor="k",
            linestyle="--",
            linewidth=1,
            label="Simulation",
        )
        ax.add_patch(sim_roi)

        xlabel, ylabel = poly._get_plot_labels(axis=axis)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        if legend:
            ax.legend(fancybox=True, framealpha=1.0)

        return ax
