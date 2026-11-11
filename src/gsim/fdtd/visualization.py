"""Static and interactive setup views for GDSFactory FDTD simulations."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from gsim.fdtd.api import PortSource
from gsim.fdtd.mesh import generate_mesh
from gsim.fdtd.runtime import RuntimeConfigMixin
from gsim.fdtd.viz import Axis, ViewerMode, validate_slice

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from gsim.common.materials import MaterialSnapshot
    from gsim.common.pdk import ResolvedPassivePcell
    from gsim.fdtd.api import Geometry
    from gsim.fdtd.models import SimulationArtifacts
    from gsim.fdtd.viz import MeshViewer


class VisualizationMixin(RuntimeConfigMixin):
    """Render FDTD setup geometry without importing Matplotlib until requested."""

    geometry: Geometry
    _resolved: ResolvedPassivePcell | None
    _last_artifacts: SimulationArtifacts | None
    _last_artifacts_domain: dict[str, Any] | None

    def _resolve_geometry(self) -> ResolvedPassivePcell:
        """Resolve the geometry supplied by the concrete simulation."""
        raise NotImplementedError

    def _material_snapshots(self) -> dict[str, MaterialSnapshot]:
        """Resolve optical properties supplied by the concrete simulation."""
        raise NotImplementedError

    def write(self, output_dir: str | Path) -> SimulationArtifacts:
        """Write artifacts using the concrete simulation workflow."""
        raise NotImplementedError

    def plot_2d_static(
        self,
        *,
        axis: Axis = "z",
        position_um: float | None = None,
        show_mesh: bool = False,
        outline_only: bool = False,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
        ax: Axes | None = None,
        cmap: str = "Blues",
        legend: bool = True,
        aspect: Literal["equal", "auto"] = "equal",
        figsize: tuple[float, float] | None = None,
    ) -> Axes:
        """Plot a static refractive-index cross-section with setup annotations.

        The normal axis defaults to z (an XY view). Positions and axis labels
        are in micrometers; an omitted position uses the geometry midpoint.
        Colors show the real refractive index at the source center wavelength.
        PML is hatched outside the physical domain. Source and monitor marks
        appear only when intersected by the slice.

        The returned Matplotlib Axes supports normal figure export. Pass ax
        to compose subplots, show_mesh=True to draw transfer-mesh edges, or
        use plot_2d_interactive() for the original HTML slider viewer.
        Geometry is resolved afresh so edits to the setup are reflected. A
        temporary mesh is generated without writing run artifacts or requiring
        a selected source port.

        Use outline_only=True to draw material boundaries over field data on
        ax, without fills, setup annotations, a legend or an index colorbar.
        Supplied axes retain their limits, labels, title and aspect. Group
        filters still apply; outline_only cannot be combined with show_mesh.
        """
        from gsim.fdtd.slice_viz import plot_mesh_slice

        validate_slice(axis, position_um)
        if aspect not in {"equal", "auto"}:
            raise ValueError("aspect must be 'equal' or 'auto'")
        if outline_only and show_mesh:
            raise ValueError("outline_only=True cannot be combined with show_mesh=True")
        resolved = self._resolve_geometry()
        if isinstance(self.source, PortSource) and self.source.port is not None:
            self._selected_port_name()
        self._validate_explicit_domain_contents()
        self._validate_monitor_regions()
        materials = self._material_snapshots()
        vertical_ports = self._vertical_port_configs(materials)
        if position_um is None:
            position_um = cast(float, self._default_slice_position(axis, None))
        with TemporaryDirectory(prefix="gsim-fdtd-slice-") as directory:
            mesh_path = Path(directory) / "mesh.msh"
            manifest = generate_mesh(
                resolved,
                mesh_path,
                background_material=self._background_material_name,
                padding=self.domain.padding,
                padding_x=self.domain.padding_x,
                padding_y=self.domain.padding_y,
                padding_z=self.domain.padding_z,
                mesh_size_nm=self.geometry.mesh_size_nm,
                geometry_tolerance_nm=self.geometry.geometry_tolerance_nm,
                x_bounds=self.domain.x_bounds,
                y_bounds=self.domain.y_bounds,
                z_bounds=self.domain.z_bounds,
                clip_geometry=self.domain.clip_geometry,
            )
            return plot_mesh_slice(
                mesh_path,
                manifest,
                materials,
                self,
                axis=axis,
                position_um=position_um,
                show_mesh=show_mesh,
                outline_only=outline_only,
                show_groups=show_groups,
                hide_groups=hide_groups,
                ax=ax,
                cmap=cmap,
                legend=legend,
                aspect=aspect,
                figsize=figsize,
                vertical_ports=vertical_ports,
            )

    def _plot_viewer(
        self,
        *,
        mode: ViewerMode,
        axis: Axis = "z",
        position_um: float | None = None,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
        zoom_to_cursor: bool = True,
        height: int = 600,
        use_pdk_styles: bool = False,
    ) -> Any:
        """Render a fresh temporary mesh without changing saved run artifacts."""
        from gsim.fdtd.display import layer_group_styles
        from gsim.fdtd.viz import plot_mesh

        def render(mesh_path: Path) -> Any:
            metadata_path = mesh_path.with_name("metadata.json")
            grid_shape = None
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf8"))
                grid_shape = metadata["grid"]["cell_counts"]["total_cells_xyz"]
            return plot_mesh(
                mesh_path,
                mode=mode,
                axis=axis,
                position_um=self._default_slice_position(axis, position_um)
                if mode in {"slice", "mesh_slice"}
                else position_um,
                show_groups=show_groups,
                hide_groups=hide_groups,
                group_styles=layer_group_styles(self._resolved)
                if use_pdk_styles
                else None,
                zoom_to_cursor=zoom_to_cursor,
                cell_size_nm=self.solver.cell_size_nm,
                pml_cells=self.domain.pml_cells,
                grid_shape=grid_shape,
                height=height,
            )

        previous_artifacts = self._last_artifacts
        with TemporaryDirectory(prefix="gsim-fdtd-view-") as directory:
            try:
                artifacts = self.write(directory)
                return render(artifacts.mesh_path)
            finally:
                self._last_artifacts = previous_artifacts

    def plot_3d(
        self,
        *,
        show_mesh: bool = False,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
        zoom_to_cursor: bool = True,
        height: int = 600,
    ) -> Any:
        """Show 3D geometry with PDK color, opacity and visibility defaults.

        Stack info.color overrides named and physical layer-view colors. Named
        stack views are preferred, then derived/source fill or frame colors.
        Shared drawing colors for blanket films of distinct materials are
        ambiguous and use the fallback palette. Color alpha does not
        change opacity. Stack info.opacity controls surface opacity;
        info.visible overrides layer-view visibility. Zero opacity starts a
        layer hidden and becomes fully opaque when enabled in the viewer.
        Explicit show_groups and
        hide_groups take precedence over PDK visibility, with hide_groups
        taking priority.
        """
        return self._plot_viewer(
            mode="mesh" if show_mesh else "surface",
            show_groups=show_groups,
            hide_groups=hide_groups,
            zoom_to_cursor=zoom_to_cursor,
            height=height,
            use_pdk_styles=True,
        )

    @overload
    def plot_2d(
        self,
        *,
        height: int,
        zoom_to_cursor: bool | None = None,
        axis: Axis = "z",
        position_um: float | None = None,
        show_mesh: bool = False,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
    ) -> MeshViewer: ...

    @overload
    def plot_2d(
        self,
        *,
        zoom_to_cursor: bool,
        height: int | None = None,
        axis: Axis = "z",
        position_um: float | None = None,
        show_mesh: bool = False,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
    ) -> MeshViewer: ...

    @overload
    def plot_2d(
        self,
        *,
        axis: Axis = "z",
        position_um: float | None = None,
        show_mesh: bool = False,
        outline_only: bool = False,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
        ax: Axes | None = None,
        cmap: str = "Blues",
        legend: bool = True,
        aspect: Literal["equal", "auto"] = "equal",
        figsize: tuple[float, float] | None = None,
    ) -> Axes: ...

    def plot_2d(
        self,
        *,
        axis: Axis = "z",
        position_um: float | None = None,
        show_mesh: bool = False,
        outline_only: bool = False,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
        ax: Axes | None = None,
        cmap: str = "Blues",
        legend: bool = True,
        aspect: Literal["equal", "auto"] = "equal",
        figsize: tuple[float, float] | None = None,
        height: int | None = None,
        zoom_to_cursor: bool | None = None,
    ) -> Axes | MeshViewer:
        """Plot a static Matplotlib cross-section, as with plot_2d_static().

        Use plot_2d_interactive() for the HTML viewer with a movable slice.
        Legacy height or zoom_to_cursor arguments retain the HTML viewer with
        a DeprecationWarning; they cannot be combined with static-only options.
        Their removal targets 0.12.0 or later, after at least three months since
        the first released warning in 0.10.0.
        """
        if height is not None or zoom_to_cursor is not None:
            if (
                outline_only
                or ax is not None
                or cmap != "Blues"
                or not legend
                or aspect != "equal"
                or figsize is not None
            ):
                raise TypeError(
                    "Legacy viewer keywords cannot be combined "
                    "with static plot options. "
                    "Use plot_2d_static() or plot_2d_interactive() explicitly."
                )
            warnings.warn(
                "The height and zoom_to_cursor viewer keywords "
                "on plot_2d() are deprecated. "
                "Use plot_2d_interactive() for the HTML viewer; "
                "plot_2d() without viewer keywords now returns a Matplotlib Axes. "
                "Removal targets 0.12.0 or later, after two feature releases and "
                "at least three months since the first released warning in 0.10.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.plot_2d_interactive(
                axis=axis,
                position_um=position_um,
                show_mesh=show_mesh,
                show_groups=show_groups,
                hide_groups=hide_groups,
                height=600 if height is None else height,
                zoom_to_cursor=True if zoom_to_cursor is None else zoom_to_cursor,
            )
        return self.plot_2d_static(
            axis=axis,
            position_um=position_um,
            show_mesh=show_mesh,
            outline_only=outline_only,
            show_groups=show_groups,
            hide_groups=hide_groups,
            ax=ax,
            cmap=cmap,
            legend=legend,
            aspect=aspect,
            figsize=figsize,
        )

    def plot_2d_interactive(
        self,
        *,
        axis: Axis = "z",
        position_um: float | None = None,
        show_mesh: bool = False,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
        zoom_to_cursor: bool = True,
        height: int = 600,
    ) -> Any:
        """Show the legacy interactive cross-section with a movable slice plane."""
        validate_slice(axis, position_um)
        return self._plot_viewer(
            mode="mesh_slice" if show_mesh else "slice",
            axis=axis,
            position_um=position_um,
            show_groups=show_groups,
            hide_groups=hide_groups,
            zoom_to_cursor=zoom_to_cursor,
            height=height,
        )

    def _default_slice_position(
        self, axis: Axis, position_um: float | None
    ) -> float | None:
        """Prefer the configured geometry midpoint over the padded domain center."""
        if position_um is not None or self._resolved is None:
            return position_um
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        explicit_bounds = getattr(self.domain, f"{axis}_bounds")
        lower, upper = (
            getattr(self.resolved, "layout_bounds", None) or self.resolved.bounds
        )
        clipped_axes = getattr(self.resolved, "clipped_layers", {}).values()
        if (
            explicit_bounds is not None
            and self.domain.clip_geometry
            and any(axis in axes for axes in clipped_axes)
        ):
            return (explicit_bounds[0] + explicit_bounds[1]) / 2
        return (lower[axis_index] + upper[axis_index]) / 2
