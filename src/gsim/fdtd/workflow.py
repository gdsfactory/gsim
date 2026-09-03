"""Ergonomic simulation workflow for cloud-hosted GDSFactory FDTD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import gdsfactory as gf
from pdk_schema import Index, ScalarValue

from gsim.common.materials import (
    MaterialResolutionError,
    MaterialSnapshot,
    get_project_material_cards,
    resolve_material_snapshot,
)
from gsim.common.materials._helpers import material_card
from gsim.common.pdk import ResolvedPassivePcell, resolve_passive_pcell
from gsim.common.viz.gmsh import DEFAULT_COLOR_PALETTE
from gsim.fdtd.api import (
    DipoleSource,
    Domain,
    GaussianBeamSource,
    Geometry,
    LineCurrentSource,
    Materials,
    Monitors,
    PlaneMonitor,
    PortSource,
    Solver,
    SourceType,
)
from gsim.fdtd.cloud import CloudWorkflowMixin
from gsim.fdtd.mesh import generate_mesh
from gsim.fdtd.models import (
    FDTDConfigError,
    FDTDGeometryError,
    SimulationArtifacts,
)
from gsim.fdtd.runtime import RuntimeConfigMixin
from gsim.fdtd.viz import Axis, ViewerMode


def _source_from_value(value: SourceType | Mapping[str, Any] | None) -> SourceType:
    """Validate a source mapping using its explicit public type."""
    if value is None:
        return PortSource()
    if isinstance(
        value, (PortSource, DipoleSource, LineCurrentSource, GaussianBeamSource)
    ):
        return value
    source_data = dict(value)
    source_type = source_data.get("type", "port")
    source_classes = {
        "port": PortSource,
        "dipole": DipoleSource,
        "line_current": LineCurrentSource,
        "gaussian_beam": GaussianBeamSource,
    }
    try:
        source_class = source_classes[source_type]
    except KeyError as error:
        raise ValueError(f"Unknown FDTD source type {source_type!r}") from error
    return source_class.model_validate(source_data)


class _PdkWithMaterialOverrides:
    """Delegate to a PDK while exposing merged project material cards."""

    def __init__(self, pdk: Any, material_cards: Mapping[str, Any]) -> None:
        """Wrap a PDK and expose the merged material-card mapping."""
        self._pdk = pdk
        self.material_cards = material_cards

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped PDK."""
        return getattr(self._pdk, name)


def _constant_material_card(name: str, refractive_index: float) -> Any:
    """Represent one public scalar override at the canonical PDK boundary."""
    return material_card(
        name=name,
        temperature_ref=None,
        permittivity=Index(
            validity=None,
            variation=None,
            conductivity=None,
            n=ScalarValue(unit="", value=refractive_index),
            k=None,
        ),
    )


class Simulation(CloudWorkflowMixin, RuntimeConfigMixin):
    """Configure, serialize, and run a GDSFactory FDTD simulation."""

    def __init__(
        self,
        pdk: Any | None = None,
        *,
        geometry: Geometry | Mapping[str, Any] | None = None,
        materials: Materials | Mapping[str, Any] | None = None,
        source: SourceType | Mapping[str, Any] | None = None,
        monitors: Monitors | list[PlaneMonitor | Mapping[str, Any]] | None = None,
        domain: Domain | Mapping[str, Any] | None = None,
        solver: Solver | Mapping[str, Any] | None = None,
        **legacy: Any,
    ) -> None:
        """Create a simulation, translating the original flat keyword API."""
        geometry_data = self._model_data(geometry)
        materials_data = self._model_data(materials)
        source_data = self._model_data(source)
        domain_data = self._model_data(domain)
        solver_data = self._model_data(solver)
        self._translate_legacy(
            legacy,
            geometry_data,
            materials_data,
            source_data,
            domain_data,
            solver_data,
        )
        self.pdk = pdk
        self.geometry = Geometry.model_validate(geometry_data)
        self.materials = Materials.model_validate(materials_data)
        self.source = _source_from_value(source_data)
        self.monitors = (
            monitors if isinstance(monitors, Monitors) else Monitors(monitors)
        )
        self.domain = Domain.model_validate(domain_data)
        self.solver = Solver.model_validate(solver_data)
        self.geometry.bind(self._resolve_geometry)
        self._resolved: ResolvedPassivePcell | None = None
        self._job_id: str | None = None
        self._input_hash: str | None = None
        self._config_dir: Path | None = None
        self._last_artifacts: SimulationArtifacts | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Validate source replacements while keeping concern mutation concise."""
        if name == "source" and hasattr(self, "source"):
            value = _source_from_value(value)
        object.__setattr__(self, name, value)

    @staticmethod
    def _model_data(value: Any) -> dict[str, Any]:
        """Normalize an optional model or mapping to mutable input data."""
        if value is None:
            return {}
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return dict(value)

    @staticmethod
    def _translate_legacy(
        legacy: dict[str, Any],
        geometry: dict[str, Any],
        materials: dict[str, Any],
        source: dict[str, Any],
        domain: dict[str, Any],
        solver: dict[str, Any],
    ) -> None:
        """Map the original flat constructor fields into concern objects."""
        mappings = {
            "mesh_size_nm": (geometry, "mesh_size_nm"),
            "geometry_tolerance_nm": (geometry, "geometry_tolerance_nm"),
            "background_material": (materials, "background"),
            "wavelength_um": (source, "wavelength_um"),
            "num_wavelengths": (source, "num_wavelengths"),
            "default_port": (source, "port"),
            "vertical_port_axis": (source, "vertical_axis"),
            "vertical_port_aperture_width_um": (
                source,
                "vertical_aperture_width_um",
            ),
            "vertical_port_waist_radius_um": (source, "vertical_waist_radius_um"),
            "background_padding_um": (domain, "padding_um"),
            "pml_cells": (domain, "pml_cells"),
            "x_bounds": (domain, "x_bounds"),
            "y_bounds": (domain, "y_bounds"),
            "z_bounds": (domain, "z_bounds"),
            "nanometers_per_cell": (solver, "cell_size_nm"),
            "max_timesteps": (solver, "max_timesteps"),
            "energy_decay_fraction": (solver, "energy_decay_fraction"),
            "max_wall_seconds": (solver, "max_wall_seconds"),
        }
        for old_name, (target, new_name) in mappings.items():
            if old_name in legacy:
                target.setdefault(new_name, legacy.pop(old_name))
        if "wavelength_halfspan_um" in legacy:
            source.setdefault(
                "wavelength_span_um", 2 * legacy.pop("wavelength_halfspan_um")
            )
        if legacy:
            names = ", ".join(sorted(legacy))
            raise TypeError(f"Unexpected Simulation arguments: {names}")

    @property
    def resolved(self) -> ResolvedPassivePcell:
        """Return the most recently resolved canonical geometry."""
        if self._resolved is None:
            raise FDTDGeometryError(
                "No geometry is configured. Call Simulation.geometry(...) first."
            )
        return self._resolved

    @property
    def job_id(self) -> str | None:
        """Cloud job identifier after upload or run."""
        return self._job_id

    def _resolve_geometry(self) -> ResolvedPassivePcell:
        """Resolve current geometry at the current source wavelength."""
        if self.geometry.component is None:
            raise FDTDGeometryError(
                "No geometry is configured. Call Simulation.geometry(...) first."
            )
        pdk = self._pdk_for_resolution()
        self._resolved = resolve_passive_pcell(
            self.geometry.component,
            pdk=pdk,
            settings=self.geometry.settings,
            wavelength_um=self.source.wavelength_um,
        )
        return self._resolved

    def _pdk_for_resolution(self) -> Any:
        """Return a delegating PDK with public material overrides attached."""
        if not self.materials.overrides:
            return self.pdk
        pdk_object = (
            gf.get_active_pdk()
            if self.pdk is None
            else getattr(self.pdk, "PDK", self.pdk)
        )
        cards = dict(get_project_material_cards(self.pdk))
        cards.update(
            {
                name: _constant_material_card(name, material.refractive_index)
                for name, material in self.materials.overrides.items()
            }
        )
        return _PdkWithMaterialOverrides(pdk_object, cards)

    def _material_snapshots(self) -> dict[str, MaterialSnapshot]:
        """Resolve project-first materials, then apply explicit index overrides."""
        snapshots = dict(self.resolved.materials)
        background = self.materials.background
        if background not in snapshots:
            try:
                project_cards = dict(get_project_material_cards(self.pdk))
                if background in self.materials.overrides:
                    project_cards[background] = _constant_material_card(
                        background,
                        self.materials.overrides[background].refractive_index,
                    )
                snapshots[background] = resolve_material_snapshot(
                    background,
                    self.source.wavelength_um,
                    project_cards,
                )
            except MaterialResolutionError as error:
                raise FDTDConfigError(
                    f"Could not resolve background material {background!r}: {error}"
                ) from error
        for name, material in self.materials.overrides.items():
            if name not in snapshots:
                raise FDTDConfigError(
                    f"Material override {name!r} is not used by the geometry "
                    "or background."
                )
            snapshots[name] = replace(
                snapshots[name],
                refractive_index=material.refractive_index,
                extinction_coefficient=0.0,
            )
        return snapshots

    def write(self, output_dir: str | Path) -> SimulationArtifacts:
        """Write ``mesh.msh`` and ``config.json`` for cloud execution."""
        resolved = self._resolve_geometry()
        if isinstance(self.source, PortSource):
            self._selected_port_name()
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        mesh_path = directory / "mesh.msh"
        config_path = directory / "config.json"
        material_snapshots = self._material_snapshots()
        self._validate_explicit_domain_contents()
        manifest = generate_mesh(
            resolved,
            mesh_path,
            background_material=self.materials.background,
            background_padding_um=self.domain.padding_um,
            mesh_size_nm=self.geometry.mesh_size_nm,
            geometry_tolerance_nm=self.geometry.geometry_tolerance_nm,
            x_bounds=self.domain.x_bounds,
            y_bounds=self.domain.y_bounds,
            z_bounds=self.domain.z_bounds,
        )
        config = self._config(manifest, material_snapshots)
        config_path.write_text(
            config.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf8",
        )
        self._last_artifacts = SimulationArtifacts(
            mesh_path=mesh_path,
            config_path=config_path,
            manifest=manifest,
        )
        return self._last_artifacts

    def _plot_viewer(
        self,
        *,
        mode: ViewerMode,
        axis: Axis = "z",
        position_um: float | None = None,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
        include_internal_groups: bool = False,
        color_palette: Sequence[str] = DEFAULT_COLOR_PALETTE,
        group_opacity: float | None = None,
        zoom_to_cursor: bool = True,
        footer_title: str = "Simulation estimate",
        cell_count_label: str = "Estimated Yee cells",
        height: int = 600,
    ) -> Any:
        """Render the current or a temporary mesh with the FDTD viewer."""
        from gsim.fdtd.viz import plot_mesh

        if (
            self._last_artifacts is not None
            and self._last_artifacts.mesh_path.is_file()
        ):
            return plot_mesh(
                self._last_artifacts.mesh_path,
                mode=mode,
                axis=axis,
                position_um=position_um,
                show_groups=show_groups,
                hide_groups=hide_groups,
                include_internal_groups=include_internal_groups,
                color_palette=color_palette,
                group_opacity=group_opacity,
                zoom_to_cursor=zoom_to_cursor,
                footer_title=footer_title,
                cell_count_label=cell_count_label,
                cell_size_nm=self.solver.cell_size_nm,
                pml_cells=self.domain.pml_cells,
                height=height,
            )

        previous_artifacts = self._last_artifacts
        with TemporaryDirectory(prefix="gsim-fdtd-view-") as directory:
            try:
                artifacts = self.write(directory)
                return plot_mesh(
                    artifacts.mesh_path,
                    mode=mode,
                    axis=axis,
                    position_um=position_um,
                    show_groups=show_groups,
                    hide_groups=hide_groups,
                    include_internal_groups=include_internal_groups,
                    color_palette=color_palette,
                    group_opacity=group_opacity,
                    zoom_to_cursor=zoom_to_cursor,
                    footer_title=footer_title,
                    cell_count_label=cell_count_label,
                    cell_size_nm=self.solver.cell_size_nm,
                    pml_cells=self.domain.pml_cells,
                    height=height,
                )
            finally:
                self._last_artifacts = previous_artifacts

    def plot_3d(
        self,
        *,
        show_mesh: bool = False,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
        include_internal_groups: bool = False,
        color_palette: Sequence[str] = DEFAULT_COLOR_PALETTE,
        group_opacity: float | None = None,
        zoom_to_cursor: bool = True,
        footer_title: str = "Simulation estimate",
        cell_count_label: str = "Estimated Yee cells",
        height: int = 600,
    ) -> Any:
        """Show the 3D geometry, optionally with Gmsh surface edges.

        ``footer_title`` and ``cell_count_label`` customize the estimated-grid
        summary without changing its calculation. ``color_palette`` and
        ``group_opacity`` customize physical-group rendering.
        """
        return self._plot_viewer(
            mode="mesh" if show_mesh else "surface",
            show_groups=show_groups,
            hide_groups=hide_groups,
            include_internal_groups=include_internal_groups,
            color_palette=color_palette,
            group_opacity=group_opacity,
            zoom_to_cursor=zoom_to_cursor,
            footer_title=footer_title,
            cell_count_label=cell_count_label,
            height=height,
        )

    def plot_2d(
        self,
        *,
        axis: Axis = "z",
        position_um: float | None = None,
        show_mesh: bool = False,
        show_groups: Sequence[str] | None = None,
        hide_groups: Sequence[str] = (),
        include_internal_groups: bool = False,
        color_palette: Sequence[str] = DEFAULT_COLOR_PALETTE,
        group_opacity: float | None = None,
        zoom_to_cursor: bool = True,
        footer_title: str = "Simulation estimate",
        cell_count_label: str = "Estimated Yee cells",
        height: int = 600,
    ) -> Any:
        """Show a filled cross-section, optionally with intersected cell edges."""
        position_um = self._default_slice_position(axis, position_um)
        return self._plot_viewer(
            mode="mesh_slice" if show_mesh else "slice",
            axis=axis,
            position_um=position_um,
            show_groups=show_groups,
            hide_groups=hide_groups,
            include_internal_groups=include_internal_groups,
            color_palette=color_palette,
            group_opacity=group_opacity,
            zoom_to_cursor=zoom_to_cursor,
            footer_title=footer_title,
            cell_count_label=cell_count_label,
            height=height,
        )

    def _default_slice_position(
        self, axis: Axis, position_um: float | None
    ) -> float | None:
        """Prefer the configured geometry midpoint over the padded domain center."""
        if position_um is not None or self._resolved is None:
            return position_um
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        lower, upper = self.resolved.bounds
        return (lower[axis_index] + upper[axis_index]) / 2


__all__ = ["Simulation"]
