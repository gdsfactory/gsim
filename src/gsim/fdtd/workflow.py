"""Ergonomic simulation workflow for cloud-hosted GDSFactory FDTD."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import replace
from math import isfinite
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Self

import gdsfactory as gf
from pdk_schema import Index, ScalarValue

from gsim.common.cloud import CloudJobMixin
from gsim.common.materials import (
    MaterialResolutionError,
    MaterialSnapshot,
    canonical_material_name,
    get_project_material_cards,
    resolve_material_snapshot,
)
from gsim.common.materials._helpers import material_card
from gsim.common.materials.registry import get_project_material_aliases
from gsim.common.pdk import ResolvedPassivePcell
from gsim.fdtd.api import (
    DipoleSource,
    Domain,
    GaussianBeamSource,
    Geometry,
    Heatmap,
    LineCurrentSource,
    Materials,
    Monitors,
    PlaneMonitor,
    PortSource,
    SignedAxis,
    Solver,
    SourceType,
)
from gsim.fdtd.domain_geometry import resolve_domain_geometry
from gsim.fdtd.fiber_medium import resolve_fiber_media
from gsim.fdtd.mesh import generate_mesh
from gsim.fdtd.metadata import write_metadata
from gsim.fdtd.models import (
    FDTDConfigError,
    FDTDGeometryError,
    SimulationArtifacts,
)
from gsim.fdtd.monitor_bounds import monitor_region_nm
from gsim.fdtd.monitor_names import resolve_monitor_name
from gsim.fdtd.visualization import VisualizationMixin

# Flat constructor arguments of the original API, as (concern, field).
# wavelength_halfspan_um is translated separately: it doubles into a span.
_LEGACY_FIELDS: dict[str, tuple[str, str]] = {
    "mesh_size_nm": ("geometry", "mesh_size_nm"),
    "geometry_tolerance_nm": ("geometry", "geometry_tolerance_nm"),
    "background_material": ("materials", "background"),
    "wavelength_um": ("source", "wavelength_um"),
    "num_wavelengths": ("source", "num_wavelengths"),
    "default_port": ("source", "port"),
    "vertical_port_axis": ("source", "vertical_axis"),
    "vertical_port_aperture_width_um": ("source", "vertical_aperture_width_um"),
    "vertical_port_waist_radius_um": ("source", "vertical_waist_radius_um"),
    "padding": ("domain", "padding"),
    "padding_x": ("domain", "padding_x"),
    "padding_y": ("domain", "padding_y"),
    "padding_z": ("domain", "padding_z"),
    "pml_cells": ("domain", "pml_cells"),
    "x_bounds": ("domain", "x_bounds"),
    "y_bounds": ("domain", "y_bounds"),
    "z_bounds": ("domain", "z_bounds"),
    "nanometers_per_cell": ("solver", "cell_size_nm"),
    "max_timesteps": ("solver", "max_timesteps"),
    "energy_decay_fraction": ("solver", "energy_decay_fraction"),
    "max_wall_seconds": ("solver", "max_wall_seconds"),
}


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

    def __init__(
        self,
        pdk: Any,
        material_cards: Mapping[str, Any],
        material_aliases: Mapping[str, str],
    ) -> None:
        """Wrap a PDK and expose the merged material-card mapping."""
        self._pdk = pdk
        self.material_cards = material_cards
        self.material_aliases = material_aliases

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


class Simulation(CloudJobMixin, VisualizationMixin):
    """Configure, serialize, and run a GDSFactory FDTD simulation."""

    _cloud_job_type = "fdtd"

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
        self._clipping_warning_key: tuple[Any, ...] | None = None
        self._last_artifacts: SimulationArtifacts | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Validate source replacements while keeping concern mutation concise."""
        if name in _LEGACY_FIELDS:
            concern, field = _LEGACY_FIELDS[name]
            raise AttributeError(
                f"Simulation.{name} is a constructor argument only; "
                f"set simulation.{concern}.{field} instead."
            )
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
        targets = {
            "geometry": geometry,
            "materials": materials,
            "source": source,
            "domain": domain,
            "solver": solver,
        }
        for old_name, (concern, new_name) in _LEGACY_FIELDS.items():
            if old_name in legacy:
                targets[concern].setdefault(new_name, legacy.pop(old_name))
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

    def add_monitors(
        self,
        *,
        xy: bool = True,
        xy_name: str | None = None,
        z_um: float | None = None,
        margin_um: float = 0.75,
    ) -> Self:
        """Add center-wavelength port intensity images and an optional XY field image.

        Args:
            xy: Include a component-wide XY field image (default: True).
            xy_name: XY image name. Use field_xy for new code. None retains
                overview_xy with a DeprecationWarning during the transition.
                Ignored when xy is False.
            z_um: XY image height in micrometers. None uses the resolved geometry's
                vertical midpoint within any opted-in crop. Ignored when xy is False.
            margin_um: Nonnegative cladding margin on each side of a port's
                active cross section, in micrometers (default: 0.75).

        Returns:
            This simulation, with the diagnostic planes added to its monitors.

        Notes:
            Optical port windows follow the port width and selected physical
            bands' active bounds, plus margin_um, three Yee cells inward from
            each port. Preview and output windows are cropped to the domain
            without changing the requested settings.
            Each plane records only the source center wavelength,
            without additional flux measurements. Guided S-parameter measurements
            remain implicit. Existing monitors are preserved; duplicate names
            raise ValueError before any planes are added. This method does not
            mesh, submit or run a simulation.
        """
        if not isfinite(margin_um) or margin_um < 0:
            raise ValueError("margin_um must be finite and nonnegative")
        resolved = self._resolve_geometry()
        wavelength_um = self.source.wavelength_um
        offset_um = 3 * self.solver.cell_size_nm / 1000
        planes: list[PlaneMonitor] = []
        for port in resolved.ports.values():
            if port.port_type != "optical":
                continue
            nx, ny, _ = port.normal
            if port.layer_key is None:
                raise FDTDGeometryError(
                    f"Optical port {port.name!r} has no owning layer"
                )
            owners = [resolved.layers[key] for key in port.physical_layer_keys]
            z_lower, z_upper = port.z_bounds or (
                min(layer.z_bounds[0] for layer in owners),
                max(layer.z_bounds[1] for layer in owners),
            )
            width_um = port.width + 2 * margin_um
            height_um = z_upper - z_lower + 2 * margin_um
            normal: SignedAxis = (
                ("+x" if nx > 0 else "-x") if nx else ("+y" if ny > 0 else "-y")
            )
            planes.append(
                PlaneMonitor(
                    name=f"port_{port.name}",
                    center_um=(
                        port.center[0] - nx * offset_um,
                        port.center[1] - ny * offset_um,
                        (z_lower + z_upper) / 2,
                    ),
                    size_um=(0, width_um, height_um)
                    if nx
                    else (width_um, 0, height_um),
                    normal=normal,
                    flux=False,
                    wavelengths_um=[wavelength_um],
                    heatmap=Heatmap(
                        quantity="intensity", wavelengths_um=[wavelength_um]
                    ),
                )
            )
        if xy:
            if z_um is None:
                low, high = resolved.bounds[0][2], resolved.bounds[1][2]
                if self.domain.clip_geometry and self.domain.z_bounds is not None:
                    low = max(low, self.domain.z_bounds[0])
                    high = min(high, self.domain.z_bounds[1])
                z_um = (low + high) / 2
            component = resolved.component
            planes.append(
                PlaneMonitor(
                    name="overview_xy" if xy_name is None else xy_name,
                    center_um=(component.dx, component.dy, z_um),
                    size_um=(component.dxsize, component.dysize, 0),
                    normal="+z",
                    flux=False,
                    wavelengths_um=[wavelength_um],
                    heatmap=Heatmap(
                        quantity="intensity", wavelengths_um=[wavelength_um]
                    ),
                )
            )
        existing_names = {monitor.name for monitor in self.monitors}
        for plane in planes:
            if resolve_monitor_name(plane.name, existing_names) in existing_names:
                raise ValueError(f"monitor {plane.name!r} already exists")
            existing_names.add(plane.name)
        if self.domain.clip_geometry:
            bounds = self._background_bounds_nm()
            for plane in planes:
                monitor_region_nm(plane, bounds, warn=False)
        if xy and xy_name is None:
            warnings.warn(
                "The default XY monitor name 'overview_xy' is deprecated and "
                "will change to 'field_xy' no earlier than 0.11.0 and only after "
                "at least two feature releases and 12 months have passed since "
                "the first released warning. "
                "Pass xy_name='field_xy' to adopt the new name, or "
                "xy_name='overview_xy' to retain the existing generated name.",
                DeprecationWarning,
                stacklevel=2,
            )
        for plane in planes:
            self.monitors.add(plane)
        return self

    def _resolve_geometry(self) -> ResolvedPassivePcell:
        """Resolve current geometry at the current source wavelength."""
        if self.geometry.component is None:
            raise FDTDGeometryError(
                "No geometry is configured. Call Simulation.geometry(...) first."
            )
        pdk = self._pdk_for_resolution()
        self._resolved = resolve_domain_geometry(
            self.geometry,
            pdk=pdk,
            wavelength_um=self.source.wavelength_um,
            domain=self.domain,
            background_material=self.materials.background,
        )
        from gsim.fdtd.geometry_clipping import warn_geometry_clipping

        self._clipping_warning_key = warn_geometry_clipping(
            self._resolved, self.geometry, self.domain, self._clipping_warning_key
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
        cards = dict(get_project_material_cards(self.pdk, include_aliases=False))
        cards.update(
            {
                name: _constant_material_card(name, material.refractive_index)
                for name, material in self._canonical_material_overrides().items()
            }
        )
        return _PdkWithMaterialOverrides(
            pdk_object, cards, get_project_material_aliases(self.pdk)
        )

    @property
    def _background_material_name(self) -> str:
        """Use the same canonical background for geometry and generated inputs."""
        return canonical_material_name(self.materials.background, self.pdk)

    def _canonical_material_overrides(self) -> dict[str, Any]:
        """Keep legacy override names working with canonical PDK layer names."""
        overrides = {}
        for name, material in self.materials.overrides.items():
            canonical = canonical_material_name(name, self.pdk)
            if canonical in overrides and overrides[canonical] != material:
                raise FDTDConfigError(
                    f"Conflicting material overrides for {canonical!r}."
                )
            overrides[canonical] = material
        return overrides

    def _material_snapshots(self) -> dict[str, MaterialSnapshot]:
        """Resolve project-first materials, then apply explicit index overrides."""
        snapshots = dict(self.resolved.materials)
        background = self._background_material_name
        overrides = self._canonical_material_overrides()
        if background not in snapshots:
            try:
                project_cards = dict(get_project_material_cards(self.pdk))
                if background in overrides:
                    project_cards[background] = _constant_material_card(
                        background,
                        overrides[background].refractive_index,
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
        for name, material in overrides.items():
            if name not in snapshots:
                raise FDTDConfigError(
                    f"Material override {name!r} is not used by the geometry "
                    "or background."
                )
            snapshots[name] = replace(
                snapshots[name],
                refractive_index=material.refractive_index,
                extinction_coefficient=0.0,
                card=_constant_material_card(name, material.refractive_index),
            )
        return snapshots

    def _prepare_upload_dir(self) -> Path:
        """Write the cloud artifacts to a fresh staging directory."""
        directory = Path(mkdtemp(prefix="fdtd_"))
        self.write(directory)
        return directory

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
        self._validate_monitor_regions()
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
        config = self._config(manifest, material_snapshots)
        resolve_fiber_media(
            config,
            mesh_path,
            manifest,
            material_snapshots,
            monitors=self.monitors,
            source=self.source,
        )
        config_path.write_text(
            config.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf8",
            newline="\n",
        )
        write_metadata(manifest, config, mesh_path)
        self._last_artifacts = SimulationArtifacts(
            mesh_path=mesh_path,
            config_path=config_path,
            manifest=manifest,
        )
        return self._last_artifacts


__all__ = ["Simulation"]
