"""Translation from public FDTD concerns to the GDSFactory FDTD runtime schema."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, radians, sin
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from gsim.common.materials import MaterialSnapshot
from gsim.fdtd.api import (
    DipoleSource,
    FiberMode,
    GaussianBeamSource,
    LineCurrentSource,
    PlaneMonitor,
    PortSource,
    Vector3,
)
from gsim.fdtd.config import (
    DipoleConfig,
    ExcitationConfig,
    FDTDConfig,
    FiberModeConfig,
    GaussianBeamConfig,
    HeatmapConfig,
    LineCurrentConfig,
    PlaneMonitorConfig,
    build_fdtd_config,
)
from gsim.fdtd.mesh import background_bounds_nm
from gsim.fdtd.models import FDTDConfigError, MeshManifest

if TYPE_CHECKING:
    from gsim.common.pdk import ResolvedPassivePcell
    from gsim.fdtd.api import Domain, Materials, Monitors, Solver, SourceType


def _scale_vector(vector: Vector3, factor: float) -> Vector3:
    """Scale a fixed-length three-vector without losing its tuple type."""
    return (
        vector[0] * factor,
        vector[1] * factor,
        vector[2] * factor,
    )


def _vector_bounds(center: Vector3, half_size: Vector3) -> tuple[Vector3, Vector3]:
    """Return fixed-length lower and upper bounds around a center."""
    return (
        (
            center[0] - half_size[0],
            center[1] - half_size[1],
            center[2] - half_size[2],
        ),
        (
            center[0] + half_size[0],
            center[1] + half_size[1],
            center[2] + half_size[2],
        ),
    )


class RuntimeConfigMixin:
    """Serialize sources, monitors, and solver controls for GDSFactory FDTD."""

    source: SourceType
    resolved: ResolvedPassivePcell
    materials: Materials
    monitors: Monitors
    domain: Domain
    solver: Solver

    def _background_bounds_nm(
        self,
    ) -> tuple[float, float, float, float, float, float]:
        """Return the resolved automatic or explicitly bounded domain."""
        return background_bounds_nm(
            self.resolved,
            self.materials.background,
            self.domain.padding_um,
            x_bounds=self.domain.x_bounds,
            y_bounds=self.domain.y_bounds,
            z_bounds=self.domain.z_bounds,
        )

    def _validate_explicit_domain_contents(self) -> None:
        """Require public sources and monitors to fit explicit axis bounds."""
        explicit_bounds = (
            self.domain.x_bounds,
            self.domain.y_bounds,
            self.domain.z_bounds,
        )
        if all(bounds is None for bounds in explicit_bounds):
            return
        domain_bounds_nm = self._background_bounds_nm()
        regions: list[
            tuple[str, tuple[float, float, float], tuple[float, float, float]]
        ] = []
        source = self.source
        if isinstance(source, GaussianBeamSource):
            center = _scale_vector(source.center_um, 1000)
            half_size = _scale_vector(source.size_um, 500)
            lower, upper = _vector_bounds(center, half_size)
            regions.append(
                (
                    "Gaussian source aperture",
                    lower,
                    upper,
                )
            )
            focal_point = _scale_vector(source.focal_point_um, 1000)
            regions.append(("Gaussian source focus", focal_point, focal_point))
        elif isinstance(source, DipoleSource):
            position = _scale_vector(source.position_um, 1000)
            regions.append(("Dipole source", position, position))
        elif isinstance(source, LineCurrentSource):
            position = list(_scale_vector(source.position_um, 1000))
            lower = position.copy()
            upper = position.copy()
            axis = {"x": 0, "y": 1, "z": 2}[source.line_axis]
            lower[axis] -= source.length_um * 500
            upper[axis] += source.length_um * 500
            regions.append(
                (
                    "Line-current source",
                    (lower[0], lower[1], lower[2]),
                    (upper[0], upper[1], upper[2]),
                )
            )

        for monitor in self.monitors:
            center = _scale_vector(monitor.center_um, 1000)
            half_size = _scale_vector(monitor.size_um, 500)
            lower, upper = _vector_bounds(center, half_size)
            regions.append(
                (
                    f"Monitor {monitor.name!r}",
                    lower,
                    upper,
                )
            )
            if monitor.fiber_mode is not None:
                focal_point = _scale_vector(monitor.fiber_mode.focal_point_um, 1000)
                regions.append(
                    (f"Monitor {monitor.name!r} fiber focus", focal_point, focal_point)
                )

        for label, region_min, region_max in regions:
            for axis, (axis_name, requested_bounds) in enumerate(
                zip(("x", "y", "z"), explicit_bounds, strict=True)
            ):
                if requested_bounds is None:
                    continue
                lower_nm = domain_bounds_nm[axis]
                upper_nm = domain_bounds_nm[axis + 3]
                if region_min[axis] < lower_nm or region_max[axis] > upper_nm:
                    extent_um = (region_min[axis] / 1000, region_max[axis] / 1000)
                    raise FDTDConfigError(
                        f"{label} {axis_name}-extent {extent_um} exceeds "
                        f"domain.{axis_name}_bounds {requested_bounds}."
                    )

    def _selected_port_name(self) -> str:
        """Validate and return the explicitly selected source port."""
        if not isinstance(self.source, PortSource):
            raise TypeError("Port selection requires a PortSource")
        if self.source.port is None:
            raise FDTDConfigError(
                "PortSource requires an explicit port. "
                "Call simulation.source(port='...') before write() or run()."
            )
        if not self.resolved.ports:
            raise FDTDConfigError("A PortSource requires at least one PDK port.")
        if self.source.port not in self.resolved.ports:
            raise FDTDConfigError(
                f"source port {self.source.port!r} is not present on the component."
            )
        return self.source.port

    @staticmethod
    def _vertical_polarization(port: Any) -> tuple[float, float, float]:
        """Map a vertical TE/TM marker into its in-plane E direction."""
        angle = radians(port.orientation)
        if port.port_type == "vertical_te":
            vector = (-sin(angle), cos(angle), 0.0)
        elif port.port_type == "vertical_tm":
            vector = (cos(angle), sin(angle), 0.0)
        else:
            raise FDTDConfigError(f"Unsupported vertical port type {port.port_type!r}.")
        return (
            round(vector[0], 12),
            round(vector[1], 12),
            round(vector[2], 12),
        )

    def _vertical_port_configs(
        self, material_snapshots: Mapping[str, MaterialSnapshot]
    ) -> dict[str, tuple[GaussianBeamConfig, PlaneMonitorConfig]]:
        """Translate vertical PDK ports into beam and fiber-plane settings."""
        if not isinstance(self.source, PortSource):
            return {}
        vertical_ports = {
            name: port for name, port in self.resolved.ports.items() if port.is_vertical
        }
        if not vertical_ports:
            return {}
        bounds = self._background_bounds_nm()
        background_index = material_snapshots[
            self.materials.background
        ].refractive_index
        outward_sign = 1 if self.source.vertical_axis == "+z" else -1
        aperture_z_nm = bounds[5] if outward_sign > 0 else bounds[2]
        device_z_nm = self.resolved.bounds[1 if outward_sign > 0 else 0][2] * 1000
        monitor_heatmap = self.source.vertical_monitor_heatmap
        heatmap_config = (
            HeatmapConfig(
                quantity=monitor_heatmap.quantity,
                wavelengths=[value * 1000 for value in monitor_heatmap.wavelengths_um],
            )
            if monitor_heatmap is not None
            else None
        )
        configs: dict[str, tuple[GaussianBeamConfig, PlaneMonitorConfig]] = {}
        for name, port in vertical_ports.items():
            aperture_width_um = self.source.vertical_aperture_width_um or port.width
            waist_radius_um = (
                self.source.vertical_waist_radius_um or aperture_width_um / 2
            )
            half_width_nm = aperture_width_um * 500
            center_x_nm = port.center[0] * 1000
            center_y_nm = port.center[1] * 1000
            region_min = (
                center_x_nm - half_width_nm,
                center_y_nm - half_width_nm,
                aperture_z_nm,
            )
            region_max = (
                center_x_nm + half_width_nm,
                center_y_nm + half_width_nm,
                aperture_z_nm,
            )
            if (
                region_min[0] < bounds[0]
                or region_min[1] < bounds[1]
                or region_max[0] > bounds[3]
                or region_max[1] > bounds[4]
            ):
                raise FDTDConfigError(
                    f"Vertical port {name!r} aperture exceeds the background domain; "
                    "increase domain.padding_um or reduce its aperture width."
                )
            polarization = self._vertical_polarization(port)
            focal_point = (center_x_nm, center_y_nm, device_z_nm)
            configs[name] = (
                GaussianBeamConfig(
                    region_min=region_min,
                    region_max=region_max,
                    aperture_normal="-z" if outward_sign > 0 else "+z",
                    propagation_direction=(0, 0, -outward_sign),
                    e_polarization=polarization,
                    focal_point=focal_point,
                    waist_radius=waist_radius_um * 1000,
                    refractive_index=background_index,
                ),
                PlaneMonitorConfig(
                    name=name,
                    region_min=region_min,
                    region_max=region_max,
                    normal=self.source.vertical_axis,
                    heatmap=heatmap_config,
                    fiber_mode=FiberModeConfig(
                        propagation_direction=(0, 0, outward_sign),
                        e_polarization=polarization,
                        focal_point=focal_point,
                        waist_radius=waist_radius_um * 1000,
                        refractive_index=background_index,
                    ),
                ),
            )
        return configs

    def _runtime_excitation(
        self,
        material_snapshots: Mapping[str, MaterialSnapshot],
    ) -> tuple[ExcitationConfig, list[PlaneMonitorConfig]]:
        """Translate the public source intent and additional monitors."""
        source = self.source
        common: dict[str, Any] = {
            "waveform": source.waveform,
            "center_wavelength": source.wavelength_um * 1000,
            "wavelength_halfspan": source.wavelength_halfspan_um * 1000,
            "num_wavelengths": source.num_wavelengths,
            "amplitude": source.amplitude,
        }
        vertical_configs = self._vertical_port_configs(material_snapshots)
        derived_monitors = [monitor for _, monitor in vertical_configs.values()]
        if isinstance(source, PortSource):
            selected_port = self._selected_port_name()
            if self.resolved.ports[selected_port].is_vertical:
                excitation = ExcitationConfig(
                    type="gaussian_beam",
                    gaussian_beam=vertical_configs[selected_port][0],
                    **common,
                )
            else:
                excitation = ExcitationConfig(
                    type="eigenmode", default_port=selected_port, **common
                )
        elif isinstance(source, DipoleSource):
            excitation = ExcitationConfig(
                type="dipole",
                dipole=DipoleConfig(
                    position=_scale_vector(source.position_um, 1000),
                    current_axis=source.current_axis,
                ),
                **common,
            )
        elif isinstance(source, LineCurrentSource):
            excitation = ExcitationConfig(
                type="line_current",
                line_current=LineCurrentConfig(
                    position=_scale_vector(source.position_um, 1000),
                    line_axis=source.line_axis,
                    current_axis=source.current_axis,
                    length=source.length_um * 1000,
                ),
                **common,
            )
        else:
            excitation = ExcitationConfig(
                type="gaussian_beam",
                gaussian_beam=self._explicit_gaussian_beam(source, material_snapshots),
                **common,
            )
        public_monitors = [
            self._plane_monitor_config(monitor, material_snapshots)
            for monitor in self.monitors
        ]
        names = [monitor.name for monitor in [*derived_monitors, *public_monitors]]
        if len(names) != len(set(names)):
            raise FDTDConfigError(
                "Explicit monitor names cannot shadow vertical ports."
            )
        return excitation, [*derived_monitors, *public_monitors]

    def _explicit_gaussian_beam(
        self,
        source: GaussianBeamSource,
        material_snapshots: Mapping[str, MaterialSnapshot],
    ) -> GaussianBeamConfig:
        """Translate an explicitly positioned public Gaussian beam."""
        half_size_nm = tuple(value * 500 for value in source.size_um)
        center_nm = tuple(value * 1000 for value in source.center_um)
        return GaussianBeamConfig(
            region_min=tuple(
                center - half
                for center, half in zip(center_nm, half_size_nm, strict=True)
            ),
            region_max=tuple(
                center + half
                for center, half in zip(center_nm, half_size_nm, strict=True)
            ),
            aperture_normal=source.aperture_normal,
            propagation_direction=source.propagation_direction,
            e_polarization=source.e_polarization,
            focal_point=_scale_vector(source.focal_point_um, 1000),
            waist_radius=source.waist_radius_um * 1000,
            refractive_index=source.refractive_index
            or material_snapshots[self.materials.background].refractive_index,
        )

    def _plane_monitor_config(
        self,
        monitor: PlaneMonitor,
        material_snapshots: Mapping[str, MaterialSnapshot],
    ) -> PlaneMonitorConfig:
        """Translate one public plane-monitor definition."""
        center_nm = tuple(value * 1000 for value in monitor.center_um)
        half_size_nm = tuple(value * 500 for value in monitor.size_um)
        fiber_mode = self._fiber_mode_config(monitor.fiber_mode, material_snapshots)
        return PlaneMonitorConfig(
            name=monitor.name,
            region_min=tuple(
                center - half
                for center, half in zip(center_nm, half_size_nm, strict=True)
            ),
            region_max=tuple(
                center + half
                for center, half in zip(center_nm, half_size_nm, strict=True)
            ),
            normal=monitor.normal,
            flux=monitor.flux,
            wavelengths=(
                [value * 1000 for value in monitor.wavelengths_um]
                if monitor.wavelengths_um is not None
                else None
            ),
            heatmap=(
                HeatmapConfig(
                    quantity=monitor.heatmap.quantity,
                    wavelengths=[
                        value * 1000 for value in monitor.heatmap.wavelengths_um
                    ],
                )
                if monitor.heatmap is not None
                else None
            ),
            fiber_mode=fiber_mode,
        )

    def _fiber_mode_config(
        self,
        fiber_mode: FiberMode | None,
        material_snapshots: Mapping[str, MaterialSnapshot],
    ) -> FiberModeConfig | None:
        """Translate an optional analytic fiber mode."""
        if fiber_mode is None:
            return None
        return FiberModeConfig(
            propagation_direction=fiber_mode.propagation_direction,
            e_polarization=fiber_mode.e_polarization,
            focal_point=_scale_vector(fiber_mode.focal_point_um, 1000),
            waist_radius=fiber_mode.waist_radius_um * 1000,
            refractive_index=fiber_mode.refractive_index
            or material_snapshots[self.materials.background].refractive_index,
        )

    def _config(
        self,
        manifest: MeshManifest,
        material_snapshots: Mapping[str, MaterialSnapshot],
    ) -> FDTDConfig:
        """Build and cross-validate the runtime configuration."""
        excitation, monitors = self._runtime_excitation(material_snapshots)
        try:
            return build_fdtd_config(
                manifest,
                material_snapshots,
                background_material=self.materials.background,
                center_wavelength_nm=self.source.wavelength_um * 1000,
                wavelength_halfspan_nm=self.source.wavelength_halfspan_um * 1000,
                num_wavelengths=self.source.num_wavelengths,
                default_port=None,
                nanometers_per_cell=self.solver.cell_size_nm,
                pml_cells=self.domain.pml_cells,
                max_timesteps=self.solver.max_timesteps,
                energy_decay_fraction=self.solver.energy_decay_fraction,
                max_wall_seconds=self.solver.max_wall_seconds,
                excitation=excitation,
                monitors=monitors,
            )
        except ValidationError as error:
            raise FDTDConfigError(
                f"Invalid GDSFactory FDTD configuration: {error}"
            ) from error


__all__ = ["RuntimeConfigMixin"]
