"""Public Simulation workflow for GDSFactory FDTD artifact generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import cos, isfinite, radians, sin
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from gsim.common.materials import (
    MaterialResolutionError,
    MaterialSnapshot,
    get_project_material_cards,
    resolve_material_snapshot,
)
from gsim.common.pdk import ResolvedPassivePcell, resolve_passive_pcell
from gsim.fdtd.config import (
    FDTDConfig,
    FiberModeConfig,
    GaussianBeamConfig,
    PlaneMonitorConfig,
    build_fdtd_config,
)
from gsim.fdtd.mesh import background_bounds_nm, generate_mesh
from gsim.fdtd.models import (
    FDTDConfigError,
    FDTDGeometryError,
    MeshManifest,
    SimulationArtifacts,
)


@dataclass
class Simulation:
    """Generate coarse Gmsh and config artifacts for GDSFactory FDTD runs."""

    pdk: Any | None = None
    wavelength_um: float = 1.55
    background_material: str = "SiO2"
    nanometers_per_cell: float = 60.0
    pml_cells: int = 32
    wavelength_halfspan_um: float = 0.05
    num_wavelengths: int = 101
    default_port: str | None = None
    background_padding_um: float = 1.0
    mesh_size_nm: float = 1000.0
    geometry_tolerance_nm: float = 10.0
    vertical_port_axis: Literal["+z", "-z"] = "+z"
    vertical_port_aperture_width_um: float | None = None
    vertical_port_waist_radius_um: float | None = None
    max_timesteps: int | None = None
    energy_decay_fraction: float = 1e-6
    max_wall_seconds: float = 3600.0
    _resolved: ResolvedPassivePcell | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate constructor settings that do not depend on geometry."""
        positive_values = {
            "wavelength_um": self.wavelength_um,
            "nanometers_per_cell": self.nanometers_per_cell,
            "background_padding_um": self.background_padding_um,
            "mesh_size_nm": self.mesh_size_nm,
            "geometry_tolerance_nm": self.geometry_tolerance_nm,
        }
        for name, value in positive_values.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.geometry_tolerance_nm > 30:
            raise ValueError("geometry_tolerance_nm must be at most 30.")
        if not self.background_material:
            raise ValueError("background_material cannot be empty.")
        if self.pml_cells < 0:
            raise ValueError("pml_cells cannot be negative.")
        if not 0 <= self.wavelength_halfspan_um < self.wavelength_um:
            raise ValueError(
                "wavelength_halfspan_um must be nonnegative and smaller than "
                "wavelength_um."
            )
        if self.num_wavelengths < 1:
            raise ValueError("num_wavelengths must be at least 1.")
        if self.max_timesteps is not None and self.max_timesteps <= 0:
            raise ValueError("max_timesteps must be positive when provided.")
        if not 0 < self.energy_decay_fraction < 1:
            raise ValueError("energy_decay_fraction must be between 0 and 1.")
        if self.max_wall_seconds < 0:
            raise ValueError("max_wall_seconds cannot be negative.")
        if self.vertical_port_axis not in {"+z", "-z"}:
            raise ValueError("vertical_port_axis must be '+z' or '-z'.")
        for name, value in {
            "vertical_port_aperture_width_um": self.vertical_port_aperture_width_um,
            "vertical_port_waist_radius_um": self.vertical_port_waist_radius_um,
        }.items():
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when provided.")

    @property
    def resolved(self) -> ResolvedPassivePcell:
        """Return stored canonical geometry or fail before geometry setup."""
        if self._resolved is None:
            raise FDTDGeometryError(
                "No geometry is configured. Call Simulation.geometry(...) first."
            )
        return self._resolved

    def geometry(
        self,
        component: Any,
        *,
        settings: Mapping[str, Any] | None = None,
    ) -> ResolvedPassivePcell:
        """Resolve and store a component through the canonical PDK boundary."""
        self._resolved = resolve_passive_pcell(
            component,
            pdk=self.pdk,
            settings=settings,
            wavelength_um=self.wavelength_um,
        )
        return self._resolved

    def _material_snapshots(self) -> dict[str, MaterialSnapshot]:
        """Add a strict project-first background snapshot to layer snapshots."""
        snapshots = dict(self.resolved.materials)
        if self.background_material in snapshots:
            return snapshots
        try:
            project_cards = get_project_material_cards(self.pdk)
            snapshots[self.background_material] = resolve_material_snapshot(
                self.background_material,
                self.wavelength_um,
                project_cards,
            )
        except MaterialResolutionError as error:
            raise FDTDConfigError(
                f"Could not resolve background material "
                f"{self.background_material!r}: {error}"
            ) from error
        return snapshots

    def _config(
        self,
        manifest: MeshManifest,
        material_snapshots: Mapping[str, MaterialSnapshot],
    ) -> FDTDConfig:
        """Build the GDSFactory FDTD schema after mesh group tags are known."""
        if not self.resolved.ports:
            raise FDTDConfigError(
                "GDSFactory FDTD simulations require at least one port."
            )
        selected_port_name = self._selected_port_name()
        selected_port = self.resolved.ports[selected_port_name]
        vertical_configs = self._vertical_port_configs(material_snapshots)
        gaussian_beam = (
            vertical_configs[selected_port_name][0]
            if selected_port.is_vertical
            else None
        )
        default_port = None if selected_port.is_vertical else selected_port_name
        monitors = [monitor for _, monitor in vertical_configs.values()]
        try:
            return build_fdtd_config(
                manifest,
                material_snapshots,
                background_material=self.background_material,
                center_wavelength_nm=self.wavelength_um * 1000,
                wavelength_halfspan_nm=self.wavelength_halfspan_um * 1000,
                num_wavelengths=self.num_wavelengths,
                default_port=default_port,
                nanometers_per_cell=self.nanometers_per_cell,
                pml_cells=self.pml_cells,
                max_timesteps=self.max_timesteps,
                energy_decay_fraction=self.energy_decay_fraction,
                max_wall_seconds=self.max_wall_seconds,
                gaussian_beam=gaussian_beam,
                monitors=monitors,
            )
        except ValidationError as error:
            raise FDTDConfigError(
                f"Invalid GDSFactory FDTD configuration: {error}"
            ) from error

    def _selected_port_name(self) -> str:
        """Validate and return the explicitly selected source port."""
        if self.default_port is None:
            raise FDTDConfigError(
                "default_port is required before writing or running FDTD."
            )
        if self.default_port not in self.resolved.ports:
            raise FDTDConfigError(
                f"default_port {self.default_port!r} is not present on the "
                "resolved component."
            )
        return self.default_port

    @staticmethod
    def _vertical_polarization(port: Any) -> tuple[float, float, float]:
        """Map vertical TE/TM and its in-plane orientation to an E direction."""
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
        self,
        material_snapshots: Mapping[str, MaterialSnapshot],
    ) -> dict[str, tuple[GaussianBeamConfig, PlaneMonitorConfig]]:
        """Translate vertical markers into free-space source/monitor settings."""
        vertical_ports = {
            name: port for name, port in self.resolved.ports.items() if port.is_vertical
        }
        if not vertical_ports:
            return {}
        domain_bounds = background_bounds_nm(
            self.resolved,
            self.background_material,
            self.background_padding_um,
        )
        background_index = material_snapshots[self.background_material].refractive_index
        outward_sign = 1 if self.vertical_port_axis == "+z" else -1
        inward_axis = "-z" if outward_sign > 0 else "+z"
        propagation_inward = (0.0, 0.0, float(-outward_sign))
        propagation_outward = (0.0, 0.0, float(outward_sign))
        aperture_z_nm = domain_bounds[5] if outward_sign > 0 else domain_bounds[2]
        device_z_nm = (
            self.resolved.bounds[1][2]
            if outward_sign > 0
            else self.resolved.bounds[0][2]
        ) * 1000
        configs = {}
        for name, port in vertical_ports.items():
            aperture_width_um = self.vertical_port_aperture_width_um or port.width
            waist_radius_um = (
                self.vertical_port_waist_radius_um or aperture_width_um / 2
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
                region_min[0] < domain_bounds[0]
                or region_min[1] < domain_bounds[1]
                or region_max[0] > domain_bounds[3]
                or region_max[1] > domain_bounds[4]
            ):
                raise FDTDConfigError(
                    f"Vertical port {name!r} aperture exceeds the background "
                    "domain; increase background_padding_um or reduce "
                    "vertical_port_aperture_width_um."
                )
            focal_point = (center_x_nm, center_y_nm, device_z_nm)
            polarization = self._vertical_polarization(port)
            common_fiber = FiberModeConfig(
                propagation_direction=propagation_outward,
                e_polarization=polarization,
                focal_point=focal_point,
                waist_radius=waist_radius_um * 1000,
                refractive_index=background_index,
            )
            configs[name] = (
                GaussianBeamConfig(
                    region_min=region_min,
                    region_max=region_max,
                    aperture_normal=inward_axis,
                    propagation_direction=propagation_inward,
                    e_polarization=polarization,
                    focal_point=focal_point,
                    waist_radius=waist_radius_um * 1000,
                    refractive_index=background_index,
                ),
                PlaneMonitorConfig(
                    name=name,
                    region_min=region_min,
                    region_max=region_max,
                    normal=self.vertical_port_axis,
                    fiber_mode=common_fiber,
                ),
            )
        return configs

    def write(self, output_dir: str | Path) -> SimulationArtifacts:
        """Write ``mesh.msh`` and ``config.json`` into an output directory."""
        resolved = self.resolved
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        mesh_path = directory / "mesh.msh"
        config_path = directory / "config.json"
        material_snapshots = self._material_snapshots()
        manifest = generate_mesh(
            resolved,
            mesh_path,
            background_material=self.background_material,
            background_padding_um=self.background_padding_um,
            mesh_size_nm=self.mesh_size_nm,
            geometry_tolerance_nm=self.geometry_tolerance_nm,
        )
        config = self._config(manifest, material_snapshots)
        config_path.write_text(
            config.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf8",
        )
        return SimulationArtifacts(
            mesh_path=mesh_path,
            config_path=config_path,
            manifest=manifest,
        )


# Keep the historical internal import path aligned with the public workflow.
# The original artifact-only implementation above remains available while the
# compatibility window is open, but new imports receive the concern-based API.
ArtifactSimulation = Simulation
from gsim.fdtd.workflow import (  # noqa: E402
    Simulation as Simulation,  # ty: ignore[invalid-assignment]
)

__all__ = ["ArtifactSimulation", "Simulation"]
