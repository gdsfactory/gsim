"""Public Simulation workflow for ZapFDTD artifact generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from gsim.common.materials import (
    MaterialResolutionError,
    MaterialSnapshot,
    get_project_material_cards,
    resolve_material_snapshot,
)
from gsim.common.pdk import ResolvedPassivePcell, resolve_passive_pcell
from gsim.fdtd.config import ZapConfig, build_zap_config
from gsim.fdtd.mesh import generate_mesh
from gsim.fdtd.models import (
    FDTDConfigError,
    FDTDGeometryError,
    MeshManifest,
    SimulationArtifacts,
)


@dataclass
class Simulation:
    """Generate coarse Gmsh and config artifacts for passive ZapFDTD runs."""

    pdk: Any | None = None
    wavelength_um: float = 1.55
    background_material: str = "SiO2"
    nanometers_per_cell: float = 31.25
    pml_cells: int = 32
    wavelength_halfspan_um: float = 0.05
    num_wavelengths: int = 11
    default_port: str | None = None
    background_padding_um: float = 1.0
    mesh_size_nm: float = 500.0
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
        }
        for name, value in positive_values.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
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
    ) -> ZapConfig:
        """Build the validated Zap schema after mesh group tags are known."""
        if not self.resolved.ports:
            raise FDTDConfigError("Eigenmode FDTD requires at least one port.")
        default_port = self.default_port or next(iter(self.resolved.ports))
        try:
            return build_zap_config(
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
            )
        except ValidationError as error:
            raise FDTDConfigError(f"Invalid ZapFDTD configuration: {error}") from error

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
        )
        config = self._config(manifest, material_snapshots)
        config_path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf8")
        return SimulationArtifacts(
            mesh_path=mesh_path,
            config_path=config_path,
            manifest=manifest,
        )


__all__ = ["Simulation"]
