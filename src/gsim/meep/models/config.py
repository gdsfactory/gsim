"""Configuration models for MEEP photonic simulation.

Defines serializable Pydantic models for the complete simulation config
that gets written as JSON and consumed by the cloud MEEP runner script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class SymmetryEntry(BaseModel):
    """One MEEP mirror symmetry plane."""

    model_config = ConfigDict(validate_assignment=True)

    direction: Literal["X", "Y", "Z"]
    phase: Literal[1, -1]


class DomainConfig(BaseModel):
    """Simulation domain sizing: margins around geometry + PML thickness.

    Margins control how much material (from the layer stack) is kept around
    the waveguide core.  ``set_z_crop()`` uses ``margin_z_above`` /
    ``margin_z_below`` to determine the crop window.  Along XY the margin
    is the gap between the geometry bounding-box and the PML inner edge.

    Cell size formula:
        cell_x = bbox_width  + 2*(margin_xy + dpml)
        cell_y = bbox_height + 2*(margin_xy + dpml)
        cell_z = z_extent + 2*dpml  (z-margins baked into z_extent)
    """

    model_config = ConfigDict(validate_assignment=True)

    dpml: float = Field(ge=0, description="PML thickness in um")
    margin_xy: float = Field(
        ge=0, description="XY margin between geometry and PML in um"
    )
    margin_z_above: float = Field(
        ge=0, description="Z margin above core kept by set_z_crop in um"
    )
    margin_z_below: float = Field(
        ge=0, description="Z margin below core kept by set_z_crop in um"
    )
    port_margin: float = Field(
        ge=0,
        description="Margin on each side of port width for monitors (um)",
    )
    extend_ports: float = Field(
        ge=0,
        description="Length to extend waveguide ports into PML in um. "
        "0 = auto (margin_xy + dpml).",
    )
    source_port_offset: float = Field(
        ge=0,
        description="Distance to offset source from port center into device (um).",
    )
    distance_source_to_monitors: float = Field(
        ge=0,
        description="Distance between source and its port monitor (um). "
        "Source-port monitor is placed this far past the source into the device.",
    )


class StoppingConfig(BaseModel):
    """Controls when the MEEP simulation stops.

    ``field_decay`` mode (recommended, matches MEEP tutorials) monitors
    a field component at a point and stops when |component|² decays by
    ``threshold`` from its peak, with ``max_time`` as a numeric time cap
    (whichever fires first).

    ``energy_decay`` mode monitors total electromagnetic energy in the
    cell and stops when it decays by ``threshold`` from its peak.

    ``dft_decay`` mode monitors convergence of all DFT monitors and
    stops when they stabilize.  ``dft_min_run_time`` is an *absolute*
    sim time (not time-after-sources) — with a broadband source turning
    off at ~t=78, a min_run_time=100 starts checking at t=100, only ~22
    time units after the source ends.

    ``fixed`` mode runs for a fixed time after sources turn off.
    """

    model_config = ConfigDict(validate_assignment=True)

    mode: Literal["fixed", "field_decay", "dft_decay", "energy_decay"]
    max_time: float = Field(gt=0, serialization_alias="run_after_sources")
    decay_dt: float = Field(gt=0)
    decay_component: str
    threshold: float = Field(gt=0, lt=1, serialization_alias="decay_by")
    decay_monitor_port: str | None = Field(default=None)
    dft_min_run_time: float = Field(
        ge=0,
        description="Minimum absolute sim time for dft_decay mode (not "
        "time-after-sources). Must exceed pulse transit time through the "
        "device to avoid false convergence on near-zero fields.",
    )
    wall_time_max: float = Field(
        default=0.0,
        ge=0,
        description="Wall-clock time limit in seconds (0=unlimited). "
        "Orthogonal safety net for all stopping modes.",
    )


class SourceConfig(BaseModel):
    """Source excitation configuration.

    Controls the Gaussian source bandwidth and which port is excited.
    When ``bandwidth`` is ``None`` (auto), ``compute_fwidth`` returns a
    bandwidth ~3x wider than the monitor frequency span (matching
    gplugins' ``dfcen=0.2`` convention) so edge frequencies receive
    adequate spectral power.
    """

    model_config = ConfigDict(validate_assignment=True)

    bandwidth: float | None = Field(
        default=None,
        description=(
            "Source Gaussian bandwidth in wavelength um. None = auto (~3x monitor bw)."
        ),
    )
    port: str | None = Field(
        default=None,
        description="Source port name. None = auto-select first port.",
    )
    fwidth: float = Field(
        default=0.0,
        ge=0,
        description="Computed source fwidth in frequency units. "
        "Set automatically by write_config(); 0 = not yet computed.",
    )

    def compute_fwidth(self, fcen: float, monitor_df: float) -> float:
        """Compute Gaussian source fwidth in frequency units.

        When auto (bandwidth=None), returns ``max(3 * monitor_df, 0.2 * fcen)``
        to ensure edge frequencies have enough spectral power.

        Args:
            fcen: Center frequency (1/um).
            monitor_df: Monitor frequency span (1/um).

        Returns:
            Source fwidth in frequency units (1/um).
        """
        if self.bandwidth is not None:
            # Convert wavelength bandwidth to frequency bandwidth
            wl_center = 1.0 / fcen
            wl_min = wl_center - self.bandwidth / 2
            wl_max = wl_center + self.bandwidth / 2
            return 1.0 / wl_min - 1.0 / wl_max
        return max(3 * monitor_df, 0.2 * fcen)


class WavelengthConfig(BaseModel):
    """Wavelength and frequency settings for MEEP FDTD simulation.

    MEEP uses normalized units where c = 1 and lengths are in um.
    Frequency f = 1/wavelength (in 1/um).
    """

    model_config = ConfigDict(validate_assignment=True)

    wavelength: float = Field(gt=0, description="Center wavelength in um")
    bandwidth: float = Field(ge=0, description="Wavelength bandwidth in um")
    num_freqs: int = Field(ge=1, description="Number of frequency points")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fcen(self) -> float:
        """Center frequency in MEEP units (1/um, since c=1)."""
        return 1.0 / self.wavelength

    @computed_field  # type: ignore[prop-decorator]
    @property
    def df(self) -> float:
        """Frequency width in MEEP units."""
        wl_min = self.wavelength - self.bandwidth / 2
        wl_max = self.wavelength + self.bandwidth / 2
        f_max = 1.0 / wl_min
        f_min = 1.0 / wl_max
        return f_max - f_min


class ResolutionConfig(BaseModel):
    """MEEP grid resolution configuration."""

    model_config = ConfigDict(validate_assignment=True)

    pixels_per_um: int = Field(ge=4, description="Grid pixels per micrometer")

    @classmethod
    def coarse(cls) -> ResolutionConfig:
        """Coarse resolution (16 pixels/um) for quick tests."""
        return cls(pixels_per_um=16)

    @classmethod
    def default(cls) -> ResolutionConfig:
        """Default resolution (32 pixels/um)."""
        return cls(pixels_per_um=32)

    @classmethod
    def fine(cls) -> ResolutionConfig:
        """Fine resolution (64 pixels/um) for production runs."""
        return cls(pixels_per_um=64)


class AccuracyConfig(BaseModel):
    """Controls MEEP subpixel averaging and polygon simplification.

    ``eps_averaging`` toggles MEEP's subpixel smoothing (expensive for
    complex polygons).  ``subpixel_maxeval`` / ``subpixel_tol`` tune the
    integration that implements it.  ``simplify_tol`` applies Shapely
    polygon simplification before extrusion, dramatically reducing vertex
    counts on dense GDS curves.
    """

    model_config = ConfigDict(validate_assignment=True)

    eps_averaging: bool = Field(description="Toggle subpixel averaging")
    subpixel_maxeval: int = Field(
        ge=0, description="Cap on integration evaluations (0=unlimited)"
    )
    subpixel_tol: float = Field(gt=0, description="Subpixel integration tolerance")
    simplify_tol: float = Field(
        ge=0, description="Shapely simplification tolerance in um (0=off)"
    )


class DiagnosticsConfig(BaseModel):
    """Controls diagnostic outputs from the MEEP runner."""

    model_config = ConfigDict(validate_assignment=True)

    save_geometry: bool = Field(description="Pre-run geometry cross-section plots")
    save_fields: bool = Field(description="Post-run field snapshot")
    save_epsilon_raw: bool = Field(description="Raw epsilon .npy (advanced)")
    save_animation: bool = Field(
        description="Field animation MP4 (heavy, needs ffmpeg)"
    )
    animation_interval: float = Field(
        gt=0,
        description="MEEP time units between animation frames",
    )
    preview_only: bool = Field(
        description="Init sim and save geometry diagnostics, skip FDTD run",
    )
    verbose_interval: float = Field(
        ge=0,
        description="MEEP time units between progress prints (0=off)",
    )


class PortData(BaseModel):
    """Serializable port data for the config JSON."""

    model_config = ConfigDict(validate_assignment=True)

    name: str
    center: list[float] = Field(description="[x, y, z] center coordinates")
    orientation: float = Field(description="Port orientation in degrees")
    width: float = Field(gt=0)
    normal_axis: int = Field(ge=0, le=1, description="0=x, 1=y")
    direction: Literal["+", "-"] = Field(description="Direction along normal axis")
    is_source: bool = False


class LayerStackEntry(BaseModel):
    """One layer in the stack config sent to the cloud runner.

    The cloud runner reads these entries alongside layout.gds to
    extract polygons per layer and extrude them to 3D prisms.
    """

    model_config = ConfigDict(validate_assignment=True)

    layer_name: str
    gds_layer: list[int] = Field(description="[layer_number, datatype]")
    zmin: float
    zmax: float
    material: str
    sidewall_angle: float = Field(default=0.0, description="Sidewall angle in degrees")


class MaterialData(BaseModel):
    """Optical material data for config JSON."""

    model_config = ConfigDict(validate_assignment=True)

    refractive_index: float = Field(gt=0)
    extinction_coeff: float = Field(default=0.0, ge=0)


class SimConfig(BaseModel):
    """Complete serializable simulation config written as JSON.

    This is the top-level config that the cloud MEEP runner reads.
    The geometry is NOT included here — it's in the GDS file.
    The layer_stack tells the runner how to extrude each GDS layer.
    """

    model_config = ConfigDict(validate_assignment=True)

    gds_filename: str = Field(description="GDS file with 2D layout")
    component_bbox: list[float] | None = Field(
        default=None,
        description=(
            "Original component bbox [xmin, ymin, xmax, ymax] before port extension."
        ),
    )
    layer_stack: list[LayerStackEntry]
    dielectrics: list[dict[str, Any]]
    ports: list[PortData]
    materials: dict[str, MaterialData]
    wavelength: WavelengthConfig = Field(serialization_alias="fdtd")
    source: SourceConfig
    stopping: StoppingConfig
    resolution: ResolutionConfig
    domain: DomainConfig
    accuracy: AccuracyConfig
    verbose_interval: float = Field(
        ge=0, description="MEEP time units between progress prints (0=off)"
    )
    diagnostics: DiagnosticsConfig
    symmetries: list[SymmetryEntry]
    split_chunks_evenly: bool = Field(default=False)
    meep_np: int = Field(default=1, ge=1, description="Recommended MPI process count")

    def to_json(self, path: str | Path) -> Path:
        """Write config to JSON file.

        Args:
            path: Output file path

        Returns:
            Path to the written file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(by_alias=True), indent=2))
        return path
