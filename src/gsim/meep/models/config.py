"""Configuration models for MEEP photonic simulation.

Defines serializable Pydantic models for the complete simulation config
that gets written as JSON and consumed by the cloud MEEP runner script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DomainConfig(BaseModel):
    """Simulation domain sizing: margins around geometry + PML thickness.

    Margins control how much material (from the layer stack) is kept around
    the waveguide core.  ``set_z_crop()`` uses ``margin_z_above`` /
    ``margin_z_below`` to determine the crop window.  Along XY the margin
    is the gap between the geometry bounding-box and the PML inner edge.

    Cell size formula:
        cell_x = bbox_width  + 2*(margin_xy + dpml)
        cell_y = bbox_height + 2*(margin_xy + dpml)
        cell_z = z_extent + 2*dpml          (z-margins baked into z_extent via set_z_crop)
    """

    model_config = ConfigDict(validate_assignment=True)

    dpml: float = Field(default=1.0, ge=0, description="PML thickness in um")
    margin_xy: float = Field(
        default=1.0, ge=0, description="XY margin between geometry and PML in um"
    )
    margin_z_above: float = Field(
        default=1.0, ge=0, description="Z margin above core kept by set_z_crop in um"
    )
    margin_z_below: float = Field(
        default=1.0, ge=0, description="Z margin below core kept by set_z_crop in um"
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON config."""
        return self.model_dump()


class FDTDConfig(BaseModel):
    """Wavelength and frequency settings for MEEP FDTD simulation.

    MEEP uses normalized units where c = 1 and lengths are in um.
    Frequency f = 1/wavelength (in 1/um).
    """

    model_config = ConfigDict(validate_assignment=True)

    wavelength: float = Field(default=1.55, gt=0, description="Center wavelength in um")
    bandwidth: float = Field(
        default=0.1, ge=0, description="Wavelength bandwidth in um"
    )
    num_freqs: int = Field(default=21, ge=1, description="Number of frequency points")
    run_after_sources: float = Field(
        default=100.0,
        gt=0,
        description="Time units to run after sources turn off",
    )

    @property
    def fcen(self) -> float:
        """Center frequency in MEEP units (1/um, since c=1)."""
        return 1.0 / self.wavelength

    @property
    def df(self) -> float:
        """Frequency width in MEEP units."""
        wl_min = self.wavelength - self.bandwidth / 2
        wl_max = self.wavelength + self.bandwidth / 2
        f_max = 1.0 / wl_min
        f_min = 1.0 / wl_max
        return f_max - f_min

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON config."""
        return {
            "wavelength": self.wavelength,
            "bandwidth": self.bandwidth,
            "num_freqs": self.num_freqs,
            "fcen": self.fcen,
            "df": self.df,
            "run_after_sources": self.run_after_sources,
        }


class ResolutionConfig(BaseModel):
    """MEEP grid resolution configuration."""

    model_config = ConfigDict(validate_assignment=True)

    pixels_per_um: int = Field(
        default=32, ge=4, description="Grid pixels per micrometer"
    )

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON config."""
        return {"pixels_per_um": self.pixels_per_um}


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return self.model_dump()


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return self.model_dump()


class MaterialData(BaseModel):
    """Optical material data for config JSON."""

    model_config = ConfigDict(validate_assignment=True)

    refractive_index: float = Field(gt=0)
    extinction_coeff: float = Field(default=0.0, ge=0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return self.model_dump()


class SimConfig(BaseModel):
    """Complete serializable simulation config written as JSON.

    This is the top-level config that the cloud MEEP runner reads.
    The geometry is NOT included here — it's in the GDS file.
    The layer_stack tells the runner how to extrude each GDS layer.
    """

    model_config = ConfigDict(validate_assignment=True)

    gds_filename: str = Field(
        default="layout.gds", description="GDS file with 2D layout"
    )
    layer_stack: list[dict[str, Any]] = Field(default_factory=list)
    dielectrics: list[dict[str, Any]] = Field(default_factory=list)
    ports: list[dict[str, Any]] = Field(default_factory=list)
    materials: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fdtd: dict[str, Any] = Field(default_factory=dict)
    resolution: dict[str, Any] = Field(default_factory=dict)
    domain: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, path: str | Path) -> Path:
        """Write config to JSON file.

        Args:
            path: Output file path

        Returns:
            Path to the written file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2))
        return path
