"""Declarative API models for MEEP photonic simulation.

Typed physics objects that map to the underlying SimConfig JSON contract.
These models are the user-facing API; the translation to SimConfig happens
in ``Simulation.write_config()``.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class Geometry(BaseModel):
    """Physical layout: component + layer stack + optional z-crop."""

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    component: Any = None
    stack: Any = None
    z_crop: str | None = Field(
        default=None,
        description='Z-crop mode: "auto" | layer_name | None (no crop)',
    )


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


class Material(BaseModel):
    """Optical material properties."""

    model_config = ConfigDict(validate_assignment=True)

    n: float = Field(gt=0, description="Refractive index")
    k: float = Field(default=0.0, ge=0, description="Extinction coefficient")


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class ModeSource(BaseModel):
    """Mode source excitation and spectral measurement window."""

    model_config = ConfigDict(validate_assignment=True)

    port: str | None = Field(
        default=None,
        description="Source port name. None = auto-select first port.",
    )
    wavelength: float = Field(
        default=1.55,
        gt=0,
        description="Center wavelength in um",
    )
    bandwidth: float = Field(
        default=0.1,
        ge=0,
        description="Measurement wavelength bandwidth in um",
    )
    num_freqs: int = Field(
        default=11,
        ge=1,
        description="Number of frequency points",
    )


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


class Symmetry(BaseModel):
    """Mirror symmetry plane."""

    model_config = ConfigDict(validate_assignment=True)

    direction: Literal["X", "Y", "Z"]
    phase: Literal[1, -1] = Field(default=1)


class Domain(BaseModel):
    """Computational domain sizing: PML + margins + symmetries."""

    model_config = ConfigDict(validate_assignment=True)

    pml: float = Field(default=1.0, ge=0, description="PML thickness in um")
    margin: float = Field(
        default=0.5,
        ge=0,
        description="XY margin between geometry and PML in um",
    )
    margin_z_above: float = Field(
        default=0.5, ge=0, description="Z margin above core in um"
    )
    margin_z_below: float = Field(
        default=0.5, ge=0, description="Z margin below core in um"
    )
    port_margin: float = Field(
        default=0.5,
        ge=0,
        description="Margin on each side of port waveguide width for mode monitors in um",
    )
    extend_ports: float = Field(
        default=0.0,
        ge=0,
        description="Length to extend waveguide ports into PML in um. 0 = auto (margin + pml).",
    )
    symmetries: list[Symmetry] = Field(
        default_factory=list,
        description="Mirror symmetry planes. Not yet used in production runs.",
    )


# ---------------------------------------------------------------------------
# Stopping variants
# ---------------------------------------------------------------------------


class FixedTime(BaseModel):
    """Run for a fixed time after sources turn off."""

    model_config = ConfigDict(validate_assignment=True)

    max_time: float = Field(default=100.0, gt=0)


class FieldDecay(BaseModel):
    """Stop when field amplitude decays at a monitor point."""

    model_config = ConfigDict(validate_assignment=True)

    max_time: float = Field(default=100.0, gt=0)
    threshold: float = Field(default=1e-3, gt=0, lt=1)
    component: str = Field(default="Ey")
    dt: float = Field(default=50.0, gt=0)
    monitor_port: str | None = Field(default=None)


class DFTDecay(BaseModel):
    """Stop when all DFT monitors converge. Best for S-parameters."""

    model_config = ConfigDict(validate_assignment=True)

    max_time: float = Field(default=100.0, gt=0)
    threshold: float = Field(default=1e-3, gt=0, lt=1)
    min_time: float = Field(
        default=100.0,
        ge=0,
        description="Minimum run time after sources. "
        "Must exceed pulse transit time through the device.",
    )


# ---------------------------------------------------------------------------
# FDTD solver
# ---------------------------------------------------------------------------


class FDTD(BaseModel):
    """Solver numerics: resolution, stopping, subpixel, diagnostics."""

    model_config = ConfigDict(validate_assignment=True)

    resolution: int = Field(default=32, ge=4, description="Pixels per micrometer")
    stopping: Union[FixedTime, FieldDecay, DFTDecay] = Field(
        default_factory=FixedTime
    )
    subpixel: bool = Field(default=False, description="Toggle subpixel averaging")
    subpixel_maxeval: int = Field(
        default=0, ge=0, description="Cap on integration evaluations (0=unlimited)"
    )
    subpixel_tol: float = Field(
        default=1e-4, gt=0, description="Subpixel integration tolerance"
    )
    simplify_tol: float = Field(
        default=0.0,
        ge=0,
        description="Shapely simplification tolerance in um (0=off)",
    )

    # Diagnostics — output control for plots, fields, animations
    save_geometry: bool = Field(default=True)
    save_fields: bool = Field(default=True)
    save_epsilon_raw: bool = Field(default=False)
    save_animation: bool = Field(default=False)
    animation_interval: float = Field(default=0.5, gt=0)
    preview_only: bool = Field(default=False)
    verbose_interval: float = Field(default=0, ge=0)
