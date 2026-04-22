"""Declarative API models for MEEP photonic simulation.

Typed physics objects that map to the underlying SimConfig JSON contract.
These models are the user-facing API; the translation to SimConfig happens
in ``Simulation.write_config()``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class Geometry(BaseModel):
    """Physical layout: component + layer stack + optional z-crop."""

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    component: Any = None
    stack: Any = None
    z_crop: str | None = Field(
        default=None,
        description='Z-crop mode: "auto" | layer_name | None (no crop)',
    )
    y_cut: float | None = Field(
        default=None,
        description=(
            "Y coordinate of the XZ cross-section cut (um). "
            "Only meaningful when solver.is_3d=False and solver.plane='xz'. "
            "None -> resolved to the component bbox Y-center at build time."
        ),
    )

    def __call__(self, **kwargs: Any) -> Geometry:
        """Update fields in place. Returns self for chaining."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


class Material(BaseModel):
    """Optical material properties."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    n: float = Field(gt=0, description="Refractive index")
    k: float = Field(default=0.0, ge=0, description="Extinction coefficient")


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class ModeSource(BaseModel):
    """Mode source excitation and spectral measurement window."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    port: str | None = Field(
        default=None,
        description="Source port name. None = auto-select first port.",
    )
    wavelength: float = Field(
        default=1.55,
        gt=0,
        description="Center wavelength in um",
    )
    wavelength_span: float = Field(
        default=0.1,
        ge=0,
        description="Wavelength span of the measurement frequency grid in um.",
    )

    def __call__(self, **kwargs: Any) -> ModeSource:
        """Update fields in place. Returns self for chaining."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


class FiberSource(BaseModel):
    """Tilted Gaussian-beam source above the chip (fiber-to-chip coupling).

    The beam center sits at (``x``, ``z``) in the XZ plane — ``z`` is the
    absolute Z coordinate of the beam plane (um). The beam tilts from the
    +Z normal by ``angle_deg`` toward +X. Only valid in XZ 2D mode
    (``solver.is_3d=False, solver.plane='xz'``).

    Beam waist convention (matches MEEP's ``beam_w0``):

    - ``waist`` is the 1/e² intensity *radius* — i.e. MFD / 2.
    - MFD (mode-field diameter) = 2 · ``waist``.

    Typical single-mode fibers:

    ============  ==========  =========  ==============
    Fiber         Wavelength  MFD (µm)   waist w0 (µm)
    ============  ==========  =========  ==============
    SMF-28        1310 nm     ~9.2       ~4.6
    SMF-28        1550 nm     ~10.4      ~5.2
    UHNA4         1550 nm     ~4.0       ~2.0
    ============  ==========  =========  ==============
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    x: float = Field(description="Beam-center X on the chip plane (um)")
    z: float = Field(description="Absolute Z of the beam plane (um)")
    angle_deg: float = Field(
        default=0.0,
        description="Tilt from +Z normal; positive tilts toward +X (degrees)",
    )
    waist: float = Field(
        gt=0,
        description=(
            "Gaussian beam waist w0 (um) — 1/e² intensity radius = MFD / 2. "
            "Standard SMF-28 at 1550 nm: w0 ~= 5.2 um (MFD ~= 10.4 um)."
        ),
    )
    wavelength: float = Field(default=1.55, gt=0, description="Center wavelength (um)")
    wavelength_span: float = Field(
        default=0.05, ge=0, description="Wavelength span (um)"
    )
    polarization: Literal["TE", "TM"] = Field(
        default="TE",
        description=(
            "PIC convention. 'TE' -> E along waveguide width (Ey, out of XZ "
            "plane); 'TM' -> E in the XZ plane (Ex)."
        ),
    )

    def __call__(self, **kwargs: Any) -> FiberSource:
        """Update fields in place. Returns self for chaining."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


class Symmetry(BaseModel):
    """Mirror symmetry plane."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    direction: Literal["X", "Y", "Z"]
    phase: Literal[1, -1] = Field(default=1)


class Domain(BaseModel):
    """Computational domain sizing: PML + margins + symmetries."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

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
        description="Margin on each side of port width for monitors (um)",
    )
    extend_ports: float = Field(
        default=0.0,
        ge=0,
        description="Extend ports into PML (um). 0 = auto (margin + pml).",
    )
    source_port_offset: float = Field(
        default=0.1,
        ge=0,
        description="Distance to offset source from port center into device (um). "
        "Matches gplugins port_source_offset.",
    )
    distance_source_to_monitors: float = Field(
        default=0.2,
        ge=0,
        description="Distance between source and its port monitor (um). "
        "The source-port monitor is placed this far past the source, "
        "deeper into the device. Matches gplugins distance_source_to_monitors.",
    )
    symmetries: list[Symmetry] = Field(
        default_factory=list,
        description="Mirror symmetry planes. Not yet used in production runs.",
    )

    def __call__(self, **kwargs: Any) -> Domain:
        """Update fields in place. Returns self for chaining."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# FDTD solver
# ---------------------------------------------------------------------------


class FDTD(BaseModel):
    """Solver numerics: resolution, stopping, subpixel, diagnostics."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    is_3d: bool = Field(
        default=True,
        description=(
            "Run a full 3D simulation (True) or an effective-index 2D "
            "simulation (False). 2D collapses the z-dimension, ignores "
            "sidewall angles, and enforces transverse-electric parity "
            "(EVEN_Y+ODD_Z)."
        ),
    )
    plane: Literal["xy", "xz"] = Field(
        default="xy",
        description=(
            "2D simulation plane. 'xy' is the effective-index top-down sim; "
            "'xz' is a vertical cross-section (for grating couplers and "
            "edge couplers). Only meaningful when is_3d=False."
        ),
    )
    resolution: int = Field(default=32, ge=4, description="Pixels per micrometer")

    # Stopping criteria (flat fields instead of variant classes)
    stopping: Literal["fixed", "field_decay", "dft_decay", "energy_decay"] = Field(
        default="energy_decay",
        description=(
            "Stopping mode: 'energy_decay' (recommended) monitors total EM energy; "
            "'field_decay' monitors a field component at a point; "
            "'dft_decay' waits for DFT convergence; "
            "'fixed' runs for max_time."
        ),
    )
    max_time: float = Field(
        default=2000.0, gt=0, description="Max run time after sources (um/c)"
    )
    stopping_threshold: float = Field(
        default=0.01, gt=0, lt=1, description="Decay/convergence threshold"
    )
    stopping_min_time: float = Field(
        default=100.0,
        ge=0,
        description=(
            "Minimum absolute sim time for dft_decay mode (not time-after-sources). "
            "Must exceed pulse transit time to avoid false convergence."
        ),
    )
    stopping_component: str = Field(
        default="Ey", description="Field component for field_decay mode"
    )
    stopping_dt: float = Field(
        default=20.0,
        gt=0,
        description="Decay measurement window for field_decay/energy_decay modes",
    )
    stopping_monitor_port: str | None = Field(
        default=None, description="Port to monitor for field_decay mode"
    )
    wall_time_max: float = Field(
        default=0.0,
        ge=0,
        description="Wall-clock time limit in seconds (0=unlimited). "
        "Orthogonal safety net — stops the sim if real time exceeds this.",
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

    # -- Convenience methods for stopping configuration --

    def stop_when_energy_decayed(
        self, dt: float = 20.0, decay_by: float = 0.01
    ) -> FDTD:
        """Stop when total field energy in the cell decays (recommended).

        Monitors total electromagnetic energy and stops when it has decayed
        by ``decay_by`` from its peak value.  More robust than ``dft_decay``
        for devices where DFTs can falsely converge on near-zero fields.

        Args:
            dt: Time window between energy checks (MEEP time units).
            decay_by: Fractional energy decay threshold (e.g. 0.01 = 1%).

        Returns:
            self (for fluent chaining).
        """
        self.stopping = "energy_decay"
        self.stopping_dt = dt
        self.stopping_threshold = decay_by
        return self

    def stop_when_dft_decayed(self, tol: float = 1e-3, min_time: float = 100.0) -> FDTD:
        """Stop when all DFT monitors converge.

        Args:
            tol: DFT convergence tolerance.
            min_time: Minimum absolute sim time before checking convergence.

        Returns:
            self (for fluent chaining).
        """
        self.stopping = "dft_decay"
        self.stopping_threshold = tol
        self.stopping_min_time = min_time
        return self

    def stop_when_fields_decayed(
        self,
        dt: float = 50.0,
        component: str = "Ey",
        decay_by: float = 0.05,
        monitor_port: str | None = None,
    ) -> FDTD:
        """Stop when a field component decays at a point (recommended).

        Matches the standard MEEP tutorial stopping condition.  Monitors
        |component|² at a point and stops when it decays by ``decay_by``
        from its peak value.

        Args:
            dt: Decay measurement time window.
            component: Field component name (e.g. "Ey", "Hz").
            decay_by: Fractional decay threshold (e.g. 0.05 = 5%).
            monitor_port: Port to monitor (None = first non-source port).

        Returns:
            self (for fluent chaining).
        """
        self.stopping = "field_decay"
        self.stopping_dt = dt
        self.stopping_component = component
        self.stopping_threshold = decay_by
        self.stopping_monitor_port = monitor_port
        return self

    def stop_after_sources(self, time: float) -> FDTD:
        """Run for a fixed sim-time after sources turn off.

        Args:
            time: Run time after sources in MEEP time units (um/c).

        Returns:
            self (for fluent chaining).
        """
        self.stopping = "fixed"
        self.max_time = time
        return self

    def stop_after_walltime(self, seconds: float) -> FDTD:
        """Set a wall-clock time limit (safety net).

        This is orthogonal to the sim-time stopping mode — it caps
        how long the FDTD run is allowed to take in real (wall) seconds.
        Combine with any other stopping method.

        Args:
            seconds: Maximum wall-clock seconds for the FDTD run.

        Returns:
            self (for fluent chaining).
        """
        self.wall_time_max = seconds
        return self

    def __call__(self, **kwargs: Any) -> FDTD:
        """Update fields in place. Returns self for chaining."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    # Diagnostics — output control for plots, fields, animations
    save_geometry: bool = Field(default=True)
    save_fields: bool = Field(default=True)
    save_epsilon_raw: bool = Field(default=False)
    save_animation: bool = Field(default=False)
    animation_interval: float = Field(default=0.5, gt=0)
    preview_only: bool = Field(default=False)
    verbose_interval: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_plane_vs_3d(self) -> FDTD:
        """Reject ``plane='xz'`` when ``is_3d`` is True."""
        if self.is_3d and self.plane == "xz":
            raise ValueError("plane='xz' requires is_3d=False")
        return self
