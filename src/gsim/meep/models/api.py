"""Declarative API models for MEEP photonic simulation.

Typed physics objects that map to the underlying SimConfig JSON contract.
These models are the user-facing API; the translation to SimConfig happens
in ``Simulation.write_config()``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class Geometry(BaseModel):
    """Physical layout: component + layer stack.

    The vertical crop reference lives on ``Domain.z_ref`` and the 2D cut
    plane lives on ``FDTD`` (``solver.y_cut`` / ``z_cut``).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    component: Any = None
    stack: Any = None

    def __call__(self, **kwargs: Any) -> Geometry:
        """Update fields in place. Returns self for chaining."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


class Material(BaseModel):
    """EM material properties for MEEP simulation overrides.

    Initialize via **exactly one** of:

    - ``refractive_index`` (+ optional ``extinction_coeff``) — converted to
      permittivity internally.
    - ``permittivity`` — used directly.

    Use ``loss_tangent`` for dielectric loss (converted to conductivity at
    the simulation frequency).
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    permittivity: float | None = Field(
        default=None, gt=0, description="Relative permittivity"
    )
    refractive_index: float | None = Field(
        default=None, gt=0, description="Refractive index (n)"
    )
    extinction_coeff: float | None = Field(
        default=None, ge=0, description="Extinction coefficient (k)"
    )
    loss_tangent: float | None = Field(
        default=None, ge=0, description="Dielectric loss tangent"
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_material(cls, data: Any) -> Any:
        """Derive permittivity from refractive_index / extinction_coeff."""
        if not isinstance(data, dict):
            return data

        n = data.get("refractive_index")
        k = data.get("extinction_coeff")
        eps = data.get("permittivity")
        tan_delta = data.get("loss_tangent")

        if n is not None and eps is not None:
            raise ValueError(
                "Provide either 'refractive_index' or 'permittivity', not both."
            )

        if n is not None:
            k_val = k if k is not None else 0.0
            derived_eps = n**2 - k_val**2
            if derived_eps <= 0:
                raise ValueError(
                    f"refractive_index={n} and extinction_coeff={k_val} "
                    f"yield non-positive permittivity ({derived_eps})."
                )
            data["permittivity"] = derived_eps
            if tan_delta is None:
                data["loss_tangent"] = (2 * n * k_val) / derived_eps
        elif eps is not None:
            data["permittivity"] = eps
            if tan_delta is None:
                data["loss_tangent"] = 0.0
        else:
            raise ValueError("Provide either 'refractive_index' or 'permittivity'.")

        # Ensure loss_tangent is set
        if data.get("loss_tangent") is None:
            data["loss_tangent"] = 0.0

        return data


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
    (``solver.mode='2d'`` with ``solver.y_cut`` set).

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

    z_ref: str | None = Field(
        default=None,
        description=(
            "Vertical crop reference: None = auto (highest-n layer actually "
            "drawn by the component, i.e. the photonic core), 'stack' = full "
            "non-air material stack, or a specific layer name. "
            "margin_z is measured from this reference."
        ),
    )
    pml: float = Field(default=1.0, ge=0, description="PML thickness in um")
    margin_x: float | tuple[float, float] = Field(
        default=0.5,
        description=(
            "Air gap between geometry and PML along X in um. Scalar = both "
            "sides equal; (low, high) = (-x side, +x side)."
        ),
    )
    margin_y: float | tuple[float, float] = Field(
        default=0.5,
        description=(
            "Air gap between geometry and PML along Y in um. Scalar = both "
            "sides equal; (low, high) = (-y side, +y side)."
        ),
    )
    margin_z: float | tuple[float, float] = Field(
        default=0.5,
        description=(
            "Vertical margin around the z_ref reference in um. Scalar = both "
            "sides equal; (low, high) = (below, above)."
        ),
    )
    port_margin: float = Field(
        default=0.5,
        ge=0,
        description="Margin on each side of port width for monitors (um)",
    )
    extend_ports: float = Field(
        default=0.0,
        ge=0,
        description="Extend ports into PML (um). 0 = auto (max XY margin + pml).",
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

    @field_validator("margin_x", "margin_y", "margin_z", mode="before")
    @classmethod
    def _validate_margin(cls, v: Any) -> Any:
        """Accept a non-negative scalar or a (low, high) tuple of them."""
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("margin must be >= 0")
            return float(v)
        try:
            low, high = v
        except (TypeError, ValueError):
            raise ValueError("margin must be a number or a (low, high) tuple") from None
        if low < 0 or high < 0:
            raise ValueError("margin sides must be >= 0")
        return (float(low), float(high))

    @staticmethod
    def _as_pair(v: float | tuple[float, float]) -> tuple[float, float]:
        """Normalize a scalar or (low, high) tuple to a (low, high) pair."""
        if isinstance(v, tuple):
            return v
        return (float(v), float(v))

    def resolved_margin_x(self) -> tuple[float, float]:
        """Return XY margin as ``(-x side, +x side)``."""
        return self._as_pair(self.margin_x)

    def resolved_margin_y(self) -> tuple[float, float]:
        """Return XY margin as ``(-y side, +y side)``."""
        return self._as_pair(self.margin_y)

    def resolved_margin_z(self) -> tuple[float, float]:
        """Return vertical margin as ``(below, above)``."""
        return self._as_pair(self.margin_z)

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

    mode: Literal["2d", "3d"] = Field(
        default="3d",
        description=(
            "Dimensionality: '3d' runs a full 3D simulation; '2d' runs an "
            "effective-index / cross-section simulation. In 2D the plane is "
            "implied by which of x_cut/y_cut/z_cut is set."
        ),
    )
    x_cut: float | Literal["auto"] | None = Field(
        default=None,
        description=(
            "YZ cross-section plane (perpendicular to X). Reserved — not yet "
            "implemented. 'auto' = bbox-centered."
        ),
    )
    y_cut: float | Literal["auto"] | None = Field(
        default=None,
        description=(
            "XZ cross-section plane (perpendicular to Y), for grating and "
            "edge couplers. 'auto' = component bbox Y-center; a float sets the "
            "Y coordinate of the cross-section (um)."
        ),
    )
    z_cut: float | Literal["auto"] | None = Field(
        default=None,
        description=(
            "XY top-down plane (perpendicular to Z), the effective-index sim. "
            "'auto' = centered; a float sets the Z coordinate (advisory)."
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
        """Update fields in place (validated as one batch). Returns self.

        Unlike per-field assignment, the ``mode``/cut invariant is checked
        against the *final* state, so ``solver(mode="2d", y_cut="auto")``
        never trips on an invalid intermediate (mode set, cut not yet).
        """
        validated = FDTD.model_validate({**self.__dict__, **kwargs})
        for name in FDTD.model_fields:
            object.__setattr__(self, name, getattr(validated, name))
        return self

    # Dispersion control
    dispersion: Literal["auto", "true", "false"] = Field(
        default="auto",
        description=(
            "Dispersion mode: 'auto' evaluates deps/eps across the source "
            "bandwidth and enables dispersion per material when >0.5%; 'true' "
            "forces full dispersion for all materials; 'false' forces "
            "constant-epsilon for speed."
        ),
    )
    dispersion_threshold: float = Field(
        default=0.005,
        gt=0,
        lt=1,
        description="deps/eps threshold for auto-dispersion (default 0.5%).",
    )

    # Diagnostics — output control for plots, fields, animations
    save_geometry: bool = Field(default=True)
    save_fields: bool = Field(default=True)
    save_epsilon_raw: bool = Field(default=False)
    save_animation: bool = Field(default=False)
    animation_interval: float = Field(default=0.5, gt=0)
    preview_only: bool = Field(default=False)
    verbose_interval: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_mode(self) -> FDTD:
        """Validate mode against the set of active cut planes.

        - ``mode='3d'`` -> none of x_cut/y_cut/z_cut may be set.
        - ``mode='2d'`` -> exactly one of x_cut/y_cut/z_cut must be set.
        - ``x_cut`` (YZ plane) is reserved and not yet implemented.
        """
        cuts = {
            "x_cut": self.x_cut,
            "y_cut": self.y_cut,
            "z_cut": self.z_cut,
        }
        n_set = sum(v is not None for v in cuts.values())
        if self.mode == "3d":
            if n_set:
                raise ValueError(
                    "3d mode requires x_cut/y_cut/z_cut to all be None "
                    f"(got {n_set} set)"
                )
        else:  # mode == "2d"
            if n_set != 1:
                raise ValueError(
                    "2d mode requires exactly one of x_cut/y_cut/z_cut "
                    f"(got {n_set} set)"
                )
        if self.x_cut is not None:
            raise NotImplementedError("YZ plane (x_cut) not yet supported")
        return self

    # -- Translation helpers (mode/cuts -> internal is_3d/plane/cut) --

    def resolved_is_3d(self) -> bool:
        """Whether this solves a full 3D simulation."""
        return self.mode == "3d"

    def resolved_plane(self) -> Literal["xy", "xz"] | None:
        """Internal 2D plane name implied by the active cut ('xz'|'xy'|None)."""
        if self.y_cut is not None:
            return "xz"
        if self.z_cut is not None:
            return "xy"
        return None

    def resolved_cut(self) -> float | str | None:
        """Value of the active cut ('auto', a float, or None in 3D)."""
        if self.y_cut is not None:
            return self.y_cut
        if self.z_cut is not None:
            return self.z_cut
        if self.x_cut is not None:
            return self.x_cut
        return None
