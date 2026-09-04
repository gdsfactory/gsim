"""Port configuration models for Palace simulations.

This module contains Pydantic models for port definitions:
- PortConfig: Single-element lumped port configuration
- CPWPortConfig: Coplanar waveguide (two-element) port configuration
- TerminalConfig: Terminal for electrostatic simulations
- WavePortConfig: Wave port (domain boundary with mode solving)
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PortConfig(BaseModel):
    """Configuration for a single-element lumped port.

    Lumped ports can be inplane (horizontal, on single layer) or
    via (vertical, between two layers).

    Attributes:
        name: Port name (must match component port name)
        layer: Target layer for inplane ports
        from_layer: Bottom layer for via ports
        to_layer: Top layer for via ports
        length: Port extent along direction (um)
        offset: Shift port inward along the waveguide (um).
            Positive = away from boundary, into conductor.
        impedance: Port impedance (Ohms)
        excited: Whether this port is excited
        geometry: Port geometry type ("inplane" or "via")
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    layer: str | None = None
    from_layer: str | None = None
    to_layer: str | None = None
    length: float | None = Field(default=None, gt=0)
    impedance: float = Field(default=50.0, gt=0)
    resistance: float | None = Field(
        default=None, ge=0, description="Resistance in Ohms"
    )
    inductance: float | None = Field(default=None, ge=0, description="Inductance in H")
    capacitance: float | None = Field(
        default=None, ge=0, description="Capacitance in F"
    )
    excited: bool = True
    geometry: Literal["inplane", "via"] = "inplane"
    offset: float = Field(
        default=0.0,
        description="Shift port inward along the waveguide (um). "
        "Positive = away from boundary, into conductor.",
    )

    @model_validator(mode="after")
    def validate_layer_config(self) -> Self:
        """Validate layer configuration based on geometry type."""
        if self.geometry == "inplane" and self.layer is None:
            raise ValueError("Inplane ports require 'layer' to be specified")
        if self.geometry == "via" and (
            self.from_layer is None or self.to_layer is None
        ):
            raise ValueError("Via ports require both 'from_layer' and 'to_layer'")
        return self


class CPWPortConfig(BaseModel):
    """Configuration for a coplanar waveguide (CPW) port.

    CPW ports consist of two elements (upper and lower gaps) that are
    excited with opposite E-field directions to create the CPW mode.

    The port is placed at the center of the signal conductor. The two
    gap element surfaces are computed from s_width and gap_width.

    Attributes:
        name: Port name (must match a single component port at the signal center)
        layer: Target conductor layer
        s_width: Width of the signal (center) conductor (um)
        gap_width: Width of each gap between signal and ground (um)
        length: Port extent along direction (um)
        offset: Shift the port along the waveguide direction (um).
            Positive moves in the port orientation direction.
        impedance: Port impedance (Ohms)
        excited: Whether this port is excited
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(description="Port name matching component port")
    layer: str = Field(description="Target conductor layer")
    s_width: float = Field(gt=0, description="Signal conductor width (um)")
    gap_width: float = Field(
        gt=0, description="Gap width between signal and ground (um)"
    )
    length: float = Field(default=2.0, gt=0, description="Port extent in um")
    offset: float | None = Field(
        default=None,
        description="Shift port inward along the waveguide (um). "
        "Positive = away from boundary, into conductor. "
        "Defaults to length/2 (port flush with conductor edge).",
    )

    @model_validator(mode="after")
    def _default_offset(self) -> Self:
        """Default offset to length/2 so the port is flush with the conductor edge."""
        if self.offset is None:
            self.offset = self.length / 2
        return self

    impedance: float = Field(default=50.0, gt=0)
    excited: bool = True


class TerminalConfig(BaseModel):
    """Configuration for a terminal (for electrostatic capacitance extraction).

    Terminals define conductor surfaces for capacitance matrix extraction
    in electrostatic simulations.

    Attributes:
        name: Terminal name
        layer: Target conductor layer
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    layer: str


class ImpedanceBoundaryConfig(BaseModel):
    """Configuration for an Impedance boundary on a dielectric-dielectric interface.

    Specifies a surface impedance boundary condition (Rs, Ls, Cs) applied to
    the shared boundary curve between two dielectric layers.  Capacitance,
    resistance, and inductance values are given as *absolute* quantities and
    are divided by the interface curve length internally to obtain the
    per-unit-length values (Rs, Ls, Cs) expected by Palace.

    Alternatively, ``attributes`` can be specified directly (with Rs/Ls/Cs
    already in per-unit-length form) for cases where the user already knows
    the Palace boundary attribute numbers.

    Attributes:
        layer_a: Name of the first dielectric layer.
        layer_b: Name of the second dielectric layer.
        capacitance: Absolute capacitance [F] (divided by curve length).
        resistance: Absolute resistance [Ohm] (divided by curve length).
        inductance: Absolute inductance [H] (divided by curve length).
        attributes: Direct Palace boundary attribute list (bypasses layer lookup).
            When set, Rs/Ls/Cs must be in per-unit-length values.
        name: Optional display name.
    """

    model_config = ConfigDict(validate_assignment=True)

    layer_a: str | None = None
    layer_b: str | None = None
    capacitance: float | None = Field(default=None, ge=0, description="Capacitance [F]")
    resistance: float | None = Field(default=None, ge=0, description="Resistance [Ohm]")
    inductance: float | None = Field(default=None, ge=0, description="Inductance [H]")
    attributes: list[int] | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> Self:
        """Validate that exactly one of (layer_a+layer_b) or attributes is set."""
        if (self.layer_a is None or self.layer_b is None) and self.attributes is None:
            raise ValueError("Either layer_a+layer_b or attributes must be provided")
        if (
            self.layer_a is not None or self.layer_b is not None
        ) and self.attributes is not None:
            raise ValueError("Cannot specify both layer_a/layer_b and attributes")
        if self.layer_a is not None and self.layer_b is None:
            raise ValueError("Both layer_a and layer_b must be provided together")
        if self.layer_a is None and self.layer_b is not None:
            raise ValueError("Both layer_a and layer_b must be provided together")
        return self


class WavePortConfig(BaseModel):
    """Configuration for a wave port (domain boundary with mode solving).

    Wave ports are used for domain-boundary ports where mode solving
    is needed. This is an alternative to lumped ports for more accurate
    S-parameter extraction.

    Attributes:
        name: Port name (must match component port name)
        layer: Target conductor layer
        z_margin: Margin to extend port geometry in z-direction (um)
        max_size: If True, set z_margin and lateral_margin to
        fill the full simulation domain
        mode: Mode number to excite
        offset: De-embedding distance in um
        excited: Whether this port is excited
        eigensolver_type: Palace SolverType for this port's 2D mode
            eigenproblem ("Default", "SLEPc" or "ARPACK"). None uses
            Palace's own default.
        eigensolver_tol: Palace EigenTol (eigenvalue solver relative
            tolerance) for this port's mode solve. None uses Palace's
            own default.
        eigensolver_ksp_tol: Palace KSPTol (linear solver tolerance used
            inside the eigenvalue iteration) for this port's mode solve.
            None uses Palace's own default.
        eigensolver_max_size: Palace MaxSize (eigensolver subspace
            dimension) for this port's mode solve - unrelated to the
            `max_size` domain-filling flag above. None lets Palace pick
            its own default (max(2 x Mode, Mode + 15)).
        eigensolver_verbose: Palace Verbose level for this port's mode
            solve. None uses Palace's own default.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    layer: str | None = None
    z_margin: float = Field(default=0, ge=0)
    lateral_margin: float = Field(default=0.0, ge=0)
    max_size: bool = Field(
        default=False,
        description=(
            "When True, set z_margin and lateral_margin"
            " to fill the full simulation domain"
        ),
    )
    mode: int = Field(default=1, ge=1, description="Mode number to excite")
    offset: float = Field(default=0.0, ge=0, description="De-embedding distance in um")
    excited: bool = True
    eigensolver_type: Literal["Default", "SLEPc", "ARPACK"] | None = Field(
        default=None,
        description="Palace SolverType for this port's 2D mode eigenproblem",
    )
    eigensolver_tol: float | None = Field(
        default=None, gt=0, description="Palace EigenTol for this port's mode solve"
    )
    eigensolver_ksp_tol: float | None = Field(
        default=None, gt=0, description="Palace KSPTol for this port's mode solve"
    )
    eigensolver_max_size: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Palace MaxSize (eigensolver subspace dimension) for this "
            "port's mode solve - unrelated to the max_size domain-filling "
            "flag above"
        ),
    )
    eigensolver_verbose: int | None = Field(
        default=None,
        ge=0,
        description="Palace Verbose level for this port's mode solve",
    )


__all__ = [
    "CPWPortConfig",
    "ImpedanceBoundaryConfig",
    "PortConfig",
    "TerminalConfig",
    "WavePortConfig",
]
