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
    length: float = Field(gt=0, description="Port extent in um")
    offset: float = Field(
        default=0.0,
        description="Shift port inward along the waveguide (um). "
        "Positive = away from boundary, into conductor.",
    )
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


__all__ = [
    "CPWPortConfig",
    "PortConfig",
    "TerminalConfig",
    "WavePortConfig",
]
