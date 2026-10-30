"""Port configuration models for Palace simulations.

This module contains Pydantic models for port definitions:
- PortConfig: Single-element lumped port configuration
- CPWPortConfig: Coplanar waveguide (two-element) port configuration
- TerminalConfig: Terminal for electrostatic simulations
- CurrentSourceConfig: Current source for magnetostatic simulations
- WavePortConfig: Wave port (domain boundary with mode solving)
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gsim.palace.ports.modes import check_wave_port_modes

if TYPE_CHECKING:
    from shapely import Polygon


class PortConfig(BaseModel):
    """Configuration for a single-element lumped port.

    Lumped ports can be inplane, gap, or interlayer. CPW ports use
    :class:`CPWPortConfig` because they require two gap elements.

    Attributes:
        name: Port name (must match component port name)
        layer: Target layer for inplane ports
        from_layer: First conductor layer for interlayer ports
        to_layer: Second conductor layer for interlayer ports
        length: Port extent along direction (um)
        offset: Shift port inward along the waveguide (um).
            Positive = away from boundary, into conductor.
        impedance: Port impedance (Ohms)
        excited: Whether this port is excited
        geometry: Lumped-port surface geometry. ``"via"`` remains as a deprecated
            alias for ``"interlayer"``.
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
    geometry: Literal["inplane", "gap", "interlayer", "via"] = "inplane"
    offset: float = Field(
        default=0.0,
        description="Shift port inward along the waveguide (um). "
        "Positive = away from boundary, into conductor.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_geometry(cls, data: Any) -> Any:
        """Normalize deprecated geometry names before field validation."""
        if isinstance(data, dict) and data.get("geometry") == "via":
            warnings.warn(
                "Port geometry 'via' is deprecated; use 'interlayer' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return {**data, "geometry": "interlayer"}
        return data

    @model_validator(mode="after")
    def validate_layer_config(self) -> Self:
        """Validate layer configuration based on geometry type."""
        if self.geometry == "inplane" and self.layer is None:
            raise ValueError("Inplane ports require 'layer' to be specified")
        if self.geometry == "interlayer" and (
            self.from_layer is None or self.to_layer is None
        ):
            raise ValueError(
                "Interlayer ports require both 'from_layer' and 'to_layer'"
            )
        if self.geometry == "gap":
            if self.layer is None:
                raise ValueError("Gap ports require 'layer' to be specified")
            if self.from_layer is not None or self.to_layer is not None:
                raise ValueError("Gap ports use a single conductor layer")
            if self.length is not None:
                raise ValueError(
                    "Gap ports use GDS port width as the gap span; omit length"
                )
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
    geometry: Literal["cpw"] = "cpw"
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
    in electrostatic simulations. A terminal selects conductors in one of
    three ways, and at least one of ``layer``, ``net`` and ``region`` is
    needed:

    - ``layer`` alone: every conductor shape on that layer (the layer mode;
      two separate electrodes on one layer are then one terminal).
    - ``net``: the connected conductor with that name, metal and vias on every
      layer it spans (see ``ElectrostaticSim.nets()``).
    - ``region``: every connected conductor with a shape inside the region,
      optionally only on ``layer``.

    ``net`` cannot be combined with ``region`` or ``layer``.

    Attributes:
        name: Terminal name
        layer: Target conductor layer. With ``region`` it limits the shapes
            that are tested against the region.
        net: Name of a net (the names of its ports joined by ``+``, or
            ``net<k>``), or the name of a port on it.
        region: Box ``(xmin, ymin, xmax, ymax)`` or list of at least three
            ``(x, y)`` points of a polygon, in um.
        floating: The conductor is not driven: it is solved as a terminal and
            eliminated from the capacitance matrix afterwards.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    layer: str | None = None
    net: str | None = None
    region: tuple[float, float, float, float] | list[tuple[float, float]] | None = None
    floating: bool = False

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        """Check that the terminal selects conductors in exactly one way."""
        if self.layer is None and self.net is None and self.region is None:
            raise ValueError(
                f"Terminal '{self.name}' needs layer, net or region to select "
                "its conductors"
            )
        if self.net is not None and (self.region is not None or self.layer is not None):
            raise ValueError(
                f"Terminal '{self.name}': net cannot be combined with region or layer"
            )
        if self.region is not None:
            self._validate_region()
        return self

    def _validate_region(self) -> None:
        """Check the box or polygon of ``region``."""
        region = self.region
        if region is None:
            return
        if isinstance(region, tuple):
            x0, y0, x1, y1 = region
            if not all(math.isfinite(v) for v in region):
                raise ValueError(f"Terminal '{self.name}': region must be finite")
            if not (x0 < x1 and y0 < y1):
                raise ValueError(
                    f"Terminal '{self.name}': a region box needs xmin < xmax and "
                    f"ymin < ymax, got {region}"
                )
            return
        if len(region) < 3:
            raise ValueError(
                f"Terminal '{self.name}': a region polygon needs at least 3 "
                f"points, got {len(region)}"
            )
        if not all(math.isfinite(v) for point in region for v in point):
            raise ValueError(f"Terminal '{self.name}': region must be finite")
        polygon = self.region_polygon()
        if polygon is None or not polygon.is_valid or polygon.area <= 0:
            raise ValueError(
                f"Terminal '{self.name}': the region polygon is degenerate or "
                "self-intersecting"
            )

    def region_polygon(self) -> Polygon | None:
        """The region as a shapely polygon (um), or None when there is none."""
        from shapely import Polygon, box

        if self.region is None:
            return None
        if isinstance(self.region, tuple):
            return box(*self.region)
        return Polygon(self.region)


class CurrentSourceConfig(BaseModel):
    """A current source between two ports on one conductor (magnetostatics).

    gsim bridges the gap between the two port faces with a vertical sheet
    through the thickness of the conductor layer. A unit current enters the
    conductor at ``plus_port`` and leaves it at ``minus_port`` (passive sign
    convention); the drawn metal that joins the two ports is the return path.

    Attributes:
        name: Source name; it labels the row of the inductance matrix.
        plus_port: Name of the component port where the current enters.
        minus_port: Name of the component port where the current leaves.
        layer: Conductor layer of the sheet; ``None`` infers it from the GDS
            layer of the ports.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    plus_port: str
    minus_port: str
    layer: str | None = None

    @model_validator(mode="after")
    def _validate_ports(self) -> Self:
        """Check that the source names two different ports."""
        if self.plus_port == self.minus_port:
            raise ValueError(
                f"Current source '{self.name}': plus_port and minus_port are "
                f"both {self.plus_port!r}; a source needs two different ports"
            )
        return self


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
        modes: Palace mode numbers followed on this port's face, for a
            multimode wave port. Empty (the default) is an ordinary wave
            port that follows ``mode`` only.
        active_mode: With ``modes``, the mode whose entry terminates the
            face; it must be in ``modes`` and equal to ``mode``. The other
            modes are observation-only entries (``"Active": false``).
        excite_modes: ``"active"`` excites only the active entry, ``"all"``
            every entry of the port.
        pilot_paths: Optional ``{mode: points}``: an open polyline of
            ``(x, y, z)`` points in um from the signal to the reference
            conductor, written as that entry's ``VoltagePath``.
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
    modes: tuple[int, ...] = Field(
        default=(),
        description=(
            "Palace mode numbers followed on this port's face (multimode port)"
        ),
    )
    active_mode: int | None = Field(
        default=None,
        ge=1,
        description="Mode whose entry terminates the face (multimode port)",
    )
    excite_modes: Literal["active", "all"] = Field(
        default="active",
        description="Excite only the active entry or every entry of the port",
    )
    pilot_paths: dict[int, list[list[float]]] | None = Field(
        default=None,
        description=(
            "Per-mode VoltagePath: (x, y, z) points in um from the signal to "
            "the reference conductor"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _active_mode_sets_mode(cls, data: Any) -> Any:
        """Let ``active_mode`` stand in for ``mode`` when only it is given."""
        if (
            isinstance(data, dict)
            and data.get("active_mode") is not None
            and "mode" not in data
        ):
            return {**data, "mode": data["active_mode"]}
        return data

    @model_validator(mode="after")
    def _validate_modes(self) -> Self:
        """Check the mode list, the active mode and the pilot paths."""
        check_wave_port_modes(
            mode=self.mode,
            modes=self.modes,
            active_mode=self.active_mode,
            pilot_paths=self.pilot_paths,
        )
        return self


class TwoTerminalPortConfig(BaseModel):
    """Configuration for a two-terminal lumped port (single Palace port, two elements).

    Both terminals are combined into one Palace LumpedPort with two EDGE-geometry
    elements (vertical surfaces spanning the conductor thickness), collapsing the
    simulation to a true 1-port S11 measurement on a coplanar device.

    Attributes:
        plus_port: GDS port name for the + (excitation) terminal
        minus_port: GDS port name for the - (reference) terminal
        layer: Conductor layer containing both terminals (e.g., "metal1")
        impedance: Port impedance in Ohms
        excited: Whether this port is excited
    """

    model_config = ConfigDict(validate_assignment=True)

    plus_port: str = Field(description="GDS port name for the + terminal")
    minus_port: str = Field(description="GDS port name for the - terminal")
    layer: str = Field(description="Conductor layer containing both terminals")
    impedance: float = Field(default=50.0, gt=0)
    excited: bool = True


__all__ = [
    "CPWPortConfig",
    "ImpedanceBoundaryConfig",
    "PortConfig",
    "TerminalConfig",
    "TwoTerminalPortConfig",
    "WavePortConfig",
]
