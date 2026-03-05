"""Driven simulation class for frequency-domain S-parameter extraction.

This module provides the DrivenSim class for running frequency-sweep
simulations to extract S-parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.models import (
    CPWPortConfig,
    DrivenConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    PortConfig,
)


class DrivenSim(PalaceSimMixin, BaseModel):
    """Frequency-domain driven simulation for S-parameter extraction.

    This class configures and runs driven simulations that sweep through
    frequencies to compute S-parameters. Uses composition (no inheritance)
    with shared Geometry and Stack components from gsim.common.

    Example:
        >>> from gsim.palace import DrivenSim
        >>>
        >>> sim = DrivenSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack(air_above=300.0)
        >>> sim.add_cpw_port("o1", layer="topmetal2", s_width=10, gap_width=6, length=5)
        >>> sim.add_cpw_port("o2", layer="topmetal2", s_width=10, gap_width=6, length=5)
        >>> sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
        >>> sim.set_output_dir("./sim")
        >>> sim.mesh(preset="default")
        >>> results = sim.run()

    Attributes:
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        ports: List of single-element port configurations
        cpw_ports: List of CPW (two-element) port configurations
        driven: Driven simulation configuration (frequencies, etc.)
        mesh: Mesh configuration
        materials: Material property overrides
        numerical: Numerical solver configuration
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Composed objects (from common)
    geometry: Geometry | None = None
    stack: LayerStack | None = None

    # Port configurations
    ports: list[PortConfig] = Field(default_factory=list)
    cpw_ports: list[CPWPortConfig] = Field(default_factory=list)

    # Driven simulation config
    driven: DrivenConfig = Field(default_factory=DrivenConfig)

    # Mesh config
    mesh_config: MeshConfig = Field(default_factory=MeshConfig.default)

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_ports: bool = PrivateAttr(default=False)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)

    # Cloud job state (set by upload/run)
    _job_id: str | None = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Port methods
    # -------------------------------------------------------------------------

    def add_port(
        self,
        name: str,
        *,
        layer: str | None = None,
        from_layer: str | None = None,
        to_layer: str | None = None,
        length: float | None = None,
        impedance: float = 50.0,
        resistance: float | None = None,
        inductance: float | None = None,
        capacitance: float | None = None,
        excited: bool = True,
        geometry: Literal["inplane", "via"] = "inplane",
    ) -> None:
        """Add a single-element lumped port.

        Args:
            name: Port name (must match component port name)
            layer: Target layer for inplane ports
            from_layer: Bottom layer for via ports
            to_layer: Top layer for via ports
            length: Port extent along direction (um)
            impedance: Port impedance (Ohms)
            resistance: Series resistance (Ohms)
            inductance: Series inductance (H)
            capacitance: Shunt capacitance (F)
            excited: Whether this port is excited
            geometry: Port geometry type ("inplane" or "via")

        Example:
            >>> sim.add_port("o1", layer="topmetal2", length=5.0)
            >>> sim.add_port(
            ...     "feed", from_layer="metal1", to_layer="topmetal2", geometry="via"
            ... )
        """
        # Remove existing config for this port if any
        self.ports = [p for p in self.ports if p.name != name]

        self.ports.append(
            PortConfig(
                name=name,
                layer=layer,
                from_layer=from_layer,
                to_layer=to_layer,
                length=length,
                impedance=impedance,
                resistance=resistance,
                inductance=inductance,
                capacitance=capacitance,
                excited=excited,
                geometry=geometry,
            )
        )

    def add_cpw_port(
        self,
        name: str,
        *,
        layer: str,
        s_width: float,
        gap_width: float,
        length: float,
        offset: float = 0.0,
        impedance: float = 50.0,
        excited: bool = True,
    ) -> None:
        """Add a coplanar waveguide (CPW) port.

        CPW ports consist of two elements (upper and lower gaps) that are
        excited with opposite E-field directions to create the CPW mode.

        Place a single gdsfactory port at the center of the signal conductor.
        The two gap element surfaces are computed from s_width and gap_width.

        Args:
            name: Port name (must match a component port at the signal center)
            layer: Target conductor layer (e.g., "topmetal2")
            s_width: Width of the signal (center) conductor (um)
            gap_width: Width of each gap between signal and ground (um)
            length: Port extent along direction (um)
            offset: Shift the port inward along the waveguide (um).
                Positive moves away from the boundary, into the conductor.
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited

        Example:
            >>> sim.add_cpw_port(
            ...     "left", layer="topmetal2", s_width=20, gap_width=15, length=5.0
            ... )
        """
        # Remove existing CPW port with same name if any
        self.cpw_ports = [p for p in self.cpw_ports if p.name != name]

        self.cpw_ports.append(
            CPWPortConfig(
                name=name,
                layer=layer,
                s_width=s_width,
                gap_width=gap_width,
                length=length,
                offset=offset,
                impedance=impedance,
                excited=excited,
            )
        )

    # -------------------------------------------------------------------------
    # Driven configuration
    # -------------------------------------------------------------------------

    def set_driven(
        self,
        *,
        fmin: float = 1e9,
        fmax: float = 100e9,
        num_points: int = 40,
        scale: Literal["linear", "log"] = "linear",
        adaptive_tol: float = 0.02,
        adaptive_max_samples: int = 20,
        compute_s_params: bool = True,
        reference_impedance: float = 50.0,
        excitation_port: str | None = None,
    ) -> None:
        """Configure driven (frequency sweep) simulation.

        Args:
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            num_points: Number of frequency points
            scale: "linear" or "log" frequency spacing
            adaptive_tol: Adaptive frequency tolerance (0 disables adaptive)
            adaptive_max_samples: Max samples for adaptive refinement
            compute_s_params: Compute S-parameters
            reference_impedance: Reference impedance for S-params (Ohms)
            excitation_port: Port to excite (None = first port)

        Example:
            >>> sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
        """
        self.driven = DrivenConfig(
            fmin=fmin,
            fmax=fmax,
            num_points=num_points,
            scale=scale,
            adaptive_tol=adaptive_tol,
            adaptive_max_samples=adaptive_max_samples,
            compute_s_params=compute_s_params,
            reference_impedance=reference_impedance,
            excitation_port=excitation_port,
        )


__all__ = ["DrivenSim"]
