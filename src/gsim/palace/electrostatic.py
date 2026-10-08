"""Electrostatic simulation class for capacitance extraction.

This module provides the ElectrostaticSim class for extracting
capacitance matrices between terminals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.capacitance import CapacitanceMatrices, load_capacitance
from gsim.palace.models import (
    ElectrostaticConfig,
    MaterialConfig,
    NumericalConfig,
    RefinementConfig,
    TerminalConfig,
    WavePortConfig,
)

if TYPE_CHECKING:
    from gsim.palace.mesh.nets import Nets

logger = logging.getLogger(__name__)


class ElectrostaticSim(PalaceSimMixin, BaseModel):
    """Electrostatic simulation for capacitance matrix extraction.

    This class configures and runs electrostatic simulations to extract
    the capacitance matrix between conductor terminals. Unlike driven
    and eigenmode simulations, this does not use ports.

    Example:
        >>> from gsim.palace import ElectrostaticSim
        >>>
        >>> sim = ElectrostaticSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack()
        >>> sim.set_airbox(margin_x=120.0, margin_above=120.0, margin_below=20.0)
        >>> sim.add_terminal("T1", layer="topmetal2")
        >>> sim.add_terminal("T2", layer="topmetal2")
        >>> sim.set_electrostatic()
        >>> sim.set_output_dir("./sim")
        >>> sim.mesh(preset="default")
        >>> results = sim.run()  # dict[str, Path]
        >>> print(results["terminal-C.csv"])

    Attributes:
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        terminals: List of terminal configurations
        electrostatic: Electrostatic simulation configuration
        materials: Material property overrides
        numerical: Numerical solver configuration
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    simulation_type: Literal["electrostatic"] = "electrostatic"

    driven: None = None
    ports: None = None
    cpw_ports: None = None
    two_terminal_ports: None = None
    wave_ports: list[WavePortConfig] = Field(default_factory=list)
    # Composed objects (from common)
    geometry: Geometry | None = None
    stack: LayerStack | None = None

    # Terminal configurations (no ports in electrostatic)
    terminals: list[TerminalConfig] = Field(default_factory=list)

    # Electrostatic simulation config
    electrostatic: ElectrostaticConfig = Field(default_factory=ElectrostaticConfig)
    eigenmode: None = None
    absorbing_boundary: bool = False

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _airbox_config: dict[str, Any] = PrivateAttr(default_factory=dict)
    _pec_blocks: list = PrivateAttr(default_factory=list)
    _symmetry_planes: list = PrivateAttr(default_factory=list)
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)
    _impedance_boundaries: list = PrivateAttr(default_factory=list)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_terminals: bool = PrivateAttr(default=False)

    # -------------------------------------------------------------------------
    # Terminal methods
    # -------------------------------------------------------------------------

    def add_terminal(
        self,
        name: str,
        *,
        layer: str,
    ) -> None:
        """Add a terminal for capacitance extraction.

        Terminals define conductor surfaces for capacitance matrix extraction.

        Args:
            name: Terminal name
            layer: Target conductor layer

        Example:
            >>> sim.add_terminal("T1", layer="topmetal2")
            >>> sim.add_terminal("T2", layer="topmetal2")
        """
        # Remove existing terminal with same name
        self.terminals = [t for t in self.terminals if t.name != name]
        self.terminals.append(
            TerminalConfig(
                name=name,
                layer=layer,
            )
        )

    def nets(self) -> Nets:
        """Find the electrically connected conductor shapes of the geometry.

        Metal that touches on one layer, and metal on different layers that a
        via joins, is one net; disconnected electrodes stay separate even when
        they share a layer. Electrical ports name the nets they sit on.

        A terminal still selects every shape on its layer, so two disconnected
        electrodes on one layer end up as a single terminal (gsim#273). This
        shows how many electrodes each layer holds.

        Returns:
            The nets, in the order their first shape appears in the layout.

        Raises:
            ValueError: If no geometry has been set.

        Example:
            >>> print(sim.nets())
        """
        from gsim.palace.mesh.geometry import extract_geometry
        from gsim.palace.mesh.nets import extract_nets

        component = self.component
        if component is None:
            msg = "No component set. Call set_geometry(component) first."
            raise ValueError(msg)
        stack = self._resolve_stack()
        ports = [port for port in component.ports if port.port_type == "electrical"]
        return extract_nets(extract_geometry(component, stack), stack, ports)

    # -------------------------------------------------------------------------
    # Electrostatic configuration
    # -------------------------------------------------------------------------

    def set_electrostatic(
        self,
        *,
        save_fields: int = 0,
    ) -> None:
        """Configure electrostatic simulation.

        Args:
            save_fields: Number of field solutions to save

        Example:
            >>> sim.set_electrostatic(save_fields=1)
        """
        self.electrostatic = ElectrostaticConfig(
            save_fields=save_fields,
        )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    def load_capacitance(
        self, results: dict[str, Path] | str | Path
    ) -> CapacitanceMatrices:
        """Load the capacitance matrices of a run, labelled with the terminals.

        The terminal names are the ones given to :meth:`add_terminal`, in the
        order they were added, which is Palace's terminal order.

        Args:
            results: The results dict that ``run()`` returns, or the output
                directory of the simulation.

        Returns:
            The Maxwell and mutual capacitance matrices, with checks; see
            :class:`~gsim.palace.capacitance.CapacitanceMatrices`.

        Example:
            >>> cap = sim.load_capacitance(sim.run())
            >>> cap.between("T1", "T2")  # plate to plate, in F
            >>> cap.to_ground("T1")  # to the substrate, in F
            >>> assert not cap.problems()
        """
        return load_capacitance(
            results, terminal_names=[terminal.name for terminal in self.terminals]
        )


__all__ = ["ElectrostaticSim"]
