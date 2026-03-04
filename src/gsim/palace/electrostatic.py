"""Electrostatic simulation class for capacitance extraction.

This module provides the ElectrostaticSim class for extracting
capacitance matrices between terminals.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.palace.base import PalaceSimMixin
from gsim.palace.models import (
    ElectrostaticConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    TerminalConfig,
)

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
        >>> sim.set_stack(air_above=300.0)
        >>> sim.add_terminal("T1", layer="topmetal2")
        >>> sim.add_terminal("T2", layer="topmetal2")
        >>> sim.set_electrostatic()
        >>> sim.set_output_dir("./sim")
        >>> sim.mesh(preset="default")
        >>> results = sim.run()

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

    # Terminal configurations (no ports in electrostatic)
    terminals: list[TerminalConfig] = Field(default_factory=list)

    # Electrostatic simulation config
    electrostatic: ElectrostaticConfig = Field(default_factory=ElectrostaticConfig)

    # Mesh config
    mesh_config: MeshConfig = Field(default_factory=MeshConfig.default)

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Any = PrivateAttr(default=None)
    _configured_terminals: bool = PrivateAttr(default=False)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)
    _last_terminals: list = PrivateAttr(default_factory=list)

    # Cloud job state
    _job_id: str | None = PrivateAttr(default=None)

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


__all__ = ["ElectrostaticSim"]
