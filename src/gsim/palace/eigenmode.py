"""Eigenmode simulation class for resonance/mode finding.

This module provides the EigenmodeSim class for finding resonant
frequencies and mode shapes.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.palace.base import PalaceSimMixin
from gsim.palace.models import (
    CPWPortConfig,
    EigenmodeConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    PortConfig,
)

logger = logging.getLogger(__name__)


class EigenmodeSim(PalaceSimMixin, BaseModel):
    """Eigenmode simulation for finding resonant frequencies.

    This class configures and runs eigenmode simulations to find
    resonant frequencies and mode shapes of structures.

    Example:
        >>> from gsim.palace import EigenmodeSim
        >>>
        >>> sim = EigenmodeSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack(air_above=300.0)
        >>> sim.add_port("o1", layer="topmetal2", length=5.0)
        >>> sim.set_eigenmode(num_modes=10, target=50e9)
        >>> sim.set_output_dir("./sim")
        >>> sim.mesh(preset="default")
        >>> results = sim.run()

    Attributes:
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        ports: List of single-element port configurations
        cpw_ports: List of CPW (two-element) port configurations
        eigenmode: Eigenmode simulation configuration
        materials: Material property overrides
        numerical: Numerical solver configuration
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Port configurations (eigenmode can have ports for Q-factor calculation)
    ports: list[PortConfig] = Field(default_factory=list)
    cpw_ports: list[CPWPortConfig] = Field(default_factory=list)

    # Eigenmode simulation config
    eigenmode: EigenmodeConfig = Field(default_factory=EigenmodeConfig)

    # Mesh config
    mesh_config: MeshConfig = Field(default_factory=MeshConfig.default)

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Any = PrivateAttr(default=None)
    _configured_ports: bool = PrivateAttr(default=False)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)

    # Cloud job state
    _job_id: str | None = PrivateAttr(default=None)

    def set_eigenmode(
        self,
        *,
        num_modes: int = 10,
        target: float | None = None,
        tolerance: float = 1e-6,
    ) -> None:
        """Configure eigenmode simulation.

        Args:
            num_modes: Number of modes to find
            target: Target frequency in Hz for mode search
            tolerance: Eigenvalue solver tolerance

        Example:
            >>> sim.set_eigenmode(num_modes=10, target=50e9)
        """
        self.eigenmode = EigenmodeConfig(
            num_modes=num_modes,
            target=target,
            tolerance=tolerance,
        )


__all__ = ["EigenmodeSim"]
