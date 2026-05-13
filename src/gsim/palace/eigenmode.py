"""Eigenmode simulation class for resonance/mode finding.

This module provides the EigenmodeSim class for finding resonant
frequencies and mode shapes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, overload

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.models import (
    CPWPortConfig,
    EigenmodeConfig,
    MaterialConfig,
    NumericalConfig,
    PortConfig,
    WavePortConfig,
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
        >>> results = sim.run()  # dict[str, Path]
        >>> print(results["eig.csv"])

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
    simulation_type: Literal["eigenmode"] = "eigenmode"

    driven: None = None
    terminals: None = None
    wave_ports: list[WavePortConfig] = Field(default_factory=list)
    # Composed objects (from common)
    geometry: Geometry | None = None
    stack: LayerStack | None = None
    absorbing_boundary: bool = False

    # Port configurations (eigenmode can have ports for Q-factor calculation)
    ports: list[PortConfig] = Field(default_factory=list)
    cpw_ports: list[CPWPortConfig] = Field(default_factory=list)

    # Eigenmode simulation config
    eigenmode: EigenmodeConfig = Field(default_factory=EigenmodeConfig)

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _pec_blocks: list = PrivateAttr(default_factory=list)
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_ports: bool = PrivateAttr(default=False)

    # -------------------------------------------------------------------------
    # Cloud run (narrowed return type)
    # -------------------------------------------------------------------------

    @overload
    def run(
        self,
        parent_dir: str | Path | None = ...,
        *,
        verbose: Literal["quiet", "status", "full"] = ...,
        wait: Literal[True] = ...,
    ) -> dict[str, Path]: ...
    @overload
    def run(
        self,
        parent_dir: str | Path | None = ...,
        *,
        verbose: Literal["quiet", "status", "full"] = ...,
        wait: Literal[False],
    ) -> str: ...
    def run(  # ty: ignore[invalid-method-override]
        self,
        parent_dir: str | Path | None = None,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        wait: bool = True,
    ) -> dict[str, Path] | str:
        """Run the eigenmode sim on GDSFactory+ cloud.

        Thin wrapper over :meth:`PalaceSimMixin.run` that narrows the
        return type: an eigenmode run returns a ``dict[str, Path]`` of
        output files keyed by name (e.g. ``"eig.csv"``), or the
        ``job_id`` string when ``wait=False``.
        """
        from gsim.palace.results import SParams

        result = super().run(parent_dir, verbose=verbose, wait=wait)
        if isinstance(result, SParams):
            msg = (
                "EigenmodeSim.run got SParams from the cloud, but an "
                "eigenmode job is expected to produce eig.csv outputs."
            )
            raise TypeError(msg)
        return result

    # -------------------------------------------------------------------------
    # Eigenmode configuration
    # -------------------------------------------------------------------------

    def set_eigenmode(
        self,
        *,
        num_modes: int = 10,
        target: float | None = None,
        tolerance: float = 1e-6,
        save: int = 0,
    ) -> None:
        """Configure eigenmode simulation.

        Args:
            num_modes: Number of modes to find
            target: Target frequency in Hz for mode search
            tolerance: Eigenvalue solver tolerance
            save: Number of eigenmodes to save as ParaView fields (0 = disabled)

        Example:
            >>> sim.set_eigenmode(num_modes=10, target=50e9)
        """
        self.eigenmode = EigenmodeConfig(
            num_modes=num_modes,
            target=target,
            tolerance=tolerance,
            save=save,
        )


__all__ = ["EigenmodeSim"]
