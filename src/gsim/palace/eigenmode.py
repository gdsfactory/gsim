"""Eigenmode simulation class for resonance/mode finding.

This module provides the EigenmodeSim class for finding resonant
frequencies and mode shapes.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.models import (
    CPWPortConfig,
    EigenmodeConfig,
    MaterialConfig,
    NumericalConfig,
    PortConfig,
    RefinementConfig,
    TwoTerminalPortConfig,
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
        >>> sim.set_stack()
        >>> sim.set_airbox(margin_x=120.0, margin_above=120.0, margin_below=20.0)
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
    two_terminal_ports: list[TwoTerminalPortConfig] = Field(default_factory=list)

    # Eigenmode simulation config
    eigenmode: EigenmodeConfig = Field(default_factory=EigenmodeConfig)

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _airbox_config: dict[str, Any] = PrivateAttr(default_factory=dict)
    _pec_blocks: list = PrivateAttr(default_factory=list)
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)
    _impedance_boundaries: list = PrivateAttr(default_factory=list)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_ports: bool = PrivateAttr(default=False)

    # -------------------------------------------------------------------------
    # Cloud run (narrowed return type)
    # -------------------------------------------------------------------------

    def run(
        self,
        parent_dir: str | Path | None = None,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        wait: bool = True,
        check_cache: bool = False,
    ) -> dict[str, Path] | str:
        """Run the eigenmode sim on GDSFactory+ cloud.

        Thin wrapper over :meth:`PalaceSimMixin.run` that narrows the
        return type: an eigenmode run returns a ``dict[str, Path]`` of
        output files keyed by name (e.g. ``"eig.csv"``), or the
        ``job_id`` string when ``wait=False``.
        """
        from gsim.palace.results import SParams

        result = super().run(
            parent_dir, verbose=verbose, wait=wait, check_cache=check_cache
        )
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
        floquet: bool = False,
        phi_target: float = math.pi / 2,
        periodic_length: float | None = None,
        n_eff_guess: float | None = None,
    ) -> None:
        """Configure eigenmode simulation.

        Args:
            num_modes: Number of modes to find
            target: Positive target frequency in Hz for mode search.
                Required before meshing or exporting the configuration.
            tolerance: Eigenvalue solver tolerance
            save: Number of eigenmodes to save as ParaView fields (0 = disabled)
            floquet: Enable Floquet periodic boundary setup in config generation
                (requires mesh(periodic_axis=...)).
            phi_target: Signed Bloch phase per cell in radians (Floquet only).
                Zero and +/-pi are valid. The wave vector is phase / mesh period.
                Palace applies E(receiver) = exp(-i * phi_target) * E(donor).
            periodic_length: Optional expected period in mesh units (um). Must
                match the measured mesh translation, including domain padding.
                Omit to use the measured period directly.
            n_eff_guess: Deprecated compatibility argument, ignored with a warning.
                Target frequency controls the eigenvalue search, not the wave vector.

        Example:
            >>> sim.set_eigenmode(num_modes=10, target=50e9)
        """
        if n_eff_guess is not None:
            warnings.warn(
                "n_eff_guess is deprecated and ignored; "
                "Floquet uses the actual mesh period.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.eigenmode = EigenmodeConfig(
            num_modes=num_modes,
            target=target,
            tolerance=tolerance,
            save=save,
            floquet=floquet,
            phi_target=phi_target,
            periodic_length=periodic_length,
            n_eff_guess=2.0 if n_eff_guess is None else n_eff_guess,
        )


__all__ = ["EigenmodeSim"]
