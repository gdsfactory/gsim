"""Simulation submodule for FDTD simulations.

This module provides the main simulation classes and utilities.
"""

from gsim.fdtd.simulation.core import (
    FDTDSimulation,
    Material,
    Mesh,
    Physics,
    Results,
    Solver,
)
from gsim.fdtd.simulation.legacy import write_sparameters
from gsim.fdtd.simulation.modes import (
    Waveguide,
    WaveguideCoupler,
    sweep_bend_mismatch,
    sweep_coupling_length,
    sweep_fraction_te,
    sweep_mode_area,
    sweep_n_eff,
    sweep_n_group,
)
from gsim.fdtd.simulation.results import (
    get_results,
    get_results_batch,
    get_sim_hash,
)

__all__ = [
    # Core simulation classes
    "FDTDSimulation",
    "Material",
    "Mesh",
    "Physics",
    "Results",
    "Solver",
    # Mode solver
    "Waveguide",
    "WaveguideCoupler",
    "sweep_bend_mismatch",
    "sweep_coupling_length",
    "sweep_fraction_te",
    "sweep_mode_area",
    "sweep_n_eff",
    "sweep_n_group",
    # Results
    "get_results",
    "get_results_batch",
    "get_sim_hash",
    # Legacy (deprecated)
    "write_sparameters",
]
