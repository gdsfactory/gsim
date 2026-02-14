"""MEEP photonic FDTD simulation module.

Provides a fluent API for configuring and running MEEP FDTD simulations
on the GDSFactory+ cloud. No local MEEP installation required.

Example:
    >>> from gsim.meep import MeepSim
    >>>
    >>> sim = MeepSim()
    >>> sim.set_geometry(component)
    >>> sim.set_stack()
    >>> sim.set_material("si", refractive_index=3.47)
    >>> sim.set_wavelength(wavelength=1.55, bandwidth=0.1)
    >>> sim.set_resolution(pixels_per_um=32)
    >>> sim.set_output_dir("./meep-sim")
    >>> result = sim.simulate()
    >>> result.plot()
"""

from gsim.meep.models import (
    FDTDConfig,
    DomainConfig,
    ResolutionConfig,
    SimConfig,
    SParameterResult,
    SymmetryEntry,
)
from gsim.meep.sim import MeepSim

__all__ = [
    "FDTDConfig",
    "DomainConfig",
    "MeepSim",
    "ResolutionConfig",
    "SParameterResult",
    "SimConfig",
    "SymmetryEntry",
]
