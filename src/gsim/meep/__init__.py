"""MEEP photonic FDTD simulation module.

Provides a declarative API (``Simulation``) for configuring and running
MEEP FDTD simulations on the GDSFactory+ cloud. No local MEEP
installation required.

Example::

    from gsim import meep

    sim = meep.Simulation()
    sim.geometry.component = ybranch
    sim.source.port = "o1"
    sim.monitors = ["o1", "o2"]
    sim.solver.stopping = meep.DFTDecay(threshold=1e-3, min_time=100)
    result = sim.run("./meep-sim")
"""

from gsim.meep.models import (
    FDTD,
    DFTDecay,
    Domain,
    DomainConfig,
    FieldDecay,
    FixedTime,
    Geometry,
    Material,
    ModeSource,
    ResolutionConfig,
    SimConfig,
    SourceConfig,
    SParameterResult,
    Symmetry,
    # Config models used by Simulation.write_config()
    WavelengthConfig,
)
from gsim.meep.simulation import BuildResult, Simulation

__all__ = [
    "FDTD",
    "BuildResult",
    "DFTDecay",
    "Domain",
    "DomainConfig",
    "FieldDecay",
    "FixedTime",
    "Geometry",
    "Material",
    "ModeSource",
    "ResolutionConfig",
    "SParameterResult",
    "SimConfig",
    "Simulation",
    "SourceConfig",
    "Symmetry",
    "WavelengthConfig",
]
