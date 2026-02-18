"""MEEP photonic FDTD simulation module.

Provides a declarative API (``Simulation``) for configuring and running
MEEP FDTD simulations on the GDSFactory+ cloud. No local MEEP
installation required.

Example::

    from gsim import meep

    sim = meep.Simulation()
    sim.geometry(component=ybranch, z_crop="auto")
    sim.materials = {"si": 3.47, "SiO2": 1.44}
    sim.source(port="o1", wavelength=1.55, wavelength_span=0.01, num_freqs=11)
    sim.monitors = ["o1", "o2"]
    sim.domain(pml=1.0, margin=0.5)
    sim.solver(resolution=32, simplify_tol=0.01)
    result = sim.run()
"""

from gsim.meep.models import (
    FDTD,
    Domain,
    DomainConfig,
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
    "Domain",
    "DomainConfig",
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
