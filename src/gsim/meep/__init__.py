"""MEEP photonic FDTD simulation module.

Provides both a declarative API (``Simulation``) and the legacy fluent
API (``MeepSim``) for configuring and running MEEP FDTD simulations
on the GDSFactory+ cloud. No local MEEP installation required.

Declarative API example::

    from gsim import meep

    sim = meep.Simulation()
    sim.geometry.component = ybranch
    sim.source.port = "o1"
    sim.monitors = [meep.ModeMonitor(port="o1"), meep.ModeMonitor(port="o2")]
    sim.solver.stopping = meep.DFTDecay(threshold=1e-3, min_time=100)
    result = sim.run()

Legacy API example::

    from gsim.meep import MeepSim

    sim = MeepSim()
    sim.set_geometry(component)
    sim.set_stack()
    sim.set_material("si", refractive_index=3.47)
    sim.set_wavelength(wavelength=1.55, bandwidth=0.1)
    sim.set_resolution(pixels_per_um=32)
    sim.set_output_dir("./meep-sim")
    result = sim.simulate()
"""

from gsim.meep.models import (
    # New declarative API models
    DFTDecay,
    Diagnostics,
    Domain,
    FDTD,
    FieldDecay,
    FixedTime,
    Geometry,
    Material,
    ModeMonitor,
    ModeSource,
    # Legacy config models
    AccuracyConfig,
    DiagnosticsConfig,
    FDTDConfig,
    WavelengthConfig,
    DomainConfig,
    ResolutionConfig,
    SimConfig,
    SourceConfig,
    SParameterResult,
    SymmetryEntry,
)
from gsim.meep.sim import MeepSim
from gsim.meep.simulation import Simulation

__all__ = [
    # New declarative API
    "DFTDecay",
    "Diagnostics",
    "Domain",
    "FDTD",
    "FieldDecay",
    "FixedTime",
    "Geometry",
    "Material",
    "ModeMonitor",
    "ModeSource",
    "Simulation",
    # Legacy
    "AccuracyConfig",
    "DiagnosticsConfig",
    "FDTDConfig",
    "WavelengthConfig",
    "DomainConfig",
    "MeepSim",
    "ResolutionConfig",
    "SParameterResult",
    "SimConfig",
    "SourceConfig",
    "SymmetryEntry",
]
