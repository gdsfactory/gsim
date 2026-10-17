"""gsim.tcad — DEVSIM charge-transport backend.

Solves Poisson + drift-diffusion on the same native-2D cross-section mesh
the Palace BoundaryMode pipeline generates, producing carrier maps
n(x, y), p(x, y), terminal currents, and small-signal C(V) per bias point.

DEVSIM itself is optional: install the ``tcad`` extra
(``pip install 'gsim[tcad]'``). Importing this package works without it;
the solver methods raise an actionable error when DEVSIM is missing.

Example::

    from gsim import tcad

    sim = tcad.Simulation()
    sim.geometry(component, stack=stack)
    sim.domain(plane="x=0", window=(-25.0, -15.0))
    sim.doping.add(
        tcad.StepDoping(region="p_rib", dopant_type="acceptor", concentration_cm3=1e18)
    )
    sim.doping.add(
        tcad.StepDoping(region="n_rib", dopant_type="donor", concentration_cm3=1e18)
    )
    sim.contacts.add(name="anode", semiconductor="p_rib", electrode="metal_p")
    sim.contacts.add(name="cathode", semiconductor="n_rib", electrode="metal_n")
    sim.interfaces.add(name="junction", region_a="p_rib", region_b="n_rib")
    sim.solver(mesh_preset="coarse")
    sim.mesh(output_dir="./tcad")
    result = sim.sweep([0.0, 1.0, 2.0], contact="cathode")
"""

from gsim.common.carriers import MobilityModel
from gsim.tcad.api import (
    Contact,
    Contacts,
    Domain,
    Doping,
    Geometry,
    Insulator,
    Interface,
    Interfaces,
    Materials,
    Solver,
    TCADMesh,
)
from gsim.tcad.doping import (
    CallableDoping,
    DopingProfile,
    GaussianDoping,
    ImplantDoping,
    StepDoping,
    TableDoping,
    acceptor_donor_concentrations,
    net_doping_cm3,
)
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap
from gsim.tcad.simulation import Simulation
from gsim.tcad.validation import (
    CapacitanceComparison,
    analytic_capacitance_f_per_cm,
    compare_capacitance,
    estimate_depletion_width_um,
)

__all__ = [
    "BiasPoint",
    "BiasSweepResult",
    "CallableDoping",
    "CapacitanceComparison",
    "CarrierMap",
    "Contact",
    "Contacts",
    "Domain",
    "Doping",
    "DopingProfile",
    "GaussianDoping",
    "Geometry",
    "ImplantDoping",
    "Insulator",
    "Interface",
    "Interfaces",
    "Materials",
    "MobilityModel",
    "Simulation",
    "Solver",
    "StepDoping",
    "TCADMesh",
    "TableDoping",
    "acceptor_donor_concentrations",
    "analytic_capacitance_f_per_cm",
    "compare_capacitance",
    "estimate_depletion_width_um",
    "net_doping_cm3",
]
