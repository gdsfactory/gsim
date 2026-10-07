"""gsim.tcad — DEVSIM charge-transport backend.

Solves Poisson + drift-diffusion on the same native-2D cross-section mesh
the Palace BoundaryMode pipeline generates, producing carrier maps
n(x, y), p(x, y), terminal currents, and small-signal C(V) per bias point.

DEVSIM itself is optional: install the ``tcad`` extra
(``pip install 'gsim[tcad]'``). Importing this package works without it;
the solver methods raise an actionable error when DEVSIM is missing.

Example::

    from gsim import tcad

    sim = tcad.ChargeTransportSim()
    sim.set_geometry(component)
    sim.set_stack(stack)
    sim.set_cross_section("x=0", window=(-25.0, -15.0))
    sim.add_contact(name="anode", layer_a="metal1", layer_b="p_rib")
    sim.add_contact(name="cathode", layer_a="metal1", layer_b="n_rib")
    sim.add_doping(
        tcad.StepDoping(region="p_rib", dopant_type="acceptor", concentration_cm3=1e18)
    )
    sim.set_output_dir("./tcad")
    sim.mesh(preset="coarse")
    result = sim.sweep([0.0, -1.0, -2.0], contact="cathode")
"""

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
from gsim.tcad.sim import ChargeTransportSim, Insulator
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
    "ChargeTransportSim",
    "DopingProfile",
    "GaussianDoping",
    "ImplantDoping",
    "Insulator",
    "StepDoping",
    "TableDoping",
    "acceptor_donor_concentrations",
    "analytic_capacitance_f_per_cm",
    "compare_capacitance",
    "estimate_depletion_width_um",
    "net_doping_cm3",
]
