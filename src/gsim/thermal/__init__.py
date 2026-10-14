"""gsim.thermal -- 2D thermal cross-sections of heaters and phase shifters.

The physics nouns: :class:`ThermalSim` sets up a heater over waveguides
and solves it, steady (:class:`ThermalResult`: P_pi, crosstalk,
resistance, V_pi, :class:`TemperatureMap`) and switched on
(:class:`StepResponse`: tau); :class:`HeaterReport` puts the figures in
one table and :class:`ThermalOptions` holds the numerical knobs.

Underneath is a pure functional core, public for custom pipelines:
:func:`build_cross_section_mesh`, :func:`solve_steady`,
:func:`solve_transient`, :func:`joule_heat_density`,
:func:`drive_power_per_length` (a voltage, current or total power as
the power per length the core takes) and
:func:`solve_thermo_optic_shift`.

Units: lengths in um, power in mW, temperature rise in K, time in us.
scikit-fem and femwell are optional (the ``femwell`` extra); importing
this package does not need them, only meshing or solving does.
"""

from gsim.thermal.mesh import CrossSectionMesh, build_cross_section_mesh
from gsim.thermal.optical import (
    ModeShift,
    sample_temperature,
    solve_thermo_optic_shift,
    thermo_optic_permittivity,
)
from gsim.thermal.results import (
    HeaterReport,
    StepResponse,
    TemperatureMap,
    ThermalResult,
)
from gsim.thermal.sim import ThermalOptions, ThermalSim
from gsim.thermal.solver import (
    HeatOperator,
    drive_power_per_length,
    joule_heat_density,
    solve_steady,
    solve_transient,
)

__all__ = [
    "CrossSectionMesh",
    "HeatOperator",
    "HeaterReport",
    "ModeShift",
    "StepResponse",
    "TemperatureMap",
    "ThermalOptions",
    "ThermalResult",
    "ThermalSim",
    "build_cross_section_mesh",
    "drive_power_per_length",
    "joule_heat_density",
    "sample_temperature",
    "solve_steady",
    "solve_thermo_optic_shift",
    "solve_transient",
    "thermo_optic_permittivity",
]
