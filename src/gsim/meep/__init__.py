"""MEEP photonic FDTD simulation module.

Provides a declarative API (``Simulation``) for configuring and running
MEEP FDTD simulations on the GDSFactory+ cloud. No local MEEP
installation required.

Example::

    from gsim import meep

    sim = meep.Simulation()
    sim.geometry(component=ybranch, z_crop="auto")
    sim.materials = {
        "si": Material(permittivity=12.0),
        "SiO2": Material(permittivity=2.1),
    }
    sim.source(port="o1", wavelength=1.55, wavelength_span=0.01)
    sim.num_freqs = 11
    sim.monitors = ["o1", "o2"]
    sim.domain(pml=1.0, margin=0.5)
    sim.solver(resolution=32, simplify_tol=0.01)
    result = sim.run()
"""

from gsim.gcloud import RunResult, register_result_parser
from gsim.meep.mode_solver import (
    mode_x_grid,
    mode_y_grid,
    mode_z_grid,
    refractive_index_profile,
    solve_cross_section_mode,
    solve_slab_mode,
    solve_slab_modes,
    solve_slab_wavelength_sweep,
)
from gsim.meep.models import (
    FDTD,
    Domain,
    DomainConfig,
    Geometry,
    Material,
    ModeResult,
    ModeSolver,
    ModeSource,
    ResolutionConfig,
    SimConfig,
    SourceConfig,
    SParameterResult,
    Symmetry,
    WavelengthConfig,
)
from gsim.meep.results import ModeSweepResult
from gsim.meep.simulation import BuildResult, Simulation


def _parse_meep_result(run_result: RunResult) -> SParameterResult:
    """Parse MEEP cloud results into an SParameterResult."""
    csv_path = run_result.files.get("s_parameters.csv")
    if csv_path is not None:
        return SParameterResult.from_csv(csv_path)
    return SParameterResult()


register_result_parser("meep", _parse_meep_result)

__all__ = [
    "FDTD",
    "BuildResult",
    "Domain",
    "DomainConfig",
    "Geometry",
    "Material",
    "ModeResult",
    "ModeSolver",
    "ModeSource",
    "ModeSweepResult",
    "ResolutionConfig",
    "SParameterResult",
    "SimConfig",
    "Simulation",
    "SourceConfig",
    "Symmetry",
    "WavelengthConfig",
    "mode_x_grid",
    "mode_y_grid",
    "mode_z_grid",
    "refractive_index_profile",
    "solve_cross_section_mode",
    "solve_slab_mode",
    "solve_slab_modes",
    "solve_slab_wavelength_sweep",
]
