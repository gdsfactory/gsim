"""FDTD simulation module for gsim.

This module provides a modular API for FDTD (Finite-Difference Time-Domain)
electromagnetic simulations using Tidy3D.

Example:
    ```python
    from gsim.fdtd import FDTDSimulation, Geometry, Material, Physics

    # Create geometry from gdsfactory component
    geometry = Geometry(
        component=my_component,
        layer_stack=stack,
    )

    # Configure materials
    material = Material(mapping={
        "si": td.Medium(permittivity=3.47**2),
        "sio2": td.Medium(permittivity=1.47**2),
    })

    # Set physics parameters
    physics = Physics(wavelength=1.55, bandwidth=0.2)

    # Create and run simulation
    sim = FDTDSimulation(
        geometry=geometry,
        material=material,
        physics=physics,
    )
    tidy3d_sim = sim.get_simulation()
    ```

Submodules:
    - geometry: 3D component modeling and visualization
    - materials: Material definitions and utilities
    - simulation: Simulation classes and mode solvers
"""

from __future__ import annotations

# Geometry
from gsim.fdtd.geometry import (
    Geometry,
    create_web_export,
    export_3d_mesh,
    plot_prism_slices,
    plot_prisms_3d,
    plot_prisms_3d_open3d,
    serve_threejs_visualization,
)

# Materials
from gsim.fdtd.materials import (
    MaterialSpecTidy3d,
    Sparameters,
    Tidy3DElementMapping,
    Tidy3DMedium,
    get_epsilon,
    get_index,
    get_medium,
    get_nk,
    material_name_to_medium,
    material_name_to_tidy3d,
)

# Simulation
from gsim.fdtd.simulation import (
    FDTDSimulation,
    Material,
    Mesh,
    Physics,
    Results,
    Solver,
    Waveguide,
    WaveguideCoupler,
    get_results,
    get_results_batch,
    get_sim_hash,
    sweep_bend_mismatch,
    sweep_coupling_length,
    sweep_fraction_te,
    sweep_mode_area,
    sweep_n_eff,
    sweep_n_group,
    write_sparameters,
)

# Utilities
from gsim.fdtd.util import get_mode_solvers, get_port_normal, sort_layers

__all__ = [
    # Main simulation class
    "FDTDSimulation",
    # Simulation components (Pydantic models)
    "Geometry",
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
    # Materials
    "MaterialSpecTidy3d",
    "Sparameters",
    "Tidy3DElementMapping",
    "Tidy3DMedium",
    "get_epsilon",
    "get_index",
    "get_medium",
    "get_nk",
    "material_name_to_medium",
    "material_name_to_tidy3d",
    # Results
    "get_results",
    "get_results_batch",
    "get_sim_hash",
    # Visualization
    "create_web_export",
    "export_3d_mesh",
    "plot_prism_slices",
    "plot_prisms_3d",
    "plot_prisms_3d_open3d",
    "serve_threejs_visualization",
    # Utilities
    "get_mode_solvers",
    "get_port_normal",
    "sort_layers",
    # Legacy (deprecated)
    "write_sparameters",
]
