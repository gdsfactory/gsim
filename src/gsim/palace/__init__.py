"""Palace API module for EM simulation with gdsfactory.

This module provides a comprehensive API for setting up and running
electromagnetic simulations using the Palace solver with gdsfactory components.

Features:
    - Problem-specific simulation classes (DrivenSim, EigenmodeSim, ElectrostaticSim)
    - Layer stack extraction from PDK
    - Port configuration (inplane, via, CPW)
    - Mesh generation with COMSOL-style presets
    - Palace config file generation

Usage:
    from gsim.palace import DrivenSim

    # Create and configure simulation
    sim = DrivenSim()
    sim.set_geometry(component)
    sim.set_stack(air_above=300.0)
    sim.add_cpw_port("o1", layer="topmetal2", s_width=10, gap_width=6, length=5)
    sim.set_driven(fmin=1e9, fmax=100e9)

    # Generate mesh and run
    sim.set_output_dir("./sim")
    sim.mesh(preset="fine")
    results = sim.run()
"""

from __future__ import annotations

from functools import partial

# Common components (shared with FDTD)
from gsim.common import Geometry, LayerStack, Stack

# Stack utilities (from common, shared with FDTD)
from gsim.common.stack import (
    MATERIALS_DB,
    Layer,
    MaterialProperties,
    StackLayer,
    extract_from_pdk,
    extract_layer_stack,
    get_material_properties,
    get_stack,
    load_stack_yaml,
    material_is_conductor,
    material_is_dielectric,
    parse_layer_stack,
    plot_stack,
    print_stack,
    print_stack_table,
)
from gsim.gcloud import RunResult, print_job_summary, register_result_parser
from gsim.gcloud import run_simulation as _run_simulation

# New simulation classes (composition, no inheritance)
from gsim.palace.driven import DrivenSim
from gsim.palace.eigenmode import EigenmodeSim
from gsim.palace.electrostatic import ElectrostaticSim

# Mesh utilities
from gsim.palace.mesh import (
    GroundPlane,
    MeshConfig,
    MeshPreset,
    MeshResult,
    generate_mesh,
)

# Models (new submodule)
from gsim.palace.models import (
    CPWPortConfig,
    DrivenConfig,
    EigenmodeConfig,
    ElectrostaticConfig,
    GeometryConfig,
    MagnetostaticConfig,
    MaterialConfig,
    NumericalConfig,
    PortConfig,
    SimulationResult,
    TerminalConfig,
    TransientConfig,
    ValidationResult,
    WavePortConfig,
)
from gsim.palace.models import (
    MeshConfig as MeshConfigModel,
)

# Port utilities
from gsim.palace.ports import (
    PalacePort,
    PortGeometry,
    PortType,
    configure_cpw_port,
    configure_inplane_port,
    configure_via_port,
    extract_ports,
)

# Visualization
from gsim.viz import plot_mesh

__all__ = [
    "MATERIALS_DB",
    "CPWPortConfig",
    "DrivenConfig",
    "DrivenSim",
    "EigenmodeConfig",
    "EigenmodeSim",
    "ElectrostaticConfig",
    "ElectrostaticSim",
    "Geometry",
    "GeometryConfig",
    "GroundPlane",
    "Layer",
    "LayerStack",
    "MagnetostaticConfig",
    "MaterialConfig",
    "MaterialProperties",
    "MeshConfig",
    "MeshConfigModel",
    "MeshPreset",
    "MeshResult",
    "NumericalConfig",
    "PalacePort",
    "PortConfig",
    "PortGeometry",
    "PortType",
    "SimulationResult",
    "Stack",
    "StackLayer",
    "TerminalConfig",
    "TransientConfig",
    "ValidationResult",
    "WavePortConfig",
    "configure_cpw_port",
    "configure_inplane_port",
    "configure_via_port",
    "extract_from_pdk",
    "extract_layer_stack",
    "extract_ports",
    "generate_mesh",
    "get_material_properties",
    "get_stack",
    "load_stack_yaml",
    "material_is_conductor",
    "material_is_dielectric",
    "parse_layer_stack",
    "plot_mesh",
    "plot_stack",
    "print_job_summary",
    "print_stack",
    "print_stack_table",
    "run_simulation",
]


def _parse_palace_result(run_result: RunResult) -> dict:
    """Parse Palace cloud results — returns the files dict."""
    return run_result.files


register_result_parser("palace", _parse_palace_result)

# Palace-specific run_simulation with job_type preset
run_simulation = partial(_run_simulation, job_type="palace")
