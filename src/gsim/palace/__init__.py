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
    sim.add_cpw_port("P2", "P1", layer="topmetal2", length=5.0)
    sim.set_driven(fmin=1e9, fmax=100e9)

    # Generate mesh and run
    sim.mesh("./sim", preset="fine")
    results = sim.simulate()
"""

from __future__ import annotations

from functools import partial

from gsim.gcloud import print_job_summary
from gsim.gcloud import run_simulation as _run_simulation

# Common components (shared with FDTD)
from gsim.common import Geometry, LayerStack, Stack

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
    MeshConfig as MeshConfigModel,
    NumericalConfig,
    PortConfig,
    SimulationResult,
    TerminalConfig,
    TransientConfig,
    ValidationResult,
    WavePortConfig,
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

# Stack utilities (from common, shared with FDTD)
from gsim.common.stack import (
    MATERIALS_DB,
    Layer,
    LayerStack,
    MaterialProperties,
    StackLayer,
    ValidationResult as StackValidationResult,
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

# Visualization
from gsim.viz import plot_mesh


__all__ = [
    # Primary simulation classes (new API)
    "DrivenSim",
    "EigenmodeSim",
    "ElectrostaticSim",
    # Common components (shared with FDTD)
    "Geometry",
    "Stack",
    # Problem configs
    "DrivenConfig",
    "EigenmodeConfig",
    "ElectrostaticConfig",
    "MagnetostaticConfig",
    "TransientConfig",
    # Port configs
    "CPWPortConfig",
    "PortConfig",
    "TerminalConfig",
    "WavePortConfig",
    # Other configs
    "GeometryConfig",
    "MaterialConfig",
    "MeshConfigModel",
    "NumericalConfig",
    "SimulationResult",
    "ValidationResult",
    # Stack utilities
    "MATERIALS_DB",
    "Layer",
    "LayerStack",
    "MaterialProperties",
    "extract_from_pdk",
    "extract_layer_stack",
    "get_material_properties",
    "get_stack",
    "load_stack_yaml",
    "material_is_conductor",
    "material_is_dielectric",
    "parse_layer_stack",
    "plot_stack",
    "print_stack",
    "print_stack_table",
    # Mesh utilities
    "GroundPlane",
    "MeshConfig",
    "MeshPreset",
    "MeshResult",
    "generate_mesh",
    "plot_mesh",
    # Port utilities
    "PalacePort",
    "PortGeometry",
    "PortType",
    "StackLayer",
    "configure_cpw_port",
    "configure_inplane_port",
    "configure_via_port",
    "extract_ports",
    # Cloud
    "print_job_summary",
    "run_simulation",
]

# Palace-specific run_simulation with job_type preset
run_simulation = partial(_run_simulation, job_type="palace")
