"""Palace API module for EM simulation with gdsfactory.

This module provides a comprehensive API for setting up and running
electromagnetic simulations using the Palace solver with gdsfactory components.

Features:
    - Layer stack extraction from PDK
    - Port configuration (inplane, via, CPW)
    - Mesh generation with COMSOL-style presets
    - Palace config file generation

Usage:
    from gsim.palace import (
        get_stack, configure_inplane_port, configure_via_port,
        extract_ports, generate_mesh
    )

    # 1. Get layer stack from active PDK
    stack = get_stack()

    # 2. Configure ports on component
    # Inplane port (horizontal, on single layer - for CPW gaps)
    configure_inplane_port(c.ports['o1'], layer='topmetal2', length=5.0)
    configure_inplane_port(c.ports['o2'], layer='topmetal2', length=5.0)

    # Via port (vertical, between two layers - for microstrip feed)
    configure_via_port(c.ports['feed'], from_layer='metal1', to_layer='topmetal2')

    # 3. Extract configured ports
    ports = extract_ports(c, stack)

    # 4. Generate mesh
    result = generate_mesh(
        component=c,
        stack=stack,
        ports=ports,
        output_dir="./simulation",
    )
"""

from __future__ import annotations

from functools import partial

from gsim.gcloud import print_job_summary
from gsim.gcloud import run_simulation as _run_simulation
from gsim.palace.mesh import (
    GroundPlane,
    MeshConfig,
    MeshPreset,
    MeshResult,
    generate_mesh,
)
from gsim.palace.models import (
    DrivenConfig,
    EigenmodeConfig,
    ElectrostaticConfig,
    LayerModel,
    LayerStackModel,
    MagnetostaticConfig,
    MaterialPropertiesModel,
    MeshConfigModel,
    NumericalConfig,
    PhysicsConfig,
    PortConfigModel,
    ProblemType,
    SimulationResult,
    TerminalConfig,
    TransientConfig,
    ValidationResultModel,
    WavePortConfig,
)
from gsim.palace.ports import (
    PalacePort,
    PortGeometry,
    PortType,
    configure_cpw_port,
    configure_inplane_port,
    configure_via_port,
    extract_ports,
)
from gsim.palace.sim import PalaceSim
from gsim.palace.stack import (
    MATERIALS_DB,
    Layer,
    LayerStack,
    MaterialProperties,
    StackLayer,
    ValidationResult,
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
from gsim.viz import plot_mesh

__all__ = [
    # New Pydantic-based fluent API
    "PalaceSim",
    # Problem type configs
    "DrivenConfig",
    "EigenmodeConfig",
    "ElectrostaticConfig",
    "MagnetostaticConfig",
    "TransientConfig",
    "ProblemType",
    # Port configs
    "WavePortConfig",
    "TerminalConfig",
    # Other models
    "LayerModel",
    "LayerStackModel",
    "MaterialPropertiesModel",
    "MeshConfigModel",
    "NumericalConfig",
    "PhysicsConfig",  # Deprecated: use DrivenConfig, etc.
    "PortConfigModel",
    "SimulationResult",
    "ValidationResultModel",
    # Legacy API (still supported)
    "MATERIALS_DB",
    "GroundPlane",
    "Layer",
    "LayerStack",
    "MaterialProperties",
    "MeshConfig",
    "MeshPreset",
    "MeshResult",
    "PalacePort",
    "PortGeometry",
    "PortType",
    "StackLayer",
    "ValidationResult",
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

# Palace-specific run_simulation with job_type preset
run_simulation = partial(_run_simulation, job_type="palace")
