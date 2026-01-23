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

import warnings
from functools import partial

from gsim.gcloud import print_job_summary
from gsim.gcloud import run_simulation as _run_simulation

# New simulation classes
from gsim.palace.base import SimBase
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
    LayerConfig,
    MagnetostaticConfig,
    MaterialConfig,
    MeshConfig as MeshConfigModel,
    NumericalConfig,
    PortConfig,
    SimulationResult,
    StackConfig,
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

# Stack utilities
from gsim.palace.stack import (
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


# Backward compatibility: PalaceSim as alias for DrivenSim
class PalaceSim(DrivenSim):
    """Legacy alias for DrivenSim.

    DEPRECATED: Use DrivenSim directly instead.

    This class provides backward compatibility with the old fluent API.
    It wraps the old method names to the new API.
    """

    def __init__(self, **kwargs):
        warnings.warn(
            "PalaceSim is deprecated. Use DrivenSim instead:\n"
            "  from gsim.palace import DrivenSim\n"
            "  sim = DrivenSim()",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)

    # Legacy fluent methods that return self
    def geometry(self, component):
        """DEPRECATED: Use set_geometry() instead."""
        self.set_geometry(component)
        return self

    def stack(self, **kwargs):
        """DEPRECATED: Use set_stack() instead."""
        self.set_stack(**kwargs)
        return self

    def material(self, name, **kwargs):
        """DEPRECATED: Use set_material() instead."""
        self.set_material(name, **kwargs)
        return self

    def materials(self, overrides):
        """DEPRECATED: Use set_material() for each material instead."""
        for name, props in overrides.items():
            self.set_material(name, **props)
        return self

    def numerical(self, **kwargs):
        """DEPRECATED: Use set_numerical() instead."""
        self.set_numerical(**kwargs)
        return self

    def port(self, name, **kwargs):
        """DEPRECATED: Use add_port() instead."""
        # Map old parameter names to new
        if "port_type" in kwargs:
            del kwargs["port_type"]  # Not used in new API (always lumped)
        self.add_port(name, **kwargs)
        return self

    def ports(self, **kwargs):
        """DEPRECATED: Configure ports individually with add_port()."""
        if self._component is None:
            raise ValueError("Must call geometry(component) before ports()")
        for gf_port in self._component.ports:
            self.add_port(
                gf_port.name,
                layer=kwargs.get("layer"),
                length=kwargs.get("length"),
                impedance=kwargs.get("impedance", 50.0),
                excited=kwargs.get("excited", True),
            )
        return self

    def cpw_port(
        self,
        port_upper,
        port_lower,
        *,
        layer,
        length,
        impedance=50.0,
        excited=True,
        name=None,
    ):
        """DEPRECATED: Use add_cpw_port() instead."""
        self.add_cpw_port(
            port_upper,
            port_lower,
            layer=layer,
            length=length,
            impedance=impedance,
            excited=excited,
            name=name,
        )
        return self

    def driven(self, **kwargs):
        """DEPRECATED: Use set_driven() instead."""
        self.set_driven(**kwargs)
        return self

    def physics(self, **kwargs):
        """DEPRECATED: Use set_driven() or other problem-specific methods."""
        warnings.warn(
            "physics() is deprecated. Use set_driven() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Map old physics kwargs to driven config
        self.set_driven(
            fmin=kwargs.get("fmin", 1e9),
            fmax=kwargs.get("fmax", 100e9),
            num_points=kwargs.get("num_frequency_points", 40),
            scale=kwargs.get("frequency_scale", "linear"),
        )
        return self


__all__ = [
    # Primary simulation classes (new API)
    "DrivenSim",
    "EigenmodeSim",
    "ElectrostaticSim",
    "SimBase",
    # Legacy (deprecated)
    "PalaceSim",
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
    "LayerConfig",
    "MaterialConfig",
    "MeshConfigModel",
    "NumericalConfig",
    "SimulationResult",
    "StackConfig",
    "ValidationResult",
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
