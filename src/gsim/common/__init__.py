"""Common components shared between Palace and FDTD solvers.

This module provides shared data models and utilities that can be used
across different electromagnetic solvers (Palace, FDTD, etc.).

Classes:
    Geometry: Wrapper for gdsfactory Component with computed properties
    LayerStack: Layer stack data model with extraction from PDK
"""

from __future__ import annotations

from gsim.common.geometry import Geometry
from gsim.common.stack import (
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

# Alias for backward compatibility
Stack = LayerStack

__all__ = [
    # Geometry
    "Geometry",
    # Stack (LayerStack is the primary name, Stack is an alias)
    "LayerStack",
    "Stack",
    "Layer",
    "StackLayer",
    "ValidationResult",
    # Stack functions
    "get_stack",
    "load_stack_yaml",
    "extract_from_pdk",
    "extract_layer_stack",
    "parse_layer_stack",
    "print_stack",
    "print_stack_table",
    "plot_stack",
    # Materials
    "MATERIALS_DB",
    "MaterialProperties",
    "get_material_properties",
    "material_is_conductor",
    "material_is_dielectric",
]
