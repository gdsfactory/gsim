"""Common components shared between Palace and FDTD solvers.

This module provides shared data models and utilities that can be used
across different electromagnetic solvers (Palace, FDTD, etc.).

Classes:
    Geometry: Wrapper for gdsfactory Component with computed properties
    LayerStack: Layer stack data model with extraction from PDK
"""

from __future__ import annotations

from gsim.common.geometry import Geometry
from gsim.common.geometry_model import GeometryModel, Prism, extract_geometry_model
from gsim.common.stack import LayerStack, Layer, ValidationResult

# Alias for backward compatibility
Stack = LayerStack

__all__ = [
    "Geometry",
    "GeometryModel",
    "LayerStack",
    "Layer",
    "Prism",
    "Stack",
    "ValidationResult",
    "extract_geometry_model",
]
