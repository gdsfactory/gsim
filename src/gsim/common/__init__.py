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
from gsim.common.polygon_utils import decimate, klayout_to_shapely, shapely_to_klayout
from gsim.common.qpdk import create_etched_component
from gsim.common.rf_layers import cpw_layer_stack
from gsim.common.stack import LayerStack, ValidationResult

# Alias for backward compatibility
Stack = LayerStack

__all__ = [
    "Geometry",
    "GeometryModel",
    "LayerStack",
    "Prism",
    "Stack",
    "ValidationResult",
    "cpw_layer_stack",
    "create_etched_component",
    "decimate",
    "extract_geometry_model",
    "klayout_to_shapely",
    "shapely_to_klayout",
]
