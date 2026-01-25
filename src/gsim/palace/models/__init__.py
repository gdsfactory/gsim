"""Pydantic models for Palace EM simulation configuration.

This module provides Pydantic v2 models for configuring Palace simulations,
offering validation, serialization, and a clean API.

Submodules:
    - geometry: GeometryConfig
    - stack: MaterialConfig (Layer/Stack are in gsim.common.stack)
    - ports: PortConfig, CPWPortConfig, TerminalConfig, WavePortConfig
    - mesh: MeshConfig
    - numerical: NumericalConfig
    - problems: DrivenConfig, EigenmodeConfig, ElectrostaticConfig, etc.
    - results: SimulationResult, ValidationResult
"""

from __future__ import annotations

from gsim.palace.models.geometry import GeometryConfig
from gsim.palace.models.mesh import MeshConfig
from gsim.palace.models.numerical import NumericalConfig
from gsim.palace.models.ports import (
    CPWPortConfig,
    PortConfig,
    TerminalConfig,
    WavePortConfig,
)
from gsim.palace.models.problems import (
    DrivenConfig,
    EigenmodeConfig,
    ElectrostaticConfig,
    MagnetostaticConfig,
    TransientConfig,
)
from gsim.palace.models.results import SimulationResult, ValidationResult
from gsim.palace.models.stack import MaterialConfig

__all__ = [
    # Geometry
    "GeometryConfig",
    # Stack
    "MaterialConfig",
    # Ports
    "CPWPortConfig",
    "PortConfig",
    "TerminalConfig",
    "WavePortConfig",
    # Mesh
    "MeshConfig",
    # Numerical
    "NumericalConfig",
    # Problems
    "DrivenConfig",
    "EigenmodeConfig",
    "ElectrostaticConfig",
    "MagnetostaticConfig",
    "TransientConfig",
    # Results
    "SimulationResult",
    "ValidationResult",
]
