"""Pydantic models for Palace EM simulation configuration.

This module provides Pydantic v2 models for configuring Palace simulations,
offering validation, serialization, and a clean API.

Submodules:
    - geometry: GeometryConfig
    - stack: MaterialConfig (Layer/Stack are in gsim.common.stack)
    - ports: PortConfig, CPWPortConfig, TerminalConfig, WavePortConfig
    - mesh: MeshConfig
    - solver: SolverConfig, LinearSolverConfig and problem-specific subclasses
    - numerical: NumericalConfig (deprecated compatibility model), RefinementConfig
    - problems: DrivenConfig, EigenmodeConfig, ElectrostaticConfig, etc.
    - results: SimulationResult, ValidationResult
"""

from __future__ import annotations

from gsim.palace.models.cross_section import CrossSectionPlaneConfig
from gsim.palace.models.geometry import GeometryConfig
from gsim.palace.models.mesh import MeshConfig
from gsim.palace.models.numerical import NumericalConfig, RefinementConfig
from gsim.palace.models.pec import PECBlockConfig
from gsim.palace.models.ports import (
    CPWPortConfig,
    ImpedanceBoundaryConfig,
    PortConfig,
    TerminalConfig,
    TwoTerminalPortConfig,
    WavePortConfig,
)
from gsim.palace.models.problems import (
    BoundaryModeConfig,
    DrivenConfig,
    EigenmodeConfig,
    ElectrostaticConfig,
    MagnetostaticConfig,
    TransientConfig,
)
from gsim.palace.models.results import SimulationResult, ValidationResult
from gsim.palace.models.solver import (
    BoundaryModeSolverConfig,
    DrivenSolverConfig,
    EigenmodeSolverConfig,
    ElectrostaticSolverConfig,
    LinearSolverConfig,
    SolverConfig,
)
from gsim.palace.models.stack import MaterialConfig

__all__ = [
    "BoundaryModeConfig",
    "BoundaryModeSolverConfig",
    "CPWPortConfig",
    "CrossSectionPlaneConfig",
    "DrivenConfig",
    "DrivenSolverConfig",
    "EigenmodeConfig",
    "EigenmodeSolverConfig",
    "ElectrostaticConfig",
    "ElectrostaticSolverConfig",
    "GeometryConfig",
    "ImpedanceBoundaryConfig",
    "LinearSolverConfig",
    "MagnetostaticConfig",
    "MaterialConfig",
    "MeshConfig",
    "NumericalConfig",
    "PECBlockConfig",
    "PortConfig",
    "RefinementConfig",
    "SimulationResult",
    "SolverConfig",
    "TerminalConfig",
    "TransientConfig",
    "TwoTerminalPortConfig",
    "ValidationResult",
    "WavePortConfig",
]
