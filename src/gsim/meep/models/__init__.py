"""MEEP simulation models."""

from gsim.meep.models.api import (
    FDTD,
    Domain,
    FiberSource,
    Geometry,
    Material,
    ModeSource,
    Symmetry,
)
from gsim.meep.models.config import (
    AccuracyConfig,
    DiagnosticsConfig,
    DomainConfig,
    FiberSourceConfig,
    LayerStackEntry,
    MaterialData,
    PortData,
    ResolutionConfig,
    SimConfig,
    SourceConfig,
    StoppingConfig,
    SymmetryEntry,
    WavelengthConfig,
)
from gsim.meep.models.results import CouplingResult, SParameterResult

# Backward compatibility alias
FDTDConfig = WavelengthConfig

__all__ = [
    "FDTD",
    "AccuracyConfig",
    "CouplingResult",
    "DiagnosticsConfig",
    "Domain",
    "DomainConfig",
    "FDTDConfig",
    "FiberSource",
    "FiberSourceConfig",
    "Geometry",
    "LayerStackEntry",
    "Material",
    "MaterialData",
    "ModeSource",
    "PortData",
    "ResolutionConfig",
    "SParameterResult",
    "SimConfig",
    "SourceConfig",
    "StoppingConfig",
    "Symmetry",
    "SymmetryEntry",
    "WavelengthConfig",
]
