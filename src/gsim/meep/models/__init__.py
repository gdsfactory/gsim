"""MEEP simulation models."""

from gsim.meep.models.api import (
    FDTD,
    Domain,
    Geometry,
    Material,
    ModeSolver,
    ModeSource,
    Symmetry,
)
from gsim.meep.models.config import (
    AccuracyConfig,
    DiagnosticsConfig,
    DomainConfig,
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
from gsim.meep.models.results import ModeResult, SParameterResult

# Backward compatibility alias
FDTDConfig = WavelengthConfig

__all__ = [
    "FDTD",
    "AccuracyConfig",
    "DiagnosticsConfig",
    "Domain",
    "DomainConfig",
    "FDTDConfig",
    "Geometry",
    "LayerStackEntry",
    "Material",
    "MaterialData",
    "ModeResult",
    "ModeSolver",
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
