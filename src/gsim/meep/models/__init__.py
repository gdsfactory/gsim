"""MEEP simulation models."""

from gsim.meep.models.config import (
    AccuracyConfig,
    DiagnosticsConfig,
    WavelengthConfig,
    LayerStackEntry,
    DomainConfig,
    MaterialData,
    PortData,
    ResolutionConfig,
    SimConfig,
    SourceConfig,
    StoppingConfig,
    SymmetryEntry,
)
from gsim.meep.models.results import SParameterResult

# Backward compatibility alias
FDTDConfig = WavelengthConfig

__all__ = [
    "AccuracyConfig",
    "DiagnosticsConfig",
    "FDTDConfig",
    "WavelengthConfig",
    "LayerStackEntry",
    "DomainConfig",
    "MaterialData",
    "PortData",
    "ResolutionConfig",
    "SParameterResult",
    "SimConfig",
    "SourceConfig",
    "StoppingConfig",
    "SymmetryEntry",
]
