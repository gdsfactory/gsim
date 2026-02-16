"""MEEP simulation models."""

from gsim.meep.models.api import (
    DFTDecay,
    Diagnostics,
    Domain,
    FDTD,
    FieldDecay,
    FixedTime,
    Geometry,
    Material,
    ModeMonitor,
    ModeSource,
)
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
    # New declarative API models
    "DFTDecay",
    "Diagnostics",
    "Domain",
    "FDTD",
    "FieldDecay",
    "FixedTime",
    "Geometry",
    "Material",
    "ModeMonitor",
    "ModeSource",
    # Legacy config models
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
