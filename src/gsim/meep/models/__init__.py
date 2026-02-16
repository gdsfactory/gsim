"""MEEP simulation models."""

from gsim.meep.models.api import (
    DFTDecay,
    Domain,
    FDTD,
    FieldDecay,
    FixedTime,
    Geometry,
    Material,
    ModeSource,
    Symmetry,
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
    "Domain",
    "FDTD",
    "FieldDecay",
    "FixedTime",
    "Geometry",
    "Material",
    "ModeSource",
    "Symmetry",
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
