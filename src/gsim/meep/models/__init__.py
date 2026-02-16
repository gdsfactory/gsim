"""MEEP simulation models."""

from gsim.meep.models.api import (
    FDTD,
    DFTDecay,
    Domain,
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
from gsim.meep.models.results import SParameterResult

# Backward compatibility alias
FDTDConfig = WavelengthConfig

__all__ = [
    "FDTD",
    "AccuracyConfig",
    "DFTDecay",
    "DiagnosticsConfig",
    "Domain",
    "DomainConfig",
    "FDTDConfig",
    "FieldDecay",
    "FixedTime",
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
