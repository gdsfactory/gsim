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
    CrossSectionBlock,
    CrossSectionGeometry,
    DiagnosticsConfig,
    DielectricEntry,
    DomainConfig,
    LayerStackEntry,
    MaterialData,
    ModeSolverConfig,
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
    "CrossSectionBlock",
    "CrossSectionGeometry",
    "DiagnosticsConfig",
    "DielectricEntry",
    "Domain",
    "DomainConfig",
    "FDTDConfig",
    "Geometry",
    "LayerStackEntry",
    "Material",
    "MaterialData",
    "ModeResult",
    "ModeSolver",
    "ModeSolverConfig",
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
