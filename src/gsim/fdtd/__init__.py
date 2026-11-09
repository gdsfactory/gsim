"""Ergonomic setup, cloud execution, and results for GDSFactory FDTD."""

from gsim.fdtd.api import (
    DipoleSource,
    Domain,
    FiberMode,
    Fields,
    GaussianBeamSource,
    Geometry,
    Heatmap,
    LineCurrentSource,
    Material,
    Materials,
    ModeSelection,
    Monitors,
    PlaneMonitor,
    PortSource,
    Solver,
)
from gsim.fdtd.config import FDTDConfig
from gsim.fdtd.field_results import ComplexFieldResult, FieldCoordinates
from gsim.fdtd.models import (
    FDTDArtifactError,
    FDTDConfigError,
    FDTDGeometryError,
    GeometryClippingWarning,
    MeshManifest,
    SimulationArtifacts,
)
from gsim.fdtd.normalization import (
    CouplingEfficiencyResult,
    fiber_coupling_efficiency,
    gaussian_coupling_efficiency,
)
from gsim.fdtd.results import (
    ComplexTrace,
    FDTDResult,
    HeatmapResult,
    MonitorResults,
    PlaneMonitorResult,
    PortOutputResults,
    SParameterResults,
    s_matrix,
)
from gsim.fdtd.viz import MeshViewer
from gsim.fdtd.workflow import Simulation
from gsim.gcloud import RunResult, register_result_parser


def _parse_fdtd_result(run_result: RunResult) -> FDTDResult:
    """Parse downloaded GDSFactory FDTD JSON and sidecar outputs."""
    return FDTDResult.from_run_result(run_result)


register_result_parser("fdtd", _parse_fdtd_result)

__all__ = [
    "ComplexFieldResult",
    "ComplexTrace",
    "CouplingEfficiencyResult",
    "DipoleSource",
    "Domain",
    "FDTDArtifactError",
    "FDTDConfig",
    "FDTDConfigError",
    "FDTDGeometryError",
    "FDTDResult",
    "FiberMode",
    "FieldCoordinates",
    "Fields",
    "GaussianBeamSource",
    "Geometry",
    "GeometryClippingWarning",
    "Heatmap",
    "HeatmapResult",
    "LineCurrentSource",
    "Material",
    "Materials",
    "MeshManifest",
    "MeshViewer",
    "ModeSelection",
    "MonitorResults",
    "Monitors",
    "PlaneMonitor",
    "PlaneMonitorResult",
    "PortOutputResults",
    "PortSource",
    "SParameterResults",
    "Simulation",
    "SimulationArtifacts",
    "Solver",
    "fiber_coupling_efficiency",
    "gaussian_coupling_efficiency",
    "s_matrix",
]
