"""Canonical PDK resolution for simulation backends."""

from gsim.common.pdk.adapters import build_layer_stack
from gsim.common.pdk.diagnostics import (
    FeatureResolution,
    feature_resolution,
    geometry_report,
    material_report,
    stack_report,
)
from gsim.common.pdk.models import (
    ComponentResolutionError,
    LayerResolutionError,
    PdkResolutionError,
    ResolvedLayer,
    ResolvedPassivePcell,
    ResolvedPort,
    UnsupportedPortError,
)
from gsim.common.pdk.overlays import StackOverlay, get_stack_overlay
from gsim.common.pdk.resolve import resolve_passive_pcell
from gsim.common.pdk.stack import (
    ResolvedDielectric,
    ResolvedProcessLayer,
    ResolvedStack,
    classify_layer_role,
    resolve_layer_stack,
    resolve_material_properties,
)

__all__ = [
    "ComponentResolutionError",
    "FeatureResolution",
    "LayerResolutionError",
    "PdkResolutionError",
    "ResolvedDielectric",
    "ResolvedLayer",
    "ResolvedPassivePcell",
    "ResolvedPort",
    "ResolvedProcessLayer",
    "ResolvedStack",
    "StackOverlay",
    "UnsupportedPortError",
    "build_layer_stack",
    "classify_layer_role",
    "feature_resolution",
    "geometry_report",
    "get_stack_overlay",
    "material_report",
    "resolve_layer_stack",
    "resolve_material_properties",
    "resolve_passive_pcell",
    "stack_report",
]
