"""Canonical passive-PCell resolution models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry.base import BaseGeometry

from gsim.common.materials import MaterialSnapshot

type Bounds3D = tuple[tuple[float, float, float], tuple[float, float, float]]


class PdkResolutionError(ValueError):
    """Base error for canonical PDK resolution."""


class ComponentResolutionError(PdkResolutionError):
    """Raised when a PDK cannot instantiate the requested component."""


class LayerResolutionError(PdkResolutionError):
    """Raised when physical layer information is incomplete or invalid."""


class UnsupportedPortError(PdkResolutionError):
    """Raised when a port cannot be represented by the FDTD contract."""


@dataclass(frozen=True)
class ResolvedLayer:
    """One material-bearing LayerStack entry with resolved geometry."""

    key: str
    declared_name: str | None
    layer: Any
    derived_layer: Any
    geometry: BaseGeometry
    material: str
    zmin: float
    thickness: float
    zmax: float
    sidewall_angle: float
    width_to_z: float
    bias: float | tuple[float, float] | None
    z_to_bias: tuple[list[float], list[float]] | None
    mesh_order: int

    @property
    def z_bounds(self) -> tuple[float, float]:
        """Return ascending vertical bounds, including negative thicknesses."""
        return min(self.zmin, self.zmax), max(self.zmin, self.zmax)


@dataclass(frozen=True)
class ResolvedPort:
    """An axis-aligned port with its active, optionally clipped vertical bounds."""

    name: str
    center: tuple[float, float, float]
    width: float
    orientation: float
    normal: tuple[int, int, int]
    port_type: str
    layer_key: str | None
    material: str | None
    layer_keys: tuple[str, ...] = ()
    z_bounds: tuple[float, float] | None = None

    @property
    def physical_layer_keys(self) -> tuple[str, ...]:
        """Return all owning bands, retaining the original single-layer API."""
        return self.layer_keys or ((self.layer_key,) if self.layer_key else ())

    @property
    def is_vertical(self) -> bool:
        """Return whether this is a free-space vertical optical port."""
        return self.port_type.startswith("vertical_")


@dataclass(frozen=True)
class ResolvedPassivePcell:
    """Canonical physical representation consumed by simulation backends.

    ``layout_bounds`` retains the unpadded XY footprint and resolved material
    height when a backend extends blanket films over its domain. Retain an
    originating PDK module for material-card and alias lookup, while keeping
    ``pdk`` as the authoritative inner PDK object.
    """

    component: Any
    derived_component: Any
    pdk: Any
    layer_stack: Any
    layers: Mapping[str, ResolvedLayer]
    materials: Mapping[str, MaterialSnapshot]
    ports: Mapping[str, ResolvedPort]
    bounds: Bounds3D
    wavelength_um: float
    pdk_module: Any | None = None
    layout_bounds: Bounds3D | None = None
    clipped_layers: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


__all__ = [
    "ComponentResolutionError",
    "LayerResolutionError",
    "PdkResolutionError",
    "ResolvedLayer",
    "ResolvedPassivePcell",
    "ResolvedPort",
    "UnsupportedPortError",
]
