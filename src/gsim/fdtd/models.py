"""Shared models and errors for GDSFactory FDTD artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class FDTDArtifactError(ValueError):
    """Base error for invalid or unsupported GDSFactory FDTD artifacts."""


class FDTDGeometryError(FDTDArtifactError):
    """Raised when geometry cannot produce a valid GDSFactory FDTD mesh."""


class FDTDConfigError(FDTDArtifactError):
    """Raised when settings cannot produce a valid GDSFactory FDTD config."""


@dataclass(frozen=True)
class MeshGroup:
    """One named three-dimensional Gmsh physical group."""

    name: str
    physical_tag: int
    material: str
    priority: int


@dataclass(frozen=True)
class PortMeshGroup:
    """One named two-dimensional Gmsh port physical group."""

    name: str
    physical_name: str
    physical_tag: int
    layer: str
    normal: tuple[int, int, int]


@dataclass(frozen=True)
class MeshManifest:
    """Authoritative physical-group mapping emitted with a Gmsh mesh."""

    volumes: dict[str, MeshGroup]
    layers: dict[str, MeshGroup]
    ports: dict[str, PortMeshGroup]
    domain_min_nm: tuple[float, float, float]
    domain_max_nm: tuple[float, float, float]


@dataclass(frozen=True)
class SimulationArtifacts:
    """Paths and metadata produced by :meth:`Simulation.write`."""

    mesh_path: Path
    config_path: Path
    manifest: MeshManifest


__all__ = [
    "FDTDArtifactError",
    "FDTDConfigError",
    "FDTDGeometryError",
    "MeshGroup",
    "MeshManifest",
    "PortMeshGroup",
    "SimulationArtifacts",
]
