"""Mesh bounds and user grid settings accompanying FDTD inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import meshio
import numpy as np

from gsim.fdtd.config import FDTDConfig
from gsim.fdtd.models import FDTDConfigError, MeshManifest


def _validate_manifest(manifest: MeshManifest, config: FDTDConfig) -> None:
    """Require diagnostics to describe the exact physical groups in config."""
    for groups, regions in (
        (manifest.volumes, config.geometry.volumes),
        (manifest.layers, config.geometry.layers),
    ):
        if set(groups) != set(regions) or any(
            (group.physical_tag, group.material, group.priority)
            != (
                regions[name].phys_group,
                regions[name].material,
                regions[name].priority,
            )
            for name, group in groups.items()
        ):
            raise FDTDConfigError("FDTD config and mesh manifest regions disagree")
    ports = config.geometry.ports
    if set(manifest.ports) != set(ports) or any(
        (port.physical_tag, port.layer, port.normal)
        != (ports[name].phys_group, ports[name].layer, ports[name].normal)
        for name, port in manifest.ports.items()
    ):
        raise FDTDConfigError("FDTD config and mesh manifest ports disagree")


def _bounds(points: np.ndarray) -> dict[str, list[float]]:
    """Return finite coordinate bounds for referenced mesh vertices."""
    if points.size == 0 or not np.isfinite(points).all():
        raise FDTDConfigError("Sizing requires nonempty, finite mesh physical groups")
    return {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}


def _mesh_bounds(mesh_path: Path, manifest: MeshManifest) -> dict[str, list[float]]:
    """Measure configured physical groups without including unused vertices."""
    mesh = meshio.read(mesh_path)
    physical_tags = mesh.cell_data.get("gmsh:physical", [])
    groups = [
        *[(group, "tetra", group.name) for group in manifest.volumes.values()],
        *[(group, "tetra", group.name) for group in manifest.layers.values()],
        *[(port, "triangle", port.physical_name) for port in manifest.ports.values()],
    ]
    group_bounds = []
    for group, cell_type, physical_name in groups:
        field = mesh.field_data.get(physical_name)
        if field is None or int(field[0]) != group.physical_tag:
            raise FDTDConfigError(
                f"Mesh physical group {physical_name!r} disagrees with manifest"
            )
        cells = [
            block.data[tags == group.physical_tag]
            for block, tags in zip(mesh.cells, physical_tags, strict=True)
            if block.type == cell_type
        ]
        if not cells or not any(len(block) for block in cells):
            raise FDTDConfigError(
                f"Mesh physical group {physical_name!r} has no {cell_type} cells"
            )
        vertices = np.unique(np.concatenate(cells))
        bounds = _bounds(mesh.points[vertices])
        group_bounds.extend((bounds["min"], bounds["max"]))
    return _bounds(np.asarray(group_bounds))


def generate_metadata(
    manifest: MeshManifest, config: FDTDConfig, mesh_path: str | Path
) -> dict[str, Any]:
    """Describe the written mesh and requested grid without predicting resources."""
    _validate_manifest(manifest, config)
    return {
        "schema_version": 1,
        "solver": "fdtd",
        "mesh": {"bounds_nm": _mesh_bounds(Path(mesh_path), manifest)},
        "grid": {
            "nanometers_per_cell": config.grid.nanometers_per_cell,
            "pml_cells": config.grid.pml_cells,
        },
    }


def write_metadata(
    manifest: MeshManifest, config: FDTDConfig, mesh_path: str | Path
) -> dict[str, Any]:
    """Write measured metadata beside the mesh and configuration."""
    path = Path(mesh_path)
    metadata = generate_metadata(manifest, config, path)
    path.with_name("metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    return metadata
