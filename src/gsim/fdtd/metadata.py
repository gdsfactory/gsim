"""Mesh bounds and client grid counts accompanying FDTD inputs."""

from __future__ import annotations

import json
import math
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
    """Describe the written mesh and grid without deciding cloud resources."""
    _validate_manifest(manifest, config)
    return {
        "schema_version": 1,
        "solver": "fdtd",
        "mesh": {"bounds_nm": _mesh_bounds(Path(mesh_path), manifest)},
        "grid": {
            "nanometers_per_cell": config.grid.nanometers_per_cell,
            "pml_cells": config.grid.pml_cells,
            "non_pml_bounds": config.grid.non_pml_bounds.model_dump(mode="json"),
            "cell_counts": _cell_counts(config),
        },
    }


def _cell_counts(config: FDTDConfig) -> dict[str, Any]:
    """Estimate the engine's z16 layout; the server must verify this hint."""
    # Match FDTD's MaterialGridBuilderV2::ComputeGridShape / ComputeAxisLayout:
    # https://github.com/doplaydo/FDTD/blob/c7d7443/src/fdtd_grid_layout.cc
    # The field path aligns Z to kMaxSimdWidth=16 on every CPU architecture.
    bounds = config.grid.non_pml_bounds
    cell_size = config.grid.nanometers_per_cell
    interior = [
        math.ceil((upper - lower) / cell_size)
        for lower, upper in zip(bounds.min, bounds.max, strict=True)
    ]
    alignment = [1, 1, 16]
    aligned_interior = [
        ((cells + group - 1) // group) * group
        for cells, group in zip(interior, alignment, strict=True)
    ]
    pml_per_face = [
        ((config.grid.pml_cells + group - 1) // group) * group for group in alignment
    ]
    total = [
        cells + 2 * (pml + guard)
        for cells, pml, guard in zip(
            aligned_interior, pml_per_face, alignment, strict=True
        )
    ]
    return {
        "scope": "client_estimate",
        "layout": "fdtd-z16-v1",
        "interior_cells_xyz": interior,
        "alignment_cells_xyz": [
            aligned - cells
            for aligned, cells in zip(aligned_interior, interior, strict=True)
        ],
        "pml_cells_per_face_xyz": pml_per_face,
        "guard_cells_per_face_xyz": alignment,
        "total_cells_xyz": total,
        "total_cells": math.prod(total),
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
