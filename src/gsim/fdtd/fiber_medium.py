"""Resolve analytic fiber media from the actual exported transfer mesh."""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Mapping
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

import meshio
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from gsim.fdtd.api import PortSource
from gsim.fdtd.mesh_sections import tetrahedron_section

if TYPE_CHECKING:
    from gsim.common.materials import MaterialSnapshot
    from gsim.fdtd.api import Monitors, SourceType
    from gsim.fdtd.config import FDTDConfig, PlaneMonitorConfig
    from gsim.fdtd.models import MeshGroup, MeshManifest

_GSIM_DIR = str(Path(__file__).parents[1])
_INTERFACE_OFFSET_UM = 1e-6


def _sections(mesh: Any, axis: int, position: float) -> dict[int, Any]:
    """Collect exact material footprints at a plane, in micrometers."""
    points = np.asarray(mesh.points, dtype=float) / 1000
    polygons: dict[int, list[Polygon]] = defaultdict(list)
    for block, tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True):
        if block.type != "tetra":
            continue
        tetrahedra = points[block.data]
        coordinates = tetrahedra[:, :, axis]
        intersects = (coordinates.min(axis=1) <= position + 1e-9) & (
            coordinates.max(axis=1) >= position - 1e-9
        )
        for vertices, tag in zip(tetrahedra[intersects], tags[intersects], strict=True):
            section = tetrahedron_section(vertices, axis, position)
            if section is not None:
                polygons[int(tag)].append(Polygon(section))
    return {tag: unary_union(parts) for tag, parts in polygons.items()}


def _plane_indices(
    sections: dict[int, Any],
    aperture: Any,
    groups: list[MeshGroup],
    materials: Mapping[str, MaterialSnapshot],
) -> set[float] | None:
    """Resolve overlapping footprints with the solver's material priorities."""
    remaining = aperture
    tolerance = max(aperture.area * 1e-10, 1e-18)
    indices = set()
    for _, same_priority in groupby(groups, key=lambda group: group.priority):
        claimed = []
        for group in same_priority:
            section = sections.get(group.physical_tag)
            if section is None:
                continue
            occupied = section.intersection(remaining)
            if occupied.area > tolerance:
                indices.add(materials[group.material].refractive_index)
                claimed.append(occupied)
        if claimed:
            remaining = remaining.difference(unary_union(claimed))
        if remaining.area <= tolerance:
            return indices
    return None


def _receiver_index(
    mesh: Any,
    monitor: PlaneMonitorConfig,
    manifest: MeshManifest,
    materials: Mapping[str, MaterialSnapshot],
) -> float | None:
    """Require one real index across the aperture and both sides of its plane."""
    axis = "xyz".index(monitor.normal[-1])
    plane_axes = [index for index in range(3) if index != axis]
    lower = np.asarray(monitor.region_min) / 1000
    upper = np.asarray(monitor.region_max) / 1000
    aperture = box(*lower[plane_axes], *upper[plane_axes])
    if aperture.area == 0:
        return None
    position = lower[axis]
    domain_lower = manifest.domain_min_nm[axis] / 1000
    domain_upper = manifest.domain_max_nm[axis] / 1000
    positions = {
        max(domain_lower, min(domain_upper, position + offset))
        for offset in (-_INTERFACE_OFFSET_UM, 0, _INTERFACE_OFFSET_UM)
    }
    groups = sorted(
        [*manifest.volumes.values(), *manifest.layers.values()],
        key=lambda group: group.priority,
        reverse=True,
    )
    indices: set[float] = set()
    for sample_position in positions:
        plane_indices = _plane_indices(
            _sections(mesh, axis, sample_position), aperture, groups, materials
        )
        if plane_indices is None:
            return None
        indices.update(plane_indices)
    return indices.pop() if len(indices) == 1 else None


def resolve_fiber_media(
    config: FDTDConfig,
    mesh_path: Path,
    manifest: MeshManifest,
    materials: Mapping[str, MaterialSnapshot],
    *,
    monitors: Monitors,
    source: SourceType,
) -> None:
    """Infer unset indices without mutating public settings or explicit overrides."""
    explicit_indices = {
        monitor.name: monitor.fiber_mode.refractive_index
        for monitor in monitors
        if monitor.fiber_mode is not None
    }
    automatic = []
    for monitor in config.monitors:
        if monitor.fiber_mode is None:
            continue
        index = explicit_indices.get(monitor.name)
        if monitor.name not in explicit_indices and isinstance(source, PortSource):
            index = source.vertical_refractive_index
        if index is None:
            automatic.append(monitor)
    if not automatic:
        return
    mesh = meshio.read(mesh_path)
    for monitor in automatic:
        index = _receiver_index(mesh, monitor, manifest, materials)
        if index is None:
            warnings.warn(
                "Cannot infer a homogeneous medium for fiber monitor "
                f"{monitor.name!r}; "
                "retaining the background refractive index. Set "
                "FiberMode.refractive_index for a custom receiver or "
                "PortSource.vertical_refractive_index for an implicit receiver.",
                UserWarning,
                skip_file_prefixes=(_GSIM_DIR,),
            )
            continue
        fiber_mode = monitor.fiber_mode
        if fiber_mode is not None:
            fiber_mode.refractive_index = index
        if isinstance(source, PortSource) and source.port == monitor.name:
            beam = config.excitation.gaussian_beam
            if beam is not None:
                beam.refractive_index = index
