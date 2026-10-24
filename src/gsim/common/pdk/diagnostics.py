"""Backend-neutral diagnostics for PDK-authoritative simulation stacks.

These helpers turn a :class:`~gsim.common.pdk.stack.ResolvedStack` and an
optional component into auditable reports used by tests and validation runs.
They deliberately do not import a solver: stack, material, geometry, and mesh
resolution can all be checked before a simulation is submitted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gsim.common.pdk.stack import ResolvedStack
from gsim.common.stack.materials import resolve_material_at_wavelength


def stack_report(resolved: ResolvedStack) -> list[dict[str, Any]]:
    """Return one row per authoritative process layer."""
    rows: list[dict[str, Any]] = []
    for key, layer in resolved.layers.items():
        rows.append(
            {
                "layer": key,
                "gds_layer": list(layer.gds_layer),
                "material": layer.material,
                "zmin_um": layer.zmin,
                "zmax_um": layer.zmax,
                "thickness_um": layer.thickness,
                "sidewall_angle_deg": layer.sidewall_angle,
                "mesh_order": layer.mesh_order,
                "layer_type": layer.layer_type,
                "drawn": layer.drawn,
            }
        )
    return rows


def material_report(
    resolved: ResolvedStack,
    *,
    wavelength_um: float = 1.55,
) -> dict[str, dict[str, Any]]:
    """Return resolved optical/RF properties for every material in the stack."""
    report: dict[str, dict[str, Any]] = {}
    for name in sorted(resolved.materials):
        material = resolve_material_at_wavelength(name, wavelength_um)
        if material is None:
            report[name] = {
                "found": False,
                "wavelength_um": wavelength_um,
            }
            continue
        epsilon = material.permittivity_scalar
        report[name] = {
            "found": True,
            "wavelength_um": wavelength_um,
            "permittivity": material.permittivity,
            "refractive_index": math.sqrt(epsilon) if epsilon is not None else None,
            "conductivity": material.conductivity,
            "loss_tangent": material.loss_tangent,
            "behavior": material.behavior,
            "model_type": material.model_type,
            "model_source": material.model_source,
            "within_validity": material.within_validity,
            "validity_note": material.validity_note,
        }
    return report


def geometry_report(
    component: Any,
    resolved: ResolvedStack,
    *,
    wavelength_um: float | None = None,
) -> dict[str, Any]:
    """Return bbox, port, and per-physical-layer polygon statistics.

    Layer regions are evaluated through the PDK LayerLevel expression, so a
    derived layer such as Cornerstone ``core = WG - GRA`` is measured as the
    material region the solver will actually see.
    """
    from gsim.common.polygon import layer_level_region

    bbox = component.dbbox()
    layer_rows: list[dict[str, Any]] = []
    for key, layer in resolved.layers.items():
        level = resolved.layer_stack.layers[key]
        region = layer_level_region(component, level)
        layer_rows.append(
            {
                "layer": key,
                "gds_layer": list(layer.gds_layer),
                "material": layer.material,
                "polygons": int(region.count()),
                "area_um2": float(region.area()) / 1e6,
                "drawn": layer.drawn,
            }
        )

    ports = [
        {
            "name": port.name,
            "center": [float(value) for value in port.dcenter],
            "width": float(port.width),
            "orientation": None
            if port.orientation is None
            else float(port.orientation),
            "layer": (
                list(port.layer) if isinstance(port.layer, tuple) else str(port.layer)
            ),
        }
        for port in component.ports
    ]

    return {
        "bbox": [bbox.left, bbox.bottom, bbox.right, bbox.top],
        "size_um": [bbox.right - bbox.left, bbox.top - bbox.bottom],
        "wavelength_um": wavelength_um,
        "layers": layer_rows,
        "ports": ports,
    }


@dataclass(frozen=True)
class FeatureResolution:
    """Grid resolution of one critical geometric feature."""

    name: str
    size_um: float
    cells: float
    required_cells: float
    resolved: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable row."""
        return {
            "name": self.name,
            "size_um": self.size_um,
            "cells": self.cells,
            "required_cells": self.required_cells,
            "resolved": self.resolved,
        }


def feature_resolution(
    features: Mapping[str, float],
    *,
    resolution: float,
    required_cells: float = 4.0,
) -> dict[str, Any]:
    """Check that each feature spans enough grid cells.

    Args:
        features: ``{feature_name: size_um}`` for the smallest critical
            dimensions (gaps, widths, grating periods, layer thicknesses).
        resolution: Grid cells per micrometre.
        required_cells: Minimum number of cells across each feature.

    Returns:
        Report with the grid step, per-feature rows, and an overall
        ``resolved`` flag.  The smallest feature drives the suggested
        resolution.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if required_cells <= 0:
        raise ValueError("required_cells must be positive")

    rows = []
    for name, size_um in features.items():
        size = float(size_um)
        cells = size * resolution
        rows.append(
            FeatureResolution(
                name=name,
                size_um=size,
                cells=cells,
                required_cells=required_cells,
                resolved=cells >= required_cells,
            )
        )

    smallest = min((row.size_um for row in rows), default=None)
    suggested = (
        required_cells / smallest if smallest is not None and smallest > 0 else None
    )
    return {
        "resolution": float(resolution),
        "grid_step_um": 1.0 / resolution,
        "required_cells": required_cells,
        "resolved": all(row.resolved for row in rows),
        "features": [row.to_dict() for row in rows],
        "suggested_resolution": suggested,
    }


__all__ = [
    "FeatureResolution",
    "feature_resolution",
    "geometry_report",
    "material_report",
    "stack_report",
]
