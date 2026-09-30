"""Evaluating a Carrier map on a mesh other than the one it was solved on.

Each Stage of a modulator Study meshes its own Window (ADR 0002), so the
charge solve's Carrier map has to be carried onto the optical and RF
meshes before either solver can see it. :func:`transfer_carriers` is that
step: it interpolates the solved concentrations onto the target mesh's
nodes or element centroids, resolves physical-group names internally so
callers never map gmsh tags themselves, and fills points the source
domain does not cover with a value the caller chooses.

The source domain is the convex hull of the Carrier map's sample points.
Interpolation is linear, so a linear field is reproduced exactly and a
transfer onto the source mesh itself is a no-op to floating-point
tolerance.

Pure meshio/scipy: no solver runtime is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import meshio
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from gsim.common.interpolate import sample_at
from gsim.common.mesh_regions import element_regions, group_names, node_regions

__all__ = ["CarrierMapLike", "TransferredCarriers", "transfer_carriers"]

Granularity = Literal["nodes", "elements"]
Fill = float | tuple[float, float] | Literal["nearest"]


@runtime_checkable
class CarrierMapLike(Protocol):
    """The part of a Carrier map this module reads.

    :class:`gsim.tcad.results.CarrierMap` satisfies it; so does any object
    carrying the same four arrays.
    """

    x_um: NDArray[np.float64]
    y_um: NDArray[np.float64]
    electrons_cm3: NDArray[np.float64]
    holes_cm3: NDArray[np.float64]


class TransferredCarriers(BaseModel):
    """Carrier concentrations sampled on a target mesh.

    Attributes:
        x_um: Target point coordinates (um), in mesh order.
        y_um: Target point coordinates (um), in mesh order.
        electrons_cm3: Electron concentration at each target point (cm^-3).
        holes_cm3: Hole concentration at each target point (cm^-3).
        region: Physical-group name of each target point. Element points
            carry their own group; node points carry the group of one
            incident element (a node on a region boundary belongs to
            several, and the lowest-numbered incident element wins).
        filled: True where the value came from ``fill`` rather than from
            the Carrier map — outside the source domain, or outside the
            requested regions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x_um: NDArray[np.float64]
    y_um: NDArray[np.float64]
    electrons_cm3: NDArray[np.float64]
    holes_cm3: NDArray[np.float64]
    region: list[str]
    filled: NDArray[np.bool_]


def _read_mesh(mesh: meshio.Mesh | str | Path) -> meshio.Mesh:
    """Load the target mesh if it was given as a path."""
    return mesh if isinstance(mesh, meshio.Mesh) else meshio.read(str(mesh))


def _validated_samples(
    carriers: CarrierMapLike | Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Source points and the two concentration columns, length-checked."""
    try:
        x = np.asarray(carriers.x_um, dtype=np.float64).ravel()
        y = np.asarray(carriers.y_um, dtype=np.float64).ravel()
        electrons = np.asarray(carriers.electrons_cm3, dtype=np.float64).ravel()
        holes = np.asarray(carriers.holes_cm3, dtype=np.float64).ravel()
    except AttributeError as err:
        raise TypeError(
            "carriers must expose x_um, y_um, electrons_cm3 and holes_cm3 "
            "(e.g. a gsim.tcad.results.CarrierMap)."
        ) from err
    if not (x.size == y.size == electrons.size == holes.size):
        raise ValueError(
            "Carrier map coordinates and concentrations must have the same "
            f"length; got {x.size}, {y.size}, {electrons.size}, {holes.size}."
        )
    if x.size < 3:
        raise ValueError("At least three carrier samples are required.")
    return np.column_stack([x, y]), electrons, holes


def _fill_values(fill: Fill) -> tuple[float, float] | None:
    """Split the fill argument into (electrons, holes), or None for nearest."""
    if isinstance(fill, str):
        if fill != "nearest":
            raise ValueError(f"Unknown fill {fill!r}; use a number or 'nearest'.")
        return None
    if isinstance(fill, (int, float)):
        return float(fill), float(fill)
    electrons, holes = fill
    return float(electrons), float(holes)


def _target_points(
    mesh: meshio.Mesh, at: Granularity
) -> tuple[NDArray[np.float64], list[str]]:
    """Target coordinates and their region names, in mesh order."""
    points = np.asarray(mesh.points, dtype=np.float64)
    if at == "nodes":
        regions = node_regions(mesh)
        return np.asarray(points[:, :2], dtype=np.float64), regions
    if at != "elements":
        raise ValueError(f"Unknown granularity {at!r}; use 'nodes' or 'elements'.")
    tris, regions = element_regions(mesh)
    centroids = points[tris][:, :, :2].mean(axis=1)
    return np.asarray(centroids, dtype=np.float64), regions


def transfer_carriers(
    carriers: CarrierMapLike | Any,
    mesh: meshio.Mesh | str | Path,
    *,
    at: Granularity = "nodes",
    fill: Fill = 0.0,
    regions: Sequence[str] | None = None,
) -> TransferredCarriers:
    """Evaluate a Carrier map on another mesh.

    Args:
        carriers: The solved Carrier map (anything exposing ``x_um``,
            ``y_um``, ``electrons_cm3`` and ``holes_cm3``).
        mesh: Target mesh (msh v2.2 path or a loaded meshio mesh), in um.
        at: ``"nodes"`` for one value per mesh node, ``"elements"`` for
            one value per triangle, in the mesh's own order.
        fill: Value for target points the source domain does not cover:
            one number for both carriers, an ``(electrons, holes)`` pair,
            or ``"nearest"`` to extend the nearest solved sample.
        regions: Restrict carriers to these physical groups; points in any
            other group take ``fill``. Group names are resolved from the
            mesh, so callers never handle gmsh tags.

    Returns:
        The :class:`TransferredCarriers` sampled on the target mesh.

    Raises:
        ValueError: When the mesh has no triangles, a requested region is
            not on the mesh, or the Carrier map columns disagree in length.
    """
    source, electrons, holes = _validated_samples(carriers)
    target_mesh = _read_mesh(mesh)
    target, point_regions = _target_points(target_mesh, at)
    fill_pair = _fill_values(fill)

    # Both carriers share one hull and one triangulation; the fill is
    # applied below, over the mask this returns and the region
    # restriction together.
    sampled, missing = sample_at(
        source, np.column_stack([electrons, holes]), target, fill=np.nan
    )
    n_electrons = np.asarray(sampled[:, 0], dtype=np.float64)
    n_holes = np.asarray(sampled[:, 1], dtype=np.float64)

    if regions is not None:
        available = sorted(set(group_names(target_mesh, dim=2).values()))
        unknown = [name for name in regions if name not in available]
        if unknown:
            raise ValueError(
                f"Regions {unknown} are not physical groups on the mesh. "
                f"Available regions: {available}"
            )
        outside_regions = np.asarray(
            [name not in set(regions) for name in point_regions], dtype=np.bool_
        )
        missing = np.asarray(missing | outside_regions, dtype=np.bool_)

    if missing.any():
        if fill_pair is None:
            # Not sample_at again: a point the regions exclude can sit
            # well inside the hull, where the interpolant has an answer,
            # and the nearest fill has to override it rather than defer
            # to it.
            from scipy.interpolate import NearestNDInterpolator

            nearest = NearestNDInterpolator(
                source, np.column_stack([electrons, holes])
            )(target[missing])
            n_electrons[missing] = nearest[:, 0]
            n_holes[missing] = nearest[:, 1]
        else:
            n_electrons[missing] = fill_pair[0]
            n_holes[missing] = fill_pair[1]

    return TransferredCarriers(
        x_um=np.asarray(target[:, 0], dtype=np.float64),
        y_um=np.asarray(target[:, 1], dtype=np.float64),
        electrons_cm3=n_electrons,
        holes_cm3=n_holes,
        region=list(point_regions),
        filled=missing,
    )
