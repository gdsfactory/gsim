"""Sampling a scattered cloud of values at other points.

Every transfer between two meshes in gsim is the same operation: a
solver leaves values at its own points, and something else — another
mesh's nodes, its triangle centroids, a band of sample columns — asks
what they are there. The linear interpolant over the cloud's Delaunay
triangulation answers inside its convex hull and returns NaN outside
it, and what to do about the outside is the caller's policy, not the
interpolator's.

:func:`sample_at` is that operation, and it always hands the missing
mask back. A caller with its own notion of missing — a region
restriction, a per-column fill, a band that is no rectangle — composes
with the mask instead of recomputing NaNs, and a caller with none
ignores it.

What it does not do is decide what the columns mean. ``fill`` is one
scalar or ``"nearest"``; a caller wanting a different value per column
applies it over the mask itself, which is already how a region
restriction is applied.

This module is imported by module path: it pulls in scipy, which
``import gsim.common`` does not, so nothing joins
``gsim.common.__all__``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["DegenerateSampleCloudError", "sample_at"]


class DegenerateSampleCloudError(ValueError):
    """The sample points span no area, so no interpolant exists.

    Raised rather than quietly falling back to nearest-neighbour: a
    cloud whose points are collinear, coincident or too few is a
    statement about whatever produced it, and a caller that can carry on
    without the sample says so by catching this.
    """


def sample_at(
    points: ArrayLike,
    values: ArrayLike,
    targets: ArrayLike,
    *,
    fill: complex | Literal["nearest"] = "nearest",
) -> tuple[NDArray[Any], NDArray[np.bool_]]:
    """Sample scattered values at other points, saying which were missed.

    Args:
        points: ``(n, 2)`` sample coordinates.
        values: ``(n,)`` or ``(n, k)`` values at those points, real or
            complex. Several columns share one hull and one
            triangulation, which is the whole reason to pass them
            together.
        targets: ``(m, 2)`` coordinates to sample at.
        fill: What a target outside the hull takes — one scalar, or
            ``"nearest"`` to extend the nearest sample.

    Returns:
        ``(sampled, missing)``: the values at the targets — real or
        complex, whichever ``values`` were — shaped like
        ``values`` per target, and the ``(m,)`` mask of targets the hull
        does not cover. The mask is about the hull alone, so it is the
        same for every column.

    Raises:
        DegenerateSampleCloudError: When the points span no area.
        ValueError: When the points and values disagree in length.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import QhullError

    pts = np.asarray(points, dtype=np.float64)
    vals = np.asarray(values)
    targ = np.asarray(targets, dtype=np.float64)
    if vals.shape[0] != pts.shape[0]:
        raise ValueError(
            "points and values must have the same length, got "
            f"{pts.shape[0]} and {vals.shape[0]}."
        )

    dtype = np.complex128 if np.iscomplexobj(vals) else np.float64
    try:
        linear = LinearNDInterpolator(pts, vals)
        sampled = np.asarray(linear(targ), dtype=dtype)
    except QhullError as err:
        raise DegenerateSampleCloudError(
            "The sample points span no area, so they have no linear "
            "interpolant: they are collinear, coincident, or too few."
        ) from err

    nan = np.isnan(sampled.real)
    missing = np.asarray(nan.any(axis=1) if nan.ndim > 1 else nan, dtype=np.bool_)
    if missing.any():
        if fill == "nearest":
            nearest = NearestNDInterpolator(pts, vals)
            sampled[missing] = np.asarray(nearest(targ[missing]), dtype=dtype)
        else:
            sampled[missing] = fill
    return sampled, missing
