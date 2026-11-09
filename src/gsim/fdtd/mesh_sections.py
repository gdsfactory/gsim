"""Geometric plane sections shared by material checks and setup plots."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def tetrahedron_section(vertices: Any, axis: int, position: float) -> Any:
    """Intersect a tetrahedron with a plane, retaining coplanar faces."""
    tolerance = 1e-9
    distances = vertices[:, axis] - position
    points = [
        point
        for point, distance in zip(vertices, distances, strict=True)
        if abs(distance) <= tolerance
    ]
    for start, end in combinations(range(4), 2):
        if distances[start] * distances[end] < -(tolerance**2):
            fraction = distances[start] / (distances[start] - distances[end])
            points.append(
                vertices[start] + fraction * (vertices[end] - vertices[start])
            )
    if len(points) < 3:
        return None
    plane_axes = [index for index in range(3) if index != axis]
    unique_points = np.unique(np.round(np.asarray(points)[:, plane_axes], 12), axis=0)
    if len(unique_points) < 3:
        return None
    centered = unique_points - unique_points.mean(axis=0)
    polygon = unique_points[np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))]
    area = (
        abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1))
        )
        / 2
    )
    return polygon if area > tolerance**2 else None
