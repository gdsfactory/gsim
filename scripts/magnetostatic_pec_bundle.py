#!/usr/bin/env python3
"""Perfect-conductor filament references for ``nbs/palace_magnetostatic.ipynb``.

The notebook compares Palace against the numbers this script prints, for two
side-by-side inductor2 coils and for two concentric rings. The script uses no finite
elements and no Palace output. Each loop is a bundle of closed ribbons on the surface
of its 2 x 3 um trace, and the inductance of the bundle comes from Neumann's formula
(for the rings, its closed form by Maxwell):

- Partial inductances are closed form for parallel segments and Gauss-Legendre
  otherwise, in pH with lengths in um.
- A ribbon's self term is its mutual with a copy shifted by the ribbon's geometric
  mean distance, 0.2231 s. Ribbons on one face use the exact mean distance of two
  coplanar strips.
- A perfect conductor takes the current split of least magnetic energy at fixed loop
  currents, so every ribbon of a loop links the same flux, and the loop matrix is
  ``(C^T L^-1 C)^-1``, where ``C`` maps ribbons to loops.

The references are for open space. The notebook removes the outer wall from Palace's
side instead, with the mean of a PEC and a PMC run.

Run ``python scripts/magnetostatic_pec_bundle.py``; it takes a few minutes.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk

MU0_4PI = 0.1  # mu0/4pi in pH/um
WIDTH, THICKNESS, Z_MID = 2.0, 3.0, 12.53  # TopMetal2 trace, um
CENTRE_Y, APOTHEM, LEAD_X = 44.675, 13.675, 2.05  # inductor2, um
PITCH = 39.35  # um
GL_X, GL_W = np.polynomial.legendre.leggauss(4)


class Ribbon(NamedTuple):
    """One surface ribbon: its cross-section offset, face and width in um."""

    u: float
    v: float
    face: str
    width: float


def centreline() -> np.ndarray:
    """The open path from lead tip P1 round the octagon to lead tip P2, clockwise."""
    radius = APOTHEM / np.cos(np.pi / 8)
    angles = np.deg2rad(22.5 + 45 * np.array([5, 4, 3, 2, 1, 0, 7, 6]))
    octagon = np.c_[radius * np.cos(angles), CENTRE_Y + radius * np.sin(angles)]
    bottom = CENTRE_Y - APOTHEM
    return np.array(
        [(-LEAD_X, 0.0), (-LEAD_X, bottom), *octagon, (LEAD_X, bottom), (LEAD_X, 0.0)]
    )


def _offset(path: np.ndarray, u: float) -> np.ndarray:
    """Mitred shift of an open clockwise path by ``u`` towards the loop interior."""
    tangent = np.diff(path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.c_[tangent[:, 1], -tangent[:, 0]]
    out = np.empty_like(path)
    out[0] = path[0] + u * normal[0]
    out[-1] = path[-1] + u * normal[-1]
    for k in range(1, len(path) - 1):
        out[k] = path[k] + u * (normal[k - 1] + normal[k]) / (
            1 + normal[k - 1] @ normal[k]
        )
    return out


def _family(
    offsets: list[tuple[float, float]], x0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Segment starts and ends, ``(filament, segment, 3)``, of closed filaments."""
    starts, ends = [], []
    for u, v in offsets:
        path = _offset(centreline(), u)
        path = np.vstack([path, path[:1]])  # closed across the gap at y = 0
        points = np.c_[path[:, 0] + x0, path[:, 1], np.full(len(path), Z_MID + v)]
        starts.append(points[:-1])
        ends.append(points[1:])
    return np.array(starts), np.array(ends)


def _parallel_mutual(a0, a1, b0, b1) -> np.ndarray:
    """Mutual in pH of parallel segments, closed form, signed by orientation."""

    def primitive(x, d):
        return x * np.arcsinh(x / d) - np.sqrt(x * x + d * d)

    along = a1 - a0
    length = np.linalg.norm(along, axis=-1)
    unit = along / length[..., None]
    other = b1 - b0
    sign = np.sign(np.sum(other * unit, axis=-1))
    start = np.sum((b0 - a0) * unit, axis=-1)
    end = start + sign * np.linalg.norm(other, axis=-1)
    low, high = np.minimum(start, end), np.maximum(start, end)
    apart = (b0 - a0) - start[..., None] * unit
    d = np.maximum(np.linalg.norm(apart, axis=-1), 1e-7)
    return (
        MU0_4PI
        * sign
        * (
            primitive(length - low, d)
            - primitive(length - high, d)
            - primitive(-low, d)
            + primitive(-high, d)
        )
    )


def _graded_nodes(length: float, far: float, near: float) -> np.ndarray:
    """Piece boundaries on ``[0, length]``, graded towards both ends."""
    points = [0.0]
    x = near
    while x < min(length / 2, far):
        points.append(x)
        x *= 2
    left = np.array(points)
    pieces = (
        max(1, int(np.ceil((length - 2 * left[-1]) / far)))
        if length - 2 * left[-1] > 0
        else 0
    )
    middle = (
        np.linspace(left[-1], length - left[-1], pieces + 1)[1:-1]
        if pieces
        else np.array([])
    )
    return np.unique(np.r_[left, middle, (length - left)[::-1]])


def _quadrature(a0, a1, far: float, near: float | None):
    """Gauss-Legendre points and weighted tangents of one segment per filament."""
    length = np.linalg.norm(a1 - a0, axis=-1).max()
    nodes = (
        _graded_nodes(length, far, near) / length
        if near
        else np.linspace(0, 1, max(1, int(np.ceil(length / far))) + 1)
    )
    low, high = nodes[:-1], nodes[1:]
    t = (low[:, None] + (high - low)[:, None] * (GL_X + 1) / 2).ravel()
    weight = ((high - low)[:, None] * GL_W / 2).ravel()
    segment = a1 - a0
    points = a0[:, None, :] + t[None, :, None] * segment[:, None, :]
    return points, weight[None, :, None] * segment[:, None, :]


def _quadrature_mutual(a0, a1, b0, b1, far: float, near: float | None) -> np.ndarray:
    """Mutual in pH between two families of segments, by quadrature."""
    points_a, tangents_a = _quadrature(a0, a1, far, near)
    points_b, tangents_b = _quadrature(b0, b1, far, near)
    out = np.empty((len(a0), len(b0)))
    for i in range(len(a0)):
        apart = points_a[i][:, None, None, :] - points_b[None, :, :, :]
        distance = np.sqrt(np.sum(apart * apart, axis=-1))
        dots = np.einsum("pk,fqk->pfq", tangents_a[i], tangents_b)
        out[i] = np.sum(dots / distance, axis=(0, 2))
    return MU0_4PI * out


def _block(first, second, *, same_loop: bool, diagonal: bool = False) -> np.ndarray:
    """Partial inductances between two families of closed filaments."""
    a0, a1 = first
    b0, b1 = second
    segments_a, segments_b = a0.shape[1], b0.shape[1]
    out = np.zeros(len(a0)) if diagonal else np.zeros((len(a0), len(b0)))
    for k in range(segments_a):
        along_a = a1[0, k] - a0[0, k]
        along_a /= np.linalg.norm(along_a)
        for m in range(segments_b):
            along_b = b1[0, m] - b0[0, m]
            along_b /= np.linalg.norm(along_b)
            if np.linalg.norm(np.cross(along_a, along_b)) < 1e-9:
                if diagonal:
                    out += _parallel_mutual(a0[:, k], a1[:, k], b0[:, m], b1[:, m])
                else:
                    out += _parallel_mutual(
                        a0[:, None, k], a1[:, None, k], b0[None, :, m], b1[None, :, m]
                    )
                continue
            if abs(along_a @ along_b) < 1e-12:
                continue  # perpendicular segments do not couple
            adjacent = same_loop and abs(k - m) in (1, segments_a - 1)
            if adjacent:
                far, near = 0.5, 0.01
            else:
                far, near = (1.0 if same_loop else 2.0), None
            if diagonal:
                out += np.array(
                    [
                        _quadrature_mutual(
                            a0[i : i + 1, k],
                            a1[i : i + 1, k],
                            b0[i : i + 1, m],
                            b1[i : i + 1, m],
                            far,
                            near,
                        )[0, 0]
                        for i in range(len(a0))
                    ]
                )
            else:
                out += _quadrature_mutual(
                    a0[:, k], a1[:, k], b0[:, m], b1[:, m], far, near
                )
    return out


def _ribbons(s: float) -> list[Ribbon]:
    """Ribbons of width about ``s`` around the trace surface."""
    across, up = round(WIDTH / s), round(THICKNESS / s)
    width, height = WIDTH / across, THICKNESS / up
    out = []
    for k in range(across):
        u = -WIDTH / 2 + width * (k + 0.5)
        out += [
            Ribbon(u, -THICKNESS / 2, "bottom", width),
            Ribbon(u, THICKNESS / 2, "top", width),
        ]
    for k in range(up):
        v = -THICKNESS / 2 + height * (k + 0.5)
        out += [
            Ribbon(-WIDTH / 2, v, "outer", height),
            Ribbon(WIDTH / 2, v, "inner", height),
        ]
    return out


def _log_mean_distance(d: float, s: float) -> float:
    """Log of the geometric mean distance of two coplanar strips ``d`` apart."""

    def primitive(x):
        x = abs(x)
        return 0.5 * x * x * np.log(x) - 0.75 * x * x if x > 0 else 0.0

    return (primitive(d + s) - 2 * primitive(d) + primitive(d - s)) / (s * s)


def self_and_mutual(s: float) -> tuple[float, float]:
    """Self inductance of one coil and mutual of the pair, in pH.

    Args:
        s: Ribbon width, in um.

    Returns:
        ``(L, M)`` of two coils ``PITCH`` apart.
    """
    ribbons = _ribbons(s)
    offsets = [(ribbon.u, ribbon.v) for ribbon in ribbons]
    count = len(ribbons)
    coil_a, coil_b = _family(offsets), _family(offsets, PITCH)
    shifted = _family(
        [(r.u, r.v + np.exp(-1.5) * r.width) for r in ribbons],
    )
    own = _block(coil_a, coil_a, same_loop=True)
    np.fill_diagonal(own, _block(coil_a, shifted, same_loop=True, diagonal=True))
    lengths = np.linalg.norm(coil_a[1] - coil_a[0], axis=-1).sum(axis=1)
    for i, first in enumerate(ribbons):
        for j, second in enumerate(ribbons):
            if i != j and first.face == second.face:
                d = np.hypot(first.u - second.u, first.v - second.v)
                own[i, j] += (
                    MU0_4PI
                    * (lengths[i] + lengths[j])
                    * (np.log(d) - _log_mean_distance(d, first.width))
                )
    across = _block(coil_a, coil_b, same_loop=False)
    full = np.block([[own, across], [across.T, own]])
    loops = np.zeros((2 * count, 2))
    loops[:count, 0] = loops[count:, 1] = 1.0
    matrix = np.linalg.inv(loops.T @ np.linalg.solve(full, loops))
    return float(matrix[0, 0]), float(matrix[0, 1])


def references(coarse: float = 0.25, fine: float = 0.2) -> dict[str, float]:
    """The open-space references quoted by the notebook, in pH.

    The error of a ribbon solution is proportional to ``s``, so the values
    extrapolate two ribbon widths linearly to ``s = 0``.

    Args:
        coarse: The coarser ribbon width, in um.
        fine: The finer ribbon width, in um.

    Returns:
        ``M`` and ``L`` of two coils ``PITCH`` apart in open space.
    """
    l_coarse, m_coarse = self_and_mutual(coarse)
    l_fine, m_fine = self_and_mutual(fine)
    weight = fine / (coarse - fine)
    return {
        "M": m_fine + weight * (m_fine - m_coarse),
        "L": l_fine + weight * (l_fine - l_coarse),
    }


def equivalent_radius(width: float, thickness: float) -> float:
    """Radius of the round wire with the external inductance of a bar, in um.

    This is the logarithmic capacity of the rectangle, from its Schwarz-Christoffel
    map: 0.59017 of the side for a square, a quarter of the width for a thin strip.

    Args:
        width: One side of the bar, in um.
        thickness: The other side, in um.
    """

    def sides(theta: float) -> tuple[float, float]:
        cosine = np.cos(2 * theta)
        short = quad(lambda p: np.sqrt(2 * (np.cos(2 * p) - cosine)), -theta, theta)
        long = quad(
            lambda p: np.sqrt(2 * (cosine - np.cos(2 * p))), theta, np.pi - theta
        )
        return short[0], long[0]

    aspect = min(width, thickness) / max(width, thickness)
    theta = brentq(lambda t: sides(t)[0] / sides(t)[1] - aspect, 1e-6, np.pi / 2 - 1e-6)
    return max(width, thickness) / sides(theta)[1]


def maxwell_mutual(first: float, second: float, height: float = 0.0) -> float:
    """Maxwell's mutual inductance of two coaxial circles, in pH.

    Args:
        first: Radius of one circle, in um.
        second: Radius of the other, in um.
        height: Distance between their planes, in um.
    """
    m = 4 * first * second / ((first + second) ** 2 + height**2)
    k = np.sqrt(m)
    return float(
        4
        * np.pi
        * MU0_4PI
        * np.sqrt(first * second)
        * ((2 / k - k) * ellipk(m) - 2 / k * ellipe(m))
    )


def ring_bundle(inner: float, outer: float, s: float) -> np.ndarray:
    """Inductance matrix of two concentric rings of 2 x 3 um section, in pH.

    Args:
        inner: Radius of the first ring, in um.
        outer: Radius of the second ring, in um.
        s: Ribbon width, in um.

    Returns:
        The 2 x 2 matrix in open space, in the order inner, outer.
    """
    ribbons = _ribbons(s)
    placed = [
        (radius + r.u, r, loop)
        for loop, radius in enumerate((inner, outer))
        for r in ribbons
    ]
    count = len(placed)
    full = np.empty((count, count))
    for i, (radius_i, ribbon_i, loop_i) in enumerate(placed):
        for j, (radius_j, ribbon_j, loop_j) in enumerate(placed):
            if i == j:
                full[i, j] = maxwell_mutual(
                    radius_i, radius_i, np.exp(-1.5) * ribbon_i.width
                )
                continue
            height = ribbon_i.v - ribbon_j.v
            full[i, j] = maxwell_mutual(radius_i, radius_j, height)
            if loop_i == loop_j and ribbon_i.face == ribbon_j.face:
                d = np.hypot(radius_i - radius_j, height)
                full[i, j] += (
                    2
                    * np.pi
                    * MU0_4PI
                    * (radius_i + radius_j)
                    * (np.log(d) - _log_mean_distance(d, ribbon_i.width))
                )
    full = (full + full.T) / 2
    loops = np.zeros((count, 2))
    loops[: count // 2, 0] = loops[count // 2 :, 1] = 1.0
    return np.linalg.inv(loops.T @ np.linalg.solve(full, loops))


def ring_references(inner: float = 25.0, outer: float = 50.0) -> np.ndarray:
    """The rings' matrix in pH, extrapolated linearly in the ribbon width to zero.

    Args:
        inner: Radius of the first ring, in um.
        outer: Radius of the second ring, in um.
    """
    return 2 * ring_bundle(inner, outer, 0.0625) - ring_bundle(inner, outer, 0.125)


if __name__ == "__main__":
    rings = ring_references()
    print(
        f"rings 25 and 50 um: L = {rings[0, 0]:.3f} and {rings[1, 1]:.3f} pH, "
        f"M = {rings[0, 1]:.3f} pH"
    )
    for width in (1.0, 0.5, 0.25, 0.2):
        self_inductance, mutual_inductance = self_and_mutual(width)
        print(
            f"s = {width:<4} um: L = {self_inductance:.3f} pH, "
            f"M = {mutual_inductance:.5f} pH"
        )
    for name, value in references().items():
        print(f"{name:<6} {value:.4f} pH")
