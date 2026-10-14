"""Thermo-optic coupling: a temperature rise to an effective-index shift.

The optical step of a thermal phase-shifter solve, as a pure core:

- :func:`thermo_optic_permittivity` maps a per-element rise dT to the
  permittivity ``eps = (n0 + dn/dT * dT)^2`` (complex ``n0`` kept, so
  absorption is carried through unchanged);
- :func:`sample_temperature` reads the P1 rise of the thermal solve at
  arbitrary points of its own mesh -- exactly, by the P1 interpolant --
  and :func:`element_centroids` gives the points an optical Window asks
  for;
- :func:`solve_thermo_optic_shift` solves the optical Window's Mode with
  and without the perturbation on the **same** mesh (so the
  discretisation error cancels in the difference), tracks the perturbed
  Mode by its overlap with the unperturbed one, and reports the
  first-order perturbation estimate alongside as a cross-check.

Lengths are in um like the meshes. The sign convention is
``exp(+i omega t)``: lossy media have ``Im(n) < 0``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gsim.common.modes import propagating_modes
from gsim.femwell.adapter import boundary_field_ratio
from gsim.femwell.runtime import require_femwell, require_skfem

if TYPE_CHECKING:
    from gsim.thermal.mesh import CrossSectionMesh

__all__ = [
    "MAX_BOUNDARY_RATIO",
    "ModeShift",
    "element_centroids",
    "sample_temperature",
    "solve_thermo_optic_shift",
    "thermo_optic_permittivity",
]

MAX_BOUNDARY_RATIO: float = 1e-3
"""Peak field at the optical Window edge over the peak field, above which
the Window is too small for the Mode (:func:`solve_thermo_optic_shift`
warns). A 0.5 x 0.22 um Si strip in a 3 x 2.2 um oxide Window sits at
about 4e-4; halving the Window puts it near 0.1."""

# How far outside a triangle (in barycentric coordinates) a point may sit
# and still count as inside it: absorbs the round-off of a point on an
# edge or a Window corner, far below any element's size.
_BARYCENTRIC_TOL = 1e-9
# Nearest-centroid candidates tried per point before a full search.
_CANDIDATES = 8


def thermo_optic_permittivity(
    n0: ArrayLike, dn_dT: ArrayLike, delta_T: ArrayLike
) -> NDArray[np.complex128]:
    """Relative permittivity of a heated dielectric, per element.

    ``n = n0 + dn_dT * delta_T`` and ``eps = n^2``. Only the real part of
    the index moves: the imaginary part of a complex ``n0`` is kept as it
    is, since v1 has no thermo-optic absorption.

    Args:
        n0: Unperturbed (complex) refractive index of every element.
        dn_dT: Thermo-optic coefficient of every element (1/K).
        delta_T: Temperature rise of every element (K).

    Returns:
        The complex relative permittivity of every element.

    Raises:
        ValueError: When the three arrays do not have the same shape.
    """
    n0_arr = np.asarray(n0, dtype=np.complex128)
    dn_arr = np.asarray(dn_dT, dtype=np.float64)
    dt_arr = np.asarray(delta_T, dtype=np.float64)
    if not n0_arr.shape == dn_arr.shape == dt_arr.shape:
        raise ValueError(
            "n0, dn_dT and delta_T must have one value per element each, got "
            f"shapes {n0_arr.shape}, {dn_arr.shape} and {dt_arr.shape}."
        )
    n = n0_arr + dn_arr * dt_arr
    return np.asarray(n * n, dtype=np.complex128)


def element_centroids(mesh: CrossSectionMesh) -> NDArray[np.float64]:
    """Centroid of every triangle of a Cross-section mesh.

    Args:
        mesh: The meshed Cross-section (an optical Window, typically).

    Returns:
        ``(2, n_elements)`` centroids in um, in element order.
    """
    m = mesh.mesh
    return np.asarray(m.p[:, m.t].mean(axis=1), dtype=np.float64)


def sample_temperature(
    thermal_mesh: CrossSectionMesh,
    nodal_rise: ArrayLike,
    points: ArrayLike,
) -> NDArray[np.float64]:
    """Temperature rise of a thermal solve at arbitrary points of its mesh.

    The value is the P1 interpolant the thermal solve is made of, read
    exactly: the containing triangle's barycentric weights on its three
    nodal values. Nothing is extrapolated -- the thermal Window must
    contain every point asked for.

    Args:
        thermal_mesh: The mesh the thermal solve ran on.
        nodal_rise: Temperature rise at every mesh node (K).
        points: ``(2, N)`` coordinates in um, in the mesh's
            ``(in-plane, z)`` axes.

    Returns:
        The rise at every point (K).

    Raises:
        ValueError: When ``nodal_rise`` does not have one value per node,
            ``points`` is not ``(2, N)``, or a point lies outside the
            thermal mesh (the first such point is named).
    """
    m = thermal_mesh.mesh
    rise = np.asarray(nodal_rise, dtype=np.float64)
    if rise.shape != (m.p.shape[1],):
        raise ValueError(
            f"nodal_rise must have one value per mesh node ({m.p.shape[1]}), "
            f"got shape {rise.shape}."
        )
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] != 2:
        raise ValueError(f"points must have shape (2, N), got {pts.shape}.")

    elements, weights = _locate(m.p, m.t, pts)
    missing = np.flatnonzero(elements < 0)
    if missing.size:
        x, z = pts[:, missing[0]]
        raise ValueError(
            f"Point ({x:g}, {z:g}) um lies outside the thermal mesh "
            f"(window {thermal_mesh.window}, z_range {thermal_mesh.z_range}); "
            f"{missing.size} of {pts.shape[1]} points are outside. The "
            "thermal Window must contain every optical Window."
        )
    nodes = m.t[:, elements]
    return np.asarray((weights * rise[nodes]).sum(axis=0), dtype=np.float64)


def _barycentric(
    p: NDArray[np.float64], t: NDArray[np.int64], elements: NDArray, pts: NDArray
) -> NDArray[np.float64]:
    """Barycentric coordinates ``(3, ...)`` of ``pts`` in ``elements``.

    ``elements`` and ``pts[0]`` broadcast against each other, so one point
    can be tested against many triangles or many points against theirs.
    """
    a, b, c = (p[:, t[i, elements]] for i in range(3))
    det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    l1 = ((pts[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (pts[1] - a[1])) / det
    l2 = ((b[0] - a[0]) * (pts[1] - a[1]) - (pts[0] - a[0]) * (b[1] - a[1])) / det
    return np.stack([1.0 - l1 - l2, l1, l2])


def _locate(
    p: NDArray[np.float64], t: NDArray[np.int64], pts: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Containing triangle and barycentric weights of every point.

    Tries the triangles with the nearest centroids first, then every
    triangle for the points those miss. Returns ``-1`` (and zero weights)
    for a point no triangle contains.
    """
    from scipy.spatial import cKDTree

    n_points = pts.shape[1]
    n_elements = t.shape[1]
    found = np.full(n_points, -1, dtype=np.int64)
    weights = np.zeros((3, n_points), dtype=np.float64)

    centroids = p[:, t].mean(axis=1)
    k = min(_CANDIDATES, n_elements)
    _dist, cand = cKDTree(centroids.T).query(pts.T, k=k)
    cand = np.asarray(cand, dtype=np.int64).reshape(n_points, k).T  # (k, N)
    lam = _barycentric(p, t, cand, pts[:, None, :])  # (3, k, N)
    inside = (lam >= -_BARYCENTRIC_TOL).all(axis=0)  # (k, N)
    hit = inside.any(axis=0)
    first = inside.argmax(axis=0)
    cols = np.flatnonzero(hit)
    found[cols] = cand[first[cols], cols]
    weights[:, cols] = lam[:, first[cols], cols]

    every = np.arange(n_elements)
    for i in np.flatnonzero(~hit):
        lam_i = _barycentric(p, t, every, pts[:, i])
        inside_i = np.flatnonzero((lam_i >= -_BARYCENTRIC_TOL).all(axis=0))
        if inside_i.size:
            found[i] = inside_i[0]
            weights[:, i] = lam_i[:, inside_i[0]]
    return found, weights


@dataclass(frozen=True)
class ModeShift:
    """The effective-index shift a temperature rise gives one optical Mode.

    Attributes:
        neff0: Effective index of the unperturbed Mode.
        neff: Effective index of the perturbed Mode, re-solved on the same
            mesh and tracked by overlap with the unperturbed one.
        delta_neff_first_order: First-order perturbation estimate of the
            shift (overlap of the permittivity change with the
            unperturbed field) -- the cross-check of the re-solve, and the
            cheap path for transients.
        mode0: The unperturbed femwell ``Mode``.
        mode: The perturbed femwell ``Mode``.
        overlap: ``|<mode0|mode>|`` of the power-normalised Modes; near
            one when the tracking is unambiguous.
        boundary_ratio: Peak field at the optical Window edge over the
            peak field, of ``mode0``
            (:func:`gsim.femwell.adapter.boundary_field_ratio`).
    """

    neff0: complex
    neff: complex
    delta_neff_first_order: complex
    mode0: Any
    mode: Any
    overlap: float
    boundary_ratio: float

    @property
    def delta_neff(self) -> complex:
        """Effective-index shift from the re-solve, ``neff - neff0``."""
        return self.neff - self.neff0


def solve_thermo_optic_shift(
    optical_mesh: CrossSectionMesh,
    epsilon0: ArrayLike,
    dn_dT: ArrayLike,
    delta_T: ArrayLike,
    *,
    wavelength_um: float,
    polarization: Literal["TE", "TM"] | None = "TE",
    mode_order: int = 0,
    num_modes: int = 4,
    n_guess: float | None = None,
) -> ModeShift:
    """Effective-index shift of one Mode under a temperature rise.

    The Mode is solved twice on the same optical mesh, with ``epsilon0``
    and with :func:`thermo_optic_permittivity` of ``sqrt(epsilon0)``, so
    the discretisation error cancels in the difference. The perturbed
    Mode is the one overlapping the unperturbed Mode most; its solve is
    seeded at the unperturbed index, which the small perturbation
    barely moves.

    Args:
        optical_mesh: The optical Window (from
            :func:`gsim.thermal.mesh.build_cross_section_mesh`).
        epsilon0: Unperturbed complex relative permittivity per element
            (``exp(+i omega t)``: lossy means ``Im < 0``).
        dn_dT: Thermo-optic coefficient per element (1/K).
        delta_T: Temperature rise per element (K), e.g.
            :func:`sample_temperature` at :func:`element_centroids`.
        wavelength_um: Vacuum wavelength in um.
        polarization: ``"TE"`` keeps Modes whose transverse field is
            mostly along the in-plane axis (``Ex``, femwell's
            ``te_fraction > 0.5``), ``"TM"`` the others, ``None`` all.
        mode_order: Which Mode of that polarization, by descending
            ``Re(neff)``: 0 is the fundamental.
        num_modes: How many Modes each solve computes to choose from.
        n_guess: Effective-index guess the unperturbed solve looks for
            Modes around, e.g. the waveguide core's index; ``None`` uses
            femwell's default (just above the largest index in the
            Window).

    Returns:
        The two indices, the first-order estimate and both Modes.

    Raises:
        ValueError: When a per-element array does not have one value per
            element, or no Mode of the requested polarization and order
            was found among ``num_modes``.
        ImportError: When femwell (the ``femwell`` extra) is missing.
    """
    n_elements = optical_mesh.mesh.t.shape[1]
    eps0 = _per_element("epsilon0", epsilon0, n_elements, np.complex128)
    dn = _per_element("dn_dT", dn_dT, n_elements, np.float64)
    rise = _per_element("delta_T", delta_T, n_elements, np.float64)
    eps = thermo_optic_permittivity(np.sqrt(eps0), dn, rise)

    require_femwell()
    skfem = require_skfem()
    from femwell.maxwell.waveguide import compute_modes

    # One loader path: the optical mesh as built, so element i of every
    # per-element array is element i of the basis.
    basis0 = skfem.Basis(optical_mesh.mesh, skfem.ElementTriP0())
    modes0 = compute_modes(
        basis0, eps0, wavelength=wavelength_um, num_modes=num_modes, n_guess=n_guess
    )
    mode0 = _select(modes0, polarization, mode_order)
    modes = compute_modes(
        basis0,
        eps,
        wavelength=wavelength_um,
        num_modes=num_modes,
        n_guess=complex(mode0.n_eff).real,
    )
    overlaps = [abs(complex(mode0.calculate_overlap(m))) for m in modes]
    best = int(np.argmax(overlaps))
    mode = modes[best]

    ratio = boundary_field_ratio(mode0)
    if ratio > MAX_BOUNDARY_RATIO:
        warnings.warn(
            f"The Mode's field at the optical Window edge is {ratio:.2g} of "
            f"its peak (above {MAX_BOUNDARY_RATIO:g}): the optical Window "
            f"(window {optical_mesh.window}, z_range {optical_mesh.z_range}) "
            "is too small for the Mode and truncates it. Enlarge it.",
            UserWarning,
            stacklevel=2,
        )

    neff0 = complex(mode0.n_eff)
    return ModeShift(
        neff0=neff0,
        neff=complex(mode.n_eff),
        delta_neff_first_order=complex(mode0.calculate_pertubated_neff(eps - eps0))
        - neff0,
        mode0=mode0,
        mode=mode,
        overlap=overlaps[best],
        boundary_ratio=ratio,
    )


def _per_element(name: str, values: ArrayLike, n: int, dtype: Any) -> NDArray[Any]:
    """One value per element of the optical mesh, or a ValueError."""
    arr = np.asarray(values, dtype=dtype)
    if arr.shape != (n,):
        raise ValueError(
            f"{name} must have one value per element of the optical mesh "
            f"({n}), got shape {arr.shape}."
        )
    return arr


def _select(
    modes: Any, polarization: Literal["TE", "TM"] | None, mode_order: int
) -> Any:
    """The ``mode_order``-th propagating Mode of one polarization."""
    candidates = sorted(
        propagating_modes(list(modes)), key=lambda m: -complex(m.n_eff).real
    )
    if polarization == "TE":
        candidates = [m for m in candidates if m.te_fraction > 0.5]
    elif polarization == "TM":
        candidates = [m for m in candidates if m.te_fraction <= 0.5]
    if mode_order >= len(candidates):
        found = ", ".join(
            f"{complex(m.n_eff).real:.4f} (TE fraction {m.te_fraction:.2f})"
            for m in modes
        )
        raise ValueError(
            f"No {polarization or 'propagating'} Mode of order {mode_order} "
            f"among the {len(modes)} solved: {found}. Raise num_modes."
        )
    return candidates[mode_order]
