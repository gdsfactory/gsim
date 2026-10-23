"""The fields of a solved Palace boundary Mode, read back off disk.

Palace's ``BoundaryMode`` text results carry effective indices and
nothing else, so everything a line parameter needs beyond the
propagation constant — the power the Mode carries, the current on its
signal conductor, and therefore its characteristic impedance — used to be
a femwell-Route capability. It need not be: asked to save its Modes,
Palace writes each one's transverse and longitudinal fields to ParaView,
and the Marks-Williams integrals are the same integrals on either
solver's fields.

The fields arrive as nodal values on second-order Lagrange triangles
covering the meshed Cross-section, in um coordinates and SI units. The
integrals here therefore land in the same scale the femwell adapter's
do, and both Routes divide them through the one definition
:func:`gsim.common.modes.z0_power_current` holds, which is what lets the
two be compared as numbers rather than as pictures.

Nothing has to be undone on the way in. Palace writes the two transverse
components of a boundary Mode in the same order as the coordinates of
the points it writes them at, and an earlier version of this reader
swapped them: on the fields it was diagnosed against, the electric field
came out tangential to a perfect conductor and the magnetic field normal
to it, which is the wrong way round on both counts. Those fields came
from meshes whose electrode outlines had lost their perfect-conductor
groups (the native-2D mesher read gmsh's ``getAdjacencies`` result
backwards), so what looked like a swapped write was a faithful read of a
wrongly conditioned solve. On a correctly conditioned one the components
as written put the boundary conditions the right way round on faces of
either orientation.

The sign convention is ``exp(+i omega t)``, matching the rest of gsim.

:class:`BoundaryModeField` and :func:`load_boundary_mode_field` are
re-exported from :mod:`gsim.palace`; the integrals are not, because
they share a unit contract — um coordinates against SI fields, which
cancels only in the ratio :func:`z0_power_current` takes — and are meant
to be read together with it. The terminal-port helpers
:func:`voltage_line_integral` and :func:`reaction_matrix` follow the same
contract: a voltage comes back in V/m times um and a reaction in
(V/m)(A/m) times um^2, so that ``V I`` and ``2 P`` are in one scale.

:func:`check_field_is_the_mode` sits here rather than beside whoever
reads a Mode back, because asking a saved field whether it is the Mode
it was fetched for is a statement about the field alone: it is the same
question for a boundary Mode's impedance and for any later eigen-field
read.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import epsilon_0 as EPS0  # noqa: N812
from scipy.constants import mu_0 as MU0  # noqa: N812

from gsim.common.modes import z0_power_current as common_z0

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "DEFAULT_PERIMETER_TOL_UM",
    "FIELD_INDEX_RTOL",
    "VOLTAGE_NSAMPLES",
    "BoundaryModeField",
    "check_field_is_the_mode",
    "contour_current",
    "field_index_ratio",
    "load_boundary_mode_field",
    "load_port_mode_fields",
    "port_mode_fields_from_grid",
    "power_flux",
    "reaction_matrix",
    "voltage_line_integral",
    "z0_power_current",
]

#: How far a mesh edge may sit off a conductor's perimeter and still
#: count as on it (um). Palace writes its ParaView coordinates in single
#: precision, which rounds a coordinate tens of um from the origin by
#: about a nanometre; this is loose enough to absorb that and far tighter
#: than any drawn feature.
DEFAULT_PERIMETER_TOL_UM: float = 1e-4

#: Default sample points per path segment of :func:`voltage_line_integral`.
#: On a path that runs a few thousandths of a gap from a metal corner, as the
#: detour paths of a multi-terminal port do, the integral is not converged at
#: 200 (0.9 % in a voltage of a three-terminal port, 3.5e-3 in a terminal S at
#: 50 GHz) and is converged to well under 1e-3 at 800.
VOLTAGE_NSAMPLES: int = 800

#: How far the points of a saved 3D port face may differ in the coordinate
#: normal to it (um); Palace writes single-precision coordinates.
_FACE_FLAT_TOL_UM: float = 1e-3

#: Barycentric points and weights of a triangle rule exact to degree 4,
#: which is what the product of two second-order fields asks for.
_QUADRATURE: tuple[tuple[float, float, float, float], ...] = (
    (0.445948490915965, 0.445948490915965, 0.108103018168070, 0.223381589678011),
    (0.445948490915965, 0.108103018168070, 0.445948490915965, 0.223381589678011),
    (0.108103018168070, 0.445948490915965, 0.445948490915965, 0.223381589678011),
    (0.091576213509771, 0.091576213509771, 0.816847572980459, 0.109951743655322),
    (0.091576213509771, 0.816847572980459, 0.091576213509771, 0.109951743655322),
    (0.816847572980459, 0.091576213509771, 0.091576213509771, 0.109951743655322),
)


@dataclass(frozen=True)
class BoundaryModeField:
    """One saved Palace boundary Mode, as fields on its own mesh.

    Attributes:
        points_um: ``(N, 2)`` node coordinates of the Cross-section (um).
        cells: ``(M, 6)`` second-order Lagrange triangles — three corner
            nodes, then the midside node of each edge in turn.
        attribute: ``(M,)`` Palace domain attribute of each triangle,
            which is the mesh Region it belongs to.
        e_t: ``(N, 2)`` transverse electric field (V/m).
        e_n: ``(N,)`` longitudinal electric field (V/m).
        h_t: ``(N, 2)`` transverse magnetic field (A/m).
        h_n: ``(N,)`` longitudinal magnetic field (A/m), or ``None``.
            No integral here reads it, and which name Palace writes it
            under has moved between versions, so a Mode saved without it
            is still a Mode this can integrate.
    """

    points_um: NDArray[np.float64]
    cells: NDArray[np.int64]
    attribute: NDArray[np.int64]
    e_t: NDArray[np.complex128]
    e_n: NDArray[np.complex128]
    h_t: NDArray[np.complex128]
    h_n: NDArray[np.complex128] | None


def _complex_array(grid: Any, name: str) -> NDArray[np.complex128]:
    """Join Palace's ``<name>_real`` / ``<name>_imag`` pair into one array."""
    missing = [
        key for key in (f"{name}_real", f"{name}_imag") if key not in grid.point_data
    ]
    if missing:
        raise ValueError(
            f"The saved mode carries no {', '.join(missing)}; its point data is "
            f"{sorted(grid.point_data)}. Palace writes mode fields only when the "
            "boundary-mode block asks it to save them (set_boundary_mode(save=...))."
        )
    real = np.asarray(grid.point_data[f"{name}_real"], dtype=np.float64)
    imag = np.asarray(grid.point_data[f"{name}_imag"], dtype=np.float64)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def _transverse(grid: Any, name: str) -> NDArray[np.complex128]:
    """One transverse field, in the order of the points it sits on.

    Palace writes a boundary Mode's two transverse components in the
    same order as the coordinates of its points. An earlier version of
    this reader swapped them, compensating for what looked like a
    swapped write — but the fields it was diagnosed on came from meshes
    whose electrode outlines had lost their perfect-conductor groups
    (the ``getAdjacencies`` result was read backwards), so the "wrong
    way round" fields were a correct read of a wrongly conditioned
    solve. On a correctly conditioned solve the components as written
    put the electric field normal to a perfect conductor and the
    magnetic field along it, which is the right way round on both
    counts.
    """
    values = _complex_array(grid, name)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            f"The saved mode's {name} field has shape {values.shape}, not the "
            "two transverse components of a boundary mode."
        )
    return np.asarray(values, dtype=np.complex128)


def load_boundary_mode_field(
    source: str | Path | dict[str, Any], *, mode_id: int
) -> BoundaryModeField:
    """Read one saved Mode of a ``BoundaryMode`` solve back off disk.

    Palace writes one ParaView cycle per saved Mode, in the solver's own
    mode order, so Mode ``m`` is cycle ``m``. Reading a Mode the solve
    did not save is an error rather than a silent fallback to another
    one.

    Args:
        source: What ``run_local`` returned, or the simulation
            directory.
        mode_id: Palace's own mode number, 1-based.

    Returns:
        The Mode's fields on the meshed Cross-section.

    Raises:
        FileNotFoundError: When the solve saved no fields at all.
        ValueError: When the requested Mode was not among the saved
            ones, or the mesh is not the second-order triangulation a
            boundary-mode solve writes.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy

    from gsim.palace.results import load_fields

    grid = load_fields(source, cycle=int(mode_id))
    cell_types = np.unique(np.asarray(grid.celltypes))
    # 22 is VTK_QUADRATIC_TRIANGLE, 69 its Lagrange spelling; both carry
    # three corner nodes then three midside ones. The Lagrange spelling
    # has no fixed node count of its own, so the connectivity is read
    # through the offsets rather than through pyvista's cells_dict — and
    # off VTK's own offsets array, since pyvista's ``offset`` is deprecated.
    offsets = vtk_to_numpy(grid.GetCells().GetOffsetsArray())
    counts = np.diff(np.asarray(offsets, dtype=np.int64))
    if cell_types.size != 1 or int(cell_types[0]) not in (22, 69):
        raise ValueError(
            f"The saved mode's mesh has cell types {cell_types.tolist()}, not the "
            "second-order triangles a boundary-mode solve writes."
        )
    if counts.size == 0 or not np.all(counts == 6):
        raise ValueError(
            "The saved mode's triangles carry "
            f"{sorted(set(counts.tolist()))} nodes each, not the six a "
            "second-order triangle carries."
        )
    cells = np.asarray(grid.cell_connectivity, dtype=np.int64).reshape(-1, 6)

    # The longitudinal magnetic field is not read by any integral here,
    # and Palace has spelled it both "B" and "Bn"; take whichever is
    # there and do not fail the whole read for a field nothing wants.
    b_n = next(
        (
            _complex_array(grid, name)
            for name in ("B", "Bn")
            if f"{name}_real" in grid.point_data
        ),
        None,
    )
    return BoundaryModeField(
        points_um=np.asarray(grid.points[:, :2], dtype=np.float64),
        cells=cells,
        attribute=np.asarray(grid.cell_data["attribute"], dtype=np.int64),
        e_t=_transverse(grid, "E"),
        e_n=_complex_array(grid, "En"),
        h_t=np.asarray(_transverse(grid, "Bt") / MU0, dtype=np.complex128),
        h_n=None if b_n is None else np.asarray(b_n / MU0, dtype=np.complex128),
    )


def port_mode_fields_from_grid(
    grid: Any, *, indices: Sequence[int], attributes: Sequence[int]
) -> list[BoundaryModeField]:
    """Take the saved port-mode fields of one wave-port face off a boundary dataset.

    The boundary ParaView output of a driven run carries, for every
    ``WavePort`` entry, the unit-power field of the port mode that entry's
    eigenproblem found: ``E0_<index>_real`` and ``E0_<index>_imag``, 3-vectors
    in V/m at the nodes of the boundary mesh (which is in um), where
    ``<index>`` is the entry's Palace ``Index``. The face is a plane of the 3D
    mesh; its cells are the boundary cells of the given attributes.

    The result is a :class:`BoundaryModeField` in the face's own ``(h, v)``
    coordinates, the ones of the matching 2D cross-section (``y, z`` for a face
    normal to x, ``x, z`` for y, ``x, y`` for z): ``e_t`` is the in-plane part
    of ``E0``, ``e_n`` the part along the normal. The magnetic field is not
    saved, so ``h_t`` is zero and ``h_n`` is ``None``: the field is good for
    :func:`voltage_line_integral`, not for the power or current integrals.

    Args:
        grid: The boundary dataset (what ``load_fields(..., boundary=True)``
            returns).
        indices: Palace ``Index`` of each entry wanted, one field per index,
            in this order.
        attributes: Boundary attributes of the port face.

    Returns:
        One field per index.

    Raises:
        ValueError: When no cell carries the attributes, the face is not made
            of second-order triangles or does not lie in a coordinate plane, or
            an entry has no ``E0`` field in the dataset (the run did not save
            fields at that frequency, or the index is not an entry of the face).
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy

    wanted = sorted({int(a) for a in attributes})
    if not wanted:
        raise ValueError("No boundary attribute was given for the port face.")
    if "attribute" not in grid.cell_data:
        raise ValueError("The saved boundary dataset carries no 'attribute' cell data.")
    labels = np.asarray(grid.cell_data["attribute"])
    on_face = np.flatnonzero(np.isin(labels, wanted))
    if on_face.size == 0:
        present = sorted({int(a) for a in np.unique(labels)})
        raise ValueError(
            f"The saved boundary has no cell on the attributes {wanted}; it has "
            f"{present}."
        )
    face = grid.extract_cells(on_face)
    cell_types = {int(t) for t in np.unique(np.asarray(face.celltypes))}
    offsets = vtk_to_numpy(face.GetCells().GetOffsetsArray())
    counts = np.diff(np.asarray(offsets, dtype=np.int64))
    if not cell_types <= {22, 69} or counts.size == 0 or not np.all(counts == 6):
        raise ValueError(
            f"The port face has cell types {sorted(cell_types)} with "
            f"{sorted(set(counts.tolist()))} nodes each, not the second-order "
            "triangles a driven run writes."
        )
    cells = np.asarray(face.cell_connectivity, dtype=np.int64).reshape(-1, 6)
    points = np.asarray(face.points, dtype=np.float64)
    flat = np.flatnonzero(np.ptp(points, axis=0) < _FACE_FLAT_TOL_UM)
    if flat.size != 1:
        raise ValueError(
            "The port face does not lie in one coordinate plane (extent per axis "
            f"{np.ptp(points, axis=0).tolist()} um)."
        )
    normal = int(flat[0])
    plane = [axis for axis in range(3) if axis != normal]
    labels_face = np.asarray(face.cell_data["attribute"], dtype=np.int64)

    fields: list[BoundaryModeField] = []
    for index in indices:
        name = f"E0_{int(index)}"
        missing = [
            k for k in (f"{name}_real", f"{name}_imag") if k not in face.point_data
        ]
        if missing:
            raise ValueError(
                f"The saved boundary carries no {', '.join(missing)}; its point "
                f"data is {sorted(face.point_data)}. The port-mode fields are "
                "written for the frequencies of set_driven(save_fields_at=...)."
            )
        e = _complex_array(face, name)
        if e.ndim != 2 or e.shape[1] != 3:
            raise ValueError(f"{name} has shape {e.shape}, not (N, 3).")
        fields.append(
            BoundaryModeField(
                points_um=points[:, plane],
                cells=cells,
                attribute=labels_face,
                e_t=np.asarray(e[:, plane], dtype=np.complex128),
                e_n=np.asarray(e[:, normal], dtype=np.complex128),
                h_t=np.zeros((points.shape[0], 2), dtype=np.complex128),
                h_n=None,
            )
        )
    return fields


def load_port_mode_fields(
    source: str | Path | dict[str, Any],
    *,
    indices: Sequence[int],
    attributes: Sequence[int],
    cycle: int,
    excitation: int = 1,
) -> list[BoundaryModeField]:
    """Read the saved port-mode fields of one wave-port face of a driven run.

    Palace writes one ParaView cycle per saved frequency
    (``set_driven(save_fields_at=...)``), in ascending order: cycle ``1`` is the
    lowest saved frequency. Every excitation's boundary output carries the
    port-mode fields ``E0_<index>`` of the entries; the fields do not depend on
    which entry is excited, so one excitation is enough. See
    :func:`port_mode_fields_from_grid` for what is returned.

    Args:
        source: What ``run_local`` or the cloud returned, or the simulation
            directory.
        indices: Palace ``Index`` of each entry wanted.
        attributes: Boundary attributes of the port face (the sidecar
            ``port_modes.json`` lists them per entry).
        cycle: ParaView cycle number, 1-based.
        excitation: Excitation whose output is read (1-based).

    Returns:
        One field per index, in the order of ``indices``.

    Raises:
        FileNotFoundError: When the run saved no boundary fields at that cycle.
        ValueError: As :func:`port_mode_fields_from_grid`.
    """
    from gsim.palace.results import load_fields

    grid = load_fields(source, excitation=excitation, cycle=int(cycle), boundary=True)
    return port_mode_fields_from_grid(grid, indices=indices, attributes=attributes)


def _triangle_areas(field: BoundaryModeField) -> NDArray[np.float64]:
    """Signed-free area of each triangle from its corner nodes (um^2)."""
    corners = field.points_um[field.cells[:, :3]]
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 2] - corners[:, 0]
    cross = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    return np.asarray(0.5 * np.abs(cross), dtype=np.float64)


def _shape_values(l0: float, l1: float, l2: float) -> NDArray[np.float64]:
    """The six second-order shape functions at one barycentric point."""
    return np.asarray(
        [
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        ],
        dtype=np.float64,
    )


def power_flux(field: BoundaryModeField) -> complex:
    """Complex power the Mode carries along the propagation direction.

    ``P = (1/2) integral (E_t x H_t*) . z dA`` over the whole meshed
    Cross-section — the numerator of the Marks-Williams power-current
    impedance, and the same integral the femwell Route assembles.

    Args:
        field: The Mode's saved fields.

    Returns:
        The complex power, in the um-area scale
        :func:`z0_power_current` divides out.
    """
    areas = _triangle_areas(field)
    e_t = field.e_t[field.cells]
    h_t = np.conj(field.h_t[field.cells])
    total = np.zeros(field.cells.shape[0], dtype=np.complex128)
    for l0, l1, l2, weight in _QUADRATURE:
        shape = _shape_values(l0, l1, l2)
        e_q = np.einsum("n,cnk->ck", shape, e_t)
        h_q = np.einsum("n,cnk->ck", shape, h_t)
        total += weight * (e_q[:, 0] * h_q[:, 1] - e_q[:, 1] * h_q[:, 0])
    return complex(0.5 * np.sum(total * areas))


def _shape_values_at(bary: NDArray[np.float64]) -> NDArray[np.float64]:
    """The six second-order shape functions at many barycentric points.

    Args:
        bary: ``(P, 3)`` barycentric coordinates.

    Returns:
        ``(P, 6)`` shape-function values, in the node order of a
        second-order triangle (the layout :func:`_shape_values` uses).
    """
    l0, l1, l2 = bary[:, 0], bary[:, 1], bary[:, 2]
    return np.asarray(
        np.stack(
            [
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l2 * l0,
            ],
            axis=1,
        ),
        dtype=np.float64,
    )


def _locate_points(
    field: BoundaryModeField, points: NDArray[np.float64], tol_um: float
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Find the triangle each point sits in, and its barycentric coordinates.

    A point counts as inside a triangle when it is no further than
    ``tol_um`` outside its worst edge, and where several triangles
    qualify (a point on a shared edge) the one it is deepest inside wins.
    The triangles are taken as straight-sided on their corner nodes,
    which is how :func:`power_flux` reads them.

    Args:
        field: The Mode's saved fields.
        points: ``(P, 2)`` points (um).
        tol_um: How far outside the mesh a point may sit and still be
            located (um).

    Returns:
        ``(cell, bary)``: the triangle index of each point and its
        ``(P, 3)`` barycentric coordinates in that triangle.

    Raises:
        ValueError: When a point is further than ``tol_um`` outside every
            triangle, or the mesh has no triangle with an area.
    """
    corners = field.points_um[field.cells[:, :3]]
    origin = corners[:, 0]
    edge_a = corners[:, 1] - origin
    edge_b = corners[:, 2] - origin
    det = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    usable = np.flatnonzero(det != 0.0)
    if usable.size == 0:
        raise ValueError("The saved mode's mesh has no triangle with an area.")
    corners, origin = corners[usable], origin[usable]
    edge_a, edge_b, det = edge_a[usable], edge_b[usable], det[usable]
    # Length of the edge opposite each corner; twice the area over it is
    # the altitude, which turns a barycentric coordinate into a distance.
    opposite = np.stack(
        [
            np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1),
            np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1),
            np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1),
        ],
        axis=1,
    )
    altitude = np.abs(det)[:, None] / opposite

    count = points.shape[0]
    found = np.empty(count, dtype=np.int64)
    bary = np.empty((count, 3), dtype=np.float64)
    chunk = max(1, 1_000_000 // usable.size)
    for start in range(0, count, chunk):
        offset = points[start : start + chunk, None, :] - origin[None]
        lam1 = (
            offset[..., 0] * edge_b[None, :, 1] - offset[..., 1] * edge_b[None, :, 0]
        ) / det
        lam2 = (
            edge_a[None, :, 0] * offset[..., 1] - edge_a[None, :, 1] * offset[..., 0]
        ) / det
        lam = np.stack([1.0 - lam1 - lam2, lam1, lam2], axis=-1)
        depth = np.min(lam * altitude[None], axis=-1)
        best = np.argmax(depth, axis=1)
        rows = np.arange(best.size)
        outside = depth[rows, best] < -tol_um
        if np.any(outside):
            bad = points[start + int(np.argmax(outside))]
            raise ValueError(
                f"The point ({bad[0]:g}, {bad[1]:g}) um lies outside the meshed "
                f"cross-section (by more than {tol_um:g} um): it is off the "
                "domain or inside a conductor that was left out of the mesh as "
                "a perfect conductor."
            )
        found[start : start + chunk] = usable[best]
        bary[start : start + chunk] = lam[rows, best]
    return found, bary


def voltage_line_integral(
    field: BoundaryModeField,
    path_um: ArrayLike,
    *,
    nsamples: int = VOLTAGE_NSAMPLES,
    tol_um: float = DEFAULT_PERIMETER_TOL_UM,
) -> complex:
    """Line integral of the transverse electric field along a polyline.

    ``V = integral of E_t . dl`` from the first point of the path to the
    last. The path is sampled with composite four-point Gauss-Legendre
    panels, and ``E_t`` is evaluated at each sample from the second-order
    shape functions of the triangle that contains it, so a field that
    the mesh represents exactly (any quadratic) is integrated exactly
    along a straight segment inside one polynomial piece; across element
    boundaries the quadrature converges as ``nsamples`` grows.

    The result is in the um-length scale of :func:`contour_current` (V/m
    times um): divide by ``1e6`` for volts. Taking the voltage and the
    reaction from the same saved field keeps ``V I`` and ``2 P`` in one
    scale.

    Args:
        field: The Mode's saved fields.
        path_um: ``(M, 2)`` polyline points in the cross-section's own
            ``(h, v)`` coordinates (um), ``M >= 2``, consecutive points
            distinct.
        nsamples: Sample points per segment, rounded up to a multiple of
            four. The default is :data:`VOLTAGE_NSAMPLES` (800).
        tol_um: How far outside the mesh a sample may sit and still be
            evaluated (um). Palace writes its coordinates in single
            precision, so a path a hair inside a gap may land a hair off
            the written mesh.

    Returns:
        The complex line integral.

    Raises:
        ValueError: When the path is malformed, a segment has zero length,
            ``nsamples`` is below one, or a sample falls outside the
            mesh (off the domain, or inside a ``"pec"`` conductor's hole).
    """
    path = np.asarray(path_um, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        raise ValueError(
            "A voltage path is an (M, 2) polyline of (h, v) points with M >= 2; "
            f"got an array of shape {path.shape}."
        )
    if not np.all(np.isfinite(path)):
        raise ValueError("The voltage path has non-finite coordinates.")
    if nsamples < 1:
        raise ValueError(f"nsamples must be at least 1, got {nsamples}.")
    steps = np.diff(path, axis=0)
    if np.any(np.linalg.norm(steps, axis=1) == 0.0):
        raise ValueError("The voltage path repeats a point: a segment has no length.")

    panels = -(-int(nsamples) // 4)
    nodes, gauss = np.polynomial.legendre.leggauss(4)
    edges = np.linspace(0.0, 1.0, panels + 1)
    width = edges[1:] - edges[:-1]
    centre = 0.5 * (edges[1:] + edges[:-1])
    params = (centre[:, None] + 0.5 * width[:, None] * nodes[None, :]).ravel()
    weights = (0.5 * width[:, None] * gauss[None, :]).ravel()

    total = 0.0 + 0.0j
    for k in range(len(steps)):
        start, step = path[k], steps[k]
        samples = start[None, :] + params[:, None] * step[None, :]
        cell, bary = _locate_points(field, samples, tol_um)
        e_at = np.einsum(
            "pn,pnk->pk", _shape_values_at(bary), field.e_t[field.cells[cell]]
        )
        total += np.sum(weights * (e_at @ step))
    return complex(total)


def reaction_matrix(fields: Sequence[BoundaryModeField]) -> NDArray[np.complex128]:
    """Unconjugated reaction of every pair of Modes on one cross-section.

    ``R_jk = integral of (E_t,j x H_t,k) . z dA`` over the whole meshed
    cross-section, with the quadrature :func:`power_flux` uses. For
    distinct Modes of a reciprocal line ``R`` is diagonal, and it is the
    ``T_V^T T_I`` that :func:`gsim.palace.terminal.currents_from_reaction`
    takes. For one real Mode ``R[0, 0] = 2 * power_flux``. For a complex
    Mode the two differ: ``power_flux`` conjugates ``H``, the reaction
    does not.

    Args:
        fields: The Modes' saved fields, all on the same mesh.

    Returns:
        ``(n, n)`` complex matrix in the um-area scale of
        :func:`power_flux`.

    Raises:
        ValueError: When no field is given, or the fields do not share one
            mesh.
    """
    modes = list(fields)
    if not modes:
        raise ValueError("reaction_matrix needs at least one Mode.")
    first = modes[0]
    for index, other in enumerate(modes[1:], start=1):
        if not (
            np.array_equal(other.cells, first.cells)
            and np.array_equal(other.points_um, first.points_um)
        ):
            raise ValueError(
                f"Mode {index} is not on the mesh of Mode 0: a reaction pairs "
                "Modes of one meshed cross-section."
            )
    areas = _triangle_areas(first)
    e_all = np.stack([m.e_t[first.cells] for m in modes])
    h_all = np.stack([m.h_t[first.cells] for m in modes])
    reaction = np.zeros((len(modes), len(modes)), dtype=np.complex128)
    for l0, l1, l2, weight in _QUADRATURE:
        shape = _shape_values(l0, l1, l2)
        e_q = np.einsum("n,jcnk->jck", shape, e_all)
        h_q = np.einsum("n,jcnk->jck", shape, h_all)
        cross = (
            e_q[:, None, :, 0] * h_q[None, :, :, 1]
            - e_q[:, None, :, 1] * h_q[None, :, :, 0]
        )
        reaction += weight * np.einsum("jkc,c->jk", cross, areas)
    return reaction


def _rectangle_side_normals(
    midpoints: NDArray[np.float64],
    *,
    h_span: tuple[float, float],
    v_span: tuple[float, float],
    tol_um: float,
) -> NDArray[np.float64]:
    """Outward normal of the rectangle side each midpoint sits on.

    Rows for midpoints that sit on no side come back as zeros, which
    contribute nothing to the contour integral.
    """
    normals = np.zeros_like(midpoints)
    on_low_h = np.abs(midpoints[:, 0] - h_span[0]) <= tol_um
    on_high_h = np.abs(midpoints[:, 0] - h_span[1]) <= tol_um
    on_low_v = np.abs(midpoints[:, 1] - v_span[0]) <= tol_um
    on_high_v = np.abs(midpoints[:, 1] - v_span[1]) <= tol_um
    normals[on_low_h] = (-1.0, 0.0)
    normals[on_high_h] = (1.0, 0.0)
    normals[on_low_v] = (0.0, -1.0)
    normals[on_high_v] = (0.0, 1.0)
    return normals


def contour_current(
    field: BoundaryModeField,
    *,
    h_span: tuple[float, float],
    v_span: tuple[float, float],
    tol_um: float = DEFAULT_PERIMETER_TOL_UM,
) -> complex:
    """Ampere's contour integral of ``H`` around one rectangular conductor.

    ``I = contour integral of H . dl``, written with the side's outward
    normal as ``(n x H) . z``. This is the current definition a perfect
    conductor has — it carries no volume current to integrate — and it
    reads the same enclosed current for a conductor meshed as a Region.

    Each mesh edge lying on the rectangle's perimeter contributes; edges
    shared by two triangles (which is how a conductor meshed as a Region
    presents its outline) are counted once.

    Args:
        field: The Mode's saved fields.
        h_span: ``(min, max)`` of the conductor along the first
            coordinate (um).
        v_span: ``(min, max)`` along the second coordinate (um).
        tol_um: How far an edge may sit off the perimeter and still
            count as on it (um).

    Returns:
        The enclosed current, in the um-length scale
        :func:`z0_power_current` divides out.

    Raises:
        ValueError: When no mesh edge lies on the rectangle, so the
            conductor named is not on this Cross-section.
    """
    # Edge n of a second-order triangle runs between corners n and n+1
    # with its midside node at 3 + n.
    edges = np.concatenate(
        [
            field.cells[:, [0, 1, 3]],
            field.cells[:, [1, 2, 4]],
            field.cells[:, [2, 0, 5]],
        ]
    )
    ends = field.points_um[edges[:, :2]]
    midpoints = ends.mean(axis=1)
    inside = (
        (midpoints[:, 0] >= h_span[0] - tol_um)
        & (midpoints[:, 0] <= h_span[1] + tol_um)
        & (midpoints[:, 1] >= v_span[0] - tol_um)
        & (midpoints[:, 1] <= v_span[1] + tol_um)
    )
    normals = _rectangle_side_normals(
        midpoints, h_span=h_span, v_span=v_span, tol_um=tol_um
    )
    on_perimeter = inside & np.any(normals != 0.0, axis=1)
    if not np.any(on_perimeter):
        raise ValueError(
            f"No mesh edge lies on the rectangle h={h_span}, v={v_span}, so no "
            "conductor of the cross-section has that outline."
        )

    edges = edges[on_perimeter]
    normals = normals[on_perimeter]
    ends = ends[on_perimeter]
    # One conductor outline, not two: a Region's outline is an edge of
    # the triangles on both sides of it and would otherwise count twice.
    _unique, first = np.unique(np.sort(edges[:, :2], axis=1), axis=0, return_index=True)
    edges, normals, ends = edges[first], normals[first], ends[first]

    lengths = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
    h_edge = field.h_t[edges]
    # Simpson over each edge's two ends and its midside node, which is
    # exact for the second-order field they interpolate.
    weights = np.asarray([1.0 / 6.0, 1.0 / 6.0, 4.0 / 6.0])
    h_mean = np.einsum("n,cnk->ck", weights, h_edge)
    tangential = normals[:, 0] * h_mean[:, 1] - normals[:, 1] * h_mean[:, 0]
    return complex(np.sum(tangential * lengths))


def z0_power_current(
    field: BoundaryModeField,
    *,
    h_span: tuple[float, float],
    v_span: tuple[float, float],
    tol_um: float = DEFAULT_PERIMETER_TOL_UM,
) -> complex:
    """Marks-Williams power-current impedance of a saved Palace Mode.

    This Backend's two integrals — :func:`power_flux` over the whole
    Cross-section and :func:`contour_current` around the signal
    conductor — divided through the shared definition
    :func:`gsim.common.modes.z0_power_current`, which owns
    ``Z_0 = 2 P / |I|^2``, the zero-current refusal and the sign flip.
    The ratio is invariant to the mode's field normalization, and the um
    coordinates cancel between the area and the length integral, so the
    result is in ohms.

    Args:
        field: The Mode's saved fields.
        h_span: ``(min, max)`` of the signal conductor along the first
            coordinate (um).
        v_span: ``(min, max)`` along the second coordinate (um).
        tol_um: Perimeter tolerance passed to :func:`contour_current`.

    Returns:
        The complex characteristic impedance in ohms, its real part
        positive.

    Raises:
        ValueError: When the contour integral comes out zero, which
            means the Mode carries no current on that conductor.
    """
    current = contour_current(field, h_span=h_span, v_span=v_span, tol_um=tol_um)
    return common_z0(power_flux(field), current)


def field_index_ratio(field: BoundaryModeField) -> float:
    """The effective index the saved fields themselves imply.

    A Mode's transverse fields are related by its own index —
    ``H_t = (n_eff / eta_0) z-hat x E_t`` exactly for a TEM Mode and
    approximately for a quasi-TEM one — so ``eta_0 |H_t| / |E_t|``
    recovers roughly ``|n_eff|`` from the fields alone. That makes it a
    check on *which* Mode was read: the fields of one Mode put through
    another one's index disagree grossly, which is the failure a
    mode-to-cycle mapping can have.

    It is an order-of-magnitude check and nothing more. The relation is
    exact only for a TEM Mode, and the ratio of two field norms says
    nothing about a Mode's loss.

    Args:
        field: The Mode's saved fields.

    Returns:
        ``eta_0`` times the ratio of the RMS transverse fields, and NaN
        when the Mode carries no electric field to divide by.
    """
    e_rms = float(np.sqrt(np.mean(np.abs(field.e_t) ** 2)))
    h_rms = float(np.sqrt(np.mean(np.abs(field.h_t) ** 2)))
    if e_rms <= 0.0:
        return float("nan")
    return float(np.sqrt(MU0 / EPS0) * h_rms / e_rms)


#: How far the index a saved Mode's fields imply may sit from the index
#: its mode table reports before the two are called different Modes.
#: Loose on purpose: the relation behind :func:`field_index_ratio` is
#: exact only for a TEM Mode, so this catches a wrong file rather than a
#: quasi-TEM Mode's own departure from it.
FIELD_INDEX_RTOL: float = 2.0


def check_field_is_the_mode(
    field: BoundaryModeField, *, mode_id: int, n_eff: complex, context: str
) -> None:
    """Warn when the fields read back are not the Mode they were fetched for.

    Which ParaView cycle holds which Mode is a convention — Palace
    writes them in mode order, so cycle ``m`` is Mode ``m`` — and a
    convention is the kind of thing that changes without an error. The
    fields carry their own index (:func:`field_index_ratio`), so they
    can be asked whether they are the Mode they were fetched for.

    Args:
        field: The fields that were read back.
        mode_id: Palace's own mode number they were fetched under.
        n_eff: The effective index that Mode's table reports.
        context: Who is asking, opening the warning.
    """
    implied = field_index_ratio(field)
    expected = abs(n_eff)
    if not math.isfinite(implied) or expected <= 0.0:
        return
    if 1.0 / (1.0 + FIELD_INDEX_RTOL) <= implied / expected <= 1.0 + FIELD_INDEX_RTOL:
        return
    warnings.warn(
        f"{context} read back fields for palace mode {mode_id} whose own "
        f"effective index is about {implied:.3g}, against the {expected:.3g} "
        "its mode table reports: these are most likely a different mode's "
        "fields, and the characteristic impedance taken from them belongs to "
        "that one. Palace writes one paraview cycle per saved mode in mode "
        "order; check that it still does.",
        stacklevel=2,
    )
