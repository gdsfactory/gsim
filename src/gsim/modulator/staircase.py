"""Staircase route: carrier maps as piecewise-constant Palace domains.

Palace only accepts piecewise-constant materials per mesh domain, so a
continuously varying carrier distribution is represented as N adjacent
strips along the junction axis. Each strip becomes a patterned-dielectric
region through the same ``Layer``/``MaterialProperties`` machinery the
doped cross-section builder uses, so the result plugs straight into
``build_doped_cross_section(doping=...)`` and the native ``BoundaryMode``
solver.

Everything that turns a Carrier map into per-Strip averages lives here:

- :func:`staircase_profile` averages a sampled one-dimensional profile over N
  equal-width Strips, each carrying the exact average of the
  piecewise-linear interpolant over it.
- :func:`strip_averages_from_nodes` reduces a scattered two-dimensional
  node cloud (e.g. a :class:`gsim.tcad.results.CarrierMap`) to that
  one-dimensional profile first — the tested, reusable mesh-transfer
  step — and averages it over the Strips.
- :func:`build_staircase_cross_section` does the whole job in one call: a
  Carrier map, a strip count and the Junction extent in, a meshable
  Staircase cross-section out — the strip Regions, their material
  response for *both* EM Stages (:class:`Strips`), and the flanking
  electrodes. The drawing of the Strips and the material of each one are
  its own private steps.
- :func:`surroundings_from_section` supplies what the Strips are *not*:
  every other Region of the drawn Cross-section, cut against the Strip
  footprint, so the Staircase is the drawn waveguide with its doped
  silicon replaced by Strips rather than a bare silicon wire in the
  background medium. A Staircase built without them answers for a
  different guide, and the difference does not shrink with strip count.

``n_strips=1`` recovers the uniform-strip model: one rectangle spanning
the window carrying the profile average.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gsim.common.carriers import permittivity_perturbation
from gsim.common.cross_section import build_doped_cross_section
from gsim.common.interpolate import DegenerateSampleCloudError, sample_at
from gsim.common.stack.materials import MaterialProperties, make_doped_materials

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.modes import Conductor
    from gsim.common.stack.extractor import Layer, LayerStack

__all__ = [
    "BAND_TOL_UM",
    "COLUMN_TOL_FRACTION",
    "DEFAULT_DRAWING",
    "DEFAULT_ELECTRODES",
    "DEFAULT_SI_INDEX",
    "DEFAULT_STRIP_LAYER",
    "DEFAULT_SURROUND_LAYER",
    "SEGMENT_TOL_UM",
    "STRIP_LENGTH_UM",
    "CarrierCoupling",
    "ConductorModel",
    "CrossSectionOrientationError",
    "ElectrodeSpec",
    "MaterialResponseLike",
    "OpticalStripMaterial",
    "RFStripMaterial",
    "StaircaseCrossSection",
    "StaircaseDrawing",
    "StripMaterial",
    "StripSegment",
    "Strips",
    "SurroundingRegion",
    "build_staircase_cross_section",
    "carrier_map_extent",
    "staircase_profile",
    "strip_averages_from_nodes",
    "surroundings_from_section",
]

#: Unperturbed silicon refractive index near 1.55 um: what a Strip
#: carries before the carriers move it, when the drawn stack cannot say
#: what its own silicon is.
DEFAULT_SI_INDEX: float = 3.4757

#: ``(layer, datatype)`` Strip 0 is drawn on, Strip ``i`` taking
#: ``datatype + i``. Deliberately outside the range gdsfactory's generic
#: PDK uses: a Strip landing on a PDK metal or via layer is resolved as
#: that conductor, which a solver reading the stack (Palace) then honours
#: and one reading only the mesh regions (femwell) does not.
DEFAULT_STRIP_LAYER: tuple[int, int] = (300, 0)

#: ``(layer, datatype)`` the first surrounding Region is drawn on, the
#: next taking ``datatype + 1``. Outside the generic PDK's own layers for
#: the same reason :data:`DEFAULT_STRIP_LAYER` is, and distinct from it so
#: a Strip and a surrounding Region never share a GDS layer.
DEFAULT_SURROUND_LAYER: tuple[int, int] = (310, 0)

#: Extents closer than this count as coincident when a surrounding
#: Region is cut against the Strip footprint (um).
SURROUND_TOL_UM: float = 1e-9

#: Fraction of the sampled extent within which two nodes count as one
#: column of the mesh, when no explicit tolerance is given. Node columns
#: are what a 2D cloud is averaged over before it is averaged over Strips.
COLUMN_TOL_FRACTION: float = 1e-6

#: How far outside a band a node still belongs to it (um). A mesh hands a
#: surface over at its coordinate to rounding, and the surface rows are
#: the band's own edges.
BAND_TOL_UM: float = 1e-9

#: Heights the band average samples the interpolant at, per ``h``. The
#: interpolant is piecewise along a vertical line, with a kink at every
#: element edge it crosses; this many samples put several inside each
#: element of a mesh refined to a fortieth of the band.
BAND_AVERAGE_HEIGHT_SAMPLES: int = 129

#: Samples of the band average inside each Strip, along the junction axis,
#: on top of the cloud's own columns.
BAND_AVERAGE_SAMPLES_PER_STRIP: int = 8

#: Drawn length of a Staircase along the propagation direction (um).
#: The Cross-section is invariant along it, so it is not a setting;
#: a Stage meshing a Staircase cuts through the middle of it.
STRIP_LENGTH_UM: float = 10.0


class MaterialResponseLike(Protocol):
    """What a carrier coupling answers, sample by sample.

    The shape of :class:`gsim.modulator.carriers.MaterialResponse`, stated
    here so the Staircase depends on the answer and not on the Stage.
    """

    @property
    def index_shift(self) -> NDArray[np.float64]:
        """Refractive-index shift per sample."""
        ...

    @property
    def absorption_cm(self) -> NDArray[np.float64]:
        """Free-carrier absorption per sample (cm^-1)."""
        ...

    @property
    def conductivity_s_per_m(self) -> NDArray[np.float64]:
        """Drude conductivity per sample (S/m)."""
        ...


class CarrierCoupling(Protocol):
    """The one thing that turns concentrations into material response.

    The carriers Stage's ``response`` method satisfies it; a test hands in
    any callable of the same shape.
    """

    def __call__(self, n_cm3: ArrayLike, p_cm3: ArrayLike) -> MaterialResponseLike:
        """Couple electron and hole concentrations (cm^-3) to a response."""
        ...


@dataclass(frozen=True)
class OpticalStripMaterial:
    """What the optical Stage adds to a Strip's material.

    Attributes:
        wavelength_um: Vacuum wavelength the Stage solves at (um). Each
            Strip's extinction ``kappa = dalpha lambda / 4 pi`` is built
            at it, so the Staircase and a continuous ``eps(x, y)`` carry
            the same loss.
        index: Unperturbed refractive index of the Strips, which the
            plasma dispersion perturbs.
    """

    wavelength_um: float
    index: float = DEFAULT_SI_INDEX


@dataclass(frozen=True)
class RFStripMaterial:
    """What the RF Stage adds to a Strip's material.

    Attributes:
        permittivity: Relative permittivity of the Strip lattice, which
            the carrier conductivity loads.
        fmax_hz: Upper validity frequency (Hz) of the Drude material.
    """

    permittivity: float = 11.9
    fmax_hz: float = 200e9


#: The per-Stage strip input: one typed value per EM Stage.
StripMaterial = OpticalStripMaterial | RFStripMaterial


@dataclass(frozen=True)
class StaircaseDrawing:
    """How a Staircase is drawn, apart from what it carries.

    Attributes:
        length_um: Drawn length along the propagation direction (um).
        base_layer: ``(layer, datatype)`` of Strip 0; Strip ``i`` takes
            ``datatype + i``.
        surround_layer: ``(layer, datatype)`` of the first surrounding
            Region, the next taking ``datatype + 1``.
        name_prefix: Region-name prefix of the Strips.
        mesh_resolution: Mesh resolution assigned to the Strip layers.
        axis: Cross-section normal axis of the resolved stack.
        value: Cross-section plane coordinate (um): the middle of the
            drawn length, so the plane cuts through the Strips.
        substrate_thickness: Substrate thickness of the resolved stack (um).
        component: Component to draw on; a new one when ``None``.
    """

    length_um: float = STRIP_LENGTH_UM
    base_layer: tuple[int, int] = DEFAULT_STRIP_LAYER
    surround_layer: tuple[int, int] = DEFAULT_SURROUND_LAYER
    name_prefix: str = "strip_"
    mesh_resolution: str | float = "fine"
    axis: Literal["x", "y", "z"] = "x"
    value: float = STRIP_LENGTH_UM / 2.0
    substrate_thickness: float = 2.0
    component: gf.Component | None = None


#: The drawing every Stage uses unless it says otherwise.
DEFAULT_DRAWING = StaircaseDrawing()


def staircase_profile(
    h: ArrayLike,
    values: ArrayLike,
    *,
    n_bins: int,
    h_min: float | None = None,
    h_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average a sampled 1D profile over N equal-width piecewise-constant Strips.

    The samples are interpreted as a piecewise-linear function of ``h``; each
    strip value is the exact average of that interpolant over the strip, so
    ``n_bins=1`` recovers the exact mean of the profile and increasing
    ``n_bins`` converges to the continuous profile.

    Args:
        h: Sample coordinates along the Strip axis (um), any order.
        values: Sample values (e.g. carrier concentration, sigma, Delta n).
        n_bins: Number of strips (>= 1).
        h_min: Window start; defaults to ``min(h)``.
        h_max: Window end; defaults to ``max(h)``.

    Returns:
        ``(edges, means)`` — strip edges of length ``n_bins + 1`` and the
        per-strip averages of length ``n_bins``.
    """
    h_arr = np.asarray(h, dtype=np.float64).ravel()
    v_arr = np.asarray(values, dtype=np.float64).ravel()
    if h_arr.size != v_arr.size:
        raise ValueError("h and values must have the same length.")
    if h_arr.size < 2:
        raise ValueError("At least two samples are required.")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    order = np.argsort(h_arr)
    h_arr = h_arr[order]
    v_arr = v_arr[order]

    lo = float(h_arr[0]) if h_min is None else float(h_min)
    hi = float(h_arr[-1]) if h_max is None else float(h_max)
    if hi <= lo:
        raise ValueError("h_max must exceed h_min.")

    edges = np.asarray(np.linspace(lo, hi, n_bins + 1), dtype=np.float64)

    # Exact average of the piecewise-linear interpolant over each strip:
    # trapezoids between the samples and the edges, summed strip by strip.
    # Differencing one running integral instead would lose a depleted
    # strip, decades below its neighbours, to cancellation.
    dense = np.union1d(edges, h_arr[(h_arr > lo) & (h_arr < hi)])
    dense_v = np.interp(dense, h_arr, v_arr)
    areas = 0.5 * (dense_v[1:] + dense_v[:-1]) * np.diff(dense)
    midpoints = 0.5 * (dense[1:] + dense[:-1])
    strip = np.clip(np.searchsorted(edges, midpoints) - 1, 0, n_bins - 1)
    integrals = np.bincount(strip, weights=areas, minlength=n_bins)
    means = np.asarray(integrals / np.diff(edges), dtype=np.float64)
    return edges, means


def strip_averages_from_nodes(
    h_um: ArrayLike,
    values: ArrayLike,
    *,
    n_strips: int,
    h_min: float | None = None,
    h_max: float | None = None,
    v_um: ArrayLike | None = None,
    v_range: tuple[float, float] | None = None,
    column_tol_um: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average scattered node values into N strips along the junction axis.

    Node values (e.g. carrier concentrations on the charge-solve mesh) are
    reduced to a 1D profile of ``h`` and binned with
    :func:`staircase_profile`, whose strip values are
    exact averages of the piecewise-linear interpolant — so ``n_strips=1``
    recovers the profile mean and increasing N converges to the continuous
    profile.

    A band selected with ``v_range`` is averaged over its area: the
    profile at each ``h`` is the mean over the band's height of the
    linear interpolant of the node cloud — the field a linear transfer
    reads off the same nodes. Counting nodes instead would weight
    the field by the mesh's refinement: a charge-solve mesh puts a third
    of its nodes on the silicon's top and bottom lines, which have no
    area, and the carriers there are not the Strip's once they vary with
    depth. A band whose nodes span no height (a single row) has no area
    to weigh, and its nodes are the profile.

    The band is taken as a rectangle of silicon, the height of its nodes.
    Where the silicon changes height along the axis — a rib beside a
    thinner slab — one band over both would average across the oxide
    above the slab, on values interpolated between slab and rib; give
    each height its own call and ``v_range``, as a Staircase's segments
    do.

    Args:
        h_um: Node coordinates along the junction (binning) axis in um.
        values: Node values (same length as ``h_um``).
        n_strips: Number of strips (>= 1).
        h_min: Window start along the axis; defaults to ``min(h_um)``.
        h_max: Window end along the axis; defaults to ``max(h_um)``.
        v_um: Optional node coordinates transverse to the binning axis
            (e.g. z); used with ``v_range`` to select a band of a 2D node
            cloud before binning.
        v_range: Optional ``(min, max)`` band in the ``v_um`` coordinate.
        column_tol_um: Nodes whose ``h`` agree to within this (um) are one
            column of the mesh and are averaged together; defaults to
            :data:`COLUMN_TOL_FRACTION` of the sampled extent.

    Returns:
        ``(edges, means)`` — strip edges of length ``n_strips + 1`` and
        per-strip averages of length ``n_strips``.
    """
    h_arr: NDArray[np.float64] = np.asarray(h_um, dtype=np.float64).ravel()
    v_arr: NDArray[np.float64] = np.asarray(values, dtype=np.float64).ravel()
    if h_arr.size != v_arr.size:
        raise ValueError("h_um and values must have the same length.")
    if v_range is not None:
        if v_um is None:
            raise ValueError("v_range requires v_um node coordinates.")
        band = np.asarray(v_um, dtype=np.float64).ravel()
        if band.size != h_arr.size:
            raise ValueError("v_um must have the same length as h_um.")
        lo, hi = v_range
        mask = (band >= lo - BAND_TOL_UM) & (band <= hi + BAND_TOL_UM)
        if not np.any(mask):
            raise ValueError("No nodes inside v_range.")
        h_arr = np.asarray(h_arr[mask], dtype=np.float64)
        v_arr = np.asarray(v_arr[mask], dtype=np.float64)
        profile = _band_averaged_profile(
            h_arr,
            np.asarray(band[mask], dtype=np.float64),
            v_arr,
            n_strips=n_strips,
            h_min=h_min,
            h_max=h_max,
            column_tol_um=column_tol_um,
        )
        if profile is not None:
            return staircase_profile(
                *profile, n_bins=n_strips, h_min=h_min, h_max=h_max
            )
    # A 2D node cloud carries many nodes per h coordinate. staircase_profile
    # reads its samples as a piecewise-linear function of h, so a column of
    # nodes would leave one arbitrary node standing per h and discard the
    # rest of the band: average each column here instead. Columns are
    # grouped within a tolerance rather than by exact equality — a mesh
    # generator is free to place a column's nodes at coordinates agreeing
    # only to rounding.
    span = float(np.ptp(h_arr))
    tol = column_tol_um if column_tol_um is not None else COLUMN_TOL_FRACTION * span
    keys = np.round(h_arr / tol).astype(np.int64) if tol > 0.0 else h_arr
    _unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    if _unique.size != h_arr.size:
        h_arr = np.asarray(
            np.bincount(inverse, weights=h_arr) / counts, dtype=np.float64
        )
        v_arr = np.asarray(
            np.bincount(inverse, weights=v_arr) / counts, dtype=np.float64
        )
    return staircase_profile(h_arr, v_arr, n_bins=n_strips, h_min=h_min, h_max=h_max)


def _band_averaged_profile(
    h_um: NDArray[np.float64],
    v_um: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    n_strips: int,
    h_min: float | None,
    h_max: float | None,
    column_tol_um: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """The height-averaged profile of a 2D node cloud, or None without area.

    Args:
        h_um: Node coordinates of the band along the junction axis (um).
        v_um: Node coordinates of the band across it (um).
        values: Node values.
        n_strips: Number of Strips the profile is for; sets the sampling.
        h_min: Start of the extent along the axis; the cloud's when None.
        h_max: End of the extent along the axis; the cloud's when None.
        column_tol_um: Nodes closer than this (um) are one point.

    Returns:
        ``(h, mean)`` — sample coordinates inside the extent and the mean
        of the cloud's linear interpolant over the band's height at each; None
        when the nodes span no height or no width.
    """
    v_lo, v_hi = float(v_um.min()), float(v_um.max())
    h_lo, h_hi = float(h_um.min()), float(h_um.max())
    span = h_hi - h_lo
    if span <= 0.0 or v_hi - v_lo <= COLUMN_TOL_FRACTION * span:
        return None

    # Nodes on an Interface appear once per Region; one value per point.
    tol = column_tol_um if column_tol_um is not None else COLUMN_TOL_FRACTION * span
    if tol <= 0.0:
        # Exact coordinates: the smallest spacing that still tells them apart.
        tol = float(np.spacing(max(abs(h_lo), abs(h_hi), abs(v_lo), abs(v_hi))))
    keys = np.round(np.column_stack([h_um, v_um]) / tol).astype(np.int64)
    _unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    points = np.column_stack(
        [np.bincount(inverse, weights=c) / counts for c in (h_um, v_um)]
    )
    node_values = np.bincount(inverse, weights=values) / counts

    # The cloud's own columns, and samples inside every Strip: the
    # interpolant kinks wherever a vertical line meets an element edge,
    # which is between the columns as well as on them.
    lo = h_lo if h_min is None else max(h_lo, float(h_min))
    hi = h_hi if h_max is None else min(h_hi, float(h_max))
    inside = points[:, 0][(points[:, 0] >= lo) & (points[:, 0] <= hi)]
    regular = np.linspace(lo, hi, n_strips * BAND_AVERAGE_SAMPLES_PER_STRIP + 1)
    columns = np.unique(np.round(np.concatenate([inside, regular]) / tol)) * tol
    columns = np.clip(columns, lo, hi)
    heights = np.linspace(v_lo, v_hi, BAND_AVERAGE_HEIGHT_SAMPLES)

    hh, vv = np.meshgrid(columns, heights, indexing="ij")
    # A sample outside the cloud's hull takes its nearest node: rounding
    # leaves one on the hull's edge outside it now and then, and a band
    # that is no rectangle (see the caller) has whole corners out there.
    try:
        flat, _outside = sample_at(
            points,
            node_values,
            np.column_stack([hh.ravel(), vv.ravel()]),
            fill="nearest",
        )
    except DegenerateSampleCloudError:
        return None
    sampled = np.asarray(flat, dtype=np.float64).reshape(hh.shape)
    # Trapezoids on equal steps: the end samples count half.
    means = (sampled.sum(axis=1) - 0.5 * (sampled[:, 0] + sampled[:, -1])) / (
        heights.size - 1
    )
    return columns, np.asarray(means, dtype=np.float64)


@dataclass(frozen=True)
class StripSegment:
    """A run of equal Strips across one stretch of the drawn device.

    A Staircase built from segments rather than from one extent follows
    the device: each segment is tiled with its own number of equal
    Strips, and every Strip in it is drawn at the segment's own height
    and averages the Carrier map over that height. A rib beside a
    thinner slab is then two segments, not a slab drawn as tall as the
    rib; a lightly doped stretch where the carriers move and a heavily
    doped one where they do not take as many Strips as each needs.

    Attributes:
        span: ``(min, max)`` extent along the junction axis (um).
        z: ``(min, max)`` vertical extent of the silicon there (um).
        n_strips: Number of equal Strips tiling the span.
    """

    span: tuple[float, float]
    z: tuple[float, float]
    n_strips: int

    def __post_init__(self) -> None:
        """An ascending span and height, and at least one Strip."""
        if self.span[1] <= self.span[0]:
            raise ValueError(f"Segment span {self.span} must be ascending.")
        if self.z[1] <= self.z[0]:
            raise ValueError(f"Segment height {self.z} must be ascending.")
        if self.n_strips < 1:
            raise ValueError("A segment needs at least one strip.")


#: Segments closer than this count as touching (um).
SEGMENT_TOL_UM: float = 1e-9


def _check_segments(segments: Sequence[StripSegment]) -> None:
    """Segments tile one extent end to end, low edge first.

    Raises:
        ValueError: When two segments overlap or leave a gap.
    """
    for low, high in pairwise(segments):
        step = high.span[0] - low.span[1]
        if step < -SEGMENT_TOL_UM:
            raise ValueError(
                f"Strip segments {low.span} and {high.span} overlap; segments "
                "tile one extent end to end, low edge first."
            )
        if step > SEGMENT_TOL_UM:
            raise ValueError(
                f"Strip segments {low.span} and {high.span} leave a gap; "
                "segments tile one extent end to end, low edge first."
            )


@dataclass(frozen=True)
class Strips:
    """The Strips of a Staircase: what each carries, Strip by Strip.

    One entry per Strip, low edge first. The concentrations are the
    Carrier map's averages over the Strip; the response is what those
    averages turn into — the Drude conductivity the RF Stage meshes and
    the plasma-dispersion permittivity the optical Stage meshes.

    Attributes:
        edges_um: Ascending Strip edges along the junction axis (um),
            one more than the Strip count.
        zmin_um: Bottom of each Strip (um).
        zmax_um: Top of each Strip (um): the height of the silicon the
            Strip stands for, which a Staircase built from segments
            varies along the junction axis.
        electrons_cm3: Average electron concentration per Strip (cm^-3).
        holes_cm3: Average hole concentration per Strip (cm^-3).
        index_shift: Refractive-index shift per Strip.
        absorption_cm: Free-carrier absorption per Strip (cm^-1).
        conductivity_s_per_m: Drude conductivity per Strip (S/m).
        permittivity: Complex relative permittivity per Strip at the
            optical wavelength (``exp(+i omega t)``: lossy means
            ``Im < 0``).
    """

    edges_um: NDArray[np.float64]
    zmin_um: NDArray[np.float64]
    zmax_um: NDArray[np.float64]
    electrons_cm3: NDArray[np.float64]
    holes_cm3: NDArray[np.float64]
    index_shift: NDArray[np.float64]
    absorption_cm: NDArray[np.float64]
    conductivity_s_per_m: NDArray[np.float64]
    permittivity: NDArray[np.complex128]

    def __post_init__(self) -> None:
        """Every per-Strip array has one entry per Strip."""
        count = self.edges_um.size - 1
        if count < 1:
            raise ValueError("edges_um needs at least two values (one strip).")
        if np.any(np.diff(self.edges_um) <= 0):
            raise ValueError("edges_um must be strictly ascending.")
        for name in (
            "zmin_um",
            "zmax_um",
            "electrons_cm3",
            "holes_cm3",
            "index_shift",
            "absorption_cm",
            "conductivity_s_per_m",
            "permittivity",
        ):
            if getattr(self, name).size != count:
                raise ValueError(
                    f"{name} has {getattr(self, name).size} values for {count} strips."
                )

    @property
    def count(self) -> int:
        """Number of Strips."""
        return int(self.edges_um.size - 1)

    @property
    def span(self) -> tuple[float, float]:
        """``(min, max)`` extent the Strips tile (um)."""
        return (float(self.edges_um[0]), float(self.edges_um[-1]))


def _strip_response(
    edges: ArrayLike,
    n_strips_cm3: ArrayLike,
    p_strips_cm3: ArrayLike,
    *,
    zmin: ArrayLike,
    zmax: ArrayLike,
    response: CarrierCoupling,
    material: StripMaterial,
) -> Strips:
    """Per-strip material response of a Staircase.

    The coupling is evaluated here and nowhere else, so a wavelength or
    background-index mistake cannot recur per caller.

    Args:
        edges: Ascending strip edges (um), length N+1.
        n_strips_cm3: Per-strip average electron concentration (cm^-3).
        p_strips_cm3: Per-strip average hole concentration (cm^-3).
        zmin: Per-strip bottom (um).
        zmax: Per-strip top (um).
        response: The carriers Stage's coupling.
        material: The per-Stage strip input: an optical one builds each
            Strip's complex permittivity at the solve wavelength from
            the coupling's index shift and absorption; an RF one gives
            every Strip the lattice permittivity beside its Drude
            conductivity.

    Returns:
        The typed :class:`Strips` record.
    """
    edge_arr = np.asarray(edges, dtype=np.float64).ravel()
    n_arr = np.asarray(n_strips_cm3, dtype=np.float64).ravel()
    p_arr = np.asarray(p_strips_cm3, dtype=np.float64).ravel()
    coupled = response(n_arr, p_arr)
    dn = np.asarray(coupled.index_shift, dtype=np.float64).ravel()
    dalpha = np.asarray(coupled.absorption_cm, dtype=np.float64).ravel()
    sigma = np.asarray(coupled.conductivity_s_per_m, dtype=np.float64).ravel()
    if isinstance(material, OpticalStripMaterial):
        permittivity = np.asarray(
            [
                permittivity_perturbation(
                    n0=material.index,
                    dn=float(dn[i]),
                    dalpha_cm=float(dalpha[i]),
                    wavelength_um=material.wavelength_um,
                )
                for i in range(n_arr.size)
            ],
            dtype=np.complex128,
        )
    else:
        permittivity = np.full(n_arr.size, material.permittivity, dtype=np.complex128)
    return Strips(
        edges_um=edge_arr,
        zmin_um=np.asarray(zmin, dtype=np.float64).ravel(),
        zmax_um=np.asarray(zmax, dtype=np.float64).ravel(),
        electrons_cm3=n_arr,
        holes_cm3=p_arr,
        index_shift=dn,
        absorption_cm=dalpha,
        conductivity_s_per_m=sigma,
        permittivity=permittivity,
    )


def _strip_material(
    name: str, strips: Strips, index: int, *, material: StripMaterial
) -> dict[str, MaterialProperties]:
    """Material of one Strip, for the Stage the Staircase was built for.

    Args:
        name: Region (and material) name of the strip.
        strips: The Staircase's Strips.
        index: Strip index within them.
        material: The per-Stage strip input the Staircase was built with.

    Returns:
        ``{name: MaterialProperties}`` for that one strip.
    """
    if isinstance(material, RFStripMaterial):
        return make_doped_materials(
            [
                (
                    name,
                    material.permittivity,
                    float(strips.conductivity_s_per_m[index]),
                    f"carrier staircase ({name}) -- Drude sigma",
                )
            ],
            fmax=material.fmax_hz,
        )
    eps = complex(strips.permittivity[index])
    eps_re = float(eps.real)
    # exp(+i omega t) convention: lossy medium has Im(eps) < 0.
    loss_tangent = -float(eps.imag) / eps_re if eps_re > 0 else 0.0
    return {
        name: MaterialProperties(
            permittivity=eps_re,
            loss_tangent=loss_tangent,
            dispersion_models=[],
        )
    }


def _draw_strips(
    comp: gf.Component,
    *,
    edges: NDArray[np.float64],
    length: float,
    base_layer: tuple[int, int],
    zmin: NDArray[np.float64],
    zmax: NDArray[np.float64],
    name_prefix: str,
    mesh_resolution: str | float,
) -> tuple[dict[str, Layer], dict[str, float]]:
    """Draw N Strips on a component and build their layer specs.

    Strip ``i`` spans ``[edges[i], edges[i+1]]`` along y and gets a
    rectangle on GDS layer ``(base_layer[0], base_layer[1] + i)`` and a
    ``Layer`` spec named ``"{name_prefix}{i}"`` whose material shares the
    name, so the per-Strip material registered under it is what the
    mesher resolves.

    Args:
        comp: gdsfactory component the strip rectangles are added to.
        edges: Ascending strip edges along y (um), length N+1.
        length: Rectangle length along the propagation direction (um).
        base_layer: ``(layer, datatype)`` of strip 0; strip ``i`` uses
            ``datatype + i``.
        zmin: Bottom z of each strip (um).
        zmax: Top z of each strip (um).
        name_prefix: Region-name prefix.
        mesh_resolution: Mesh resolution assigned to the strip layers.

    Returns:
        ``(layer_specs, centres)`` keyed by Strip name, low edge first.
    """
    from gsim.common.stack.extractor import Layer

    if length <= 0:
        raise ValueError("length must be positive.")
    if np.any(zmax <= zmin):
        raise ValueError("zmax must exceed zmin.")

    layer_specs: dict[str, Layer] = {}
    centres: dict[str, float] = {}
    for i in range(edges.size - 1):
        name = f"{name_prefix}{i}"
        gds_layer = (base_layer[0], base_layer[1] + i)
        y0, y1 = float(edges[i]), float(edges[i + 1])

        # Drawn from its two edges rather than from a width and a centre:
        # adjacent Strips then hand the GDS grid the identical coordinate
        # for the edge they share, so no strip count can snap a sliver of
        # background between them.
        comp.add_polygon(
            [(0.0, y0), (length, y0), (length, y1), (0.0, y1)], layer=gds_layer
        )
        centres[name] = (y0 + y1) / 2
        bottom, top = float(zmin[i]), float(zmax[i])
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=bottom,
            zmax=top,
            thickness=top - bottom,
            material=name,
            layer_type="dielectric",
            mesh_resolution=mesh_resolution,
        )
    return layer_specs, centres


#: How the drawn conductors of a Traveling-wave electrode are expressed
#: in the meshed Cross-section. See ADR 0003.
ConductorModel = Literal["volume", "pec"]


@dataclass(frozen=True)
class ElectrodeSpec:
    """The Traveling-wave electrodes flanking a Staircase.

    Attributes:
        width_um: Width of each electrode along the junction axis (um).
        gap_um: Gap between the Junction extent and the electrode edge (um).
        thickness_um: Electrode thickness (um).
        conductor_model: How the metal is expressed in the mesh (ADR 0003).
            ``"volume"`` meshes each electrode as a Region of lossy metal
            carrying :attr:`sigma_s_per_m`; ``"pec"`` leaves its interior
            out of the meshed domain and makes its outline a perfect
            conductor. Only ``"pec"`` is a Cross-section both first-class
            Routes express identically, and only ``"pec"`` keeps an
            eigenvalue search off the metal-dominated modes a
            ``|eps| ~ 1e7`` Region carries.
        sigma_s_per_m: Electrode conductivity (S/m; aluminium by default),
            used for the RF target of the ``"volume"`` model. A ``"pec"``
            electrode has no conductivity to carry.
        optical_permittivity: Complex relative permittivity of the
            electrode metal at the optical wavelength, in the
            ``exp(+i omega t)`` convention (``Im < 0`` is lossy). The RF
            Drude conductivity above is meaningless at optical
            frequencies, so an optical ``"volume"`` Staircase that
            contains the electrodes needs this value; leave it unset when
            the optical Window excludes them, or when the electrodes are
            ``"pec"`` and so carry no permittivity at all.
        zmin: Bottom z of the electrodes (um); defaults to the strip zmin.
        names: Region names of the low-side and high-side electrode.
        gds_layer: ``(layer, datatype)`` of the low-side electrode; the
            high-side one uses ``datatype + 1``.
    """

    width_um: float = 2.0
    gap_um: float = 0.0
    thickness_um: float = 0.5
    conductor_model: ConductorModel = "volume"
    sigma_s_per_m: float = 3.8e7
    optical_permittivity: complex | None = None
    zmin: float | None = None
    names: tuple[str, str] = ("electrode_low", "electrode_high")
    gds_layer: tuple[int, int] = (42, 0)


#: Default Traveling-wave electrodes: 2 um wide, touching the Junction extent.
DEFAULT_ELECTRODES = ElectrodeSpec()


@dataclass
class StaircaseCrossSection:
    """A Carrier map reduced to Strips, ready to mesh.

    The strip Regions and the electrodes are drawn once on
    :attr:`component`; :meth:`stack` resolves them to a ``LayerStack``
    carrying the Strip materials of the Stage the Staircase was built
    for — the RF Stage's Drude conductivity or the optical Stage's
    perturbed permittivity, chosen by the :data:`StripMaterial` handed
    to the builder.

    Attributes:
        component: The gdsfactory component carrying the strip and
            electrode rectangles.
        strips: The :class:`Strips` — edges, average concentrations, and
            the conductivity and permittivity each carries.
        strip_names: Region names of the strips, low edge first.
        electrode_names: Region names of the electrodes (empty when the
            Staircase was built without them).
        electrode_spans: ``(min, max)`` extent of each electrode along the
            junction axis (um), in the same order as the names.
        surroundings: The drawn device's own Regions redrawn around the
            Strips (empty on a Staircase built without them).
        layers: Every Region the Staircase drew — Strips, electrodes and
            surroundings — as the layer spec the mesher reads, by name.
    """

    component: gf.Component
    strips: Strips
    strip_names: list[str]
    electrode_names: tuple[str, ...]
    electrode_spans: tuple[tuple[float, float], ...]
    surroundings: tuple[SurroundingRegion, ...]
    layers: dict[str, Layer]
    _centres: dict[str, float]
    _electrodes: ElectrodeSpec | None
    _material: StripMaterial
    _drawing: StaircaseDrawing
    _respond: Callable[[ArrayLike, ArrayLike], Strips]
    _stack: LayerStack | None = field(default=None)

    @property
    def strip_span(self) -> tuple[float, float]:
        """``(min, max)`` extent the Strips actually tile (um)."""
        return self.strips.span

    @property
    def conductor_model(self) -> ConductorModel | None:
        """How the electrodes are expressed in the mesh, or None.

        Returns:
            The :class:`ElectrodeSpec`'s model, and ``None`` for a
            Staircase drawn without electrodes.
        """
        return self._electrodes.conductor_model if self._electrodes else None

    def electrode_extent(
        self, name: str
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """The rectangle one electrode occupies on the Cross-section.

        A downstream integral over a conductor — the line current the
        characteristic impedance divides by — needs the conductor's
        outline, and under the ``"pec"`` model the mesh no longer carries
        it as a Region to look up.

        Args:
            name: Region name of the electrode.

        Returns:
            ``((h_min, h_max), (v_min, v_max))`` in um: the extent along
            the junction axis, then the vertical one.

        Raises:
            ValueError: When the Staircase has no electrode of that name.
        """
        if name not in self.electrode_names:
            raise ValueError(
                f"The staircase has no electrode named '{name}'; it drew "
                f"{list(self.electrode_names)}."
            )
        index = self.electrode_names.index(name)
        layer = self.layers[name]
        return (self.electrode_spans[index], (layer.zmin, layer.zmax))

    def conductor(self, name: str) -> Conductor:
        """One electrode, named the way a current integral reads it.

        Args:
            name: Region name of the electrode.

        Returns:
            The :class:`~gsim.common.modes.Conductor`: its Region, its
            extent and the model it reached the mesh under.

        Raises:
            ValueError: When the Staircase has no electrode of that name.
        """
        from gsim.common.modes import Conductor

        model = self.conductor_model
        if model is None:
            raise ValueError(
                f"The staircase drew no electrodes, so there is no conductor "
                f"named '{name}'."
            )
        return Conductor(name=name, extent=self.electrode_extent(name), model=model)

    def unloaded(self) -> StaircaseCrossSection:
        """The same Staircase with its carriers switched off.

        Same Strips, same electrodes, same surroundings and the same
        drawing — every Strip's electron and hole concentration set to
        zero, so it carries no Drude conductivity and no plasma
        dispersion, and a solve of it answers for the bare
        Traveling-wave electrode. The "EM solve of the bare electrode"
        half of the classic loaded-line workflow.

        Returns:
            A new Staircase over the same component.
        """
        zeros = np.zeros(self.strips.count, dtype=np.float64)
        bare = self._respond(zeros, zeros)
        return replace(self, strips=bare, _stack=None)

    @property
    def material(self) -> StripMaterial:
        """The per-Stage strip input this Staircase was built with."""
        return self._material

    def _doping(self) -> dict[str, Any]:
        """Layer specs, materials and centres for ``build_doped_cross_section``.

        Returns:
            The doping mapping the cross-section builder consumes.
        """
        materials: dict[str, MaterialProperties] = {}
        for index, name in enumerate(self.strip_names):
            materials.update(
                _strip_material(name, self.strips, index, material=self._material)
            )
        materials.update(self._electrode_materials())
        surrounding: dict[str, Any] = {
            region.material: region.properties
            for region in self.surroundings
            if region.properties is not None
        }
        return {
            "layer_specs": dict(self.layers),
            # The drawn stack's own material entries first, so a Strip or
            # an electrode sharing a name still wins: the Staircase is
            # what the Carrier map speaks for.
            "materials": surrounding | materials,
            "centres": dict(self._centres),
        }

    def _electrode_materials(self) -> dict[str, MaterialProperties]:
        """Electrode materials for the Stage the Staircase was built for.

        Returns:
            ``{name: MaterialProperties}`` for every electrode.

        Raises:
            ValueError: For an optical ``"volume"`` Staircase whose
                electrodes have no optical permittivity — the RF
                conductivity would model them as a near-transparent
                dielectric.
        """
        spec = self._electrodes
        if spec is None or not self.electrode_names:
            return {}
        if spec.conductor_model == "pec":
            # No conductivity and no loss: the native-2D mesher reads the
            # material to decide whether an electrode's outline carries a
            # finite-conductivity surface impedance or is a perfect
            # conductor, and a perfect conductor is the one both Routes
            # express identically (ADR 0003).
            return {
                name: MaterialProperties(
                    permittivity=1.0, loss_tangent=0.0, dispersion_models=[]
                )
                for name in self.electrode_names
            }
        if isinstance(self._material, RFStripMaterial):
            return make_doped_materials(
                [(name, spec.sigma_s_per_m) for name in self.electrode_names],
                permittivity=1.0,
                source_prefix="electrode",
            )
        if spec.optical_permittivity is None:
            raise ValueError(
                "The staircase electrodes only carry an RF Drude "
                "conductivity, which is meaningless at optical "
                "frequencies. Give the metal its optical permittivity "
                "(ElectrodeSpec(optical_permittivity=...)), or build the "
                "staircase with electrodes=None when the optical window "
                "excludes them."
            )
        eps = complex(spec.optical_permittivity)
        eps_re = float(eps.real)
        # exp(+i omega t) convention: lossy medium has Im(eps) < 0.
        loss_tangent = -float(eps.imag) / eps_re if eps_re != 0.0 else 0.0
        return {
            name: MaterialProperties(
                permittivity=eps_re,
                loss_tangent=loss_tangent,
                dispersion_models=[],
            )
            for name in self.electrode_names
        }

    def stack(self) -> LayerStack:
        """Resolve the Staircase into a meshable layer stack.

        Returns:
            The ``LayerStack``, built once and cached, carrying the Strip
            materials of the Stage this Staircase was built for.
        """
        if self._stack is None:
            stack, _section = build_doped_cross_section(
                self.component,
                axis=self._drawing.axis,
                value=self._drawing.value,
                substrate_thickness=self._drawing.substrate_thickness,
                doping=self._doping(),
                verbose=False,
            )
            self._stack = stack
        return self._stack


def _electrode_layers(
    comp: gf.Component,
    spec: ElectrodeSpec,
    *,
    junction: tuple[float, float],
    length: float,
    zmin: float,
    mesh_resolution: str | float,
) -> tuple[
    dict[str, Layer],
    dict[str, float],
    tuple[tuple[float, float], ...],
]:
    """Draw the flanking electrodes and build their layer specs."""
    from gsim.common.stack.extractor import Layer

    if spec.width_um <= 0:
        raise ValueError("Electrode width_um must be positive.")
    if spec.gap_um < 0:
        raise ValueError("Electrode gap_um must be non-negative.")
    if spec.thickness_um <= 0:
        raise ValueError("Electrode thickness_um must be positive.")

    h_min, h_max = junction
    spans = (
        (h_min - spec.gap_um - spec.width_um, h_min - spec.gap_um),
        (h_max + spec.gap_um, h_max + spec.gap_um + spec.width_um),
    )
    base_z = spec.zmin if spec.zmin is not None else zmin

    layer_specs: dict[str, Layer] = {}
    centres: dict[str, float] = {}
    for index, (name, (y0, y1)) in enumerate(zip(spec.names, spans, strict=True)):
        gds_layer = (spec.gds_layer[0], spec.gds_layer[1] + index)
        # From its edges, like the Strips: an electrode is meant to touch
        # the Strip lattice, not to sit a grid rounding away from it.
        comp.add_polygon(
            [(0.0, y0), (length, y0), (length, y1), (0.0, y1)], layer=gds_layer
        )
        centres[name] = (y0 + y1) / 2
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=base_z,
            zmax=base_z + spec.thickness_um,
            thickness=spec.thickness_um,
            material=name,
            # A conductor layer is what the native-2D mesher meshes as an
            # outline rather than as a domain, which is what makes the
            # "pec" model a boundary condition instead of a Region
            # (ADR 0003).
            layer_type="conductor" if spec.conductor_model == "pec" else "dielectric",
            mesh_resolution=mesh_resolution,
        )
    return layer_specs, centres, spans


def carrier_map_extent(
    carriers: Any, band: tuple[float, float] | None = None
) -> tuple[float, float]:
    """The extent a Carrier map covers along the junction axis (um).

    Strips average the Carrier map, so they cannot reach past it: the
    extent here is the widest one
    :func:`build_staircase_cross_section` will accept.

    Args:
        carriers: The Carrier map (anything exposing ``x_um`` and
            ``y_um``).
        band: ``(min, max)`` vertical band of samples to measure across;
            the whole map when omitted.

    Returns:
        ``(min, max)`` along the junction axis.

    Raises:
        ValueError: When the band holds no sample of the map.
    """
    h_um = np.asarray(carriers.x_um, dtype=np.float64).ravel()
    v_um = np.asarray(carriers.y_um, dtype=np.float64).ravel()
    if band is None:
        inside = np.ones(v_um.shape, dtype=bool)
    else:
        inside = (v_um >= band[0] - BAND_TOL_UM) & (v_um <= band[1] + BAND_TOL_UM)
    if not np.any(inside):
        raise ValueError(
            f"No carrier samples inside the vertical band {band}; "
            f"the map spans z in [{v_um.min():.3g}, {v_um.max():.3g}] um."
        )
    return (float(h_um[inside].min()), float(h_um[inside].max()))


@dataclass(frozen=True)
class SurroundingRegion:
    """A Region of the drawn device redrawn beside the Strips.

    The Strips carry the Carrier map, and nothing else. Everything the
    drawn Cross-section has around them — the undoped silicon the guide
    slab is made of, the Traveling-wave metal landing on the pads, an
    implant the charge solve never covered — guides the Mode just as much,
    and a Staircase that omits it solves a different waveguide. Each such
    Region reaches the Staircase as one of these, cut against the Strip
    footprint so the two never overlap.

    Attributes:
        name: Region name on the meshed Cross-section.
        h: ``(min, max)`` extent along the junction axis (um).
        z: ``(min, max)`` vertical extent (um).
        material: Material name, as the drawn stack names it.
        layer_type: How the mesher expresses it — ``"dielectric"`` for a
            meshed domain, ``"conductor"`` for metal meshed as an outline
            (ADR 0003).
        properties: The material's own entry from the drawn stack, for a
            material the base materials database does not already carry
            (a doped-silicon material, say). ``None`` leaves the lookup to
            the database.
        mesh_resolution: Mesh resolution assigned to the Region.
    """

    name: str
    h: tuple[float, float]
    z: tuple[float, float]
    material: str
    layer_type: Literal["conductor", "via", "dielectric", "substrate"] = "dielectric"
    properties: Any | None = None
    mesh_resolution: str | float = "fine"


class CrossSectionOrientationError(ValueError):
    """A drawn Cross-section was not cut on an x-normal plane."""


def _cut_against(
    h: tuple[float, float],
    z: tuple[float, float],
    *,
    box_h: tuple[float, float],
    box_z: tuple[float, float],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The parts of one axis-aligned rectangle outside another.

    Args:
        h: ``(min, max)`` in-plane extent of the rectangle (um).
        z: ``(min, max)`` vertical extent of the rectangle (um).
        box_h: In-plane extent of the rectangle cut out of it.
        box_z: Vertical extent of the rectangle cut out of it.

    Returns:
        Up to four disjoint rectangles covering exactly the part of the
        first that the second does not; the rectangle itself when the two
        do not overlap, and nothing when it is entirely inside.
    """
    overlap_h = (max(h[0], box_h[0]), min(h[1], box_h[1]))
    overlap_z = (max(z[0], box_z[0]), min(z[1], box_z[1]))
    if (
        overlap_h[1] - overlap_h[0] <= SURROUND_TOL_UM
        or overlap_z[1] - overlap_z[0] <= SURROUND_TOL_UM
    ):
        return [(h, z)]

    pieces: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if overlap_h[0] - h[0] > SURROUND_TOL_UM:
        pieces.append(((h[0], overlap_h[0]), z))
    if h[1] - overlap_h[1] > SURROUND_TOL_UM:
        pieces.append(((overlap_h[1], h[1]), z))
    if overlap_z[0] - z[0] > SURROUND_TOL_UM:
        pieces.append((overlap_h, (z[0], overlap_z[0])))
    if z[1] - overlap_z[1] > SURROUND_TOL_UM:
        pieces.append((overlap_h, (overlap_z[1], z[1])))
    return pieces


def surroundings_from_section(
    section: Iterable[Any],
    *,
    strip_span: tuple[float, float],
    strip_z: tuple[float, float],
    stack: LayerStack | None = None,
) -> tuple[SurroundingRegion, ...]:
    """Everything a drawn Cross-section has around the Strips.

    Each rectangle of the drawn section is cut against the Strip
    footprint: the part the Strips replace is dropped, and the rest
    becomes a :class:`SurroundingRegion` carrying the drawn material.
    That is one rule for every Region. The doped silicon the Carrier map
    speaks for disappears wherever the Strips cover it and survives
    wherever they do not, so a Strip extent narrower than the doped slab
    leaves unperturbed silicon rather than a hole.

    Args:
        section: Rectangles of the drawn Cross-section, each exposing
            ``layer_name``, ``material``, ``y0``, ``y1``, ``zmin`` and
            ``zmax`` (the output of
            :func:`gsim.common.cross_section.extract_plane_section` on an
            x-normal plane).
        strip_span: ``(min, max)`` extent the Strips tile (um).
        strip_z: ``(min, max)`` vertical extent of the Strips (um).
        stack: The drawn layer stack, read for two things the section
            rectangles do not carry: how each Region is meshed
            (``"conductor"`` metal becomes an outline rather than a
            domain, ADR 0003), and the material entry of a material the
            base database does not know.

    Returns:
        The surrounding Regions, in section order, each named after the
        Region it came from (suffixed when one rectangle cuts into
        several pieces).

    Raises:
        CrossSectionOrientationError: The Cross-section was not cut on an
            x-normal plane. No caller builds a y-normal Staircase, so the
            contract is enforced rather than generalised.
    """
    material_map = dict(stack.materials) if stack is not None else {}
    layers = dict(stack.layers) if stack is not None else {}
    rectangles = tuple(section)
    if rectangles and not hasattr(rectangles[0], "y0"):
        raise CrossSectionOrientationError(
            "A Staircase is cut on an x-normal plane, where every "
            "Cross-section rectangle carries a y extent. This one carries "
            "none: extract_plane_section returns RectYZ2D for axis 'x' "
            "alone (Rect2D for 'y', PolygonXY2D for 'z'), and the Staircase "
            "is built on the x-normal Cross-section only."
        )
    regions: list[SurroundingRegion] = []
    for rect in rectangles:
        name = str(rect.layer_name)
        h = (float(rect.y0), float(rect.y1))
        z = (float(rect.zmin), float(rect.zmax))
        if h[1] - h[0] <= SURROUND_TOL_UM or z[1] - z[0] <= SURROUND_TOL_UM:
            continue
        pieces = _cut_against(h, z, box_h=strip_span, box_z=strip_z)
        layer = layers.get(name)
        layer_type = layer.layer_type if layer is not None else "dielectric"
        for index, (piece_h, piece_z) in enumerate(pieces):
            regions.append(
                SurroundingRegion(
                    name=name if len(pieces) == 1 else f"{name}_{index}",
                    h=piece_h,
                    z=piece_z,
                    material=str(rect.material),
                    layer_type=layer_type,
                    properties=material_map.get(str(rect.material)),
                )
            )
    return tuple(regions)


def _surrounding_layers(
    comp: gf.Component,
    surroundings: Sequence[SurroundingRegion],
    *,
    length: float,
    base_layer: tuple[int, int],
) -> tuple[dict[str, Layer], dict[str, float]]:
    """Draw the surrounding Regions and build their layer specs."""
    from gsim.common.stack.extractor import Layer

    layer_specs: dict[str, Layer] = {}
    centres: dict[str, float] = {}
    for index, region in enumerate(surroundings):
        if region.h[1] <= region.h[0]:
            raise ValueError(
                f"Surrounding region '{region.name}' has a non-ascending "
                f"in-plane extent {region.h}."
            )
        if region.z[1] <= region.z[0]:
            raise ValueError(
                f"Surrounding region '{region.name}' has a non-ascending "
                f"vertical extent {region.z}."
            )
        if region.name in layer_specs:
            raise ValueError(
                f"Two surrounding regions are both named '{region.name}'; "
                "region names have to be unique on the cross-section."
            )
        gds_layer = (base_layer[0], base_layer[1] + index)
        y0, y1 = region.h
        comp.add_polygon(
            [(0.0, y0), (length, y0), (length, y1), (0.0, y1)], layer=gds_layer
        )
        centres[region.name] = 0.5 * (y0 + y1)
        layer_specs[region.name] = Layer(
            name=region.name,
            gds_layer=gds_layer,
            zmin=region.z[0],
            zmax=region.z[1],
            thickness=region.z[1] - region.z[0],
            material=region.material,
            layer_type=region.layer_type,
            mesh_resolution=region.mesh_resolution,
        )
    return layer_specs, centres


def _resolve_segments(
    segments: Sequence[StripSegment],
    *,
    n_strips: int | None,
    junction: tuple[float, float] | None,
    zmin: float | None,
    zmax: float | None,
    band: tuple[float, float] | None,
) -> tuple[tuple[StripSegment, ...], tuple[tuple[float, float], ...]]:
    """The segments a Staircase is built from, and the band each averages.

    One extent is one segment; segments are taken as given. The two ways
    are exclusive, so a Staircase never silently drops one of them.

    Returns:
        ``(segments, bands)``, one band per segment.
    """
    single = (n_strips, junction, zmin, zmax)
    if segments:
        if any(value is not None for value in single):
            raise ValueError(
                "Give the strips either as segments or as n_strips, junction, "
                "zmin and zmax, not both."
            )
        runs = tuple(segments)
        _check_segments(runs)
        return runs, tuple(run.z for run in runs)
    if n_strips is None or junction is None or zmin is None or zmax is None:
        raise ValueError(
            "A staircase needs its strips: either segments, or n_strips, "
            "junction, zmin and zmax."
        )
    if junction[1] <= junction[0]:
        raise ValueError("junction must be an ascending (min, max) extent.")
    run = StripSegment(
        span=(float(junction[0]), float(junction[1])),
        z=(float(zmin), float(zmax)),
        n_strips=n_strips,
    )
    return (run,), (band if band is not None else run.z,)


def build_staircase_cross_section(
    carriers: Any,
    *,
    n_strips: int | None = None,
    junction: tuple[float, float] | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    segments: Sequence[StripSegment] = (),
    response: CarrierCoupling,
    material: StripMaterial,
    band: tuple[float, float] | None = None,
    electrodes: ElectrodeSpec | None = DEFAULT_ELECTRODES,
    surroundings: Sequence[SurroundingRegion] = (),
    drawing: StaircaseDrawing = DEFAULT_DRAWING,
) -> StaircaseCrossSection:
    """Turn a Carrier map into a meshable Staircase cross-section.

    The Carrier map is binned into *n_strips* piecewise-constant Strips
    across the Junction extent, the carriers Stage's coupling turns each
    Strip's averages into its material, and each Strip is drawn as its
    own Region beside the Traveling-wave electrodes and whatever of the
    drawn device is redrawn around them.

    The Strips come either from one extent — *n_strips* equal Strips
    across *junction*, all between *zmin* and *zmax* — or from
    *segments*, each tiled with its own Strips at its own height, which
    is how a Staircase follows a rib standing beside a thinner slab.

    Coordinates follow the Carrier map's own frame: ``carriers.x_um`` runs
    along the junction axis (the in-plane coordinate of the Cross-section)
    and ``carriers.y_um`` is the vertical one, which is how the
    charge-transport backend reports a Carrier map.

    Args:
        carriers: The Carrier map to staircase (anything exposing
            ``x_um``, ``y_um``, ``electrons_cm3`` and ``holes_cm3``).
        n_strips: Number of Strips; ``1`` recovers the uniform model.
        junction: ``(min, max)`` Junction extent along the junction axis
            (um) the Strips tile.
        zmin: Bottom z of the Strips (um).
        zmax: Top z of the Strips (um).
        segments: The Strips as :class:`StripSegment` runs, low edge
            first, tiling one extent end to end — in place of
            *n_strips*, *junction*, *zmin* and *zmax*, which then stay
            unset.
        response: The carriers Stage's coupling — the one thing that
            turns electron and hole concentrations into index shift,
            absorption and conductivity.
        material: The per-Stage strip input: an
            :class:`OpticalStripMaterial` (wavelength and unperturbed
            index) or an :class:`RFStripMaterial` (lattice permittivity
            and top frequency).
        band: ``(min, max)`` vertical band of Carrier-map samples averaged
            into the Strips; defaults to ``(zmin, zmax)``. Segments
            always average over their own height.
        electrodes: Flanking electrodes; ``None`` draws none.
        surroundings: The drawn device's own Regions to redraw around the
            Strips — see :func:`surroundings_from_section`. Empty leaves
            the Staircase as Strips alone in the background medium, which
            is the right Cross-section only when the drawn device has
            nothing else inside the meshed Window.
        drawing: How the Staircase is drawn — layers, names, length,
            plane and substrate; :data:`DEFAULT_DRAWING` otherwise.

    Returns:
        The :class:`StaircaseCrossSection`.

    Raises:
        ValueError: When the extent reaches outside the Carrier map, the
            segments overlap or leave a gap, both ways of giving the
            Strips are used at once, or the strip count or geometry is
            not usable.
    """
    import gdsfactory as gf

    runs, bands = _resolve_segments(
        segments,
        n_strips=n_strips,
        junction=junction,
        zmin=zmin,
        zmax=zmax,
        band=band,
    )
    h_min, h_max = runs[0].span[0], runs[-1].span[1]

    h_um = np.asarray(carriers.x_um, dtype=np.float64).ravel()
    v_um = np.asarray(carriers.y_um, dtype=np.float64).ravel()
    edge_runs: list[NDArray[np.float64]] = []
    n_runs: list[NDArray[np.float64]] = []
    p_runs: list[NDArray[np.float64]] = []
    bottoms: list[NDArray[np.float64]] = []
    tops: list[NDArray[np.float64]] = []
    for run, band_range in zip(runs, bands, strict=True):
        covered = carrier_map_extent(carriers, band_range)
        if run.span[0] < covered[0] or run.span[1] > covered[1]:
            raise ValueError(
                f"Strip extent {run.span} reaches outside the carrier map, "
                f"which covers [{covered[0]:.3g}, {covered[1]:.3g}] um along the "
                "junction axis. Widen the charge Window or narrow the extent."
            )
        averages = [
            strip_averages_from_nodes(
                h_um,
                values,
                n_strips=run.n_strips,
                h_min=run.span[0],
                h_max=run.span[1],
                v_um=v_um,
                v_range=band_range,
            )
            for values in (carriers.electrons_cm3, carriers.holes_cm3)
        ]
        run_edges = averages[0][0]
        # A segment shares its low edge with the one before it.
        edge_runs.append(run_edges if not edge_runs else run_edges[1:])
        n_runs.append(averages[0][1])
        p_runs.append(averages[1][1])
        bottoms.append(np.full(run.n_strips, run.z[0], dtype=np.float64))
        tops.append(np.full(run.n_strips, run.z[1], dtype=np.float64))
    edges = np.concatenate(edge_runs)
    strip_zmin = np.concatenate(bottoms)
    strip_zmax = np.concatenate(tops)

    def respond(n_cm3: ArrayLike, p_cm3: ArrayLike) -> Strips:
        """The Strips these averages make, under this Staircase's coupling."""
        return _strip_response(
            edges,
            n_cm3,
            p_cm3,
            zmin=strip_zmin,
            zmax=strip_zmax,
            response=response,
            material=material,
        )

    strips = respond(np.concatenate(n_runs), np.concatenate(p_runs))
    length = drawing.length_um
    comp = drawing.component if drawing.component is not None else gf.Component()
    layer_specs, centres = _draw_strips(
        comp,
        edges=strips.edges_um,
        length=length,
        base_layer=drawing.base_layer,
        zmin=strip_zmin,
        zmax=strip_zmax,
        name_prefix=drawing.name_prefix,
        mesh_resolution=drawing.mesh_resolution,
    )
    strip_names = list(layer_specs)

    electrode_names: tuple[str, ...] = ()
    electrode_spans: tuple[tuple[float, float], ...] = ()
    if electrodes is not None:
        specs, electrode_centres, electrode_spans = _electrode_layers(
            comp,
            electrodes,
            junction=(h_min, h_max),
            length=length,
            zmin=float(strip_zmin.min()),
            mesh_resolution=drawing.mesh_resolution,
        )
        layer_specs.update(specs)
        centres.update(electrode_centres)
        electrode_names = tuple(specs)

    if surroundings:
        clashes = [region.name for region in surroundings if region.name in layer_specs]
        if clashes:
            raise ValueError(
                f"Surrounding region(s) {clashes} share a name with a strip "
                "or an electrode of this staircase; rename them so every "
                "region on the cross-section is distinct."
            )
        specs, surround_centres = _surrounding_layers(
            comp,
            tuple(surroundings),
            length=length,
            base_layer=drawing.surround_layer,
        )
        layer_specs.update(specs)
        centres.update(surround_centres)

    return StaircaseCrossSection(
        component=comp,
        strips=strips,
        strip_names=strip_names,
        electrode_names=electrode_names,
        electrode_spans=electrode_spans,
        surroundings=tuple(surroundings),
        layers=layer_specs,
        _centres=centres,
        _electrodes=electrodes,
        _material=material,
        _drawing=drawing,
        _respond=respond,
    )
