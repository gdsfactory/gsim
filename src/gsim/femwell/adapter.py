"""femwell adapter on the shared native-2D cross-section mesh.

Loads the exact msh v2.2 mesh the BoundaryMode pipeline generates into
skfem/femwell, with two epsilon paths:

- **piecewise-constant**: every named 2D physical group gets the complex
  relative permittivity of its stack material at the target wavelength or
  frequency (:func:`epsilon_by_region`) — the configuration Palace can
  also express, used for cross-solver validation.
- **continuous**: carrier-derived eps(x, y) node values (e.g. from a
  :class:`gsim.tcad.results.CarrierMap` through
  :func:`gsim.common.carriers.permittivity_perturbation`) projected onto a
  piecewise-element basis (:func:`elementwise_epsilon`) — the configuration
  Palace cannot express.

Both epsilon paths are pure meshio/scipy functions testable without the
femwell runtime; only :func:`solve_modes` needs femwell/skfem installed
(``pip install 'gsim[femwell]'``).

What comes back from a solve is read here too: how much of a Mode's
field is left at the Window boundary (:func:`boundary_field_ratio`) or
outside a band of the Cross-section (:func:`field_fraction_outside`),
the current one conductor carries (:func:`electrode_current`) and the
Marks-Williams power-current impedance it implies
(:func:`z0_power_current`), and — for an RF line — the one reading of a
selected Mode the RF Stage asks for: index, impedance and whether it is
the wall Mode (:func:`line_reading`). A conductor is named to every
current integral the same way, as a
:class:`~gsim.common.modes.Conductor`.

The sign convention is ``exp(+i omega t)``: lossy media have
``Im(eps) < 0``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import meshio
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import epsilon_0 as EPS0  # noqa: N812
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.interpolate import sample_at
from gsim.common.mesh_regions import (
    cell_blocks,
    group_names,
    group_tags,
    region_elements,
)
from gsim.common.modes import (
    Conductor,
    LineReading,
    wall_mode_from_currents,
)
from gsim.common.modes import z0_power_current as common_z0
from gsim.common.stack.materials import (
    MaterialProperties,
    ResolvedMaterial,
    region_material_map,
    resolve_stack_material,
)
from gsim.femwell.runtime import require_femwell, require_skfem

if TYPE_CHECKING:
    from gsim.common.stack.extractor import LayerStack

__all__ = [
    "boundary_facets_within",
    "boundary_field_ratio",
    "electrode_current",
    "elementwise_epsilon",
    "epsilon_by_region",
    "field_fraction_outside",
    "line_reading",
    "solve_modes",
    "z0_power_current",
]


def _complex_permittivity(
    resolved: ResolvedMaterial, *, frequency_hz: float
) -> complex:
    """Complex relative permittivity in the exp(+i omega t) convention."""
    eps_re = resolved.permittivity_scalar
    if eps_re is None:
        eps_re = 1.0
    loss_tangent = (
        resolved.loss_tangent_scalar
        if resolved.loss_tangent_scalar is not None
        else 0.0
    )
    sigma = (
        resolved.conductivity_scalar
        if resolved.conductivity_scalar is not None
        else 0.0
    )
    omega = 2.0 * np.pi * frequency_hz
    return complex(eps_re * (1.0 - 1j * loss_tangent) - 1j * sigma / (omega * EPS0))


def epsilon_by_region(
    mesh: meshio.Mesh | str | Path,
    stack: LayerStack,
    *,
    wavelength_um: float | None = None,
    frequency_hz: float | None = None,
    overrides: dict[str, MaterialProperties] | None = None,
) -> dict[str, complex]:
    """Resolve the complex permittivity of every 2D mesh region.

    Materials come from the stack's materials (the same database the
    Palace config generator uses), evaluated at the target wavelength or
    frequency, so both solvers see identical piecewise-constant epsilon.

    Args:
        mesh: The shared msh v2.2 mesh (path or loaded meshio mesh).
        stack: Layer stack the mesh was generated from (region-to-material
            mapping and material property source).
        wavelength_um: Target vacuum wavelength in um (optical).
        frequency_hz: Target frequency in Hz (RF). Exactly one of
            ``wavelength_um`` / ``frequency_hz`` must be given.
        overrides: Optional material-property overrides by material name.

    Returns:
        ``{region_name: complex_relative_permittivity}`` for every dim-2
        physical group (``exp(+i omega t)``: lossy means ``Im < 0``).
    """
    if (wavelength_um is None) == (frequency_hz is None):
        raise ValueError("Give exactly one of wavelength_um or frequency_hz.")
    if wavelength_um is not None:
        frequency = C0 / (wavelength_um * 1e-6)
        wavelength = wavelength_um
    else:
        assert frequency_hz is not None  # noqa: S101 - guarded above
        frequency = float(frequency_hz)
        wavelength = C0 / frequency * 1e6

    if not isinstance(mesh, meshio.Mesh):
        mesh = meshio.read(str(mesh))
    regions = list(group_tags(mesh, dim=2))
    if not regions:
        raise ValueError("Mesh has no 2D physical groups.")

    materials = region_material_map(stack, regions)
    stack_materials = dict(stack.materials or {})

    result: dict[str, complex] = {}
    for region in regions:
        material = materials[region]
        entry = (overrides or {}).get(material, stack_materials.get(material))
        resolved = resolve_stack_material(material, entry, wavelength)
        if resolved is None:
            raise ValueError(
                f"Region '{region}' maps to material '{material}' which is "
                "not resolvable from the stack materials or the built-in "
                "database."
            )
        result[region] = _complex_permittivity(resolved, frequency_hz=frequency)
    return result


def elementwise_epsilon(
    mesh: meshio.Mesh | str | Path,
    x_um: ArrayLike,
    y_um: ArrayLike,
    eps_values: ArrayLike,
    *,
    fill: complex | None = None,
) -> NDArray[np.complex128]:
    """Project scattered eps(x, y) samples onto per-element (P0) values.

    Each triangle of the mesh gets the value of the linear interpolant of
    the samples at its centroid — the continuous-epsilon path Palace
    cannot express. Centroids outside the convex hull of the samples fall
    back to nearest-neighbour (or ``fill`` when given).

    Args:
        mesh: The shared msh v2.2 mesh (path or loaded meshio mesh).
        x_um: Sample x coordinates in um (e.g. charge-solve nodes).
        y_um: Sample y coordinates in um.
        eps_values: Complex permittivity samples at those points.
        fill: Value for centroids outside the sample hull; defaults to
            nearest-neighbour extrapolation.

    Returns:
        Complex epsilon per triangle, in the mesh's triangle order.
    """
    if not isinstance(mesh, meshio.Mesh):
        mesh = meshio.read(str(mesh))
    tris, _tags = cell_blocks(mesh, "triangle")
    centroids = mesh.points[tris][:, :, :2].mean(axis=1)

    points = np.column_stack(
        [
            np.asarray(x_um, dtype=np.float64).ravel(),
            np.asarray(y_um, dtype=np.float64).ravel(),
        ]
    )
    values = np.asarray(eps_values, dtype=np.complex128).ravel()
    if points.shape[0] != values.size:
        raise ValueError("x_um, y_um and eps_values must have the same length.")
    if points.shape[0] < 3:
        raise ValueError("At least three sample points are required.")

    sampled, _missing = sample_at(
        points, values, centroids, fill="nearest" if fill is None else fill
    )
    return np.asarray(sampled, dtype=np.complex128)


def solve_modes(
    msh_path: str | Path,
    *,
    epsilon: dict[str, complex] | ArrayLike,
    wavelength_um: float,
    num_modes: int = 1,
    order: int = 1,
    metallic_boundaries: bool = False,
    n_guess: float | None = None,
) -> Any:
    """Solve waveguide modes with femwell on the shared mesh.

    Requires the femwell runtime (``pip install 'gsim[femwell]'``).

    Args:
        msh_path: Path of the shared msh v2.2 mesh (um coordinates).
        epsilon: Either ``{region_name: eps}`` (piecewise-constant, from
            :func:`epsilon_by_region`) or per-triangle values (continuous,
            from :func:`elementwise_epsilon`).
        wavelength_um: Vacuum wavelength in um. For RF, pass the free-space
            wavelength of the target frequency (``c0 / f`` in um).
        num_modes: Number of modes to compute.
        order: Finite element order of the mode solve.
        metallic_boundaries: Enforce PEC on the outer boundary.
        n_guess: Effective-index guess centering the eigenvalue search.
            femwell's default guess tracks the largest permittivity, which
            for RF materials with big conductive ``|Im(eps)|`` can put the
            shift far from the physical quasi-TEM mode; pass an explicit
            guess (e.g. the expected slow-wave index) there.

    Returns:
        The femwell ``Modes`` result (each mode carries ``n_eff``).

    Raises:
        ValueError: If the mesh has no triangles, if a region in ``epsilon``
            is not on the mesh, if a 2D region on the mesh is missing from
            ``epsilon``, or if per-element values do not match the element
            count.
    """
    require_femwell()
    skfem = require_skfem()
    from femwell.maxwell.waveguide import compute_modes

    # Build the skfem mesh and the region-to-element mapping directly from
    # meshio: skfem's own msh subdomain parsing is version-fragile, and the
    # explicit construction keeps the meshio triangle order == skfem element
    # order (which the per-element epsilon path relies on).
    mio = meshio.read(str(msh_path))
    tris, phys_tags = cell_blocks(mio, "triangle")

    # Drop points no triangle references (e.g. nodes only line/contact
    # groups use): they would become zero rows in the eigenproblem and make
    # the shift-invert factorization exactly singular.
    points = mio.points
    used = np.unique(tris)
    if used.size != points.shape[0]:
        remap = np.full(points.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        tris = remap[tris]
        points = points[used]

    mesh = skfem.MeshTri(
        np.ascontiguousarray(points[:, :2].T, dtype=np.float64),
        np.ascontiguousarray(tris.T, dtype=np.int64),
    )
    basis0 = skfem.Basis(mesh, skfem.ElementTriP0())

    if isinstance(epsilon, dict):
        tags_by_name = group_tags(mio, dim=2)
        eps = basis0.zeros(dtype=complex)
        # cast: ty cannot narrow the dict half of the union on its own.
        eps_map = cast("dict[str, complex]", epsilon)  # type: ignore[redundant-cast]
        for region, value in eps_map.items():
            if region not in tags_by_name:
                raise ValueError(
                    f"Region '{region}' not found on the mesh. "
                    f"Available Regions: {sorted(tags_by_name)}"
                )
            # ElementTriP0: one dof per element, in element order.
            eps[phys_tags == tags_by_name[region]] = value
        # Every element must get a permittivity: a region left out of the map
        # keeps eps = 0, which is not a material at all -- it either makes the
        # shift-invert factorization singular or returns modes of a structure
        # the caller never described.
        mapped = {tags_by_name[region] for region in eps_map}
        present = {int(tag) for tag in np.unique(phys_tags)}
        missing = sorted(present - mapped)
        if missing:
            tag_names = group_names(mio, dim=2)
            named = [
                f"'{tag_names[tag]}'" if tag in tag_names else f"tag {tag}"
                for tag in missing
            ]
            raise ValueError(
                f"No permittivity given for mesh region(s) {', '.join(named)}. "
                "Every 2D region on the mesh must appear in the epsilon map."
            )
    else:
        eps = np.asarray(epsilon, dtype=np.complex128)
        if eps.size != basis0.N:
            raise ValueError(
                f"Per-element epsilon has {eps.size} values but the mesh "
                f"has {basis0.N} elements."
            )

    return compute_modes(
        basis0,
        eps,
        wavelength=wavelength_um,
        num_modes=num_modes,
        order=order,
        metallic_boundaries=metallic_boundaries,
        n_guess=n_guess,
    )


def boundary_field_ratio(mode: Any) -> float:
    """How much of a solved Mode's field is left at the Window boundary.

    A mode solve is only as trustworthy as the box it was solved in: a
    domain too small for the Mode truncates the evanescent tail and
    returns a confident effective index for a field that was never
    allowed to decay. The ratio is the peak electric-field magnitude over
    the elements touching the outer boundary of the meshed domain,
    divided by the peak over the whole cross-section — a well-contained
    Mode gives a number orders of magnitude below one.

    Only the outer boundary counts. Cut-outs inside the domain (a metal
    electrode is meshed as a void) are boundaries too, and the field
    peaks at their corners, but they are the structure rather than the
    edge of the Window.

    Args:
        mode: A femwell ``Mode`` from :func:`solve_modes`.

    Returns:
        Peak boundary field over peak field, in ``[0, 1]``.

    Raises:
        ValueError: When the Mode carries no field, or when no facet of
            the mesh lies on the domain's bounding box.
    """
    require_skfem()
    mesh = mode.basis.mesh
    facets = mesh.boundary_facets()
    midpoints = mesh.p[:, mesh.facets[:, facets]].mean(axis=1)
    lows = mesh.p.min(axis=1)
    highs = mesh.p.max(axis=1)
    tol = 1e-6 * float(np.max(highs - lows))
    on_window = np.zeros(facets.size, dtype=bool)
    for axis in range(mesh.p.shape[0]):
        on_window |= np.abs(midpoints[axis] - lows[axis]) <= tol
        on_window |= np.abs(midpoints[axis] - highs[axis]) <= tol
    if not on_window.any():
        raise ValueError(
            "No mesh facet lies on the domain's bounding box, so its outer "
            "boundary could not be identified."
        )

    (e_x, e_y), e_z = mode.basis.interpolate(mode.E)
    magnitude = np.abs(e_x) ** 2 + np.abs(e_y) ** 2 + np.abs(e_z) ** 2
    peak = float(magnitude.max())
    if peak <= 0.0:
        raise ValueError("The mode carries no field; nothing to compare.")
    elements = np.unique(mesh.f2t[0, facets[on_window]])
    return float(np.sqrt(float(magnitude[elements].max()) / peak))


def field_fraction_outside(
    mode: Any,
    span: tuple[float, float],
    *,
    axis: int = 0,
) -> float:
    """How much of a solved Mode's power sits outside an interval.

    A Staircase carries the carrier response only where its Strips are;
    a Mode that mostly lives elsewhere is being solved on a
    representation that cannot answer for it, however finely the Strips
    are binned. The fraction here is the Mode's power — ``|E|^2``
    integrated over the elements — outside *span* along one axis, over
    the power everywhere.

    Args:
        mode: A femwell ``Mode`` from :func:`solve_modes`.
        span: ``(min, max)`` interval in mesh units (um).
        axis: Mesh axis the interval is on; ``0`` is the in-plane one on
            a cross-section mesh, ``1`` the vertical one.

    Returns:
        The fraction in ``[0, 1]``; ``0`` when every element centroid
        falls inside the interval.

    Raises:
        ValueError: When the interval is not ascending, or the Mode
            carries no field.
    """
    require_skfem()
    low, high = float(span[0]), float(span[1])
    if high <= low:
        raise ValueError(f"span must be an ascending (min, max) interval, got {span}.")

    basis = mode.basis
    (e_x, e_y), e_z = basis.interpolate(mode.E)
    intensity = np.abs(e_x) ** 2 + np.abs(e_y) ** 2 + np.abs(e_z) ** 2
    power = np.asarray((intensity * basis.dx).sum(axis=1), dtype=np.float64)
    total = float(power.sum())
    if total <= 0.0:
        raise ValueError("The mode carries no field; nothing to compare.")

    mesh = basis.mesh
    centroids = mesh.p[axis, mesh.t].mean(axis=0)
    outside = (centroids < low) | (centroids > high)
    return float(power[outside].sum() / total)


def boundary_facets_within(
    mesh: Any,
    *,
    h_span: tuple[float, float],
    v_span: tuple[float, float],
    tol_um: float = 1e-4,
) -> NDArray[np.int64]:
    """Facets of the domain boundary lying within one axis-aligned rectangle.

    A conductor the mesh leaves out of its meshed domain — a Staircase
    electrode under the ``"pec"`` conductor model — is a hole, and its
    outline is part of the domain boundary. This finds that outline from
    the rectangle the conductor occupies, which is what
    :func:`electrode_current` integrates the line current around. The
    outline need not be the rectangle itself: any hole inside it counts,
    so a round conductor is found by its bounding box.

    Args:
        mesh: The skfem mesh a Mode was solved on
            (``mode.basis.mesh``).
        h_span: ``(min, max)`` of the rectangle along the first
            coordinate (um).
        v_span: ``(min, max)`` along the second coordinate (um).
        tol_um: How far a facet midpoint may sit outside the rectangle
            and still count as inside it (um). Loose enough to absorb
            the rounding a mesh file's own coordinate precision leaves,
            and far tighter than any drawn feature.

    Returns:
        The facet indices, ascending.

    Raises:
        ValueError: When no boundary facet lies within the rectangle,
            which means the conductor was meshed as a domain rather than
            left out of one.
    """
    facets = mesh.boundary_facets()
    midpoints = mesh.p[:, mesh.facets[:, facets]].mean(axis=1)
    h, v = midpoints[0], midpoints[1]
    inside_h = (h >= h_span[0] - tol_um) & (h <= h_span[1] + tol_um)
    inside_v = (v >= v_span[0] - tol_um) & (v <= v_span[1] + tol_um)
    selected = facets[inside_h & inside_v]
    if selected.size == 0:
        raise ValueError(
            f"No boundary facet lies within the rectangle h={h_span}, v={v_span}, "
            "so that conductor is not a hole in the meshed domain. Its "
            "current is a conduction integral over its elements rather than "
            "a contour integral around it."
        )
    return np.asarray(np.sort(selected), dtype=np.int64)


def z0_power_current(
    mode: Any,
    *,
    frequency_hz: float,
    conductor: Conductor,
    mesh: meshio.Mesh | str | Path,
) -> complex:
    """Marks-Williams power-current characteristic impedance of an RF mode.

    This Backend's two integrals — the complex Poynting flux
    ``P = (1/2) integral (E_t x H_t*) . z dA`` over the whole
    cross-section, and the longitudinal current ``I`` on the signal
    conductor (:func:`electrode_current`) — divided through the shared
    definition :func:`gsim.common.modes.z0_power_current`, which owns
    ``Z_0 = 2 P / |I|^2`` and the zero-current refusal. The ratio is
    invariant to the mode's field normalization; the mesh coordinates
    are in um and the unit conversion is internal.

    Args:
        mode: A femwell ``Mode`` from :func:`solve_modes` (fields solved
            with the complex permittivity that encodes the conductivity,
            ``exp(+i omega t)``: ``Im(eps) < 0``).
        frequency_hz: RF frequency of the solve in Hz.
        conductor: The signal conductor, named the way its current is
            read.
        mesh: The shared msh v2.2 mesh the Mode was solved on (path or
            loaded meshio mesh), whose Region names a ``"volume"``
            conductor's elements are found by.

    Returns:
        Complex characteristic impedance in ohms.

    Raises:
        ValueError: When the conductor has nothing to integrate over
            (from :func:`electrode_current`), or the current comes out
            zero.
    """
    skfem = require_skfem()
    from skfem.helpers import cross

    basis = mode.basis

    @skfem.Functional(dtype=np.complex128)  # type: ignore[untyped-decorator]
    def _power_form(w: Any) -> Any:
        return cross(w["E"][0], np.conj(w["H"][0]))

    power = 0.5 * _power_form.assemble(
        basis,
        E=basis.interpolate(mode.E),
        H=basis.interpolate(mode.H),
    )

    current = electrode_current(
        mode, frequency_hz=frequency_hz, conductor=conductor, mesh=mesh
    )
    return common_z0(power, current)


def electrode_current(
    mode: Any,
    *,
    frequency_hz: float,
    conductor: Conductor,
    mesh: meshio.Mesh | str | Path,
) -> complex:
    """Longitudinal current one conductor of a Mode carries.

    The current definition :func:`z0_power_current` divides the power
    by, on its own, so that a second conductor's current can be read the
    same way and compared with the first's: the line Mode's signal and
    return electrodes carry equal and opposite currents, the Mode
    between both electrodes together and the shielding wall has them
    alike (:func:`gsim.common.modes.common_mode_fraction`).

    A ``"volume"`` conductor carries the conduction current
    ``integral sigma E_z dA`` over the elements of its Region, with
    ``sigma`` implied by the Mode's own epsilon; a ``"pec"`` conductor
    left out of the meshed domain carries Ampere's contour integral of
    ``H`` around the outline its extent locates
    (:func:`boundary_facets_within`). Both land in the same scale, and
    both sign conventions are consistent from one conductor to the
    next — the conduction integral through ``E_z``, the contour through
    the boundary normal that points into every conductor alike — so two
    conductors' currents read the same way are comparable.

    Args:
        mode: A femwell ``Mode`` from :func:`solve_modes`.
        frequency_hz: RF frequency of the solve in Hz.
        conductor: The conductor, named the way its current is read.
        mesh: The shared msh v2.2 mesh the Mode was solved on (path or
            loaded meshio mesh).

    Returns:
        The complex current, in the um-coordinate scale
        :func:`z0_power_current` divides out.

    Raises:
        ValueError: When there is nothing to integrate over: a
            ``"volume"`` conductor with no Region on the mesh, or a
            ``"pec"`` conductor that is not a hole in it.
    """
    if conductor.model == "pec":
        h_span, v_span = conductor.extent
        facets = boundary_facets_within(mode.basis.mesh, h_span=h_span, v_span=v_span)
        return _contour_current(mode, facets)
    return _conduction_current(
        mode,
        frequency_hz=frequency_hz,
        current_elements=region_elements(mesh, conductor.name),
    )


def line_reading(
    mode: Any,
    *,
    frequency_hz: float,
    mesh: meshio.Mesh | str | Path,
    signal: Conductor,
    return_: Conductor | None = None,
) -> LineReading:
    """What the RF Stage asks of one selected Mode, in one reading.

    The index off the Mode, the impedance off its fields over the signal
    conductor, and — given the return electrode — whether the Mode is
    the wall Mode, from the balance of the two electrodes' currents
    (:func:`gsim.common.modes.wall_mode_from_currents`). A line with no
    single return electrode has no pair to compare, and is not checked.

    Args:
        mode: A femwell ``Mode`` from :func:`solve_modes`.
        frequency_hz: RF frequency of the solve in Hz.
        mesh: The shared msh v2.2 mesh the Mode was solved on.
        signal: The signal conductor.
        return_: The return conductor, or ``None`` when there is not
            exactly one.

    Returns:
        The reading.
    """
    z0 = z0_power_current(mode, frequency_hz=frequency_hz, conductor=signal, mesh=mesh)
    if return_ is None:
        return LineReading(
            n_eff=complex(mode.n_eff),
            z0_ohm=z0,
            wall_mode=None,
            diagnostic="the line has no single return electrode to compare "
            "the signal current against",
        )
    wall_mode, diagnostic = wall_mode_from_currents(
        electrode_current(mode, frequency_hz=frequency_hz, conductor=signal, mesh=mesh),
        electrode_current(
            mode, frequency_hz=frequency_hz, conductor=return_, mesh=mesh
        ),
    )
    return LineReading(
        n_eff=complex(mode.n_eff), z0_ohm=z0, wall_mode=wall_mode, diagnostic=diagnostic
    )


def _conduction_current(
    mode: Any,
    *,
    frequency_hz: float,
    current_elements: ArrayLike,
) -> complex:
    """Longitudinal conduction current over a conductor's own elements.

    The conductivity is the one the Mode was solved with: ``sigma =
    -Im(eps_r) omega eps_0`` where the imaginary part is negative.
    """
    skfem = require_skfem()

    omega = 2.0 * np.pi * float(frequency_hz)
    eps = np.asarray(mode.epsilon_r, dtype=np.complex128)
    sigma = np.where(eps.imag < 0.0, -eps.imag, 0.0) * omega * EPS0

    elements = np.atleast_1d(np.asarray(current_elements))
    if elements.dtype == bool:
        elements = np.flatnonzero(elements)
    if elements.size == 0:
        raise ValueError("The conductor has no elements to integrate the current over.")

    @skfem.Functional(dtype=np.complex128)  # type: ignore[untyped-decorator]
    def _current_form(w: Any) -> Any:
        return w["sigma"] * w["E"][1]

    sub = mode.basis.with_elements(elements)
    sub_sigma = mode.basis_epsilon_r.with_elements(elements)
    # Mesh coordinates are um: S/m -> S/um so the um^2 area integral is in A.
    return complex(
        1e-6
        * _current_form.assemble(
            sub,
            E=sub.interpolate(mode.E),
            sigma=sub_sigma.interpolate(np.asarray(sigma, dtype=np.float64)),
        )
    )


def _contour_current(mode: Any, facets: ArrayLike) -> complex:
    """Ampere's contour integral of ``H`` around a closed set of facets.

    The facets bound the conductor, so the enclosed current is
    ``I = contour integral of H . dl``, written with the facet normal as
    ``(n x H) . z``. The normal a boundary facet carries points out of
    the meshed domain and so consistently into the conductor, which sets
    the sign of the whole contour and leaves ``|I|`` — the only part
    ``Z_0`` reads — right either way.

    Args:
        mode: The solved Mode.
        facets: Facet indices of the closed contour
            (:func:`boundary_facets_within`).

    Returns:
        The enclosed current, in the same scale as the conduction
        integral (um-coordinate mesh, SI fields).
    """
    skfem = require_skfem()

    selected = np.atleast_1d(np.asarray(facets, dtype=np.int64))
    facet_basis = skfem.FacetBasis(mode.basis.mesh, mode.basis.elem, facets=selected)

    @skfem.Functional(dtype=np.complex128)  # type: ignore[untyped-decorator]
    def _ampere_form(w: Any) -> Any:
        (h_x, h_y), _h_z = w["H"]
        return w.n[0] * h_y - w.n[1] * h_x

    # Mesh coordinates are um and H is in A/m, so the um line integral is
    # 1e6 times the current in A -- the same scale the conduction integral
    # above lands in, which is what makes 2 P / |I|^2 come out in ohms.
    return complex(
        _ampere_form.assemble(facet_basis, H=facet_basis.interpolate(mode.H))
    )
