"""Thermal solver core: Joule source, steady and transient heat solves.

A pure functional core, independent of how the mesh was made. Inputs are
a ``skfem.MeshTri`` with coordinates in µm, per-element (P0) material
arrays in SI units, and named boundary facet groups
(``dict[str, NDArray[int]]`` of facet indices into ``mesh.facets``).

Physics (v1, 2D cross-section of an infinitely long heater):

- The unknown is the temperature **rise** dT above ambient (K), solved
  with P1 Lagrange elements; there is no absolute temperature in the
  linear problem.
- ``-div(k grad dT) = q`` with dT = 0 on the heat-sink facets (Dirichlet),
  ``-k d(dT)/dn = h dT`` on convective facets (Robin) and adiabatic
  everywhere else. The transient adds ``rho*c_p d(dT)/dt``.
- The source is per-length Joule heat from a **longitudinal current**:
  the field E is common to every conductor of a heater, so
  ``q_i = sigma_i E²`` exactly and no conduction solve is needed.

Units: every public argument is SI except lengths, which are in µm like
the mesh (areas in µm², heater length in µm), and transient times, which
are in µs. Internally the weak form is assembled in a µm-consistent
system — k in W/(µm·K), q in W/µm³, h in W/(µm²·K), rho*c_p in J/(µm³·K),
time in s — so dT comes out in K. That conversion happens once, in
:class:`HeatOperator`, and nowhere else.

scikit-fem is imported lazily so ``import gsim.thermal`` works without
it; the solve paths raise an ``ImportError`` naming the extra.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gsim.common.optional import require_module

if TYPE_CHECKING:
    from skfem import MeshTri

_UM2_TO_M2 = 1e-12
# SI -> µm-consistent weak-form units (see module docstring), applied once.
_K_SI_TO_UM = 1e-6  # W/(m·K) -> W/(µm·K)
_Q_SI_TO_UM = 1e-18  # W/m³ -> W/µm³
_H_SI_TO_UM = 1e-12  # W/(m²·K) -> W/(µm²·K)
_RHOCP_SI_TO_UM = 1e-18  # J/(m³·K) -> J/(µm³·K)


def _skfem() -> Any:
    """Import scikit-fem, naming the optional extra when it is missing."""
    return require_module(
        "skfem",
        extra="femwell",
        hint=(
            "scikit-fem is required for the gsim.thermal solver but is not "
            "installed. Install the optional extra: pip install 'gsim[femwell]' "
            "(or: pip install scikit-fem)."
        ),
    )


@dataclass(frozen=True)
class JouleSource:
    """Per-element Joule heat of a longitudinal-current heater.

    Attributes:
        heat_density: Volumetric heat per element (W/m³); zero outside the
            heater conductors.
        field: Longitudinal electric field common to all heater
            conductors (V/m).
        resistance_per_length: Heater resistance per unit length
            R' = 1 / sum sigma_i A_i (ohm/m).
    """

    heat_density: NDArray[np.float64]
    field: float
    resistance_per_length: float


def _element_array(name: str, values: ArrayLike, n: int | None) -> NDArray:
    """Return ``values`` as a 1D per-element array, checking its length."""
    arr = np.asarray(values)
    if arr.ndim != 1 or (n is not None and arr.shape[0] != n):
        expected = f"({n},)" if n is not None else "a 1D array"
        raise ValueError(
            f"{name} must have shape {expected} (one value per element), "
            f"got shape {arr.shape}"
        )
    return arr


def joule_heat_density(
    conductivity_S_per_m: ArrayLike,
    heater_mask: ArrayLike,
    element_area_um2: ArrayLike,
    *,
    power_per_length_W_per_m: float,
) -> JouleSource:
    """Spread a heater's power per length over its conductors.

    The current flows along the waveguide, out of the cross-section, so
    every heater conductor sees the same field E and carries
    ``J_i = sigma_i E``. With ``G' = sum sigma_i A_i`` over the heater elements,
    ``P/L = G' E²``, hence ``E = sqrt((P/L) / G')`` and
    ``q_i = sigma_i E²``: parallel strips split the power in proportion to
    their conductance.

    Args:
        conductivity_S_per_m: Electrical conductivity per element (S/m).
            Only elements in ``heater_mask`` are used.
        heater_mask: Boolean per element, True on the heater conductors.
        element_area_um2: Area of each element (µm²).
        power_per_length_W_per_m: Dissipated power per unit heater
            length (W/m), >= 0.

    Returns:
        The per-element heat density, the common field and the
        resistance per length.

    Raises:
        ValueError: On shape mismatches, a negative power or area, or a
            heater with no positive conductance.
    """
    sigma = _element_array("conductivity_S_per_m", conductivity_S_per_m, None)
    n = sigma.shape[0]
    mask = _element_array("heater_mask", heater_mask, n).astype(bool)
    area = _element_array("element_area_um2", element_area_um2, n).astype(float)
    if not np.isfinite(power_per_length_W_per_m) or power_per_length_W_per_m < 0:
        raise ValueError(
            f"power_per_length_W_per_m must be finite and >= 0, "
            f"got {power_per_length_W_per_m}"
        )
    if np.any(area < 0) or np.any(sigma[mask] < 0):
        raise ValueError("element areas and heater conductivities must be >= 0")
    conductance = float(np.sum(sigma[mask] * area[mask] * _UM2_TO_M2))
    if not conductance > 0:
        raise ValueError(
            "heater has no conductance: the heater mask selects no element "
            "with positive conductivity and area"
        )
    field = float(np.sqrt(power_per_length_W_per_m / conductance))
    q = np.where(mask, sigma * field**2, 0.0).astype(float)
    return JouleSource(
        heat_density=q,
        field=field,
        resistance_per_length=1.0 / conductance,
    )


def drive_power_per_length(
    resistance_per_length_ohm_per_m: float,
    *,
    power_per_length_W_per_m: float | None = None,
    power_W: float | None = None,
    voltage_V: float | None = None,
    current_A: float | None = None,
    length_um: float | None = None,
) -> float:
    """Convert one heater drive into dissipated power per length (W/m).

    With R = R'·L: a total power gives ``P/L``; a voltage gives
    ``V² / (R' L²)``; a current gives ``I² R'`` (independent of L).

    Args:
        resistance_per_length_ohm_per_m: Heater R' (ohm/m), e.g. from
            :func:`joule_heat_density`.
        power_per_length_W_per_m: Drive given directly as P/L (W/m).
        power_W: Total dissipated power (W); needs ``length_um``.
        voltage_V: Voltage across the heater (V); needs ``length_um``.
        current_A: Current through the heater (A).
        length_um: Heater length (µm).

    Returns:
        Power per unit length (W/m).

    Raises:
        ValueError: Unless exactly one drive is given, or when a drive
            that needs ``length_um`` lacks a positive one.
    """
    drives = {
        "power_per_length_W_per_m": power_per_length_W_per_m,
        "power_W": power_W,
        "voltage_V": voltage_V,
        "current_A": current_A,
    }
    given = [name for name, value in drives.items() if value is not None]
    if len(given) != 1:
        raise ValueError(
            f"give exactly one drive out of {sorted(drives)}, got {given or 'none'}"
        )
    if not resistance_per_length_ohm_per_m > 0:
        raise ValueError("resistance_per_length_ohm_per_m must be > 0")
    if power_per_length_W_per_m is not None:
        return float(power_per_length_W_per_m)
    if current_A is not None:
        return float(current_A**2 * resistance_per_length_ohm_per_m)
    if length_um is None or not length_um > 0:
        raise ValueError(f"drive {given[0]} needs a positive length_um")
    length_m = length_um * 1e-6
    if power_W is not None:
        return float(power_W / length_m)
    return float(
        cast("float", voltage_V) ** 2 / (resistance_per_length_ohm_per_m * length_m**2)
    )


# --- Steady solve -------------------------------------------------------------


def _factorise(matrix: Any) -> Any:
    """Sparse LU factorisation with the ordering that suits these meshes."""
    from scipy.sparse.linalg import splu

    return splu(matrix, permc_spec="MMD_AT_PLUS_A")


def _group_names(groups: str | Sequence[str]) -> tuple[str, ...]:
    """Normalise one facet group name or several to a tuple."""
    return (groups,) if isinstance(groups, str) else tuple(groups)


def _facets_of(
    boundary_facets: Mapping[str, ArrayLike], name: str, n_facets: int, role: str
) -> NDArray[np.int64]:
    """Look up and validate the facet indices of one named group."""
    if name not in boundary_facets:
        raise ValueError(
            f"{role} facet group {name!r} not in boundary_facets "
            f"(available: {sorted(boundary_facets)})"
        )
    idx = np.asarray(boundary_facets[name])
    if idx.ndim != 1 or (idx.size and not np.issubdtype(idx.dtype, np.integer)):
        raise ValueError(f"facet group {name!r} must be a 1D integer index array")
    idx = idx.astype(np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= n_facets):
        raise ValueError(
            f"facet group {name!r} has indices outside [0, {n_facets}) "
            "(indices into mesh.facets)"
        )
    return idx


class HeatOperator:
    """Factorised steady heat operator, reusable across many sources.

    Assembles ``-div(k grad dT)`` with P1 Lagrange elements on a µm mesh,
    dT = 0 on the ``sink`` facets (eliminated), a Robin term ``h dT``
    on each ``convective`` group and adiabatic facets everywhere else,
    then factorises it once, on the first :meth:`solve`, with
    ``splu(permc_spec="MMD_AT_PLUS_A")``.
    The problem is linear, so :meth:`solve` costs only a load assembly
    and two triangular solves; dT scales linearly with q.

    Args:
        mesh: Triangle mesh, coordinates in µm.
        conductivity_W_per_mK: Thermal conductivity per element
            (W/(m·K)), > 0.
        boundary_facets: Named facet groups, indices into ``mesh.facets``.
        sink: Facet group name(s) held at dT = 0 (heat sink).
        convective: Facet group name -> heat-transfer coefficient h
            (W/(m²·K), >= 0) of a Robin condition towards ambient.

    Raises:
        ValueError: On a shape mismatch, a non-positive or non-finite
            conductivity, an unknown or out-of-range facet group, a
            negative h, or when there is no heat path to ambient (no
            sink facets and no positive h), which leaves dT undefined.
    """

    def __init__(
        self,
        mesh: MeshTri,
        conductivity_W_per_mK: ArrayLike,
        *,
        boundary_facets: Mapping[str, ArrayLike],
        sink: str | Sequence[str] = "bottom",
        convective: Mapping[str, float] | None = None,
    ) -> None:
        """Assemble the operator; see the class docstring for arguments."""
        skfem = _skfem()

        if mesh.t.shape[0] != 3 or mesh.p.shape[0] != 2:
            raise ValueError("mesh must be a 2D triangle mesh (skfem.MeshTri)")
        self.mesh = mesh
        self.n_elements = int(mesh.t.shape[1])
        self.n_nodes = int(mesh.p.shape[1])
        k = _element_array(
            "conductivity_W_per_mK", conductivity_W_per_mK, self.n_elements
        ).astype(float)
        if not np.all(np.isfinite(k)) or np.any(k <= 0):
            raise ValueError("conductivity_W_per_mK must be finite and > 0")

        n_facets = int(mesh.facets.shape[1])
        sink_facets = [
            _facets_of(boundary_facets, name, n_facets, "sink")
            for name in _group_names(sink)
        ]
        robin = {
            name: (_facets_of(boundary_facets, name, n_facets, "convective"), h)
            for name, h in (convective or {}).items()
        }
        for name, (_, h) in robin.items():
            if not np.isfinite(h) or h < 0:
                raise ValueError(f"convective h for {name!r} must be finite and >= 0")
        sink_idx = (
            np.unique(np.concatenate(sink_facets))
            if sink_facets
            else np.zeros(0, np.int64)
        )
        if sink_idx.size == 0 and not any(h > 0 and f.size for f, h in robin.values()):
            raise ValueError(
                "no heat path to ambient: give a non-empty sink facet group or a "
                "convective group with h > 0, otherwise the steady dT is undefined"
            )

        p1 = skfem.ElementTriP1()
        self._basis = skfem.Basis(mesh, p1)
        p0_basis = self._basis.with_element(skfem.ElementTriP0())

        @skfem.BilinearForm
        def _diffusion(u, v, w):
            return w.k * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])

        @skfem.BilinearForm
        def _mass(u, v, w):
            return w.c * u * v

        @skfem.BilinearForm
        def _p0_to_p1(u, v, _):
            return u * v

        self._mass_form = _mass
        self._p0_basis = p0_basis
        stiffness = _diffusion.assemble(
            self._basis, k=p0_basis.interpolate(k * _K_SI_TO_UM)
        )
        for facets, h in robin.values():
            if h > 0 and facets.size:
                fbasis = skfem.FacetBasis(mesh, p1, facets=facets)
                stiffness = stiffness + _mass.assemble(fbasis, c=h * _H_SI_TO_UM)
        self.stiffness = stiffness.tocsr()
        # Maps per-element q (W/µm³) to the P1 load vector: integral of q phi_j dA.
        self._load = skfem.asm(_p0_to_p1, p0_basis, self._basis).tocsr()

        # P1 dofs are the mesh vertices, so the sink nodes are the facet ends.
        self.sink_nodes = np.unique(mesh.facets[:, sink_idx].ravel())
        free = np.ones(self.n_nodes, bool)
        free[self.sink_nodes] = False
        self.free_nodes = np.flatnonzero(free)
        self._lu: Any = None

    def restrict(self, matrix: Any) -> Any:
        """Restrict a nodal matrix to the free (non-sink) nodes.

        Args:
            matrix: Sparse ``(n_nodes, n_nodes)`` matrix.

        Returns:
            The free-free block as CSC, ready to factorise.
        """
        return matrix[self.free_nodes][:, self.free_nodes].tocsc()

    def mass_matrix(self, heat_capacity_J_per_m3K: ArrayLike) -> Any:
        """Assemble the consistent P1 heat-capacity matrix.

        The matrix is ``integral of rho*c_p phi_i phi_j``.

        Args:
            heat_capacity_J_per_m3K: Volumetric heat capacity rho*c_p per
                element (J/(m³·K)), > 0.

        Returns:
            Sparse CSR matrix in the internal µm units (J/(µm·K)).

        Raises:
            ValueError: On a shape mismatch or a non-positive value.
        """
        rho_cp = _element_array(
            "heat_capacity_J_per_m3K", heat_capacity_J_per_m3K, self.n_elements
        ).astype(float)
        if not np.all(np.isfinite(rho_cp)) or np.any(rho_cp <= 0):
            raise ValueError("heat_capacity_J_per_m3K must be finite and > 0")
        return self._mass_form.assemble(
            self._basis, c=self._p0_basis.interpolate(rho_cp * _RHOCP_SI_TO_UM)
        ).tocsr()

    def load_vector(self, heat_W_per_m3: ArrayLike) -> NDArray[np.float64]:
        """Assemble the nodal load ``integral of q phi_j`` (W/µm) of a per-element q.

        Args:
            heat_W_per_m3: Volumetric heat per element (W/m³).

        Returns:
            The P1 load vector in the internal µm units.

        Raises:
            ValueError: On a shape mismatch or non-finite values.
        """
        q = _element_array("heat_W_per_m3", heat_W_per_m3, self.n_elements)
        q = q.astype(float)
        if not np.all(np.isfinite(q)):
            raise ValueError("heat_W_per_m3 must be finite")
        return self._load @ (q * _Q_SI_TO_UM)

    def solve(self, heat_W_per_m3: ArrayLike) -> NDArray[np.float64]:
        """Steady temperature rise for one heat source.

        Args:
            heat_W_per_m3: Volumetric heat per element (W/m³), e.g.
                :attr:`JouleSource.heat_density`.

        Returns:
            Nodal dT (K), aligned with the columns of ``mesh.p``; exactly
            zero on the sink nodes.
        """
        load = self.load_vector(heat_W_per_m3)
        if self._lu is None:
            self._lu = _factorise(self.restrict(self.stiffness))
        rise = np.zeros(self.n_nodes)
        rise[self.free_nodes] = self._lu.solve(load[self.free_nodes])
        return rise


def solve_steady(
    mesh: MeshTri,
    conductivity_W_per_mK: ArrayLike,
    heat_W_per_m3: ArrayLike,
    *,
    boundary_facets: Mapping[str, ArrayLike],
    sink: str | Sequence[str] = "bottom",
    convective: Mapping[str, float] | None = None,
) -> NDArray[np.float64]:
    """Steady temperature rise of a 2D cross-section (one-shot).

    Builds a :class:`HeatOperator` and solves once; build the operator
    yourself to reuse its factorisation across many sources or powers.

    Args:
        mesh: Triangle mesh, coordinates in µm.
        conductivity_W_per_mK: Thermal conductivity per element (W/(m·K)).
        heat_W_per_m3: Volumetric heat per element (W/m³).
        boundary_facets: Named facet groups, indices into ``mesh.facets``.
        sink: Facet group name(s) held at dT = 0.
        convective: Facet group name -> h (W/(m²·K)) of a Robin condition.

    Returns:
        Nodal dT (K), aligned with the columns of ``mesh.p``.
    """
    operator = HeatOperator(
        mesh,
        conductivity_W_per_mK,
        boundary_facets=boundary_facets,
        sink=sink,
        convective=convective,
    )
    return operator.solve(heat_W_per_m3)


# --- Transient solve ----------------------------------------------------------


@dataclass(frozen=True)
class TransientHistory:
    """Temperature-rise history of a step-on transient.

    Attributes:
        time_us: Sample times (µs), ``steps + 1`` values from 0 to the
            duration; index 0 is the initial state dT = 0.
        values: dT history (K): ``(steps + 1, n_nodes)`` nodal values
            when no ``sample`` was given, else the sampled values
            stacked along the first axis.
    """

    time_us: NDArray[np.float64]
    values: NDArray[np.float64]


def solve_transient(
    mesh: MeshTri,
    conductivity_W_per_mK: ArrayLike,
    heat_capacity_J_per_m3K: ArrayLike,
    heat_W_per_m3: ArrayLike,
    *,
    duration_us: float,
    steps: int,
    theta: float = 1.0,
    boundary_facets: Mapping[str, ArrayLike],
    sink: str | Sequence[str] = "bottom",
    convective: Mapping[str, float] | None = None,
    sample: Callable[[NDArray[np.float64]], ArrayLike] | ArrayLike | None = None,
) -> TransientHistory:
    """Temperature-rise response to a heat source stepped on at t = 0.

    Theta-scheme on ``M d(dT)/dt + K dT = F`` from dT = 0, with F constant for
    t > 0 and a constant step ``dt = duration / steps``, so one
    ``splu`` factorisation of ``M/dt + theta*K`` serves every step:
    ``(M/dt + theta*K) dT_next = (M/dt - (1-theta)K) dT + F``. Boundary conditions
    are those of :class:`HeatOperator`. In this linear model the fall
    response after switching off from steady state is the complement of
    the rise, so one run gives both.

    Args:
        mesh: Triangle mesh, coordinates in µm.
        conductivity_W_per_mK: Thermal conductivity per element (W/(m·K)).
        heat_capacity_J_per_m3K: Volumetric heat capacity rho*c_p per
            element (J/(m³·K)).
        heat_W_per_m3: Volumetric heat per element while on (W/m³).
        duration_us: Simulated time (µs), > 0.
        steps: Number of time steps, >= 1.
        theta: 1.0 for backward Euler (default, robust for a step
            drive), 0.5 for Crank-Nicolson (second order, may ring at
            the step); must lie in [0.5, 1].
        boundary_facets: Named facet groups, indices into ``mesh.facets``.
        sink: Facet group name(s) held at dT = 0.
        convective: Facet group name -> h (W/(m²·K)) of a Robin condition.
        sample: What to record per step. ``None`` keeps the full nodal
            field (memory ``(steps + 1) x n_nodes``); an array of nodal
            weights records ``weights @ dT``; a callable records
            ``sample(dT)``.

    Returns:
        The sample times and the recorded dT history.

    Raises:
        ValueError: On invalid materials, boundaries (see
            :class:`HeatOperator`), duration, steps, theta or sample
            weights.
    """
    if not np.isfinite(duration_us) or duration_us <= 0:
        raise ValueError(f"duration_us must be finite and > 0, got {duration_us}")
    if int(steps) != steps or steps < 1:
        raise ValueError(f"steps must be an integer >= 1, got {steps}")
    if not 0.5 <= theta <= 1.0:
        raise ValueError(f"theta must lie in [0.5, 1] (0.5 = CN, 1 = BE), got {theta}")
    operator = HeatOperator(
        mesh,
        conductivity_W_per_mK,
        boundary_facets=boundary_facets,
        sink=sink,
        convective=convective,
    )
    mass = operator.mass_matrix(heat_capacity_J_per_m3K)
    load = operator.load_vector(heat_W_per_m3)[operator.free_nodes]
    record = _sampler(sample, operator.n_nodes)

    dt_s = duration_us * 1e-6 / steps
    mass_free = operator.restrict(mass) / dt_s
    stiff_free = operator.restrict(operator.stiffness)
    lu = _factorise((mass_free + theta * stiff_free).tocsc())
    explicit = (mass_free - (1.0 - theta) * stiff_free).tocsr()

    rise = np.zeros(operator.n_nodes)
    free = np.zeros(operator.free_nodes.size)
    history = [record(rise)]
    for _ in range(int(steps)):
        free = lu.solve(explicit @ free + load)
        rise[operator.free_nodes] = free
        history.append(record(rise))
    return TransientHistory(
        time_us=np.linspace(0.0, duration_us, int(steps) + 1, dtype=np.float64),
        values=np.asarray(history, dtype=float),
    )


def _sampler(
    sample: Callable[[NDArray[np.float64]], ArrayLike] | ArrayLike | None,
    n_nodes: int,
) -> Callable[[NDArray[np.float64]], Any]:
    """Build the per-step recorder for ``solve_transient``'s ``sample``."""
    if sample is None:
        return lambda rise: rise.copy()
    if callable(sample):
        return lambda rise: np.asarray(sample(rise), dtype=float)
    weights = np.asarray(sample, dtype=float)
    if weights.shape != (n_nodes,):
        raise ValueError(
            f"sample weights must have shape ({n_nodes},) (one per mesh node), "
            f"got {weights.shape}"
        )
    return lambda rise: float(weights @ rise)


def _crossing_time(t: NDArray, s: NDArray, level: float) -> float:
    """First time the normalised step ``s`` reaches ``level``, interpolated."""
    above = np.flatnonzero(s >= level)
    i = int(above[0])
    if i == 0:
        return float(t[0])
    frac = (level - s[i - 1]) / (s[i] - s[i - 1])
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def _normalised_step(t: ArrayLike, y: ArrayLike) -> tuple[NDArray, NDArray]:
    """Validate a step response and rescale it to run from 0 to 1."""
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if t_arr.ndim != 1 or t_arr.shape != y_arr.shape or t_arr.size < 2:
        raise ValueError(
            "t and y must be 1D arrays of the same length >= 2, "
            f"got shapes {t_arr.shape} and {y_arr.shape}"
        )
    span = y_arr[-1] - y_arr[0]
    if span == 0 or not np.isfinite(span):
        raise ValueError("y has no net change between its first and last sample")
    return t_arr, (y_arr - y_arr[0]) / span


def rise_time_10_90(t: ArrayLike, y: ArrayLike) -> float:
    """10-90 % rise time of a step response.

    Levels are taken relative to the first and last samples (the last is
    assumed settled); each crossing is the first one, linearly
    interpolated between samples.

    Args:
        t: Sample times (any unit; the result has the same unit).
        y: Response values, same length as ``t``.

    Returns:
        ``t90 - t10``.

    Raises:
        ValueError: On mismatched shapes or a response with no net change.
    """
    t_arr, s = _normalised_step(t, y)
    return _crossing_time(t_arr, s, 0.9) - _crossing_time(t_arr, s, 0.1)


def time_constant_1e(t: ArrayLike, y: ArrayLike) -> float:
    """1/e time constant of a step response.

    The time, from ``t[0]``, at which the response first covers
    ``1 - 1/e`` (~ 63.2 %) of its change from the first to the last
    sample; equal to tau for a single exponential.

    Args:
        t: Sample times (any unit; the result has the same unit).
        y: Response values, same length as ``t``.

    Returns:
        The 1/e time constant.

    Raises:
        ValueError: On mismatched shapes or a response with no net change.
    """
    t_arr, s = _normalised_step(t, y)
    return _crossing_time(t_arr, s, 1.0 - np.exp(-1.0)) - float(t_arr[0])
