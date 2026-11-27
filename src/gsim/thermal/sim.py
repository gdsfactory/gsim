"""Simulation: the heater phase shifter as a physicist sets it up.

A 2D cross-section of an infinitely long heater over one or more
waveguides. The heater's power heats the cross-section through the
oxide into the substrate (the heat sink at the bottom of the Modelled
depth); the temperature rise shifts the waveguides' effective index
through dn/dT, and the phase follows over the heater length::

    from gsim import thermal

    sim = thermal.Simulation()
    sim.geometry(component, stack=stack)
    sim.domain(plane="x=0", window=(-40.0, 40.0))
    sim.heater(layers="heater", length_um=320.0)
    sim.waveguides.add(name="arm", position_um=0.0)
    sim.waveguides.add(name="neighbour", position_um=10.0)
    result = sim.solve(wavelength_um=1.55)
    result.p_pi_mw, result.crosstalk_rad["neighbour"]
    step = sim.solve_transient(power_mw=20.0, duration_us=100.0)
    print(result.report(step))

Physics (v1): steady and transient heat conduction with constant k,
rho*c_p and dn/dT, so everything is linear in heater power; the heater
current runs along the waveguide (``q_i = sigma_i E^2`` over its
conductors); no feedback of temperature into carriers or loss.

Units: lengths in um, power in mW (total over the heater length),
temperature rise in K, time in us, resistance in ohm, voltage in V,
phase in rad.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from gsim.common.config import Concern, Same, component_digest, fingerprint
from gsim.common.cross_section import parse_plane_spec
from gsim.common.stack.materials import (
    MaterialProperties,
    ThermalProperties,
    _get_ci,
    get_material_properties,
    thermal_properties,
)
from gsim.femwell.runtime import require_femwell, require_skfem
from gsim.thermal.api import (
    STEADY_FIELDS,
    Boundary,
    Domain,
    Geometry,
    Heater,
    Materials,
    Solver,
    Waveguide,
    Waveguides,
)
from gsim.thermal.mesh import CrossSectionMesh, _drawn_rects, build_cross_section_mesh
from gsim.thermal.optical import (
    ModeShift,
    element_centroids,
    interpolation_weights,
    sample_temperature,
    solve_thermo_optic_shift,
)
from gsim.thermal.results import (
    _UM_TO_M,
    StepResponse,
    ThermalResult,
    power_per_length,
)
from gsim.thermal.solver import (
    HeatOperator,
    JouleSource,
    graded_times,
    joule_heat_density,
    rise_time_10_90,
    solve_transient,
    time_constant_1e,
)

__all__ = ["Simulation"]

# Optical Window around a waveguide: enough oxide for a 500 x 220 nm Si
# strip's TE0 and TM0 tails to fall below MAX_BOUNDARY_RATIO at its edges
# (TM0 reaches further above and below the core than TE0).
_OPTICAL_MARGIN_LATERAL_UM = 1.5
_OPTICAL_MARGIN_VERTICAL_UM = 2.0
_FEATURE_DIVISIONS = 4.0
_COARSE_DIVISIONS = 10.0
_SETTLED_FRACTION = 0.99
# Largest backward Euler delay, as a fraction of tau_1e, solved without a warning.
_MAX_TIME_STEP_BIAS = 0.02
_EPS0 = 8.8541878128e-12
_C0 = 299792458.0

# Depth below z = 0 of the silicon substrate taken from the active PDK.
_PDK_SUBSTRATE_DEPTH_UM = 100.0
# Warnings point at the user's call, past every frame of this package.
_SKIP_FRAMES = (str(Path(__file__).parent),)

# The override field for a ThermalProperties field, where they differ.
_OVERRIDE_NAME = {"electrical_conductivity": "conductivity"}

_CONCERNS = (
    "geometry",
    "domain",
    "heater",
    "waveguides",
    "materials",
    "boundary",
    "solver",
)


@dataclass(frozen=True)
class _Strip:
    """The drawn rectangle of a waveguide in the cut (um)."""

    layer: str
    material: str
    h0: float
    h1: float
    v0: float
    v1: float

    @property
    def centre(self) -> tuple[float, float]:
        """Centre of the rectangle, ``(in-plane, z)``."""
        return (0.5 * (self.h0 + self.h1), 0.5 * (self.v0 + self.v1))


@dataclass(frozen=True)
class _Thermal:
    """The thermal mesh, its heat operator and the rise per unit drive."""

    mesh: CrossSectionMesh
    conductivity: NDArray[np.float64]
    convective: dict[str, float]
    joule: JouleSource
    unit_rise: NDArray[np.float64]


@dataclass(frozen=True)
class _Optical:
    """One waveguide's optical Window, base permittivity and dn/dT."""

    strip: _Strip
    mesh: CrossSectionMesh
    epsilon0: NDArray[np.complex128]
    dn_dt: NDArray[np.float64]
    unit_rise: NDArray[np.float64]
    bounded_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Steady:
    """A steady solve and what the transient reuses from it."""

    result: ThermalResult
    shift: ModeShift
    slope: float


class Simulation:
    """A 2D thermal cross-section of a heater over waveguides.

    Example:
        >>> from gsim import thermal
        >>> sim = thermal.Simulation()
        >>> sim.geometry(component, stack=stack)
        >>> sim.domain(plane="x=160", window=(-30.0, 30.0))
        >>> sim.heater(layers="heater", length_um=320.0)
        >>> sim.waveguides.add(name="arm", position_um=0.0)
        >>> result = sim.solve(wavelength_um=1.55)
        >>> step = sim.solve_transient(power_mw=result.p_pi_mw, duration_us=300.0)

    :meth:`solve` gives the steady figures (P_pi, crosstalk, resistance,
    V_pi, temperature field) and :meth:`solve_transient` the switching
    time; :meth:`mesh` builds the thermal mesh alone, to inspect it
    before a solve. The meshes are built on the first call and kept, with
    the steady solve, until an input they depend on changes: a concern
    can be edited at any time and the next call starts again from it.

    Units: lengths in um, power in mW, temperature rise in K, time in us.
    """

    geometry = Concern(Geometry)
    domain = Concern(Domain)
    heater = Concern(Heater)
    waveguides = Concern(Waveguides)
    materials = Concern(Materials)
    boundary = Concern(Boundary)
    solver = Concern(Solver)

    def __init__(
        self,
        *,
        geometry: Geometry | Mapping[str, Any] | None = None,
        domain: Domain | Mapping[str, Any] | None = None,
        heater: Heater | Mapping[str, Any] | None = None,
        waveguides: Waveguides | Iterable[Waveguide | Mapping[str, Any]] | None = None,
        materials: Materials | Mapping[str, Any] | None = None,
        boundary: Boundary | Mapping[str, Any] | None = None,
        solver: Solver | Mapping[str, Any] | None = None,
    ) -> None:
        """Create a simulation from its concerns, as models or mappings."""
        self.geometry = geometry
        self.domain = domain
        self.heater = heater
        self.waveguides = waveguides
        self.materials = materials
        self.boundary = boundary
        self.solver = solver
        self._cache: dict[tuple[Any, ...], Any] = {}
        self._warned: set[str] = set()
        self._fingerprint: Any = None
        self._resolved_stack: Any = None
        self._verbose = False

    def __repr__(self) -> str:
        """Show the concerns."""
        concerns = ", ".join(f"{name}={getattr(self, name)!r}" for name in _CONCERNS)
        return f"Simulation({concerns})"

    # ------------------------------------------------------------------ mesh

    def mesh(self, *, verbose: bool = False) -> CrossSectionMesh:
        """Build the thermal mesh of the cut, or return the kept one.

        The solves build the same mesh on their own; call this to look at
        it first, e.g. ``sim.mesh().plot()``. Resolution comes from the
        ``solver`` concern.

        Args:
            verbose: Print gmsh's meshing progress.

        Returns:
            The :class:`~gsim.thermal.CrossSectionMesh` of the thermal
            Window.
        """
        require_skfem()
        self._sync()
        with self._verbosity(verbose):
            return self._thermal_mesh()

    # ------------------------------------------------------------------ solves

    def solve(
        self, *, wavelength_um: float = 1.55, verbose: bool = False
    ) -> ThermalResult:
        """Steady heater solve: P_pi, crosstalk, resistance and temperature.

        One heat solve at a unit power per length gives the temperature
        rise per watt per metre (the model is linear); each waveguide's
        Mode is then re-solved with the heated index on its optical
        Window at a power near P_pi, giving the effective-index shift
        per unit power.

        Args:
            wavelength_um: Vacuum wavelength in um.
            verbose: Print gmsh's meshing progress (silent by default).

        Returns:
            The :class:`~gsim.thermal.ThermalResult`.
        """
        wavelength = _positive(wavelength_um, "wavelength_um", "um")
        self._sync()
        with self._verbosity(verbose):
            return self._steady(wavelength).result

    def solve_transient(
        self,
        *,
        power_mw: float,
        duration_us: float,
        wavelength_um: float = 1.55,
        verbose: bool = False,
    ) -> StepResponse:
        """Switch-on response to a heater power stepped on at t = 0.

        The phase over time comes from first-order perturbation of the
        phase shifter's Mode by the temperature history, scaled so that
        its steady value matches the steady Mode re-solve. The ``solver``
        concern sets the number of time steps; they are graded
        (:func:`~gsim.thermal.graded_times`): short after switch-on, where
        the response changes fastest, and doubling in length seven times
        towards ``duration_us``. The default backward Euler scheme reads
        ``tau_us`` and ``tau_1e_us`` long by about half the step in use at
        that time; the result warns when that exceeds 2 % of ``tau_1e_us``
        (``theta=0.5`` removes the bias).

        Args:
            power_mw: Total heater power stepped on (mW).
            duration_us: Simulated time (us); long enough to settle (the
                result warns otherwise).
            wavelength_um: Vacuum wavelength in um.
            verbose: Print gmsh's meshing progress (silent by default).

        Returns:
            The :class:`~gsim.thermal.StepResponse`.
        """
        power = _positive(power_mw, "power_mw", "mW")
        duration = _positive(duration_us, "duration_us", "us")
        wavelength = _positive(wavelength_um, "wavelength_um", "um")
        self._sync()
        with self._verbosity(verbose):
            steady = self._steady(wavelength)
            thermal = self._thermal()
        solver = self.solver
        result = steady.result
        shifter = result.phase_shifter
        optical = self._optical(self.waveguides[shifter], thermal, result.wavelength_um)

        weights = _first_order_weights(steady.shift.mode0, optical)
        n_dofs = thermal.unit_rise.size
        shift_weights = _dof_weights(
            thermal.mesh, n_dofs, element_centroids(optical.mesh), weights
        )
        strips = self._strips()
        probes = [
            _dof_weights(
                thermal.mesh, n_dofs, np.array([[c] for c in strips[n].centre]), [1.0]
            )
            for n in strips
        ]
        sampler = np.vstack([shift_weights, *probes])

        drive = power_per_length(power, result.length_um)
        history = solve_transient(
            thermal.mesh.mesh,
            thermal.conductivity,
            self._heat_capacity(thermal.mesh),
            thermal.joule.heat_density * drive,
            time_us=graded_times(duration, solver.steps),
            theta=solver.theta,
            boundary_facets=thermal.mesh.boundary_facets,
            sink="bottom",
            convective=thermal.convective,
            sample=lambda rise: sampler @ rise,
            degree=solver.degree,
        )
        first_order_slope = float(shift_weights @ thermal.unit_rise)
        scale = steady.slope / first_order_slope
        phase = np.asarray(
            2.0
            * math.pi
            * history.values[:, 0]
            * scale
            * result.length_um
            / result.wavelength_um,
            dtype=np.float64,
        )
        steady_phase = result.phase(power)[shifter]
        if abs(phase[-1]) < _SETTLED_FRACTION * abs(steady_phase):
            warnings.warn(
                f"The phase reaches {phase[-1] / steady_phase:.1%} of its steady "
                f"value after {duration:g} us: increase duration so the step "
                "settles, or tau is underestimated.",
                UserWarning,
                stacklevel=2,
            )
        tau_1e = time_constant_1e(history.time_us, phase)
        self._warn_time_step_bias(history.time_us, tau_1e)
        return StepResponse(
            time_us=history.time_us,
            power_mw=power,
            phase_shifter=shifter,
            phase_rad=phase,
            steady_phase_rad=steady_phase,
            temperature_rise_k={
                name: history.values[:, i + 1] for i, name in enumerate(strips)
            },
            tau_us=rise_time_10_90(history.time_us, phase),
            tau_1e_us=tau_1e,
        )

    def _warn_time_step_bias(self, time_us: NDArray[np.float64], tau_1e: float) -> None:
        """Warn when backward Euler's delay of half a step biases tau_1e."""
        theta = self.solver.theta
        step = float(np.diff(time_us)[np.searchsorted(time_us, tau_1e) - 1])
        bias = (theta - 0.5) * step
        if bias > _MAX_TIME_STEP_BIAS * tau_1e:
            warnings.warn(
                f"The time step near tau_1e ({step:.3g} us) delays the step "
                f"response by about {bias:.2g} us, {bias / tau_1e:.1%} of "
                f"tau_1e: raise solver.time_steps, shorten duration_us, or set "
                "solver.theta=0.5.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------ internals

    @contextmanager
    def _verbosity(self, verbose: bool) -> Generator[None]:
        """Let the meshes built inside print gmsh's progress."""
        self._verbose = bool(verbose)
        try:
            yield
        finally:
            self._verbose = False

    def _resolve_stack(self) -> Any:
        """The geometry's stack as a LayerStack, read from a path or the PDK."""
        from gsim.common.stack import LayerStack, get_stack

        stack = self.geometry.stack
        if isinstance(stack, LayerStack):
            return stack
        if stack is None:
            return get_stack(
                include_substrate=True, substrate_thickness=_PDK_SUBSTRATE_DEPTH_UM
            )
        return get_stack(yaml_path=stack)

    def _steady_inputs(self) -> Any:
        """Snapshot of everything the meshes and the steady solve use."""
        component = self.geometry.component
        return fingerprint(
            (
                Same(component),
                component_digest(component),
                self._resolved_stack,
                self.domain,
                self.heater,
                list(self.waveguides),
                self.materials,
                self.boundary,
                {name: getattr(self.solver, name) for name in STEADY_FIELDS},
            )
        )

    def _sync(self) -> None:
        """Check the setup; forget the kept solves if an input changed."""
        self._require_setup()
        self._resolved_stack = self._resolve_stack()
        inputs = self._steady_inputs()
        if inputs != self._fingerprint:
            self._cache.clear()
            self._warned.clear()
            self._fingerprint = inputs

    def _require_setup(self) -> None:
        """Raise naming the first concern a solve still needs."""
        missing = [
            (self.geometry.component is None, "sim.geometry(component, stack=stack)"),
            (self.domain.plane is None, "sim.domain(plane='x=<um>')"),
            (not self.heater.layers, "sim.heater(layers=..., length_um=...)"),
            (self.heater.length_um is None, "sim.heater(length_um=...)"),
            (
                not self.waveguides,
                "sim.waveguides.add(name=..., position_um=...) -- P_pi needs a "
                "phase-shifter waveguide",
            ),
        ]
        for absent, call in missing:
            if absent:
                raise ValueError(f"The simulation is not set up: call {call} first.")

    @property
    def _component(self) -> Any:
        """The component, once :meth:`_require_setup` has seen it."""
        return self.geometry.component

    @property
    def _stack(self) -> Any:
        """The resolved stack, once :meth:`_sync` has read it."""
        return self._resolved_stack

    def _section(self) -> list[Any]:
        """The drawn rectangles of the cut, one per layer interval."""
        axis, value = parse_plane_spec(str(self.domain.plane))
        return _drawn_rects(self._component, self._stack, axis, value)

    def _locate_strip(self, spec: Waveguide) -> _Strip:
        """The drawn rectangle of a waveguide, found by in-plane position."""
        rects = self._section()
        at = [r for r in rects if r.h0 <= spec.position_um <= r.h1]
        if spec.layer is not None:
            hits = [r for r in at if r.group == spec.layer]
            if not hits:
                raise ValueError(
                    f"Waveguide {spec.name!r}: layer {spec.layer!r} is not drawn at "
                    f"{spec.position_um:g} um in the cut {self.domain.plane!r}. "
                    "Drawn there: "
                    f"{sorted({r.group for r in at}) or 'nothing'}."
                )
        else:
            hits = [r for r in at if r.group not in self.heater.layers]
            if not hits:
                raise ValueError(
                    f"Waveguide {spec.name!r}: no drawn non-heater layer at "
                    f"{spec.position_um:g} um in the cut {self.domain.plane!r}. "
                    "Waveguide cores drawn in the cut: "
                    + _describe(r for r in rects if r.group not in self.heater.layers)
                    + "."
                )
            if len(hits) > 1:
                raise ValueError(
                    f"Waveguide {spec.name!r}: several layers are drawn at "
                    f"{spec.position_um:g} um ({_describe(hits)}); name the core "
                    "with sim.waveguides.add(..., layer=...)."
                )
        r = hits[0]
        return _Strip(r.group, r.material, r.h0, r.h1, r.v0, r.v1)

    def _strips(self) -> dict[str, _Strip]:
        """Every waveguide's drawn rectangle, by name, in the order added."""
        return {wg.name: self._locate_strip(wg) for wg in self.waveguides}

    def _overrides(self) -> dict[str, dict[str, float]]:
        """Per-run material overrides, as ``thermal_properties`` takes them."""
        return self.materials.as_records()

    def _thermal_props(
        self, material: str, region: str, required: tuple[str, ...]
    ) -> ThermalProperties:
        """Thermal properties of a Region's material, or a physics error."""
        try:
            return thermal_properties(
                material, self._stack, self._overrides(), required=required
            )
        except ValueError as error:
            hint = ""
            if required:
                fields = ", ".join(
                    f"{_OVERRIDE_NAME.get(n, n)!r}: ..." for n in required
                )
                hint = (
                    f" For this run only: sim.materials[{material!r}] = {{{fields}}}."
                )
            raise ValueError(
                f"Region {region!r} is {material!r}: {error}{hint}"
            ) from error

    def _mesh_sizes(
        self, rects: list[Any], extent: float
    ) -> tuple[dict[str, float], float]:
        """Per-Region and coarse element sizes for the solver's settings.

        The preset scales the defaults; ``max_mesh_size_um`` caps every
        size, ``refined_mesh_size_um`` sets the heater conductors and the
        waveguide cores, and ``region_mesh_size_um`` wins over both.
        """
        solver = self.solver
        f = solver.size_factor
        coarse = f * extent / _COARSE_DIVISIONS
        if solver.max_mesh_size_um is not None:
            coarse = min(coarse, float(solver.max_mesh_size_um))
        sizes: dict[str, float] = {}
        for r in rects:
            own = f * min(r.h1 - r.h0, r.v1 - r.v0) / _FEATURE_DIVISIONS
            sizes[r.group] = min(sizes.get(r.group, coarse), own, coarse)
        if solver.refined_mesh_size_um is not None:
            fine = {*self.heater.layers, *(s.layer for s in self._strips().values())}
            refined = float(solver.refined_mesh_size_um)
            for group in fine & set(sizes):
                sizes[group] = refined
        sizes.update(solver.region_mesh_size_um)
        return sizes, coarse

    def _thermal_mesh(self) -> CrossSectionMesh:
        """The thermal mesh of the cut (kept)."""
        key = ("mesh",)
        if key in self._cache:
            return cast("CrossSectionMesh", self._cache[key])
        rects = self._section()
        drawn = {r.group for r in rects}
        absent = [name for name in self.heater.layers if name not in drawn]
        if len(absent) == len(self.heater.layers):
            raise ValueError(
                f"Heater layer(s) {absent} are not drawn in the cut "
                f"{self.domain.plane!r}; layers drawn there: {sorted(drawn)}."
            )
        if absent:
            warnings.warn(
                f"Heater layer(s) {absent} are not drawn in the cut "
                f"{self.domain.plane!r}; only "
                f"{sorted(set(self.heater.layers) - set(absent))} "
                "carry the heater current there.",
                UserWarning,
                skip_file_prefixes=_SKIP_FRAMES,
            )
        h0 = min(r.h0 for r in rects)
        h1 = max(r.h1 for r in rects)
        window = self.domain.window or (h0 - 20.0, h1 + 20.0)
        self._check_heater_inside(rects, window)
        sizes, coarse = self._mesh_sizes(rects, window[1] - window[0])
        mesh = build_cross_section_mesh(
            self._component,
            self._stack,
            str(self.domain.plane),
            window=window,
            depth=self.domain.depth_um,
            max_size=sizes,
            default_max_size=coarse,
            verbose=self._verbose,
        )
        self._cache[key] = mesh
        return mesh

    def _thermal(self) -> _Thermal:
        """The thermal mesh and its rise per unit power per length (kept)."""
        key = ("thermal",)
        if key in self._cache:
            return cast("_Thermal", self._cache[key])
        mesh = self._thermal_mesh()
        regions = mesh.element_region
        k_by_region: dict[str, float] = {}
        sigma_by_region: dict[str, float] = {}
        for region, material in mesh.region_material.items():
            heater = region in self.heater.layers
            required = ("thermal_conductivity",) + (
                ("electrical_conductivity",) if heater else ()
            )
            props = self._thermal_props(material, region, required)
            k_by_region[region] = float(props.thermal_conductivity or 0.0)
            sigma_by_region[region] = (
                float(props.electrical_conductivity or 0.0) if heater else 0.0
            )
        self._check_heater_conducts(mesh.region_material, sigma_by_region)
        conductivity = np.array([k_by_region[r] for r in regions])
        sigma = np.array([sigma_by_region[r] for r in regions])
        heater_mask = np.isin(regions, self.heater.layers)
        joule = joule_heat_density(
            sigma, heater_mask, _element_areas(mesh), power_per_length_w_per_m=1.0
        )
        convective = self.boundary.convective_edges()
        operator = HeatOperator(
            mesh.mesh,
            conductivity,
            boundary_facets=mesh.boundary_facets,
            sink="bottom",
            convective=convective,
            degree=self.solver.degree,
        )
        thermal = _Thermal(
            mesh=mesh,
            conductivity=conductivity,
            convective=convective,
            joule=joule,
            unit_rise=operator.solve(joule.heat_density),
        )
        self._cache[key] = thermal
        return thermal

    def _check_heater_inside(
        self, rects: list[Any], window: tuple[float, float]
    ) -> None:
        """Raise when the thermal Window cuts off part of a heater conductor.

        The resistance, P_pi and V_pi come from the drawn conductors, so
        a heater rectangle the Window clips would silently change them.
        """
        for r in rects:
            if r.group in self.heater.layers and (r.h0 < window[0] or r.h1 > window[1]):
                raise ValueError(
                    f"Heater layer {r.group!r} is drawn over [{r.h0:g}, {r.h1:g}] um "
                    f"in the cut {self.domain.plane!r}, outside the thermal Window "
                    f"({window[0]:g}, {window[1]:g}) um: the heater's resistance "
                    "and power would be wrong. Widen the Window with "
                    "sim.domain(window=(min, max)) to contain every "
                    "heater conductor."
                )

    def _check_heater_conducts(
        self, region_material: Mapping[str, str], sigma: Mapping[str, float]
    ) -> None:
        """Raise when no heater conductor has a positive conductivity.

        No current would flow, so the heater could dissipate no power.
        """
        heaters = {
            region: material
            for region, material in region_material.items()
            if region in self.heater.layers
        }
        if any(sigma[region] > 0 for region in heaters):
            return
        found = ", ".join(
            f"{region!r} is {material!r} ({sigma[region]:g} S/m)"
            for region, material in heaters.items()
        )
        materials = sorted(set(heaters.values()))
        hint = " or ".join(
            f"sim.materials[{m!r}] = {{'conductivity': ...}}" for m in materials
        )
        raise ValueError(
            f"Heater has no electrical conductivity: {found}, so no current "
            f"flows and no power is dissipated. For this run only: {hint} (S/m)."
        )

    def _check_waveguide_inside(
        self, spec: Waveguide, strip: _Strip, window: tuple[float, float]
    ) -> None:
        """Raise when a waveguide core lies outside the thermal Window."""
        if strip.h0 < window[0] or strip.h1 > window[1]:
            raise ValueError(
                f"Waveguide {spec.name!r}: its core {strip.layer!r} is drawn over "
                f"[{strip.h0:g}, {strip.h1:g}] um in the cut {self.domain.plane!r}, "
                f"outside the thermal Window ({window[0]:g}, {window[1]:g}) um. "
                "Widen the Window with sim.domain(window=(min, max)) to contain "
                "every waveguide core."
            )

    def _optical_window(
        self, spec: Waveguide, strip: _Strip, thermal: _Thermal
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """The optical Window of a waveguide, clipped to the thermal Window."""
        lateral = spec.window or (
            strip.h0 - _OPTICAL_MARGIN_LATERAL_UM,
            strip.h1 + _OPTICAL_MARGIN_LATERAL_UM,
        )
        vertical = spec.z_range or (
            strip.v0 - _OPTICAL_MARGIN_VERTICAL_UM,
            strip.v1 + _OPTICAL_MARGIN_VERTICAL_UM,
        )
        (w0, w1), (z0, z1) = thermal.mesh.window, thermal.mesh.z_range
        return (
            (max(lateral[0], w0), min(lateral[1], w1)),
            (max(vertical[0], z0), min(vertical[1], z1)),
        )

    def _optical(
        self,
        spec: Waveguide,
        thermal: _Thermal,
        wavelength: float,
    ) -> _Optical:
        """A waveguide's optical Window with its permittivity and dn/dT (kept)."""
        key = ("optical", spec.name, wavelength)
        if key in self._cache:
            return cast("_Optical", self._cache[key])
        from gsim.femwell.adapter import epsilon_by_region

        strip = self._locate_strip(spec)
        self._check_waveguide_inside(spec, strip, thermal.mesh.window)
        window, z_range = self._optical_window(spec, strip, thermal)

        def build(z: tuple[float, float]) -> tuple[CrossSectionMesh, dict[str, Any]]:
            """The optical mesh over a vertical extent, and its permittivity."""
            rects = [
                r
                for r in self._section()
                if r.h1 > window[0] and r.h0 < window[1] and r.v1 > z[0]
            ]
            sizes, coarse = self._mesh_sizes(
                rects, max(window[1] - window[0], z[1] - z[0])
            )
            mesh = build_cross_section_mesh(
                self._component,
                self._stack,
                str(self.domain.plane),
                window=window,
                z_range=z,
                max_size=sizes,
                default_max_size=coarse,
                verbose=self._verbose,
            )
            eps = epsilon_by_region(
                mesh.msh_path,
                self._stack,
                wavelength_um=wavelength,
                overrides=self._optical_overrides(mesh),
            )
            return mesh, eps

        mesh, eps = build(z_range)
        rivals = _high_index_beyond_core(mesh, eps, strip)
        bounded_by: tuple[str, ...] = ()
        if rivals and spec.z_range is None:
            # Stop the default extent at the nearest one: a Mode solve that
            # includes it finds that region's own Mode, not the waveguide's.
            below = [face for face, side, _ in rivals if side == "below"]
            above = [face for face, side, _ in rivals if side == "above"]
            z_range = (
                max([z_range[0], *below]),
                min([z_range[1], *above]),
            )
            mesh, eps = build(z_range)
            bounded_by = tuple(sorted({name for _, _, name in rivals}))
        elif rivals:
            names = ", ".join(sorted({name for _, _, name in rivals}))
            warnings.warn(
                f"Waveguide {spec.name!r}: its optical Window (z_range "
                f"{z_range}) includes {names}, at least as high in index as the "
                "core, above or below it. The Mode solve may find that region's "
                "Mode instead of the waveguide's; keep z_range inside the "
                "cladding.",
                UserWarning,
                stacklevel=4,
            )
        core = self._thermal_props(strip.material, strip.layer, ())
        if core.dn_dT is None or core.dn_dT == 0:
            raise ValueError(
                f"Waveguide {spec.name!r}: its core {strip.layer!r} is "
                f"{strip.material!r}, which has no (or a zero) dn_dT, so heat "
                "cannot shift its index. Give it with "
                f"sim.materials[{strip.material!r}] = {{'dn_dT': ...}} (1/K)."
            )
        dn_by_region: dict[str, float] = {}
        missing: list[str] = []
        for region, material in mesh.region_material.items():
            dn = thermal_properties(material, self._stack, self._overrides()).dn_dT
            if dn is None:
                missing.append(material)
            dn_by_region[region] = float(dn or 0.0)
        self._warn_missing_dn_dT(missing)
        optical = _Optical(
            strip=strip,
            mesh=mesh,
            epsilon0=np.array([eps[r] for r in mesh.element_region]),
            dn_dt=np.array([dn_by_region[r] for r in mesh.element_region]),
            unit_rise=sample_temperature(
                thermal.mesh, thermal.unit_rise, element_centroids(mesh)
            ),
            bounded_by=bounded_by,
        )
        self._cache[key] = optical
        return optical

    def _optical_overrides(self, mesh: CrossSectionMesh) -> dict[str, Any]:
        """Material records carrying this run's optical overrides."""
        records: dict[str, Any] = {}
        for material in set(mesh.region_material.values()):
            override = self.materials.get(material)
            if override is None:
                continue
            props = override.properties()
            base = _stack_record(material, self._stack)
            update: dict[str, Any] = dict(props)
            if "permittivity" in props:
                # A given permittivity replaces the dispersion models.
                update["dispersion_models"] = []
            records[material] = base.model_copy(update=update)
        return records

    def _warn_missing_dn_dT(self, materials: list[str]) -> None:  # noqa: N802
        """Warn once per material that has no dn/dT and is taken as 0."""
        new = sorted(set(materials) - self._warned)
        if not new:
            return
        self._warned.update(new)
        warnings.warn(
            f"Material(s) {new} have no dn_dT; their index is taken as "
            "independent of temperature. Set it with sim.materials[name] = "
            "{'dn_dT': ...} if they carry optical field.",
            UserWarning,
            skip_file_prefixes=_SKIP_FRAMES,
        )

    def _heat_capacity(self, mesh: CrossSectionMesh) -> NDArray[np.float64]:
        """Volumetric heat capacity rho*c_p of every element (J/m^3/K)."""
        by_region = {}
        for region, material in mesh.region_material.items():
            props = self._thermal_props(
                material, region, ("specific_heat", "mass_density")
            )
            by_region[region] = float(props.specific_heat or 0.0) * float(
                props.mass_density or 0.0
            )
        return np.array([by_region[r] for r in mesh.element_region])

    def _steady(self, wavelength: float) -> _Steady:
        """The steady solve, kept per wavelength."""
        key = ("steady", wavelength)
        if key in self._cache:
            return cast("_Steady", self._cache[key])
        require_skfem()
        require_femwell()
        length_um = float(self.heater.length_um or 0.0)
        length_m = length_um * _UM_TO_M
        wavelength_m = wavelength * _UM_TO_M
        thermal = self._thermal()
        strips = self._strips()
        optics = {
            wg.name: self._optical(wg, thermal, wavelength) for wg in self.waveguides
        }
        waveguides = list(self.waveguides)
        shifter = waveguides[0]
        optical = optics[shifter.name]

        # Re-solve at a drive near P_pi, so the perturbation is the real
        # one: guess it from the core's dn/dT and rise at the core centre.
        core = thermal_properties(
            optical.strip.material, self._stack, self._overrides()
        )
        core_rise = float(
            sample_temperature(
                thermal.mesh,
                thermal.unit_rise,
                np.array([[c] for c in optical.strip.centre]),
            )[0]
        )
        guess = abs(float(core.dn_dT or 0.0) * core_rise)
        reference = wavelength_m / (2.0 * length_m * guess) if guess > 0 else 1.0
        shift = _shift_of(
            optical,
            optical.mesh,
            optical.epsilon0,
            optical.dn_dt,
            optical.unit_rise * reference,
            wavelength_um=wavelength,
            polarization=shifter.polarization,
            n_guess=_core_index(optical),
            order=self.solver.optical_order,
        )
        slope = shift.delta_neff.real / reference  # per W/m
        if slope == 0:
            raise ValueError(
                f"The heater does not shift waveguide {shifter.name!r}'s index: "
                "check that heat reaches it and its materials have dn_dT."
            )
        p_pi_w = wavelength_m / (2.0 * abs(slope))
        slopes = {shifter.name: slope}
        first_order = {shifter.name: shift.delta_neff_first_order.real / reference}
        drive_pi = p_pi_w / length_m
        for wg in waveguides[1:]:
            victim = optics[wg.name]
            vshift = _shift_of(
                victim,
                victim.mesh,
                victim.epsilon0,
                victim.dn_dt,
                victim.unit_rise * drive_pi,
                wavelength_um=wavelength,
                polarization=wg.polarization,
                n_guess=_core_index(victim),
                order=self.solver.optical_order,
            )
            slopes[wg.name] = vshift.delta_neff.real / drive_pi
            first_order[wg.name] = vshift.delta_neff_first_order.real / drive_pi

        per_mw = 1e-3 / length_m  # W/m of drive per mW of total power
        rise_at_cores = sample_temperature(
            thermal.mesh,
            thermal.unit_rise,
            np.array([s.centre for s in strips.values()]).T,
        )
        resistance = thermal.joule.resistance_per_length * length_m
        p_pi = p_pi_w * 1e3
        phases_at_pi = {
            name: 2.0 * math.pi * s * p_pi_w / wavelength_m
            for name, s in slopes.items()
        }
        result = ThermalResult(
            phase_shifter=shifter.name,
            wavelength_um=wavelength,
            length_um=length_um,
            p_pi_mw=p_pi,
            efficiency_rad_per_mw=math.pi / p_pi,
            resistance_ohm=resistance,
            v_pi_v=math.sqrt(p_pi_w * resistance),
            crosstalk_rad={
                name: phase
                for name, phase in phases_at_pi.items()
                if name != shifter.name
            },
            crosstalk_ratio={
                name: phase / phases_at_pi[shifter.name]
                for name, phase in phases_at_pi.items()
                if name != shifter.name
            },
            thermal_resistance_k_per_mw={
                name: float(rise) * per_mw
                for name, rise in zip(strips, rise_at_cores, strict=True)
            },
            dneff_per_mw={name: s * per_mw for name, s in slopes.items()},
            dneff_first_order_per_mw={
                name: s * per_mw for name, s in first_order.items()
            },
            heater_layers=self.heater.layers,
            mesh=thermal.mesh,
            rise_per_power_per_length=thermal.unit_rise,
        )
        steady = _Steady(result=result, shift=shift, slope=slope)
        self._cache[key] = steady
        return steady


def _stack_record(material: str, stack: Any) -> MaterialProperties:
    """The stack's record for a material, as the other material paths read it.

    The stack entry (matched case-insensitively) when it is a valid
    :class:`MaterialProperties` record; otherwise -- no entry, or one that
    does not validate -- the database record, else an empty record.
    """
    entry = _get_ci(stack.materials, material)
    if isinstance(entry, MaterialProperties):
        return entry
    if entry is not None:
        try:
            return MaterialProperties.model_validate(entry)
        except ValidationError:
            pass
    return get_material_properties(material) or MaterialProperties()


def _describe(rects: Any) -> str:
    """Name and in-plane extent of drawn rectangles, for error messages."""
    items = [f"{r.group} [{r.h0:g}, {r.h1:g}] um" for r in rects]
    return ", ".join(items) if items else "none"


def _shift_of(optical: _Optical, *args: Any, **kwargs: Any) -> Any:
    """``solve_thermo_optic_shift``, explaining a truncation it cannot fix.

    When the optical Window stops at a high-index region next to the core,
    a Mode that reaches that edge leaks into the region; enlarging the
    Window, which the solver's warning suggests, would not help.
    """
    if not optical.bounded_by:
        return solve_thermo_optic_shift(*args, **kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shift = solve_thermo_optic_shift(*args, **kwargs)
    names = ", ".join(optical.bounded_by)
    for warning in caught:
        message = str(warning.message)
        if "optical Window edge" in message:
            message = (
                f"{message.split(':')[0]}: the optical Window stops at {names}, "
                f"next to waveguide core {optical.strip.layer!r}, because a Mode "
                "solve that includes it finds that region's Mode. The waveguide's "
                f"Mode reaches {names} and leaks into it, which this model "
                "ignores; a thicker cladding between them removes the leak."
            )
        warnings.warn(message, warning.category, stacklevel=4)
    return shift


def _high_index_beyond_core(
    mesh: CrossSectionMesh, eps: Mapping[str, complex], strip: _Strip
) -> list[tuple[float, str, str]]:
    """Regions wholly above or below the core with an index at least the core's.

    Returns ``(face, side, region)`` per such region: ``side`` is
    ``"below"`` or ``"above"`` the core, and ``face`` is the region's
    z nearest the core (um).
    """
    core = eps[strip.layer].real
    z = mesh.mesh.p[1][mesh.mesh.t]  # (3, n_elements) vertex heights
    found: list[tuple[float, str, str]] = []
    for region, value in eps.items():
        if region == strip.layer or value.real < core:
            continue
        mask = mesh.region_mask(region)
        if not mask.any():
            continue
        top, bottom = float(z[:, mask].max()), float(z[:, mask].min())
        if top <= strip.v0:
            found.append((top, "below", region))
        elif bottom >= strip.v1:
            found.append((bottom, "above", region))
    return found


def _core_index(optical: _Optical) -> float:
    """Largest real index of a waveguide's core in its optical Window."""
    core = optical.mesh.region_mask(optical.strip.layer)
    return float(np.sqrt(optical.epsilon0[core]).real.max())


def _element_areas(mesh: CrossSectionMesh) -> NDArray[np.float64]:
    """Area of every triangle (um^2)."""
    p, t = mesh.mesh.p, mesh.mesh.t
    a, b, c = p[:, t[0]], p[:, t[1]], p[:, t[2]]
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return np.asarray(0.5 * np.abs(cross), dtype=np.float64)


def _dof_weights(
    mesh: CrossSectionMesh,
    n_dofs: int,
    points: NDArray[np.float64],
    weights: Any,
) -> NDArray[np.float64]:
    """DOF vector v with ``v @ rise == sum_i weights_i * rise(points_i)``.

    The Lagrange interpolant is linear in the DOF values, so a weighted
    sum of point samples is one dot product with the field's DOF vector.
    """
    dofs, phi = interpolation_weights(mesh, n_dofs, points)
    out = np.zeros(n_dofs)
    np.add.at(out, dofs, phi * np.asarray(weights, dtype=float))
    return out


def _first_order_weights(mode0: Any, optical: _Optical) -> NDArray[np.float64]:
    """d(n_eff)/d(dT) of every optical element, to first order.

    First-order perturbation gives ``dn_eff = (eps0 c / 2) integral of
    d_eps |E|^2`` over the power-normalised Mode, and the heated index
    gives ``d_eps = 2 n0 dn/dT dT`` to first order, so the shift is a
    fixed weighted sum of the element rises.
    """
    skfem = require_skfem()
    energy = skfem.Functional(_energy_density, dtype=complex)
    per_element = energy.elemental(mode0.basis, E=mode0.basis.interpolate(mode0.E))
    d_eps = 2.0 * np.sqrt(optical.epsilon0) * optical.dn_dt
    return np.asarray(np.real(0.5 * _EPS0 * _C0 * per_element * d_eps), dtype=float)


def _energy_density(w: Any) -> Any:
    """Transverse plus longitudinal |E|^2 density of a femwell Mode field."""
    from skfem.helpers import dot

    return dot(np.conj(w["E"][0]), w["E"][0]) + np.conj(w["E"][1]) * w["E"][1]


def _positive(value: float, name: str, unit: str) -> float:
    """A finite positive input as a float, or a ValueError naming it."""
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be positive ({unit}), got {value}.")
    return number
