"""ThermalSim: the heater phase shifter as a physicist sets it up.

A 2D cross-section of an infinitely long heater over one or more
waveguides. The heater's power heats the cross-section through the
oxide into the substrate (the heat sink at the bottom of the Modelled
depth); the temperature rise shifts the waveguides' effective index
through dn/dT, and the phase follows over the heater length::

    sim = ThermalSim()
    sim.set_geometry(component)
    sim.set_stack(stack)
    sim.set_cross_section("x=0", window=(-40.0, 40.0))
    sim.set_heater("heater", length=320.0)
    sim.add_waveguide("arm", position=0.0)
    sim.add_waveguide("neighbour", position=10.0)
    result = sim.solve(wavelength=1.55)
    result.p_pi, result.crosstalk["neighbour"]
    step = sim.solve_transient(power=20.0, duration=100.0)
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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from gsim.common.cross_section import parse_plane_spec
from gsim.common.stack.materials import (
    MaterialProperties,
    ThermalProperties,
    _get_ci,
    get_material_properties,
    thermal_properties,
)
from gsim.femwell.runtime import require_femwell, require_skfem
from gsim.thermal.mesh import CrossSectionMesh, _drawn_rects, build_cross_section_mesh
from gsim.thermal.optical import (
    ModeShift,
    element_centroids,
    interpolation_weights,
    sample_temperature,
    solve_thermo_optic_shift,
)
from gsim.thermal.results import (
    StepResponse,
    ThermalResult,
    power_per_length,
)
from gsim.thermal.solver import (
    LAGRANGE_DEGREES,
    HeatOperator,
    JouleSource,
    joule_heat_density,
    rise_time_10_90,
    solve_transient,
    time_constant_1e,
)

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack.extractor import LayerStack

__all__ = ["ThermalOptions", "ThermalSim"]

# Per precision: mesh-size factor on the mesher's defaults, transient steps.
_PRECISION: dict[str, tuple[float, int]] = {
    "fast": (1.0, 200),
    "default": (0.6, 400),
    "accurate": (0.35, 1000),
}
# Optical Window around a waveguide: enough oxide for a 500 x 220 nm Si
# strip's TE0 tail to fall below MAX_BOUNDARY_RATIO at its edges.
_OPTICAL_MARGIN_LATERAL_UM = 1.5
_OPTICAL_MARGIN_VERTICAL_UM = 1.0
_FEATURE_DIVISIONS = 4.0
_COARSE_DIVISIONS = 10.0
_UM_TO_M = 1e-6
_SETTLED_FRACTION = 0.99
_EPS0 = 8.8541878128e-12
_C0 = 299792458.0

# The properties set_material accepts, all MaterialProperties fields.
_OVERRIDABLE = (
    "thermal_conductivity",
    "specific_heat",
    "mass_density",
    "dn_dT",
    "conductivity",
    "permittivity",
    "loss_tangent",
)
# set_material's name for a ThermalProperties field, where they differ.
_SET_MATERIAL_NAME = {"electrical_conductivity": "conductivity"}
_DEFAULT_HEAT_TRANSFER_COEFFICIENT = 10.0  # W/m^2/K, free convection in air


@dataclass(frozen=True)
class ThermalOptions:
    """Numerical knobs of a thermal solve; the defaults converge for a strip heater.

    Attributes:
        precision: ``"fast"``, ``"default"`` or ``"accurate"``: scales the
            mesh size everywhere (x1, x0.6, x0.35 of the mesher's
            defaults) and sets the transient's time steps (200, 400,
            1000).
        max_size: Maximum element size in um per Region name (a stack
            layer, or a slab by its material), overriding the precision
            on both the thermal and the optical meshes.
        time_steps: Number of constant transient time steps, overriding
            the precision.
        theta: Time scheme of the transient: 1.0 backward Euler (robust
            for a step drive), 0.5 Crank-Nicolson.
        degree: Lagrange element degree of the heat solve: 1 (linear),
            2 (quadratic, default) or 3. A higher degree reaches a given
            accuracy on a coarser mesh; the optical Mode solve is
            unaffected.
    """

    precision: Literal["fast", "default", "accurate"] = "default"
    max_size: Mapping[str, float] | None = None
    time_steps: int | None = None
    theta: float = 1.0
    degree: int = 2

    def __post_init__(self) -> None:
        """Check the knobs before any mesh is built."""
        if self.precision not in _PRECISION:
            raise ValueError(
                f"precision must be one of {sorted(_PRECISION)}, "
                f"got {self.precision!r}."
            )
        if self.time_steps is not None and (
            int(self.time_steps) != self.time_steps or self.time_steps < 1
        ):
            raise ValueError(
                f"time_steps must be a positive integer, got {self.time_steps}."
            )
        if not 0.5 <= self.theta <= 1.0:
            raise ValueError(
                "theta must lie in [0.5, 1] (0.5 Crank-Nicolson, 1 backward "
                f"Euler), got {self.theta}."
            )
        if self.degree not in LAGRANGE_DEGREES:
            raise ValueError(
                f"degree must be one of {LAGRANGE_DEGREES}, got {self.degree!r}."
            )

    @property
    def size_factor(self) -> float:
        """Factor on the mesher's default element sizes."""
        return _PRECISION[self.precision][0]

    @property
    def steps(self) -> int:
        """Number of transient time steps."""
        if self.time_steps is not None:
            return int(self.time_steps)
        return _PRECISION[self.precision][1]

    def key(self) -> tuple[Any, ...]:
        """Hashable identity of the mesh-relevant knobs."""
        return (
            self.precision,
            tuple(sorted((self.max_size or {}).items())),
            self.degree,
        )


class _WaveguideSpec(BaseModel):
    """One waveguide of interest, as the user named it."""

    model_config = ConfigDict(frozen=True)

    name: str
    position: float
    layer: str | None = None
    window: tuple[float, float] | None = None
    polarization: Literal["TE", "TM"] = "TE"


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


@dataclass(frozen=True)
class _Steady:
    """A steady solve and what the transient reuses from it."""

    result: ThermalResult
    shift: ModeShift
    slope: float


class ThermalSim(BaseModel):
    """A 2D thermal cross-section of a heater over waveguides.

    Configure with the setters, then :meth:`solve` (steady: P_pi,
    crosstalk, resistance, V_pi, temperature field) and
    :meth:`solve_transient` (switching time). Every setter can be called
    in any order before a solve; a setter called -- or a field assigned
    directly -- after a solve clears the cached meshes and solves.

    Units: lengths in um, power in mW, temperature rise in K, time in us.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    component: Any = None
    stack: Any = None
    plane: str | None = None
    window: tuple[float, float] | None = None
    depth: float | None = None
    heater_layers: tuple[str, ...] = ()
    heater_length: float | None = None
    waveguides: tuple[_WaveguideSpec, ...] = ()
    materials: dict[str, dict[str, float]] = Field(default_factory=dict)
    boundary: dict[str, float] = Field(default_factory=dict)

    _cache: dict[tuple[Any, ...], Any] = PrivateAttr(default_factory=dict)
    _warned: set[str] = PrivateAttr(default_factory=set)
    _h: float = PrivateAttr(default=_DEFAULT_HEAT_TRANSFER_COEFFICIENT)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a field; changing any setup field forgets the cached solves."""
        super().__setattr__(name, value)
        if not name.startswith("_"):
            self._changed()

    # ------------------------------------------------------------------ setup

    def set_geometry(self, component: gf.Component) -> None:
        """Set the component whose cross-section is solved.

        Args:
            component: The gdsfactory component (heater and waveguides).
        """
        self.component = component

    def set_stack(self, stack: LayerStack) -> None:
        """Set the layer stack: layer heights, materials and slabs.

        Args:
            stack: A gsim ``LayerStack``, e.g. from
                :func:`gsim.common.stack.get_stack`.
        """
        self.stack = stack

    def set_cross_section(
        self,
        plane: str,
        *,
        window: tuple[float, float] | None = None,
        depth: float | None = None,
    ) -> None:
        """Choose the cut and the thermal Window.

        Args:
            plane: The cut, ``"x=<um>"`` or ``"y=<um>"``, across the
                heater and waveguides.
            window: In-plane extent ``(min, max)`` of the thermal Window
                in um. Defaults to the drawn cut plus 20 um each side.
                The Window's sides are adiabatic, so make it wide next
                to the oxide thickness (a half-width of 5x is ample).
            depth: Modelled depth of the substrate below the stack in
                um, the bottom of which is the heat sink. Defaults to
                the stack's substrate thickness (100 um without one).
        """
        axis, _ = parse_plane_spec(plane)
        if axis not in {"x", "y"}:
            raise ValueError(
                "A heater cross-section is vertical: use an 'x=<um>' or "
                f"'y=<um>' plane, not {plane!r}."
            )
        if window is not None and not window[1] > window[0]:
            raise ValueError(f"window must be (min, max) with max > min, got {window}.")
        if depth is not None and not depth > 0:
            raise ValueError(f"depth must be positive (um), got {depth}.")
        self.plane = plane
        self.window = None if window is None else (float(window[0]), float(window[1]))
        self.depth = None if depth is None else float(depth)

    def set_heater(self, layers: str | list[str], *, length: float) -> None:
        """Name the heater: the layers whose drawn conductors carry its current.

        The current runs along the waveguide, so every conductor of the
        heater in the cut sees the same field and they share the power
        in proportion to their conductance.

        Args:
            layers: Stack layer name(s) of the heater conductors.
            length: Heater length along the waveguide in um; it sets the
                resistance, V_pi and the phase (P_pi does not depend on
                it in 2D).
        """
        names = (layers,) if isinstance(layers, str) else tuple(layers)
        if not names:
            raise ValueError("set_heater needs at least one layer name.")
        if not length > 0:
            raise ValueError(f"Heater length must be positive (um), got {length}.")
        self.heater_layers = names
        self.heater_length = float(length)

    def add_waveguide(
        self,
        name: str,
        position: float,
        *,
        layer: str | None = None,
        window: tuple[float, float] | None = None,
        polarization: Literal["TE", "TM"] = "TE",
    ) -> None:
        """Add a waveguide of interest at an in-plane position of the cut.

        The first waveguide added is the phase shifter; every other one
        is a crosstalk victim. Each gets its own optical Window, by
        default the waveguide's drawn rectangle plus 1.5 um each side
        and 1 um above and below, clipped to the thermal Window.

        Args:
            name: Name for the results (``result.crosstalk[name]``).
            position: In-plane coordinate in um (``y`` for an ``"x=..."``
                cut) inside the waveguide's drawn rectangle.
            layer: Stack layer of the waveguide core. Needed only when
                several non-heater layers are drawn at ``position``.
            window: In-plane extent ``(min, max)`` of the optical Window
                in um, replacing the default.
            polarization: ``"TE"`` or ``"TM"`` fundamental Mode.
        """
        if any(wg.name == name for wg in self.waveguides):
            raise ValueError(f"A waveguide named {name!r} was already added.")
        spec = _WaveguideSpec(
            name=name,
            position=float(position),
            layer=layer,
            window=window,
            polarization=polarization,
        )
        if self._can_locate():
            self._locate_strip(spec)
        self.waveguides = (*self.waveguides, spec)

    def set_material(self, name: str, **properties: float) -> None:
        """Override material properties for this simulation only.

        Calls for the same material merge, whatever the case of
        ``name``; the override keeps the spelling of the first call.

        Args:
            name: Material name as the stack names it (case-insensitive).
            **properties: Any of ``thermal_conductivity`` (W/m/K),
                ``specific_heat`` (J/kg/K), ``mass_density`` (kg/m^3),
                ``dn_dT`` (1/K), ``conductivity`` (electrical, S/m),
                ``permittivity`` and ``loss_tangent`` (optical, at the
                solve wavelength).
        """
        unknown = sorted(set(properties) - set(_OVERRIDABLE))
        if unknown:
            raise ValueError(
                f"Unknown material properties {unknown}; set_material accepts "
                f"{list(_OVERRIDABLE)}."
            )
        key = next(
            (k for k in self.materials if k.lower().strip() == name.lower().strip()),
            name,
        )
        merged = {**self.materials.get(key, {}), **properties}
        MaterialProperties.model_validate(merged)
        self.materials = {**self.materials, key: merged}

    def set_boundary(
        self,
        *,
        top: Literal["adiabatic", "convective"] | None = None,
        sides: Literal["adiabatic", "convective"] | None = None,
        heat_transfer_coefficient: float | None = None,
    ) -> None:
        """Choose the top and side boundary conditions.

        The bottom of the Modelled depth is always the heat sink, at
        ambient. A convective edge loses ``h dT`` to ambient; free
        convection in air (h ~ 10 W/m^2/K) is negligible next to the
        substrate path, hence the adiabatic default. Calls are
        cumulative: a setting left as ``None`` keeps its current value
        (initially adiabatic top and sides, h = 10 W/m^2/K).

        Args:
            top: Condition on the top of the stack.
            sides: Condition on the two sides of the thermal Window.
            heat_transfer_coefficient: h in W/m^2/K, shared by every
                convective edge.
        """
        if top is None:
            top = "convective" if "top" in self.boundary else "adiabatic"
        if sides is None:
            sides = "convective" if "left" in self.boundary else "adiabatic"
        if heat_transfer_coefficient is None:
            heat_transfer_coefficient = self._h
        for label, value in (("top", top), ("sides", sides)):
            if value not in {"adiabatic", "convective"}:
                raise ValueError(
                    f"{label} must be 'adiabatic' or 'convective', got {value!r}."
                )
        if not heat_transfer_coefficient >= 0:
            raise ValueError(
                "heat_transfer_coefficient must be >= 0 (W/m^2/K), "
                f"got {heat_transfer_coefficient}."
            )
        h = float(heat_transfer_coefficient)
        self._h = h
        boundary: dict[str, float] = {}
        if top == "convective":
            boundary["top"] = h
        if sides == "convective":
            boundary["left"] = h
            boundary["right"] = h
        self.boundary = boundary

    # ------------------------------------------------------------------ solves

    def solve(
        self, *, wavelength: float = 1.55, options: ThermalOptions | None = None
    ) -> ThermalResult:
        """Steady heater solve: P_pi, crosstalk, resistance and temperature.

        One heat solve at a unit power per length gives the temperature
        rise per watt per metre (the model is linear); each waveguide's
        Mode is then re-solved with the heated index on its optical
        Window at a power near P_pi, giving the effective-index shift
        per unit power.

        Args:
            wavelength: Vacuum wavelength in um.
            options: Numerical knobs; :class:`ThermalOptions` defaults.

        Returns:
            The :class:`~gsim.thermal.ThermalResult`.
        """
        return self._steady(float(wavelength), options or ThermalOptions()).result

    def solve_transient(
        self,
        *,
        power: float,
        duration: float,
        wavelength: float = 1.55,
        options: ThermalOptions | None = None,
    ) -> StepResponse:
        """Switch-on response to a heater power stepped on at t = 0.

        The phase over time comes from first-order perturbation of the
        phase shifter's Mode by the temperature history, scaled so that
        its steady value matches the steady Mode re-solve.

        Args:
            power: Total heater power stepped on (mW).
            duration: Simulated time (us); long enough to settle (the
                result warns otherwise).
            wavelength: Vacuum wavelength in um.
            options: Numerical knobs; ``precision`` sets the time steps.

        Returns:
            The :class:`~gsim.thermal.StepResponse`.
        """
        if not power > 0:
            raise ValueError(f"power must be positive (mW), got {power}.")
        if not duration > 0:
            raise ValueError(f"duration must be positive (us), got {duration}.")
        opts = options or ThermalOptions()
        steady = self._steady(float(wavelength), opts)
        thermal = self._thermal(opts)
        result = steady.result
        shifter = result.phase_shifter
        optical = self._optical(self._spec(shifter), thermal, result.wavelength, opts)

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

        drive = power_per_length(power, result.length)
        history = solve_transient(
            thermal.mesh.mesh,
            thermal.conductivity,
            self._heat_capacity(thermal.mesh),
            thermal.joule.heat_density * drive,
            duration_us=float(duration),
            steps=opts.steps,
            theta=opts.theta,
            boundary_facets=thermal.mesh.boundary_facets,
            sink="bottom",
            convective=thermal.convective,
            sample=lambda rise: sampler @ rise,
            degree=opts.degree,
        )
        first_order_slope = float(shift_weights @ thermal.unit_rise)
        scale = steady.slope / first_order_slope
        phase = np.asarray(
            2.0
            * math.pi
            * history.values[:, 0]
            * scale
            * result.length
            / result.wavelength,
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
        return StepResponse(
            time=history.time_us,
            power=float(power),
            phase_shifter=shifter,
            phase=phase,
            steady_phase=steady_phase,
            temperature={
                name: history.values[:, i + 1] for i, name in enumerate(strips)
            },
            tau=rise_time_10_90(history.time_us, phase),
            tau_1e=time_constant_1e(history.time_us, phase),
        )

    # ------------------------------------------------------------------ internals

    def _changed(self) -> None:
        """Forget every cached solve and once-only warning after a setup change."""
        self._cache.clear()
        self._warned.clear()

    def _spec(self, name: str) -> _WaveguideSpec:
        """The waveguide added under ``name``."""
        return next(wg for wg in self.waveguides if wg.name == name)

    def _can_locate(self) -> bool:
        """Whether the cut is known well enough to find a waveguide in it."""
        return (
            self.component is not None
            and self.stack is not None
            and self.plane is not None
            and bool(self.heater_layers)
        )

    def _require_setup(self) -> None:
        """Raise naming the first setter a solve still needs."""
        missing = [
            (self.component is None, "set_geometry(component)"),
            (self.stack is None, "set_stack(stack)"),
            (self.plane is None, "set_cross_section('x=0', ...)"),
            (not self.heater_layers, "set_heater(layers, length=...)"),
            (
                not self.waveguides,
                "add_waveguide(name, position) -- P_pi needs a phase-shifter waveguide",
            ),
        ]
        for absent, call in missing:
            if absent:
                raise ValueError(f"The simulation is not set up: call {call} first.")

    def _section(self) -> list[Any]:
        """The drawn rectangles of the cut, one per layer interval."""
        axis, value = parse_plane_spec(str(self.plane))
        return _drawn_rects(self.component, self.stack, axis, value)

    def _locate_strip(self, spec: _WaveguideSpec) -> _Strip:
        """The drawn rectangle of a waveguide, found by in-plane position."""
        rects = self._section()
        at = [r for r in rects if r.h0 <= spec.position <= r.h1]
        if spec.layer is not None:
            hits = [r for r in at if r.group == spec.layer]
            if not hits:
                raise ValueError(
                    f"Waveguide {spec.name!r}: layer {spec.layer!r} is not drawn at "
                    f"{spec.position:g} um in the cut {self.plane!r}. Drawn there: "
                    f"{sorted({r.group for r in at}) or 'nothing'}."
                )
        else:
            hits = [r for r in at if r.group not in self.heater_layers]
            if not hits:
                raise ValueError(
                    f"Waveguide {spec.name!r}: no drawn non-heater layer at "
                    f"{spec.position:g} um in the cut {self.plane!r}. Waveguide "
                    "cores drawn in the cut: "
                    + _describe(r for r in rects if r.group not in self.heater_layers)
                    + "."
                )
            if len(hits) > 1:
                raise ValueError(
                    f"Waveguide {spec.name!r}: several layers are drawn at "
                    f"{spec.position:g} um ({_describe(hits)}); name the core "
                    "with add_waveguide(..., layer=...)."
                )
        r = hits[0]
        return _Strip(r.group, r.material, r.h0, r.h1, r.v0, r.v1)

    def _strips(self) -> dict[str, _Strip]:
        """Every waveguide's drawn rectangle, by name, in the order added."""
        return {wg.name: self._locate_strip(wg) for wg in self.waveguides}

    def _overrides(self) -> dict[str, dict[str, float]]:
        """Per-run material overrides, as ``thermal_properties`` takes them."""
        return self.materials

    def _override_for(self, material: str) -> dict[str, float] | None:
        """This run's overrides of one material, matched case-insensitively."""
        for name, props in self.materials.items():
            if name.lower().strip() == material.lower().strip():
                return props
        return None

    def _thermal_props(
        self, material: str, region: str, required: tuple[str, ...]
    ) -> ThermalProperties:
        """Thermal properties of a Region's material, or a physics error."""
        try:
            return thermal_properties(
                material, self.stack, self._overrides(), required=required
            )
        except ValueError as error:
            hint = ""
            if required:
                kwargs = ", ".join(
                    f"{_SET_MATERIAL_NAME.get(n, n)}=..." for n in required
                )
                hint = f" For this run only: sim.set_material({material!r}, {kwargs})."
            raise ValueError(
                f"Region {region!r} is {material!r}: {error}{hint}"
            ) from error

    def _mesh_sizes(
        self, rects: list[Any], extent: float, opts: ThermalOptions
    ) -> tuple[dict[str, float], float]:
        """Per-Region and coarse element sizes for one precision."""
        f = opts.size_factor
        coarse = f * extent / _COARSE_DIVISIONS
        sizes: dict[str, float] = {}
        for r in rects:
            own = f * min(r.h1 - r.h0, r.v1 - r.v0) / _FEATURE_DIVISIONS
            sizes[r.group] = min(sizes.get(r.group, coarse), own, coarse)
        sizes.update(opts.max_size or {})
        return sizes, coarse

    def _thermal(self, opts: ThermalOptions) -> _Thermal:
        """The thermal mesh and its rise per unit power per length (cached)."""
        key = ("thermal", opts.key())
        if key in self._cache:
            return cast("_Thermal", self._cache[key])
        rects = self._section()
        drawn = {r.group for r in rects}
        absent = [name for name in self.heater_layers if name not in drawn]
        if len(absent) == len(self.heater_layers):
            raise ValueError(
                f"Heater layer(s) {absent} are not drawn in the cut "
                f"{self.plane!r}; layers drawn there: {sorted(drawn)}."
            )
        if absent:
            warnings.warn(
                f"Heater layer(s) {absent} are not drawn in the cut "
                f"{self.plane!r}; only {sorted(set(self.heater_layers) - set(absent))} "
                "carry the heater current there.",
                UserWarning,
                stacklevel=4,
            )
        h0 = min(r.h0 for r in rects)
        h1 = max(r.h1 for r in rects)
        window = self.window or (h0 - 20.0, h1 + 20.0)
        self._check_heater_inside(rects, window)
        sizes, coarse = self._mesh_sizes(rects, window[1] - window[0], opts)
        mesh = build_cross_section_mesh(
            self.component,
            self.stack,
            str(self.plane),
            window=window,
            depth=self.depth,
            max_size=sizes,
            default_max_size=coarse,
        )
        regions = mesh.element_region
        k_by_region: dict[str, float] = {}
        sigma_by_region: dict[str, float] = {}
        for region, material in mesh.region_material.items():
            heater = region in self.heater_layers
            required = ("thermal_conductivity",) + (
                ("electrical_conductivity",) if heater else ()
            )
            props = self._thermal_props(material, region, required)
            k_by_region[region] = float(props.thermal_conductivity or 0.0)
            sigma_by_region[region] = (
                float(props.electrical_conductivity or 0.0) if heater else 0.0
            )
        conductivity = np.array([k_by_region[r] for r in regions])
        sigma = np.array([sigma_by_region[r] for r in regions])
        heater_mask = np.isin(regions, self.heater_layers)
        joule = joule_heat_density(
            sigma, heater_mask, _element_areas(mesh), power_per_length_W_per_m=1.0
        )
        convective = dict(self.boundary)
        operator = HeatOperator(
            mesh.mesh,
            conductivity,
            boundary_facets=mesh.boundary_facets,
            sink="bottom",
            convective=convective,
            degree=opts.degree,
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
            if r.group in self.heater_layers and (r.h0 < window[0] or r.h1 > window[1]):
                raise ValueError(
                    f"Heater layer {r.group!r} is drawn over [{r.h0:g}, {r.h1:g}] um "
                    f"in the cut {self.plane!r}, outside the thermal Window "
                    f"({window[0]:g}, {window[1]:g}) um: the heater's resistance "
                    "and power would be wrong. Widen the Window with "
                    "set_cross_section(..., window=(min, max)) to contain every "
                    "heater conductor."
                )

    def _optical_window(
        self, spec: _WaveguideSpec, strip: _Strip, thermal: _Thermal
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """The optical Window of a waveguide, clipped to the thermal Window."""
        lateral = spec.window or (
            strip.h0 - _OPTICAL_MARGIN_LATERAL_UM,
            strip.h1 + _OPTICAL_MARGIN_LATERAL_UM,
        )
        vertical = (
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
        spec: _WaveguideSpec,
        thermal: _Thermal,
        wavelength: float,
        opts: ThermalOptions,
    ) -> _Optical:
        """A waveguide's optical Window with its permittivity and dn/dT (cached)."""
        key = ("optical", spec.name, wavelength, opts.key())
        if key in self._cache:
            return cast("_Optical", self._cache[key])
        from gsim.femwell.adapter import epsilon_by_region

        strip = self._locate_strip(spec)
        window, z_range = self._optical_window(spec, strip, thermal)
        rects = [
            r
            for r in self._section()
            if r.h1 > window[0] and r.h0 < window[1] and r.v1 > z_range[0]
        ]
        sizes, coarse = self._mesh_sizes(
            rects, max(window[1] - window[0], z_range[1] - z_range[0]), opts
        )
        mesh = build_cross_section_mesh(
            self.component,
            self.stack,
            str(self.plane),
            window=window,
            z_range=z_range,
            max_size=sizes,
            default_max_size=coarse,
        )
        eps = epsilon_by_region(
            mesh.msh_path,
            self.stack,
            wavelength_um=wavelength,
            overrides=self._optical_overrides(mesh),
        )
        core = self._thermal_props(strip.material, strip.layer, ())
        if core.dn_dT is None or core.dn_dT == 0:
            raise ValueError(
                f"Waveguide {spec.name!r}: its core {strip.layer!r} is "
                f"{strip.material!r}, which has no (or a zero) dn_dT, so heat "
                "cannot shift its index. Give it with "
                f"sim.set_material({strip.material!r}, dn_dT=...) (1/K)."
            )
        dn_by_region: dict[str, float] = {}
        missing: list[str] = []
        for region, material in mesh.region_material.items():
            dn = thermal_properties(material, self.stack, self._overrides()).dn_dT
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
        )
        self._cache[key] = optical
        return optical

    def _optical_overrides(self, mesh: CrossSectionMesh) -> dict[str, Any]:
        """Material records carrying this run's optical overrides."""
        records: dict[str, Any] = {}
        for material in set(mesh.region_material.values()):
            props = self._override_for(material)
            if not props:
                continue
            base = _stack_record(material, self.stack)
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
            "independent of temperature. Set it with sim.set_material(name, "
            "dn_dT=...) if they carry optical field.",
            UserWarning,
            stacklevel=5,
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

    def _steady(self, wavelength: float, opts: ThermalOptions) -> _Steady:
        """The steady solve, cached per wavelength and mesh knobs."""
        key = ("steady", wavelength, opts.key())
        if key in self._cache:
            return cast("_Steady", self._cache[key])
        require_skfem()
        require_femwell()
        self._require_setup()
        length_um = float(self.heater_length or 0.0)
        length_m = length_um * _UM_TO_M
        wavelength_m = wavelength * _UM_TO_M
        thermal = self._thermal(opts)
        strips = self._strips()
        optics = {
            wg.name: self._optical(wg, thermal, wavelength, opts)
            for wg in self.waveguides
        }
        shifter = self.waveguides[0]
        optical = optics[shifter.name]

        # Re-solve at a drive near P_pi, so the perturbation is the real
        # one: guess it from the core's dn/dT and rise at the core centre.
        core = thermal_properties(optical.strip.material, self.stack, self._overrides())
        core_rise = float(
            sample_temperature(
                thermal.mesh,
                thermal.unit_rise,
                np.array([[c] for c in optical.strip.centre]),
            )[0]
        )
        guess = abs(float(core.dn_dT or 0.0) * core_rise)
        reference = wavelength_m / (2.0 * length_m * guess) if guess > 0 else 1.0
        shift = solve_thermo_optic_shift(
            optical.mesh,
            optical.epsilon0,
            optical.dn_dt,
            optical.unit_rise * reference,
            wavelength_um=wavelength,
            polarization=shifter.polarization,
            n_guess=_core_index(optical),
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
        for wg in self.waveguides[1:]:
            victim = optics[wg.name]
            vshift = solve_thermo_optic_shift(
                victim.mesh,
                victim.epsilon0,
                victim.dn_dt,
                victim.unit_rise * drive_pi,
                wavelength_um=wavelength,
                polarization=wg.polarization,
                n_guess=_core_index(victim),
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
            wavelength=wavelength,
            length=length_um,
            p_pi=p_pi,
            efficiency=math.pi / p_pi,
            resistance=resistance,
            v_pi=math.sqrt(p_pi_w * resistance),
            crosstalk={
                name: phase
                for name, phase in phases_at_pi.items()
                if name != shifter.name
            },
            crosstalk_ratio={
                name: phase / phases_at_pi[shifter.name]
                for name, phase in phases_at_pi.items()
                if name != shifter.name
            },
            thermal_resistance={
                name: float(rise) * per_mw
                for name, rise in zip(strips, rise_at_cores, strict=True)
            },
            dneff_per_power={name: s * per_mw for name, s in slopes.items()},
            dneff_first_order={name: s * per_mw for name, s in first_order.items()},
            heater_layers=self.heater_layers,
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
