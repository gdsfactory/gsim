"""Tests for the physics-noun layer of gsim.thermal (Simulation and results)."""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager

import gdsfactory as gf
import numpy as np
import pytest
from scipy.integrate import quad

from gsim.common.stack.extractor import Layer, LayerStack

pytest.importorskip("skfem")
pytest.importorskip("femwell")

from gsim.thermal import (
    HeaterReport,
    Simulation,
    Solver,
    StepResponse,
    TemperatureMap,
    ThermalResult,
)

WAVELENGTH = 1.55
# femwell's metal_heater_phase_shifter.py inputs (Jacques et al. 2019 TiN heater):
# 0.5 x 0.22 Si core (k 90), 2 um BOX, 0.5 um Si wafer (k 148) on the sink,
# 2 x 0.14 TiN (k 28, sigma 2.3e6 S/m) 2 um above the core, oxide (k 1.38) to
# 2.8 um above the core bottom, 16 um wide. Its output, 3.786 rad at 7.4 mA
# over 320 um, is P_pi = 22.6 mW.
FEMWELL_P_PI_MW = 22.6
SIGMA_TIN = 2.3e6
HEATER_W, HEATER_T = 2.0, 0.14
CORE_W, CORE_T = 0.5, 0.22

_SI = {"specific_heat": 711.0, "mass_density": 2330.0, "dn_dT": 1.86e-4}
_OX = {"specific_heat": 709.0, "mass_density": 2203.0}


def _layer(name, gds, zmin, zmax, material, kind="dielectric"):
    """One drawn layer of a test stack."""
    return Layer(
        name=name,
        gds_layer=gds,
        zmin=zmin,
        zmax=zmax,
        thickness=zmax - zmin,
        material=material,
        layer_type=kind,
    )


def femwell_stack() -> LayerStack:
    """The femwell TiN example's stack, with its own material values."""
    materials = {
        "si_core": {"permittivity": 3.4777**2, "thermal_conductivity": 90.0, **_SI},
        "si_wafer": {"permittivity": 3.4777**2, "thermal_conductivity": 148.0, **_SI},
        "oxide": {
            "permittivity": 1.444**2,
            "thermal_conductivity": 1.38,
            "dn_dT": 1e-5,
            **_OX,
        },
        "tin": {
            "conductivity": SIGMA_TIN,
            "thermal_conductivity": 28.0,
            "specific_heat": 598.0,
            "mass_density": 5240.0,
        },
    }
    layers = {
        "core": _layer("core", (1, 0), 0.0, CORE_T, "si_core"),
        "heater": _layer(
            "heater", (47, 0), 2.22, 2.22 + HEATER_T, "tin", kind="conductor"
        ),
    }
    dielectrics = [
        {"name": "substrate", "zmin": -2.5, "zmax": -2.0, "material": "si_wafer"},
        {"name": "box", "zmin": -2.0, "zmax": 0.0, "material": "oxide"},
        {"name": "clad", "zmin": 0.0, "zmax": 2.8, "material": "oxide"},
    ]
    return LayerStack(
        pdk_name="test", layers=layers, materials=materials, dielectrics=dielectrics
    )


def _rect(c: gf.Component, center: float, width: float, layer) -> None:
    """A strip of the given width along x, centred at y = center."""
    y0, y1 = center - width / 2, center + width / 2
    c.add_polygon([(-5, y0), (5, y0), (5, y1), (-5, y1)], layer=layer)


def heater_over_strip(neighbour: float | None = None) -> gf.Component:
    """A TiN strip over a Si strip, and optionally a second strip beside it."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    _rect(c, 0.0, CORE_W, (1, 0))
    _rect(c, 0.0, HEATER_W, (47, 0))
    if neighbour is not None:
        _rect(c, neighbour, CORE_W, (1, 0))
    return c


def femwell_sim(length: float = 320.0, neighbour: float | None = None) -> Simulation:
    """The femwell TiN case, 16 um wide, configured but not solved."""
    sim = Simulation()
    sim.geometry(heater_over_strip(neighbour), stack=femwell_stack())
    sim.domain(plane="x=0", window=(-8.0, 8.0))
    sim.heater(layers="heater", length_um=length)
    sim.waveguides.add(name="arm", position_um=0.0)
    if neighbour is not None:
        sim.waveguides.add(name="neighbour", position_um=neighbour)
    return sim


def stack_of(sim: Simulation) -> LayerStack:
    """The sim's stack, to edit in place."""
    stack = sim.geometry.stack
    assert isinstance(stack, LayerStack)
    return stack


@contextmanager
def solver_settings(sim: Simulation, **settings):
    """Change a shared sim's solver for one block, then put it back."""
    saved = dict(sim.solver)
    sim.solver(**settings)
    try:
        yield sim
    finally:
        sim.solver(**saved)


# --------------------------------------------------------------- canonical script


@pytest.fixture(scope="module")
def canonical():
    """The canonical script of ticket 06, on a TiN heater with a neighbour."""
    sim = Simulation()
    sim.geometry(heater_over_strip(neighbour=5.0), stack=femwell_stack())
    sim.domain(plane="x=0", window=(-15.0, 15.0))
    sim.heater(layers="heater", length_um=100.0)
    sim.waveguides.add(name="arm", position_um=0.0)
    sim.waveguides.add(name="neighbour", position_um=5.0)

    sim.solver(precision="fast")
    result = sim.solve(wavelength_um=WAVELENGTH)
    step = sim.solve_transient(power_mw=20.0, duration_us=60.0)
    return sim, result, step


def test_the_canonical_script_gives_labelled_physics(canonical):
    _sim, result, step = canonical

    assert isinstance(result, ThermalResult)
    assert 15.0 < result.p_pi_mw < 30.0
    assert result.efficiency_rad_per_mw == pytest.approx(math.pi / result.p_pi_mw)
    assert 0.0 < result.crosstalk_rad["neighbour"] < math.pi
    assert result.v_pi_v == pytest.approx(
        math.sqrt(result.p_pi_mw * 1e-3 * result.resistance_ohm)
    )
    tmap = result.temperature(power_mw=result.p_pi_mw)
    assert isinstance(tmap, TemperatureMap)
    # The heater is the hottest thing in the cross-section.
    assert tmap.peak("heater") == pytest.approx(tmap.peak())

    assert isinstance(step, StepResponse)
    assert step.tau_us > 0.0
    report = result.report(step)
    assert isinstance(report, HeaterReport)
    text = str(report)
    for label in ("P_pi (mW)", "V_pi (V)", "tau 10-90 % (us)", "neighbour"):
        assert label in text
    assert "<table" in report._repr_html_()


def test_the_phase_shifter_picks_up_exactly_pi_at_p_pi(canonical):
    _sim, result, _step = canonical
    phase = result.phase(result.p_pi_mw)
    assert phase["arm"] == pytest.approx(math.pi, rel=1e-9)
    assert phase["neighbour"] == pytest.approx(result.crosstalk_rad["neighbour"])


def test_the_temperature_map_and_step_response_plot_on_matplotlib_axes(canonical):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _sim, result, step = canonical
    ax = result.temperature(power_mw=result.p_pi_mw).plot()
    assert ax.get_xlabel().startswith("y")
    assert step.plot().get_xlabel() == "Time (µs)"
    plt.close("all")


# --------------------------------------------------------------- unit consistency


@pytest.fixture(scope="module")
def femwell_result() -> ThermalResult:
    """The femwell TiN case at default precision."""
    return femwell_sim().solve(wavelength_um=WAVELENGTH)


def test_femwell_tin_heater_p_pi_matches_femwell_within_3_percent(femwell_result):
    assert femwell_result.p_pi_mw == pytest.approx(FEMWELL_P_PI_MW, rel=0.03)


def test_resistance_is_length_over_conductance_of_the_drawn_heater(femwell_result):
    by_hand = 320e-6 / (SIGMA_TIN * HEATER_W * 1e-6 * HEATER_T * 1e-6)
    assert femwell_result.resistance_ohm == pytest.approx(by_hand, rel=1e-9)
    assert by_hand == pytest.approx(496.9, abs=0.1)


def test_doubling_power_doubles_the_temperature_rise(femwell_result):
    one = femwell_result.temperature(power_mw=10.0)
    two = femwell_result.temperature(power_mw=20.0)
    np.testing.assert_allclose(two.values, 2.0 * one.values, rtol=1e-12)
    assert two.at(0.0, 0.11) == pytest.approx(2.0 * one.at(0.0, 0.11))


def test_thermal_resistance_is_the_core_rise_per_milliwatt(femwell_result):
    # Core centre rise per mW of total heater power over 320 um. From the
    # map at 1 mW: same number, different route through the units.
    rise = femwell_result.temperature(power_mw=1.0).at(0.0, CORE_T / 2)
    assert femwell_result.thermal_resistance_k_per_mw["arm"] == pytest.approx(rise)
    # Order of magnitude by hand: dT_pi = lambda / (2 L dn/dT) = 13 K at P_pi.
    assert femwell_result.thermal_resistance_k_per_mw["arm"] == pytest.approx(
        13.0 / FEMWELL_P_PI_MW, rel=0.15
    )


def test_p_pi_does_not_depend_on_the_heater_length(femwell_result):
    short = femwell_sim(length=100.0).solve(wavelength_um=WAVELENGTH)
    assert short.p_pi_mw == pytest.approx(femwell_result.p_pi_mw, rel=2e-3)
    assert short.resistance_ohm == pytest.approx(
        femwell_result.resistance_ohm * 100 / 320
    )


def test_p_pi_scales_with_the_oxide_conductivity_override(femwell_result):
    sim = femwell_sim()
    sim.materials["oxide"] = {"thermal_conductivity": 2 * 1.38}
    sim.solver(precision="fast")
    stiffer = sim.solve(wavelength_um=WAVELENGTH)
    # Heat leaves through the oxide: P_pi ~ k_oxide to first order.
    assert 1.8 < stiffer.p_pi_mw / femwell_result.p_pi_mw < 2.05


# --------------------------------------------------------------- analytic references

H_OX = 3.0  # oxide thickness between the adiabatic top and the sink (um)
K_OX = 1.38
STRIP_W = 0.4


def uniform_oxide_sim() -> Simulation:
    """Uniform-k oxide over a sink, a thin strip heater on its top surface."""
    gf.gpdk.PDK.activate()
    uniform = {"thermal_conductivity": K_OX, **_OX}
    materials = {
        "ox_u": {"permittivity": 1.444**2, "dn_dT": 1e-5, **uniform},
        "si_u": {"permittivity": 3.4777**2, **_SI, **uniform},
        "metal_u": {"conductivity": 1e6, **uniform},
    }
    layers = {
        "core": _layer("core", (1, 0), 1.0, 1.0 + CORE_T, "si_u"),
        "heater": _layer(
            "heater", (47, 0), H_OX - 0.04, H_OX, "metal_u", kind="conductor"
        ),
    }
    dielectrics = [
        {"name": "substrate", "zmin": 0.0, "zmax": 0.5, "material": "ox_u"},
        {"name": "clad", "zmin": 0.5, "zmax": H_OX, "material": "ox_u"},
    ]
    stack = LayerStack(
        pdk_name="test", layers=layers, materials=materials, dielectrics=dielectrics
    )
    c = gf.Component()
    _rect(c, 0.0, STRIP_W, (47, 0))
    _rect(c, -6.0, CORE_W, (1, 0))
    sim = Simulation()
    sim.geometry(c, stack=stack)
    sim.domain(plane="x=0", window=(-8 * H_OX, 8 * H_OX))
    sim.heater(layers="heater", length_um=100.0)
    sim.waveguides.add(name="probe", position_um=-6.0)
    return sim


def strip_on_surface(x: float, power_per_length: float) -> float:
    """Surface rise of a strip heater on an adiabatic top over a sink.

    The line-source result (P'/pi k) ln coth(pi x / 4H) of research/03
    section 2.2, averaged over the strip width.
    """

    def line(s: float) -> float:
        """Rise at x of a line source at s on the surface."""
        u = math.pi * abs(x - s) / (4 * H_OX)
        return math.log(1.0 / math.tanh(u)) / (math.pi * K_OX)

    value, _ = quad(line, -STRIP_W / 2, STRIP_W / 2, limit=200)
    return power_per_length * value / STRIP_W


@pytest.fixture(scope="module")
def uniform_result() -> ThermalResult:
    """The uniform-oxide sim solved at default precision."""
    return uniform_oxide_sim().solve(wavelength_um=WAVELENGTH)


def test_surface_rise_matches_the_ln_coth_strip_solution(uniform_result):
    power = 1.0  # mW over 100 um: P' = 10 W/m
    tmap = uniform_result.temperature(power_mw=power)
    x = np.linspace(1.0, 2.0 * H_OX, 11)
    fem = tmap.at(x, np.full_like(x, H_OX))
    exact = np.array([strip_on_surface(xi, 10.0) for xi in x])
    np.testing.assert_allclose(fem, exact, rtol=0.01)


def test_crosstalk_tail_decays_over_2h_over_pi(uniform_result):
    tmap = uniform_result.temperature(power_mw=1.0)
    x = np.linspace(2 * H_OX, 4 * H_OX, 21)
    rise = tmap.at(x, np.full_like(x, 1.5))
    slope = np.polyfit(x, np.log(rise), 1)[0]
    assert -1.0 / slope == pytest.approx(2 * H_OX / math.pi, rel=0.02)


# --------------------------------------------------------------- element degree


@pytest.fixture(scope="module")
def uniform_by_degree() -> dict[int, ThermalResult]:
    """The uniform-oxide sim at fast precision, solved at every degree."""
    sim = uniform_oxide_sim()
    results = {}
    for degree in (1, 2, 3):
        sim.solver(precision="fast", degree=degree)
        results[degree] = sim.solve(wavelength_um=WAVELENGTH)
    return results


def test_degree_2_converges_closer_to_degree_3_than_degree_1(uniform_by_degree):
    x = np.array([1.0, 2.0, 4.0, 8.0])
    z = np.full_like(x, H_OX)
    rise = {
        d: r.temperature(power_mw=1.0).at(x, z) for d, r in uniform_by_degree.items()
    }
    error_1 = np.max(np.abs(rise[1] / rise[3] - 1))
    error_2 = np.max(np.abs(rise[2] / rise[3] - 1))
    assert error_2 < 1e-3
    assert error_2 < error_1 / 10


def test_a_degree_2_temperature_map_peaks_and_plots(uniform_by_degree):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    tmap = uniform_by_degree[2].temperature(power_mw=1.0)
    assert tmap.values.size > tmap.mesh.mesh.p.shape[1]
    assert tmap.peak("heater") == pytest.approx(tmap.peak())
    assert tmap.peak("heater") == pytest.approx(
        uniform_by_degree[3].temperature(power_mw=1.0).peak("heater"), rel=1e-3
    )
    tmap.plot()
    plt.close("all")


def test_a_degree_2_step_response_matches_degree_1(canonical):
    sim, _result, step = canonical
    with solver_settings(sim, precision="fast", degree=2):
        quadratic = sim.solve_transient(power_mw=20.0, duration_us=60.0)
    assert quadratic.tau_us == pytest.approx(step.tau_us, rel=0.03)
    assert quadratic.phase_rad[-1] == pytest.approx(step.phase_rad[-1], rel=0.02)


def test_the_solver_rejects_an_unsupported_degree():
    with pytest.raises(ValueError, match="degree"):
        Solver(degree=4)  # type: ignore[arg-type]


# --------------------------------------------------------------- optical order


def test_the_optical_mode_solve_defaults_to_order_2():
    assert Solver().optical_order == 2


def test_the_solver_rejects_an_optical_order_femwell_lacks():
    with pytest.raises(ValueError, match="optical_order"):
        Solver(optical_order=3)  # type: ignore[arg-type]


def test_changing_the_optical_order_re_solves_and_agrees_on_p_pi(canonical):
    sim, result, _step = canonical
    with solver_settings(sim, optical_order=1):
        linear = sim.solve(wavelength_um=WAVELENGTH)
    assert linear is not result
    assert linear.p_pi_mw != result.p_pi_mw
    assert linear.p_pi_mw == pytest.approx(result.p_pi_mw, rel=0.03)


# --------------------------------------------------------------- transient


def test_the_step_response_settles_to_the_steady_phase(canonical):
    _sim, result, step = canonical
    assert step.time_us[0] == 0.0
    assert step.time_us[-1] == pytest.approx(60.0)
    assert step.phase_rad[0] == 0.0
    steady = result.phase(20.0)["arm"]
    assert step.steady_phase_rad == pytest.approx(steady, rel=1e-9)
    assert step.phase_rad[-1] == pytest.approx(steady, rel=0.01)
    assert step.tau_1e_us < step.tau_us
    assert set(step.temperature_k) == {"arm", "neighbour"}
    # The arm under the heater warms first and most.
    assert step.temperature_k["arm"][-1] > step.temperature_k["neighbour"][-1]


def test_tau_is_stable_under_precision(canonical):
    sim, _result, step = canonical
    with solver_settings(sim, precision="default"):
        finer = sim.solve_transient(power_mw=20.0, duration_us=60.0)
    assert finer.tau_us == pytest.approx(step.tau_us, rel=0.03)


def test_a_too_short_transient_warns_to_lengthen_it(canonical):
    sim, _result, _step = canonical
    with pytest.warns(UserWarning, match="duration"):
        sim.solve_transient(power_mw=20.0, duration_us=2.0)


# --------------------------------------------------------------- errors


def test_a_waveguide_where_only_the_heater_is_drawn_is_an_error():
    sim = femwell_sim()
    sim.solver(precision="fast")
    # Between the core and the heater edge only the heater is drawn, and
    # a heater is never a waveguide.
    sim.waveguides.add(name="ghost", position_um=0.8)
    with pytest.raises(ValueError, match="no drawn"):
        sim.solve(wavelength_um=WAVELENGTH)


def test_a_waveguide_core_without_dn_dt_is_an_error():
    sim = femwell_sim()
    sim.materials["si_core"] = {"dn_dT": 0.0}
    sim.solver(precision="fast")
    with pytest.raises(ValueError, match=r"dn_dT.*si_core"):
        sim.solve(wavelength_um=WAVELENGTH)


def test_missing_femwell_names_the_extra(monkeypatch):
    sim = femwell_sim()
    monkeypatch.setitem(sys.modules, "femwell", None)
    with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
        sim.solve(wavelength_um=WAVELENGTH)


def test_unknown_material_properties_are_rejected():
    sim = Simulation()
    with pytest.raises(ValueError, match="thermal_conductivty"):
        sim.materials["oxide"] = {"thermal_conductivty": 1.0}


def test_a_material_override_is_held_to_the_material_constraints():
    sim = Simulation()
    with pytest.raises(ValueError, match="mass_density"):
        sim.materials["oxide"] = {"mass_density": -1.0}


def test_the_solver_rejects_an_unknown_precision():
    with pytest.raises(ValueError, match="precision"):
        Solver(precision="ultra")  # type: ignore[arg-type]


def test_solving_with_no_waveguide_says_what_is_missing():
    sim = Simulation()
    sim.geometry(heater_over_strip(), stack=femwell_stack())
    sim.domain(plane="x=0", window=(-8.0, 8.0))
    sim.heater(layers="heater", length_um=320.0)
    with pytest.raises(ValueError, match=r"waveguides\.add"):
        sim.solve(wavelength_um=WAVELENGTH)


# --------------------------------------------------------------- setup changes


def test_editing_a_concern_in_place_invalidates_the_kept_solve():
    sim = femwell_sim(length=320.0)
    sim.solver(precision="fast")
    first = sim.solve(wavelength_um=WAVELENGTH)

    sim.heater.length_um = 640.0
    longer = sim.solve(wavelength_um=WAVELENGTH)
    assert longer.length_um == 640.0
    assert longer.resistance_ohm == pytest.approx(2.0 * first.resistance_ohm, rel=1e-9)

    sim.materials["tin"] = {"conductivity": SIGMA_TIN / 2}
    halved = sim.solve(wavelength_um=WAVELENGTH)
    assert halved.resistance_ohm == pytest.approx(4.0 * first.resistance_ohm, rel=1e-9)


def test_a_transient_only_solver_change_keeps_the_steady_solve(canonical):
    sim, _result, _step = canonical
    first = sim.solve(wavelength_um=WAVELENGTH)
    assert sim.solve(wavelength_um=WAVELENGTH) is first
    with solver_settings(sim, time_steps=50, theta=0.5):
        assert sim.solve(wavelength_um=WAVELENGTH) is first


def test_an_edit_to_the_component_in_place_is_seen():
    sim = femwell_sim()
    sim.solver(precision="fast")
    first = sim.solve(wavelength_um=WAVELENGTH)
    # A second heater strip beside the first, outside the arm's optical
    # Window: the conductors are in parallel, so R halves.
    _rect(sim.geometry.component, 5.0, HEATER_W, (47, 0))
    second = sim.solve(wavelength_um=WAVELENGTH)
    assert second.resistance_ohm == pytest.approx(first.resistance_ohm / 2, rel=1e-9)


def test_the_concerns_accept_mappings_and_validate_them():
    sim = Simulation(
        geometry={"component": heater_over_strip(), "stack": femwell_stack()},
        domain={"plane": "x=0", "window": (-8.0, 8.0)},
        heater={"layers": "heater", "length_um": 320.0},
        waveguides=[{"name": "arm", "position_um": 0.0}],
        solver={"precision": "fast"},
    )
    assert sim.heater.layers == ("heater",)
    assert sim.waveguides.phase_shifter is not None
    assert sim.waveguides.phase_shifter.name == "arm"
    assert "Simulation(geometry=" in repr(sim)
    with pytest.raises(ValueError, match="already exists"):
        sim.waveguides.add(name="arm", position_um=1.0)
    with pytest.raises(ValueError, match="vertical"):
        sim.domain(plane="z=0")
    with pytest.raises(ValueError, match="ascending"):
        sim.domain(window=(8.0, -8.0))
    with pytest.raises(ValueError, match="greater than 0"):
        sim.heater.length_um = 0.0
    with pytest.raises(ValueError, match="LayerStack"):
        sim.geometry(stack="stack.yaml")


def _heater_strips_sim(strips, window=(-8.0, 8.0), layers="heater") -> Simulation:
    """The femwell case with the heater drawn as ``(centre, width)`` strips."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    _rect(c, 0.0, CORE_W, (1, 0))
    for centre, width in strips:
        _rect(c, centre, width, (47, 0))
    sim = Simulation()
    sim.geometry(c, stack=femwell_stack())
    sim.domain(plane="x=0", window=window)
    sim.heater(layers=layers, length_um=320.0)
    sim.waveguides.add(name="arm", position_um=0.0)
    return sim


@pytest.mark.parametrize(
    ("strips", "extent"),
    [
        ([(0.0, 2.0), (7.5, 2.0)], r"\[6\.5, 8\.5\]"),  # half outside
        ([(0.0, 2.0), (12.0, 2.0)], r"\[11, 13\]"),  # wholly outside
    ],
)
def test_a_heater_conductor_the_window_cuts_is_an_error_naming_the_remedy(
    strips, extent
):
    sim = _heater_strips_sim(strips)
    sim.solver(precision="fast")
    with pytest.raises(ValueError, match=extent) as error:
        sim.solve(wavelength_um=WAVELENGTH)
    message = str(error.value)
    assert "'heater'" in message
    assert "(-8, 8)" in message
    assert "sim.domain(window=" in message


def test_two_heater_strips_conduct_in_parallel():
    sim = _heater_strips_sim([(-1.5, 1.0), (1.5, 1.0)])
    sim.solver(precision="fast")
    result = sim.solve(wavelength_um=WAVELENGTH)
    # Same total conductor cross-section as the single 2 um strip.
    by_hand = 320e-6 / (SIGMA_TIN * 2 * 1e-6 * HEATER_T * 1e-6)
    assert result.resistance_ohm == pytest.approx(by_hand, rel=1e-9)


def test_a_heater_layer_the_cut_misses_is_left_out_with_a_warning():
    sim = _heater_strips_sim([(0.0, HEATER_W)], layers=["heater", "lead"])
    stack_of(sim).layers["lead"] = _layer(
        "lead", (49, 0), 2.4, 2.6, "tin", kind="conductor"
    )
    sim.solver(precision="fast")
    with pytest.warns(UserWarning, match="lead"):
        result = sim.solve(wavelength_um=WAVELENGTH)
    assert result.peak_temperature_k == pytest.approx(
        result.temperature(result.p_pi_mw).peak("heater")
    )
    assert "Peak heater rise" in str(result.report())


# --------------------------------------------------------------- result read-outs


def test_crosstalk_ratio_is_the_victim_phase_over_the_shifter_phase(canonical):
    _sim, result, _step = canonical
    # At P_pi the phase shifter turns exactly pi.
    assert result.crosstalk_ratio == {
        "neighbour": pytest.approx(result.crosstalk_rad["neighbour"] / math.pi)
    }
    assert 0.0 < result.crosstalk_ratio["neighbour"] < 1.0
    assert "Crosstalk ratio, neighbour" in str(result.report())


def test_temperatures_are_rises_unless_an_ambient_is_given(femwell_result):
    tmap = femwell_result.temperature(power_mw=10.0)
    rise = tmap.at(0.0, 0.11)
    assert 0.0 < rise < 50.0
    assert tmap.at(0.0, 0.11, ambient_k=293.15) == pytest.approx(293.15 + rise)
    np.testing.assert_allclose(
        tmap.at([0.0, 1.0], [0.11, 0.11], ambient_k=300.0),
        300.0 + tmap.at([0.0, 1.0], [0.11, 0.11]),
    )
    assert tmap.peak(ambient_k=300.0) == pytest.approx(300.0 + tmap.peak())
    assert tmap.peak("heater", ambient_k=300.0) == pytest.approx(
        300.0 + tmap.peak("heater")
    )
    with pytest.raises(ValueError, match="kelvin"):
        tmap.peak(ambient_k=-20.0)


def test_drive_power_from_a_voltage_or_a_current(femwell_result):
    r = femwell_result.resistance_ohm
    # V = 5 V across R: V^2 / R; I = 10 mA through R: I^2 R (in mW).
    assert femwell_result.drive_power(voltage_v=5.0) == pytest.approx(25.0 / r * 1e3)
    assert femwell_result.drive_power(current_a=0.01) == pytest.approx(1e-4 * r * 1e3)
    assert femwell_result.drive_power(voltage_v=femwell_result.v_pi_v) == pytest.approx(
        femwell_result.p_pi_mw
    )
    with pytest.raises(ValueError, match="exactly one"):
        femwell_result.drive_power(voltage_v=1.0, current_a=0.01)
    with pytest.raises(ValueError, match="exactly one"):
        femwell_result.drive_power()


def test_drive_power_per_length_is_public():
    from gsim.thermal import drive_power_per_length

    assert drive_power_per_length(1.0, current_A=2.0) == pytest.approx(4.0)


# --------------------------------------------------------------- materials


def test_an_anisotropic_conductivity_matters_only_on_the_heater():
    sim = femwell_sim()
    sim.solver(precision="fast")
    stack_of(sim).materials["si_core"]["conductivity"] = [1e3, 1e3, 10.0]
    assert 15.0 < sim.solve(wavelength_um=WAVELENGTH).p_pi_mw < 30.0

    sim = femwell_sim()
    sim.solver(precision="fast")
    stack_of(sim).materials["tin"]["conductivity"] = [SIGMA_TIN, SIGMA_TIN, 1e3]
    with pytest.raises(ValueError, match=r"heater.*anisotropic"):
        sim.solve(wavelength_um=WAVELENGTH)


def test_an_optical_override_skips_an_invalid_stack_entry_like_the_rest():
    sim = femwell_sim()
    # Not a valid MaterialProperties record: the thermal path falls back to
    # the database's oxide, and so must the optical override.
    stack_of(sim).materials["oxide"]["mass_density"] = -1.0
    sim.materials["oxide"] = {"permittivity": 1.444**2}
    sim.solver(precision="fast")
    result = sim.solve(wavelength_um=WAVELENGTH)
    assert 15.0 < result.p_pi_mw < 30.0


def test_a_material_override_replaces_one_spelled_in_another_case():
    sim = Simulation()
    sim.materials["TiN"] = {"conductivity": 1e6}
    sim.materials["tin"] = {"thermal_conductivity": 20.0}
    assert list(sim.materials.overrides) == ["tin"]
    assert sim.materials["TIN"].properties() == {"thermal_conductivity": 20.0}
    with pytest.raises(ValueError, match="several spellings"):
        sim.materials(overrides={"TiN": {}, "tin": {}})


def test_a_setup_change_warns_again_about_a_material_without_dn_dt():
    sim = femwell_sim()
    stack_of(sim).dielectrics[2]["material"] = "clad_ox"
    stack_of(sim).materials["clad_ox"] = {
        "permittivity": 1.444**2,
        "thermal_conductivity": 1.38,
        **_OX,
    }
    sim.solver(precision="fast")
    with pytest.warns(UserWarning, match="clad_ox"):
        sim.solve(wavelength_um=WAVELENGTH)
    sim.heater.length_um = 330.0
    with pytest.warns(UserWarning, match="clad_ox"):
        sim.solve(wavelength_um=WAVELENGTH)


# --------------------------------------------------------------- boundaries, cut


def test_the_boundary_keeps_what_a_call_leaves_out():
    sim = Simulation()
    sim.boundary(top="convective", heat_transfer_coefficient=20.0)
    sim.boundary(sides="convective")
    edges = sim.boundary.convective_edges()
    assert edges == {"top": 20.0, "left": 20.0, "right": 20.0}
    sim.boundary(top="adiabatic")
    assert sim.boundary.convective_edges() == {"left": 20.0, "right": 20.0}
    sim.boundary(heat_transfer_coefficient=5.0)
    assert sim.boundary.convective_edges() == {"left": 5.0, "right": 5.0}


def test_a_waveguide_on_the_shared_edge_of_two_touching_strips_is_found():
    gf.gpdk.PDK.activate()
    c = gf.Component()
    _rect(c, -CORE_W / 4, CORE_W / 2, (1, 0))
    _rect(c, CORE_W / 4, CORE_W / 2, (1, 0))
    _rect(c, 0.0, HEATER_W, (47, 0))
    sim = Simulation()
    sim.geometry(c, stack=femwell_stack())
    sim.domain(plane="x=0", window=(-8.0, 8.0))
    sim.heater(layers="heater", length_um=320.0)
    sim.waveguides.add(name="arm", position_um=0.0)
    assert [wg.name for wg in sim.waveguides] == ["arm"]


# --------------------------------------------------------------- plots


def test_the_step_plot_marks_the_10_90_points_and_the_1e_time(canonical):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _sim, _result, step = canonical
    ax = step.plot()
    marks = {line.get_label(): line for line in ax.get_lines()}
    t10, t90 = marks["10 % and 90 %"].get_xdata()
    assert t90 - t10 == pytest.approx(step.tau_us)
    np.testing.assert_allclose(
        marks["10 % and 90 %"].get_ydata(),
        [0.1 * step.phase_rad[-1], 0.9 * step.phase_rad[-1]],
    )
    (t_1e,) = marks["1/e"].get_xdata()
    assert t_1e == pytest.approx(step.tau_1e_us)
    plt.close("all")


def test_the_temperature_plot_zooms_to_given_limits(canonical):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _sim, result, _step = canonical
    ax = result.temperature(power_mw=result.p_pi_mw).plot(
        xlim=(-3.0, 3.0), zlim=(-1.0, 3.0)
    )
    assert ax.get_xlim() == (-3.0, 3.0)
    assert ax.get_ylim() == (-1.0, 3.0)
    plt.close("all")
