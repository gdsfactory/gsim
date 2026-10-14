"""Tests for the physics-noun layer of gsim.thermal (ThermalSim and results)."""

from __future__ import annotations

import math
import sys

import gdsfactory as gf
import numpy as np
import pytest
from scipy.integrate import quad

from gsim.common.stack.extractor import Layer, LayerStack

pytest.importorskip("skfem")
pytest.importorskip("femwell")

from gsim.thermal import (
    HeaterReport,
    StepResponse,
    TemperatureMap,
    ThermalOptions,
    ThermalResult,
    ThermalSim,
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


def femwell_sim(length: float = 320.0, neighbour: float | None = None) -> ThermalSim:
    """The femwell TiN case, 16 um wide, configured but not solved."""
    sim = ThermalSim()
    sim.set_geometry(heater_over_strip(neighbour))
    sim.set_stack(femwell_stack())
    sim.set_cross_section("x=0", window=(-8.0, 8.0))
    sim.set_heater("heater", length=length)
    sim.add_waveguide("arm", position=0.0)
    if neighbour is not None:
        sim.add_waveguide("neighbour", position=neighbour)
    return sim


# --------------------------------------------------------------- canonical script


@pytest.fixture(scope="module")
def canonical():
    """The canonical script of ticket 06, on a TiN heater with a neighbour."""
    sim = ThermalSim()
    sim.set_geometry(heater_over_strip(neighbour=5.0))
    sim.set_stack(femwell_stack())
    sim.set_cross_section("x=0", window=(-15.0, 15.0))
    sim.set_heater("heater", length=100.0)
    sim.add_waveguide("arm", position=0.0)
    sim.add_waveguide("neighbour", position=5.0)

    fast = ThermalOptions(precision="fast")
    result = sim.solve(wavelength=WAVELENGTH, options=fast)
    step = sim.solve_transient(power=20.0, duration=60.0, options=fast)
    return sim, result, step


def test_the_canonical_script_gives_labelled_physics(canonical):
    _sim, result, step = canonical

    assert isinstance(result, ThermalResult)
    assert 15.0 < result.p_pi < 30.0
    assert result.efficiency == pytest.approx(math.pi / result.p_pi)
    assert 0.0 < result.crosstalk["neighbour"] < math.pi
    assert result.v_pi == pytest.approx(
        math.sqrt(result.p_pi * 1e-3 * result.resistance)
    )
    tmap = result.temperature(power=result.p_pi)
    assert isinstance(tmap, TemperatureMap)
    # The heater is the hottest thing in the cross-section.
    assert tmap.peak("heater") == pytest.approx(tmap.peak())

    assert isinstance(step, StepResponse)
    assert step.tau > 0.0
    report = result.report(step)
    assert isinstance(report, HeaterReport)
    text = str(report)
    for label in ("P_pi (mW)", "V_pi (V)", "tau 10-90 % (us)", "neighbour"):
        assert label in text
    assert "<table" in report._repr_html_()


def test_the_phase_shifter_picks_up_exactly_pi_at_p_pi(canonical):
    _sim, result, _step = canonical
    phase = result.phase(result.p_pi)
    assert phase["arm"] == pytest.approx(math.pi, rel=1e-9)
    assert phase["neighbour"] == pytest.approx(result.crosstalk["neighbour"])


def test_the_temperature_map_and_step_response_plot_on_matplotlib_axes(canonical):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _sim, result, step = canonical
    ax = result.temperature(power=result.p_pi).plot()
    assert ax.get_xlabel().startswith("y")
    assert step.plot().get_xlabel() == "Time (µs)"
    plt.close("all")


# --------------------------------------------------------------- unit consistency


@pytest.fixture(scope="module")
def femwell_result() -> ThermalResult:
    """The femwell TiN case at default precision."""
    return femwell_sim().solve(wavelength=WAVELENGTH)


def test_femwell_tin_heater_p_pi_matches_femwell_within_3_percent(femwell_result):
    assert femwell_result.p_pi == pytest.approx(FEMWELL_P_PI_MW, rel=0.03)


def test_resistance_is_length_over_conductance_of_the_drawn_heater(femwell_result):
    by_hand = 320e-6 / (SIGMA_TIN * HEATER_W * 1e-6 * HEATER_T * 1e-6)
    assert femwell_result.resistance == pytest.approx(by_hand, rel=1e-9)
    assert by_hand == pytest.approx(496.9, abs=0.1)


def test_doubling_power_doubles_the_temperature_rise(femwell_result):
    one = femwell_result.temperature(power=10.0)
    two = femwell_result.temperature(power=20.0)
    np.testing.assert_allclose(two.values, 2.0 * one.values, rtol=1e-12)
    assert two.at(0.0, 0.11) == pytest.approx(2.0 * one.at(0.0, 0.11))


def test_thermal_resistance_is_the_core_rise_per_milliwatt(femwell_result):
    # Core centre rise per mW of total heater power over 320 um. From the
    # map at 1 mW: same number, different route through the units.
    rise = femwell_result.temperature(power=1.0).at(0.0, CORE_T / 2)
    assert femwell_result.thermal_resistance["arm"] == pytest.approx(rise)
    # Order of magnitude by hand: dT_pi = lambda / (2 L dn/dT) = 13 K at P_pi.
    assert femwell_result.thermal_resistance["arm"] == pytest.approx(
        13.0 / FEMWELL_P_PI_MW, rel=0.15
    )


def test_p_pi_does_not_depend_on_the_heater_length(femwell_result):
    short = femwell_sim(length=100.0).solve(wavelength=WAVELENGTH)
    assert short.p_pi == pytest.approx(femwell_result.p_pi, rel=2e-3)
    assert short.resistance == pytest.approx(femwell_result.resistance * 100 / 320)


def test_p_pi_scales_with_the_oxide_conductivity_override(femwell_result):
    sim = femwell_sim()
    sim.set_material("oxide", thermal_conductivity=2 * 1.38)
    stiffer = sim.solve(wavelength=WAVELENGTH, options=ThermalOptions("fast"))
    # Heat leaves through the oxide: P_pi ~ k_oxide to first order.
    assert 1.8 < stiffer.p_pi / femwell_result.p_pi < 2.05


# --------------------------------------------------------------- analytic references

H_OX = 3.0  # oxide thickness between the adiabatic top and the sink (um)
K_OX = 1.38
STRIP_W = 0.4


def uniform_oxide_sim() -> ThermalSim:
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
    sim = ThermalSim()
    sim.set_geometry(c)
    sim.set_stack(stack)
    sim.set_cross_section("x=0", window=(-8 * H_OX, 8 * H_OX))
    sim.set_heater("heater", length=100.0)
    sim.add_waveguide("probe", position=-6.0)
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
    return uniform_oxide_sim().solve(wavelength=WAVELENGTH)


def test_surface_rise_matches_the_ln_coth_strip_solution(uniform_result):
    power = 1.0  # mW over 100 um: P' = 10 W/m
    tmap = uniform_result.temperature(power=power)
    x = np.linspace(1.0, 2.0 * H_OX, 11)
    fem = tmap.at(x, np.full_like(x, H_OX))
    exact = np.array([strip_on_surface(xi, 10.0) for xi in x])
    np.testing.assert_allclose(fem, exact, rtol=0.01)


def test_crosstalk_tail_decays_over_2h_over_pi(uniform_result):
    tmap = uniform_result.temperature(power=1.0)
    x = np.linspace(2 * H_OX, 4 * H_OX, 21)
    rise = tmap.at(x, np.full_like(x, 1.5))
    slope = np.polyfit(x, np.log(rise), 1)[0]
    assert -1.0 / slope == pytest.approx(2 * H_OX / math.pi, rel=0.02)


# --------------------------------------------------------------- transient


def test_the_step_response_settles_to_the_steady_phase(canonical):
    _sim, result, step = canonical
    assert step.time[0] == 0.0
    assert step.time[-1] == pytest.approx(60.0)
    assert step.phase[0] == 0.0
    steady = result.phase(20.0)["arm"]
    assert step.steady_phase == pytest.approx(steady, rel=1e-9)
    assert step.phase[-1] == pytest.approx(steady, rel=0.01)
    assert step.tau_1e < step.tau
    assert set(step.temperature) == {"arm", "neighbour"}
    # The arm under the heater warms first and most.
    assert step.temperature["arm"][-1] > step.temperature["neighbour"][-1]


def test_tau_is_stable_under_precision(canonical):
    sim, _result, step = canonical
    finer = sim.solve_transient(power=20.0, duration=60.0)
    assert finer.tau == pytest.approx(step.tau, rel=0.03)


def test_a_too_short_transient_warns_to_lengthen_it(canonical):
    sim, _result, _step = canonical
    with pytest.warns(UserWarning, match="duration"):
        sim.solve_transient(
            power=20.0, duration=2.0, options=ThermalOptions(precision="fast")
        )


# --------------------------------------------------------------- errors


def test_a_waveguide_where_only_the_heater_is_drawn_is_an_error():
    sim = femwell_sim()
    # Between the core and the heater edge only the heater is drawn, and
    # a heater is never a waveguide.
    with pytest.raises(ValueError, match="no drawn"):
        sim.add_waveguide("ghost", position=0.8)


def test_a_waveguide_core_without_dn_dt_is_an_error():
    sim = femwell_sim()
    sim.set_material("si_core", dn_dT=0.0)
    with pytest.raises(ValueError, match=r"dn_dT.*si_core"):
        sim.solve(wavelength=WAVELENGTH, options=ThermalOptions("fast"))


def test_missing_femwell_names_the_extra(monkeypatch):
    sim = femwell_sim()
    monkeypatch.setitem(sys.modules, "femwell", None)
    with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
        sim.solve(wavelength=WAVELENGTH)


def test_unknown_material_properties_are_rejected():
    sim = ThermalSim()
    with pytest.raises(ValueError, match="thermal_conductivity"):
        sim.set_material("oxide", thermal_conductivty=1.0)


def test_options_reject_an_unknown_precision():
    with pytest.raises(ValueError, match="precision"):
        ThermalOptions(precision="ultra")  # type: ignore[arg-type]


def test_solving_with_no_waveguide_says_what_is_missing():
    sim = ThermalSim()
    sim.set_geometry(heater_over_strip())
    sim.set_stack(femwell_stack())
    sim.set_cross_section("x=0", window=(-8.0, 8.0))
    sim.set_heater("heater", length=320.0)
    with pytest.raises(ValueError, match="add_waveguide"):
        sim.solve(wavelength=WAVELENGTH)


# --------------------------------------------------------------- setup changes


def test_assigning_a_setup_field_directly_invalidates_the_cached_solve():
    fast = ThermalOptions(precision="fast")
    sim = femwell_sim(length=320.0)
    first = sim.solve(wavelength=WAVELENGTH, options=fast)

    sim.heater_length = 640.0
    longer = sim.solve(wavelength=WAVELENGTH, options=fast)
    assert longer.length == 640.0
    assert longer.resistance == pytest.approx(2.0 * first.resistance, rel=1e-9)

    sim.materials = {"tin": {"conductivity": SIGMA_TIN / 2}}
    halved = sim.solve(wavelength=WAVELENGTH, options=fast)
    assert halved.resistance == pytest.approx(4.0 * first.resistance, rel=1e-9)


def _heater_strips_sim(strips, window=(-8.0, 8.0), layers="heater") -> ThermalSim:
    """The femwell case with the heater drawn as ``(centre, width)`` strips."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    _rect(c, 0.0, CORE_W, (1, 0))
    for centre, width in strips:
        _rect(c, centre, width, (47, 0))
    sim = ThermalSim()
    sim.set_geometry(c)
    sim.set_stack(femwell_stack())
    sim.set_cross_section("x=0", window=window)
    sim.set_heater(layers, length=320.0)
    sim.add_waveguide("arm", position=0.0)
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
    with pytest.raises(ValueError, match=extent) as error:
        sim.solve(wavelength=WAVELENGTH, options=ThermalOptions("fast"))
    message = str(error.value)
    assert "'heater'" in message
    assert "(-8, 8)" in message
    assert "set_cross_section" in message


def test_two_heater_strips_conduct_in_parallel():
    sim = _heater_strips_sim([(-1.5, 1.0), (1.5, 1.0)])
    result = sim.solve(wavelength=WAVELENGTH, options=ThermalOptions("fast"))
    # Same total conductor cross-section as the single 2 um strip.
    by_hand = 320e-6 / (SIGMA_TIN * 2 * 1e-6 * HEATER_T * 1e-6)
    assert result.resistance == pytest.approx(by_hand, rel=1e-9)


def test_a_heater_layer_the_cut_misses_is_left_out_with_a_warning():
    sim = _heater_strips_sim([(0.0, HEATER_W)], layers=["heater", "lead"])
    sim.stack.layers["lead"] = _layer(
        "lead", (49, 0), 2.4, 2.6, "tin", kind="conductor"
    )
    with pytest.warns(UserWarning, match="lead"):
        result = sim.solve(wavelength=WAVELENGTH, options=ThermalOptions("fast"))
    assert result.peak_temperature == pytest.approx(
        result.temperature(result.p_pi).peak("heater")
    )
    assert "Peak heater rise" in str(result.report())


# --------------------------------------------------------------- result read-outs


def test_crosstalk_ratio_is_the_victim_phase_over_the_shifter_phase(canonical):
    _sim, result, _step = canonical
    # At P_pi the phase shifter turns exactly pi.
    assert result.crosstalk_ratio == {
        "neighbour": pytest.approx(result.crosstalk["neighbour"] / math.pi)
    }
    assert 0.0 < result.crosstalk_ratio["neighbour"] < 1.0
    assert "Crosstalk ratio, neighbour" in str(result.report())


def test_temperatures_are_rises_unless_an_ambient_is_given(femwell_result):
    tmap = femwell_result.temperature(power=10.0)
    rise = tmap.at(0.0, 0.11)
    assert 0.0 < rise < 50.0
    assert tmap.at(0.0, 0.11, ambient=293.15) == pytest.approx(293.15 + rise)
    np.testing.assert_allclose(
        tmap.at([0.0, 1.0], [0.11, 0.11], ambient=300.0),
        300.0 + tmap.at([0.0, 1.0], [0.11, 0.11]),
    )
    assert tmap.peak(ambient=300.0) == pytest.approx(300.0 + tmap.peak())
    assert tmap.peak("heater", ambient=300.0) == pytest.approx(
        300.0 + tmap.peak("heater")
    )
    with pytest.raises(ValueError, match="kelvin"):
        tmap.peak(ambient=-20.0)


def test_drive_power_from_a_voltage_or_a_current(femwell_result):
    r = femwell_result.resistance
    # V = 5 V across R: V^2 / R; I = 10 mA through R: I^2 R (in mW).
    assert femwell_result.drive_power(voltage=5.0) == pytest.approx(25.0 / r * 1e3)
    assert femwell_result.drive_power(current=0.01) == pytest.approx(1e-4 * r * 1e3)
    assert femwell_result.drive_power(voltage=femwell_result.v_pi) == pytest.approx(
        femwell_result.p_pi
    )
    with pytest.raises(ValueError, match="exactly one"):
        femwell_result.drive_power(voltage=1.0, current=0.01)
    with pytest.raises(ValueError, match="exactly one"):
        femwell_result.drive_power()


def test_drive_power_per_length_is_public():
    from gsim.thermal import drive_power_per_length

    assert drive_power_per_length(1.0, current_A=2.0) == pytest.approx(4.0)


# --------------------------------------------------------------- materials


def test_an_anisotropic_conductivity_matters_only_on_the_heater():
    fast = ThermalOptions(precision="fast")
    sim = femwell_sim()
    sim.stack.materials["si_core"]["conductivity"] = [1e3, 1e3, 10.0]
    assert 15.0 < sim.solve(wavelength=WAVELENGTH, options=fast).p_pi < 30.0

    sim = femwell_sim()
    sim.stack.materials["tin"]["conductivity"] = [SIGMA_TIN, SIGMA_TIN, 1e3]
    with pytest.raises(ValueError, match=r"heater.*anisotropic"):
        sim.solve(wavelength=WAVELENGTH, options=fast)


def test_an_optical_override_skips_an_invalid_stack_entry_like_the_rest():
    sim = femwell_sim()
    # Not a valid MaterialProperties record: the thermal path falls back to
    # the database's oxide, and so must the optical override.
    sim.stack.materials["oxide"]["mass_density"] = -1.0
    sim.set_material("oxide", permittivity=1.444**2)
    result = sim.solve(wavelength=WAVELENGTH, options=ThermalOptions("fast"))
    assert 15.0 < result.p_pi < 30.0


def test_set_material_merges_names_that_differ_only_in_case():
    sim = ThermalSim()
    sim.set_material("TiN", conductivity=1e6)
    sim.set_material("tin", thermal_conductivity=20.0)
    assert sim.materials == {"TiN": {"conductivity": 1e6, "thermal_conductivity": 20.0}}


def test_a_setup_change_warns_again_about_a_material_without_dn_dt():
    sim = femwell_sim()
    sim.stack.dielectrics[2]["material"] = "clad_ox"
    sim.stack.materials["clad_ox"] = {
        "permittivity": 1.444**2,
        "thermal_conductivity": 1.38,
        **_OX,
    }
    fast = ThermalOptions(precision="fast")
    with pytest.warns(UserWarning, match="clad_ox"):
        sim.solve(wavelength=WAVELENGTH, options=fast)
    sim.heater_length = 330.0
    with pytest.warns(UserWarning, match="clad_ox"):
        sim.solve(wavelength=WAVELENGTH, options=fast)


# --------------------------------------------------------------- boundaries, cut


def test_set_boundary_keeps_what_a_call_leaves_out():
    sim = ThermalSim()
    sim.set_boundary(top="convective", heat_transfer_coefficient=20.0)
    sim.set_boundary(sides="convective")
    assert sim.boundary == {"top": 20.0, "left": 20.0, "right": 20.0}
    sim.set_boundary(top="adiabatic")
    assert sim.boundary == {"left": 20.0, "right": 20.0}
    sim.set_boundary(heat_transfer_coefficient=5.0)
    assert sim.boundary == {"left": 5.0, "right": 5.0}


def test_a_waveguide_on_the_shared_edge_of_two_touching_strips_is_found():
    gf.gpdk.PDK.activate()
    c = gf.Component()
    _rect(c, -CORE_W / 4, CORE_W / 2, (1, 0))
    _rect(c, CORE_W / 4, CORE_W / 2, (1, 0))
    _rect(c, 0.0, HEATER_W, (47, 0))
    sim = ThermalSim()
    sim.set_geometry(c)
    sim.set_stack(femwell_stack())
    sim.set_cross_section("x=0", window=(-8.0, 8.0))
    sim.set_heater("heater", length=320.0)
    sim.add_waveguide("arm", position=0.0)
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
    assert t90 - t10 == pytest.approx(step.tau)
    np.testing.assert_allclose(
        marks["10 % and 90 %"].get_ydata(),
        [0.1 * step.phase[-1], 0.9 * step.phase[-1]],
    )
    (t_1e,) = marks["1/e"].get_xdata()
    assert t_1e == pytest.approx(step.tau_1e)
    plt.close("all")


def test_the_temperature_plot_zooms_to_given_limits(canonical):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    _sim, result, _step = canonical
    ax = result.temperature(power=result.p_pi).plot(xlim=(-3.0, 3.0), zlim=(-1.0, 3.0))
    assert ax.get_xlim() == (-3.0, 3.0)
    assert ax.get_ylim() == (-1.0, 3.0)
    plt.close("all")
