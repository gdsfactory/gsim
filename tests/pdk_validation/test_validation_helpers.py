"""Unit tests for the MEEP validation harness (:mod:`_validation`).

These exercise the pure post-processing maths (de-embedding, normalisation,
agreement metrics) and the simulation *configuration*; none of them run MEEP
and none require ``pymeep``.  Tests that need a PDK skip cleanly when it is
not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.pdk_validation import _validation as validation


class _Port:
    def __init__(self, name: str) -> None:
        self.name = name


class _Component:
    def __init__(self, *names: str) -> None:
        self.ports = [_Port(name) for name in names]


class _Result:
    """Minimal stand-in for SParameterResult."""

    def __init__(self, wavelengths, s_params, debug_info=None):
        self.wavelengths = list(wavelengths)
        self.s_params = {k: list(v) for k, v in s_params.items()}
        self.debug_info = debug_info or {}


# ---------------------------------------------------------------------------
# Facts derived from device sizes in square wavelengths
# ---------------------------------------------------------------------------


def test_lambda_squared_reference_value():
    lam = validation.VALIDATION_WAVELENGTH_UM
    assert lam == 1.55
    assert math.isclose(lam**2, 2.4025, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Port monitor offsets
# ---------------------------------------------------------------------------


def test_monitor_distances_source_gets_extra_offset():
    component = _Component("o1", "o2", "o3")
    distances = validation.monitor_distances(
        component,
        source_port="o1",
        source_port_offset=0.1,
        distance_source_to_monitors=0.2,
    )
    assert distances == {
        "o1": pytest.approx(0.3),
        "o2": pytest.approx(0.1),
        "o3": pytest.approx(0.1),
    }


def test_monitor_distances_defaults_to_first_port():
    component = _Component("o1", "o2")
    distances = validation.monitor_distances(component)
    assert distances["o1"] > distances["o2"]


# ---------------------------------------------------------------------------
# De-embedding
# ---------------------------------------------------------------------------


def test_s_param_indices():
    assert validation.s_param_indices("S11") == (0, 0)
    assert validation.s_param_indices("S41") == (3, 0)
    with pytest.raises(ValueError):
        validation.s_param_indices("x11")


def test_deembed_is_identity_without_offsets():
    result = _Result(
        [1.53, 1.55, 1.57],
        {"S11": [0.1 + 0j, 0.2 + 0j, 0.3 + 0j], "S21": [1 + 0j, 1 + 0j, 1 + 0j]},
        {"eigenmode_info": {"o1": {"n_eff": [2.4, 2.4, 2.4]}}},
    )
    out = validation.deembed_s_params(
        result, ["o1", "o2"], {"o1": 0.0, "o2": 0.0}, source_port="o1"
    )
    np.testing.assert_allclose(out["S11"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(out["S21"], [1.0, 1.0, 1.0])


def test_deembed_applies_expected_phase():
    wavelengths = np.array([1.55])
    result = _Result(
        wavelengths,
        {"S11": [1 + 0j]},
        {"eigenmode_info": {"o1": {"n_eff": [2.0]}}},
    )
    distances = {"o1": 0.1, "o2": 0.1}
    out = validation.deembed_s_params(result, ["o1", "o2"], distances, source_port="o1")
    k = 2 * math.pi / 1.55
    expected = np.exp(1j * k * (2.0 * 0.1 + 2.0 * 0.1))
    np.testing.assert_allclose(out["S11"], [expected], rtol=1e-12)


def test_deembed_sign_flips_phase():
    result = _Result(
        [1.55], {"S11": [1 + 0j]}, {"eigenmode_info": {"o1": {"n_eff": [2.0]}}}
    )
    distances = {"o1": 0.1, "o2": 0.1}
    plus = validation.deembed_s_params(result, ["o1", "o2"], distances, sign=1.0)[
        "S11"
    ][0]
    minus = validation.deembed_s_params(result, ["o1", "o2"], distances, sign=-1.0)[
        "S11"
    ][0]
    np.testing.assert_allclose(minus, np.conj(plus), rtol=1e-12)


def test_deembed_only_touches_source_row():
    result = _Result(
        [1.55],
        {"S12": [0.5 + 0j]},  # excited from o2 -> not the configured source
        {"eigenmode_info": {"o1": {"n_eff": [2.0]}, "o2": {"n_eff": [2.0]}}},
    )
    out = validation.deembed_s_params(
        result, ["o1", "o2"], {"o1": 0.1, "o2": 0.1}, source_port="o1"
    )
    np.testing.assert_allclose(out["S12"], [0.5 + 0j])


def test_deembed_falls_back_when_no_n_eff():
    result = _Result([1.55], {"S11": [1 + 0j]}, {})
    out = validation.deembed_s_params(
        result, ["o1", "o2"], {"o1": 0.1, "o2": 0.0}, source_port="o1"
    )
    k = 2 * math.pi / 1.55
    # S11 has both the receive (o1) and source (o1) monitors offset by 0.1 um.
    expected = np.exp(1j * k * validation.FALLBACK_N_EFF * (0.1 + 0.1))
    np.testing.assert_allclose(out["S11"], [expected], rtol=1e-12)


# ---------------------------------------------------------------------------
# Metrics and normalisation
# ---------------------------------------------------------------------------


def test_to_db():
    assert math.isclose(float(validation.to_db([1.0])[0]), 0.0, abs_tol=1e-9)
    assert math.isclose(float(validation.to_db([0.1])[0]), -20.0, rel_tol=1e-6)


def test_normalize_measured_spectrum_max():
    power = np.array([-30.0, -20.0, -10.0])
    out = validation.normalize_measured_spectrum(power, method="max")
    np.testing.assert_allclose(out, [-20.0, -10.0, 0.0])


def test_normalize_measured_spectrum_percentile_is_robust():
    rng = np.random.default_rng(0)
    power = -30.0 + rng.normal(0.0, 0.1, 200)
    power[0] = 5.0  # single launch spike
    robust = validation.normalize_measured_spectrum(
        power, method="percentile", percentile=99.0
    )
    naive = validation.normalize_measured_spectrum(power, method="max")
    # The percentile reference ignores the spike, so the baseline sits near 0;
    # the max reference is dragged down to the spike.
    assert abs(np.median(robust)) < 1.0
    assert np.median(naive) < -30.0


def test_agreement_identical_curves():
    wl = np.linspace(1.5, 1.6, 51)
    db = -3 * np.ones_like(wl)
    metrics = validation.agreement_db(wl, db, wl.copy(), db.copy())
    assert metrics is not None
    assert metrics["rms_db"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["max_abs_db"] == pytest.approx(0.0, abs=1e-9)


def test_agreement_constant_bias():
    wl = np.linspace(1.5, 1.6, 51)
    metrics = validation.agreement_db(wl, np.zeros_like(wl), wl, np.full_like(wl, 2.0))
    assert metrics is not None
    assert metrics["bias_db"] == pytest.approx(-2.0, abs=1e-9)
    assert metrics["rms_db"] == pytest.approx(2.0, abs=1e-9)


def test_agreement_no_overlap_returns_none():
    metrics = validation.agreement_db(
        np.array([1.5, 1.55, 1.6]), np.zeros(3), np.array([1.8, 1.9, 2.0]), np.zeros(3)
    )
    assert metrics is None


def test_agreement_normalize_removes_offset():
    wl = np.linspace(1.5, 1.6, 51)
    sim = -3 * np.ones_like(wl)  # flat
    ref = 7 * np.ones_like(wl)  # flat, offset by 10 dB
    metrics = validation.agreement_db(wl, sim, wl, ref, normalize=True)
    assert metrics is not None
    assert metrics["rms_db"] == pytest.approx(0.0, abs=1e-9)


def test_power_conservation():
    total = validation.power_conservation({"S21": [0.6], "S31": [0.8]})
    np.testing.assert_allclose(total, [1.0])


# ---------------------------------------------------------------------------
# Size in wavelengths and timing record
# ---------------------------------------------------------------------------


def test_simulation_size_in_wavelengths():
    size = validation.SimulationSize(
        cell_x_um=23.0, cell_y_um=3.5, cell_z_um=3.22, resolution=20
    )
    lam = validation.VALIDATION_WAVELENGTH_UM
    assert size.size_lambda == pytest.approx((23.0 / lam, 3.5 / lam, 3.22 / lam))
    assert size.area_lambda2 == pytest.approx(23.0 * 3.5 / lam**2)
    assert size.cells == 460 * 70 * 65


def test_simulation_size_2d_ignores_z():
    size = validation.SimulationSize(
        cell_x_um=10.0, cell_y_um=20.0, cell_z_um=0.0, resolution=10, dimension="2d"
    )
    assert size.cells == 100 * 200


def test_simulation_size_reads_runner_metadata():
    result = _Result(
        [1.5, 1.55, 1.6],
        {"S11": [1, 1, 1]},
        {
            "metadata": {
                "cell_size": [23.0, 3.5, 3.22],
                "wall_seconds": 321.5,
                "meep_time": 180.0,
                "timesteps": 7200,
            }
        },
    )
    size = validation.simulation_size(result, resolution=20, num_freqs=101)
    assert size.cell_x_um == pytest.approx(23.0)
    assert size.cell_y_um == pytest.approx(3.5)
    assert size.cell_z_um == pytest.approx(3.22)
    timing = validation.run_metadata(result)
    assert timing == {
        "wall_seconds": pytest.approx(321.5),
        "meep_time": pytest.approx(180.0),
        "timesteps": 7200,
    }


def test_run_metadata_defaults_when_absent():
    assert validation.run_metadata(_Result([1.55], {}, {})) == {
        "wall_seconds": 0.0,
        "meep_time": 0.0,
        "timesteps": 0,
    }


def test_format_summary_reports_time_and_size():
    outcome = validation.ValidationOutcome(
        case="straight",
        wavelengths=np.array([1.5, 1.55, 1.6]),
        s_params={"S21": np.ones(3)},
        reference=validation.Reference(np.array([1.5, 1.6]), {"S21": np.ones(2)}),
        metrics={"S21": {"rms_db": 0.25, "corr": 0.99}},
        best_port="S21",
        resolution=30,
        num_freqs=201,
        dimension="3d",
        wall_seconds=123.4,
        meep_time=90.0,
        timesteps=3600,
        size=validation.SimulationSize(23.0, 3.5, 3.22, 30, 201, "3d"),
    )
    table = validation.format_summary({"straight": outcome})
    assert "compute time (s)" in table
    assert "cell x/y/z (lambda)" in table
    assert "123.4" in table
    assert "14.8 x 2.3 x 2.1" in table


def test_format_summary_lists_failures():
    table = validation.format_summary({"mzi": RuntimeError("boom")})
    assert "FAILED" in table


def test_high_resolution_overrides_are_accuracy_first():
    overrides = validation.high_resolution_overrides()
    assert set(overrides) == {
        "straight",
        "coupler",
        "ring_coupling",
        "mzi",
        "ring",
        "bragg",
        "cspdk_coupler",
        "cspdk_ring_coupling",
        "siepic_coupler",
    }
    for settings in overrides.values():
        assert settings["resolution"] > validation.VALIDATION_RESOLUTION
        assert settings["num_freqs"] >= validation.VALIDATION_NUM_FREQS
        assert settings["decay_by"] <= validation.VALIDATION_DECAY_BY
    # The large MZI is pushed hardest on the cheap 2-D plane.
    assert overrides["mzi"]["resolution"] >= 40
    # The DBR cannot use subpixel averaging (10 nm end-caps hang init_sim).
    assert overrides["bragg"]["subpixel"] is False


def test_high_resolution_presets_resolve_critical_features():
    overrides = validation.high_resolution_overrides()
    for name, settings in overrides.items():
        report = validation.case_feature_report(name, resolution=settings["resolution"])
        assert report["resolved"] is True, name


def test_ring_gap_preset_needs_four_cells():
    report = validation.case_feature_report("ring", resolution=25)
    assert report["resolved"] is False
    assert report["suggested_resolution"] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Simulation configuration (no run)
# ---------------------------------------------------------------------------


def test_build_validation_simulation_defaults():
    pytest.importorskip("gsim")
    component = _Component("o1", "o2")
    sim = validation.build_validation_simulation(component, stack=object())
    assert sim.solver.resolution == validation.VALIDATION_RESOLUTION == 20
    assert sim.num_freqs == validation.VALIDATION_NUM_FREQS == 101
    assert sim.solver.stopping == "energy_decay"
    assert sim.solver.stopping_threshold == pytest.approx(
        validation.VALIDATION_DECAY_BY
    )
    assert sim.solver.stopping_dt == pytest.approx(validation.VALIDATION_DECAY_DT)
    assert sim.source.wavelength_span == pytest.approx(validation.VALIDATION_SPAN_UM)
    assert sim.monitors == ["o1", "o2"]
    assert sim.source.port == "o1"


def test_2d_effective_index_overrides_only_the_core():
    pytest.importorskip("ubcpdk")
    from tests.pdk_validation._pdks import ubc_target

    target = ubc_target()
    case = validation.get_case("straight")
    component = case.make_component()
    sim = validation.build_validation_simulation(
        component, dimension="2d", pdk=target.pdk
    )

    core = sim.geometry.stack.layers["core"]
    substrate = sim.geometry.stack.layers["substrate"]
    assert core.material.endswith("__core_2d")
    assert substrate.material == "si"
    assert sim.geometry.stack.layers["box"].material == "sio2"
    assert core.material in sim.materials
    assert "si" not in sim.materials
    # The thick silicon substrate also supports a mode. A full-stack solve
    # can select it and silently replace the core slab index with bulk Si.
    core_index = sim.materials[core.material].refractive_index
    assert core_index == pytest.approx(2.847475, abs=1e-4)
    finer = validation.build_validation_simulation(
        component, dimension="2d", resolution=50, pdk=target.pdk
    )
    assert finer.materials[core.material].refractive_index == pytest.approx(core_index)


def test_build_validation_simulation_2d_plane():
    pytest.importorskip("gsim")
    component = _Component("o1", "o2")
    sim = validation.build_validation_simulation(
        component, dimension="2d", stack=object()
    )
    assert sim.solver.mode == "2d"
    assert sim.solver.z_cut == "auto"


# ---------------------------------------------------------------------------
# Matched-geometry devices (need gdsfactory, no MEEP)
# ---------------------------------------------------------------------------


def test_bragg_case_geometry_matches_pinned_set1_design():
    pytest.importorskip("ubcpdk")
    component = validation.get_case("bragg").make_component()
    assert sorted(port.name for port in component.ports) == ["o1", "o2"]
    assert component.xsize == pytest.approx(200 * 0.318, abs=0.002)
    assert component.ysize == pytest.approx(0.5 + 0.12 / 2, abs=0.002)


def test_published_layouts_are_the_recorded_meep_geometries():
    from scripts import build_pdk_validation_variant as page

    for name in (
        "straight",
        "coupler",
        "ring_coupling",
        "ring",
        "bragg",
        "cspdk_coupler",
        "cspdk_ring_coupling",
        "siepic_coupler",
    ):
        recorded = page.RUNS / "runs" / f"2d_{name}" / "layout.gds"
        published = page.GDS_STORE / f"{name}.gds"
        assert published.read_bytes() == recorded.read_bytes(), name
        assert page.resolve_gds(name) == recorded


def test_layout_lookup_stays_within_repository_data(tmp_path, monkeypatch):
    from scripts import build_pdk_validation_variant as page

    monkeypatch.setattr(page, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(page, "GDS_STORE", tmp_path / "gds")
    assert page.resolve_gds("missing") is None


@pytest.mark.parametrize(
    "name", ["straight", "coupler", "ring_coupling", "mzi", "ring", "bragg"]
)
def test_case_registry(name):
    pytest.importorskip("ubcpdk")
    case = validation.get_case(name)
    assert case.name == name
    assert case.resolution >= 20
    assert case.num_freqs >= 101


def test_mzi_case_uses_2d_for_size():
    pytest.importorskip("ubcpdk")
    assert validation.get_case("mzi").dimension == "2d"


@pytest.mark.pdk_data
def test_ring_through_uses_measured_detector_one():
    pytest.importorskip("ubcpdk")
    from tests.pdk_validation import _fetch

    case = validation.get_case("ring")
    reference = case.reference(np.array([1.55]))
    path = _fetch.fetch_ubc(_fetch.UBC_RING_MAT)
    wavelength_nm, detector_one = _fetch.load_ubc_spectrum(path, port=1)
    _, detector_zero = _fetch.load_ubc_spectrum(path, port=0)
    assert reference.sim_ports == ("S21",)
    assert reference.transmission_db is not None
    assert np.array_equal(reference.transmission_db[0], wavelength_nm / 1000)
    assert np.array_equal(reference.transmission_db[1], detector_one)
    assert not np.array_equal(detector_zero, detector_one)


def test_available_cases_use_their_target_pdk():
    from tests.pdk_validation._pdks import available_targets

    targets = available_targets()
    if "ubc" not in targets:
        pytest.skip("ubcpdk unavailable")

    cases = {case.name: case for case in validation.build_cases()}
    assert cases["straight"].target.key == "ubc"
    if "cspdk" in targets:
        assert cases["cspdk_coupler"].target.key == "cspdk"
        assert cases["cspdk_ring_coupling"].target.key == "cspdk"
        assert cases["cspdk_coupler"].target.pdk.name.startswith("cspdk.si220")
    assert cases["siepic_coupler"].target.key == "generic"
    assert cases["siepic_coupler"].target.pdk.name == "generic"


def test_siepic_case_uses_generic_pdk_geometry():
    from tests.pdk_validation._pdks import available_targets

    targets = available_targets()
    if "ubc" not in targets:
        pytest.skip("ubcpdk unavailable")
    case = validation.get_case("siepic_coupler")
    target = targets["generic"]
    case.target.activate()
    component = case.make_component()

    assert component.layers
    assert all(tuple(layer) == (1, 0) for layer in component.layers)
    assert target.dielectrics
    assert {region["name"] for region in target.dielectrics} == {
        "substrate",
        "box",
        "clad",
    }
