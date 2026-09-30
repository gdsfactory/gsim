"""The RF Stage on the real pipeline: its own Staircase mesh, real solves.

Gated on gmsh and the femwell runtime, and standing on a synthetic
Carrier map so it needs no DEVSIM: what it proves is the Stage's own
chain — pick the Bias point, staircase its Carrier map, mesh the
electrode-loaded Cross-section, select the line Mode at every frequency
and extract the impedance over the signal conductor. The end-to-end run
off a real charge solve is the ``tcad_local`` test at the bottom.

The fixtures solve the line Mode, not the wall Mode (ADR 0005, ticket 24):
the Bias is the depleted 4 V point, where the loaded line Mode sits inside
the default loss bound, and the band starts at 20 GHz: at 10 GHz the lossy
2 um electrodes of the ``"volume"`` model make even the bare line an RC
slow wave (loss ratio 0.55), which the band was chosen to stay clear of.
The ``tcad_local`` end-to-end runs stand on a real charge solve: one at
the three Strips of these fixtures, which only reaches the line
parameters, and one at the preset's defaults, which lands on the line
Mode across the preset's band (``TestCrosscheck`` compares the two
loaded-line routes at the Strip count that resolves the junction).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gsim.common.modes import Conductor, NoLineModeError
from gsim.common.transmission_line import RFLineParams
from gsim.modulator import Device, Study, pn_phase_shifter

from .conftest import (
    CENTER_Y,
    HALF_WIDTH,
    PAD_WIDTH,
    RIB_HEIGHT,
    WALL_MODE_WARNING,
    build_demo,
)

pytest.importorskip("gmsh")
pytest.importorskip("femwell")
pytest.importorskip("skfem")

from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

DOPING_CM3 = 1e18
DEPLETED_CM3 = 1e10
SLAB = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)
FREQS_HZ = [20e9, 40e9]
#: The depleted Bias the fixtures solve at; the line Mode is inside the bound.
DEPLETED_V = 4.0
#: Strips across the Junction: three put a depleted Strip in the middle,
#: where two average it away and land the solve on the wall Mode.
N_STRIPS = 3
#: One-frequency, one-Bias sweep for the tests that build their own Study.
ONE_FREQ_HZ = [20e9]
ONE_BIAS = [DEPLETED_V]


def depletion_carriers(bias_v: float) -> CarrierMap:
    """A Carrier map whose depletion region widens with reverse bias."""
    y = np.linspace(SLAB[0], SLAB[1], 121)
    z = np.linspace(0.0, RIB_HEIGHT, 9)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()

    depleted = np.abs(yy - CENTER_Y) < 0.05 * np.sqrt(1.0 + abs(bias_v))
    n_side = yy < CENTER_Y
    electrons = np.where(n_side & ~depleted, DOPING_CM3, DEPLETED_CM3)
    holes = np.where(~n_side & ~depleted, DOPING_CM3, DEPLETED_CM3)

    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in n_side],
        electrons_cm3=electrons,
        holes_cm3=holes,
    )


#: Frequency the synthetic small-signal admittances are fitted at (Hz).
JUNCTION_FREQ_HZ = 1e9


def junction_admittance_per_cm(bias_v: float) -> complex:
    """A series-RC admittance whose capacitance falls with reverse bias."""
    r_s_ohm_m = 1.1e-4
    c_j_f_per_m = 3.0e-10 / np.sqrt(1.0 + abs(bias_v))
    omega = 2.0 * np.pi * JUNCTION_FREQ_HZ
    return 1.0 / (r_s_ohm_m - 1j / (omega * c_j_f_per_m)) / 1e2


def canned_sweep(biases) -> BiasSweepResult:
    """A Bias sweep of synthetic Carrier maps and series-RC admittances."""
    return BiasSweepResult(
        contact="cathode",
        points=[
            BiasPoint(
                bias_v=bias,
                carriers=depletion_carriers(bias),
                admittance_s_per_cm=junction_admittance_per_cm(bias),
                admittance_freq_hz=JUNCTION_FREQ_HZ,
            )
            for bias in biases
        ],
    )


def build_study(output_dir, biases=(0.0, DEPLETED_V)):
    """A Study whose charge Stage already holds a synthetic Bias sweep."""
    demo = build_demo()
    component, stack = demo.component, demo.stack
    study = Study(
        component=component,
        stack=stack,
        device=Device(p_regions=["p_rib", "p_pad"], n_regions=["n_rib", "n_pad"]),
        output_dir=output_dir,
    )
    study.charge.seed(canned_sweep(list(biases)))
    return study


def run_recording(solve):
    """Run one solve with every warning it emits recorded alongside."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = solve()
    return result, [str(w.message) for w in record]


def wall_mode_warnings(messages):
    """The recorded warnings saying the solve selected the wall Mode."""
    return [m for m in messages if WALL_MODE_WARNING in m]


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """The RF Stage run across two frequencies, with what it warned."""
    study = build_study(tmp_path_factory.mktemp("modulator-rf"))
    study.rf(frequencies_hz=FREQS_HZ, n_strips=N_STRIPS)
    line, messages = run_recording(study.rf.run)
    return study, line, messages


class TestLineParameters:
    def test_the_result_is_the_line_parameter_type(self, solved):
        _, line, _ = solved

        assert isinstance(line, RFLineParams)
        assert line.freq_hz == pytest.approx(FREQS_HZ)

    def test_the_loaded_line_is_slow_and_lossy(self, solved):
        _, line, _ = solved

        # The carrier-loaded junction slows the wave well past the light
        # line, and the conductive strips make it lossy.
        assert np.all(line.n_rf > 1.0)
        assert np.all(line.alpha_rf_np_m > 0.0)

    def test_the_impedance_is_physical(self, solved):
        _, line, _ = solved

        assert np.all(line.z0_ohm.real > 0.0)
        assert np.all(line.z0_ohm.real < 1e3)

    def test_the_line_parameters_reach_the_rlgc_relations(self, solved):
        _, line, _ = solved

        rlgc = line.rlgc
        assert np.all(np.isfinite(rlgc["C"]))
        assert np.all(rlgc["C"] > 0.0)

    def test_the_bias_it_was_solved_at_is_reported(self, solved):
        _, line, _ = solved

        assert line.bias_v == DEPLETED_V

    def test_it_is_the_line_mode_not_the_wall_mode(self, solved):
        """Ticket 24: the fixture exercises the Mode the Stage exists for."""
        _, line, messages = solved

        assert not wall_mode_warnings(messages)
        # Measured 3.74 / 3.52 and 46.5 / 43.8 ohm at 20 / 40 GHz: the
        # loaded line Mode, not the wall Mode's ~2.2 and ~180 ohm.
        assert np.all(line.n_rf > 3.0)
        assert np.all(line.z0_ohm.real < 80.0)

    def test_the_carriers_stage_ran_first(self, solved):
        study, _, _ = solved
        assert study.carriers.has_run is True

    def test_running_twice_solves_once(self, solved):
        study, line, _ = solved
        assert study.rf.run() is line


class TestItsOwnStaircase:
    def test_the_rf_mesh_carries_the_strips_and_the_electrodes(self, solved):
        import meshio

        study, _, _ = solved
        mesh = meshio.read(str(study.stage_dir("rf") / "palace.msh"))
        regions = {
            str(name)
            for name, data in mesh.field_data.items()
            if int(np.asarray(data)[1]) == 2
        }

        assert {"strip_0", "strip_1", "strip_2"} <= regions
        assert {"electrode_low", "electrode_high"} <= regions

    def test_the_rf_mesh_is_not_the_optical_one(self, solved):
        """ADR 0002: the RF solve meshes its own Window."""
        study, _, _ = solved

        assert (study.stage_dir("rf") / "palace.msh").exists()
        assert not (study.stage_dir("optical") / "palace.msh").exists()

    def test_the_window_spans_the_electrodes_and_their_surroundings(self, solved):
        import meshio

        study, _, _ = solved
        points = np.asarray(
            meshio.read(str(study.stage_dir("rf") / "palace.msh")).points
        )
        low, high = study.rf.staircase().electrode_spans

        assert points[:, 0].min() < low[0]
        assert points[:, 0].max() > high[1]


class TestSignalConductor:
    def test_the_impedance_integrates_over_the_signal_electrode(
        self, tmp_path, monkeypatch
    ):
        """The conductor comes from the device description, not the caller."""
        from gsim.femwell import adapter

        captured: dict[str, Conductor] = {}
        extract = adapter.z0_power_current

        def spy(mode, *, frequency_hz, conductor, mesh):
            captured["conductor"] = conductor
            return extract(
                mode, frequency_hz=frequency_hz, conductor=conductor, mesh=mesh
            )

        monkeypatch.setattr(adapter, "z0_power_current", spy)
        study = build_study(tmp_path, biases=ONE_BIAS)
        study.rf(frequencies_hz=ONE_FREQ_HZ, n_strips=N_STRIPS)
        study.rf.run()

        conductor = captured["conductor"]
        assert conductor.name == "electrode_low"
        assert conductor.model == "volume"
        assert conductor.extent == study.rf.staircase().electrode_extent(
            "electrode_low"
        )


@pytest.fixture(scope="module")
def pec_solved(tmp_path_factory):
    """The same RF Stage run with perfect electrodes instead of lossy ones."""
    study = build_study(tmp_path_factory.mktemp("modulator-rf-pec"))
    study.rf(
        frequencies_hz=FREQS_HZ,
        n_strips=N_STRIPS,
        conductor_model="pec",
        # The contour integral reads h on the domain boundary, where
        # femwell's first-order h is piecewise constant.
        order=2,
    )
    line, messages = run_recording(study.rf.run)
    return study, line, messages


class TestPerfectConductorElectrodes:
    """The ``"pec"`` conductor model on the femwell Route (ADR 0003).

    A perfect electrode is left out of the meshed domain, so the Stage
    has no conduction current to integrate and reads Ampere's contour
    integral around the hole instead. What that must not do is change
    the answer: the same line, modelled with lossless metal instead of
    lossy metal, has to come out with much the same impedance and much
    less loss.
    """

    def test_the_electrodes_are_not_regions_of_the_mesh(self, pec_solved):
        import meshio

        study, _, _ = pec_solved
        mesh = meshio.read(str(study.stage_dir("rf") / "palace.msh"))
        regions = {
            str(name)
            for name, data in mesh.field_data.items()
            if int(np.asarray(data)[1]) == 2
        }

        assert {"strip_0", "strip_1", "strip_2"} <= regions
        assert not {"electrode_low", "electrode_high"} & regions

    def test_the_contour_current_reaches_a_physical_impedance(self, pec_solved):
        _, line, _ = pec_solved

        assert np.all(np.isfinite(line.z0_ohm))
        assert np.all(line.z0_ohm.real > 10.0)
        assert np.all(line.z0_ohm.real < 1e3)

    def test_it_is_the_line_mode_not_the_wall_mode(self, pec_solved):
        _, line, messages = pec_solved

        assert not wall_mode_warnings(messages)
        # Measured n_eff = 2.9015 and 41.0 ohm at both frequencies, the
        # line Mode ticket 23 pinned the two Routes to.
        assert np.all(line.n_rf > 2.5)
        assert np.all(line.z0_ohm.real < 80.0)

    def test_it_lands_where_the_lossy_metal_model_does(self, pec_solved, solved):
        """Two models of the same electrode, one line: same impedance.

        The lossy 2 um metal carries an internal impedance the perfect
        conductor does not, and it matters most where the skin depth is
        deepest: measured 46.5 ohm against 41.0 at 20 GHz (13%) and
        43.8 against 41.0 at 40 GHz (7%).
        """
        _, pec, _ = pec_solved
        _, volume, _ = solved

        np.testing.assert_allclose(pec.z0_ohm.real, volume.z0_ohm.real, rtol=0.2)

    def test_the_line_is_far_less_lossy_without_the_metal(self, pec_solved, solved):
        """The metal's own loss is what the pec model drops."""
        _, pec, _ = pec_solved
        _, volume, _ = solved

        assert np.all(pec.alpha_rf_np_m > 0.0)
        # What is left is the slab's own loss — the junction charging through
        # its series resistance, which grows as f^2 — so it is a few percent
        # of the metal's at 40 GHz rather than a fixed fraction of it.
        assert np.all(pec.alpha_rf_np_m < 0.05 * volume.alpha_rf_np_m)

    def test_a_first_order_solve_says_the_impedance_is_biased(self, tmp_path):
        study = build_study(tmp_path, biases=ONE_BIAS)
        study.rf(
            frequencies_hz=ONE_FREQ_HZ,
            n_strips=N_STRIPS,
            conductor_model="pec",
            order=1,
        )
        with pytest.warns(UserWarning, match="biased high by tens of percent"):
            study.rf.run()


class TestWindowTooSmall:
    def test_the_default_window_does_not_warn(self, tmp_path):
        """The full extent shields the line rather than squeezing it."""
        study = build_study(tmp_path, biases=ONE_BIAS)
        study.rf(frequencies_hz=ONE_FREQ_HZ, n_strips=N_STRIPS)

        _, messages = run_recording(study.rf.run)

        assert not [m for m in messages if "window boundary" in m]

    def test_a_squeezed_mode_warns_naming_the_stage(self, tmp_path):
        study = build_study(tmp_path, biases=ONE_BIAS)
        # A window clearing the electrodes by 0.2 um presses the line
        # Mode against the wall (ADR 0002): measured 36% of its peak
        # field there. The squeeze is in-plane on purpose: closing the
        # wall in z instead pushes the line Mode's loss past the default
        # bound first, and the solve falls back on the wall Mode (ADR
        # 0005), which is not the Mode this test is about.
        study.rf(
            frequencies_hz=ONE_FREQ_HZ,
            n_strips=N_STRIPS,
            window=(CENTER_Y - 2.5, CENTER_Y + 2.5),
            window_z=(-2.0, 2.0),
        )

        line, messages = run_recording(study.rf.run)

        assert [
            m for m in messages if "rf stage's mode" in m and "window boundary" in m
        ]
        assert not wall_mode_warnings(messages)
        # Still the line Mode: measured 3.72 and 35.4 ohm.
        assert line.n_rf[0] > 3.0
        assert line.z0_ohm[0].real < 80.0


class TestModeSelection:
    def test_an_override_rule_selects_the_mode(self, tmp_path):
        study = build_study(tmp_path, biases=ONE_BIAS)
        study.rf(
            frequencies_hz=ONE_FREQ_HZ,
            n_strips=N_STRIPS,
            num_modes=2,
            rule=lambda modes: [],
        )

        with pytest.raises(NoLineModeError):
            study.rf.run()

    def test_ambiguous_candidates_warn_through_the_stage(self, tmp_path):
        """The shared rule's degeneracy warning reaches the user."""
        study = build_study(tmp_path, biases=ONE_BIAS)
        study.rf(
            frequencies_hz=ONE_FREQ_HZ,
            n_strips=N_STRIPS,
            num_modes=2,
            rule=list,
            degeneracy_rtol=1e3,
        )

        with pytest.warns(UserWarning, match="degenerate"):
            study.rf.run()


@pytest.mark.tcad_local
class TestEndToEnd:
    def test_a_real_charge_solve_reaches_the_line_parameters(self, tmp_path):
        pytest.importorskip("devsim")
        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(p_regions=["p_rib", "p_pad"], n_regions=["n_rib", "n_pad"]),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0, DEPLETED_V])
        study.rf(frequencies_hz=FREQS_HZ, n_strips=N_STRIPS)

        line = study.rf.run()

        assert study.charge.has_run is True
        assert line.freq_hz == pytest.approx(FREQS_HZ)
        assert np.all(line.n_rf > 1.0)
        assert line.bias_v == DEPLETED_V

    def test_the_presets_defaults_land_on_the_line_mode(self, tmp_path):
        """The notebook's path: nothing configured, the line Mode found.

        The preset's Strips tile the whole doped slab and one of them still
        sits inside the 2 V depletion region, so no Strip shunts the
        electrodes; the loaded line's loss ratio at 10 GHz is inside the
        default bound, so the wall Mode (n_eff ~ 2.2, Z0 ~ 170 ohm) is not
        what the rule falls back to at any frequency of the band.
        """
        pytest.importorskip("devsim")
        demo = build_demo()
        study = pn_phase_shifter(
            component=demo.component,
            stack=demo.stack,
            device=demo.device,
            output_dir=tmp_path,
        )

        line, messages = run_recording(study.rf.run)

        assert not wall_mode_warnings(messages)
        assert not [m for m in messages if "no depleted strip" in m]
        assert np.all(line.n_rf > 4.0)
        assert np.all((line.z0_ohm.real > 20.0) & (line.z0_ohm.real < 80.0))


class TestTwoPortExport:
    """The Study hands the Traveling-wave electrode over, no hand-carried arrays."""

    def test_the_touchstone_export_reads_back_as_the_solved_line(self, solved):
        skrf = pytest.importorskip("skrf")
        study, line, _ = solved

        path = study.line.export_touchstone()
        network = skrf.Network(str(path))

        assert path == study.stage_dir("line") / "electrode.s2p"
        np.testing.assert_allclose(network.f, line.freq_hz)
        # A passive line referenced to a real impedance transmits at most
        # what it is fed.
        assert np.all(np.isfinite(network.s))
        assert np.all(np.abs(network.s[:, 1, 0]) <= 1.0 + 1e-9)

    def test_the_sax_model_is_evaluated_off_the_study(self, solved):
        study, _, _ = solved

        sdict = study.line.sax_model()(f=np.linspace(FREQS_HZ[0], FREQS_HZ[-1], 7))

        s21 = sdict[("o2", "o1")]
        assert s21.shape == (7,)
        assert np.all(np.isfinite(s21))
        assert np.all(np.abs(s21) <= 1.0 + 1e-9)

    def test_the_exports_round_trip_on_the_real_solve(self, solved):
        """Ticket: the handoff reassembles to the Study's own answers."""
        study, _, _ = solved

        comparison = study.line.verify_exports(quiet=True)

        assert comparison.check() is comparison
        assert comparison.freq_hz == pytest.approx(FREQS_HZ)


@pytest.fixture(scope="module")
def unloaded(tmp_path_factory):
    """The unloaded solve on the same Study, carriers switched off."""
    study = build_study(tmp_path_factory.mktemp("modulator-rf-unloaded"))
    study.rf(frequencies_hz=FREQS_HZ, n_strips=N_STRIPS)
    line, messages = run_recording(study.rf.run_unloaded)
    return study, line, messages


class TestUnloadedSolve:
    def test_the_result_is_flagged_unloaded(self, unloaded):
        _, line, _ = unloaded

        assert isinstance(line, RFLineParams)
        assert line.unloaded is True
        assert line.freq_hz == pytest.approx(FREQS_HZ)

    def test_it_is_the_line_mode_not_the_wall_mode(self, unloaded):
        _, _, messages = unloaded

        assert not wall_mode_warnings(messages)

    def test_the_bare_line_sits_between_oxide_and_the_strip_dielectric(self, unloaded):
        study, line, _ = unloaded

        # No carriers: the mode's index sits between the light lines of
        # the materials it spreads over — above the oxide's sqrt(3.9),
        # below the strips' sqrt(11.7) — nowhere near the slow-wave
        # index a loaded junction produces. Measured 2.88 / 2.72 at
        # 20 / 40 GHz on the demo device.
        assert np.all(line.n_rf > 1.9)
        assert np.all(line.n_rf < np.sqrt(study.rf.strip_permittivity))

    def test_the_impedance_is_a_plausible_bare_lines(self, unloaded):
        _, line, _ = unloaded

        # The demo device measures 60.3 / 56.8 ohm at 20 / 40 GHz: two
        # narrow (2 um) conductors on a thin stack. The bracket rules
        # out a metal-shorted (~0) answer and the wall Mode's ~180 ohm.
        assert np.all(line.z0_ohm.real > 30.0)
        assert np.all(line.z0_ohm.real < 120.0)

    def test_the_junction_loads_the_bare_line(self, unloaded, solved):
        """Carriers on: slower and lower impedance than the bare electrode."""
        _, bare, _ = unloaded
        _, loaded, _ = solved

        assert np.all(loaded.n_rf > bare.n_rf)
        assert np.all(loaded.z0_ohm.real < bare.z0_ohm.real)

    def test_the_unloaded_solve_is_cached_and_invalidated(self, unloaded):
        study, line, _ = unloaded

        assert study.rf.run_unloaded() is line
        study.rf(n_strips=4)
        assert study.rf._unloaded_result is None

    def test_the_loaded_result_is_untouched(self, unloaded):
        study, _, _ = unloaded

        assert study.rf.has_run is False


@pytest.mark.tcad_local
class TestCrosscheck:
    """Ticket: the two loaded-line routes agree on the demo device.

    The comparison needs the direct solve to actually resolve the
    junction, which the defaults do not attempt: the Strips must tile
    the whole doped slab (so the pads' series resistance is in), be
    narrower than the depletion region (61 across 1.2 um), and the
    eigenvalue search must be aimed at the slow wave. 20-30 GHz is the
    band the gate's tolerances were set on: higher, the unloaded solve
    wanders onto a substrate branch.
    """

    def test_the_routes_agree_within_the_gates_tolerance(self, tmp_path):
        pytest.importorskip("devsim")
        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(p_regions=["p_rib", "p_pad"], n_regions=["n_rib", "n_pad"]),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0, 2.0])
        study.rf(
            frequencies_hz=[20e9, 30e9],
            n_strips=61,
            strip_span=SLAB,
            num_modes=8,
            n_guess=6.0,
        )

        comparison = study.rf.crosscheck()

        assert comparison.direct.unloaded is False
        assert comparison.assembled.unloaded is False
        assert comparison.freq_hz == pytest.approx([20e9, 30e9])
        # The junction loads the line: the direct solve is slower and
        # lossier than the bare electrode by far more than the routes'
        # residual disagreement.
        unloaded = study.rf.run_unloaded()
        assert np.all(comparison.direct.n_rf > 1.5 * unloaded.n_rf)
        # The gate: both routes' n_RF, loss and Z0 within the stated
        # tolerances, or check() names the diverging quantity.
        comparison.check()


class TestWallMode:
    """The femwell Route says when it selected the wall Mode (ADR 0005).

    At 0 V the undepleted 1e18 Strips join the two electrodes through a
    resistive slab, and the loaded line is an RC slow wave losing as fast
    as it advances (``n_eff ~ 31.9 - 31.9j``), at the edge of the default
    loss bound. The Staircase says so before anything is meshed; a
    tighter bound drops the RC wave and leaves the Mode running between
    both electrodes together and the metallic Window wall, which the
    Route reads off the electrode currents. Depleting the Junction puts
    the line Mode well inside the bound, where the default rule finds it.
    """

    def test_an_undepleted_bias_is_reported_before_the_solve(self, tmp_path):
        study = build_study(tmp_path)
        study.rf(
            frequencies_hz=[10e9],
            n_strips=N_STRIPS,
            conductor_model="pec",
            order=2,
            bias_v=0.0,
        )
        with pytest.warns(UserWarning, match="no depleted strip"):
            study.rf.staircase()

    def test_a_tighter_bound_lands_on_the_wall_mode_and_says_so(self, tmp_path):
        study = build_study(tmp_path)
        study.rf(
            frequencies_hz=[10e9],
            n_strips=N_STRIPS,
            conductor_model="pec",
            order=2,
            bias_v=0.0,
            # Aimed straight at the wall Mode's index, with the bound the
            # Stage defaulted to before the RC wave was admitted.
            n_guess=2.0,
            max_loss_ratio=0.5,
        )
        with pytest.warns(UserWarning, match=WALL_MODE_WARNING):
            study.rf.run()

    def test_a_depleted_bias_lands_on_the_line_mode(self, tmp_path):
        study = build_study(tmp_path)
        study.rf(
            frequencies_hz=[10e9],
            n_strips=N_STRIPS,
            conductor_model="pec",
            order=2,
            bias_v=DEPLETED_V,
        )
        line, messages = run_recording(study.rf.run)
        assert not wall_mode_warnings(messages)
        # The loaded line Mode, not the wall Mode's ~180 ohm.
        assert 20.0 < line.z0_ohm[0].real < 80.0
        assert line.n_rf[0] > 2.5
