"""The line Stage: the electrode, the terminations, and the device report.

Nothing here solves anything. Both EM Stages are stubbed with canned
results, so what is under test is the assembly the line Stage does — the
optical sweep and the RF line parameters turned into the existing
:class:`~gsim.modulator.report.TWMZMReport` — and the lifecycle around
it: the report runs whatever upstream Stage has not run, and re-configuring
the line throws the report away without touching either solve.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gsim.common.transmission_line import (
    line_params_from_neff,
    segmented_period_abcd,
)
from gsim.modulator.optical import (
    GroupIndex,
    OpticalMode,
    OpticalStage,
    OpticalSweep,
)
from gsim.modulator.report import OpticalPhaseSweep, twmzm_figures_of_merit
from gsim.modulator.rf import RFStage
from gsim.modulator.twmzm import mzm_transfer, mzm_transfer_figures

WAVELENGTH_UM = 1.55
N_GROUP = 3.8
#: What the optical Stage computes when the line Stage is given none.
COMPUTED_N_GROUP = 3.9
RF_FREQS = [10e9, 40e9, 100e9]
RF_N_EFF = [3.20 - 0.004j, 3.15 - 0.010j, 3.10 - 0.020j]
RF_Z0 = [46.0 + 1.0j, 44.0 + 0.5j, 42.0 + 0.2j]
BIASES = [0.0, 1.0, 2.0]
#: Phase index per bias: real part falls as the junction depletes.
OPTICAL_N_EFF = [2.400000 - 1e-5j, 2.399950 - 9e-6j, 2.399870 - 8e-6j]


def optical_sweep(
    biases: list[float] | None = None, n_eff: list[complex] | None = None
) -> OpticalSweep:
    """A canned optical sweep, as the optical Stage would return it."""
    biases = BIASES if biases is None else biases
    n_eff = OPTICAL_N_EFF if n_eff is None else n_eff
    reference = n_eff[0].real
    return OpticalSweep(
        contact="cathode",
        wavelength_um=WAVELENGTH_UM,
        reference_bias_v=biases[0],
        points=[
            OpticalMode(
                bias_v=bias,
                n_eff=value,
                index_shift=value.real - reference,
                loss_db_cm=abs(value.imag) * 1e3,
                boundary_field_ratio=1e-4,
            )
            for bias, value in zip(biases, n_eff, strict=True)
        ],
    )


#: Provenance the canned RF result carries, as the RF Stage records it.
RF_BIAS_V = 2.0
RF_SIGNAL_CONTACT = "cathode"


def rf_params():
    """Canned RF line parameters, as the RF Stage would return them."""
    return line_params_from_neff(
        np.asarray(RF_FREQS, dtype=np.float64),
        RF_N_EFF,
        z0_ohm=RF_Z0,
        bias_v=RF_BIAS_V,
        signal_contact=RF_SIGNAL_CONTACT,
    )


@pytest.fixture
def solved(study, monkeypatch):
    """A Study whose EM Stages answer from canned results, counting solves."""
    solves = {"optical": 0, "rf": 0}
    group_index_solves: list[float] = []

    def solve_optical(_stage):
        solves["optical"] += 1
        return optical_sweep()

    def solve_group_index(_stage, sweep):
        group_index_solves.append(sweep.wavelength_um)
        return GroupIndex(
            n_group=COMPUTED_N_GROUP,
            wavelength_um=sweep.wavelength_um,
            step_um=0.01,
            bias_v=sweep.reference_bias_v,
            wavelengths_um=(sweep.wavelength_um - 0.01, sweep.wavelength_um + 0.01),
            n_eff=(2.41, 2.39),
            core_index=(3.48, 3.47),
        )

    def solve_rf(_stage):
        solves["rf"] += 1
        return rf_params()

    monkeypatch.setattr(OpticalStage, "_solve", solve_optical)
    monkeypatch.setattr(OpticalStage, "_solve_group_index", solve_group_index)
    monkeypatch.setattr(RFStage, "_solve", solve_rf)
    study.solves = solves
    study.group_index_solves = group_index_solves
    return study


def hand_assembled(
    *,
    length_m: float,
    z_load_ohm: complex = 50.0,
    z_gen_ohm: complex = 50.0,
    n_group: float = N_GROUP,
):
    """The report the notebook assembles by hand from the same inputs."""
    sweep = optical_sweep()
    return twmzm_figures_of_merit(
        rf_params(),
        OpticalPhaseSweep(
            voltages_v=sweep.voltages,
            dn_eff=sweep.index_shift,
            alpha_opt_db_cm=sweep.loss_db_cm,
            wavelength_um=WAVELENGTH_UM,
            n_group=n_group,
        ),
        length_m=length_m,
        z_load_ohm=z_load_ohm,
        z_gen_ohm=z_gen_ohm,
    )


class TestConfiguration:
    def test_defaults_are_readable(self, study):
        assert study.line.length_um == 3000.0
        assert study.line.z_load_ohm == 50.0
        assert study.line.z_gen_ohm == 50.0
        assert study.line.n_group is None
        assert study.line.response_frequencies_hz is None
        assert study.line.has_run is False

    def test_the_section_is_callable(self, study):
        assert study.line(length_um=5000.0, z_load_ohm=75.0) is study.line
        assert study.line.length_um == 5000.0
        assert study.line.z_load_ohm == 75.0

    def test_unknown_setting_is_rejected(self, study):
        with pytest.raises(ValueError, match="nope"):
            study.line(nope=1)

    def test_a_nonpositive_length_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.line(length_um=0.0)

    def test_a_nonpositive_group_index_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.line(n_group=0.0)

    def test_a_response_grid_that_is_empty_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.line(response_frequencies_hz=[])

    def test_a_nonpositive_response_frequency_is_rejected(self, study):
        with pytest.raises(ValueError, match="positive"):
            study.line(response_frequencies_hz=[0.0])

    def test_the_response_grid_is_kept_ascending(self, study):
        study.line(response_frequencies_hz=[40e9, 10e9])
        assert study.line.response_frequencies_hz == [10e9, 40e9]


class TestReport:
    def test_the_report_matches_the_hand_assembled_call(self, solved):
        solved.line(length_um=3000.0, n_group=N_GROUP)

        report = solved.line.run()
        expected = hand_assembled(length_m=3e-3)

        assert report.length_m == pytest.approx(3e-3)
        assert report.freq_hz == pytest.approx(expected.freq_hz)
        assert report.response == pytest.approx(expected.response)
        assert report.bandwidth_3db_hz == pytest.approx(expected.bandwidth_3db_hz)
        assert report.walkoff_bandwidth_hz == pytest.approx(
            expected.walkoff_bandwidth_hz
        )
        assert report.velocity_mismatch == pytest.approx(expected.velocity_mismatch)
        assert report.vpi_l_vcm == pytest.approx(expected.vpi_l_vcm)
        assert report.voltages_v == pytest.approx(expected.voltages_v)
        for name, values in expected.rlgc.items():
            assert report.rlgc[name] == pytest.approx(values)

    def test_the_terminations_reach_the_response(self, solved):
        solved.line(length_um=3000.0, n_group=N_GROUP, z_load_ohm=75.0, z_gen_ohm=25.0)

        report = solved.line.run()
        expected = hand_assembled(length_m=3e-3, z_load_ohm=75.0, z_gen_ohm=25.0)

        assert report.z_load_ohm == 75.0
        assert report.z_gen_ohm == 25.0
        assert report.response == pytest.approx(expected.response)

    def test_the_length_reaches_the_figures_of_merit(self, solved):
        solved.line(length_um=1000.0, n_group=N_GROUP)
        short = solved.line.run()
        solved.line(length_um=6000.0)
        long = solved.line.run()

        # Walk-off scales as 1/L at a fixed Velocity mismatch. This line is
        # dispersive, so the two limits read the mismatch at different
        # frequencies and the factor is not exactly six; each is what the
        # hand assembly gives at its own length.
        assert short.walkoff_bandwidth_hz > long.walkoff_bandwidth_hz
        assert short.walkoff_bandwidth_hz == pytest.approx(
            hand_assembled(length_m=1e-3).walkoff_bandwidth_hz
        )
        assert long.walkoff_bandwidth_hz == pytest.approx(
            hand_assembled(length_m=6e-3).walkoff_bandwidth_hz
        )

    def test_the_study_exposes_the_report(self, solved):
        solved.line(n_group=N_GROUP)
        assert solved.report() is solved.line.run()

    def test_an_optical_sweep_of_one_bias_point_is_an_actionable_error(
        self, study, monkeypatch
    ):
        monkeypatch.setattr(
            OpticalStage,
            "_solve",
            lambda _stage: optical_sweep(biases=[0.0], n_eff=[OPTICAL_N_EFF[0]]),
        )
        monkeypatch.setattr(RFStage, "_solve", lambda _stage: rf_params())
        study.line(n_group=N_GROUP)

        with pytest.raises(ValueError, match="charge"):
            study.line.run()


class TestMachZehnder:
    """The Phase shifter in the arms of a Mach-Zehnder: transfer and chirp."""

    #: Long enough for the canned sweep's 1.3e-4 index shift to reach V_pi.
    LENGTH_UM = 30000.0

    def test_push_pull_at_quadrature_is_the_default(self, study):
        assert study.line.drive == "push-pull"
        assert study.line.arm_bias_v is None
        assert study.line.arm_imbalance_db == 0.0
        assert study.line.phase_offset_rad == pytest.approx(np.pi / 2.0)

    def test_an_unknown_drive_configuration_is_rejected(self, study):
        with pytest.raises(ValueError, match="drive"):
            study.line(drive="dual-drive")

    def test_the_report_carries_the_transfer_and_its_figures(self, solved):
        solved.line(length_um=self.LENGTH_UM, n_group=N_GROUP)
        report = solved.line.run()

        sweep = optical_sweep()
        settings = {"length_m": self.LENGTH_UM * 1e-6, "wavelength_um": WAVELENGTH_UM}
        figures = mzm_transfer_figures(
            sweep.voltages, sweep.index_shift, sweep.loss_db_cm, **settings
        )
        assert report.drive == "push-pull"
        assert report.arm_bias_v == pytest.approx(1.0)
        assert report.drive_v[[0, -1]] == pytest.approx([-2.0, 2.0])
        assert report.transfer == pytest.approx(
            mzm_transfer(
                report.drive_v,
                sweep.voltages,
                sweep.index_shift,
                sweep.loss_db_cm,
                **settings,
            )
        )
        assert report.v_pi_v == pytest.approx(figures.v_pi_v)
        assert report.insertion_loss_db == pytest.approx(figures.insertion_loss_db)
        assert report.extinction_ratio_db == pytest.approx(figures.extinction_ratio_db)
        assert report.transfer_message is None
        # 0.01 dB/cm over 3 cm: next to nothing lost, and a loss that barely
        # moves with bias leaves the arms all but balanced at the null.
        assert 0.35 < report.v_pi_v < 0.45
        assert report.insertion_loss_db == pytest.approx(0.027, abs=1e-3)
        assert report.extinction_ratio_db > 60.0

    def test_both_drive_configurations_come_from_one_solve(self, solved):
        solved.line(length_um=self.LENGTH_UM, n_group=N_GROUP, drive="push-pull")
        push_pull = solved.line.run()
        solved.line(drive="single-drive")
        single = solved.line.run()

        assert solved.solves == {"optical": 1, "rf": 1}
        assert single.drive == "single-drive"
        # The voltage between the arms reaches half as far on one arm alone.
        assert single.drive_v[[0, -1]] == pytest.approx([-1.0, 1.0])
        # Chirp per Bias point: none push-pull, unit at quadrature driven
        # from one side, each up to the canned sweep's slight loss slope.
        assert push_pull.chirp.shape == push_pull.voltages_v.shape
        assert push_pull.chirp == pytest.approx(0.0, abs=1e-3)
        assert single.chirp == pytest.approx(1.0, abs=1e-3)

    def test_the_other_quadrature_point_flips_the_single_drive_chirp(self, solved):
        solved.line(n_group=N_GROUP, drive="single-drive", phase_offset_rad=-np.pi / 2)
        assert solved.line.run().chirp == pytest.approx(-1.0, abs=1e-3)

    def test_the_arm_imbalance_limits_the_extinction_ratio(self, solved):
        solved.line(length_um=self.LENGTH_UM, n_group=N_GROUP, arm_imbalance_db=0.5)
        report = solved.line.run()
        # 0.5 dB between the arms alone allows 30.8 dB.
        assert report.extinction_ratio_db == pytest.approx(30.8, abs=0.1)

    def test_the_arm_bias_is_where_a_single_drive_starts_from(self, solved):
        solved.line(
            length_um=self.LENGTH_UM,
            n_group=N_GROUP,
            drive="single-drive",
            arm_bias_v=0.0,
        )
        report = solved.line.run()
        assert report.arm_bias_v == 0.0
        assert report.drive_v[[0, -1]] == pytest.approx([0.0, 2.0])

    def test_an_arm_bias_outside_the_sweep_is_an_error(self, solved):
        solved.line(n_group=N_GROUP, arm_bias_v=5.0)
        with pytest.raises(ValueError, match="outside the bias sweep"):
            solved.line.run()

    def test_a_sweep_too_short_to_reach_v_pi_says_so(self, solved):
        solved.line(length_um=1000.0, n_group=N_GROUP)
        report = solved.line.run()

        assert report.v_pi_v is None
        assert report.insertion_loss_db is None
        assert report.extinction_ratio_db is None
        assert "too short to reach V_pi" in report.transfer_message
        # What the sweep does cover is still there to plot.
        assert np.all(np.isfinite(report.transfer))
        assert np.all(np.isfinite(report.chirp))


class TestBiasOrder:
    def test_an_unordered_bias_sweep_is_differentiated_in_order(
        self, study, monkeypatch
    ):
        """V_pi L is a slope, so the sweep is sorted before differentiating."""
        shuffled = [0.0, 2.0, 1.0]
        monkeypatch.setattr(
            OpticalStage,
            "_solve",
            lambda _stage: optical_sweep(
                biases=shuffled,
                n_eff=[OPTICAL_N_EFF[0], OPTICAL_N_EFF[2], OPTICAL_N_EFF[1]],
            ),
        )
        monkeypatch.setattr(RFStage, "_solve", lambda _stage: rf_params())
        study.line(length_um=3000.0, n_group=N_GROUP)

        report = study.line.run()
        expected = hand_assembled(length_m=3e-3)

        assert report.voltages_v == pytest.approx(expected.voltages_v)
        assert report.vpi_l_vcm == pytest.approx(expected.vpi_l_vcm)

    def test_a_bias_visited_twice_is_an_actionable_error(self, study, monkeypatch):
        monkeypatch.setattr(
            OpticalStage,
            "_solve",
            lambda _stage: optical_sweep(biases=[0.0, 0.0], n_eff=OPTICAL_N_EFF[:2]),
        )
        monkeypatch.setattr(RFStage, "_solve", lambda _stage: rf_params())
        study.line(n_group=N_GROUP)

        with pytest.raises(ValueError, match=r"twice|repeat"):
            study.line.run()


class TestUnmeasuredImpedance:
    def test_a_nan_impedance_is_named_rather_than_reported_as_a_result(
        self, study, monkeypatch
    ):
        """The palace RF route leaves Z0 NaN; the report says so."""
        nan_z0 = [complex(float("nan"), float("nan"))] * len(RF_FREQS)
        monkeypatch.setattr(OpticalStage, "_solve", lambda _stage: optical_sweep())
        monkeypatch.setattr(
            RFStage,
            "_solve",
            lambda _stage: line_params_from_neff(
                np.asarray(RF_FREQS, dtype=np.float64), RF_N_EFF, z0_ohm=nan_z0
            ),
        )
        study.line(n_group=N_GROUP)

        with pytest.warns(UserWarning, match="impedance"):
            report = study.line.run()

        # The figures that do not need Z0 stay usable.
        assert report.walkoff_bandwidth_hz > 0.0
        assert np.all(np.isfinite(report.vpi_l_vcm))


class TestGroupIndex:
    def test_the_configured_group_index_sets_the_velocity_mismatch(self, solved):
        solved.line(n_group=2.5)

        report = solved.line.run()

        assert report.velocity_mismatch == pytest.approx(
            np.asarray(RF_N_EFF, dtype=complex).real - 2.5
        )

    def test_a_configured_group_index_pays_for_no_extra_optical_solve(self, solved):
        solved.line(n_group=N_GROUP)

        solved.line.run()

        assert solved.group_index_solves == []
        assert solved.optical.result.group_index is None

    def test_without_one_the_optical_stage_computes_it(self, solved):
        report = solved.line.run()

        assert len(solved.group_index_solves) == 1
        assert report.velocity_mismatch == pytest.approx(
            np.asarray(RF_N_EFF, dtype=complex).real - COMPUTED_N_GROUP
        )

    def test_the_computed_one_is_not_the_phase_index_and_warns_about_nothing(
        self, solved
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = solved.line.run()

        assert report.velocity_mismatch != pytest.approx(
            np.asarray(RF_N_EFF, dtype=complex).real - OPTICAL_N_EFF[0].real
        )

    def test_re_configuring_the_line_keeps_the_computed_one(self, solved):
        """The two extra solves are the optical Stage's, not the line's."""
        solved.line.run()

        solved.line(length_um=1000.0)
        solved.line.run()

        assert len(solved.group_index_solves) == 1

    def test_a_configured_group_index_warns_about_nothing(self, solved):
        solved.line(n_group=N_GROUP)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solved.line.run()


class TestResponseGrid:
    def test_the_solved_frequencies_are_the_default_grid(self, solved):
        solved.line(n_group=N_GROUP)
        assert solved.line.run().freq_hz == pytest.approx(RF_FREQS)

    def test_a_dense_grid_interpolates_the_line_parameters(self, solved):
        grid = np.linspace(10e9, 100e9, 91)
        solved.line(n_group=N_GROUP, response_frequencies_hz=list(grid))

        report = solved.line.run()

        assert report.freq_hz == pytest.approx(grid)
        n_rf = np.asarray(RF_N_EFF, dtype=complex).real
        assert report.velocity_mismatch + N_GROUP == pytest.approx(
            np.interp(grid, RF_FREQS, n_rf)
        )
        assert report.z0_ohm[0] == pytest.approx(RF_Z0[0])

    def test_a_grid_past_the_solved_range_warns_about_the_clamp(self, solved):
        solved.line(n_group=N_GROUP, response_frequencies_hz=[10e9, 200e9])

        with pytest.warns(UserWarning, match="200|solved"):
            report = solved.line.run()

        # numpy clamps rather than extrapolating: the last solved value holds.
        assert report.velocity_mismatch[-1] + N_GROUP == pytest.approx(
            RF_N_EFF[-1].real
        )


#: The bare electrode: faster, less lossy, and well above 50 ohm.
UNLOADED_N_EFF = [2.30 - 0.001j, 2.28 - 0.002j, 2.26 - 0.004j]
UNLOADED_Z0 = [72.0 + 0.4j, 71.0 + 0.2j, 70.0 + 0.1j]


def unloaded_params():
    """Canned unloaded line parameters, as ``rf.run_unloaded`` returns them."""
    return line_params_from_neff(
        np.asarray(RF_FREQS, dtype=np.float64),
        UNLOADED_N_EFF,
        z0_ohm=UNLOADED_Z0,
        unloaded=True,
        bias_v=RF_BIAS_V,
        signal_contact=RF_SIGNAL_CONTACT,
    )


@pytest.fixture
def segmented(solved, monkeypatch):
    """The canned Study, its RF Stage answering the unloaded solve too."""
    solved.solves["unloaded"] = 0

    def run_unloaded(stage, *, force=False):
        if stage._unloaded_result is None or force:
            solved.solves["unloaded"] += 1
            stage._unloaded_result = unloaded_params()
        return stage._unloaded_result

    monkeypatch.setattr(RFStage, "run_unloaded", run_unloaded)
    return solved


class TestSegmentedElectrode:
    """A Traveling-wave electrode the Junction loads part of the way."""

    def hand_assembled(self, *, fill_factor, period_m, rf=None, unloaded=None):
        sweep = optical_sweep()
        return twmzm_figures_of_merit(
            rf_params() if rf is None else rf,
            OpticalPhaseSweep(
                voltages_v=sweep.voltages,
                dn_eff=sweep.index_shift,
                alpha_opt_db_cm=sweep.loss_db_cm,
                wavelength_um=WAVELENGTH_UM,
                n_group=N_GROUP,
            ),
            length_m=3e-3,
            unloaded=unloaded_params() if unloaded is None else unloaded,
            fill_factor=fill_factor,
            period_m=period_m,
        )

    def test_the_electrode_is_loaded_all_the_way_by_default(self, study):
        assert study.line.fill_factor == 1.0
        assert study.line.period_um == 50.0

    def test_a_fill_factor_outside_the_period_is_rejected(self, study):
        for fill_factor in (0.0, -0.2, 1.1):
            with pytest.raises(ValueError):
                study.line(fill_factor=fill_factor)

    def test_a_nonpositive_period_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.line(period_um=0.0)

    def test_a_period_longer_than_the_electrode_is_rejected(self, study):
        with pytest.raises(ValueError, match="period"):
            study.line(length_um=1000.0, fill_factor=0.5, period_um=2000.0)

    def test_a_segmented_electrode_is_a_whole_number_of_periods(self, study):
        study.line(length_um=3020.0, fill_factor=0.5, period_um=50.0)
        assert study.line.length_m == pytest.approx(3e-3)
        # Loaded all the way, the length is the configured one.
        study.line(fill_factor=1.0)
        assert study.line.length_m == pytest.approx(3.02e-3)

    def test_a_fill_factor_of_one_is_todays_report_exactly(self, segmented):
        segmented.line(length_um=3000.0, n_group=N_GROUP, fill_factor=1.0)

        report = segmented.line.run()
        today = hand_assembled(length_m=3e-3)

        assert np.array_equal(report.response, today.response)
        assert report.bandwidth_3db_hz == today.bandwidth_3db_hz
        assert report.walkoff_bandwidth_hz == today.walkoff_bandwidth_hz
        assert np.array_equal(report.velocity_mismatch, today.velocity_mismatch)
        assert np.array_equal(report.z0_ohm, today.z0_ohm)
        assert np.array_equal(report.vpi_l_vcm, today.vpi_l_vcm)
        for name, values in today.rlgc.items():
            assert np.array_equal(report.rlgc[name], values)

    def test_an_electrode_loaded_all_the_way_solves_no_unloaded_line(self, segmented):
        segmented.line(n_group=N_GROUP)
        segmented.line.run()
        assert segmented.solves["unloaded"] == 0

    def test_a_fill_factor_below_one_runs_the_unloaded_solve(self, segmented):
        segmented.line(n_group=N_GROUP, fill_factor=0.5)
        segmented.line.run()
        assert segmented.solves == {"optical": 1, "rf": 1, "unloaded": 1}

    def test_the_unloaded_solve_the_rf_stage_holds_is_reused(self, segmented):
        segmented.line(n_group=N_GROUP, fill_factor=0.5)
        segmented.line.run()
        segmented.line(fill_factor=0.7, period_um=100.0)
        segmented.line.run()
        assert segmented.solves == {"optical": 1, "rf": 1, "unloaded": 1}

    def test_the_report_is_the_periodic_lines(self, segmented):
        segmented.line(
            length_um=3000.0, n_group=N_GROUP, fill_factor=0.5, period_um=100.0
        )

        report = segmented.line.run()
        expected = self.hand_assembled(fill_factor=0.5, period_m=100e-6)

        assert report.fill_factor == 0.5
        assert report.period_m == pytest.approx(100e-6)
        assert report.z0_ohm == pytest.approx(expected.z0_ohm)
        assert report.n_rf == pytest.approx(expected.n_rf)
        assert report.alpha_rf_np_m == pytest.approx(expected.alpha_rf_np_m)
        assert report.response == pytest.approx(expected.response)

    def test_partial_loading_reaches_toward_fifty_ohm(self, segmented):
        segmented.line(n_group=N_GROUP)
        full = segmented.line.run()
        segmented.line(fill_factor=0.5)
        half = segmented.line.run()

        assert np.all(half.z0_ohm.real > full.z0_ohm.real)
        assert np.all(half.n_rf < full.n_rf)

    def test_modulation_efficiency_scales_with_the_fill_factor(self, segmented):
        segmented.line(n_group=N_GROUP)
        full = segmented.line.run()
        segmented.line(fill_factor=0.4)
        partial = segmented.line.run()

        assert partial.vpi_l_vcm == pytest.approx(full.vpi_l_vcm / 0.4)

    def test_both_lines_are_read_on_the_response_grid(self, segmented):
        grid = np.linspace(10e9, 100e9, 19)
        segmented.line(
            n_group=N_GROUP, fill_factor=0.5, response_frequencies_hz=grid.tolist()
        )

        report = segmented.line.run()
        expected = self.hand_assembled(
            fill_factor=0.5,
            period_m=50e-6,
            rf=rf_params().resampled(grid),
            unloaded=unloaded_params().resampled(grid),
        )

        assert report.freq_hz == pytest.approx(grid)
        assert report.z0_ohm == pytest.approx(expected.z0_ohm)
        assert report.response == pytest.approx(expected.response)

    def test_a_period_approaching_the_bragg_condition_warns(self, segmented):
        # 300 um at 100 GHz and an index near 2.7: over half way to Bragg.
        segmented.line(n_group=N_GROUP, fill_factor=0.5, period_um=300.0)
        with pytest.warns(UserWarning, match="Bragg"):
            segmented.line.run()

    def test_the_exported_two_port_is_the_cascade_of_its_periods(self, segmented):
        segmented.line(length_um=3000.0, fill_factor=0.5, period_um=100.0)
        loaded, unloaded = rf_params(), unloaded_params()
        period = segmented_period_abcd(
            gamma_loaded_per_m=loaded.gamma_per_m,
            z0_loaded_ohm=loaded.z0_ohm,
            gamma_unloaded_per_m=unloaded.gamma_per_m,
            z0_unloaded_ohm=unloaded.z0_ohm,
            fill_factor=0.5,
            period_m=100e-6,
        )
        cascade = np.linalg.matrix_power(period, 30)
        a, b = cascade[:, 0, 0], cascade[:, 0, 1]
        c, d = cascade[:, 1, 0], cascade[:, 1, 1]
        z_gen, z_load = segmented.line.z_gen_ohm, segmented.line.z_load_ohm

        assert segmented.line.driven_response() == pytest.approx(
            z_load / (a * z_load + b + z_gen * (c * z_load + d)), rel=1e-9
        )

    def test_the_export_says_the_line_is_segmented(self, segmented, tmp_path):
        segmented.line(fill_factor=0.5, period_um=100.0)
        text = segmented.line.export_touchstone(tmp_path / "line.s2p").read_text()
        assert "fill_factor = 0.5" in text

    def test_a_period_far_below_the_wavelength_warns_about_nothing(self, segmented):
        segmented.line(n_group=N_GROUP, fill_factor=0.5, period_um=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            segmented.line.run()


class TestLifecycle:
    def test_the_report_runs_the_stages_that_have_not_run(self, solved):
        solved.line(n_group=N_GROUP)

        solved.line.run()

        assert solved.solves == {"optical": 1, "rf": 1}
        assert solved.optical.has_run is True
        assert solved.rf.has_run is True

    def test_asking_twice_solves_once(self, solved):
        solved.line(n_group=N_GROUP)

        first = solved.line.run()

        assert solved.line.run() is first
        assert solved.solves == {"optical": 1, "rf": 1}

    def test_re_configuring_the_line_drops_the_report_and_no_solve(self, solved):
        solved.line(n_group=N_GROUP)
        solved.line.run()

        solved.line(length_um=5000.0)

        assert solved.line.has_run is False
        assert solved.optical.has_run is True
        assert solved.rf.has_run is True

        solved.line.run()
        assert solved.solves == {"optical": 1, "rf": 1}

    def test_re_configuring_a_stage_upstream_drops_the_report(self, solved):
        solved.line(n_group=N_GROUP)
        solved.line.run()

        solved.optical(wavelength_um=1.31)

        assert solved.optical.has_run is False
        assert solved.line.has_run is False
        assert solved.rf.has_run is True

    def test_the_line_is_the_last_stage_of_the_study(self, study):
        assert list(study.stages) == ["charge", "carriers", "optical", "rf", "line"]
        assert study.stages["line"] is study.line


class TestTwoPortExport:
    """The solved line leaves the Study as a circuit-simulator two-port."""

    def test_the_touchstone_file_carries_the_solved_line(self, solved, tmp_path):
        from gsim.common.circuit import line_smatrix, read_touchstone

        path = solved.line.export_touchstone(tmp_path / "electrode.s2p")

        rf = rf_params()
        expected = line_smatrix(
            rf.gamma_per_m, rf.z0_ohm, length_m=solved.line.length_m
        )
        two_port = read_touchstone(path)
        assert two_port.z_ref_ohm == 50.0
        np.testing.assert_allclose(two_port.freq_hz, RF_FREQS)
        np.testing.assert_allclose(two_port.s, expected, rtol=1e-10, atol=1e-15)

    def test_without_a_path_it_lands_in_the_line_stage_directory(self, solved):
        path = solved.line.export_touchstone()

        assert path == solved.stage_dir("line") / "electrode.s2p"
        assert path.exists()

    def test_the_export_runs_the_rf_stage_first(self, solved, tmp_path):
        assert solved.rf.has_run is False

        solved.line.export_touchstone(tmp_path / "line.s2p")

        assert solved.rf.has_run is True
        assert solved.solves == {"optical": 0, "rf": 1}

    def test_the_sax_model_reproduces_the_solved_matrix(self, solved):
        from gsim.common.circuit import line_smatrix

        model = solved.line.sax_model(z_ref_ohm=75.0)
        sdict = model()

        rf = rf_params()
        expected = line_smatrix(
            rf.gamma_per_m,
            rf.z0_ohm,
            length_m=solved.line.length_m,
            z_ref_ohm=75.0,
        )
        np.testing.assert_allclose(sdict[("o2", "o1")], expected[:, 1, 0])
        np.testing.assert_allclose(sdict[("o1", "o1")], expected[:, 0, 0])

    def test_the_export_reports_the_solved_length(self, solved, tmp_path):
        from gsim.common.circuit import read_touchstone

        solved.line(length_um=5000.0)

        path = solved.line.export_touchstone(tmp_path / "line.s2p")

        assert "length_m = 0.005" in read_touchstone(path).comments

    def test_the_export_records_the_bias_and_contact_off_the_rf_result(
        self, solved, tmp_path
    ):
        """The provenance is the record's, not the RF Stage's private state."""
        from gsim.common.circuit import read_touchstone

        path = solved.line.export_touchstone(tmp_path / "line.s2p")

        comments = read_touchstone(path).comments
        assert f"signal contact: {RF_SIGNAL_CONTACT}" in comments
        assert f"bias_v = {RF_BIAS_V:g}" in comments

    def test_a_result_without_provenance_records_none(
        self, study, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            RFStage,
            "_solve",
            lambda _stage: line_params_from_neff(
                np.asarray(RF_FREQS, dtype=np.float64), RF_N_EFF, z0_ohm=RF_Z0
            ),
        )

        path = study.line.export_touchstone(tmp_path / "line.s2p")

        text = path.read_text()
        assert "signal contact" not in text
        assert "bias_v" not in text


class TestExportRoundTrip:
    """The handoff artifacts reassemble to the Study's own answers."""

    @pytest.fixture
    def exported(self, solved, monkeypatch):
        """A Study whose EM and charge Stages answer from canned results."""
        from gsim.modulator.charge import ChargeStage

        from .conftest import junction_sweep

        monkeypatch.setattr(ChargeStage, "_solve", lambda _stage: junction_sweep())
        solved.line(n_group=N_GROUP, z_load_ohm=45.0, z_gen_ohm=50.0)
        return solved

    def test_the_reassembled_response_matches_the_internal_one(self, exported):
        comparison = exported.line.verify_exports(quiet=True)

        assert comparison.check() is comparison
        assert np.max(comparison.response_rel_diff) < 1e-8
        np.testing.assert_allclose(comparison.freq_hz, RF_FREQS, rtol=1e-12)

    def test_the_junction_file_matches_the_charge_stage_exactly(self, exported):
        comparison = exported.line.verify_exports(quiet=True)

        assert (
            comparison.r_s_file_ohm_m.tolist() == comparison.r_s_internal_ohm_m.tolist()
        )
        assert (
            comparison.c_j_file_f_per_m.tolist()
            == comparison.c_j_internal_f_per_m.tolist()
        )

    def test_the_command_prints_the_side_by_side_table(self, exported, capsys):
        exported.line.verify_exports()

        out = capsys.readouterr().out
        assert "internal" in out
        assert "rel diff" in out
        assert "Junction branch" in out
        assert "exact" in out

    def test_the_internal_response_follows_the_terminations(self, exported):
        from gsim.common.circuit import line_driven_response

        rf = rf_params()
        expected = line_driven_response(
            rf.gamma_per_m,
            rf.z0_ohm,
            length_m=exported.line.length_m,
            z_gen_ohm=50.0,
            z_load_ohm=45.0,
        )
        np.testing.assert_allclose(exported.line.driven_response(), expected)

    def test_the_files_land_in_the_stage_directories_by_default(self, exported):
        comparison = exported.line.verify_exports(quiet=True)

        assert comparison.touchstone_path == (
            exported.stage_dir("line") / "electrode.s2p"
        )
        assert comparison.junction_path == (
            exported.stage_dir("charge") / "junction.json"
        )

    def test_a_diverging_response_names_the_quantity_and_frequency(self, exported):
        comparison = exported.line.verify_exports(quiet=True)
        comparison.reassembled = comparison.reassembled * 1.01

        with pytest.raises(ValueError, match=r"[Dd]riven response.*GHz"):
            comparison.check()

    def test_a_diverging_junction_column_names_the_bias(self, exported):
        comparison = exported.line.verify_exports(quiet=True)
        tampered = comparison.c_j_file_f_per_m.copy()
        tampered[1] *= 1.001
        comparison.c_j_file_f_per_m = tampered

        with pytest.raises(ValueError, match=r"C_j.*1 V"):
            comparison.check()

    def test_the_junction_columns_come_through_the_rf_stage(
        self, exported, monkeypatch
    ):
        """The line Stage reads the EM Stages; the charge sweep is the RF Stage's."""
        seen = []
        branches = RFStage.junction_branches

        def spy(stage):
            seen.append(stage)
            return branches(stage)

        monkeypatch.setattr(RFStage, "junction_branches", spy)

        comparison = exported.line.verify_exports(quiet=True)

        assert seen == [exported.rf]
        assert comparison.bias_v.tolist() == [0.0, 1.0, 2.0]
        assert comparison.check() is comparison

    def test_a_nan_response_fails_the_check_rather_than_passing(self, exported):
        # NaN compares False against any tolerance; the gate must not
        # read that as agreement.
        comparison = exported.line.verify_exports(quiet=True)
        tampered = comparison.internal.copy()
        tampered[0] = complex(float("nan"), float("nan"))
        comparison.internal = tampered

        with pytest.raises(ValueError, match="not finite"):
            comparison.check()
