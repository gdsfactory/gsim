"""Hermetic tests: solver outputs wired into the TW-MZM device report.

Synthetic mode results drive the wiring; results are checked against the
analytic limits of the physics functions in :mod:`gsim.modulator.twmzm`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.transmission_line import (
    line_params_from_neff,
    segmented_line_params,
)
from gsim.modulator.report import (
    SILICON_ONLY_RTOL,
    LoadedLineComparison,
    OpticalPhaseSweep,
    twmzm_figures_of_merit,
)
from gsim.modulator.twmzm import (
    SINC_3DB_ARGUMENT,
    segmented_eo_response,
    walkoff_bandwidth,
)

FREQ = np.linspace(1e9, 100e9, 200)
WL_UM = 1.55


def _optical(n_group=3.8, slope=-1e-4):
    voltages = np.linspace(0.0, 4.0, 9)
    return OpticalPhaseSweep(
        voltages_v=voltages,
        dn_eff=slope * voltages,
        wavelength_um=WL_UM,
        n_group=n_group,
    )


class TestFiguresOfMerit:
    def test_matched_lossless_velocity_matched_is_flat(self):
        optical = _optical(n_group=2.5)
        rf = line_params_from_neff(FREQ, 2.5 + 0j, z0_ohm=50.0)
        report = twmzm_figures_of_merit(
            rf, optical, length_m=5e-3, z_load_ohm=50.0, z_gen_ohm=50.0
        )
        np.testing.assert_allclose(np.abs(report.response), 1.0, atol=1e-9)
        assert report.bandwidth_3db_hz is None
        assert report.walkoff_bandwidth_hz is None
        np.testing.assert_allclose(report.velocity_mismatch, 0.0)

    def test_velocity_mismatch_reproduces_walkoff_limit(self):
        n_rf, n_opt, length = 6.0, 3.8, 10e-3
        optical = _optical(n_group=n_opt)
        rf = line_params_from_neff(FREQ, n_rf + 0j, z0_ohm=50.0)
        report = twmzm_figures_of_merit(rf, optical, length_m=length)
        expected = walkoff_bandwidth(length_m=length, n_rf=n_rf, n_opt=n_opt)
        assert report.walkoff_bandwidth_hz == pytest.approx(expected)
        # Lossless matched line: the full response's 3 dB point is the
        # analytic sinc walk-off frequency.
        assert report.bandwidth_3db_hz == pytest.approx(expected, rel=1e-2)
        # And the sinc shape itself is reproduced.
        u = np.pi * FREQ * length * abs(n_rf - n_opt) / C0
        np.testing.assert_allclose(
            np.abs(report.response), np.abs(np.sinc(u / np.pi)), atol=1e-6
        )

    def test_vpi_l_matches_closed_form_for_linear_sweep(self):
        slope = -2e-4
        optical = _optical(slope=slope)
        rf = line_params_from_neff(FREQ, 3.8 + 0j, z0_ohm=50.0)
        report = twmzm_figures_of_merit(rf, optical, length_m=5e-3)
        expected_vcm = WL_UM / (2.0 * abs(slope)) / 1e4
        np.testing.assert_allclose(report.vpi_l_vcm, expected_vcm, rtol=1e-9)

    def test_sinc_3db_argument_constant(self):
        u = SINC_3DB_ARGUMENT
        assert abs(np.sin(u) / u) == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-12)

    def test_rejects_nonpositive_length(self):
        rf = line_params_from_neff(FREQ, 2.5 + 0j, z0_ohm=50.0)
        with pytest.raises(ValueError, match="length_m"):
            twmzm_figures_of_merit(rf, _optical(), length_m=0.0)

    def test_report_carries_line_and_bias_axes(self):
        optical = _optical()
        rf = line_params_from_neff(FREQ, 4.0 - 0.02j, z0_ohm=42.0 + 3.0j)
        report = twmzm_figures_of_merit(rf, optical, length_m=3e-3)
        assert report.freq_hz.shape == FREQ.shape
        assert report.vpi_l_vcm.shape == optical.voltages_v.shape
        assert set(report.rlgc) == {"R", "L", "G", "C"}
        assert report.z_load_ohm == 50.0 + 0j

    def test_walkoff_limit_is_not_read_off_the_mean_index(self):
        # n_RF crossing n_g inside the band has a mean mismatch near zero;
        # the limit must follow the index the line actually has.
        freq = np.linspace(10e9, 100e9, 50)
        n_rf = np.linspace(4.2, 3.4, 50)
        rf = line_params_from_neff(freq, n_rf + 0j, z0_ohm=50.0)
        report = twmzm_figures_of_merit(rf, _optical(n_group=3.8), length_m=3e-3)
        assert report.walkoff_bandwidth_hz == pytest.approx(
            walkoff_bandwidth(length_m=3e-3, n_rf=3.4, n_opt=3.8)
        )


class TestMachZehnder:
    """The Phase shifter put in the arms of a Mach-Zehnder, in the report."""

    RF = line_params_from_neff(FREQ, 3.8 + 0j, z0_ohm=50.0)
    SLOPE = -2e-4
    LENGTH_M = 5e-3
    #: lambda / (2 |slope| L) for the linear sweep of ``_optical``: 0.775 V.
    V_PI = WL_UM * 1e-6 / (2.0 * abs(SLOPE) * LENGTH_M)

    def _report(self, optical=None, **settings):
        optical = _optical(slope=self.SLOPE) if optical is None else optical
        return twmzm_figures_of_merit(
            self.RF, optical, length_m=self.LENGTH_M, **settings
        )

    def test_push_pull_at_quadrature_is_the_default(self):
        report = self._report()
        assert report.drive == "push-pull"
        # The arms rest on the middle of the 0-4 V sweep, and the voltage
        # between them can then reach +/- 4 V.
        assert report.arm_bias_v == pytest.approx(2.0)
        assert report.drive_v[[0, -1]] == pytest.approx([-4.0, 4.0])
        assert report.transfer.shape == report.drive_v.shape
        assert np.interp(0.0, report.drive_v, report.transfer) == pytest.approx(0.5)

    def test_v_pi_is_the_modulation_efficiency_over_the_length(self):
        report = self._report()
        assert report.v_pi_v == pytest.approx(self.V_PI)
        assert report.v_pi_v == pytest.approx(
            report.vpi_l_vcm[0] / (self.LENGTH_M * 1e2)
        )
        assert report.transfer_message is None

    def test_a_lossless_balanced_modulator_has_no_loss_and_full_extinction(self):
        report = self._report()
        assert report.insertion_loss_db == pytest.approx(0.0, abs=1e-12)
        assert report.extinction_ratio_db == np.inf
        assert report.transfer.max() == pytest.approx(1.0, abs=1e-4)
        assert report.transfer.min() == pytest.approx(0.0, abs=1e-4)

    def test_the_loss_sweep_reaches_the_insertion_loss(self):
        optical = _optical(slope=self.SLOPE)
        optical.alpha_opt_db_cm = np.full_like(optical.voltages_v, 8.0)
        report = self._report(optical)
        # 8 dB/cm over 5 mm, in both arms alike.
        assert report.insertion_loss_db == pytest.approx(4.0)
        assert report.extinction_ratio_db == np.inf

    def test_the_arm_imbalance_reaches_the_extinction_ratio(self):
        report = self._report(arm_imbalance_db=0.5)
        q = 10.0 ** (-0.5 / 20.0)
        assert report.extinction_ratio_db == pytest.approx(
            20.0 * np.log10((1.0 + q) / (1.0 - q))
        )

    def test_the_phase_offset_moves_the_rest_point(self):
        report = self._report(phase_offset_rad=0.0)
        assert np.interp(0.0, report.drive_v, report.transfer) == pytest.approx(1.0)

    def test_the_chirp_follows_the_drive_configuration(self):
        push_pull = self._report()
        single = self._report(drive="single-drive")
        assert push_pull.chirp.shape == push_pull.voltages_v.shape
        assert np.all(push_pull.chirp == 0.0)
        assert single.chirp == pytest.approx(1.0)
        assert self._report(
            drive="single-drive", phase_offset_rad=-np.pi / 2.0
        ).chirp == pytest.approx(-1.0)

    def test_a_single_drive_rests_where_it_is_told_to(self):
        report = self._report(drive="single-drive", arm_bias_v=0.0)
        assert report.arm_bias_v == 0.0
        assert report.drive_v[[0, -1]] == pytest.approx([0.0, 4.0])

    def test_a_sweep_too_short_to_reach_v_pi_says_so(self):
        # A tenth of the index shift: V_pi is 7.75 V and the sweep spans 4.
        report = self._report(_optical(slope=self.SLOPE / 10.0), drive="single-drive")
        assert report.v_pi_v is None
        assert report.insertion_loss_db is None
        assert report.extinction_ratio_db is None
        assert "too short to reach V_pi" in report.transfer_message
        # The transfer it did cover is still reported.
        assert np.all(np.isfinite(report.transfer))

    def test_a_sweep_out_of_bias_order_is_read_in_order(self):
        ordered = _optical(slope=self.SLOPE)
        ordered.alpha_opt_db_cm = 8.0 - ordered.voltages_v + 0.1 * ordered.voltages_v**2
        shuffle = np.array([3, 0, 8, 1, 5, 2, 7, 4, 6])
        shuffled = OpticalPhaseSweep(
            voltages_v=ordered.voltages_v[shuffle],
            dn_eff=ordered.dn_eff[shuffle],
            alpha_opt_db_cm=ordered.alpha_opt_db_cm[shuffle],
            wavelength_um=WL_UM,
            n_group=3.8,
        )
        expected = self._report(ordered, drive="single-drive")
        report = self._report(shuffled, drive="single-drive")
        assert report.transfer == pytest.approx(expected.transfer)
        assert report.v_pi_v == pytest.approx(expected.v_pi_v)
        # The chirp is per bias point, so it stays in the sweep's own order.
        assert report.chirp == pytest.approx(expected.chirp[shuffle])


class TestSegmentedElectrode:
    """The report of a Traveling-wave electrode loaded by fill factor."""

    @staticmethod
    def loaded():
        return line_params_from_neff(FREQ, 4.0 - 0.06j, z0_ohm=28.0 - 1.5j)

    @staticmethod
    def unloaded():
        return line_params_from_neff(
            FREQ, 2.2 - 0.01j, z0_ohm=70.0 - 0.5j, unloaded=True
        )

    def segmented(self, fill_factor, **kwargs):
        return twmzm_figures_of_merit(
            self.loaded(),
            _optical(),
            length_m=3e-3,
            unloaded=self.unloaded(),
            fill_factor=fill_factor,
            period_m=50e-6,
            **kwargs,
        )

    def test_a_fill_factor_of_one_is_todays_report_exactly(self):
        today = twmzm_figures_of_merit(self.loaded(), _optical(), length_m=3e-3)
        report = self.segmented(1.0)

        assert report.fill_factor == 1.0
        assert np.array_equal(report.response, today.response)
        assert report.bandwidth_3db_hz == today.bandwidth_3db_hz
        assert report.walkoff_bandwidth_hz == today.walkoff_bandwidth_hz
        assert np.array_equal(report.velocity_mismatch, today.velocity_mismatch)
        assert np.array_equal(report.z0_ohm, today.z0_ohm)
        assert np.array_equal(report.vpi_l_vcm, today.vpi_l_vcm)
        for name, values in today.rlgc.items():
            assert np.array_equal(report.rlgc[name], values)

    def test_the_unsegmented_report_says_it_is_loaded_all_the_way(self):
        report = twmzm_figures_of_merit(self.loaded(), _optical(), length_m=3e-3)
        assert report.fill_factor == 1.0
        assert report.period_m is None
        assert np.array_equal(report.n_rf, self.loaded().n_rf)
        assert np.array_equal(report.alpha_rf_np_m, self.loaded().alpha_rf_np_m)

    def test_it_reports_the_bloch_line(self):
        loaded, unloaded = self.loaded(), self.unloaded()
        gamma, z_bloch = segmented_line_params(
            gamma_loaded_per_m=loaded.gamma_per_m,
            z0_loaded_ohm=loaded.z0_ohm,
            gamma_unloaded_per_m=unloaded.gamma_per_m,
            z0_unloaded_ohm=unloaded.z0_ohm,
            fill_factor=0.5,
            period_m=50e-6,
        )
        report = self.segmented(0.5)

        assert report.fill_factor == 0.5
        assert report.period_m == 50e-6
        assert report.z0_ohm == pytest.approx(z_bloch)
        assert report.n_rf == pytest.approx(gamma.imag * C0 / (2 * np.pi * FREQ))
        assert report.alpha_rf_np_m == pytest.approx(gamma.real)
        assert report.velocity_mismatch == pytest.approx(report.n_rf - 3.8)
        omega = 2 * np.pi * FREQ
        assert report.rlgc["R"] + 1j * omega * report.rlgc["L"] == pytest.approx(
            gamma * z_bloch
        )

    def test_partial_loading_reaches_toward_fifty_ohm(self):
        full, half = self.segmented(1.0), self.segmented(0.5)
        assert np.all(np.abs(half.z0_ohm - 50.0) < np.abs(full.z0_ohm - 50.0))
        assert np.all(half.n_rf < full.n_rf)

    def test_modulation_efficiency_scales_with_the_fill_factor(self):
        full, half = self.segmented(1.0), self.segmented(0.5)
        assert half.vpi_l_vcm == pytest.approx(full.vpi_l_vcm / 0.5)

    def test_the_electrode_is_a_whole_number_of_periods(self):
        # 3.02 mm holds 60.4 periods of 50 um: the device reported is the 60
        # period one, and every figure is taken over its 3 mm.
        report = twmzm_figures_of_merit(
            self.loaded(),
            _optical(),
            length_m=3.02e-3,
            unloaded=self.unloaded(),
            fill_factor=0.5,
            period_m=50e-6,
        )
        whole = self.segmented(0.5)
        assert report.length_m == pytest.approx(3e-3)
        assert report.walkoff_bandwidth_hz == pytest.approx(whole.walkoff_bandwidth_hz)
        assert report.v_pi_v == pytest.approx(whole.v_pi_v)
        assert report.response == pytest.approx(whole.response)

    def test_the_mach_zehnder_swings_over_the_loaded_length_only(self):
        full, half = self.segmented(1.0), self.segmented(0.5)
        # Half the electrode modulates, so the same index shift needs twice
        # the drive; the device V_pi L over the length says the same.
        assert half.v_pi_v == pytest.approx(2.0 * full.v_pi_v)
        assert half.v_pi_v == pytest.approx(half.vpi_l_vcm[0] / (3e-3 * 1e2))

    def test_the_response_is_the_loaded_sections_own(self):
        loaded, unloaded = self.loaded(), self.unloaded()
        report = self.segmented(0.5, z_load_ohm=45.0)
        expected = segmented_eo_response(
            FREQ,
            n_periods=60,
            period_m=50e-6,
            fill_factor=0.5,
            n_opt=3.8,
            n_rf_loaded=loaded.n_rf,
            alpha_loaded_np_m=loaded.alpha_rf_np_m,
            z0_loaded_ohm=loaded.z0_ohm,
            n_rf_unloaded=unloaded.n_rf,
            alpha_unloaded_np_m=unloaded.alpha_rf_np_m,
            z0_unloaded_ohm=unloaded.z0_ohm,
            z_load_ohm=45.0,
            z_gen_ohm=50.0,
        )
        assert report.response == pytest.approx(expected)

    def test_a_segmented_electrode_needs_the_unloaded_line_and_a_period(self):
        with pytest.raises(ValueError, match="unloaded"):
            twmzm_figures_of_merit(
                self.loaded(),
                _optical(),
                length_m=3e-3,
                fill_factor=0.5,
                period_m=50e-6,
            )
        with pytest.raises(ValueError, match="period_m"):
            twmzm_figures_of_merit(
                self.loaded(),
                _optical(),
                length_m=3e-3,
                unloaded=self.unloaded(),
                fill_factor=0.5,
            )

    def test_the_two_lines_share_one_frequency_axis(self):
        with pytest.raises(ValueError, match="freq_hz"):
            twmzm_figures_of_merit(
                self.loaded(),
                _optical(),
                length_m=3e-3,
                unloaded=self.unloaded().resampled(FREQ[::2]),
                fill_factor=0.5,
                period_m=50e-6,
            )

    def test_a_fill_factor_outside_the_period_is_rejected(self):
        for fill_factor in (0.0, 1.5):
            with pytest.raises(ValueError, match="fill_factor"):
                self.segmented(fill_factor)


def _line(n_rf=2.5, alpha=40.0, z0=45.0, unloaded=False):
    freq = np.array([10e9, 40e9])
    return line_params_from_neff(
        freq,
        n_rf - 1j * alpha * C0 / (2 * np.pi * freq),
        z0_ohm=z0,
        unloaded=unloaded,
    )


class TestLoadedLineComparison:
    def test_identical_routes_have_zero_deltas_and_pass(self):
        cmp = LoadedLineComparison(direct=_line(), assembled=_line())

        assert cmp.delta_n_rf == pytest.approx([0.0, 0.0], abs=1e-15)
        assert cmp.delta_alpha == pytest.approx([0.0, 0.0], abs=1e-15)
        assert cmp.delta_z0 == pytest.approx([0.0, 0.0], abs=1e-15)
        cmp.check()

    def test_the_deltas_are_relative_to_the_direct_route(self):
        cmp = LoadedLineComparison(direct=_line(n_rf=2.0), assembled=_line(n_rf=2.2))

        assert cmp.delta_n_rf == pytest.approx([0.1, 0.1])

    def test_a_diverged_index_fails_naming_quantity_and_frequency(self):
        cmp = LoadedLineComparison(direct=_line(n_rf=2.0), assembled=_line(n_rf=3.0))

        with pytest.raises(ValueError, match=r"n_RF.*10 GHz"):
            cmp.check()

    def test_a_diverged_impedance_fails_naming_the_impedance(self):
        cmp = LoadedLineComparison(direct=_line(z0=40.0), assembled=_line(z0=80.0))

        with pytest.raises(ValueError, match=r"Z0"):
            cmp.check()

    def test_the_default_gate_sits_just_outside_the_measured_gap(self):
        """Measured on the two demo devices, the oxide in the charge solve:
        at worst 15 % on n_RF, 41 % on the loss and 23 % on |Z0|."""
        inside = LoadedLineComparison(
            direct=_line(n_rf=2.0, alpha=40.0, z0=40.0),
            assembled=_line(n_rf=1.70, alpha=23.6, z0=49.2),
        )
        inside.check()

        slow = LoadedLineComparison(direct=_line(n_rf=2.0), assembled=_line(n_rf=1.5))
        with pytest.raises(ValueError, match="n_RF"):
            slow.check()
        quiet = LoadedLineComparison(
            direct=_line(alpha=40.0), assembled=_line(alpha=18.0)
        )
        with pytest.raises(ValueError, match="loss"):
            quiet.check()
        high = LoadedLineComparison(direct=_line(z0=40.0), assembled=_line(z0=54.0))
        with pytest.raises(ValueError, match="Z0"):
            high.check()

    def test_a_silicon_only_charge_solve_needs_the_wider_gate(self):
        """Measured without the oxide: 25 % on n_RF, 59 % on the loss, 42 % on |Z0|."""
        silicon_only = LoadedLineComparison(
            direct=_line(n_rf=2.0, alpha=40.0, z0=40.0),
            assembled=_line(n_rf=1.5, alpha=16.4, z0=56.8),
        )
        with pytest.raises(ValueError, match="n_RF"):
            silicon_only.check()
        silicon_only.check(**SILICON_ONLY_RTOL)

    def test_the_tolerances_are_adjustable(self):
        cmp = LoadedLineComparison(direct=_line(n_rf=2.0), assembled=_line(n_rf=2.2))

        cmp.check(rtol_n_rf=0.2)
        with pytest.raises(ValueError, match="n_RF"):
            cmp.check(rtol_n_rf=0.05)

    def test_mismatched_frequency_axes_are_rejected(self):
        long = line_params_from_neff(np.array([1e9, 2e9, 3e9]), 2.5, z0_ohm=45.0)

        with pytest.raises(ValueError, match="freq"):
            LoadedLineComparison(direct=_line(), assembled=long)
