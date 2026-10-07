"""Tests for the line theory of gsim.common.transmission_line.

The transmission line on its own terms, with no modulator in sight: the
RLGC extraction, the series-RC junction branch and the loaded line it
makes, the periodic line a segmented electrode behaves as, and the
:class:`RFLineParams` record every Route hands its answer back in.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.transmission_line import (
    JunctionBranch,
    bragg_fraction,
    line_params_from_gamma,
    line_params_from_neff,
    loaded_line_params,
    rlgc_from_line_params,
    section_abcd,
    segmented_line,
    segmented_line_params,
    segmented_period_abcd,
    series_rc_from_admittance,
)
from tests._helpers import series_rc_admittance

FREQ = np.linspace(1e9, 100e9, 200)


class TestRLGC:
    def test_lossless_line_roundtrip(self):
        l_per_m = 2.5e-7
        c_per_m = 1e-10
        z0 = np.sqrt(l_per_m / c_per_m)  # 50 ohm
        freq = np.array([1e9, 10e9])
        omega = 2 * np.pi * freq
        gamma = 1j * omega * np.sqrt(l_per_m * c_per_m)
        rlgc = rlgc_from_line_params(freq, gamma_per_m=gamma, z0_ohm=z0)
        assert rlgc["R"] == pytest.approx(np.zeros(2), abs=1e-9)
        assert rlgc["L"] == pytest.approx(np.full(2, l_per_m), rel=1e-12)
        assert rlgc["G"] == pytest.approx(np.zeros(2), abs=1e-12)
        assert rlgc["C"] == pytest.approx(np.full(2, c_per_m), rel=1e-12)

    def test_lossy_line_has_positive_r(self):
        freq = np.array([5e9])
        omega = 2 * np.pi * freq
        gamma = 40.0 + 1j * omega * 4.0 / C0  # alpha = 40 Np/m, n_rf = 4
        rlgc = rlgc_from_line_params(freq, gamma_per_m=gamma, z0_ohm=45.0 + 2.0j)
        assert rlgc["R"][0] > 0
        assert rlgc["C"][0] > 0


class TestSeriesRCFromAdmittance:
    def test_recovers_the_branch_it_came_from(self):
        r_s, c_j = 8e-4, 2.4e-10  # 0.8 ohm mm, 0.24 fF/um
        y = series_rc_admittance(1e6, r_s, c_j)
        r_fit, c_fit = series_rc_from_admittance(y, freq_hz=1e6)
        assert r_fit == pytest.approx(r_s, rel=1e-12)
        assert c_fit == pytest.approx(c_j, rel=1e-12)

    def test_pure_capacitor_has_zero_resistance(self):
        c_j = 1e-10
        omega = 2 * np.pi * 1e6
        r_fit, c_fit = series_rc_from_admittance(1j * omega * c_j, freq_hz=1e6)
        assert r_fit == pytest.approx(0.0, abs=1e-15)
        assert c_fit == pytest.approx(c_j, rel=1e-12)

    def test_a_sweep_of_admittances_fits_pointwise(self):
        r_s = np.array([1e-3, 2e-3, 3e-3])
        c_j = np.array([3e-10, 2e-10, 1e-10])
        y = series_rc_admittance(1e6, r_s, c_j)
        r_fit, c_fit = series_rc_from_admittance(y, freq_hz=1e6)
        assert r_fit == pytest.approx(r_s, rel=1e-12)
        assert c_fit == pytest.approx(c_j, rel=1e-12)

    def test_rejects_an_inductive_admittance(self):
        with pytest.raises(ValueError, match="capacitive"):
            series_rc_from_admittance(1e-3 - 1e-4j, freq_hz=1e6)

    def test_rejects_a_negative_conductance(self):
        y = series_rc_admittance(1e6, 1e-3, 2e-10)
        with pytest.raises(ValueError, match="series RC"):
            series_rc_from_admittance(-y.real + 1j * y.imag, freq_hz=1e6)

    def test_rejects_a_nonpositive_frequency(self):
        with pytest.raises(ValueError, match="frequency"):
            series_rc_from_admittance(1e-3 + 1e-4j, freq_hz=0.0)


class TestLoadedLineParams:
    UNLOADED: ClassVar = {
        "R": np.zeros(2),
        "L": np.full(2, 4e-7),
        "G": np.zeros(2),
        "C": np.full(2, 8e-11),
    }
    FREQ = np.array([1e9, 10e9])

    def test_zero_junction_branch_recovers_the_unloaded_line(self):
        gamma, z0 = loaded_line_params(
            self.FREQ, rlgc=self.UNLOADED, junction=(0.0, 0.0)
        )
        omega = 2 * np.pi * self.FREQ
        assert gamma == pytest.approx(1j * omega * np.sqrt(4e-7 * 8e-11), rel=1e-12)
        assert z0 == pytest.approx(np.full(2, np.sqrt(4e-7 / 8e-11)), rel=1e-12)

    def test_a_lossless_junction_adds_its_capacitance(self):
        c_j = 2e-10
        gamma, z0 = loaded_line_params(
            self.FREQ, rlgc=self.UNLOADED, junction=(0.0, c_j)
        )
        omega = 2 * np.pi * self.FREQ
        c_total = 8e-11 + c_j
        assert gamma == pytest.approx(1j * omega * np.sqrt(4e-7 * c_total), rel=1e-12)
        assert z0 == pytest.approx(np.sqrt(4e-7 / c_total) * np.ones(2), rel=1e-12)

    def test_the_series_resistance_makes_the_line_lossy(self):
        gamma, z0 = loaded_line_params(
            self.FREQ, rlgc=self.UNLOADED, junction=JunctionBranch(2e-3, 2e-10)
        )
        # Hand-computed: Z = jwL', Y = jwC' + jwC_j / (1 + jwR_sC_j).
        omega = 2 * np.pi * self.FREQ
        y_j = 1j * omega * 2e-10 / (1 + 1j * omega * 2e-3 * 2e-10)
        z_series = 1j * omega * 4e-7
        y_shunt = 1j * omega * 8e-11 + y_j
        assert gamma == pytest.approx(np.sqrt(z_series * y_shunt), rel=1e-12)
        assert z0 == pytest.approx(np.sqrt(z_series / y_shunt), rel=1e-12)
        assert np.all(gamma.real > 0)

    def test_rejects_mismatched_rlgc_shapes(self):
        bad = dict(self.UNLOADED, L=np.full(3, 4e-7))
        with pytest.raises(ValueError, match="shape"):
            loaded_line_params(self.FREQ, rlgc=bad, junction=(0.0, 0.0))


class TestSegmentedLine:
    """A Traveling-wave electrode loaded part of the way, as a periodic line."""

    FREQ = np.array([1e9, 10e9, 40e9, 100e9])
    PERIOD_M = 50e-6

    @classmethod
    def lines(cls) -> dict:
        """A lossy ~28 ohm loaded line and a ~70 ohm unloaded one."""
        omega = 2 * np.pi * cls.FREQ
        return {
            "gamma_loaded_per_m": 120.0 * np.sqrt(cls.FREQ / 1e10)
            + 1j * omega * 4.0 / C0,
            "z0_loaded_ohm": np.full(cls.FREQ.shape, 28.0 - 1.5j),
            "gamma_unloaded_per_m": 25.0 * np.sqrt(cls.FREQ / 1e10)
            + 1j * omega * 2.2 / C0,
            "z0_unloaded_ohm": np.full(cls.FREQ.shape, 70.0 - 0.5j),
        }

    def test_one_period_is_a_reciprocal_symmetric_two_port(self):
        abcd = segmented_period_abcd(
            **self.lines(), fill_factor=0.6, period_m=self.PERIOD_M
        )
        a, b, c, d = abcd[:, 0, 0], abcd[:, 0, 1], abcd[:, 1, 0], abcd[:, 1, 1]
        assert abcd.shape == (self.FREQ.size, 2, 2)
        assert a * d - b * c == pytest.approx(np.ones(self.FREQ.size), rel=1e-12)
        assert a == pytest.approx(d, rel=1e-12)

    def test_the_bloch_wave_solves_the_period(self):
        lines = self.lines()
        abcd = segmented_period_abcd(**lines, fill_factor=0.6, period_m=self.PERIOD_M)
        gamma, z_bloch = segmented_line_params(
            **lines, fill_factor=0.6, period_m=self.PERIOD_M
        )
        # (V, I) = (Z_B, 1) exp(-Gamma n) is an eigenvector of the period.
        step = np.exp(gamma * self.PERIOD_M)
        assert abcd[:, 0, 0] * z_bloch + abcd[:, 0, 1] == pytest.approx(
            step * z_bloch, rel=1e-9
        )
        assert abcd[:, 1, 0] * z_bloch + abcd[:, 1, 1] == pytest.approx(step, rel=1e-9)

    def test_the_passive_branch_is_taken(self):
        gamma, z_bloch = segmented_line_params(
            **self.lines(), fill_factor=0.6, period_m=self.PERIOD_M
        )
        assert np.all(gamma.real >= 0)
        assert np.all(gamma.imag > 0)
        assert np.all(z_bloch.real > 0)

    def test_a_lossless_line_takes_the_forward_wave(self):
        omega = 2 * np.pi * self.FREQ
        gamma, z_bloch = segmented_line_params(
            gamma_loaded_per_m=1j * omega * 4.0 / C0,
            z0_loaded_ohm=28.0,
            gamma_unloaded_per_m=1j * omega * 2.2 / C0,
            z0_unloaded_ohm=70.0,
            fill_factor=0.5,
            period_m=self.PERIOD_M,
        )
        assert gamma.real == pytest.approx(np.zeros(self.FREQ.size), abs=1e-6)
        assert np.all(gamma.imag > 0)
        assert np.all(z_bloch.real > 0)

    def test_a_fill_factor_of_one_is_the_loaded_line(self):
        lines = self.lines()
        gamma, z_bloch = segmented_line_params(
            **lines, fill_factor=1.0, period_m=self.PERIOD_M
        )
        assert gamma == pytest.approx(lines["gamma_loaded_per_m"], rel=1e-12)
        assert z_bloch == pytest.approx(lines["z0_loaded_ohm"], rel=1e-12)

    def test_a_fill_factor_of_zero_is_the_unloaded_line(self):
        lines = self.lines()
        gamma, z_bloch = segmented_line_params(
            **lines, fill_factor=0.0, period_m=self.PERIOD_M
        )
        assert gamma == pytest.approx(lines["gamma_unloaded_per_m"], rel=1e-12)
        assert z_bloch == pytest.approx(lines["z0_unloaded_ohm"], rel=1e-12)

    def test_a_short_period_is_the_length_weighted_average(self):
        lines = self.lines()
        fill = 0.6
        gamma, z_bloch = segmented_line_params(
            **lines, fill_factor=fill, period_m=self.PERIOD_M
        )
        # Series impedance gamma*Z0 and shunt admittance gamma/Z0 per meter,
        # each averaged by length along the period.
        g_l, z_l = lines["gamma_loaded_per_m"], lines["z0_loaded_ohm"]
        g_u, z_u = lines["gamma_unloaded_per_m"], lines["z0_unloaded_ohm"]
        series = fill * g_l * z_l + (1 - fill) * g_u * z_u
        shunt = fill * g_l / z_l + (1 - fill) * g_u / z_u

        short = (
            bragg_fraction(
                gamma_loaded_per_m=g_l,
                gamma_unloaded_per_m=g_u,
                fill_factor=fill,
                period_m=self.PERIOD_M,
            )
            < 0.05
        )
        assert short.sum() >= 3
        assert gamma[short] == pytest.approx(np.sqrt(series * shunt)[short], rel=1e-3)
        assert z_bloch[short] == pytest.approx(np.sqrt(series / shunt)[short], rel=1e-3)

    def test_partial_loading_raises_the_impedance_and_lowers_the_index(self):
        lines = self.lines()
        gamma_half, z_half = segmented_line_params(
            **lines, fill_factor=0.5, period_m=self.PERIOD_M
        )
        assert np.all(np.abs(z_half) > np.abs(lines["z0_loaded_ohm"]))
        assert np.all(gamma_half.imag < lines["gamma_loaded_per_m"].imag)

    def test_rejects_a_fill_factor_outside_the_period(self):
        with pytest.raises(ValueError, match="fill_factor"):
            segmented_line_params(**self.lines(), fill_factor=1.2, period_m=50e-6)
        with pytest.raises(ValueError, match="period_m"):
            segmented_line_params(**self.lines(), fill_factor=0.5, period_m=0.0)


class TestBraggFraction:
    def test_it_is_the_phase_advance_per_period_over_pi(self):
        beta_l = 2 * np.pi * 30e9 * 4.0 / C0
        beta_u = 2 * np.pi * 30e9 * 2.0 / C0
        fraction = bragg_fraction(
            gamma_loaded_per_m=10.0 + 1j * beta_l,
            gamma_unloaded_per_m=1j * beta_u,
            fill_factor=0.25,
            period_m=100e-6,
        )
        assert fraction == pytest.approx(
            (0.25 * beta_l + 0.75 * beta_u) * 100e-6 / np.pi
        )

    def test_the_lossless_stop_band_is_open_where_it_reaches_one(self):
        # Two quarter-wave sections: the centre of the first stop band,
        # where the Bloch wave decays with no loss in either line.
        period, n_l, n_u = 1e-3, 4.0, 2.0
        fill = n_u / (n_l + n_u)
        freq = C0 / (4 * n_l * fill * period)
        gamma_l = 1j * 2 * np.pi * freq * n_l / C0
        gamma_u = 1j * 2 * np.pi * freq * n_u / C0
        fraction = bragg_fraction(
            gamma_loaded_per_m=gamma_l,
            gamma_unloaded_per_m=gamma_u,
            fill_factor=fill,
            period_m=period,
        )
        gamma, _ = segmented_line_params(
            gamma_loaded_per_m=gamma_l,
            z0_loaded_ohm=28.0,
            gamma_unloaded_per_m=gamma_u,
            z0_unloaded_ohm=70.0,
            fill_factor=fill,
            period_m=period,
        )
        assert fraction == pytest.approx(1.0)
        assert gamma.real > 0


class TestLineParamsExtraction:
    def test_complex_neff_hand_values(self):
        freq = np.array([10e9, 50e9])
        n_eff = np.array([2.0 - 0.1j, 2.5 - 0.2j])
        rf = line_params_from_neff(freq, n_eff, z0_ohm=40.0)
        np.testing.assert_allclose(rf.n_rf, [2.0, 2.5])
        np.testing.assert_allclose(
            rf.alpha_rf_np_m, 2.0 * np.pi * freq * [0.1, 0.2] / C0
        )
        np.testing.assert_allclose(rf.z0_ohm, 40.0)

    def test_either_imag_sign_gives_loss(self):
        freq = np.array([10e9])
        plus = line_params_from_neff(freq, [2.0 + 0.1j], z0_ohm=50.0)
        minus = line_params_from_neff(freq, [2.0 - 0.1j], z0_ohm=50.0)
        np.testing.assert_allclose(plus.alpha_rf_np_m, minus.alpha_rf_np_m)
        assert plus.alpha_rf_np_m[0] > 0

    def test_gamma_round_trips_through_rlgc(self):
        # Lossless 50-ohm line: R = G = 0, L/C give back n_rf and Z0.
        rf = line_params_from_neff(FREQ, 2.5 + 0j, z0_ohm=50.0)
        rlgc = rf.rlgc
        np.testing.assert_allclose(rlgc["R"], 0.0, atol=1e-9)
        np.testing.assert_allclose(rlgc["G"], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.sqrt(rlgc["L"] / rlgc["C"]), 50.0, rtol=1e-9)
        np.testing.assert_allclose(C0 * np.sqrt(rlgc["L"] * rlgc["C"]), 2.5, rtol=1e-9)

    def test_scalar_broadcast(self):
        rf = line_params_from_neff(FREQ, 2.0 - 0.05j, z0_ohm=45.0)
        assert rf.n_rf.shape == FREQ.shape
        assert rf.z0_ohm.shape == FREQ.shape


class TestUnloadedFlag:
    def test_line_params_are_loaded_unless_said_otherwise(self):
        rf = line_params_from_neff(FREQ, 2.5 + 0j, z0_ohm=50.0)
        assert rf.unloaded is False

    def test_an_unloaded_solve_is_flagged(self):
        rf = line_params_from_neff(FREQ, 2.5 + 0j, z0_ohm=50.0, unloaded=True)
        assert rf.unloaded is True


class TestLineParamsFromGamma:
    def test_roundtrips_the_gamma_property(self):
        rf = line_params_from_neff(FREQ, 2.5 - 0.05j, z0_ohm=45.0 + 2.0j)

        back = line_params_from_gamma(FREQ, rf.gamma_per_m, z0_ohm=rf.z0_ohm)

        assert back.n_rf == pytest.approx(rf.n_rf)
        assert back.alpha_rf_np_m == pytest.approx(rf.alpha_rf_np_m)
        assert back.z0_ohm == pytest.approx(rf.z0_ohm)
        assert back.unloaded is False

    def test_hand_values(self):
        freq = np.array([10e9])
        omega = 2 * np.pi * freq
        gamma = 30.0 + 1j * omega * 2.5 / C0

        rf = line_params_from_gamma(freq, gamma, z0_ohm=40.0)

        assert rf.n_rf == pytest.approx([2.5])
        assert rf.alpha_rf_np_m == pytest.approx([30.0])

    def test_either_sign_convention_is_loss(self):
        freq = np.array([10e9])
        omega = 2 * np.pi * freq
        gamma = -30.0 - 1j * omega * 2.5 / C0

        rf = line_params_from_gamma(freq, gamma, z0_ohm=40.0)

        assert rf.n_rf == pytest.approx([2.5])
        assert rf.alpha_rf_np_m == pytest.approx([30.0])


class TestProvenance:
    def test_line_params_carry_no_bias_or_contact_unless_given(self):
        line = line_params_from_neff([10e9], [2.0 - 0.01j], z0_ohm=[50.0])
        assert line.bias_v is None
        assert line.signal_contact is None

    def test_the_bias_and_the_contact_travel_with_the_record(self):
        line = line_params_from_neff(
            [10e9], [2.0 - 0.01j], z0_ohm=[50.0], bias_v=2.0, signal_contact="cathode"
        )
        assert line.bias_v == 2.0
        assert line.signal_contact == "cathode"
        from_gamma = line_params_from_gamma(
            line.freq_hz, line.gamma_per_m, z0_ohm=line.z0_ohm, bias_v=2.0
        )
        assert from_gamma.bias_v == 2.0


class TestResampling:
    def _line(self):
        return line_params_from_neff(
            [10e9, 20e9, 40e9],
            [3.2 - 0.004j, 3.1 - 0.010j, 3.0 - 0.020j],
            z0_ohm=[46.0 + 1.0j, 44.0 + 0.5j, 42.0 + 0.2j],
            bias_v=1.5,
            signal_contact="cathode",
        )

    def test_the_solved_points_are_reproduced(self):
        line = self._line()
        same = line.resampled(line.freq_hz)
        np.testing.assert_allclose(same.n_rf, line.n_rf)
        np.testing.assert_allclose(same.alpha_rf_np_m, line.alpha_rf_np_m)
        np.testing.assert_allclose(same.z0_ohm, line.z0_ohm)

    def test_between_points_every_quantity_interpolates_linearly(self):
        line = self._line()
        mid = line.resampled([15e9, 30e9])
        np.testing.assert_allclose(mid.n_rf, [3.15, 3.05])
        np.testing.assert_allclose(
            mid.alpha_rf_np_m,
            np.interp([15e9, 30e9], line.freq_hz, line.alpha_rf_np_m),
        )
        # Real and imaginary parts of the impedance interpolate separately.
        np.testing.assert_allclose(mid.z0_ohm, [45.0 + 0.75j, 43.0 + 0.35j])

    def test_outside_the_solved_range_the_end_values_hold(self):
        line = self._line()
        clamped = line.resampled([1e9, 100e9])
        np.testing.assert_allclose(clamped.n_rf, [3.2, 3.0])
        np.testing.assert_allclose(clamped.z0_ohm, [46.0 + 1.0j, 42.0 + 0.2j])

    def test_the_provenance_and_the_flag_travel_with_it(self):
        line = self._line().model_copy(update={"unloaded": True})
        resampled = line.resampled([12e9])
        assert resampled.bias_v == 1.5
        assert resampled.signal_contact == "cathode"
        assert resampled.unloaded is True

    def test_a_descending_grid_is_refused(self):
        with pytest.raises(ValueError, match="ascending"):
            self._line().resampled([20e9, 10e9])


class TestSectionABCD:
    """The telegrapher two-port every cascade here is assembled from."""

    FREQ = np.array([1e9, 40e9])

    def theta(self, length_m=2e-3):
        gamma = 30.0 + 1j * 2 * np.pi * self.FREQ * 3.5 / C0
        return gamma * length_m

    def test_it_is_the_telegrapher_matrix(self):
        z0 = 45.0 - 2.0j
        theta = self.theta()
        abcd = section_abcd(theta, z0)
        assert abcd.shape == (2, 2, 2)
        assert abcd[..., 0, 0] == pytest.approx(np.cosh(theta))
        assert abcd[..., 1, 1] == pytest.approx(np.cosh(theta))
        assert abcd[..., 0, 1] == pytest.approx(z0 * np.sinh(theta))
        assert abcd[..., 1, 0] == pytest.approx(np.sinh(theta) / z0)

    def test_two_halves_cascade_into_the_whole_section(self):
        z0 = np.full(self.FREQ.shape, 50.0 + 0j)
        half = section_abcd(0.5 * self.theta(), z0)
        assert half @ half == pytest.approx(section_abcd(self.theta(), z0))

    def test_a_lossless_section_is_reciprocal_and_symmetric(self):
        theta = 1j * 2 * np.pi * self.FREQ * 3.5 * 2e-3 / C0
        abcd = section_abcd(theta, 50.0)
        assert np.linalg.det(abcd) == pytest.approx(np.ones(self.FREQ.shape))
        assert abcd[..., 0, 0] == pytest.approx(abcd[..., 1, 1])

    def test_scalars_broadcast(self):
        assert section_abcd(0.1j, 50.0).shape == (2, 2)


class TestSegmentedLineRecord:
    """The Bloch line of a segmented electrode, as line parameters."""

    FREQ = np.array([10e9, 40e9])

    def loaded(self):
        return line_params_from_neff(self.FREQ, 4.0 - 0.06j, z0_ohm=28.0 - 1.5j)

    def unloaded(self):
        return line_params_from_neff(
            self.FREQ, 2.2 - 0.01j, z0_ohm=70.0 - 0.5j, unloaded=True
        )

    def test_it_carries_the_bloch_constants(self):
        fill_factor, period_m = 0.6, 50e-6
        line = segmented_line(
            self.loaded(),
            self.unloaded(),
            fill_factor=fill_factor,
            period_m=period_m,
        )
        gamma, z_bloch = segmented_line_params(
            gamma_loaded_per_m=self.loaded().gamma_per_m,
            z0_loaded_ohm=self.loaded().z0_ohm,
            gamma_unloaded_per_m=self.unloaded().gamma_per_m,
            z0_unloaded_ohm=self.unloaded().z0_ohm,
            fill_factor=fill_factor,
            period_m=period_m,
        )
        assert line.gamma_per_m == pytest.approx(gamma)
        assert line.z0_ohm == pytest.approx(z_bloch)

    def test_a_fill_factor_of_one_is_the_loaded_line(self):
        line = segmented_line(
            self.loaded(), self.unloaded(), fill_factor=1.0, period_m=50e-6
        )
        assert line.n_rf == pytest.approx(self.loaded().n_rf, rel=1e-9)
        assert line.z0_ohm == pytest.approx(self.loaded().z0_ohm, rel=1e-9)

    def test_the_loaded_line_provenance_travels_with_it(self):
        loaded = line_params_from_neff(
            self.FREQ, 4.0 - 0.06j, z0_ohm=28.0, bias_v=-2.0, signal_contact="anode"
        )
        line = segmented_line(loaded, self.unloaded(), fill_factor=0.5, period_m=50e-6)
        assert line.bias_v == -2.0
        assert line.signal_contact == "anode"
        assert line.unloaded is False

    def test_the_two_lines_must_share_one_frequency_axis(self):
        with pytest.raises(ValueError, match="one freq_hz axis"):
            segmented_line(
                self.loaded(),
                self.unloaded().resampled(self.FREQ[:1]),
                fill_factor=0.5,
                period_m=50e-6,
            )
