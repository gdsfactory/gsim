"""Tests for the TW-MZM physics layer (gsim.modulator.twmzm).

Analytic checks use the standard traveling-wave modulator limits:

- perfect velocity match, no loss, matched impedances -> flat response;
- velocity mismatch only -> |sin(u)/u| walk-off roll-off with
  u = pi f L (n_rf - n_opt) / c;
- RF loss only -> |(1 - exp(-alpha L)) / (alpha L)|;
- V_pi L = lambda / (2 |d(dn_eff)/dV|).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.modulator.twmzm import (
    eo_bandwidth,
    eo_response,
    mzm_chirp,
    mzm_drive_range,
    mzm_transfer,
    mzm_transfer_figures,
    segmented_eo_response,
    vpi_length_vcm,
    walkoff_bandwidth,
    walkoff_bandwidth_dispersive,
)


class TestEOResponse:
    def test_matched_lossless_velocity_matched_is_flat(self):
        freq = np.linspace(1e6, 100e9, 50)
        m = eo_response(
            freq,
            length_m=5e-3,
            n_rf=4.0,
            n_opt=4.0,
            alpha_rf_np_m=0.0,
            z0_ohm=50.0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        assert np.allclose(np.abs(m), 1.0, atol=1e-9)

    def test_velocity_mismatch_sinc_rolloff(self):
        length = 10e-3
        n_rf, n_opt = 4.0, 3.6
        freq = np.array([1e9, 20e9, 50e9])
        m = eo_response(
            freq,
            length_m=length,
            n_rf=n_rf,
            n_opt=n_opt,
            alpha_rf_np_m=0.0,
            z0_ohm=50.0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        u = np.pi * freq * length * (n_rf - n_opt) / C0
        expected = np.abs(np.sin(u) / u)
        assert np.abs(m) == pytest.approx(expected, rel=1e-9)

    def test_loss_only_rolloff(self):
        length = 8e-3
        alpha = 200.0  # Np/m
        freq = np.array([10e9])
        m = eo_response(
            freq,
            length_m=length,
            n_rf=3.8,
            n_opt=3.8,
            alpha_rf_np_m=alpha,
            z0_ohm=50.0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
            normalize=False,
        )
        # Matched drive halves the generator voltage; loss averages to
        # (1 - exp(-alpha L)) / (alpha L).
        expected = 0.5 * (1.0 - np.exp(-alpha * length)) / (alpha * length)
        assert np.abs(m[0]) == pytest.approx(expected, rel=1e-9)

    def test_normalized_to_unity_at_dc(self):
        freq = np.linspace(1e5, 60e9, 30)
        m = eo_response(
            freq,
            length_m=6e-3,
            n_rf=3.9,
            n_opt=3.6,
            alpha_rf_np_m=150.0,
            z0_ohm=42.0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        # DC-normalized: response tends to 1 at low frequency.
        assert np.abs(m[0]) == pytest.approx(1.0, abs=1e-3)

    def test_frequency_dependent_line_params(self):
        freq = np.linspace(1e8, 50e9, 20)
        n_rf = np.linspace(4.1, 3.9, 20)
        alpha = 30.0 * np.sqrt(freq / 1e9)
        z0 = np.linspace(48.0, 44.0, 20)
        m = eo_response(
            freq,
            length_m=4e-3,
            n_rf=n_rf,
            n_opt=3.7,
            alpha_rf_np_m=alpha,
            z0_ohm=z0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        assert m.shape == freq.shape
        assert np.all(np.isfinite(np.abs(m)))
        assert np.abs(m[-1]) < np.abs(m[0])

    def test_mismatched_load_differs_from_matched(self):
        freq = np.array([25e9])
        kwargs = dict(
            length_m=5e-3,
            n_rf=3.9,
            n_opt=3.6,
            alpha_rf_np_m=100.0,
            z0_ohm=40.0,
            z_gen_ohm=50.0,
        )
        matched = eo_response(freq, z_load_ohm=40.0, **kwargs)
        mismatched = eo_response(freq, z_load_ohm=50.0, **kwargs)
        assert np.abs(matched[0]) != pytest.approx(np.abs(mismatched[0]), rel=1e-6)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            eo_response(
                np.array([1e9]),
                length_m=0.0,
                n_rf=3.9,
                n_opt=3.6,
                alpha_rf_np_m=0.0,
                z0_ohm=50.0,
                z_load_ohm=50.0,
                z_gen_ohm=50.0,
            )


class TestEOBandwidth:
    def test_walkoff_bandwidth_analytic(self):
        # Lossless matched line: |sin u / u| = 1/sqrt(2) at u ~ 1.3916.
        length = 10e-3
        n_rf, n_opt = 4.0, 3.6
        expected = 1.3915573 * C0 / (np.pi * length * (n_rf - n_opt))
        assert walkoff_bandwidth(
            length_m=length, n_rf=n_rf, n_opt=n_opt
        ) == pytest.approx(expected, rel=1e-5)

    def test_bandwidth_from_response_matches_walkoff(self):
        length = 10e-3
        n_rf, n_opt = 4.0, 3.6
        freq = np.linspace(1e6, 40e9, 4000)
        m = eo_response(
            freq,
            length_m=length,
            n_rf=n_rf,
            n_opt=n_opt,
            alpha_rf_np_m=0.0,
            z0_ohm=50.0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        f3db = eo_bandwidth(freq, m)
        assert f3db == pytest.approx(
            walkoff_bandwidth(length_m=length, n_rf=n_rf, n_opt=n_opt), rel=1e-3
        )

    def test_no_crossing_returns_none(self):
        freq = np.linspace(1e6, 10e9, 10)
        m = np.ones(10, dtype=complex)
        assert eo_bandwidth(freq, m) is None


class TestVpiLength:
    def test_linear_dneff(self):
        # dn_eff = s * V with s = 5e-5 / V at 1.55 um:
        # V_pi L = lambda / (2 s) = 1.55 / 1e-4 um V = 1.55 V cm.
        v = np.linspace(0.0, 4.0, 9)
        dneff = 5e-5 * v
        vpil = vpi_length_vcm(v, dneff, wavelength_um=1.55)
        assert vpil == pytest.approx(np.full(9, 1.55), rel=1e-9)

    def test_sublinear_dneff_increases_with_bias(self):
        # Depletion-type saturation: slope falls, V_pi L grows with bias.
        v = np.linspace(0.0, 4.0, 41)
        dneff = 1e-4 * np.sqrt(v + 0.5)
        vpil = vpi_length_vcm(v, dneff, wavelength_um=1.55)
        assert np.all(np.diff(vpil) > 0)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            vpi_length_vcm(np.array([0.0, 1.0]), np.array([0.0]), wavelength_um=1.55)


class _MachZehnder:
    """A Phase shifter whose index shift is linear in bias, in both arms."""

    WL_UM = 1.55
    LENGTH_M = 3e-3
    SLOPE = 1e-4
    VOLTAGES = np.linspace(0.0, 8.0, 17)
    DN = SLOPE * VOLTAGES
    #: lambda / (2 |slope| L): the Modulation efficiency over the length.
    V_PI = WL_UM * 1e-6 / (2.0 * SLOPE * LENGTH_M)

    def settings(self, **overrides):
        return {"length_m": self.LENGTH_M, "wavelength_um": self.WL_UM} | overrides


class TestMZMTransfer(_MachZehnder):
    @pytest.mark.parametrize("drive", ["single-drive", "push-pull"])
    def test_lossless_balanced_arms_give_a_raised_cosine(self, drive):
        low, high = mzm_drive_range(self.VOLTAGES, drive=drive)
        v = np.linspace(low, high, 101)
        transfer = mzm_transfer(v, self.VOLTAGES, self.DN, **self.settings(drive=drive))
        # Quadrature by default: half power at rest, falling as arm 1's
        # phase delay grows.
        expected = 0.5 * (1.0 + np.cos(np.pi * v / self.V_PI + np.pi / 2.0))
        assert transfer == pytest.approx(expected, abs=1e-12)

    def test_the_drive_voltage_is_the_voltage_between_the_arms(self):
        assert mzm_drive_range(self.VOLTAGES, drive="single-drive") == (-4.0, 4.0)
        assert mzm_drive_range(self.VOLTAGES, drive="push-pull") == (-8.0, 8.0)
        # A single drive resting on the start of the sweep has all of it.
        assert mzm_drive_range(self.VOLTAGES, drive="single-drive", bias_v=0.0) == (
            0.0,
            8.0,
        )

    def test_the_static_phase_offset_moves_the_rest_point(self):
        at_rest = [
            mzm_transfer(
                0.0, self.VOLTAGES, self.DN, **self.settings(phase_offset_rad=offset)
            )
            for offset in (0.0, np.pi / 2.0, np.pi)
        ]
        assert at_rest == pytest.approx([1.0, 0.5, 0.0], abs=1e-12)

    def test_an_arm_imbalance_fills_in_the_null(self):
        # 3 dB more power down arm 1: r = 0.666, null at (sqrt(r) - sqrt(1-r))^2 / 2.
        fraction = 1.0 / (1.0 + 10.0 ** (-0.3))
        null = mzm_transfer(
            0.0,
            self.VOLTAGES,
            self.DN,
            **self.settings(arm_imbalance_db=3.0, phase_offset_rad=np.pi),
        )
        assert null == pytest.approx(
            (np.sqrt(fraction) - np.sqrt(1.0 - fraction)) ** 2 / 2.0
        )

    def test_the_transfer_is_not_extrapolated_past_the_sweep(self):
        with pytest.raises(ValueError, match="not extrapolated"):
            mzm_transfer(
                4.5, self.VOLTAGES, self.DN, **self.settings(drive="single-drive")
            )

    def test_a_push_pull_bias_on_the_end_of_the_sweep_is_refused(self):
        with pytest.raises(ValueError, match="either side"):
            mzm_drive_range(self.VOLTAGES, drive="push-pull", bias_v=0.0)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="outside the bias sweep"):
            mzm_drive_range(self.VOLTAGES, bias_v=9.0)
        with pytest.raises(ValueError, match="drive configuration"):
            mzm_drive_range(self.VOLTAGES, drive="dual")
        with pytest.raises(ValueError, match="ascending"):
            mzm_transfer(0.0, self.VOLTAGES[::-1], self.DN, **self.settings())
        with pytest.raises(ValueError, match="alpha_opt_db_cm"):
            mzm_transfer(0.0, self.VOLTAGES, self.DN, [1.0], **self.settings())
        with pytest.raises(ValueError, match="length_m"):
            mzm_transfer(0.0, self.VOLTAGES, self.DN, **self.settings(length_m=0.0))


class TestMZMTransferFigures(_MachZehnder):
    @pytest.mark.parametrize("drive", ["single-drive", "push-pull"])
    def test_v_pi_is_the_modulation_efficiency_over_the_length(self, drive):
        figures = mzm_transfer_figures(
            self.VOLTAGES, self.DN, **self.settings(drive=drive)
        )
        v_pi_l_vcm = vpi_length_vcm(self.VOLTAGES, self.DN, wavelength_um=self.WL_UM)
        assert figures.v_pi_v == pytest.approx(v_pi_l_vcm[0] / (self.LENGTH_M * 1e2))
        assert figures.v_pi_v == pytest.approx(self.V_PI)
        # At quadrature the peak and the null sit either side of rest.
        assert figures.peak_v == pytest.approx(-self.V_PI / 2.0)
        assert figures.null_v == pytest.approx(self.V_PI / 2.0)
        assert figures.message is None

    def test_lossless_balanced_arms_extinguish_completely(self):
        figures = mzm_transfer_figures(self.VOLTAGES, self.DN, **self.settings())
        assert figures.insertion_loss_db == pytest.approx(0.0, abs=1e-12)
        assert figures.extinction_ratio_db == np.inf

    def test_equal_loss_in_both_arms_costs_insertion_loss_and_no_extinction(self):
        # 10 dB/cm over 3 mm, whatever the bias: 3 dB off the peak and the
        # null alike.
        loss = np.full_like(self.VOLTAGES, 10.0)
        settings = self.settings(arm_imbalance_db=1.0)
        lossless = mzm_transfer_figures(self.VOLTAGES, self.DN, **settings)
        lossy = mzm_transfer_figures(self.VOLTAGES, self.DN, loss, **settings)
        assert lossless.insertion_loss_db is not None
        assert lossy.insertion_loss_db == pytest.approx(
            lossless.insertion_loss_db + 3.0
        )
        assert lossy.extinction_ratio_db == pytest.approx(lossless.extinction_ratio_db)
        assert lossy.v_pi_v == pytest.approx(lossless.v_pi_v)

    def test_an_arm_imbalance_limits_the_extinction(self):
        figures = mzm_transfer_figures(
            self.VOLTAGES, self.DN, **self.settings(arm_imbalance_db=0.5)
        )
        # Field ratio q = sqrt(P_2 / P_1): ER = ((1 + q) / (1 - q))^2.
        q = 10.0 ** (-0.5 / 20.0)
        assert figures.extinction_ratio_db == pytest.approx(
            20.0 * np.log10((1.0 + q) / (1.0 - q))
        )

    def test_a_loss_that_moves_with_bias_limits_the_extinction(self):
        # The arms sit at different voltages at the null, so their losses
        # differ there and the fields no longer cancel.
        loss = 10.0 - 1.0 * self.VOLTAGES
        figures = mzm_transfer_figures(self.VOLTAGES, self.DN, loss, **self.settings())
        # Push-pull peak and null at -/+ V_pi/2: the arms at 4 V -/+ V_pi/4
        # and the other way round, the same two amplitudes adding and then
        # subtracting.
        arm_v = 4.0 + np.array([1.0, -1.0]) * self.V_PI / 4.0
        one, other = 10.0 ** (-(10.0 - arm_v) * (self.LENGTH_M * 1e2) / 20.0)
        assert figures.extinction_ratio_db == pytest.approx(
            20.0 * np.log10((one + other) / abs(one - other))
        )
        assert figures.insertion_loss_db == pytest.approx(
            -20.0 * np.log10((one + other) / 2.0)
        )

    def test_a_sweep_too_short_to_reach_v_pi_says_so(self):
        # 0 to 2 V around a 1 V rest: a swing of 2 V, short of the 2.58 V.
        short = self.VOLTAGES <= 2.0
        figures = mzm_transfer_figures(
            self.VOLTAGES[short],
            self.DN[short],
            **self.settings(drive="single-drive"),
        )
        assert figures.v_pi_v is None
        assert figures.insertion_loss_db is None
        assert figures.extinction_ratio_db is None
        assert figures.message is not None
        assert "too short to reach V_pi" in figures.message
        assert "single-drive span of -1 to 1 V" in figures.message
        assert "0.77 pi" in figures.message

    def test_the_same_sweep_reaches_v_pi_driven_push_pull(self):
        short = self.VOLTAGES <= 2.0
        figures = mzm_transfer_figures(
            self.VOLTAGES[short], self.DN[short], **self.settings(drive="push-pull")
        )
        assert figures.v_pi_v == pytest.approx(self.V_PI)

    def test_a_curved_index_shift_is_read_off_the_transfer_not_the_slope(self):
        # Sub-linear, as a depleting Junction is: the swing between the peak
        # and the null is where the phase difference really moves by pi.
        voltages = np.linspace(0.0, 12.0, 49)
        dn = 3e-4 * np.sqrt(voltages)
        settings = self.settings(drive="single-drive", bias_v=0.0, phase_offset_rad=0.0)
        figures = mzm_transfer_figures(voltages, dn, **settings)
        assert figures.peak_v == pytest.approx(0.0)
        assert figures.null_v is not None
        assert figures.v_pi_v is not None
        transfer = mzm_transfer(
            [figures.peak_v, figures.null_v], voltages, dn, **settings
        )
        assert transfer == pytest.approx([1.0, 0.0], abs=1e-12)
        phase = 2.0 * np.pi * np.interp(figures.v_pi_v, voltages, dn) * self.LENGTH_M
        assert phase / (self.WL_UM * 1e-6) == pytest.approx(np.pi)


class TestMZMChirp(_MachZehnder):
    LOSS = 10.0 - 0.5 * _MachZehnder.VOLTAGES

    def test_an_ideal_push_pull_drive_has_no_chirp(self):
        chirp = mzm_chirp(
            self.VOLTAGES,
            self.DN,
            np.full_like(self.VOLTAGES, 10.0),
            wavelength_um=self.WL_UM,
            drive="push-pull",
        )
        assert np.all(chirp == 0.0)

    def test_a_lossless_single_drive_has_unit_chirp_signed_by_the_quadrature(self):
        for offset, expected in ((np.pi / 2.0, 1.0), (-np.pi / 2.0, -1.0)):
            for slope in (1.0, -1.0):
                chirp = mzm_chirp(
                    self.VOLTAGES,
                    slope * self.DN,
                    wavelength_um=self.WL_UM,
                    drive="single-drive",
                    phase_offset_rad=offset,
                )
                assert chirp == pytest.approx(expected)

    def _slope_ratio(self):
        """``lambda alpha' / (4 pi n')`` with alpha the power attenuation."""
        alpha_slope_per_m = -0.5 * 1e2 * np.log(10.0) / 10.0
        return self.WL_UM * 1e-6 * alpha_slope_per_m / (4.0 * np.pi * self.SLOPE)

    def test_the_absorption_slope_pulls_a_single_drive_off_unit_chirp(self):
        chirp = mzm_chirp(
            self.VOLTAGES,
            self.DN,
            self.LOSS,
            wavelength_um=self.WL_UM,
            drive="single-drive",
        )
        ratio = self._slope_ratio()
        assert chirp == pytest.approx((1.0 - ratio) / (1.0 + ratio))
        assert np.all(chirp != 1.0)

    def test_the_absorption_slope_leaves_a_push_pull_drive_a_residue(self):
        chirp = mzm_chirp(
            self.VOLTAGES,
            self.DN,
            self.LOSS,
            wavelength_um=self.WL_UM,
            drive="push-pull",
        )
        assert chirp == pytest.approx(-self._slope_ratio())

    def test_an_arm_imbalance_chirps_a_push_pull_drive(self):
        chirp = mzm_chirp(
            self.VOLTAGES,
            self.DN,
            wavelength_um=self.WL_UM,
            drive="push-pull",
            arm_imbalance_db=1.0,
        )
        fraction = 1.0 / (1.0 + 10.0 ** (-0.1))
        assert chirp == pytest.approx(
            (fraction - 0.5) / np.sqrt(fraction * (1.0 - fraction))
        )

    def test_the_chirp_matches_the_output_field_differentiated_numerically(self):
        # The definition itself, on the complex output field of a lossy,
        # unbalanced single-drive modulator: Im / Re of d ln(E) / dv.
        fraction = 1.0 / (1.0 + 10.0 ** (-0.1))
        offset = 0.7
        bias, step = 4.0, 1e-4

        def field(v):
            arm_1 = bias + v
            amplitude = 10.0 ** (-(10.0 - 0.5 * arm_1) * self.LENGTH_M * 1e2 / 20.0)
            phase = (
                2.0 * np.pi * self.SLOPE * arm_1 * self.LENGTH_M / (self.WL_UM * 1e-6)
            )
            rest = 10.0 ** (-(10.0 - 0.5 * bias) * self.LENGTH_M * 1e2 / 20.0)
            rest_phase = (
                2.0 * np.pi * self.SLOPE * bias * self.LENGTH_M / (self.WL_UM * 1e-6)
            )
            return np.sqrt(fraction) * amplitude * np.exp(
                -1j * (phase + offset)
            ) + np.sqrt(1.0 - fraction) * rest * np.exp(-1j * rest_phase)

        log_slope = (field(step) - field(-step)) / (2.0 * step) / field(0.0)
        chirp = mzm_chirp(
            self.VOLTAGES,
            self.DN,
            self.LOSS,
            wavelength_um=self.WL_UM,
            drive="single-drive",
            arm_imbalance_db=1.0,
            phase_offset_rad=offset,
        )
        assert chirp[8] == pytest.approx(log_slope.imag / log_slope.real, rel=1e-6)


class TestSegmentedEOResponse:
    """Only the loaded sections of a segmented electrode modulate the light."""

    FREQ = np.array([1e9, 10e9, 40e9, 100e9])
    N_OPT = 3.8
    LOADED: ClassVar = {
        "n_rf_loaded": 4.0,
        "alpha_loaded_np_m": 150.0,
        "z0_loaded_ohm": 28.0 - 1.5j,
    }
    UNLOADED: ClassVar = {
        "n_rf_unloaded": 2.2,
        "alpha_unloaded_np_m": 30.0,
        "z0_unloaded_ohm": 70.0 - 0.5j,
    }
    LOSSLESS: ClassVar = {
        "n_rf_loaded": 4.0,
        "alpha_loaded_np_m": 0.0,
        "z0_loaded_ohm": 28.0,
        "n_rf_unloaded": 2.2,
        "alpha_unloaded_np_m": 0.0,
        "z0_unloaded_ohm": 70.0,
    }

    def section_by_section(
        self, freq_hz, *, n_periods, period_m, fill_factor, z_load, z_gen
    ):
        """Cascade every section and integrate V(z) over the loaded ones."""
        omega = 2 * np.pi * freq_hz
        gamma_l = self.LOADED["alpha_loaded_np_m"] + 1j * omega * 4.0 / C0
        gamma_u = self.UNLOADED["alpha_unloaded_np_m"] + 1j * omega * 2.2 / C0
        z_l, z_u = self.LOADED["z0_loaded_ohm"], self.UNLOADED["z0_unloaded_ohm"]
        beta_opt = omega * self.N_OPT / C0
        half, bare = 0.5 * fill_factor * period_m, (1 - fill_factor) * period_m
        sections = [
            (gamma_l, z_l, half, True),
            (gamma_u, z_u, bare, False),
            (gamma_l, z_l, half, True),
        ] * n_periods

        def abcd(gamma, z0, length):
            theta = gamma * length
            return np.array(
                [
                    [np.cosh(theta), z0 * np.sinh(theta)],
                    [np.sinh(theta) / z0, np.cosh(theta)],
                ]
            )

        total = np.eye(2, dtype=complex)
        for gamma, z0, length, _ in sections:
            total = total @ abcd(gamma, z0, length)
        z_in = (total[0, 0] * z_load + total[0, 1]) / (
            total[1, 0] * z_load + total[1, 1]
        )
        state = np.array([z_in / (z_in + z_gen), 1.0 / (z_in + z_gen)])

        integral, start = 0.0j, 0.0
        for gamma, z0, length, loaded in sections:
            if loaded:
                z = np.linspace(0.0, length, 201)
                volts = state[0] * np.cosh(gamma * z) - state[1] * z0 * np.sinh(
                    gamma * z
                )
                integral += np.trapezoid(volts * np.exp(1j * beta_opt * (start + z)), z)
            state = np.linalg.solve(abcd(gamma, z0, length), state)
            start += length
        return integral / start

    def test_it_matches_a_section_by_section_integration(self):
        setup = {"n_periods": 12, "period_m": 250e-6, "fill_factor": 0.6}
        response = segmented_eo_response(
            self.FREQ,
            **setup,
            n_opt=self.N_OPT,
            **self.LOADED,
            **self.UNLOADED,
            z_load_ohm=45.0,
            z_gen_ohm=50.0,
            normalize=False,
        )
        expected = [
            self.section_by_section(f, **setup, z_load=45.0, z_gen=50.0)
            for f in self.FREQ
        ]
        assert response == pytest.approx(np.array(expected), rel=1e-5)

    def test_a_fill_factor_of_one_is_the_uniform_response(self):
        response = segmented_eo_response(
            self.FREQ,
            n_periods=60,
            period_m=50e-6,
            fill_factor=1.0,
            n_opt=self.N_OPT,
            **self.LOADED,
            **self.UNLOADED,
            z_load_ohm=45.0,
            z_gen_ohm=50.0,
        )
        uniform = eo_response(
            self.FREQ,
            length_m=3e-3,
            n_rf=4.0,
            n_opt=self.N_OPT,
            alpha_rf_np_m=150.0,
            z0_ohm=28.0 - 1.5j,
            z_load_ohm=45.0,
            z_gen_ohm=50.0,
        )
        assert response == pytest.approx(uniform, rel=1e-9)

    def test_the_drive_acts_over_the_loaded_fraction_only(self):
        # Lossless at low frequency the line voltage is uniform, so the
        # averaged drive is the fill factor times the terminated voltage.
        response = segmented_eo_response(
            [1e6],
            n_periods=60,
            period_m=50e-6,
            fill_factor=0.4,
            n_opt=self.N_OPT,
            **self.LOSSLESS,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
            normalize=False,
        )
        assert response[0] == pytest.approx(0.4 * 0.5, rel=1e-6)

    def test_normalized_to_unity_at_dc(self):
        response = segmented_eo_response(
            [1e5],
            n_periods=60,
            period_m=50e-6,
            fill_factor=0.4,
            n_opt=self.N_OPT,
            **self.LOADED,
            **self.UNLOADED,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        assert abs(response[0]) == pytest.approx(1.0, abs=1e-6)

    def test_a_lossless_line_normalizes_without_a_dc_impedance(self):
        response = segmented_eo_response(
            [1e6],
            n_periods=60,
            period_m=50e-6,
            fill_factor=0.4,
            n_opt=self.N_OPT,
            **self.LOSSLESS,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        assert abs(response[0]) == pytest.approx(1.0, abs=1e-6)

    def test_an_electrode_loaded_nowhere_is_rejected(self):
        with pytest.raises(ValueError, match="fill_factor"):
            segmented_eo_response(
                self.FREQ,
                n_periods=60,
                period_m=50e-6,
                fill_factor=0.0,
                n_opt=self.N_OPT,
                **self.LOADED,
                **self.UNLOADED,
                z_load_ohm=50.0,
                z_gen_ohm=50.0,
            )


class TestDispersiveWalkoff:
    """The walk-off limit of a line whose RF index moves with frequency."""

    FREQ = np.linspace(10e9, 100e9, 10)
    LENGTH = 10e-3
    N_OPT = 3.8

    def test_flat_index_recovers_the_closed_form(self):
        expected = walkoff_bandwidth(length_m=self.LENGTH, n_rf=6.0, n_opt=self.N_OPT)
        assert walkoff_bandwidth_dispersive(
            self.FREQ, np.full(10, 6.0), length_m=self.LENGTH, n_opt=self.N_OPT
        ) == pytest.approx(expected)

    def test_velocity_matched_everywhere_is_unbounded(self):
        assert (
            walkoff_bandwidth_dispersive(
                self.FREQ, np.full(10, self.N_OPT), length_m=1e-3, n_opt=self.N_OPT
            )
            is None
        )

    def test_limit_is_self_consistent_inside_the_solved_range(self):
        n_rf = np.linspace(4.6, 4.0, 10)
        limit = walkoff_bandwidth_dispersive(
            self.FREQ, n_rf, length_m=self.LENGTH, n_opt=self.N_OPT
        )
        assert limit is not None
        assert self.FREQ[0] < limit < self.FREQ[-1]
        mismatch = abs(np.interp(limit, self.FREQ, n_rf) - self.N_OPT)
        assert limit == pytest.approx(
            walkoff_bandwidth(
                length_m=self.LENGTH, n_rf=self.N_OPT + mismatch, n_opt=self.N_OPT
            ),
            rel=1e-6,
        )

    def test_index_crossing_the_group_index_does_not_average_away(self):
        # n_RF crosses n_g mid-band, so its mean mismatch is ~0 and a limit
        # read off the mean runs away. Past the solved range the last
        # solved index is held, and the limit follows from that one.
        n_rf = np.linspace(4.2, 3.4, 10)
        limit = walkoff_bandwidth_dispersive(
            self.FREQ, n_rf, length_m=3e-3, n_opt=self.N_OPT
        )
        assert limit == pytest.approx(
            walkoff_bandwidth(length_m=3e-3, n_rf=3.4, n_opt=self.N_OPT)
        )

    def test_limit_below_the_solved_range_holds_the_first_index(self):
        limit = walkoff_bandwidth_dispersive(
            self.FREQ, np.linspace(9.0, 8.0, 10), length_m=50e-3, n_opt=self.N_OPT
        )
        assert limit < self.FREQ[0]
        assert limit == pytest.approx(
            walkoff_bandwidth(length_m=50e-3, n_rf=9.0, n_opt=self.N_OPT)
        )
