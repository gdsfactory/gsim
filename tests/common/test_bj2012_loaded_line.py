"""Loaded-line closed form against the measured line of Baehr-Jones et al. 2012.

Baehr-Jones et al., "Ultralow drive voltage silicon traveling-wave
modulator", Opt. Express 20(11), 12014 (2012), DOI 10.1364/OE.20.012014,
report a metal-only control line and the same line loaded with its
junctions, both at 10 GHz. This builds the metal-only line's per-unit-length
circuit from the measured impedance, resistance and inductance, loads it
with the junction branch (1.6 ohm*cm in series with the 0 V capacitance of
their Fig. 2d) and compares the result with the measured loaded line. The
measurement is an independent reference for the loading algebra of
:func:`gsim.common.transmission_line.loaded_line_params`.

Everything is closed form: no solver, no extras, well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.junction import JunctionBranch
from gsim.common.transmission_line import (
    line_params_from_gamma,
    loaded_line_params,
    rlgc_from_line_params,
    segmented_line_params,
)
from gsim.common.twmzm import eo_bandwidth, eo_response

REFERENCE = json.loads(
    (Path(__file__).parents[2] / "nbs" / "data" / "bj2012_reference.json").read_text(
        encoding="utf-8"
    )
)
METAL = REFERENCE["metal_only_line_10GHz"]
LOADED = REFERENCE["loaded_line_10GHz"]
OPTICAL_N_GROUP = REFERENCE["optical"]["n_group"]
LENGTH_M = REFERENCE["geometry"]["phase_shifter_length_um"] * 1e-6
TERMINATION_OHM = REFERENCE["electro_optic"]["termination_ohm_per_arm"]
SOURCE_OHM = 50.0  # assumed, the paper does not state it

NP_PER_M_TO_DB_PER_CM = 20.0 / np.log(10.0) * 1e-2
F0_HZ = METAL["freq_ghz"] * 1e9
R_OHM_M = METAL["r_kohm_per_m"] * 1e3
L_H_M = METAL["l_nH_per_m"] * 1e-9
Z0_METAL = complex(METAL["z0_real_ohm"], METAL["z0_imag_ohm"])
# 1.6 ohm*cm -> ohm*m, and Fig. 2d at 0 V: 0.225 fF/um = 0.225 nF/m.
JUNCTION = JunctionBranch(r_s_ohm_m=1.6e-2, c_j_f_per_m=0.225e-9)


def _metal_only_rlgc() -> dict[str, np.ndarray]:
    omega = 2.0 * np.pi * F0_HZ
    gamma = (R_OHM_M + 1j * omega * L_H_M) / Z0_METAL
    return rlgc_from_line_params(
        np.array([F0_HZ]), gamma_per_m=np.array([gamma]), z0_ohm=Z0_METAL
    )


def _loaded_line(freq_hz, rlgc, junction=JUNCTION):
    gamma, z0 = loaded_line_params(freq_hz, rlgc=rlgc, junction=junction)
    return line_params_from_gamma(freq_hz, gamma, z0_ohm=z0)


def _eo_f3db_ghz(*, z_load_ohm: float, z_gen_ohm: float = SOURCE_OHM) -> float:
    """3 dB EO bandwidth of the loaded 5 mm line; R follows sqrt(f)."""
    freq = np.linspace(0.1e9, 60e9, 600)
    base = _metal_only_rlgc()
    rlgc = {
        "R": R_OHM_M * np.sqrt(freq / F0_HZ),
        "L": np.full_like(freq, base["L"][0]),
        "G": np.full_like(freq, base["G"][0]),
        "C": np.full_like(freq, base["C"][0]),
    }
    line = _loaded_line(freq, rlgc)
    response = eo_response(
        freq,
        length_m=LENGTH_M,
        n_rf=line.n_rf,
        n_opt=OPTICAL_N_GROUP,
        alpha_rf_np_m=line.alpha_rf_np_m,
        z0_ohm=line.z0_ohm,
        z_load_ohm=z_load_ohm,
        z_gen_ohm=z_gen_ohm,
    )
    bandwidth = eo_bandwidth(freq, response)
    assert bandwidth is not None
    return bandwidth / 1e9


def _brute_force_response(freq_hz, gamma, z0, z_load, z_gen, n_steps=2001):
    """DC-normalised V_eff per volt of source EMF by stepping exact ABCD sections.

    From the load (1 V, 1/z_load A) back to the input; the source condition
    V + I z_gen = EMF fixes the scale; the voltage is integrated numerically
    along the co-propagating light of group index ``OPTICAL_N_GROUP``.
    """
    z = np.linspace(0.0, LENGTH_M, n_steps)
    step = z[1] - z[0]
    cosh, sinh = np.cosh(gamma * step), np.sinh(gamma * step)
    v = np.empty(n_steps, dtype=complex)
    i = np.empty(n_steps, dtype=complex)
    v[-1], i[-1] = 1.0, 1.0 / z_load
    for k in range(n_steps - 2, -1, -1):
        v[k] = cosh * v[k + 1] + z0 * sinh * i[k + 1]
        i[k] = sinh / z0 * v[k + 1] + cosh * i[k + 1]
    v /= v[0] + i[0] * z_gen
    light = np.exp(1j * 2.0 * np.pi * freq_hz * OPTICAL_N_GROUP / C0 * z)
    samples = v * light
    averaged = np.sum(samples[1:] + samples[:-1]) * step / 2.0 / LENGTH_M
    return averaged / (z_load / (z_load + z_gen))


class TestMetalOnlyLine:
    def test_rlgc_from_measured_impedance_reproduces_the_inputs(self):
        rlgc = _metal_only_rlgc()
        assert rlgc["R"][0] == pytest.approx(R_OHM_M, rel=1e-12)
        assert rlgc["L"][0] == pytest.approx(L_H_M, rel=1e-12)

    def test_implied_capacitance_matches_the_measured_value(self):
        # Hand calculation: C = Im(gamma/Z0)/omega = 0.124 nF/m; the paper
        # quotes 0.12 nF/m.
        rlgc = _metal_only_rlgc()
        assert rlgc["C"][0] == pytest.approx(METAL["c_nF_per_m"] * 1e-9, rel=0.05)
        assert rlgc["C"][0] == pytest.approx(0.1238e-9, rel=2e-3)

    def test_implied_loss_and_index_match_the_measurement(self):
        omega = 2.0 * np.pi * F0_HZ
        gamma = (R_OHM_M + 1j * omega * L_H_M) / Z0_METAL
        alpha_db_cm = gamma.real * NP_PER_M_TO_DB_PER_CM
        n_rf = gamma.imag * C0 / omega
        # Hand calculation: 3.60 dB/cm, n = 2.26; measured 3.8 dB/cm.
        assert alpha_db_cm == pytest.approx(3.60, abs=0.02)
        assert alpha_db_cm == pytest.approx(METAL["alpha_dB_per_cm"], abs=0.3)
        assert n_rf == pytest.approx(2.262, abs=0.005)


class TestLoadedLine:
    def test_matches_the_measured_loaded_line_at_10_ghz(self):
        line = _loaded_line(np.array([F0_HZ]), _metal_only_rlgc())
        z0 = line.z0_ohm[0]
        alpha_db_cm = line.alpha_rf_np_m[0] * NP_PER_M_TO_DB_PER_CM
        # Measured: Z = 37 - 1j ohm, 10.8 dB/cm, n_RF = 3.75.
        assert abs(z0 - complex(LOADED["z0_real_ohm"], LOADED["z0_imag_ohm"])) < 2.0
        assert line.n_rf[0] == pytest.approx(LOADED["n_rf"], abs=0.1)
        assert alpha_db_cm == pytest.approx(LOADED["alpha_dB_per_cm"], abs=1.0)

    def test_loading_algebra_against_a_direct_telegrapher_evaluation(self):
        # Hand evaluation of gamma = sqrt(ZY) and Z0 = sqrt(Z/Y), written out
        # without gsim's helpers.
        rlgc = _metal_only_rlgc()
        omega = 2.0 * np.pi * F0_HZ
        r_s, c_j = JUNCTION
        z_ser = R_OHM_M + 1j * omega * L_H_M
        y_branch = 1.0 / (r_s + 1.0 / (1j * omega * c_j))
        y_sh = rlgc["G"][0] + 1j * omega * rlgc["C"][0] + y_branch
        line = _loaded_line(np.array([F0_HZ]), rlgc)
        assert line.gamma_per_m[0] == pytest.approx(np.sqrt(z_ser * y_sh), rel=1e-12)
        assert line.z0_ohm[0] == pytest.approx(np.sqrt(z_ser / y_sh), rel=1e-12)
        # Stored values of this closed form (not paper numbers).
        assert line.z0_ohm[0].real == pytest.approx(36.85, abs=0.05)
        assert line.n_rf[0] == pytest.approx(3.725, abs=0.005)
        assert line.alpha_rf_np_m[0] * NP_PER_M_TO_DB_PER_CM == pytest.approx(
            10.71, abs=0.05
        )

    def test_junction_rc_corner_is_far_above_the_bandwidth_of_interest(self):
        r_s, c_j = JUNCTION
        corner_ghz = 1.0 / (2.0 * np.pi * r_s * c_j) / 1e9
        assert corner_ghz == pytest.approx(44.2, abs=0.2)


class TestEoResponse:
    def test_matched_line_against_the_analytic_averaging_integral(self):
        # A real line terminated in its own impedance has no reflections; the
        # normalised response is (exp(u) - 1)/u with u = (j beta_o - gamma) L.
        freq = np.linspace(0.5e9, 40e9, 80)
        n_rf, alpha, z0 = 3.75, 40.0, 37.0
        omega = 2.0 * np.pi * freq
        gamma = alpha + 1j * omega * n_rf / C0
        u = (1j * omega * OPTICAL_N_GROUP / C0 - gamma) * LENGTH_M
        analytic = np.expm1(u) / u
        response = eo_response(
            freq,
            length_m=LENGTH_M,
            n_rf=n_rf,
            n_opt=OPTICAL_N_GROUP,
            alpha_rf_np_m=alpha,
            z0_ohm=z0,
            z_load_ohm=z0,
            z_gen_ohm=z0,
        )
        assert response == pytest.approx(analytic, abs=1e-12)

    def test_closed_form_bandwidth_of_the_loaded_5_mm_line(self):
        # Regression values of the closed-form model (50 ohm source assumed,
        # R scaled as sqrt(f)); these are this model's output, NOT agreement
        # with the measured ~10 GHz of Fig. 3, which this model overshoots
        # because it carries no probe, cable or photodiode roll-off.
        f_paper_term = _eo_f3db_ghz(z_load_ohm=TERMINATION_OHM)
        assert f_paper_term == pytest.approx(18.4, abs=0.4)

    def test_bandwidth_is_dominated_by_the_termination(self):
        matched = _eo_f3db_ghz(z_load_ohm=37.0)
        fifty = _eo_f3db_ghz(z_load_ohm=50.0)
        paper = _eo_f3db_ghz(z_load_ohm=TERMINATION_OHM)
        assert fifty < matched < paper
        assert matched == pytest.approx(11.6, abs=0.3)
        assert fifty == pytest.approx(5.6, abs=0.3)

    @pytest.mark.parametrize(
        ("z_load", "z_gen"), [(25.0, 50.0), (25.0, 100.0), (37.0, 50.0), (60.0, 20.0)]
    )
    def test_reflections_against_a_brute_force_integration(self, z_load, z_gen):
        # Exact line sections stepped back from the load, the source condition
        # applied at the input, the voltage integrated along the light: no
        # code shared with eo_response.
        freq = np.array([1e9, 5e9, 10e9, 20e9, 35e9])
        alpha = 124.3 * (freq / 10e9)  # Np/m, the measured 10.8 dB/cm at 10 GHz
        n_rf, z0 = LOADED["n_rf"], complex(LOADED["z0_real_ohm"], LOADED["z0_imag_ohm"])
        response = eo_response(
            freq,
            length_m=LENGTH_M,
            n_rf=n_rf,
            n_opt=OPTICAL_N_GROUP,
            alpha_rf_np_m=alpha,
            z0_ohm=z0,
            z_load_ohm=z_load,
            z_gen_ohm=z_gen,
        )
        for k, f in enumerate(freq):
            gamma = alpha[k] + 1j * 2.0 * np.pi * f * n_rf / C0
            brute = _brute_force_response(f, gamma, z0, z_load, z_gen)
            assert response[k] == pytest.approx(brute, rel=1e-6)

    def test_the_papers_own_loss_costs_a_matched_line_2_6_db_at_10_ghz(self):
        # 10.8 dB/cm over 5 mm is 5.4 dB one way; averaged over the light that
        # is (exp(u) - 1)/u = 0.744 = -2.57 dB, and it does not depend on how
        # the loss scales with frequency.
        freq = np.array([10e9])
        alpha = 10.8 / NP_PER_M_TO_DB_PER_CM
        response = eo_response(
            freq,
            length_m=LENGTH_M,
            n_rf=LOADED["n_rf"],
            n_opt=OPTICAL_N_GROUP,
            alpha_rf_np_m=alpha,
            z0_ohm=37.0,
            z_load_ohm=37.0,
            z_gen_ohm=37.0,
        )
        assert 20.0 * np.log10(abs(response[0])) == pytest.approx(-2.57, abs=0.03)

    def test_what_the_response_is_referenced_to_moves_the_bandwidth(self):
        # The paper's own loaded-line numbers, loss ~ f, a 25 ohm per-arm load:
        # referenced to the source EMF of a 50 ohm source the line impedance
        # (37 ohm) replaces the load in the divider and the response rises
        # +2.1 dB before the loss takes over; referenced to the voltage at
        # the line input (ideal source) there is no such rise.
        freq = np.linspace(0.1e9, 60e9, 600)
        kwargs = {
            "length_m": LENGTH_M,
            "n_rf": LOADED["n_rf"],
            "n_opt": OPTICAL_N_GROUP,
            "alpha_rf_np_m": 124.3 * freq / 10e9,
            "z0_ohm": complex(LOADED["z0_real_ohm"], LOADED["z0_imag_ohm"]),
            "z_load_ohm": TERMINATION_OHM,
        }
        source_ref = eo_response(freq, z_gen_ohm=SOURCE_OHM, **kwargs)
        input_ref = eo_response(freq, z_gen_ohm=1e-9, **kwargs)
        assert 20.0 * np.log10(np.abs(source_ref).max()) > 1.0
        assert 20.0 * np.log10(np.abs(input_ref).max()) < 0.2
        bw_source = eo_bandwidth(freq, source_ref)
        bw_input = eo_bandwidth(freq, input_ref)
        assert bw_source is not None
        assert bw_input is not None
        f_source = bw_source / 1e9
        f_input = bw_input / 1e9
        assert f_input == pytest.approx(10.6, abs=0.3)
        assert f_source == pytest.approx(21.3, abs=0.5)
        assert f_input < REFERENCE["electro_optic"]["f_3dB_GHz"] * 1.1


class TestStriations:
    def test_averaged_branch_equals_the_periodic_line(self):
        # A segment of 90% of the length carrying (R_s, C_j) averages to the
        # branch (R_s / fill, fill * C_j): same R_s C_j, admittance per length
        # times fill. The periodic-line (Bloch) model of the same electrode
        # must agree, far below the Bragg condition.
        fill, period_m = 1.0 - 1.0 / 10.0, 10e-6
        segment = JunctionBranch(r_s_ohm_m=1.44e-2, c_j_f_per_m=0.25e-9)
        averaged = JunctionBranch(
            r_s_ohm_m=segment.r_s_ohm_m / fill, c_j_f_per_m=segment.c_j_f_per_m * fill
        )
        freq = np.array([2e9, 10e9, 40e9])
        omega = 2.0 * np.pi * freq
        base = _metal_only_rlgc()
        rlgc = {
            "R": R_OHM_M * np.sqrt(freq / F0_HZ),
            "L": np.full_like(freq, base["L"][0]),
            "G": np.full_like(freq, base["G"][0]),
            "C": np.full_like(freq, base["C"][0]),
        }
        gamma_avg, z0_avg = loaded_line_params(freq, rlgc=rlgc, junction=averaged)
        gamma_seg, z0_seg = loaded_line_params(freq, rlgc=rlgc, junction=segment)
        gamma_open, z0_open = loaded_line_params(
            freq, rlgc=rlgc, junction=JunctionBranch(0.0, 0.0)
        )
        gamma_per, z0_per = segmented_line_params(
            gamma_loaded_per_m=gamma_seg,
            z0_loaded_ohm=z0_seg,
            gamma_unloaded_per_m=gamma_open,
            z0_unloaded_ohm=z0_open,
            fill_factor=fill,
            period_m=period_m,
        )
        assert gamma_per == pytest.approx(gamma_avg, rel=2e-3)
        assert z0_per == pytest.approx(z0_avg, rel=2e-3)
        # and the admittance statement itself, written out
        y_seg = (
            1j
            * omega
            * segment.c_j_f_per_m
            / (1.0 + 1j * omega * segment.r_s_ohm_m * segment.c_j_f_per_m)
        )
        y_avg = (
            1j
            * omega
            * averaged.c_j_f_per_m
            / (1.0 + 1j * omega * averaged.r_s_ohm_m * averaged.c_j_f_per_m)
        )
        assert y_avg == pytest.approx(fill * y_seg, rel=1e-12)
