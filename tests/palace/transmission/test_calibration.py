"""Validate calibrated held-out DUTs, reflect assumptions and reference planes."""

import numpy as np
import pytest

from gsim.palace.transmission import calibrate_multiline_trl, calibrate_trl

from .conftest import line_matrix, lumped_matrix, network_from_matrix


def _calibrate(analytical_lines, method, planes=(50e-6, 50e-6)):
    arguments = {
        "thru_length_m": 100e-6,
        "maximum_phase_index": 3,
        "reference_planes_m": planes,
        "reflect_offset_m": 50e-6,
    }
    thru = analytical_lines.line(100e-6)
    reflect = analytical_lines.isolated_reflect()
    if method == "trl":
        return calibrate_trl(
            thru,
            reflect,
            analytical_lines.line(300e-6),
            line_length_m=300e-6,
            **arguments,
        )
    return calibrate_multiline_trl(
        thru,
        reflect,
        [analytical_lines.line(300e-6), analytical_lines.line(1100e-6)],
        line_lengths_m=[300e-6, 1100e-6],
        **arguments,
    )


@pytest.mark.parametrize("method", ["trl", "multiline"])
@pytest.mark.parametrize("planes", [(50e-6, 50e-6), (0.0, 0.0), (20e-6, 70e-6)])
def test_held_out_mismatched_dut_and_explicit_planes(analytical_lines, method, planes):
    calibration = _calibrate(analytical_lines, method, planes)
    gamma = analytical_lines.gamma_per_m
    f_ghz = analytical_lines.frequency_hz / 1e9
    core = (
        line_matrix(gamma, 90e-6, 1.35)
        @ lumped_matrix(np.zeros(len(gamma)), 0.08 + 1j * f_ghz / 200)
        @ line_matrix(gamma, 35e-6, 0.75)
    )
    physical = line_matrix(gamma, 50e-6) @ core @ line_matrix(gamma, 50e-6)
    raw = network_from_matrix(
        analytical_lines.frequency_hz,
        analytical_lines.left @ physical @ analytical_lines.right,
    )
    original_s = raw.s.copy()
    corrected = calibration.apply_cal(raw)
    expected = network_from_matrix(
        analytical_lines.frequency_hz,
        line_matrix(gamma, 50e-6 - planes[0])
        @ core
        @ line_matrix(gamma, 50e-6 - planes[1]),
    )
    np.testing.assert_allclose(corrected.s, expected.s, atol=1e-10)
    np.testing.assert_allclose(calibration.gamma_per_m, gamma, rtol=1e-10)
    assert calibration.reference_planes_m == planes
    assert calibration.reflect_offset_m == 50e-6
    assert calibration.wave_basis == "normalized_line"
    np.testing.assert_array_equal(corrected.z0, 1)
    np.testing.assert_array_equal(raw.s, original_s)


@pytest.mark.parametrize("method", ["trl", "multiline"])
def test_held_out_uniform_line_rejects_wrong_physical_planes(analytical_lines, method):
    calibration = _calibrate(analytical_lines, method)
    corrected = calibration.apply_cal(analytical_lines.line(200e-6))
    expected = network_from_matrix(
        analytical_lines.frequency_hz,
        line_matrix(analytical_lines.gamma_per_m, 100e-6),
    )
    wrong_physical_length = network_from_matrix(
        analytical_lines.frequency_hz,
        line_matrix(analytical_lines.gamma_per_m, 200e-6),
    )
    np.testing.assert_allclose(corrected.s, expected.s, atol=1e-10)
    assert np.max(np.abs(corrected.s - wrong_physical_length.s)) > 0.1


def test_no_physical_impedance_is_inferred(analytical_lines):
    # The external port label is arbitrary to a normalized line-wave solution.
    thru = analytical_lines.line(100e-6)
    reflect = analytical_lines.isolated_reflect()
    line = analytical_lines.line(300e-6)
    raw = analytical_lines.line(200e-6)
    for network in (thru, reflect, line, raw):
        network.z0 = 50
    calibration = calibrate_trl(
        thru,
        reflect,
        line,
        thru_length_m=100e-6,
        line_length_m=300e-6,
        maximum_phase_index=3,
        reference_planes_m=(50e-6, 50e-6),
        reflect_offset_m=50e-6,
    )
    corrected = calibration.apply_cal(raw)
    np.testing.assert_array_equal(corrected.z0, 1)
    np.testing.assert_array_equal(raw.z0, 50)
    expected = network_from_matrix(
        analytical_lines.frequency_hz,
        line_matrix(analytical_lines.gamma_per_m, 100e-6),
    )
    np.testing.assert_allclose(corrected.s, expected.s, atol=1e-10)


@pytest.mark.parametrize("method", ["trl", "multiline"])
def test_apply_requires_same_input_basis_and_frequencies(analytical_lines, method):
    calibration = _calibrate(analytical_lines, method)
    raw = analytical_lines.line(200e-6)
    raw.z0 = 50
    with pytest.raises(ValueError, match="wave normalization"):
        calibration.apply_cal(raw)
    raw = analytical_lines.line(200e-6)[1:]
    with pytest.raises(ValueError, match="frequency grid"):
        calibration.apply_cal(raw)


@pytest.mark.parametrize("fault", ["transmission", "nonfinite", "grid", "z0"])
@pytest.mark.parametrize("method", ["trl", "multiline"])
def test_invalid_reflect(analytical_lines, method, fault, monkeypatch):
    reflect = analytical_lines.isolated_reflect()
    if fault == "transmission":
        reflect.s[:, 1, 0] = 0.01
    elif fault == "nonfinite":
        reflect.s[0, 0, 0] = np.nan
    elif fault == "grid":
        reflect = reflect[1:]
    else:
        reflect.z0 = 50
    monkeypatch.setattr(analytical_lines, "isolated_reflect", lambda: reflect)
    with pytest.raises(ValueError):
        _calibrate(analytical_lines, method)


def test_trl_nearly_pi_is_poorly_conditioned(analytical_lines):
    delta = np.pi * 0.999 / analytical_lines.gamma_per_m[-1].imag
    with pytest.raises(ValueError, match="phase singularity"):
        calibrate_trl(
            analytical_lines.line(100e-6),
            analytical_lines.isolated_reflect(),
            analytical_lines.line(100e-6 + delta),
            thru_length_m=100e-6,
            line_length_m=100e-6 + delta,
            maximum_phase_index=2.398339664,
            reference_planes_m=(50e-6, 50e-6),
            reflect_offset_m=50e-6,
        )


def test_multiline_requires_unambiguous_anchor(analytical_lines):
    with pytest.raises(ValueError, match="phase < pi"):
        calibrate_multiline_trl(
            analytical_lines.line(100e-6),
            analytical_lines.isolated_reflect(),
            [analytical_lines.line(1100e-6), analytical_lines.line(2100e-6)],
            thru_length_m=100e-6,
            line_lengths_m=[1100e-6, 2100e-6],
            maximum_phase_index=3,
            reference_planes_m=(50e-6, 50e-6),
            reflect_offset_m=50e-6,
        )


def test_multiline_requires_well_conditioned_pair(analytical_lines):
    with pytest.raises(ValueError, match="well-conditioned line pair"):
        calibrate_multiline_trl(
            analytical_lines.line(100e-6),
            analytical_lines.isolated_reflect(),
            [analytical_lines.line(101e-6), analytical_lines.line(102e-6)],
            thru_length_m=100e-6,
            line_lengths_m=[101e-6, 102e-6],
            maximum_phase_index=3,
            reference_planes_m=(50e-6, 50e-6),
            reflect_offset_m=50e-6,
        )


@pytest.mark.parametrize("planes", [(0.0,), (0.0, np.nan), (0.0, 0.0, 0.0)])
def test_bad_reference_planes(analytical_lines, planes):
    with pytest.raises(ValueError, match="reference_planes_m"):
        _calibrate(analytical_lines, "trl", planes)


def test_nist_degenerate_perfect_matches_fail_clearly(analytical_lines):
    standards = []
    for length in (100e-6, 300e-6, 1100e-6):
        network = analytical_lines.line(length)
        network.s[:] = 0
        network.s[:, 0, 1] = network.s[:, 1, 0] = np.exp(
            -analytical_lines.gamma_per_m * length
        )
        standards.append(network)
    reflect = standards[0].copy()
    reflect.s[:] = 0
    reflect.s[:, 0, 0] = reflect.s[:, 1, 1] = -np.exp(
        -analytical_lines.gamma_per_m * 100e-6
    )
    geometry = {
        "thru_length_m": 100e-6,
        "maximum_phase_index": 3,
        "reference_planes_m": (50e-6, 50e-6),
        "reflect_offset_m": 50e-6,
    }
    with pytest.raises(ValueError, match="non-finite calibration coefficients"):
        calibrate_multiline_trl(
            standards[0],
            reflect,
            standards[1:],
            line_lengths_m=[300e-6, 1100e-6],
            **geometry,
        )
    # The single-line solver supports this exact-match special case.
    calibration = calibrate_trl(
        standards[0],
        reflect,
        standards[1],
        line_length_m=300e-6,
        **geometry,
    )
    corrected = calibration.apply_cal(standards[1])
    expected = np.zeros_like(corrected.s)
    expected[:, 0, 1] = expected[:, 1, 0] = np.exp(
        -analytical_lines.gamma_per_m * 200e-6
    )
    np.testing.assert_allclose(corrected.s, expected, atol=1e-7)
