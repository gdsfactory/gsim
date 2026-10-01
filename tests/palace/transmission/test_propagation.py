"""Verify extraction and interpolation against analytical held-out networks."""

import numpy as np
import pytest

from gsim.palace.transmission import (
    extract_propagation,
    line_conditioning,
    predict_line,
)


def test_extract_with_unequal_launches(analytical_lines):
    short, long = analytical_lines.line(100e-6), analytical_lines.line(300e-6)
    result = extract_propagation(
        short,
        long,
        length_difference_m=200e-6,
        maximum_phase_index=3,
    )
    np.testing.assert_array_equal(result.frequency_hz, analytical_lines.frequency_hz)
    np.testing.assert_allclose(
        result.gamma_per_m, analytical_lines.gamma_per_m, rtol=1e-11
    )
    np.testing.assert_allclose(result.phase_index, 2.398339664)
    np.testing.assert_allclose(
        result.attenuation_db_per_m,
        analytical_lines.gamma_per_m.real * 20 / np.log(10),
    )


@pytest.mark.parametrize("fraction", [0, 0.25, 0.5, 0.75, 1])
def test_predict_independent_intermediate_line(analytical_lines, fraction):
    short, long = analytical_lines.line(100e-6), analytical_lines.line(300e-6)
    original_short = short.s.copy()
    predicted = predict_line(
        short,
        long,
        length_difference_m=200e-6,
        target_difference_m=fraction * 200e-6,
        maximum_phase_index=3,
    )
    expected = analytical_lines.line(100e-6 + fraction * 200e-6)
    np.testing.assert_allclose(predicted.s, expected.s, atol=1e-12)
    np.testing.assert_array_equal(short.s, original_short)
    np.testing.assert_array_equal(predicted.z0, short.z0)


def test_active_attenuation_is_not_clipped(analytical_lines):
    analytical_lines.gamma_per_m = (
        -analytical_lines.gamma_per_m.real + 1j * analytical_lines.gamma_per_m.imag
    )
    result = extract_propagation(
        analytical_lines.line(100e-6),
        analytical_lines.line(300e-6),
        length_difference_m=200e-6,
        maximum_phase_index=3,
    )
    np.testing.assert_allclose(
        result.gamma_per_m, analytical_lines.gamma_per_m, rtol=1e-11
    )
    assert np.all(result.attenuation_db_per_m < 0)


@pytest.mark.parametrize("delta", [0, -1, np.nan, np.inf])
def test_bad_length(analytical_lines, delta):
    with pytest.raises(ValueError, match="length_difference_m"):
        extract_propagation(
            analytical_lines.line(100e-6),
            analytical_lines.line(300e-6),
            length_difference_m=delta,
            maximum_phase_index=3,
        )


@pytest.mark.parametrize("bound", [0, -1, np.nan, np.inf])
def test_bad_phase_bound(analytical_lines, bound):
    with pytest.raises(ValueError, match="maximum_phase_index"):
        extract_propagation(
            analytical_lines.line(100e-6),
            analytical_lines.line(300e-6),
            length_difference_m=200e-6,
            maximum_phase_index=bound,
        )


def test_phase_bound_checks_highest_frequency(analytical_lines):
    # Safe at 35 GHz, ambiguous at 100 GHz. A check of only f[0] would miss it.
    with pytest.raises(ValueError, match="phase < pi"):
        extract_propagation(
            analytical_lines.line(100e-6),
            analytical_lines.line(700e-6),
            length_difference_m=600e-6,
            maximum_phase_index=3,
        )


def test_bound_cannot_be_smaller_than_extracted_index(analytical_lines):
    with pytest.raises(ValueError, match="exceeds maximum_phase_index"):
        extract_propagation(
            analytical_lines.line(100e-6),
            analytical_lines.line(300e-6),
            length_difference_m=200e-6,
            maximum_phase_index=2,
        )


def test_nearly_pi_phase(analytical_lines):
    exact_index = 2.398339664
    delta = 299792458 / (2 * analytical_lines.frequency_hz[-1] * exact_index)
    with pytest.raises(ValueError, match="phase < pi"):
        extract_propagation(
            analytical_lines.line(0),
            analytical_lines.line(delta),
            length_difference_m=delta,
            maximum_phase_index=exact_index,
        )
    result = extract_propagation(
        analytical_lines.line(0),
        analytical_lines.line(delta * 0.999),
        length_difference_m=delta * 0.999,
        maximum_phase_index=exact_index,
    )
    np.testing.assert_allclose(
        result.gamma_per_m, analytical_lines.gamma_per_m, rtol=1e-9
    )


@pytest.mark.parametrize("target", [-1, 201e-6, np.nan])
def test_prediction_is_interpolation(analytical_lines, target):
    with pytest.raises(ValueError, match="target_difference_m"):
        predict_line(
            analytical_lines.line(100e-6),
            analytical_lines.line(300e-6),
            length_difference_m=200e-6,
            target_difference_m=target,
            maximum_phase_index=3,
        )


def test_identical_standards_have_no_phase_branch(analytical_lines):
    line = analytical_lines.line(100e-6)
    with pytest.raises(ValueError, match="degenerate"):
        extract_propagation(
            line,
            line,
            length_difference_m=200e-6,
            maximum_phase_index=3,
        )


def test_zero_transmission_is_singular(analytical_lines):
    short, long = analytical_lines.line(100e-6), analytical_lines.line(300e-6)
    short.s[:, 1, 0] = 0
    with pytest.raises(ValueError, match="zero S21"):
        extract_propagation(
            short, long, length_difference_m=200e-6, maximum_phase_index=3
        )


def test_near_singular_transfer_is_rejected(analytical_lines):
    short, long = analytical_lines.line(100e-6), analytical_lines.line(300e-6)
    short.s[:] = 0
    short.s[:, 0, 1] = short.s[:, 1, 0] = 1e-20
    with pytest.raises(ValueError, match="ill-conditioned"):
        extract_propagation(
            short, long, length_difference_m=200e-6, maximum_phase_index=3
        )


def test_nonreciprocal_line_ratio_is_rejected(analytical_lines):
    short, long = analytical_lines.line(100e-6), analytical_lines.line(300e-6)
    long.s[:, 0, 1] *= 0.5
    with pytest.raises(ValueError, match="reciprocal pair"):
        extract_propagation(
            short, long, length_difference_m=200e-6, maximum_phase_index=3
        )


@pytest.mark.parametrize("fault", ["grid", "nan", "z0", "port_z0", "s_def", "ports"])
def test_network_validation(analytical_lines, fault):
    short, long = analytical_lines.line(100e-6), analytical_lines.line(300e-6)
    if fault == "grid":
        long.frequency.f[:] += 1
    elif fault == "nan":
        long.s[0, 0, 0] = np.nan
    elif fault == "z0":
        long.z0 = 50
    elif fault == "port_z0":
        short.z0 = long.z0 = [1, 2]
    elif fault == "s_def":
        long.s_def = "traveling"
    else:
        long = long.s11
    with pytest.raises(ValueError):
        extract_propagation(
            short, long, length_difference_m=200e-6, maximum_phase_index=3
        )


@pytest.mark.parametrize("frequencies", [[0], [-1], [np.nan], [2, 1], [1, 1], []])
def test_invalid_frequency_grid(frequencies):
    with pytest.raises(ValueError, match="frequency_hz"):
        line_conditioning(np.array(frequencies), 2, 100e-6)


def test_phase_conditioning():
    frequencies = np.array([1e9, 2e9, 3e9, 4e9])
    result = line_conditioning(frequencies, 1, 299792458 / 4e9)
    np.testing.assert_allclose(result.phase_difference_deg, [90, 180, 270, 360])
    np.testing.assert_allclose(result.distance_from_singularity_deg, [90, 0, 90, 0])
    np.testing.assert_array_equal(result.usable, [True, False, True, False])
