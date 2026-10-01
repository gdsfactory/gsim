"""TRL wrappers for simulated networks in the normalized line-wave basis."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from ._validation import (
    positive_scalar,
    require_skrf,
    transfer_matrices,
    validate_networks,
)
from .propagation import SPEED_OF_LIGHT_M_S, extract_propagation, line_conditioning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from skrf import Network
    from skrf.calibration import Calibration


@dataclass(frozen=True)
class LineCalibration:
    """Calibration with explicit reference planes and a normalized wave basis.

    ``reference_planes_m`` gives each plane's inward displacement from its line
    end. Both equal half the thru length at the thru midpoint. Corrected ``z0=1``
    labels normalized line waves; it does not mean a measured one-ohm impedance.
    The caller must establish common launches and equal intrinsic reflect
    standards. Their physical correctness cannot be inferred from fitted data.
    """

    frequency_hz: NDArray
    gamma_per_m: NDArray
    reference_planes_m: tuple[float, float]
    reflect_offset_m: float
    _calibration: Calibration = field(repr=False)
    _input_reference: Network = field(repr=False)
    _shift_m: tuple[float, float] = field(repr=False)
    wave_basis: str = field(default="normalized_line", init=False)

    def apply_cal(self, network: Network) -> Network:
        """Correct a DUT on the exact standard frequency grid and input basis."""
        validate_networks(standards=self._input_reference, dut=network)
        corrected = self._calibration.apply_cal(_normalized_copy(network))
        shift = np.asarray(self._shift_m)
        exponent = self.gamma_per_m[:, None, None] * (
            shift[None, :, None] + shift[None, None, :]
        )
        corrected.s = corrected.s * np.exp(exponent)
        corrected.z0 = np.ones_like(corrected.z0)
        if not np.isfinite(corrected.s).all():
            raise ValueError("Calibration produced non-finite DUT S parameters")
        return corrected


def _normalized_copy(network: Network) -> Network:
    """Relabel S waves for calibration without physical impedance renormalization."""
    result = network.copy()
    result.z0 = np.ones_like(result.z0)
    return result


def _reference_planes(values: tuple[float, float]) -> tuple[float, float]:
    """Validate explicit inward plane shifts; negative values move outward."""
    planes = np.asarray(values, dtype=float)
    if planes.shape != (2,) or not np.isfinite(planes).all():
        raise ValueError("reference_planes_m must contain two finite distances")
    return float(planes[0]), float(planes[1])


def _validate_reflect(reflect: Network, offset: float, estimate: complex) -> None:
    """Require an isolated reflect and explicit common termination information."""
    if not np.isfinite(offset):
        raise ValueError("reflect_offset_m must be finite")
    if not np.isfinite(estimate) or estimate == 0:
        raise ValueError("reflect_estimate must be finite and nonzero")
    if np.any(np.abs(reflect.s[:, (0, 1), (1, 0)]) > 1e-6):
        raise ValueError("Reflect ports must be isolated (|S21| and |S12| <= 1e-6)")


def _zero_switch_terms(frequency_hz: NDArray):
    """Construct zero instrument-switch terms for fixed-boundary simulation data."""
    rf = require_skrf()
    zero = rf.Network(f=frequency_hz, s=np.zeros((len(frequency_hz), 1, 1)), z0=1)
    return zero, zero.copy()


def _run_calibration(calibration: Calibration) -> None:
    """Run the solver and report numerically invalid calibration coefficients."""
    try:
        # Degenerate standards can produce many backend divide-by-zero warnings.
        # Validate the result below and return one contextual error instead.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            calibration.run()
    except np.linalg.LinAlgError as exc:
        raise ValueError("TRL standards produce a singular calibration") from exc
    if any(not np.isfinite(value).all() for value in calibration.coefs.values()):
        raise ValueError("TRL standards produced non-finite calibration coefficients")


def calibrate_trl(
    thru: Network,
    reflect: Network,
    line: Network,
    *,
    thru_length_m: float,
    line_length_m: float,
    maximum_phase_index: float,
    reference_planes_m: tuple[float, float],
    reflect_offset_m: float,
    reflect_estimate: complex = -1,
    minimum_line_separation_deg: float = 20.0,
) -> LineCalibration:
    """Calibrate simulated thru/reflect/line standards with a bounded phase branch.

    Lengths, reflect offsets and reference planes are in metres. Reflect offset
    is measured inward from each line end to the equal reflect terminations;
    ``reflect_estimate`` estimates their intrinsic coefficient (e.g. -1 for a
    short), whose root the solver refines. The thru/line must share a uniform
    reciprocal line and common launches. The two launches may differ.

    The bounded two-length extraction selects the propagation root, so no
    arbitrary unwrap is used. Zero instrument-switch terms assume simulated
    fixed-boundary S matrices. Output is normalized line waves, with no inferred
    physical impedance or passivity clipping.
    """
    rf = require_skrf()
    frequencies = validate_networks(thru=thru, reflect=reflect, line=line)
    thru_length = positive_scalar(thru_length_m, "thru_length_m", allow_zero=True)
    line_length = positive_scalar(line_length_m, "line_length_m")
    planes = _reference_planes(reference_planes_m)
    _validate_reflect(reflect, reflect_offset_m, reflect_estimate)
    delta = line_length - thru_length
    propagation = extract_propagation(
        thru,
        line,
        length_difference_m=delta,
        maximum_phase_index=maximum_phase_index,
    )
    condition = line_conditioning(
        frequencies,
        propagation.phase_index,
        delta,
        minimum_degrees=minimum_line_separation_deg,
    )
    if not condition.usable.all():
        raise ValueError("TRL line is too close to a 0/180-degree phase singularity")
    ideal_s = np.zeros_like(thru.s)
    ideal_s[:, 0, 1] = ideal_s[:, 1, 0] = np.exp(-propagation.gamma_per_m * delta)
    approximate_line = rf.Network(f=frequencies, s=ideal_s, z0=1)
    reflect_s = np.zeros_like(thru.s)
    reflect_s[:, 0, 0] = reflect_s[:, 1, 1] = reflect_estimate * np.exp(
        -2 * propagation.gamma_per_m * (reflect_offset_m - thru_length / 2)
    )
    approximate_reflect = rf.Network(f=frequencies, s=reflect_s, z0=1)
    calibration = rf.TRL(
        measured=[_normalized_copy(n) for n in (thru, reflect, line)],
        ideals=[None, approximate_reflect, approximate_line],
        switch_terms=_zero_switch_terms(frequencies),
        estimate_line=False,
    )
    _run_calibration(calibration)
    return LineCalibration(
        frequencies,
        propagation.gamma_per_m,
        planes,
        reflect_offset_m,
        calibration,
        thru.copy(),
        (planes[0] - thru_length / 2, planes[1] - thru_length / 2),
    )


def calibrate_multiline_trl(
    thru: Network,
    reflect: Network,
    lines: Sequence[Network],
    *,
    thru_length_m: float,
    line_lengths_m: Sequence[float],
    maximum_phase_index: float,
    reference_planes_m: tuple[float, float],
    reflect_offset_m: float,
    reflect_estimate: complex = -1,
    minimum_line_separation_deg: float = 20.0,
) -> LineCalibration:
    """Run NIST multiline TRL with a bounded short-line branch anchor.

    Supply at least two lines longer than the thru. The shortest line/thru
    separation must satisfy the ``maximum_phase_index`` bound below pi at every
    frequency. Longer lines may wrap; the bounded anchor fixes their root and
    checks the returned propagation. At least one line pair must be well
    conditioned at each frequency. Units and wave/reference-plane conventions
    match :func:`calibrate_trl`.

    No physical-ohm renormalization or assumptions about line capacitance are
    supplied to scikit-rf. Zero switch terms apply to simulated networks only.
    """
    rf = require_skrf()
    if len(lines) < 2 or len(lines) != len(line_lengths_m):
        raise ValueError("Provide at least two lines and one length for each line")
    frequencies = validate_networks(
        thru=thru,
        reflect=reflect,
        **{f"line_{index}": line for index, line in enumerate(lines)},
    )
    for index, line in enumerate(lines):
        transfer_matrices(line, f"line_{index}")
    thru_length = positive_scalar(thru_length_m, "thru_length_m", allow_zero=True)
    lengths = np.array(
        [positive_scalar(value, "line_lengths_m") for value in line_lengths_m]
    )
    if np.any(lengths <= thru_length) or len(np.unique(lengths)) != len(lengths):
        raise ValueError("Line lengths must be distinct and greater than thru_length_m")
    planes = _reference_planes(reference_planes_m)
    _validate_reflect(reflect, reflect_offset_m, reflect_estimate)
    shortest = int(np.argmin(lengths))
    anchor = extract_propagation(
        thru,
        lines[shortest],
        length_difference_m=lengths[shortest] - thru_length,
        maximum_phase_index=maximum_phase_index,
    )
    all_lengths = np.r_[thru_length, lengths]
    margins = [
        line_conditioning(
            frequencies,
            anchor.phase_index,
            abs(second - first),
            minimum_degrees=minimum_line_separation_deg,
        ).usable
        for first, second in combinations(all_lengths, 2)
    ]
    if not np.any(margins, axis=0).all():
        raise ValueError(
            "Multiline TRL has no well-conditioned line pair at a frequency"
        )
    # NIST's er_est imaginary part is specified at 1 GHz. Initialize it from
    # the bounded anchor; gamma_est also supplies an independent root at each f.
    effective_permittivity = -(
        (anchor.gamma_per_m[0] * SPEED_OF_LIGHT_M_S / (2 * np.pi * frequencies[0])) ** 2
    )
    initial_estimate = complex(
        effective_permittivity.real,
        effective_permittivity.imag * frequencies[0] / 1e9,
    )
    calibration = rf.NISTMultilineTRL(
        measured=[_normalized_copy(n) for n in (thru, reflect, *lines)],
        Grefls=[reflect_estimate],
        l=all_lengths.tolist(),
        er_est=initial_estimate,
        gamma_est=anchor.gamma_per_m,
        gamma_root_choice="estimate",
        refl_offset=reflect_offset_m,
        # Keep NIST at the midpoint, then shift both ports explicitly below.
        # This preserves reciprocal S12/S21 for unequal shifts in skrf 1.9.
        ref_plane=[thru_length / 2, thru_length / 2],
        z0_ref=1,
        switch_terms=_zero_switch_terms(frequencies),
    )
    _run_calibration(calibration)
    gamma = np.asarray(calibration.gamma)
    phase_error = np.abs(gamma.imag - anchor.gamma_per_m.imag) * np.ptp(all_lengths)
    beta_bound = 2 * np.pi * frequencies * maximum_phase_index / SPEED_OF_LIGHT_M_S
    if (
        not np.isfinite(gamma).all()
        or np.any(gamma.imag <= 0)
        or np.any(gamma.imag > beta_bound * (1 + 1e-9))
        or np.any(phase_error >= np.pi / 2)
    ):
        raise ValueError(
            "Multiline propagation disagrees with the bounded phase anchor"
        )
    return LineCalibration(
        frequencies,
        gamma.copy(),
        planes,
        reflect_offset_m,
        calibration,
        thru.copy(),
        (planes[0] - thru_length / 2, planes[1] - thru_length / 2),
    )
