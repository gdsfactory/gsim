"""Propagation and interpolation for two lengths with common launch networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._validation import (
    positive_scalar,
    require_skrf,
    transfer_matrices,
    validate_frequencies,
    validate_matrices,
    validate_networks,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from skrf import Network

SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class PropagationResult:
    """Propagation samples: ``gamma_per_m = alpha + 1j * beta``.

    ``frequency_hz`` is in Hz, alpha in Np/m, and beta in rad/m. The wave
    convention is exp(-gamma * length). Negative attenuation is preserved.
    """

    frequency_hz: NDArray
    gamma_per_m: NDArray

    @property
    def attenuation_db_per_m(self) -> NDArray:
        """Return signed amplitude attenuation in dB/m."""
        return self.gamma_per_m.real * 20 / np.log(10)

    @property
    def phase_index(self) -> NDArray:
        """Return beta / k0 for the selected positive-phase branch."""
        return (
            self.gamma_per_m.imag * SPEED_OF_LIGHT_M_S / (2 * np.pi * self.frequency_hz)
        )


@dataclass(frozen=True)
class LineConditioning:
    """Line/thru phase separation and distance from a 0/180-degree singularity."""

    phase_difference_deg: NDArray
    distance_from_singularity_deg: NDArray
    usable: NDArray


def line_conditioning(
    frequency_hz: NDArray,
    phase_index: float | NDArray,
    length_difference_m: float,
    *,
    minimum_degrees: float = 20.0,
) -> LineConditioning:
    """Assess TRL phase separation without choosing or unwrapping a branch.

    A caller-supplied phase index can be scalar or one value per frequency.
    This assesses phase conditioning only, not loss or fixture consistency.
    """
    frequencies = validate_frequencies(frequency_hz)
    length = positive_scalar(length_difference_m, "length_difference_m")
    minimum = positive_scalar(minimum_degrees, "minimum_degrees")
    if minimum > 90:
        raise ValueError("minimum_degrees must be at most 90")
    index = np.broadcast_to(np.asarray(phase_index, dtype=float), frequencies.shape)
    if not np.isfinite(index).all() or np.any(index <= 0):
        raise ValueError("phase_index must be finite and positive")
    phase = 360 * frequencies * index * length / SPEED_OF_LIGHT_M_S
    remainder = np.mod(phase, 180)
    distance = np.minimum(remainder, 180 - remainder)
    return LineConditioning(phase, distance, distance >= minimum)


def _line_eigensystem(
    short: Network,
    long: Network,
    length_difference_m: float,
    maximum_phase_index: float,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Find the bounded logarithm of the launch-cancelling transfer ratio."""
    frequencies = validate_networks(short=short, long=long)
    length = positive_scalar(length_difference_m, "length_difference_m")
    index_bound = positive_scalar(maximum_phase_index, "maximum_phase_index")
    phase_bound = 2 * np.pi * frequencies * index_bound * length / SPEED_OF_LIGHT_M_S
    if np.any(phase_bound >= np.pi):
        raise ValueError(
            "The maximum_phase_index bound must give phase < pi at every "
            "frequency; use a shorter length difference to resolve the branch"
        )
    short_transfer = transfer_matrices(short, "short")
    long_transfer = transfer_matrices(long, "long")
    ratio = np.linalg.solve(
        short_transfer.swapaxes(-1, -2), long_transfer.swapaxes(-1, -2)
    ).swapaxes(-1, -2)
    validate_matrices(ratio, "line transfer ratio")
    eigenvalues, eigenvectors = np.linalg.eig(ratio)
    validate_matrices(eigenvectors, "line eigenvectors")
    if not np.allclose(np.prod(eigenvalues, axis=1), 1, rtol=1e-3, atol=1e-9):
        raise ValueError(
            "Line transfer eigenvalues are not a reciprocal pair within 0.1%; "
            "check line/launch consistency and solver accuracy"
        )
    logarithms = np.log(eigenvalues)
    phase = logarithms.imag
    if np.any(np.sum(phase > 1e-10, axis=1) != 1) or np.any(
        np.min(np.abs(phase), axis=1) < 1e-10
    ):
        raise ValueError("Line phase is degenerate or has no unique positive branch")
    selected_phase = np.max(phase, axis=1)
    if np.any(selected_phase > phase_bound * (1 + 1e-9)):
        raise ValueError("Extracted phase exceeds maximum_phase_index")
    return frequencies, logarithms, eigenvectors, short_transfer


def extract_propagation(
    short: Network,
    long: Network,
    *,
    length_difference_m: float,
    maximum_phase_index: float,
) -> PropagationResult:
    """Cancel common launches and extract positive-phase propagation in SI units.

    The caller must supply two lengths of the same uniform reciprocal line with
    unchanged launches across lengths; the two ends may differ. An independently
    justified positive phase index bound must keep the largest difference below pi;
    wrapped data cannot establish its own correct branch. No passivity clipping
    or impedance/RLGC inference is performed. A transfer eigenvalue product
    differing from unity by more than 0.1% is rejected as nonreciprocal.
    """
    frequencies, logarithms, _, _ = _line_eigensystem(
        short, long, length_difference_m, maximum_phase_index
    )
    selected = np.argmax(logarithms.imag, axis=1)
    gamma = logarithms[np.arange(len(frequencies)), selected] / length_difference_m
    return PropagationResult(frequencies, gamma)


def predict_line(
    short: Network,
    long: Network,
    *,
    length_difference_m: float,
    target_difference_m: float,
    maximum_phase_index: float,
) -> Network:
    """Predict an intermediate uniform line including the common launch networks.

    ``target_difference_m`` is the target length minus the short length, between
    zero and ``length_difference_m``. Use an independently simulated intermediate
    line to test the common-launch/uniform-line premise. Inputs and their numeric
    wave normalization are preserved.
    """
    frequencies, logarithms, eigenvectors, short_transfer = _line_eigensystem(
        short, long, length_difference_m, maximum_phase_index
    )
    target = positive_scalar(
        target_difference_m, "target_difference_m", allow_zero=True
    )
    if target > length_difference_m:
        raise ValueError("target_difference_m must lie between the two input lengths")
    factors = np.exp(logarithms * (target / length_difference_m))
    predicted = (eigenvectors * factors[:, None, :]) @ np.linalg.solve(
        eigenvectors, short_transfer
    )
    rf = require_skrf()
    return rf.Network(
        f=frequencies,
        t=predicted,
        z0=short.z0.copy(),
        s_def=short.s_def,
        name="predicted intermediate line",
    )
