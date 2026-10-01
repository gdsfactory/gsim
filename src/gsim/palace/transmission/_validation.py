"""Shared validation and optional scikit-rf access for line analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from skrf import Network


def require_skrf():
    """Import the optional RF dependency only when a network operation needs it."""
    try:
        import skrf
    except ImportError as exc:
        raise ImportError(
            "Transmission-line analysis requires scikit-rf; install gsim[rf]."
        ) from exc
    return skrf


def positive_scalar(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Validate a finite scalar length or index."""
    if np.ndim(value) != 0 or np.iscomplexobj(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (not allow_zero and value == 0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return float(value)


def validate_frequencies(frequency_hz: NDArray) -> NDArray:
    """Require a nonempty, positive, finite and strictly increasing grid in Hz."""
    frequencies = np.asarray(frequency_hz, dtype=float)
    if (
        frequencies.ndim != 1
        or not frequencies.size
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0)
        or np.any(np.diff(frequencies) <= 0)
    ):
        raise ValueError(
            "frequency_hz must be positive, finite and strictly increasing"
        )
    return frequencies


def validate_networks(**networks: Network) -> NDArray:
    """Check two-port data, exact frequency grids and a common wave basis."""
    rf = require_skrf()
    first = next(iter(networks.values()))
    for name, network in networks.items():
        if not isinstance(network, rf.Network) or network.nports != 2:
            raise ValueError(f"{name} must be a two-port scikit-rf Network")
        validate_frequencies(network.f)
        if not np.array_equal(network.f, first.f):
            raise ValueError(f"{name} has a different frequency grid")
        if not np.isfinite(network.s).all():
            raise ValueError(f"{name} contains non-finite S parameters")
        if (
            not np.isfinite(network.z0).all()
            or np.any(network.z0.real <= 0)
            or np.any(network.z0.imag != 0)
            or not np.array_equal(network.z0[:, 0], network.z0[:, 1])
            or not np.array_equal(network.z0, first.z0)
            or network.s_def != first.s_def
        ):
            raise ValueError(
                f"{name} requires common real positive wave normalization "
                "on both ports and all networks, with the same s_def"
            )
    return np.asarray(first.f, dtype=float).copy()


def transfer_matrices(network: Network, name: str) -> NDArray:
    """Read dimensionless wave-transfer matrices without masking singular data."""
    if np.any(network.s[:, 1, 0] == 0):
        raise ValueError(f"{name} has zero S21; its transfer matrix is singular")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        matrices = network.t
    validate_matrices(matrices, name)
    return matrices


def validate_matrices(matrices: NDArray, name: str) -> None:
    """Reject nonfinite or numerically singular transfer/eigenvector matrices."""
    if not np.isfinite(matrices).all():
        raise ValueError(f"{name} has a non-finite transfer matrix")
    if np.any(np.linalg.cond(matrices) > 1e12):
        raise ValueError(f"{name} has a singular or ill-conditioned transfer matrix")
