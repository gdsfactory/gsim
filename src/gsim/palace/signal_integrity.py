"""Mixed-mode and time-domain analysis for multi-port S-parameter networks.

The helpers in this module operate on :class:`skrf.Network` objects.  They are
backend-neutral: any solver result that can be exported as a scikit-rf network
(Palace, Meep, or a measured Touchstone file) can use them.

The functions deliberately keep the conventions used by the signal-integrity
tutorial:

* ports are ordered by name before the mixed-mode transformation;
* mixed-mode order is ``(differential, differential, common, common)`` for a
  four-port network;
* TDR adds a real DC sample by linear or constant extrapolation before taking
  the step response;
* all returned impedances and delays are in ohms and seconds.

scikit-rf is imported lazily so importing :mod:`gsim.palace` does not require
the optional ``rf-analysis`` dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "check_network",
    "complete_network",
    "differential_delay",
    "gap_arrival",
    "interpolate_complex",
    "line_network",
    "mixed_mode",
    "tdr",
]


def _network_type() -> Any:
    """Import and return :class:`skrf.Network` with an actionable error."""
    try:
        from skrf import Network
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "scikit-rf is required for signal-integrity analysis. Install it "
            "with `pip install scikit-rf` (or the gsim `rf-analysis` extra)."
        ) from error
    return Network


def check_network(net: Any, *, n_ports: int | None = None) -> None:
    """Validate a network before converting or transforming it.

    Args:
        net: scikit-rf network.
        n_ports: Required number of ports, or ``None`` to accept any count.

    Raises:
        ValueError: When the network is incomplete, non-finite, non-monotonic in
            frequency, or has no unique port names.
    """
    if n_ports is not None and net.nports != n_ports:
        raise ValueError(f"Expected a {n_ports}-port network, got {net.nports}")
    if not np.isfinite(net.s).all():
        raise ValueError("Expected a complete, finite S-parameter matrix")
    if len(net.f) < 2 or not np.all(np.diff(net.f) > 0):
        raise ValueError("Frequency must be strictly increasing")
    if net.port_names is None or len(set(net.port_names)) != net.nports:
        raise ValueError("Unique port names are required for every port")


def mixed_mode(
    net: Any,
    pairs: Sequence[tuple[str, str]] | None = None,
    z_se: float | None = None,
) -> tuple[Any, Any]:
    """Return port-ordered single-ended and mixed-mode four-port networks.

    Args:
        net: Four-port scikit-rf network with names on every port.  Frequencies
            are expected in Hz and ``net.z0`` must be a real, equal reference.
        pairs: Two ordered ``(+, -)`` port-name pairs.  Defaults to
            ``((p0, p1), (p2, p3))`` in the network's current port order.
        z_se: Single-ended reference impedance.  Defaults to the network's
            reference impedance.

    Returns:
        ``(ordered, mixed)`` where ``ordered`` follows the pair order and
        ``mixed`` has ports ``d_near, d_far, c_near, c_far``.

    Raises:
        ValueError: When the ports, pairs, or reference impedance do not match.
    """
    Network = _network_type()
    check_network(net, n_ports=4)

    names = list(net.port_names)
    if pairs is None:
        pairs = ((names[0], names[1]), (names[2], names[3]))
    if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
        raise ValueError("Supply two ordered (+, -) pairs")

    ordered_names = [name for pair in pairs for name in pair]
    if len(set(ordered_names)) != 4 or set(ordered_names) != set(names):
        raise ValueError("Pair names must cover each port exactly once")

    if z_se is None:
        z_se = float(np.asarray(net.z0).reshape(-1)[0].real)
    if not np.allclose(net.z0, z_se, rtol=0, atol=1e-12):
        raise ValueError("Renormalize explicitly to equal, real reference impedances")

    indices = [names.index(name) for name in ordered_names]
    ordered = Network(
        f=net.f,
        s=net.s[:, indices][:, :, indices],
        z0=z_se,
        f_unit="Hz",
        name=getattr(net, "name", None),
    )
    ordered.port_names = ordered_names

    mixed = ordered.copy()
    mixed.se2gmm(p=2)
    mixed.port_names = ["d_near", "d_far", "c_near", "c_far"]
    return ordered, mixed


def complete_network(
    parameters: Any,
    *,
    ports: Sequence[str],
    z_se: float = 50.0,
) -> Any:
    """Convert a complete Palace ``SParams`` object to an ordered network.

    Missing entries are rejected *before* :meth:`SParams.to_skrf` can fill them
    by reciprocity or leave unobserved entries at zero.

    Args:
        parameters: ``gsim.palace.SParams``-like object with ``keys()`` and
            ``to_skrf()``.
        ports: Expected port names in tutorial order.
        z_se: Declared single-ended reference impedance.

    Returns:
        A four-port network ordered as ``ports``.

    Raises:
        ValueError: When entries are missing, the network is incomplete, or the
            solver reference impedance differs from ``z_se``.
    """
    expected = {(a, b) for a in ports for b in ports}
    if set(parameters.keys()) != expected:
        missing = sorted(expected - set(parameters.keys()))
        unexpected = sorted(set(parameters.keys()) - expected)
        raise ValueError(
            f"Incomplete/unexpected matrix: missing {missing}, unexpected {unexpected}"
        )

    net = parameters.to_skrf()
    check_network(net, n_ports=len(ports))
    if not np.allclose(net.z0, z_se):
        raise ValueError(
            "Solver reference differs from the declared reference impedance"
        )
    if len(ports) != 4:
        raise ValueError("Supply exactly four ordered port names")
    return mixed_mode(
        net, pairs=((ports[0], ports[1]), (ports[2], ports[3])), z_se=z_se
    )[0]


def line_network(
    f: Sequence[float] | np.ndarray,
    zc: float,
    delay: float,
    zref: float,
) -> Any:
    """Return the exact two-port network of an ideal uniform TEM line.

    Args:
        f: Frequency in Hz.
        zc: Characteristic impedance in ohms.
        delay: One-way delay in seconds.
        zref: Port reference impedance in ohms.
    """
    theta = 2j * np.pi * np.asarray(f) * delay
    a, b, c = np.cosh(theta), zc * np.sinh(theta), np.sinh(theta) / zc
    denominator = 2 * a + b / zref + c * zref
    reflection = (b / zref - c * zref) / denominator
    transmission = 2 / denominator
    s = np.empty((len(f), 2, 2), complex)
    s[:, 0, 0] = s[:, 1, 1] = reflection
    s[:, 0, 1] = s[:, 1, 0] = transmission
    Network = _network_type()
    return Network(f=f, s=s, z0=zref, f_unit="Hz")


def tdr(
    f: Sequence[float] | np.ndarray,
    reflection: Sequence[complex] | np.ndarray,
    zref: float,
    *,
    window: str = "hamming",
    dc: str = "linear",
    pad: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a reflection coefficient into a reflected step and impedance.

    Args:
        f: Uniform, strictly increasing frequency grid in Hz.
        reflection: Reflection coefficient at ``f``.
        zref: Reference impedance in ohms.
        window: scikit-rf window name (for example ``"hamming"`` or ``"hann"``).
        dc: DC extrapolation policy, ``"linear"`` or ``"constant"``.
        pad: Zero-padding passed to scikit-rf's step response.

    Returns:
        ``(t, rho, z)`` in seconds, reflected step / incident step, and ohms.

    Raises:
        ValueError: For a nonuniform grid, non-finite samples, invalid reference
            impedance, or an unsupported DC policy.
    """
    Network = _network_type()
    f = np.asarray(f, float)
    reflection = np.asarray(reflection, complex)
    if f.ndim != 1 or len(f) < 3 or reflection.shape != f.shape:
        raise ValueError("Need at least three scalar reflection samples")
    if not np.isfinite(f).all() or not np.isfinite(reflection).all():
        raise ValueError("Nonfinite TDR input")
    df = f[1] - f[0]
    if df <= 0 or not np.allclose(np.diff(f), df, rtol=1e-8, atol=1):
        raise ValueError("TDR requires a uniform increasing frequency grid")
    if zref <= 0 or dc not in {"linear", "constant"}:
        raise ValueError("Use a positive reference and linear/constant DC policy")

    if np.isclose(f[0], df, rtol=1e-8, atol=1):
        value = (
            2 * reflection[0].real - reflection[1].real
            if dc == "linear"
            else reflection[0].real
        )
        reflection = np.r_[value, reflection]
        f = np.r_[0.0, f]
    elif f[0] != 0:
        raise ValueError("Start at DC or exactly one frequency step above DC")
    if abs(reflection[0].imag) > 1e-10:
        raise ValueError("DC reflection must be real")

    oneport = Network(f=f, s=reflection[:, None, None], z0=zref, f_unit="Hz")
    t, rho = oneport.step_response(window=window, pad=pad)
    denominator = 1 - rho
    impedance = np.full(rho.shape, np.nan)
    np.divide(
        zref * (1 + rho),
        denominator,
        out=impedance,
        where=np.abs(denominator) > 1e-6,
    )
    return t, rho, impedance


def differential_delay(
    mixed: Any,
    *,
    min_fraction: float = 0.1,
    max_fraction: float = 0.6,
    min_magnitude: float = 0.05,
    min_samples: int = 5,
) -> float:
    """Estimate one-way group delay from the differential transmission phase.

    Args:
        mixed: Mixed-mode four-port network; ``S[:, 1, 0]`` is used.
        min_fraction: Lower frequency bound as a fraction of the maximum.
        max_fraction: Upper frequency bound as a fraction of the maximum.
        min_magnitude: Minimum ``|S21|`` used for the phase-slope estimate.
        min_samples: Minimum number of samples in the fit window.

    Returns:
        Median one-way delay in seconds.

    Raises:
        ValueError: When too few usable samples exist or the delay is nonpositive.
    """
    transmission = mixed.s[:, 1, 0]
    use = (
        (mixed.f >= min_fraction * mixed.f[-1])
        & (mixed.f <= max_fraction * mixed.f[-1])
        & (np.abs(transmission) > min_magnitude)
    )
    delays = -np.gradient(np.unwrap(np.angle(transmission)), 2 * np.pi * mixed.f)
    if use.sum() < min_samples:
        raise ValueError("Insufficient transmitted signal to estimate delay")
    tau = float(np.median(delays[use]))
    if tau <= 0:
        raise ValueError("Nonpositive estimated delay: inspect ports, phase, and sweep")
    return tau


def gap_arrival(
    baseline: Any,
    disrupted: Any,
    one_way_delay: float,
    resolution_s: float,
    threshold: float,
    *,
    zref: float | None = None,
) -> float:
    """Return the first fixed-gate half-peak arrival of a reflected-step change.

    The gate excludes the launches and the far-end reflection and is identical
    for both networks.  It is never centred on the expected feature time.

    Args:
        baseline: Unperturbed mixed-mode network.
        disrupted: Perturbed mixed-mode network on the same time grid.
        one_way_delay: One-way delay in seconds, used to set the far gate.
        resolution_s: Working time resolution in seconds.
        threshold: Minimum reflected-step difference to report an arrival.
        zref: Differential reference impedance.  Defaults to the first
            reference of ``baseline``.

    Returns:
        Arrival time in seconds, or ``nan`` when the feature is below threshold
        or outside the gate.
    """
    if zref is None:
        zref = float(np.asarray(baseline.z0).reshape(-1)[0].real)
    t, r_base, _ = tdr(baseline.f, baseline.s[:, 0, 0], zref)
    t_gap, r_gap, _ = tdr(disrupted.f, disrupted.s[:, 0, 0], zref)
    np.testing.assert_allclose(t, t_gap)
    difference = np.abs(r_gap - r_base)
    region = (t > 2 * resolution_s) & (t < 2 * one_way_delay - 2 * resolution_s)
    if not region.any() or difference[region].max() <= threshold:
        return np.nan
    level = max(threshold, 0.5 * difference[region].max())
    found = np.flatnonzero(region & (difference >= level))
    return float(t[found[0]]) if len(found) else np.nan


def interpolate_complex(
    net: Any, frequencies: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Interpolate every complex S entry onto ``frequencies``.

    Real and imaginary parts are interpolated separately.  This is intended for
    comparing direct anchor solves with a dense sweep; it does not create data
    outside the source band.

    Args:
        net: scikit-rf network.
        frequencies: Target frequencies in Hz.

    Returns:
        Array with shape ``(len(frequencies), n_ports, n_ports)``.
    """
    frequencies = np.asarray(frequencies, float)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Target frequencies must be finite")
    if np.any(frequencies < net.f[0]) or np.any(frequencies > net.f[-1]):
        raise ValueError("Target frequencies must lie within the source band")
    n_ports = net.s.shape[1]
    values = [
        np.interp(frequencies, net.f, net.s[:, i, j].real)
        + 1j * np.interp(frequencies, net.f, net.s[:, i, j].imag)
        for i in range(n_ports)
        for j in range(n_ports)
    ]
    return np.stack(values, axis=-1).reshape(-1, n_ports, n_ports)
