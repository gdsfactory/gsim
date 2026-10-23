"""Tests for mixed-mode and time-domain network analysis."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

rf = pytest.importorskip("skrf")

from gsim.palace.signal_integrity import (  # noqa: E402
    check_network,
    complete_network,
    differential_delay,
    gap_arrival,
    interpolate_complex,
    line_network,
    mixed_mode,
    tdr,
)

PORTS = ("p1", "p2", "p3", "p4")
PAIRS = (("p1", "p2"), ("p3", "p4"))
Z_SE = 50.0
F = np.arange(801) * 0.25e9
M = np.array(
    [[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float
) / np.sqrt(2)


def _network(s: np.ndarray, f: np.ndarray | None = None) -> rf.Network:
    f = F if f is None else f
    return rf.Network(f=f, s=s, z0=Z_SE, f_unit="Hz")


def test_mixed_mode_matches_matrix_and_inverse_for_reordered_ports() -> None:
    rng = np.random.default_rng(274)
    s = (rng.normal(size=(7, 4, 4)) + 1j * rng.normal(size=(7, 4, 4))) * 0.05
    trial = _network(s, np.linspace(1e9, 7e9, 7))
    trial.port_names = list(PORTS)

    for perm in ([0, 1, 2, 3], [2, 0, 3, 1], [1, 0, 3, 2]):
        shuffled = _network(s[:, perm][:, :, perm], trial.f)
        shuffled.port_names = [PORTS[i] for i in perm]

        ordered, mixed = mixed_mode(shuffled, pairs=PAIRS)
        np.testing.assert_allclose(mixed.s, M @ s @ M.T, atol=1e-12, rtol=0)

        back = mixed.copy()
        back.gmm2se(p=2)
        np.testing.assert_allclose(back.s, ordered.s, atol=1e-12, rtol=0)
        restore = [ordered.port_names.index(name) for name in shuffled.port_names]
        np.testing.assert_allclose(
            back.s[:, restore][:, :, restore], shuffled.s, atol=1e-12, rtol=0
        )


def test_mixed_mode_rejects_bad_pairs_and_reference() -> None:
    net = _network(np.zeros((3, 4, 4), complex), np.arange(3) * 1e9)
    net.port_names = list(PORTS)

    with pytest.raises(ValueError, match="cover each port"):
        mixed_mode(net, pairs=(("p1", "p2"), ("p2", "p3")))
    with pytest.raises(ValueError, match="Renormalize"):
        mixed_mode(net, z_se=75.0)


def test_check_network_rejects_incomplete_or_nonuniform_data() -> None:
    net = _network(np.zeros((3, 4, 4), complex), np.arange(3) * 1e9)
    net.port_names = list(PORTS)
    check_network(net, n_ports=4)

    net.s[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        check_network(net)
    net.s[0, 0, 0] = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = _network(np.zeros((3, 4, 4), complex), [0, 2e9, 1e9])
    with pytest.raises(ValueError, match="strictly increasing"):
        check_network(bad)


class _Parameters:
    """Minimal SParams facade for complete_network tests."""

    def __init__(
        self, net: rf.Network, keys: list[tuple[str, str]] | None = None
    ) -> None:
        self._net = net
        self._keys = keys or [(a, b) for a in PORTS for b in PORTS]

    def keys(self) -> list[tuple[str, str]]:
        return self._keys

    def to_skrf(self) -> rf.Network:
        return self._net


def test_complete_network_rejects_missing_or_unexpected_entries() -> None:
    net = _network(np.zeros((3, 4, 4), complex), np.arange(3) * 1e9)
    net.port_names = list(PORTS)
    full = [(a, b) for a in PORTS for b in PORTS]
    with pytest.raises(ValueError, match="Incomplete"):
        complete_network(_Parameters(net, full[:-1]), ports=PORTS)
    with pytest.raises(ValueError, match="Incomplete"):
        complete_network(_Parameters(net, [*full, ("p1", "p9")]), ports=PORTS)
    wrong_reference = _network(np.zeros((3, 4, 4), complex), np.arange(3) * 1e9)
    wrong_reference.port_names = list(PORTS)
    wrong_reference.z0 = 75.0
    with pytest.raises(ValueError, match="reference"):
        complete_network(_Parameters(wrong_reference), ports=PORTS)


def test_complete_network_preserves_order_and_reference() -> None:
    net = _network(np.eye(4)[None, :, :].repeat(3, axis=0) * 0.1, np.arange(3) * 1e9)
    net.port_names = list(PORTS)

    ordered = complete_network(_Parameters(net), ports=PORTS)
    assert ordered.port_names == list(PORTS)
    np.testing.assert_allclose(ordered.z0, Z_SE)


def test_line_network_tdr_recovers_impedance_delay_and_dc() -> None:
    zc, delay, zref = 130.0, 80e-12, 100.0
    uniform = line_network(F, zc, delay, zref)
    resolution = 2 / F[-1]

    t, _, impedance = tdr(F, uniform.s[:, 0, 0], zref)
    plateau = (t > 4 * resolution) & (t < 2 * delay - 4 * resolution)
    assert plateau.any()
    assert abs(np.median(impedance[plateau]) / zc - 1) < 0.02

    t_impulse, impulse = uniform.s21.impulse_response(window="hamming", bandpass=False)
    delay_estimate = t_impulse[np.argmax(np.abs(impulse))]
    assert abs(delay_estimate - delay) < resolution

    # Withhold DC and recover it by linear extrapolation.
    t_ext, _, impedance_ext = tdr(F[1:], uniform.s[1:, 0, 0], zref)
    plateau_ext = (t_ext > 4 * resolution) & (t_ext < 2 * delay - 4 * resolution)
    assert abs(np.median(impedance_ext[plateau_ext]) / zc - 1) < 0.02


def test_tdr_rejects_nonuniform_grid_and_bad_dc_policy() -> None:
    with pytest.raises(ValueError, match="uniform"):
        tdr(np.array([0.0, 1e9, 3e9]), np.zeros(3), 50.0)
    with pytest.raises(ValueError, match="linear/constant"):
        tdr(F[1:], np.zeros(len(F) - 1, complex), 50.0, dc="quadratic")


def test_differential_delay_recovers_known_phase_slope() -> None:
    delay = 37e-12
    s = np.zeros((len(F), 4, 4), complex)
    s[:, 1, 0] = np.exp(-2j * np.pi * F * delay)
    mixed = _network(s)
    mixed.port_names = ["d_near", "d_far", "c_near", "c_far"]
    mixed.z0 = np.array([[100.0, 100.0, 25.0, 25.0]] * len(F))
    assert differential_delay(mixed) == pytest.approx(delay, rel=1e-12)


def _probe(arrival_s: float, amplitude: float = 0.1) -> rf.Network:
    reflection = amplitude * np.exp(-2j * np.pi * F * arrival_s)
    return rf.Network(f=F, s=reflection[:, None, None], z0=100, f_unit="Hz")


def test_gap_arrival_uses_a_fixed_gate() -> None:
    zero = _probe(0, amplitude=0)
    first, moved = _probe(32e-12), _probe(48e-12)

    first_time = gap_arrival(zero, first, 40e-12, 10e-12, 1e-4)
    moved_time = gap_arrival(zero, moved, 40e-12, 10e-12, 1e-4)
    assert abs(first_time - 32e-12) < 10e-12
    assert abs((moved_time - first_time) - 16e-12) < 10e-12
    assert np.isnan(gap_arrival(zero, zero, 40e-12, 10e-12, 1e-4))


def test_interpolate_complex_preserves_shape_and_values() -> None:
    f = np.array([1e9, 2e9, 3e9])
    s = np.zeros((3, 2, 2), complex)
    s[:, 0, 1] = f / 1e9
    s[:, 1, 0] = 1j * f / 1e9
    net = rf.Network(f=f, s=s, z0=50, f_unit="Hz")

    out = interpolate_complex(net, [1.5e9, 2.5e9])
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out[:, 0, 1], [1.5, 2.5])
    np.testing.assert_allclose(out[:, 1, 0], [1.5j, 2.5j])


def test_network_validation_rejects_wrong_count_and_duplicate_names() -> None:
    net = _network(np.zeros((3, 4, 4), complex), np.arange(3) * 1e9)
    net.port_names = list(PORTS)

    with pytest.raises(ValueError, match="Expected a 2-port"):
        check_network(net, n_ports=2)

    net.port_names = ["p1", "p1", "p2", "p3"]
    with pytest.raises(ValueError, match="Unique port names"):
        check_network(net)


def test_mixed_mode_defaults_to_network_order_and_rejects_bad_pair_shape() -> None:
    net = _network(np.zeros((3, 4, 4), complex), np.arange(3) * 1e9)
    net.port_names = ["a", "b", "c", "d"]

    ordered, _ = mixed_mode(net)
    assert ordered.port_names == ["a", "b", "c", "d"]

    with pytest.raises(ValueError, match="ordered"):
        mixed_mode(net, pairs=(("a", "b", "c"), ("d",)))


def test_tdr_rejects_bad_inputs() -> None:
    f = F[1:]
    good = np.zeros(len(f), complex)

    with pytest.raises(ValueError, match="scalar reflection"):
        tdr(f, good[:-1], 50.0)
    with pytest.raises(ValueError, match="Nonfinite"):
        tdr(f, np.r_[np.nan, good[1:]], 50.0)
    with pytest.raises(ValueError, match="positive reference"):
        tdr(f, good, 0.0)
    with pytest.raises(ValueError, match="Start at DC"):
        tdr(np.array([2e9, 3e9, 4e9]), np.zeros(3, complex), 50.0)
    with pytest.raises(ValueError, match="DC reflection must be real"):
        tdr(np.r_[0.0, f], np.r_[1 + 1j, good], 50.0)


def test_differential_delay_rejects_unusable_transmission() -> None:
    empty = _network(np.zeros((len(F), 4, 4), complex))
    empty.port_names = ["d_near", "d_far", "c_near", "c_far"]
    with pytest.raises(ValueError, match="Insufficient"):
        differential_delay(empty)

    delayed = _network(np.zeros((len(F), 4, 4), complex))
    delayed.s[:, 1, 0] = np.exp(2j * np.pi * F * 10e-12)
    delayed.port_names = ["d_near", "d_far", "c_near", "c_far"]
    with pytest.raises(ValueError, match="Nonpositive"):
        differential_delay(delayed)
