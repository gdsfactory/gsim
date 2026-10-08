"""Tests for modal to terminal S-parameter conversion on analytic coupled lines.

References are independent of the modal route: a chain-matrix solution
(``expm`` of the telegrapher matrix) converted with scikit-rf's ``z2s``, and
scikit-rf's ``se2gmm`` mixed-mode conversion. Generic synthetic values only.
"""

from __future__ import annotations

import numpy as np
import pytest
import skrf
from scipy.linalg import expm
from skrf.media import DefinedGammaZ0
from skrf.network import s2z, z2s

from gsim.palace.terminal import (
    currents_from_reaction,
    degenerate_mode_groups,
    modal_to_terminal_s,
    terminal_to_modal_s,
    uniform_line_modal_s,
)

TOL = 1e-10
C0 = 2.998e8
FREQ = np.linspace(1e9, 100e9, 9)
OMEGA = 2 * np.pi * FREQ
ELL = 700e-6


def rlgc(kind="sym", lossy=True, homogeneous=False):
    """Per-unit-length (R, L, G, C) of a coupled line over a common reference."""
    if kind == "three":
        L = np.array(
            [
                [4.0e-7, 1.2e-7, 0.4e-7],
                [1.2e-7, 4.2e-7, 1.1e-7],
                [0.4e-7, 1.1e-7, 3.8e-7],
            ]
        )
        C = np.array(
            [
                [1.6e-10, -0.5e-10, -0.1e-10],
                [-0.5e-10, 1.9e-10, -0.4e-10],
                [-0.1e-10, -0.4e-10, 1.5e-10],
            ]
        )
        R = np.array([[40.0, 5.0, 2.0], [5.0, 45.0, 4.0], [2.0, 4.0, 38.0]])
        G = np.array(
            [[2e-4, -5e-5, -2e-5], [-5e-5, 2.2e-4, -4e-5], [-2e-5, -4e-5, 1.8e-4]]
        )
    else:
        asym = 0.4 if kind == "asym" else 0.0
        L = np.array([[4.0e-7, 1.2e-7], [1.2e-7, 4.0e-7 * (1 + asym)]])
        C = np.array([[1.6e-10, -0.5e-10], [-0.5e-10, 1.6e-10 * (1 + 0.5 * asym)]])
        R = np.array([[40.0, 5.0], [5.0, 40.0]])
        G = np.array([[2e-4, -5e-5], [-5e-5, 2e-4]])
    if homogeneous:
        C = np.linalg.inv(L) * (2.1 / C0**2)  # L C = mu eps 1: degenerate modes
    if not lossy:
        R = np.zeros_like(R)
        G = np.zeros_like(G)
    return R, L, G, C


def chain_terminal_s(R, L, G, C, ell=ELL, z_ref=50.0):
    """Terminal S from the chain matrix expm([[0, -Z], [-Y, 0]] l)."""
    n = L.shape[0]
    z_mats = []
    for w in OMEGA:
        zs = R + 1j * w * L
        ys = G + 1j * w * C
        a = np.block([[np.zeros((n, n)), -zs], [-ys, np.zeros((n, n))]])
        phi = expm(a * ell)
        p11, p12, p21, p22 = phi[:n, :n], phi[:n, n:], phi[n:, :n], phi[n:, n:]
        p21i = np.linalg.inv(p21)
        z_mats.append(
            np.block([[-p21i @ p22, -p21i], [p12 - p11 @ p21i @ p22, -p11 @ p21i]])
        )
    return z2s(np.array(z_mats), z0=z_ref, s_def="power")


def pick_gamma(g2):
    g = np.sqrt(np.asarray(g2, dtype=complex))
    tol = 1e-12 * np.abs(g)
    flip = (g.real < -tol) | ((np.abs(g.real) <= tol) & (g.imag < 0))
    return np.where(flip, -g, g)


def modal_data(R, L, G, C):
    """T_V, T_I (nf, n, n) and gamma (nf, n) from the eigenproblem of Z Y."""
    tvs, tis, gs = [], [], []
    for w in OMEGA:
        zs = R + 1j * w * L
        ys = G + 1j * w * C
        g2, tv = np.linalg.eig(zs @ ys)
        g = pick_gamma(g2)
        tvs.append(tv)
        tis.append(ys @ tv @ np.diag(1 / g))
        gs.append(g)
    return np.array(tvs), np.array(tis), np.array(gs)


def two_end(tv, ti):
    return [tv, tv], [ti, ti]


CASES = {
    "symmetric-lossless": {"kind": "sym", "lossy": False},
    "symmetric-lossy": {"kind": "sym"},
    "asymmetric-lossy": {"kind": "asym"},
    "three-conductor-lossy": {"kind": "three"},
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_chain_matrix(name):
    R, L, G, C = rlgc(**CASES[name])
    tv, ti, gamma = modal_data(R, L, G, C)
    s_t = modal_to_terminal_s(uniform_line_modal_s(gamma, ELL), *two_end(tv, ti))
    s_ref = chain_terminal_s(R, L, G, C)
    assert s_t.shape == s_ref.shape
    assert np.abs(s_t - s_ref).max() < TOL


@pytest.mark.parametrize("name", list(CASES))
def test_round_trip_and_reciprocity(name):
    R, L, G, C = rlgc(**CASES[name])
    tv, ti, gamma = modal_data(R, L, G, C)
    s_m = uniform_line_modal_s(gamma, ELL)
    s_t = modal_to_terminal_s(s_m, *two_end(tv, ti), z_ref=60.0)
    back = terminal_to_modal_s(s_t, *two_end(tv, ti), z_ref=60.0)
    assert np.abs(back - s_m).max() < TOL
    assert np.abs(s_t - s_t.transpose(0, 2, 1)).max() < TOL


def test_round_trip_with_cross_mode_terms():
    rng = np.random.default_rng(3)
    R, L, G, C = rlgc("asym")
    tv, ti, _ = modal_data(R, L, G, C)
    s_m = 0.3 * (
        rng.normal(size=(len(FREQ), 4, 4)) + 1j * rng.normal(size=(len(FREQ), 4, 4))
    )
    s_t = modal_to_terminal_s(s_m, *two_end(tv, ti))
    assert np.abs(terminal_to_modal_s(s_t, *two_end(tv, ti)) - s_m).max() < TOL


def test_joint_per_mode_scaling_invariance():
    rng = np.random.default_rng(1)
    R, L, G, C = rlgc("asym")
    tv, ti, gamma = modal_data(R, L, G, C)
    s_m = uniform_line_modal_s(gamma, ELL)
    ref = modal_to_terminal_s(s_m, *two_end(tv, ti))
    c = rng.normal(size=2) + 1j * rng.normal(size=2)
    scaled = modal_to_terminal_s(s_m, *two_end(tv * c, ti * c))
    assert np.abs(scaled - ref).max() < TOL


def test_degenerate_modes_any_basis():
    rng = np.random.default_rng(5)
    R, L, G, C = rlgc("sym", lossy=False, homogeneous=True)
    s_ref = chain_terminal_s(R, L, G, C)
    gam = np.array(
        [
            pick_gamma(np.linalg.eigvals((1j * w * L) @ (1j * w * C))).mean()
            for w in OMEGA
        ]
    )
    gamma = np.repeat(gam[:, None], 2, axis=1)
    s_m = uniform_line_modal_s(gamma, ELL)
    ys = 1j * OMEGA[:, None, None] * C[None]

    def run(basis):
        tv = np.broadcast_to(basis, (len(FREQ), 2, 2))
        ti = ys @ tv / gam[:, None, None]  # common gamma: T_I = Y T_V / gamma
        return modal_to_terminal_s(s_m, *two_end(tv, ti))

    for _ in range(5):
        basis = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        assert np.abs(run(basis) - s_ref).max() < TOL
    # terminal-aligned basis: T_V R = identity
    assert np.abs(run(np.eye(2)) - s_ref).max() < TOL


def test_per_terminal_z_ref_against_skrf():
    R, L, G, C = rlgc("asym")
    tv, ti, gamma = modal_data(R, L, G, C)
    s_m = uniform_line_modal_s(gamma, ELL)
    z_ref = np.array([50.0, 35.0, 50.0, 35.0])
    s_t = modal_to_terminal_s(s_m, *two_end(tv, ti), z_ref=z_ref)

    nf = len(FREQ)
    t_mat = np.zeros((nf, 4, 4), complex)
    w_mat = np.zeros_like(t_mat)
    for p in range(2):
        t_mat[:, 2 * p : 2 * p + 2, 2 * p : 2 * p + 2] = tv
        w_mat[:, 2 * p : 2 * p + 2, 2 * p : 2 * p + 2] = ti
    z_modal = s2z(s_m, z0=1.0, s_def="traveling")  # unit modal impedance
    z_term = t_mat @ z_modal @ np.linalg.inv(w_mat)
    s_skrf = z2s(z_term, z0=z_ref, s_def="power")
    assert np.abs(s_t - s_skrf).max() < TOL

    # frequency-dependent reference impedance
    z2 = z_ref[None, :] * (1 + 0.1 * np.linspace(0, 1, nf))[:, None]
    s_t2 = modal_to_terminal_s(s_m, *two_end(tv, ti), z_ref=z2)
    assert np.abs(s_t2 - z2s(z_term, z0=z2, s_def="power")).max() < TOL


@pytest.mark.parametrize("lossy", [False, True])
def test_mixed_mode_matches_defined_gamma_lines(lossy):
    """se2gmm of the terminal S gives the odd/even lines with 2*Zodd and Zeven/2."""
    R, L, G, C = rlgc("sym", lossy=lossy)
    tv, ti, gamma = modal_data(R, L, G, C)
    s_t = modal_to_terminal_s(uniform_line_modal_s(gamma, ELL), *two_end(tv, ti))
    # eigenvector order changes with frequency: identify the odd mode (V_A/V_B < 0)
    ratio = tv[:, 0, :] / tv[:, 1, :]
    odd = np.argmin(ratio.real, axis=1)
    even = 1 - odd
    fi = np.arange(len(FREQ))
    z_odd = tv[fi, 0, odd] / ti[fi, 0, odd]
    z_even = tv[fi, 0, even] / ti[fi, 0, even]
    freq = skrf.Frequency.from_f(FREQ, unit="Hz")
    odd_line = DefinedGammaZ0(
        freq, z0_port=100.0, z0=2 * z_odd, gamma=gamma[fi, odd]
    ).line(ELL, "m")
    even_line = DefinedGammaZ0(
        freq, z0_port=25.0, z0=z_even / 2, gamma=gamma[fi, even]
    ).line(ELL, "m")

    # ports (A1, B1, A2, B2): se2gmm pairs (0, 1) and (2, 3);
    # outputs d ports, then c ports
    net = skrf.Network(f=FREQ, s=s_t, z0=50.0, f_unit="Hz")
    mm = net.copy()
    mm.se2gmm(p=2)
    assert np.abs(mm.s[:, 0:2, 0:2] - odd_line.s).max() < TOL
    assert np.abs(mm.s[:, 2:4, 2:4] - even_line.s).max() < TOL
    assert (
        np.abs(mm.s[:, 0:2, 2:4]).max() < TOL
    )  # no mode conversion on a symmetric line


@pytest.mark.parametrize("kind", ["sym", "asym", "three"])
def test_lossy_line_is_passive(kind):
    R, L, G, C = rlgc(kind)
    tv, ti, gamma = modal_data(R, L, G, C)
    s_t = modal_to_terminal_s(uniform_line_modal_s(gamma, 5e-3), *two_end(tv, ti))
    sv = np.linalg.svd(s_t, compute_uv=False)
    assert sv.max() <= 1 + 1e-12
    assert sv.max() > 0.5  # not a vacuous check


def test_currents_from_reaction_reproduces_t_i():
    R, L, G, C = rlgc("three")
    tv, ti, _ = modal_data(R, L, G, C)
    r = np.einsum("fik,fik->fk", tv, ti)  # unconjugated reaction V_k^T I_k
    rec = currents_from_reaction(tv, r)
    assert np.abs(rec - ti).max() / np.abs(ti).max() < TOL
    # single-frequency form
    rec0 = currents_from_reaction(tv[0], r[0])
    assert np.abs(rec0 - ti[0]).max() / np.abs(ti[0]).max() < TOL


def test_uniform_line_modal_s_structure():
    gamma = np.array([[1.0 + 2.0j, 0.5 + 3.0j]])
    s = uniform_line_modal_s(gamma, 0.1)
    e = np.exp(-gamma * 0.1)
    assert s.shape == (1, 4, 4)
    assert np.allclose(s[0, 0, 2], e[0, 0])
    assert np.allclose(s[0, 3, 1], e[0, 1])
    assert np.count_nonzero(s) == 4
    with pytest.raises(ValueError, match="Re\\(gamma\\)"):
        uniform_line_modal_s(np.array([[-1.0 + 2.0j]]), 0.1)


def test_degenerate_mode_groups():
    assert degenerate_mode_groups([2.0, 3.0, 2.0 + 1e-9]) == [(0, 2), (1,)]
    assert degenerate_mode_groups([2.0, 2.1, 2.2], rtol=1e-6) == [(0,), (1,), (2,)]
    assert degenerate_mode_groups([2.0, 2.1, 2.2], rtol=0.06) == [(0, 1, 2)]
    with pytest.raises(ValueError, match="1-D"):
        degenerate_mode_groups(np.ones((2, 2)))


class TestValidation:
    def setup_method(self):
        R, L, G, C = rlgc("sym")
        self.tv, self.ti, gamma = modal_data(R, L, G, C)
        self.s_m = uniform_line_modal_s(gamma, ELL)

    def call(self, **kw):
        args = {
            "s_modal": self.s_m,
            "t_v": [self.tv, self.tv],
            "t_i": [self.ti, self.ti],
        }
        args.update(kw)
        return modal_to_terminal_s(
            args.pop("s_modal"), args.pop("t_v"), args.pop("t_i"), **args
        )

    def test_baseline_runs(self):
        assert self.call().shape == (len(FREQ), 4, 4)

    def test_port_count_mismatch(self):
        with pytest.raises(ValueError, match="terminals/modes"):
            self.call(t_v=[self.tv], t_i=[self.ti])

    def test_t_v_t_i_port_count_differ(self):
        with pytest.raises(ValueError, match="ports"):
            self.call(t_i=[self.ti])

    def test_non_square_block(self):
        with pytest.raises(ValueError, match="square"):
            self.call(
                t_v=[self.tv[:, :, :1], self.tv], t_i=[self.ti[:, :, :1], self.ti]
            )

    def test_frequency_mismatch(self):
        with pytest.raises(ValueError, match="frequencies"):
            self.call(t_v=[self.tv[:3], self.tv])

    def test_modes_do_not_span_terminals(self):
        singular = np.array([[1.0, 2.0], [1.0, 2.0]])
        with pytest.raises(ValueError, match="do not span"):
            self.call(t_v=[singular, self.tv])
        with pytest.raises(ValueError, match="do not span"):
            self.call(t_i=[singular, self.ti])

    def test_bad_z_ref(self):
        with pytest.raises(ValueError, match="> 0"):
            self.call(z_ref=-50.0)
        with pytest.raises(ValueError, match="> 0"):
            self.call(z_ref=0.0)
        with pytest.raises(ValueError, match="complex"):
            self.call(z_ref=50.0 + 5.0j)
        with pytest.raises(ValueError, match="entries"):
            self.call(z_ref=[50.0, 50.0, 50.0])
        with pytest.raises(ValueError, match="shape"):
            self.call(z_ref=np.ones((3, 4)) * 50.0)

    def test_non_finite(self):
        bad = self.s_m.copy()
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            self.call(s_modal=bad)
        bad_t = self.tv.copy()
        bad_t[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            self.call(t_v=[bad_t, self.tv])

    def test_s_not_square(self):
        with pytest.raises(ValueError, match="shape"):
            self.call(s_modal=self.s_m[:, :, :3])

    def test_complex_z_ref_with_zero_imag_is_accepted(self):
        out = self.call(z_ref=np.array(50.0 + 0.0j))
        assert np.allclose(out, self.call(z_ref=50.0))
