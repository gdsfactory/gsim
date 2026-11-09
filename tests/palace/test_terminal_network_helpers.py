"""``short_terminals`` and ``mixed_mode`` on synthetic multiconductor lines.

References are independent of the helpers: scikit-rf's ``s2y`` / ``y2s`` for the
admittance sub-block, the chain matrix of the line with the shorted conductor
eliminated (a Schur complement of the series impedance), and
``DefinedGammaZ0`` lines for the mixed-mode blocks of a symmetric pair. Generic
synthetic values only.
"""

from __future__ import annotations

import numpy as np
import pytest
import skrf
from scipy.linalg import expm
from skrf.media import DefinedGammaZ0
from skrf.network import s2y, y2s, z2s

from gsim.palace import mixed_mode as exported_mixed_mode
from gsim.palace import short_terminals as exported_short_terminals
from gsim.palace.terminal import modal_to_terminal_s, uniform_line_modal_s
from gsim.palace.terminal_ports import mixed_mode, short_terminals
from tests.palace.test_terminal import (
    ELL,
    FREQ,
    OMEGA,
    chain_terminal_s,
    modal_data,
    rlgc,
    two_end,
)

TOL = 1e-10
NF = len(FREQ)
NAMES_3 = [f"{end}:{t}" for end in "AB" for t in ("G1", "S1", "S2")]
NAMES_2 = ["A1", "B1", "A2", "B2"]


def _modes(net):
    """Port modes of a network, or None on scikit-rf versions without them."""
    modes = getattr(net, "port_modes", None)
    return None if modes is None else [str(m) for m in modes]


def make_net(s, names, z0=50.0):
    """A skrf network with named ports."""
    net = skrf.Network(f=FREQ, s=s, z0=z0, f_unit="Hz")
    net.port_names = list(names)
    return net


def chain_from_zy(zs_list, ys_list, ell, z_ref=50.0):
    """Terminal S of a line with series ``Z`` and shunt ``Y`` per frequency."""
    n = zs_list[0].shape[0]
    z_mats = []
    for zs, ys in zip(zs_list, ys_list, strict=True):
        a = np.block([[np.zeros((n, n)), -zs], [-ys, np.zeros((n, n))]])
        phi = expm(a * ell)
        p11, p12, p21, p22 = phi[:n, :n], phi[:n, n:], phi[n:, :n], phi[n:, n:]
        p21i = np.linalg.inv(p21)
        z_mats.append(
            np.block([[-p21i @ p22, -p21i], [p12 - p11 @ p21i @ p22, -p11 @ p21i]])
        )
    return z2s(np.array(z_mats), z0=z_ref, s_def="power")


def three_conductor_net(ell=ELL, z_ref=50.0):
    """Ports (A:G1, A:S1, A:S2, B:G1, B:S1, B:S2) of the lossy 3-conductor line."""
    s = chain_terminal_s(*rlgc("three"), ell=ell, z_ref=z_ref)
    return make_net(s, NAMES_3, z0=z_ref)


class TestShortTerminals:
    def test_it_equals_the_admittance_sub_block(self):
        net = three_conductor_net()
        shorted = short_terminals(net, ["A:G1", "B:G1"])
        y = s2y(net.s, z0=50.0)
        keep = [1, 2, 4, 5]
        reference = y2s(y[:, keep][:, :, keep], z0=50.0, s_def="power")
        assert np.abs(shorted.s - reference).max() < TOL
        assert shorted.port_names == ["A:S1", "A:S2", "B:S1", "B:S2"]
        assert shorted.nports == 4
        np.testing.assert_allclose(shorted.f, FREQ)

    def test_one_shorted_terminal_of_a_pair_of_ends(self):
        net = three_conductor_net()
        shorted = short_terminals(net, ["B:G1"])
        y = s2y(net.s, z0=50.0)
        keep = [0, 1, 2, 4, 5]
        reference = y2s(y[:, keep][:, :, keep], z0=50.0, s_def="power")
        assert np.abs(shorted.s - reference).max() < TOL
        assert shorted.port_names == ["A:G1", "A:S1", "A:S2", "B:S1", "B:S2"]

    def test_the_order_of_the_names_does_not_matter(self):
        net = three_conductor_net()
        first = short_terminals(net, ["A:G1", "B:G1"])
        second = short_terminals(net, ["B:G1", "A:G1"])
        # same ports shorted, solved in another order: equal up to rounding
        np.testing.assert_allclose(first.s, second.s, rtol=0, atol=1e-14)
        assert first.port_names == second.port_names

    def test_reference_impedances_per_port_are_kept_and_used(self):
        z_ref = np.array([50.0, 35.0, 60.0, 50.0, 35.0, 60.0])
        net = three_conductor_net(z_ref=z_ref)
        shorted = short_terminals(net, ["A:G1", "B:G1"])
        y = s2y(net.s, z0=np.broadcast_to(z_ref, (NF, 6)))
        keep = [1, 2, 4, 5]
        reference = y2s(
            y[:, keep][:, :, keep],
            z0=np.broadcast_to(z_ref[keep], (NF, 4)),
            s_def="power",
        )
        assert np.abs(shorted.s - reference).max() < TOL
        np.testing.assert_allclose(shorted.z0[0].real, z_ref[keep])

    def test_the_input_network_is_not_changed(self):
        net = three_conductor_net()
        before = net.s.copy()
        short_terminals(net, ["A:G1"])
        np.testing.assert_array_equal(net.s, before)
        assert net.nports == 6
        assert net.port_names == NAMES_3

    def test_the_result_is_passive_and_reciprocal(self):
        shorted = short_terminals(three_conductor_net(5e-3), ["A:G1", "B:G1"])
        assert np.abs(shorted.s - shorted.s.transpose(0, 2, 1)).max() < TOL
        assert np.linalg.svd(shorted.s, compute_uv=False).max() <= 1 + 1e-9

    def test_the_tied_line_is_the_limit_of_an_electrically_short_line(self):
        """G1 tied at the two ends against G1 tied all along the line.

        Eliminating G1 along the line (V = 0 at every z) leaves a two-conductor
        line with ``Z' = Z_ss - Z_s1 Z_11^-1 Z_1s`` and ``Y' = Y_ss``. The
        difference to the line with G1 shorted at its ends only goes as the cube
        of the length.
        """
        r, l, g, c = rlgc("three")
        zs3 = [r + 1j * w * l for w in OMEGA]
        ys3 = [g + 1j * w * c for w in OMEGA]
        zs2 = [z[1:, 1:] - np.outer(z[1:, 0], z[0, 1:]) / z[0, 0] for z in zs3]
        ys2 = [y[1:, 1:] for y in ys3]
        errors = []
        for ell in (700e-6, 70e-6, 7e-6):
            net = three_conductor_net(ell)
            shorted = short_terminals(net, ["A:G1", "B:G1"])
            errors.append(np.abs(shorted.s - chain_from_zy(zs2, ys2, ell)).max())
        assert errors[0] < 5e-2
        assert errors[1] < 1e-5
        assert errors[2] < 1e-8
        assert errors[1] < errors[0] / 1000
        assert errors[2] < errors[1] / 500

    def test_it_is_exported_from_gsim_palace(self):
        assert exported_short_terminals is short_terminals

    @pytest.mark.parametrize(
        ("names", "match"),
        [
            ([], "at least one port name"),
            (["A:G1", "A:G1"], "repeated"),
            (["nope"], "unknown ports"),
            (NAMES_3, "cannot short every port"),
        ],
    )
    def test_bad_names_are_refused(self, names, match):
        with pytest.raises(ValueError, match=match):
            short_terminals(three_conductor_net(), names)

    def test_a_network_without_port_names_is_refused(self):
        net = skrf.Network(f=FREQ, s=np.zeros((NF, 2, 2)), z0=50.0, f_unit="Hz")
        net.port_names = None
        with pytest.raises(ValueError, match="one name per port"):
            short_terminals(net, ["a"])

    def test_complex_reference_impedances_are_refused(self):
        net = make_net(np.zeros((NF, 2, 2), dtype=complex), ["a", "b"], z0=50.0 + 5.0j)
        with pytest.raises(ValueError, match="complex port impedances"):
            short_terminals(net, ["a"])

    def test_ports_that_cannot_be_tied_are_refused(self):
        """S = -1 on a port is a short already: I + S_nn is singular."""
        s = np.zeros((NF, 2, 2), dtype=complex)
        s[:, 1, 1] = -1.0
        with pytest.raises(ValueError, match="singular"):
            short_terminals(make_net(s, ["a", "b"]), ["b"])

    def test_shorting_a_matched_one_port_termination(self):
        """A through line with one end shorted is a one port with S11 = -S11 + ...

        Two ports, S21 = exp(-j theta), S11 = S22 = 0: shorting port 2 gives
        S11' = 0 - S12 (1 + 0)^-1 S21 = -exp(-2 j theta).
        """
        theta = np.linspace(0.1, 2.0, NF)
        s = np.zeros((NF, 2, 2), dtype=complex)
        s[:, 0, 1] = s[:, 1, 0] = np.exp(-1j * theta)
        out = short_terminals(make_net(s, ["a", "b"]), ["b"])
        np.testing.assert_allclose(out.s[:, 0, 0], -np.exp(-2j * theta), atol=1e-14)


def symmetric_pair_net(*, lossy):
    """The ports (A1, B1, A2, B2) of a symmetric coupled pair, and its modes."""
    r, l, g, c = rlgc("sym", lossy=lossy)
    tv, ti, gamma = modal_data(r, l, g, c)
    s_t = modal_to_terminal_s(uniform_line_modal_s(gamma, ELL), *two_end(tv, ti))
    return make_net(s_t, NAMES_2), tv, ti, gamma


class TestMixedMode:
    @pytest.mark.parametrize("lossy", [False, True])
    def test_it_matches_the_defined_gamma_lines(self, lossy):
        """The same check as the guide's section 6, through the wrapper."""
        net, tv, ti, gamma = symmetric_pair_net(lossy=lossy)
        ratio = tv[:, 0, :] / tv[:, 1, :]
        odd = np.argmin(ratio.real, axis=1)
        even = 1 - odd
        fi = np.arange(NF)
        z_odd = tv[fi, 0, odd] / ti[fi, 0, odd]
        z_even = tv[fi, 0, even] / ti[fi, 0, even]
        freq = skrf.Frequency.from_f(FREQ, unit="Hz")
        odd_line = DefinedGammaZ0(
            freq, z0_port=100.0, z0=2 * z_odd, gamma=gamma[fi, odd]
        ).line(ELL, "m")
        even_line = DefinedGammaZ0(
            freq, z0_port=25.0, z0=z_even / 2, gamma=gamma[fi, even]
        ).line(ELL, "m")
        mixed = mixed_mode(net, [("A1", "B1"), ("A2", "B2")])
        assert np.abs(mixed.s[:, 0:2, 0:2] - odd_line.s).max() < TOL
        assert np.abs(mixed.s[:, 2:4, 2:4] - even_line.s).max() < TOL
        assert np.abs(mixed.s[:, 0:2, 2:4]).max() < TOL  # no mode conversion
        assert mixed.port_names == ["d(A1,B1)", "d(A2,B2)", "c(A1,B1)", "c(A2,B2)"]
        assert _modes(mixed) in (None, ["D", "D", "C", "C"])
        np.testing.assert_allclose(mixed.z0[0].real, [100.0, 100.0, 25.0, 25.0])

    def test_it_agrees_with_se2gmm_on_the_ordered_network(self):
        net, *_ = symmetric_pair_net(lossy=True)
        reference = net.copy()
        reference.se2gmm(p=2)
        mixed = mixed_mode(net, [("A1", "B1"), ("A2", "B2")])
        np.testing.assert_allclose(mixed.s, reference.s, rtol=0, atol=1e-12)

    def test_ports_are_taken_by_name_not_by_position(self):
        net, *_ = symmetric_pair_net(lossy=True)
        order = [3, 0, 2, 1]  # B2, A1, A2, B1
        shuffled = make_net(
            net.s[:, order][:, :, order],
            [NAMES_2[i] for i in order],
        )
        mixed = mixed_mode(shuffled, [("A1", "B1"), ("A2", "B2")])
        direct = mixed_mode(net, [("A1", "B1"), ("A2", "B2")])
        np.testing.assert_allclose(mixed.s, direct.s, rtol=0, atol=1e-14)
        assert mixed.port_names == direct.port_names

    def test_the_input_network_is_not_changed(self):
        net, *_ = symmetric_pair_net(lossy=True)
        before = net.s.copy()
        mixed_mode(net, [("A1", "B1"), ("A2", "B2")])
        np.testing.assert_array_equal(net.s, before)
        assert net.port_names == NAMES_2
        assert _modes(net) in (None, ["S"] * 4)

    def test_swapping_a_pair_flips_the_sign_of_the_conversion_terms(self):
        r, l, g, c = rlgc("asym")
        tv, ti, gamma = modal_data(r, l, g, c)
        s_t = modal_to_terminal_s(uniform_line_modal_s(gamma, ELL), *two_end(tv, ti))
        net = make_net(s_t, NAMES_2)
        forward = mixed_mode(net, [("A1", "B1"), ("A2", "B2")])
        backward = mixed_mode(net, [("B1", "A1"), ("B2", "A2")])
        conversion = forward.s[:, 0:2, 2:4]
        assert np.abs(conversion).max() > 1e-3  # an asymmetric line does convert
        np.testing.assert_allclose(backward.s[:, 0:2, 2:4], -conversion, atol=1e-14)
        np.testing.assert_allclose(
            backward.s[:, 2:4, 0:2], -forward.s[:, 2:4, 0:2], atol=1e-14
        )
        np.testing.assert_allclose(
            backward.s[:, 0:2, 0:2], forward.s[:, 0:2, 0:2], atol=1e-14
        )
        np.testing.assert_allclose(
            backward.s[:, 2:4, 2:4], forward.s[:, 2:4, 2:4], atol=1e-14
        )

    def test_unpaired_terminals_come_last_and_keep_their_impedance(self):
        net = three_conductor_net(z_ref=np.array([50.0, 40.0, 40.0, 50.0, 40.0, 40.0]))
        mixed = mixed_mode(net, [("A:S1", "A:S2"), ("B:S1", "B:S2")])
        assert mixed.port_names == [
            "d(A:S1,A:S2)",
            "d(B:S1,B:S2)",
            "c(A:S1,A:S2)",
            "c(B:S1,B:S2)",
            "A:G1",
            "B:G1",
        ]
        assert _modes(mixed) in (None, ["D", "D", "C", "C", "S", "S"])
        # scikit-rf's default: 2 z0 differential, z0 / 2 common mode, with
        # z0 the mean of the pair's reference impedances
        np.testing.assert_allclose(
            mixed.z0[0].real, [80.0, 80.0, 20.0, 20.0, 50.0, 50.0]
        )
        # same as se2gmm on the network permuted by hand
        order = [1, 2, 4, 5, 0, 3]
        by_hand = make_net(
            net.s[:, order][:, :, order],
            [NAMES_3[i] for i in order],
            z0=np.array([40.0, 40.0, 40.0, 40.0, 50.0, 50.0]),
        )
        by_hand.se2gmm(p=2)
        np.testing.assert_allclose(mixed.s, by_hand.s, rtol=0, atol=1e-12)

    def test_mixed_mode_reference_impedances_can_be_given(self):
        net, *_ = symmetric_pair_net(lossy=False)
        mixed = mixed_mode(
            net, [("A1", "B1"), ("A2", "B2")], z0_mm=np.array([90.0, 90.0, 30.0, 30.0])
        )
        np.testing.assert_allclose(mixed.z0[0].real, [90.0, 90.0, 30.0, 30.0])

    def test_it_is_exported_from_gsim_palace(self):
        assert exported_mixed_mode is mixed_mode

    @pytest.mark.parametrize(
        ("pairs", "match"),
        [
            ([], "pairs of names"),
            ([("A1",)], "pairs of names"),
            ([("A1", "B1", "A2")], "pairs of names"),
            ([("A1", "nope")], "unknown ports"),
            ([("A1", "B1"), ("B1", "A2")], "repeated"),
        ],
    )
    def test_bad_pairs_are_refused(self, pairs, match):
        net, *_ = symmetric_pair_net(lossy=False)
        with pytest.raises(ValueError, match=match):
            mixed_mode(net, pairs)

    def test_a_network_without_port_names_is_refused(self):
        net = skrf.Network(f=FREQ, s=np.zeros((NF, 2, 2)), z0=50.0, f_unit="Hz")
        net.port_names = None
        with pytest.raises(ValueError, match="one name per port"):
            mixed_mode(net, [("a", "b")])
