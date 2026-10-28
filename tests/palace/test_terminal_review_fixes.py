"""Review fixes of the terminal S-parameter change.

Two behaviours that the first version got wrong:

* ``mixed_mode`` turns a pair of terminals into a differential and a common
  port by rotating the waves, which is exact only when both terminals of the
  pair have the same reference impedance. A pair with two different ones is
  refused. The reference values below come from the impedance matrix of the
  network and from ``Network.se2gmm``, not from ``mixed_mode``.
* ``align_gauge`` looks the 3D effective index up at the frequency of every 2D
  solve; a solve outside the frequency range of the 3D wave numbers used to be
  paired with the end value without a word and is now an error.

Nothing here runs Palace. Generic synthetic values only.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c as C0  # noqa: N812

from gsim.palace.terminal_ports import PortModes2D, align_gauge, mixed_mode
from tests.palace.test_terminal_fixes import FREQ_GRID, N_2D, N_3D
from tests.palace.test_terminal_network_helpers import (
    NAMES_2,
    NF,
    make_net,
    symmetric_pair_net,
)

# -- mixed_mode with different reference impedances ----------------------------------


def matched_pair_net(z_a, z_b):
    """Two ports with ``S = 0`` at the reference impedances ``z_a`` and ``z_b``."""
    return make_net(np.zeros((NF, 2, 2), dtype=complex), ["a", "b"], z0=[z_a, z_b])


def matched_pair_mixed_s(z_a, z_b):
    """Mixed-mode power-wave S of the matched pair, from its impedance matrix.

    ``S = 0`` at ``z0 = (z_a, z_b)`` means ``V_i = z_i I_i`` on both ports. With
    ``V_d = V_a - V_b``, ``I_d = (I_a - I_b) / 2``, ``V_c = (V_a + V_b) / 2`` and
    ``I_c = I_a + I_b`` the impedance matrix of the (d, c) ports is below, and the
    reference impedances are twice and half of the mean of ``z_a`` and ``z_b``.
    """
    mean = 0.5 * (z_a + z_b)
    z_mm = np.array([[z_a + z_b, (z_a - z_b) / 2], [(z_a - z_b) / 2, (z_a + z_b) / 4]])
    z_ref = np.array([2 * mean, mean / 2])
    s_pseudo = (z_mm - np.diag(z_ref)) @ np.linalg.inv(z_mm + np.diag(z_ref))
    root = np.sqrt(z_ref)
    return s_pseudo * root[None, :] / root[:, None]


class TestMixedModeReferenceImpedances:
    def test_the_reference_is_not_the_plain_rotation(self):
        """What the refused case would have returned, against the right answer.

        The plain rotation of the waves gives ``S = 0`` for the matched pair.
        The mixed-mode S of the same network is not zero (the pair is matched
        port by port, not as a differential and a common port), and ``se2gmm``
        and the impedance matrix agree on it.
        """
        expected = matched_pair_mixed_s(50.0, 75.0)
        assert expected[0, 0] == pytest.approx(-1 / 99)
        assert expected[0, 1] == pytest.approx(-10 / 99)
        reference = matched_pair_net(50.0, 75.0)
        reference.se2gmm(p=1)
        for f in range(NF):
            np.testing.assert_allclose(reference.s[f], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(reference.z0[0].real, [125.0, 31.25])

    @pytest.mark.parametrize(("z_a", "z_b"), [(50.0, 75.0), (75.0, 50.0), (50.0, 51.0)])
    def test_a_pair_with_different_impedances_is_refused(self, z_a, z_b):
        with pytest.raises(ValueError, match="same reference impedance") as raised:
            mixed_mode(matched_pair_net(z_a, z_b), [("a", "b")])
        message = str(raised.value)
        assert "(a, b)" in message
        assert f"{z_a:g} and {z_b:g} ohm" in message
        assert "renormalize" in message

    def test_the_refusal_does_not_depend_on_the_mixed_mode_impedances(self):
        with pytest.raises(ValueError, match="same reference impedance"):
            mixed_mode(
                matched_pair_net(50.0, 75.0),
                [("a", "b")],
                z0_mm=np.array([125.0, 31.25]),
            )

    def test_a_difference_at_one_frequency_is_enough(self):
        z0 = np.full((NF, 2), 50.0)
        z0[-1, 1] = 60.0
        net = make_net(np.zeros((NF, 2, 2), dtype=complex), ["a", "b"], z0=z0)
        with pytest.raises(ValueError, match="same reference impedance"):
            mixed_mode(net, [("a", "b")])

    def test_only_the_pair_that_differs_is_named(self):
        s = np.zeros((NF, 4, 4), dtype=complex)
        net = make_net(s, NAMES_2, z0=[50.0, 50.0, 40.0, 60.0])
        with pytest.raises(ValueError, match="same reference impedance") as raised:
            mixed_mode(net, [("A1", "B1"), ("A2", "B2")])
        message = str(raised.value)
        assert "(A2, B2)" in message
        assert "(A1, B1)" not in message

    def test_rounding_noise_in_the_impedances_is_not_a_difference(self):
        net = matched_pair_net(50.0, 50.0 * (1.0 + 1e-12))
        mixed = mixed_mode(net, [("a", "b")])
        np.testing.assert_allclose(mixed.z0[0].real, [100.0, 25.0], rtol=1e-9)

    def test_pairs_may_differ_from_each_other(self):
        """Equal within each pair, different between the pairs: still converted."""
        rng = np.random.default_rng(5)
        s = 0.3 * (
            rng.normal(size=(NF, 4, 4)) + 1j * rng.normal(size=(NF, 4, 4))
        ).astype(complex)
        net = make_net(s, NAMES_2, z0=[50.0, 50.0, 75.0, 75.0])
        reference = net.copy()
        reference.se2gmm(p=2)
        mixed = mixed_mode(net, [("A1", "B1"), ("A2", "B2")])
        np.testing.assert_allclose(mixed.s, reference.s, rtol=0, atol=1e-12)
        np.testing.assert_allclose(mixed.z0[0].real, [100.0, 150.0, 25.0, 37.5])

    def test_renormalising_the_pair_first_gives_the_right_mixed_mode_s(self):
        """The advice of the error message works, on the matched pair above."""
        net = matched_pair_net(50.0, 75.0)
        net.renormalize(np.array([62.5, 62.5]))
        mixed = mixed_mode(net, [("a", "b")])
        expected = matched_pair_mixed_s(50.0, 75.0)
        for f in range(NF):
            np.testing.assert_allclose(mixed.s[f], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(mixed.z0[0].real, [125.0, 31.25])

    def test_equal_impedances_still_match_the_line_model(self):
        """The accepted case is untouched: same result as before on a lossy pair."""
        net, *_ = symmetric_pair_net(lossy=True)
        reference = net.copy()
        reference.se2gmm(p=2)
        mixed = mixed_mode(net, [("A1", "B1"), ("A2", "B2")])
        np.testing.assert_allclose(mixed.s, reference.s, rtol=0, atol=1e-12)

    def test_the_docstring_states_the_condition(self):
        doc = " ".join((mixed_mode.__doc__ or "").split())
        assert "must have the same reference impedance" in doc
        assert "ValueError" in doc


# -- align_gauge with a 2D solve outside the 3D frequency range ----------------------


def solves_at(*freqs):
    """2D solves of a three-terminal port at the given frequencies (GHz)."""
    t_v = np.eye(3) + 0.2 * (np.ones((3, 3)) - np.eye(3))
    return [PortModes2D(f, N_2D, t_v, 0.02 * t_v) for f in freqs]


def wave_numbers():
    """3D wave numbers of the modes on ``FREQ_GRID`` (1, 10, 50 GHz)."""
    k0 = 2.0 * np.pi * FREQ_GRID * 1e9 / C0
    return k0[:, None] * N_3D[None, :]


def align(solves, **kwargs):
    """``align_gauge`` on ``FREQ_GRID`` with the pairing opened to the 2D error."""
    return align_gauge(
        solves,
        freq_ghz=FREQ_GRID,
        wave_numbers=wave_numbers(),
        rtol=0.15,
        **kwargs,
    )


class TestSolveOutsideTheWaveNumberRange:
    def test_a_solve_above_the_range_is_refused(self):
        with pytest.raises(ValueError, match="outside the 3D wave-number range") as e:
            align(solves_at(1.0, 60.0))
        assert "[60.0] GHz" in str(e.value)
        assert "1 to 50 GHz" in str(e.value)

    def test_a_solve_below_the_range_is_refused(self):
        with pytest.raises(ValueError, match="outside the 3D wave-number range") as e:
            align(solves_at(0.5, 50.0))
        assert "[0.5] GHz" in str(e.value)

    def test_every_offending_solve_is_listed(self):
        with pytest.raises(ValueError, match="outside the 3D wave-number range") as e:
            align(solves_at(0.5, 20.0, 60.0, 80.0))
        assert "[0.5, 60.0, 80.0] GHz" in str(e.value)

    def test_extrapolate_does_not_allow_it(self):
        """``extrapolate`` holds the 2D matrices; the 3D wave numbers stay unknown."""
        with pytest.raises(ValueError, match="outside the 3D wave-number range"):
            align(solves_at(1.0, 60.0), extrapolate=True)

    def test_the_other_error_still_comes_first_for_a_target_outside_the_2d_range(self):
        with pytest.raises(ValueError, match="add 2D solves"):
            align(solves_at(10.0, 20.0))

    @pytest.mark.parametrize("freqs", [(1.0, 50.0), (1.0, 20.0, 50.0), (10.0, 10.5)])
    def test_solves_inside_the_range_and_on_its_ends_are_accepted(self, freqs):
        if freqs[0] > FREQ_GRID[0] or freqs[-1] < FREQ_GRID[-1]:
            aligned = align(solves_at(*freqs), extrapolate=True)
        else:
            aligned = align(solves_at(*freqs))
        assert aligned.t_v.shape == (3, 3, 3)
        assert aligned.mismatch == pytest.approx(1.5837 / 1.5244 - 1.0, rel=2e-3)

    def test_the_ends_are_met_to_rounding(self):
        solves = solves_at(1.0 * (1.0 + 1e-10), 50.0 * (1.0 - 1e-10))
        assert align(solves).t_v.shape == (3, 3, 3)
        beyond = solves_at(1.0, 50.0 * (1.0 + 1e-6))
        with pytest.raises(ValueError, match="outside the 3D wave-number range"):
            align(beyond)

    def test_the_pairing_alone_would_not_have_caught_it(self):
        """The 3D n_eff is flat here, so a held end value pairs without a word.

        The solves inside the range give no error, and the only difference to
        the refused call is where the 2D frequency lies.
        """
        aligned = align(solves_at(1.0, 50.0))
        assert aligned.pairing == ((0, 1, 2), (0, 1, 2))
        with pytest.raises(ValueError, match="outside the 3D wave-number range"):
            align(solves_at(1.0, 60.0))
