"""Selecting the physical line Mode out of a set of solved Modes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gsim.common.modes import (
    MAX_COMMON_MODE_FRACTION,
    Conductor,
    LineReading,
    NoLineModeError,
    common_mode_fraction,
    propagating_modes,
    select_line_mode,
    wall_mode_from_currents,
    z0_power_current,
)


def mode(n_eff: complex):
    """A solver Mode stand-in carrying only its effective index."""
    return SimpleNamespace(n_eff=complex(n_eff))


class TestDefaultRule:
    def test_picks_the_slowest_propagating_mode(self):
        modes = [mode(3.2 - 0.01j), mode(2.1 - 0.005j)]
        assert select_line_mode(modes).n_eff == 3.2 - 0.01j

    def test_rejects_evanescent_and_spurious_modes(self):
        physical = mode(3.0 - 0.02j)
        modes = [
            mode(0.4 - 0.001j),  # below the light line: not guided
            mode(6.0 - 9.0j),  # |Im| > Re: spurious / lossy junk
            physical,
        ]
        assert select_line_mode(modes) is physical

    def test_accepts_plain_complex_numbers_and_mappings(self):
        assert select_line_mode([1.5 + 0j, 3.0 + 0j]) == 3.0 + 0j
        picked = select_line_mode([{"n_eff": 2.0 + 0j}, {"n_eff": 4.0 + 0j}])
        assert picked["n_eff"] == 4.0 + 0j

    def test_candidates_are_available_on_their_own(self):
        modes = [mode(0.5 + 0j), mode(3.0 - 0.1j)]
        assert [m.n_eff for m in propagating_modes(modes)] == [3.0 - 0.1j]


class TestCustomRule:
    def test_caller_rule_replaces_the_candidate_set(self):
        modes = [mode(3.2 + 0j), mode(2.1 + 0j)]

        def slowest_only(candidates):
            return [min(candidates, key=lambda m: m.n_eff.real)]

        assert select_line_mode(modes, rule=slowest_only).n_eff == 2.1 + 0j

    def test_caller_rule_may_keep_modes_the_default_rejects(self):
        modes = [mode(0.4 + 0j)]
        assert select_line_mode(modes, rule=list).n_eff == 0.4 + 0j

    def test_empty_result_from_a_caller_rule_is_reported(self):
        with pytest.raises(NoLineModeError):
            select_line_mode([mode(3.0 + 0j)], rule=lambda _modes: [])


class TestDegeneracy:
    def test_near_degenerate_candidates_warn_and_name_the_indices(self):
        modes = [mode(3.00 + 0j), mode(2.99 + 0j)]
        with pytest.warns(UserWarning, match="2.99"):
            picked = select_line_mode(modes)
        assert picked.n_eff == 3.00 + 0j

    def test_well_separated_candidates_are_silent(self):
        import warnings

        modes = [mode(3.0 + 0j), mode(2.0 + 0j)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert select_line_mode(modes).n_eff == 3.0 + 0j

    def test_tolerance_is_configurable(self):
        modes = [mode(3.0 + 0j), mode(2.0 + 0j)]
        with pytest.warns(UserWarning, match="degenerate"):
            select_line_mode(modes, degeneracy_rtol=0.5)


class TestNoCandidates:
    def test_error_names_every_solved_mode(self):
        modes = [mode(0.4 + 0j), mode(0.2 - 3.0j)]
        with pytest.raises(NoLineModeError) as excinfo:
            select_line_mode(modes)
        message = str(excinfo.value)
        assert "0.4" in message
        assert "2 mode" in message

    def test_error_when_nothing_was_solved_at_all(self):
        with pytest.raises(NoLineModeError, match="no modes"):
            select_line_mode([])

    def test_is_a_value_error(self):
        assert issubclass(NoLineModeError, ValueError)


class TestLossBound:
    """How much loss still counts as propagating."""

    def test_a_mode_losing_as_fast_as_it_advances_is_dropped(self):
        candidates = propagating_modes([3.0 - 3.0j, 3.0 - 0.1j])
        assert candidates == [3.0 - 0.1j]

    def test_a_tighter_bound_drops_the_modes_just_inside_the_default(self):
        """A discretization's spurious modes cluster where alpha ~ beta."""
        spurious = 31.9 - 31.8j
        assert propagating_modes([spurious, 2.0 - 1e-6j]) == [spurious, 2.0 - 1e-6j]
        assert propagating_modes([spurious, 2.0 - 1e-6j], max_loss_ratio=0.5) == [
            2.0 - 1e-6j
        ]

    def test_the_bound_reaches_the_selection(self):
        modes = [31.9 - 31.8j, 2.0 - 1e-6j]
        assert select_line_mode(modes) == 31.9 - 31.8j
        assert select_line_mode(modes, max_loss_ratio=0.5) == 2.0 - 1e-6j

    def test_a_bound_nothing_survives_is_reported(self):
        with pytest.raises(NoLineModeError, match="No propagating line mode"):
            select_line_mode([3.0 - 2.0j], max_loss_ratio=0.5)


class TestGainBound:
    """A Mode that grows along the line is a numerical artefact, not a candidate.

    In the ``exp(+i omega t)`` convention a lossy Mode has ``Im(n_eff) < 0``;
    a positive imaginary part is gain, which a passive line cannot have.
    Palace's shift-and-invert search returns such a Mode now and then, well
    inside the loss bound, and it must not be taken for the line.
    """

    def test_a_gain_mode_inside_the_loss_bound_is_dropped(self):
        physical = mode(2.03 - 6e-7j)
        spurious = mode(1170.0 + 512.0j)  # |Im|/Re = 0.44, but growing
        assert propagating_modes([physical, spurious], max_loss_ratio=0.5) == [physical]

    def test_it_never_becomes_the_selected_line_mode(self):
        physical = mode(2.03 - 6e-7j)
        assert (
            select_line_mode([physical, mode(1170.0 + 512.0j)], max_loss_ratio=0.5)
            is physical
        )

    def test_numerical_noise_on_a_lossless_mode_is_not_gain(self):
        lossless = mode(2.03 + 1e-9j)
        assert propagating_modes([lossless]) == [lossless]

    def test_the_bound_is_a_parameter(self):
        """A coarser solve's noise can be admitted without a whole rule."""
        noisy = mode(2.4 + 3e-3j)
        assert propagating_modes([noisy]) == []
        assert propagating_modes([noisy], max_gain_ratio=1e-2) == [noisy]


class TestCommonModeFraction:
    """Telling the line Mode from the wall Mode by its electrode currents.

    A shielded two-electrode line has two propagating Modes: the line
    Mode, whose signal and return electrodes carry equal and opposite
    currents, and the wall Mode, on which both electrodes carry the same
    current and return it through the metallic Window wall. The fraction
    reads that difference off the two currents alone, so either Route can
    ask it.
    """

    def test_equal_and_opposite_currents_are_the_line_mode(self):
        assert common_mode_fraction(0.156 + 0j, -0.157 + 0j) == pytest.approx(
            0.003, abs=1e-3
        )

    def test_equal_and_alike_currents_are_the_wall_mode(self):
        assert common_mode_fraction(0.075 + 0j, 0.0751 + 0j) == pytest.approx(
            1.0, abs=1e-3
        )

    def test_it_is_a_fraction_between_the_two(self):
        assert common_mode_fraction(1.0 + 0j, 0.0 + 0j) == pytest.approx(1.0)
        assert common_mode_fraction(1.0 + 0j, -0.5 + 0j) == pytest.approx(1.0 / 3.0)

    def test_it_reads_the_phase_not_only_the_magnitude(self):
        """A quarter turn between the two is neither balanced nor alike."""
        assert common_mode_fraction(1.0 + 0j, 1.0j) == pytest.approx(np.sqrt(2.0) / 2.0)

    def test_no_current_at_all_is_no_reading(self):
        assert np.isnan(common_mode_fraction(0.0 + 0j, 0.0 + 0j))

    def test_the_default_bound_sits_between_the_two_modes(self):
        assert 0.0 < MAX_COMMON_MODE_FRACTION < 1.0


class TestWallModeFromCurrents:
    """The pairing behind the femwell Route's reading, on bare currents."""

    def test_alike_currents_are_the_wall_mode(self):
        wall_mode, diagnostic = wall_mode_from_currents(0.075 + 0j, 0.0751 + 0j)
        assert wall_mode is True
        assert "window wall" in diagnostic
        assert "100%" in diagnostic

    def test_opposite_currents_are_the_line_mode(self):
        wall_mode, diagnostic = wall_mode_from_currents(0.156 + 0j, -0.157 + 0j)
        assert wall_mode is False
        assert "line mode" in diagnostic

    def test_no_current_at_all_is_no_reading(self):
        wall_mode, diagnostic = wall_mode_from_currents(0j, 0j)
        assert wall_mode is None
        assert "neither electrode" in diagnostic

    def test_the_bound_is_the_shared_one(self):
        # |1 - 0.34| / 1.34 sits just under one half, |1 - 0.32| / 1.32 just over.
        assert wall_mode_from_currents(1.0 + 0j, -0.34 + 0j)[0] is False
        assert wall_mode_from_currents(1.0 + 0j, -0.32 + 0j)[0] is True


class TestZ0PowerCurrent:
    def test_it_divides_twice_the_power_by_the_squared_current(self):
        assert z0_power_current(50.0 + 5.0j, 2.0) == pytest.approx(25.0 + 2.5j)

    def test_the_current_enters_as_a_magnitude_only(self):
        forward = z0_power_current(50.0, 2.0)
        assert z0_power_current(50.0, -2.0) == forward
        assert z0_power_current(50.0, 2.0j) == forward

    def test_a_mode_travelling_against_the_normal_is_flipped_back(self):
        flipped = z0_power_current(-41.716 - 0.067j, 1.0)
        assert flipped == z0_power_current(41.716 + 0.067j, 1.0)
        assert flipped.real > 0.0
        assert flipped.imag > 0.0

    def test_a_mode_carrying_no_current_is_reported(self):
        with pytest.raises(ValueError, match="no current"):
            z0_power_current(50.0, 0.0)


class TestDescriptors:
    def test_a_conductor_defaults_to_a_meshed_region(self):
        conductor = Conductor("electrode_low", ((-22.6, -20.6), (0.0, 0.5)))
        assert conductor.model == "volume"
        assert conductor.extent == ((-22.6, -20.6), (0.0, 0.5))

    def test_a_reading_carries_what_a_route_read(self):
        reading = LineReading(n_eff=2.9 - 1.4e-3j, z0_ohm=41.0 + 0j, wall_mode=False)
        assert reading.diagnostic == ""
        assert reading.z0_ohm == 41.0
