"""Pure-function tests for the analytic doping profiles (hand values)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from gsim.tcad.doping import (
    CallableDoping,
    GaussianDoping,
    ImplantDoping,
    StepDoping,
    TableDoping,
    acceptor_donor_concentrations,
    net_doping_cm3,
)


class TestStepDoping:
    def test_uniform_without_windows(self):
        step = StepDoping(region="slab", dopant_type="donor", concentration_cm3=1e18)
        values = step.concentration([-5.0, 0.0, 5.0], [0.0, 0.1, 0.2])
        np.testing.assert_allclose(values, 1e18)

    def test_box_windows(self):
        step = StepDoping(
            region="slab",
            dopant_type="acceptor",
            concentration_cm3=2e17,
            x_range=(-1.0, 1.0),
            y_range=(0.0, 0.2),
        )
        x = np.array([0.0, 0.0, 2.0, -1.0])
        y = np.array([0.1, 0.5, 0.1, 0.0])
        # Inside; y outside; x outside; on the closed boundary.
        np.testing.assert_allclose(step.concentration(x, y), [2e17, 0.0, 0.0, 2e17])

    def test_scalar_input(self):
        step = StepDoping(region="slab", dopant_type="donor", concentration_cm3=1e18)
        assert float(step.concentration(0.0, 0.0)) == 1e18

    def test_rejects_descending_range(self):
        with pytest.raises(ValidationError):
            StepDoping(
                region="slab",
                dopant_type="donor",
                concentration_cm3=1e18,
                x_range=(1.0, -1.0),
            )

    def test_rejects_nonpositive_concentration(self):
        with pytest.raises(ValidationError):
            StepDoping(region="slab", dopant_type="donor", concentration_cm3=0.0)


class TestGaussianDoping:
    def test_peak_and_sigma_hand_values(self):
        gauss = GaussianDoping(
            region="slab",
            dopant_type="acceptor",
            peak_cm3=1e19,
            center=(0.0, 0.0),
            sigma_x=0.5,
        )
        assert float(gauss.concentration(0.0, 0.0)) == pytest.approx(1e19)
        # One sigma off-center: peak * exp(-1/2).
        assert float(gauss.concentration(0.5, 0.0)) == pytest.approx(
            1e19 * math.exp(-0.5)
        )
        # Uniform along y when sigma_y omitted.
        assert float(gauss.concentration(0.0, 3.0)) == pytest.approx(1e19)

    def test_separable_both_axes(self):
        gauss = GaussianDoping(
            region="slab",
            dopant_type="donor",
            peak_cm3=1e18,
            center=(1.0, 0.1),
            sigma_x=0.2,
            sigma_y=0.05,
        )
        expected = 1e18 * math.exp(-0.5) * math.exp(-0.5)
        assert float(gauss.concentration(1.2, 0.15)) == pytest.approx(expected)

    def test_hard_window_clips_tail(self):
        gauss = GaussianDoping(
            region="slab",
            dopant_type="donor",
            peak_cm3=1e18,
            center=(0.0, 0.0),
            sigma_x=1.0,
            x_range=(-1.0, 1.0),
        )
        assert float(gauss.concentration(2.0, 0.0)) == 0.0

    def test_requires_a_sigma(self):
        with pytest.raises(ValidationError):
            GaussianDoping(
                region="slab",
                dopant_type="donor",
                peak_cm3=1e18,
                center=(0.0, 0.0),
            )


class TestImplantDoping:
    def test_peak_at_projected_range(self):
        implant = ImplantDoping(
            region="slab",
            dopant_type="acceptor",
            peak_cm3=5e18,
            surface_y=0.22,
            range_um=0.1,
            straggle_um=0.03,
        )
        # Peak at depth Rp below the surface.
        assert float(implant.concentration(0.0, 0.12)) == pytest.approx(5e18)
        # One straggle deeper: peak * exp(-1/2).
        assert float(implant.concentration(0.0, 0.09)) == pytest.approx(
            5e18 * math.exp(-0.5)
        )

    def test_zero_above_surface(self):
        implant = ImplantDoping(
            region="slab",
            dopant_type="acceptor",
            peak_cm3=5e18,
            surface_y=0.22,
            range_um=0.1,
            straggle_um=0.03,
        )
        assert float(implant.concentration(0.0, 0.3)) == 0.0

    def test_lateral_window(self):
        implant = ImplantDoping(
            region="slab",
            dopant_type="donor",
            peak_cm3=1e20,
            surface_y=0.0,
            range_um=0.05,
            straggle_um=0.02,
            x_range=(2.0, 4.0),
        )
        assert float(implant.concentration(3.0, -0.05)) == pytest.approx(1e20)
        assert float(implant.concentration(0.0, -0.05)) == 0.0


class TestSuperposition:
    def test_acceptors_and_donors_summed_separately(self):
        profiles = [
            StepDoping(
                region="slab",
                dopant_type="acceptor",
                concentration_cm3=1e18,
                x_range=(-2.0, 0.0),
            ),
            StepDoping(
                region="slab",
                dopant_type="donor",
                concentration_cm3=4e17,
                x_range=(0.0, 2.0),
            ),
            StepDoping(region="slab", dopant_type="donor", concentration_cm3=1e15),
        ]
        x = np.array([-1.0, 1.0])
        y = np.zeros(2)
        acceptors, donors = acceptor_donor_concentrations(profiles, x, y)
        np.testing.assert_allclose(acceptors, [1e18, 0.0])
        np.testing.assert_allclose(donors, [1e15, 4e17 + 1e15])

    def test_net_doping_sign_convention(self):
        profiles = [
            StepDoping(region="slab", dopant_type="acceptor", concentration_cm3=1e18),
            StepDoping(region="slab", dopant_type="donor", concentration_cm3=4e17),
        ]
        net = net_doping_cm3(profiles, 0.0, 0.0)
        # donors - acceptors: p-type net doping is negative.
        assert float(net) == pytest.approx(4e17 - 1e18)


class TestTableDoping:
    """Sampled profiles: a SIMS depth scan is the case this exists for."""

    def test_depth_profile_interpolates_between_samples(self):
        table = TableDoping(
            region="rib",
            dopant_type="acceptor",
            y_um=[0.0, 0.1, 0.2],
            values_cm3=[1e18, 5e17, 1e17],
        )
        # On a sample, and halfway between two of them.
        assert float(table.concentration(0.0, 0.1)) == pytest.approx(5e17)
        assert float(table.concentration(0.0, 0.05)) == pytest.approx(7.5e17)

    def test_a_single_axis_is_uniform_along_the_other(self):
        table = TableDoping(
            region="rib",
            dopant_type="donor",
            y_um=[0.0, 0.2],
            values_cm3=[1e18, 1e17],
        )
        x = np.array([-5.0, 0.0, 5.0])
        np.testing.assert_allclose(table.concentration(x, np.full(3, 0.1)), 5.5e17)

    def test_edge_fill_holds_the_boundary_sample(self):
        table = TableDoping(
            region="rib",
            dopant_type="acceptor",
            y_um=[0.0, 0.2],
            values_cm3=[1e18, 1e17],
        )
        np.testing.assert_allclose(table.concentration(0.0, [-3.0, 9.0]), [1e18, 1e17])

    def test_zero_fill_drops_outside_the_grid(self):
        table = TableDoping(
            region="rib",
            dopant_type="acceptor",
            y_um=[0.0, 0.2],
            values_cm3=[1e18, 1e17],
            fill="zero",
        )
        np.testing.assert_allclose(table.concentration(0.0, [-3.0, 9.0]), [0.0, 0.0])
        # Inside the grid, "zero" changes nothing.
        assert float(table.concentration(0.0, 0.1)) == pytest.approx(5.5e17)

    def test_two_axes_interpolate_bilinearly(self):
        table = TableDoping(
            region="rib",
            dopant_type="donor",
            x_um=[0.0, 1.0],
            y_um=[0.0, 1.0],
            values_cm3=[[0.0, 1e17], [1e17, 2e17]],
        )
        # Corner samples, then the cell center: the mean of the four.
        assert float(table.concentration(0.0, 1.0)) == pytest.approx(1e17)
        assert float(table.concentration(1.0, 1.0)) == pytest.approx(2e17)
        assert float(table.concentration(0.5, 0.5)) == pytest.approx(1e17)

    def test_hard_window_clips_the_table(self):
        table = TableDoping(
            region="rib",
            dopant_type="acceptor",
            y_um=[0.0, 0.2],
            values_cm3=[1e18, 1e18],
            x_range=(-1.0, 1.0),
        )
        values = table.concentration([0.0, 2.0], [0.1, 0.1])
        np.testing.assert_allclose(values, [1e18, 0.0])

    def test_rejects_a_table_with_no_axis(self):
        with pytest.raises(ValidationError, match="x_um and/or y_um"):
            TableDoping(region="rib", dopant_type="donor", values_cm3=[1e18, 1e17])

    def test_rejects_descending_samples(self):
        with pytest.raises(ValidationError, match="ascending"):
            TableDoping(
                region="rib",
                dopant_type="donor",
                y_um=[0.2, 0.0],
                values_cm3=[1e18, 1e17],
            )

    def test_rejects_a_single_sample(self):
        with pytest.raises(ValidationError, match="at least two samples"):
            TableDoping(
                region="rib", dopant_type="donor", y_um=[0.1], values_cm3=[1e18]
            )

    def test_rejects_mismatched_value_shape(self):
        with pytest.raises(ValidationError, match=r"values_cm3\[x\]\[y\]"):
            TableDoping(
                region="rib",
                dopant_type="donor",
                x_um=[0.0, 1.0],
                y_um=[0.0, 1.0],
                values_cm3=[1e18, 1e17],
            )

    def test_rejects_negative_concentrations(self):
        with pytest.raises(ValidationError, match="non-negative"):
            TableDoping(
                region="rib",
                dopant_type="donor",
                y_um=[0.0, 0.2],
                values_cm3=[1e18, -1e17],
            )

    def test_round_trips_through_json(self):
        """Unlike CallableDoping, a table is data and survives a dump."""
        table = TableDoping(
            region="rib",
            dopant_type="acceptor",
            y_um=[0.0, 0.2],
            values_cm3=[1e18, 1e17],
        )
        restored = TableDoping.model_validate_json(table.model_dump_json())
        assert restored == table
        assert restored.kind == "table"


class TestCallableDoping:
    """The escape hatch for a shape no other member expresses."""

    def test_evaluates_the_function_at_the_coordinates(self):
        profile = CallableDoping(
            region="rib",
            dopant_type="donor",
            function=lambda x, y: 1e18 * np.exp(-np.abs(x)),
        )
        values = profile.concentration([0.0, 1.0], [0.0, 0.0])
        np.testing.assert_allclose(values, [1e18, 1e18 * math.exp(-1.0)])

    def test_a_scalar_return_broadcasts(self):
        profile = CallableDoping(
            region="rib", dopant_type="donor", function=lambda x, y: 3e17
        )
        np.testing.assert_allclose(
            profile.concentration([0.0, 1.0, 2.0], 0.0), [3e17, 3e17, 3e17]
        )

    def test_hard_window_clips_the_function(self):
        profile = CallableDoping(
            region="rib",
            dopant_type="acceptor",
            function=lambda x, y: np.full(np.broadcast(x, y).shape, 1e18),
            y_range=(0.0, 0.2),
        )
        values = profile.concentration(0.0, [0.1, 0.5])
        np.testing.assert_allclose(values, [1e18, 0.0])

    def test_a_wrong_shaped_return_names_the_region(self):
        profile = CallableDoping(
            region="rib", dopant_type="donor", function=lambda x, y: np.zeros(3)
        )
        with pytest.raises(ValueError, match=r"'rib'.*does not broadcast"):
            profile.concentration(np.zeros(5), np.zeros(5))

    def test_negative_concentrations_are_refused(self):
        """The p/n sense is dopant_type's job, not the function's sign."""
        profile = CallableDoping(
            region="rib", dopant_type="donor", function=lambda x, y: -1e18
        )
        with pytest.raises(ValueError, match="negative concentrations"):
            profile.concentration(0.0, 0.0)

    def test_non_finite_concentrations_are_refused(self):
        profile = CallableDoping(
            region="rib", dopant_type="donor", function=lambda x, y: np.inf
        )
        with pytest.raises(ValueError, match="non-finite"):
            profile.concentration(0.0, 0.0)


class TestSuperposedShapes:
    def test_a_table_and_a_function_sum_like_any_other_shape(self):
        profiles = [
            TableDoping(
                region="rib",
                dopant_type="acceptor",
                y_um=[0.0, 0.2],
                values_cm3=[1e18, 0.0],
            ),
            CallableDoping(
                region="rib", dopant_type="donor", function=lambda x, y: 4e17
            ),
        ]
        acceptors, donors = acceptor_donor_concentrations(profiles, 0.0, 0.1)
        assert float(acceptors) == pytest.approx(5e17)
        assert float(donors) == pytest.approx(4e17)
        assert float(net_doping_cm3(profiles, 0.0, 0.1)) == pytest.approx(-1e17)
