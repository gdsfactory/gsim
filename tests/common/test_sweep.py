"""One base for a sweep of points over a scalar key."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel

from gsim.common.sweep import BIAS_TOL_V, ScalarSweep


class Point(BaseModel):
    """A swept point carrying nothing but its own key."""

    bias_v: float


class Sweep(ScalarSweep[Point]):
    """A sweep keyed on its points' bias, as a Backend's is."""

    sweep_noun = "bias sweep"
    key_unit = "V"

    def _key(self, point: Point) -> float:
        return point.bias_v


class TestScalarSweep:
    @pytest.fixture
    def sweep(self):
        return Sweep(points=[Point(bias_v=v) for v in (0.0, 1.0, 2.0)])

    def test_the_keys_come_back_in_sweep_order(self, sweep):
        assert np.array_equal(sweep.keys, np.asarray([0.0, 1.0, 2.0]))

    def test_an_empty_sweep_has_no_keys(self):
        assert Sweep().keys.shape == (0,)

    def test_a_point_is_found_by_its_key(self, sweep):
        assert sweep.point_at(1.0).bias_v == 1.0

    def test_a_key_within_the_tolerance_is_the_same_point(self, sweep):
        assert sweep.point_at(1.0 + BIAS_TOL_V / 2).bias_v == 1.0

    def test_a_key_the_sweep_never_visited_names_the_ones_it_did(self, sweep):
        with pytest.raises(
            ValueError, match=r"no point at V = 3; it visited 0, 1, 2 V"
        ):
            sweep.point_at(3.0)

    def test_a_wider_tolerance_is_the_callers_to_give(self, sweep):
        assert sweep.point_at(1.4, tol=0.5).bias_v == 1.0
