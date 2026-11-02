"""One scatter interpolator behind every mesh-to-mesh sample."""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.interpolate import DegenerateSampleCloudError, sample_at

SQUARE = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


class TestSampleAt:
    def test_it_interpolates_linearly_inside_the_hull(self):
        values = SQUARE[:, 0] + SQUARE[:, 1]
        sampled, missing = sample_at(SQUARE, values, [[0.5, 0.5]])
        assert sampled == pytest.approx([1.0])
        assert not missing.any()

    def test_a_target_outside_the_hull_is_flagged_missing(self):
        sampled, missing = sample_at(SQUARE, SQUARE[:, 0], [[0.5, 0.5], [3.0, 3.0]])
        assert missing.tolist() == [False, True]
        assert np.isfinite(sampled).all()

    def test_the_default_fill_extends_the_nearest_sample(self):
        sampled, _missing = sample_at(SQUARE, SQUARE[:, 0], [[9.0, 9.0]])
        assert sampled == pytest.approx([1.0])

    def test_a_scalar_fill_is_used_instead(self):
        sampled, _missing = sample_at(SQUARE, SQUARE[:, 0], [[9.0, 9.0]], fill=-1.0)
        assert sampled == pytest.approx([-1.0])

    def test_complex_values_stay_complex(self):
        values = SQUARE[:, 0] + 1j * SQUARE[:, 1]
        sampled, _missing = sample_at(SQUARE, values, [[0.5, 0.25]])
        assert sampled[0] == pytest.approx(0.5 + 0.25j)

    def test_several_columns_share_one_triangulation(self):
        values = np.column_stack([SQUARE[:, 0], 2.0 * SQUARE[:, 1]])
        sampled, missing = sample_at(SQUARE, values, [[0.5, 0.5], [7.0, 7.0]])
        assert sampled.shape == (2, 2)
        assert sampled[0] == pytest.approx([0.5, 1.0])
        assert missing.tolist() == [False, True]

    def test_a_scalar_fill_covers_every_column(self):
        values = np.column_stack([SQUARE[:, 0], SQUARE[:, 1]])
        sampled, _missing = sample_at(SQUARE, values, [[9.0, 9.0]], fill=0.0)
        assert sampled[0] == pytest.approx([0.0, 0.0])

    def test_a_degenerate_cloud_is_reported_rather_than_guessed_at(self):
        collinear = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        with pytest.raises(DegenerateSampleCloudError):
            sample_at(collinear, collinear[:, 0], [[0.5, 0.5]])

    def test_the_sample_count_must_match_the_values(self):
        with pytest.raises(ValueError, match="same length"):
            sample_at(SQUARE, [1.0, 2.0], [[0.5, 0.5]])
