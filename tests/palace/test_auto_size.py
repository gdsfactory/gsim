"""Tests for geometry-aware mesh auto-sizing."""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace.mesh.auto_size import (
    auto_refined_mesh_size,
    min_conductor_feature_size,
    min_conductor_gap,
)


@pytest.fixture(autouse=True)
def _activate_pdk():
    """Activate the generic PDK for all tests."""
    gf.gpdk.PDK.activate()


def _conductor_stack(gds_layer: tuple[int, int] = (1, 0)) -> LayerStack:
    """Return a stack with a single conductor layer."""
    stack = LayerStack()
    stack.layers["metal"] = Layer(
        name="metal",
        gds_layer=gds_layer,
        zmin=0.0,
        zmax=0.5,
        thickness=0.5,
        material="copper",
        layer_type="conductor",
    )
    return stack


def _narrow_trace(width: float = 2.0, length: float = 100.0) -> gf.Component:
    """Create a simple rectangular trace on layer (1, 0)."""
    c = gf.Component()
    half_w = width / 2
    half_l = length / 2
    c.add_polygon(
        [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)],
        layer=(1, 0),
    )
    return c


def _cpw_component(
    s_width: float = 20.0,
    gap: float = 15.0,
    ground_width: float = 40.0,
    length: float = 300.0,
) -> gf.Component:
    """Create a CPW layout: signal trace plus two coplanar ground planes.

    Signal centered at y=0, extending along x. Grounds on either side in y,
    separated from signal by ``gap``.
    """
    c = gf.Component()
    half_l = length / 2
    half_s = s_width / 2
    # Signal: s_width wide (y span) x length (x span)
    c.add_polygon(
        [(-half_l, -half_s), (half_l, -half_s), (half_l, half_s), (-half_l, half_s)],
        layer=(1, 0),
    )
    # Lower ground: y in [-(half_s+gap+ground_width), -(half_s+gap)]
    c.add_polygon(
        [
            (-half_l, -(half_s + gap + ground_width)),
            (half_l, -(half_s + gap + ground_width)),
            (half_l, -(half_s + gap)),
            (-half_l, -(half_s + gap)),
        ],
        layer=(1, 0),
    )
    # Upper ground: y in [half_s+gap, half_s+gap+ground_width]
    c.add_polygon(
        [
            (-half_l, half_s + gap),
            (half_l, half_s + gap),
            (half_l, half_s + gap + ground_width),
            (-half_l, half_s + gap + ground_width),
        ],
        layer=(1, 0),
    )
    return c


def _two_far_polys() -> gf.Component:
    """Two small polygons separated by a larger distance than any bbox dim."""
    c = gf.Component()
    # 50um x 50um at origin
    c.add_polygon([(0, 0), (50, 0), (50, 50), (0, 50)], layer=(1, 0))
    # 50um x 50um at (500, 0) — gap is ~450um, bigger than any bbox (50)
    c.add_polygon([(500, 0), (550, 0), (550, 50), (500, 50)], layer=(1, 0))
    return c


class TestMinConductorFeatureSize:
    """Tests for min_conductor_feature_size."""

    def test_narrow_trace_returns_width(self):
        component = _narrow_trace(width=2.0, length=100.0)
        stack = _conductor_stack()
        assert min_conductor_feature_size(component, stack) == 2.0

    def test_returns_none_when_no_conductors(self):
        component = _narrow_trace()
        empty_stack = LayerStack()
        assert min_conductor_feature_size(component, empty_stack) is None

    def test_ignores_non_conductor_layers(self):
        # Narrow polygon on a layer that's NOT a conductor must be ignored.
        component = _narrow_trace(width=2.0)
        stack = _conductor_stack(gds_layer=(99, 0))
        assert min_conductor_feature_size(component, stack) is None

    def test_cpw_gap_beats_trace_width(self):
        """For a CPW with 20 um trace + 15 um gap, should return 15 (gap)."""
        component = _cpw_component(s_width=20.0, gap=15.0, ground_width=40.0)
        stack = _conductor_stack()
        result = min_conductor_feature_size(component, stack)
        assert result == pytest.approx(15.0)

    def test_single_polygon_falls_back_to_bbox(self):
        """No gap exists — min_conductor_gap is None, bbox min wins."""
        component = _narrow_trace(width=3.0, length=50.0)
        stack = _conductor_stack()
        # No gap possible; falls back to bbox min = 3.0
        assert min_conductor_gap(component, stack) is None
        assert min_conductor_feature_size(component, stack) == pytest.approx(3.0)

    def test_far_apart_polygons_bbox_wins(self):
        """Gap larger than any bbox dim — bbox min wins."""
        component = _two_far_polys()
        stack = _conductor_stack()
        # Both polygons are 50um squares, separated by ~450um
        assert min_conductor_feature_size(component, stack) == pytest.approx(50.0)


class TestMinConductorGap:
    """Tests for min_conductor_gap."""

    def test_cpw_returns_gap(self):
        component = _cpw_component(s_width=20.0, gap=15.0)
        stack = _conductor_stack()
        assert min_conductor_gap(component, stack) == pytest.approx(15.0)

    def test_single_polygon_returns_none(self):
        component = _narrow_trace()
        stack = _conductor_stack()
        assert min_conductor_gap(component, stack) is None

    def test_no_conductor_layer_returns_none(self):
        component = _narrow_trace()
        stack = _conductor_stack(gds_layer=(99, 0))
        assert min_conductor_gap(component, stack) is None


class TestAutoRefinedMeshSize:
    """Tests for auto_refined_mesh_size."""

    def test_scales_down_for_small_features(self):
        component = _narrow_trace(width=2.0)
        stack = _conductor_stack()
        # min_feature=2um, cells_per_feature=4 -> 0.5um < preset 5.0
        assert auto_refined_mesh_size(component, stack, preset_size=5.0) == 0.5

    def test_caps_at_preset_for_large_features(self):
        component = _narrow_trace(width=100.0, length=100.0)
        stack = _conductor_stack()
        # min_feature=100um, /4 = 25um, but capped at preset 5.0
        assert auto_refined_mesh_size(component, stack, preset_size=5.0) == 5.0

    def test_falls_back_to_preset_when_no_conductors(self):
        component = _narrow_trace()
        empty_stack = LayerStack()
        assert auto_refined_mesh_size(component, empty_stack, preset_size=5.0) == 5.0

    def test_cpw_end_to_end(self):
        """CPW with 20 um trace + 15 um gap, preset=5.0 -> 15/4 = 3.75."""
        component = _cpw_component(s_width=20.0, gap=15.0, ground_width=40.0)
        stack = _conductor_stack()
        # min_feature = 15 (gap), /4 = 3.75 < preset 5.0
        assert auto_refined_mesh_size(
            component, stack, preset_size=5.0
        ) == pytest.approx(3.75)
