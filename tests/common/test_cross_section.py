"""Tests for gsim.common.cross_section."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Literal

import pytest

from gsim.common.cross_section import (
    PolygonXY2D,
    Rect2D,
    RectYZ2D,
    extract_plane_section,
    extract_xy_polygons,
    extract_xz_rectangles,
    extract_yz_rectangles,
)


def _layer(
    name: str,
    gds_layer: tuple[int, int],
    zmin: float,
    zmax: float,
    material: str,
    layer_type: Literal["conductor", "via", "dielectric", "substrate"] = "dielectric",
):
    """Build a Layer with the minimum args the real model requires."""
    from gsim.common.stack import Layer

    return Layer(
        name=name,
        gds_layer=gds_layer,
        zmin=zmin,
        zmax=zmax,
        thickness=zmax - zmin,
        material=material,
        layer_type=layer_type,
    )


def _stack(layers_list):
    """Build a LayerStack from a list of Layer objects."""
    from gsim.common.stack import LayerStack

    return LayerStack(
        pdk_name="test",
        units="um",
        layers={layer.name: layer for layer in layers_list},
        materials={},
        dielectrics=[],
        simulation={},
    )


class TestRect2D:
    """Tests for the Rect2D dataclass."""

    def test_frozen_dataclass_equal_by_value(self):
        a = Rect2D(
            x0=0.0, x1=1.0, zmin=-0.1, zmax=0.1, layer_name="core", material="si"
        )
        b = Rect2D(
            x0=0.0, x1=1.0, zmin=-0.1, zmax=0.1, layer_name="core", material="si"
        )
        assert a == b
        assert hash(a) == hash(b)

    def test_frozen_cannot_mutate(self):
        r = Rect2D(x0=0.0, x1=1.0, zmin=0.0, zmax=0.1, layer_name="core", material="si")
        with pytest.raises(FrozenInstanceError):
            r.x0 = 5.0  # ty: ignore[invalid-assignment]


class TestSimpleWaveguide:
    """Single-layer strip waveguide on the core layer."""

    def _build_stack(self):
        """Return a single-layer stack with a silicon core from 0 to 0.22 um."""
        return _stack(
            [_layer("core", (1, 0), 0.0, 0.22, "si", layer_type="dielectric")]
        )

    def _build_straight(self):
        """Build a simple straight waveguide polygon on layer (1, 0)."""
        import gdsfactory as gf

        c = gf.Component()
        c.add_polygon(
            [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
            layer=(1, 0),
        )
        return c

    def test_cut_through_center(self):
        c = self._build_straight()
        stack = self._build_stack()

        rects = extract_xz_rectangles(c, stack, y_cut=0.0)

        assert len(rects) == 1
        r = rects[0]
        assert r.layer_name == "core"
        assert r.material == "si"
        assert r.zmin == pytest.approx(0.0)
        assert r.zmax == pytest.approx(0.22)
        assert r.x0 == pytest.approx(-5.0)
        assert r.x1 == pytest.approx(5.0)

    def test_cut_misses_waveguide(self):
        c = self._build_straight()
        stack = self._build_stack()
        rects = extract_xz_rectangles(c, stack, y_cut=10.0)
        assert rects == []


class TestPartialEtch:
    """Two-layer strip + slab: core rectangle on top of a wider slab."""

    def _build_stack(self):
        """Return a two-layer stack with a slab and a core."""
        return _stack(
            [
                _layer("slab", (2, 0), 0.0, 0.09, "si", layer_type="dielectric"),
                _layer("core", (1, 0), 0.0, 0.22, "si", layer_type="dielectric"),
            ]
        )

    def _build_component(self):
        """Build a narrow core strip on top of a wider slab layer."""
        import gdsfactory as gf

        c = gf.Component()
        # Core strip: narrow, centered on y=0
        c.add_polygon(
            [(-3, -0.25), (3, -0.25), (3, 0.25), (-3, 0.25)],
            layer=(1, 0),
        )
        # Slab layer: wider, full extent
        c.add_polygon(
            [(-3, -1.5), (3, -1.5), (3, 1.5), (-3, 1.5)],
            layer=(2, 0),
        )
        return c

    def test_cut_through_both_layers(self):
        c = self._build_component()
        stack = self._build_stack()
        rects = extract_xz_rectangles(c, stack, y_cut=0.0)

        layers = {r.layer_name for r in rects}
        assert layers == {"slab", "core"}

        core = next(r for r in rects if r.layer_name == "core")
        slab = next(r for r in rects if r.layer_name == "slab")

        assert core.zmin == pytest.approx(0.0)
        assert core.zmax == pytest.approx(0.22)
        assert slab.zmin == pytest.approx(0.0)
        assert slab.zmax == pytest.approx(0.09)
        # Core extent narrower than slab extent at y=0:
        assert (core.x1 - core.x0) <= (slab.x1 - slab.x0) + 1e-6

    def test_cut_through_slab_only(self):
        c = self._build_component()
        stack = self._build_stack()
        rects = extract_xz_rectangles(c, stack, y_cut=1.0)

        layers = {r.layer_name for r in rects}
        assert layers == {"slab"}  # core polygon does not extend to y=1.0


class TestPolygonWithHole:
    """Donut polygon: outer ring with interior hole."""

    def _build_stack(self):
        """Return a single-layer stack with a silicon core."""
        return _stack(
            [_layer("core", (1, 0), 0.0, 0.22, "si", layer_type="dielectric")]
        )

    def _build_donut(self):
        """Build a rectangular donut (outer box with a rectangular hole)."""
        import gdsfactory as gf

        outer = gf.Component()
        outer.add_polygon(
            [(-5, -1), (5, -1), (5, 1), (-5, 1)],
            layer=(1, 0),
        )
        inner = gf.Component()
        inner.add_polygon(
            [(-2, -0.5), (2, -0.5), (2, 0.5), (-2, 0.5)],
            layer=(1, 0),
        )
        return gf.boolean(outer, inner, operation="not", layer=(1, 0))

    def test_cut_through_hole_splits_into_two_intervals(self):
        c = self._build_donut()
        stack = self._build_stack()
        rects = extract_xz_rectangles(c, stack, y_cut=0.0)

        core_rects = sorted(
            (r for r in rects if r.layer_name == "core"),
            key=lambda r: r.x0,
        )
        assert len(core_rects) == 2
        assert core_rects[0].x0 == pytest.approx(-5.0)
        assert core_rects[0].x1 == pytest.approx(-2.0)
        assert core_rects[1].x0 == pytest.approx(2.0)
        assert core_rects[1].x1 == pytest.approx(5.0)


class TestEdgeCaseCut:
    """Cut line exactly on a polygon edge should not crash."""

    def _build_stack(self):
        """Return a single-layer stack with a silicon core."""
        return _stack(
            [_layer("core", (1, 0), 0.0, 0.22, "si", layer_type="dielectric")]
        )

    def test_cut_on_edge(self):
        import gdsfactory as gf

        c = gf.Component()
        c.add_polygon(
            [(-5, 0.0), (5, 0.0), (5, 1.0), (-5, 1.0)],
            layer=(1, 0),
        )
        stack = self._build_stack()

        rects = extract_xz_rectangles(c, stack, y_cut=0.0)

        core_rects = [r for r in rects if r.layer_name == "core"]
        assert len(core_rects) == 1
        assert core_rects[0].x0 == pytest.approx(-5.0)
        assert core_rects[0].x1 == pytest.approx(5.0)


class TestGeneralizedPlaneExtraction:
    """Tests for axis-generalized cross-section extraction helpers."""

    def _build_stack(self):
        return _stack(
            [_layer("core", (1, 0), 0.0, 0.22, "si", layer_type="dielectric")]
        )

    def _build_rect_component(self):
        import gdsfactory as gf

        c = gf.Component()
        c.add_polygon(
            [(-5, -1), (5, -1), (5, 1), (-5, 1)],
            layer=(1, 0),
        )
        return c

    def test_extract_yz_rectangles(self):
        c = self._build_rect_component()
        stack = self._build_stack()

        rects = extract_yz_rectangles(c, stack, x_cut=0.0)

        assert len(rects) == 1
        r = rects[0]
        assert isinstance(r, RectYZ2D)
        assert r.layer_name == "core"
        assert r.material == "si"
        assert r.y0 == pytest.approx(-1.0)
        assert r.y1 == pytest.approx(1.0)
        assert r.zmin == pytest.approx(0.0)
        assert r.zmax == pytest.approx(0.22)

    def test_extract_xy_polygons(self):
        c = self._build_rect_component()
        stack = self._build_stack()

        polys = extract_xy_polygons(c, stack, z_cut=0.1)

        assert len(polys) == 1
        p = polys[0]
        assert isinstance(p, PolygonXY2D)
        assert p.layer_name == "core"
        assert p.material == "si"
        xs = [pt[0] for pt in p.exterior]
        ys = [pt[1] for pt in p.exterior]
        assert min(xs) == pytest.approx(-5.0)
        assert max(xs) == pytest.approx(5.0)
        assert min(ys) == pytest.approx(-1.0)
        assert max(ys) == pytest.approx(1.0)

    def test_extract_plane_section_dispatch(self):
        c = self._build_rect_component()
        stack = self._build_stack()

        y_rects = extract_plane_section(c, stack, axis="y", value=0.0)
        x_rects = extract_plane_section(c, stack, axis="x", value=0.0)
        z_polys = extract_plane_section(c, stack, axis="z", value=0.1)

        assert len(y_rects) == 1
        assert isinstance(y_rects[0], Rect2D)

        assert len(x_rects) == 1
        assert isinstance(x_rects[0], RectYZ2D)

        assert len(z_polys) == 1
        assert isinstance(z_polys[0], PolygonXY2D)
