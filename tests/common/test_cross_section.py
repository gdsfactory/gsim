"""Tests for gsim.common.cross_section and doping-profile helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Literal

import gdsfactory as gf
import pytest

from gsim.common.cross_section import (
    PolygonXY2D,
    Rect2D,
    RectYZ2D,
    build_doped_cross_section,
    extract_plane_section,
    extract_xy_polygons,
    extract_xz_rectangles,
    extract_yz_rectangles,
    parse_plane_spec,
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


class TestParsePlaneSpec:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("x=0", ("x", 0.0)),
            (" Y = -1.5 ", ("y", -1.5)),
            ("z=1e-3", ("z", 1e-3)),
            ("x=.5", ("x", 0.5)),
        ],
    )
    def test_it_reads_the_axis_and_the_coordinate(self, spec, expected):
        assert parse_plane_spec(spec) == expected

    @pytest.mark.parametrize("spec", ["x=", "w=0", "0", "x 0", "x=abc", ""])
    def test_a_spec_naming_no_plane_is_rejected(self, spec):
        with pytest.raises(ValueError, match="Invalid plane spec"):
            parse_plane_spec(spec)


class TestPolygonWithHole:
    """Donut polygon: outer ring with interior hole."""

    def _build_stack(self):
        """Return a single-layer stack with a silicon core."""
        return _stack(
            [_layer("core", (1, 0), 0.0, 0.22, "si", layer_type="dielectric")]
        )

    def _build_donut(self):
        """Build a rectangular donut (outer box with a rectangular hole)."""

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


def _centered_rect(wx: float, wy: float, layer: tuple[int, int]):
    r = gf.Component()
    r << gf.c.rectangle((wx, wy), centered=True, layer=layer)
    return r


class TestBuildDopedCrossSection:
    """Tests for build_doped_cross_section() using the active PDK stack."""

    def _build_component(self):
        LAYER = gf.gpdk.LAYER
        comp = gf.Component()
        wg = comp << _centered_rect(10.0, 0.4, LAYER.WG)
        wg.y = -20.0
        return comp, LAYER

    def _doping_result(self, comp, rib_center_y=-20.0):
        """Two doped slab Regions beside the rib, drawn by hand."""
        from gsim.common.stack.extractor import Layer
        from gsim.common.stack.materials import make_doped_materials

        sides = {
            "pp_slab_0": ((23, 0), (rib_center_y + 0.2, rib_center_y + 2.2)),
            "npp_slab_0": ((24, 0), (rib_center_y - 2.2, rib_center_y - 0.2)),
        }
        layer_specs = {}
        for name, (gds_layer, (y0, y1)) in sides.items():
            comp.add_polygon(
                [(-5.0, y0), (5.0, y0), (5.0, y1), (-5.0, y1)], layer=gds_layer
            )
            layer_specs[name] = Layer(
                name=name,
                gds_layer=gds_layer,
                zmin=0.0,
                zmax=0.09,
                thickness=0.09,
                material=name,
                layer_type="dielectric",
                mesh_resolution="fine",
            )
        materials = make_doped_materials(
            [(name, 2e4) for name in sides], permittivity=11.9
        )
        return {"layer_specs": layer_specs, "materials": materials}

    def test_builds_stack_with_doping_and_rib_layers(self):
        comp, LAYER = self._build_component()
        doping = self._doping_result(comp)

        stack, section = build_doped_cross_section(
            comp,
            axis="x",
            value=0.0,
            substrate_thickness=2.0,
            include_substrate=False,
            doping=doping,
            metal1=(1.1, 1.0),
            rib_layers=[
                ("p_rib", LAYER.P, 1.6e3),
                ("n_rib", LAYER.N, 1.6e3),
            ],
            permittivity=11.9,
            fmax=200e9,
            verbose=False,
        )

        assert "pp_slab_0" in stack.layers
        assert "npp_slab_0" in stack.layers
        assert "p_rib" in stack.layers
        assert "n_rib" in stack.layers

        p_rib = stack.layers["p_rib"]
        assert p_rib.gds_layer == tuple(LAYER.P)
        assert p_rib.zmax == pytest.approx(0.22)
        assert p_rib.thickness == pytest.approx(0.22)
        assert p_rib.material == "p_rib"

        layers = {r.layer_name for r in section}
        assert {"core", "pp_slab_0", "npp_slab_0"} <= layers

    def test_doping_materials_registered_on_stack(self):
        """Doping/rib materials must land on stack.materials for solver config.

        Regression: the merged materials dict used to be computed but never
        attached, so doped domains silently resolved to eps=1.0 without
        conductivity in the generated Palace config.
        """
        comp, LAYER = self._build_component()
        doping = self._doping_result(comp)

        stack, _section = build_doped_cross_section(
            comp,
            axis="x",
            value=0.0,
            substrate_thickness=2.0,
            include_substrate=False,
            doping=doping,
            metal1=(1.1, 1.0),
            rib_layers=[
                ("p_rib", LAYER.P, 1.6e3),
                ("n_rib", LAYER.N, 1.6e3),
            ],
            permittivity=11.9,
            fmax=200e9,
            verbose=False,
        )

        for name, sigma in (
            ("pp_slab_0", 2e4),
            ("p_rib", 1.6e3),
            ("n_rib", 1.6e3),
        ):
            assert name in stack.materials, f"{name} missing from stack.materials"
            props = stack.materials[name]
            assert isinstance(props, dict)
            assert props["permittivity"] == pytest.approx(11.9)
            assert props["conductivity"] == pytest.approx(sigma)

    def test_metal1_override_applied(self):
        comp, _LAYER = self._build_component()
        stack, _ = build_doped_cross_section(
            comp,
            axis="x",
            value=0.0,
            substrate_thickness=2.0,
            metal1=(1.1, 1.0),
            verbose=False,
        )
        m1 = stack.layers["metal1"]
        assert m1.zmin == pytest.approx(1.1)
        assert m1.zmax == pytest.approx(2.1)
        assert m1.thickness == pytest.approx(1.0)

    def test_no_doping_no_rib_layers(self):
        comp, _LAYER = self._build_component()
        stack, _ = build_doped_cross_section(
            comp,
            axis="x",
            value=0.0,
            substrate_thickness=2.0,
            verbose=False,
        )
        assert "p_rib" not in stack.layers
        assert "pp_slab_0" not in stack.layers
        assert "core" in stack.layers
