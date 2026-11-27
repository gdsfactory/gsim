"""The Palace route evaluates boolean (derived) stack levels like the FDTD route.

The generic gdsfactory PDK defines ``core = (WG - DEEP_ETCH) - SHALLOW_ETCH`` and
``shallow_etch = SHALLOW_ETCH & WG``. ``extract_geometry`` must hand the mesher
the evaluated region of such a level, one polygon set per stack level, and the
same area ``resolve_passive_pcell`` (the FDTD route) resolves.
"""

from __future__ import annotations

from types import SimpleNamespace

import gdsfactory as gf
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from gsim.common.materials import GSIM_MATERIAL_CARDS
from gsim.common.pdk import resolve_passive_pcell
from gsim.common.stack import Layer, LayerStack, get_stack
from gsim.palace._mesh_fingerprint import mesh_inputs_fingerprint
from gsim.palace.mesh.geometry import (
    add_metals,
    extract_geometry,
    get_layer_infos,
    polygon_keys,
)

BOOLEAN_LEVELS = ("core", "shallow_etch")
AREA_TOL = 1e-6  # um2


def _strip_with_bar(layer_name: str, bar_width: float, bar_height: float):
    """WG strip 10 x 0.5 um with a bar on *layer_name* across its middle."""
    gf.gpdk.PDK.activate()
    layer = gf.gpdk.LAYER
    component = gf.Component()
    component.add_polygon(
        [(0, -0.25), (10, -0.25), (10, 0.25), (0, 0.25)], layer=layer.WG
    )
    x0, y0 = 5 - bar_width / 2, -bar_height / 2
    component.add_polygon(
        [
            (x0, y0),
            (x0 + bar_width, y0),
            (x0 + bar_width, y0 + bar_height),
            (x0, y0 + bar_height),
        ],
        layer=getattr(layer, layer_name),
    )
    return component


def _shallow_strip():
    return _strip_with_bar("SHALLOW_ETCH", 2.0, 2.0)


def _deep_strip():
    return _strip_with_bar("DEEP_ETCH", 1.0, 2.5)


def _rib():
    gf.gpdk.PDK.activate()
    return gf.components.straight(cross_section="rib")


def _grating():
    gf.gpdk.PDK.activate()
    return gf.components.grating_coupler_elliptical_trenches()


COMPONENTS = {
    "shallow_strip": _shallow_strip,
    "deep_strip": _deep_strip,
    "rib": _rib,
    "grating_trenches": _grating,
}


@pytest.fixture
def stack() -> LayerStack:
    """The Palace stack of the generic PDK."""
    gf.gpdk.PDK.activate()
    return get_stack()


def _palace_areas(component, stack: LayerStack) -> dict[str, float]:
    """Area of the polygons ``extract_geometry`` assigns to each stack level."""
    by_key: dict = {}
    for key, xs, ys, holes in extract_geometry(component, stack).polygons:
        by_key.setdefault(key, []).append(
            Polygon(
                list(zip(xs, ys, strict=True)),
                [list(zip(hx, hy, strict=True)) for hx, hy in holes],
            )
        )
    areas: dict[str, float] = {}
    for key, polygons in by_key.items():
        for info in get_layer_infos(stack, key):
            areas[info["name"]] = unary_union(polygons).area
    return areas


def _fdtd_areas(component) -> dict[str, float]:
    """Area of every stack level the FDTD route resolves."""
    gf.gpdk.PDK.activate()
    module = SimpleNamespace(
        PDK=gf.gpdk.PDK,
        MATERIAL_CARDS={
            "si": GSIM_MATERIAL_CARDS["Si"].model_copy(update={"name": "si"}),
            "sio2": GSIM_MATERIAL_CARDS["SiO2"].model_copy(update={"name": "sio2"}),
        },
    )
    resolved = resolve_passive_pcell(component, pdk=module)
    return {key: layer.geometry.area for key, layer in resolved.layers.items()}


@pytest.mark.parametrize("case", COMPONENTS)
def test_every_level_has_the_fdtd_route_area(case: str, stack: LayerStack) -> None:
    """Boolean and plain levels alike: the mesher input equals the FDTD one."""
    component = COMPONENTS[case]()
    palace = _palace_areas(component, stack)
    fdtd = _fdtd_areas(component)
    for level in stack.layers:
        assert palace.get(level, 0.0) == pytest.approx(
            fdtd.get(level, 0.0), abs=AREA_TOL
        ), level


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        # strip 5 um2, etch bar overlap 1 um2
        ("shallow_strip", {"core": 4.0, "shallow_etch": 1.0}),
        # strip 5 um2, deep etch bar overlap 0.5 um2 (bar itself is 2.5 um2)
        ("deep_strip", {"core": 4.5, "deep_etch": 2.5}),
    ],
)
def test_synthetic_strips_match_hand_computed_areas(
    case: str, expected: dict[str, float], stack: LayerStack
) -> None:
    """The areas follow from the drawn rectangles, not from the evaluator."""
    palace = _palace_areas(COMPONENTS[case](), stack)
    for level, area in expected.items():
        assert palace[level] == pytest.approx(area, abs=AREA_TOL), level


def test_boolean_levels_of_the_grating_coupler_are_etched(stack: LayerStack) -> None:
    """The trenches are cut out of the core and fill the shallow etch level."""
    component = _grating()
    palace = _palace_areas(component, stack)
    fdtd = _fdtd_areas(component)
    assert palace["core"] < 400.0  # 453.4 um2 before: the unetched WG layer
    assert palace["shallow_etch"] > 0.0
    for level in BOOLEAN_LEVELS:
        assert palace[level] == pytest.approx(fdtd[level], abs=AREA_TOL)


def test_control_without_etch_layers_keeps_its_levels(stack: LayerStack) -> None:
    """A rib has nothing to subtract: core and slab are what is drawn."""
    palace = _palace_areas(_rib(), stack)
    fdtd = _fdtd_areas(_rib())
    assert palace["core"] == pytest.approx(5.0, abs=AREA_TOL)
    assert palace["core"] == pytest.approx(fdtd["core"], abs=AREA_TOL)
    assert palace["slab90"] == pytest.approx(fdtd["slab90"], abs=AREA_TOL)
    assert "shallow_etch" not in palace


def test_levels_sharing_a_layer_number_do_not_share_polygons(
    stack: LayerStack,
) -> None:
    """deep_etch (3, 6) and slab90 (3, 0) share GDS layer 3 but not polygons.

    Nothing is drawn on DEEP_ETCH in a rib, so it must have no polygons, where
    the slab on SLAB90 gives slab90 its own.
    """
    palace = _palace_areas(_rib(), stack)
    assert palace["slab90"] > 0.0
    assert "deep_etch" not in palace
    keys = polygon_keys(stack)
    assert keys["deep_etch"] != keys["slab90"]


def test_polygon_keys(stack: LayerStack) -> None:
    """Boolean levels are keyed by name; plain levels by number or tuple."""
    keys = polygon_keys(stack)
    assert keys["core"] == "core"
    assert keys["shallow_etch"] == "shallow_etch"
    assert keys["metal1"] == stack.layers["metal1"].gds_layer[0]  # no clash: number
    assert keys["deep_etch"] == (3, 6)  # shares number 3 with slab90
    assert keys["slab90"] == (3, 0)


def test_only_boolean_levels_remember_their_expression(stack: LayerStack) -> None:
    """``get_stack`` marks the levels whose layer is a boolean expression."""
    derived = {
        name for name, layer in stack.layers.items() if layer.derived_source is not None
    }
    assert derived == set(BOOLEAN_LEVELS)


def test_copied_stack_keeps_derived_levels(stack: LayerStack) -> None:
    """The simulation deep-copies its stack; the copy still evaluates the levels."""
    copied = stack.model_copy(deep=True)
    assert copied.layers["core"].derived_source is not None
    component = _shallow_strip()
    assert _palace_areas(component, copied) == _palace_areas(component, stack)


def test_stack_from_yaml_has_no_expressions(tmp_path) -> None:
    """A YAML stack keeps only GDS tuples: its levels stay plain layers."""
    from gsim.common.stack import load_stack_yaml

    path = tmp_path / "stack.yaml"
    path.write_text(
        "layers:\n"
        "  core:\n"
        "    gds_layer: [1, 0]\n"
        "    zmin: 0.0\n"
        "    zmax: 0.22\n"
        "    material: si\n"
        "    type: dielectric\n",
        encoding="utf-8",
    )
    loaded = load_stack_yaml(path)
    assert loaded.layers["core"].derived_source is None
    assert polygon_keys(loaded) == {"core": 1}


def test_fingerprint_tells_a_derived_level_from_a_plain_one(
    stack: LayerStack,
) -> None:
    """A mesh built from an expression is not reused for the plain layer."""

    def fingerprint(stack: LayerStack) -> dict[str, str]:
        return mesh_inputs_fingerprint(
            component=None,
            stack=stack,
            airbox_config={},
            pec_blocks=[],
            symmetry_planes=[],
            port_configs=[],
        )

    plain = stack.model_copy(deep=True)
    plain.layers["core"] = Layer(**plain.layers["core"].model_dump())
    assert plain.layers["core"].derived_source is None
    derived = fingerprint(stack)
    assert derived["layer:core"] != fingerprint(plain)["layer:core"]
    assert derived["layer:metal1"] == fingerprint(plain)["layer:metal1"]


def _two_datatypes_stack() -> LayerStack:
    """Two conductors on layer 7, datatypes 0 and 1, and one on layer 8."""
    return LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=gds,
                zmin=z,
                zmax=z + 1.0,
                thickness=1.0,
                material="aluminum",
                layer_type="conductor",
            )
            for name, gds, z in (
                ("a", (7, 0), 0.0),
                ("b", (7, 1), 2.0),
                ("c", (8, 0), 4.0),
            )
        }
    )


def test_datatypes_of_one_layer_number_get_their_own_polygons() -> None:
    """Two levels on (7, 0) and (7, 1) each mesh the polygons of their own tuple."""
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon([(0, 0), (4, 0), (4, 1), (0, 1)], layer=(7, 0))
    component.add_polygon([(0, 5), (2, 5), (2, 6), (0, 6)], layer=(7, 1))
    component.add_polygon([(0, 8), (3, 8), (3, 9), (0, 9)], layer=(8, 0))
    stack = _two_datatypes_stack()
    areas = _palace_areas(component, stack)
    assert areas == pytest.approx({"a": 4.0, "b": 2.0, "c": 3.0})
    keys = polygon_keys(stack)
    assert keys == {"a": (7, 0), "b": (7, 1), "c": 8}


def test_boolean_levels_are_meshed_as_their_evaluated_region(stack: LayerStack) -> None:
    """add_metals extrudes the etched core and the shallow etch separately."""
    import gmsh

    component = _shallow_strip()
    geometry = extract_geometry(component, stack)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("derived")
        kernel = gmsh.model.occ
        metal_tags = add_metals(kernel, geometry, stack, planar_conductors=True)
        kernel.synchronize()

        def volume(name: str) -> float:
            return sum(kernel.getMass(3, tag) for tag in metal_tags[name]["volumes"])

        core = stack.layers["core"]
        etch = stack.layers["shallow_etch"]
        assert volume("core") == pytest.approx(4.0 * core.thickness, rel=1e-6)
        assert volume("shallow_etch") == pytest.approx(1.0 * etch.thickness, rel=1e-6)
    finally:
        gmsh.clear()
        gmsh.finalize()


def test_symmetry_plane_clips_boolean_levels(stack: LayerStack) -> None:
    """A plane through the strip halves the evaluated core and shallow etch."""
    from gsim.palace.models.symmetry import SymmetryPlaneConfig

    plane = SymmetryPlaneConfig(axis="y", position=0.0)
    geometry = extract_geometry(_shallow_strip(), stack, symmetry_plane=plane)
    by_key: dict = {}
    for key, xs, ys, holes in geometry.polygons:
        by_key.setdefault(key, []).append(
            Polygon(
                list(zip(xs, ys, strict=True)),
                [list(zip(hx, hy, strict=True)) for hx, hy in holes],
            )
        )
    assert unary_union(by_key["core"]).area == pytest.approx(2.0, abs=AREA_TOL)
    assert unary_union(by_key["shallow_etch"]).area == pytest.approx(0.5, abs=AREA_TOL)


def test_bbox_still_spans_the_drawn_layout(stack: LayerStack) -> None:
    """The domain size follows the drawn WG layer, not the etched core.

    The shallow etch bar covers the last um of the strip, so the evaluated core
    ends at x = 9 um while the strip is drawn up to x = 10 um.
    """
    gf.gpdk.PDK.activate()
    covered = gf.Component()
    covered.add_polygon(
        [(0, -0.25), (10, -0.25), (10, 0.25), (0, 0.25)], layer=gf.gpdk.LAYER.WG
    )
    covered.add_polygon(
        [(9, -1), (11, -1), (11, 1), (9, 1)], layer=gf.gpdk.LAYER.SHALLOW_ETCH
    )
    areas = _palace_areas(covered, stack)
    assert areas["core"] == pytest.approx(4.5, abs=AREA_TOL)  # x from 0 to 9
    xmin, ymin, xmax, ymax = extract_geometry(covered, stack).bbox
    assert (xmin, xmax) == pytest.approx((0.0, 10.0))
    assert (ymin, ymax) == pytest.approx((-0.25, 0.25))
