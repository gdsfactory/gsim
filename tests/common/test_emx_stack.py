"""Tests for the EMX ``.proc`` importer (``gsim.common.stack.emx``).

All expected numbers are computed by hand from ``data/generic_4lm.proc``, a
synthetic file with invented values:

* z = 0 is the top of the substrate, layers are listed bottom to top.
* ``M1``: thickness 0.2 * 2 = 0.4, offset 0.2 inside ``ild0`` (z 0..1), sheet
  resistance 0.1 ohm/sq -> sigma = 1 / (0.1 * 0.4e-6) = 2.5e7 S/m.
* ``M2``: thickness 1.0 at the bottom of ``ild1`` (z 1..2), 3.5e7 S/m.
* ``M3``: thickness 1.5, offset 0.5 inside ``ild2`` (z 2..4), first table entry
  0.02 ohm/sq -> sigma = 1 / (0.02 * 1.5e-6) = 3.333e7 S/m.
* Substrate: 100 um, 10 ohm-cm -> 100 / 10 = 10 S/m.
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import gdsfactory as gf
import pytest

from gsim.common import LayerStack
from gsim.common.stack import get_material_properties
from gsim.common.stack.emx import (
    PORTABLE_SCHEMA,
    EmxImportWarning,
    load_emx_proc,
    load_portable_stackup_json,
    parse_emx_proc,
)

DATA = Path(__file__).parent / "data" / "generic_4lm.proc"

# A minimal, valid process used to probe single features.
BASE = """\
layer 10 11.9 name sub 20 ohm-cm
layer 2.0 4.0 name ox
conductor 1.0 {conductor} MA
define MA = l1t0
"""


def _write(tmp_path: Path, text: str, name: str = "case.proc") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _load_quiet(path: Path, **kwargs) -> LayerStack:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EmxImportWarning)
        return load_emx_proc(path, **kwargs)


@pytest.fixture(scope="module")
def loaded() -> tuple[LayerStack, list[str]]:
    """The synthetic stack together with the warning texts it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stack = load_emx_proc(DATA)
    texts = [str(w.message) for w in caught if w.category is EmxImportWarning]
    return stack, texts


@pytest.fixture(scope="module")
def stack(loaded) -> LayerStack:
    return loaded[0]


# ---------------------------------------------------------------------------
# Stack geometry and materials
# ---------------------------------------------------------------------------


def test_returns_layer_stack(stack):
    assert isinstance(stack, LayerStack)
    assert stack.pdk_name == "generic_4lm"
    assert stack.units == "um"


def test_dielectric_regions(stack):
    expected = [
        ("substrate", -100.0, 0.0, 11.9),
        ("ild0", 0.0, 1.0, 4.2),
        ("ild1", 1.0, 2.0, 3.9),
        ("ild2", 2.0, 4.0, 3.6),
        ("passivation", 4.0, 5.5, 6.5),
    ]
    assert len(stack.dielectrics) == len(expected)
    for region, (name, zmin, zmax, eps) in zip(
        stack.dielectrics, expected, strict=True
    ):
        assert region["name"] == name
        assert region["zmin"] == pytest.approx(zmin, abs=1e-12)
        assert region["zmax"] == pytest.approx(zmax, abs=1e-12)
        assert stack.materials[region["material"]]["permittivity"] == eps


def test_dielectrics_are_contiguous(stack):
    ordered = sorted(stack.dielectrics, key=lambda d: d["zmin"])
    for lower, upper in itertools.pairwise(ordered):
        assert lower["zmax"] == pytest.approx(upper["zmin"], abs=1e-12)
    assert stack.get_z_range() == pytest.approx((-100.0, 5.5))


def test_substrate_is_lossy(stack):
    material = stack.materials[stack.dielectrics[0]["material"]]
    assert material["type"] == "semiconductor"
    assert material["conductivity"] == pytest.approx(10.0)
    assert material["permittivity"] == 11.9


def test_air_layer_is_dropped(stack):
    assert all(
        "air" not in d["name"].lower()
        and stack.materials[d["material"]]["permittivity"] != 1.0
        for d in stack.dielectrics
    )
    assert stack.simulation["emx"]["top_of_stack_um"] == pytest.approx(5.5)


@pytest.mark.parametrize(
    ("name", "zmin", "zmax"),
    [("M1", 0.2, 0.6), ("M2", 1.0, 2.0), ("M3", 2.5, 4.0)],
)
def test_conductor_z_positions(stack, name, zmin, zmax):
    layer = stack.layers[name]
    assert layer.layer_type == "conductor"
    assert layer.zmin == pytest.approx(zmin, abs=1e-12)
    assert layer.zmax == pytest.approx(zmax, abs=1e-12)
    assert layer.thickness == pytest.approx(zmax - zmin, abs=1e-12)


@pytest.mark.parametrize(
    ("name", "sigma"),
    [
        ("M1", 1 / (0.1 * 0.4e-6)),  # sheet resistance 0.1 ohm/sq, t = 0.4 um
        ("M2", 3.5e7),  # given directly in S/m
        ("M3", 1 / (0.02 * 1.5e-6)),  # first table entry, t = 1.5 um
    ],
)
def test_conductor_conductivity(stack, name, sigma):
    material = stack.materials[stack.layers[name].material]
    assert material["type"] == "conductor"
    assert material["conductivity"] == pytest.approx(sigma, rel=1e-12)


def test_sheet_resistance_is_kept_as_information(stack):
    assert (
        stack.materials[stack.layers["M1"].material]["sheet_resistance_ohm_per_sq"]
        == 0.1
    )
    assert (
        "sheet_resistance_ohm_per_sq"
        not in stack.materials[stack.layers["M2"].material]
    )


@pytest.mark.parametrize(
    ("name", "gds"),
    [
        ("M1", (11, 0)),
        ("M2", (12, 0)),  # optional fill stream (12, 5) is not used
        ("M3", (13, 0)),  # defined through an alias
        ("V1", (21, 0)),
        ("V2", (22, 0)),
    ],
)
def test_gds_layers(stack, name, gds):
    assert stack.layers[name].gds_layer == gds


def test_unused_stream_creates_no_layer(stack):
    assert set(stack.layers) == {"M1", "M2", "M3", "V1", "V2"}
    assert stack.simulation["emx"]["gds_streams"]["MARKER"] == [[99, 0]]


@pytest.mark.parametrize(
    ("name", "zmin", "zmax", "sigma", "ends"),
    [
        ("V1", 0.6, 1.0, 1.2e6, ["M1", "M2"]),
        ("V2", 2.0, 2.5, 8.0e6, ["M2", "M3"]),
    ],
)
def test_vias(stack, name, zmin, zmax, sigma, ends):
    via = stack.layers[name]
    assert via.layer_type == "via"
    assert via.zmin == pytest.approx(zmin, abs=1e-12)
    assert via.zmax == pytest.approx(zmax, abs=1e-12)
    assert stack.materials[via.material]["conductivity"] == pytest.approx(sigma)
    assert stack.simulation["emx"]["via_connectivity"][name] == ends
    # A via touches both conductors it connects.
    lower, upper = (stack.layers[end] for end in ends)
    assert via.zmin == pytest.approx(lower.zmax, abs=1e-12)
    assert via.zmax == pytest.approx(upper.zmin, abs=1e-12)


def test_stack_validates(stack):
    result = stack.validate_stack()
    assert result.valid, str(result)


def test_materials_do_not_collide_with_database(stack):
    """A name known to the material database would silently replace our values."""
    assert stack.materials
    for name in stack.materials:
        assert name.startswith("emx_")
        assert get_material_properties(name) is None


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def test_warns_about_ignored_features(loaded):
    texts = " | ".join(loaded[1])
    assert "temperature dependence" in texts
    assert "fill and slotting" in texts
    assert "table" in texts
    assert "nominal part '3.5e7'" in texts
    assert "bias" not in texts
    assert "merge" not in texts


def test_warnings_are_stored_in_metadata(stack, loaded):
    assert stack.simulation["emx"]["warnings"] == loaded[1]
    assert stack.simulation["emx"]["detected_in_source"] == [
        "fill and slotting",
        "temperature dependence",
    ]


def test_bias_and_merge_are_reported(tmp_path):
    text = BASE.format(conductor="3e7") + "bias MA 0.01\nmerge_vias on\n"
    path = _write(tmp_path, text)
    with pytest.warns(EmxImportWarning) as record:
        load_emx_proc(path)
    joined = " | ".join(str(w.message) for w in record)
    assert "geometry bias" in joined
    assert "via merge operations" in joined


def test_clean_file_has_no_warning(tmp_path):
    path = _write(tmp_path, BASE.format(conductor="3e7 S/m"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stack = load_emx_proc(path)
    assert stack.simulation["emx"]["warnings"] == []


# ---------------------------------------------------------------------------
# Substrate override
# ---------------------------------------------------------------------------


def test_substrate_override_moves_only_the_backside():
    base = _load_quiet(DATA)
    thick = _load_quiet(DATA, substrate_thickness_um=300)
    assert thick.dielectrics[0]["zmin"] == pytest.approx(-300.0)
    assert thick.dielectrics[0]["zmax"] == 0.0
    assert thick.simulation["emx"]["substrate_thickness_um"] == pytest.approx(300.0)
    for key in ("ild0", "ild1", "ild2", "passivation"):
        a = next(d for d in base.dielectrics if d["name"] == key)
        b = next(d for d in thick.dielectrics if d["name"] == key)
        assert (a["zmin"], a["zmax"]) == (b["zmin"], b["zmax"])
    for name, layer in base.layers.items():
        other = thick.layers[name]
        assert (layer.zmin, layer.zmax) == (other.zmin, other.zmax)
    # The substrate material is unchanged by the override.
    assert thick.materials["emx_substrate"] == base.materials["emx_substrate"]


@pytest.mark.parametrize("value", [0, -5, "abc"])
def test_substrate_override_must_be_positive_number(value):
    with pytest.raises(ValueError, match=r"substrate_thickness_um|Invalid number"):
        parse_emx_proc(DATA, substrate_thickness_um=value)


def test_override_without_substrate_warns(tmp_path):
    path = _write(
        tmp_path, "layer 2.0 4.0 name ox\nconductor 1.0 3e7 S/m MA\ndefine MA = l1t0\n"
    )
    with pytest.warns(EmxImportWarning, match="no substrate"):
        stack = load_emx_proc(path, substrate_thickness_um=50)
    assert stack.dielectrics[0]["zmin"] == 0.0


def test_infinite_substrate_needs_a_thickness(tmp_path):
    text = "layer infinity 11.9 name sub 10 ohm-cm\nlayer 2.0 4.0 name ox\n"
    path = _write(tmp_path, text)
    with pytest.raises(ValueError, match="infinite thickness"):
        parse_emx_proc(path)
    stack = _load_quiet(path, substrate_thickness_um=40)
    assert stack.dielectrics[0]["zmin"] == pytest.approx(-40.0)


# ---------------------------------------------------------------------------
# Parsing details
# ---------------------------------------------------------------------------


def test_sheet_resistance_conversion_from_a_number(tmp_path):
    # Rs = 0.05 ohm/sq, t = 2 um -> sigma = 1 / (0.05 * 2e-6) = 1e7 S/m
    text = BASE.format(conductor="0.05").replace("conductor 1.0", "conductor 2.0")
    stack = _load_quiet(_write(tmp_path, text))
    assert stack.materials[stack.layers["MA"].material][
        "conductivity"
    ] == pytest.approx(1e7, rel=1e-12)


def test_sheet_resistance_needs_thickness(tmp_path):
    text = BASE.format(conductor="0.05").replace("conductor 1.0", "conductor 0")
    with pytest.raises(ValueError, match="zero thickness"):
        parse_emx_proc(_write(tmp_path, text))


def test_zero_thickness_conductor_with_conductivity(tmp_path):
    text = BASE.format(conductor="5.8e7 S/m").replace("conductor 1.0", "conductor 0")
    stack = _load_quiet(_write(tmp_path, text))
    layer = stack.layers["MA"]
    assert layer.zmin == layer.zmax == pytest.approx(0.0)
    assert layer.thickness == 0.0


def test_conductor_without_preceding_layer_starts_at_zero(tmp_path):
    text = "conductor 1.0 3e7 S/m MA\nlayer 2.0 4.0 name ox\ndefine MA = l1t0\n"
    stack = _load_quiet(_write(tmp_path, text))
    assert stack.layers["MA"].zmin == 0.0


def test_negative_offset(tmp_path):
    text = (
        "layer 10 11.9 name sub 20 ohm-cm\nlayer 2.0 4.0 name ox\n"
        "offset -0.5\nconductor 1.0 3e7 S/m MA\ndefine MA = l1t0\n"
    )
    stack = _load_quiet(_write(tmp_path, text))
    assert stack.layers["MA"].zmin == pytest.approx(-0.5)
    assert stack.layers["MA"].zmax == pytest.approx(0.5)


def test_defines_with_arithmetic(tmp_path):
    text = (
        "define t_ox = 1.5 + 0.5\ndefine er_ox = 8 / 2\n"
        "layer 10 11.9 name sub 20 ohm-cm\nlayer t_ox er_ox name ox\n"
    )
    stack = _load_quiet(_write(tmp_path, text))
    ox = next(d for d in stack.dielectrics if d["name"] == "ox")
    assert (ox["zmin"], ox["zmax"]) == (0.0, 2.0)
    assert stack.materials[ox["material"]]["permittivity"] == 4.0


def test_integer_permittivity_name_is_not_truncated(tmp_path):
    text = (
        "layer 10 11.9 name sub 20 ohm-cm\n"
        "layer 1.0 10 name hi\n"
        "layer 1.0 10.5 name hi2\n"
    )
    stack = _load_quiet(_write(tmp_path, text))
    names = {d["material"] for d in stack.dielectrics}
    assert "emx_dielectric_er_10" in names
    assert "emx_dielectric_er_10p5" in names


def test_layer_name_does_not_enter_the_resistivity(tmp_path):
    """A define that shares the layer name must not multiply the resistivity."""
    text = "define sub = 7\nlayer 10 11.9 name sub 20 ohm-cm\n"
    stack = _load_quiet(_write(tmp_path, text))
    assert stack.materials["emx_substrate"]["conductivity"] == pytest.approx(5.0)


def test_comments_and_case_are_handled(tmp_path):
    text = (
        "# a comment\nLAYER 10 11.9 NAME sub 20 OHM-CM   # trailing comment\n"
        "Layer 2.0 4.0 name ox\nCONDUCTOR 1.0 3e7 S/M ma\nDefine MA = l1t0\n"
    )
    stack = _load_quiet(_write(tmp_path, text))
    assert "MA" in stack.layers


def test_duplicate_via_names_get_unique_names(tmp_path):
    text = """\
layer 10 11.9 name sub 20 ohm-cm
layer 1.0 4.0 name o1
conductor 0.2 3e7 S/m MA
layer 1.0 4.0 name o2
conductor 0.2 3e7 S/m MB
layer 1.0 4.0 name o3
conductor 0.2 3e7 S/m MC
define MA = l1t0
define MB = l2t0
define MC = l3t0
define VX = l10t0
define VX_MB_MC = l11t0
via MA MB { 1e6 S/m } VX
via MB MC { 1e6 S/m } VX
"""
    stack = _load_quiet(_write(tmp_path, text))
    assert stack.layers["VX"].gds_layer == (10, 0)
    assert stack.layers["VX_MB_MC"].gds_layer == (11, 0)
    assert stack.simulation["emx"]["via_connectivity"] == {
        "VX": ["MA", "MB"],
        "VX_MB_MC": ["MB", "MC"],
    }


def test_conductor_without_gds_stream_is_left_out(tmp_path):
    text = BASE.format(conductor="3e7 S/m").replace(
        "define MA = l1t0", "define MA = M0 * M9"
    )
    with pytest.warns(EmxImportWarning, match="no direct GDS stream"):
        stack = load_emx_proc(_write(tmp_path, text))
    assert "MA" not in stack.layers
    assert stack.dielectrics  # the dielectric stack is unaffected


def test_via_between_touching_conductors_is_left_out(tmp_path):
    text = """\
layer 10 11.9 name sub 20 ohm-cm
layer 2.0 4.0 name ox
conductor 1.0 3e7 S/m MA
conductor 1.0 3e7 S/m MB
define MA = l1t0
define MB = l2t0
define VA = l3t0
via MA MB { 1e6 S/m } VA
"""
    with pytest.warns(EmxImportWarning, match="touch or overlap"):
        stack = load_emx_proc(_write(tmp_path, text))
    assert "VA" not in stack.layers
    assert stack.simulation["emx"]["via_connectivity"]["VA"] == ["MA", "MB"]


def test_shared_gds_layer_number_is_reported(tmp_path):
    text = BASE.format(conductor="3e7 S/m") + (
        "conductor 1.0 3e7 S/m MB\ndefine MB = l1t7\n"
    )
    with pytest.warns(EmxImportWarning, match="share GDS layer number 1"):
        load_emx_proc(_write(tmp_path, text))


def test_several_streams_use_the_first(tmp_path):
    text = BASE.format(conductor="3e7 S/m").replace(
        "define MA = l1t0", "define MA = l1t0 + l1t1"
    )
    with pytest.warns(EmxImportWarning, match="several GDS streams"):
        stack = load_emx_proc(_write(tmp_path, text))
    assert stack.layers["MA"].gds_layer == (1, 0)


def test_top_down_listing_is_flagged(tmp_path):
    text = "layer 2.0 4.0 name ox\nlayer 10 11.9 name sub 20 ohm-cm\n"
    with pytest.warns(EmxImportWarning, match="not the first layer"):
        load_emx_proc(_write(tmp_path, text))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("define a = 1\n", "No layer or conductor"),
        ("layer 1.0 4.0\nvia MA MB { 1e6 S/m \n", "Unterminated"),
        (
            BASE.format(conductor="3e7 S/m") + "via MA MA { rect 1 1 } VA\n",
            "no nominal S/m",
        ),
        (
            BASE.format(conductor="3e7 S/m") + "conductor 1.0 3e7 S/m MA\n",
            "Duplicate conductor",
        ),
        (
            "layer 1.0 4.0\nconductor 1.0 unknown_name MA\n",
            "Cannot resolve sheet resistance",
        ),
        ("layer -1.0 4.0\n", "Negative layer thickness"),
        ("layer 10 11.9 name sub ohm-cm\n", "positive resistivity"),
    ],
)
def test_errors(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        parse_emx_proc(_write(tmp_path, text))


# ---------------------------------------------------------------------------
# Portable JSON
# ---------------------------------------------------------------------------


def test_portable_document_is_json_serializable():
    document = parse_emx_proc(DATA)
    assert document["schema"] == PORTABLE_SCHEMA
    text = json.dumps(document)
    assert json.loads(text) == document
    assert [e["name"] for e in document["layer_stack"] if e["type"] == "conductor"] == [
        "M1",
        "M2",
        "M3",
    ]
    assert [v["name"] for v in document["vias"]] == ["V1", "V2"]


def test_json_round_trip_gives_the_same_stack(tmp_path):
    json_path = tmp_path / "stackup.json"
    json_path.write_text(json.dumps(parse_emx_proc(DATA), indent=2), encoding="utf-8")
    from_proc = _load_quiet(DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EmxImportWarning)
        from_json = load_portable_stackup_json(json_path)
    assert from_json.model_dump() == from_proc.model_dump()


def test_json_with_wrong_schema_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "something-else", "layer_stack": []}))
    with pytest.raises(ValueError, match="Unsupported stackup schema"):
        load_portable_stackup_json(path)


# ---------------------------------------------------------------------------
# Use with the simulation classes
# ---------------------------------------------------------------------------


def test_set_stack_accepts_the_imported_stack(stack):
    from gsim.palace import DrivenSim

    sim = DrivenSim()
    sim.set_stack(stack)
    assert sim.stack is stack
    assert sim._stack_kwargs == {"_prebuilt": True}
    assert sim._resolve_stack() is stack
    # The material values survive the resolution step.
    assert stack.materials["emx_M2_metal"]["conductivity"] == 3.5e7


def _cpw_component(
    length: float = 200.0,
    s_width: float = 10.0,
    g_width: float = 30.0,
    gap: float = 6.0,
):
    """GSG line on M3 with an M2 pad and a V2 via under the upper ground."""
    gf.gpdk.PDK.activate()
    m3, m2, v2 = (13, 0), (12, 0), (22, 0)
    c = gf.Component()
    c << gf.c.rectangle((length, s_width), centered=True, layer=m3)
    y = (g_width + s_width) / 2 + gap
    upper = c << gf.c.rectangle((length, g_width), centered=True, layer=m3)
    upper.move((0, y))
    lower = c << gf.c.rectangle((length, g_width), centered=True, layer=m3)
    lower.move((0, -y))
    pad = c << gf.c.rectangle((40, 20), centered=True, layer=m2)
    pad.move((0, y))
    via = c << gf.c.rectangle((10, 10), centered=True, layer=v2)
    via.move((0, y))
    for name, x, orientation in (("o1", -length / 2, 0), ("o2", length / 2, 180)):
        c.add_port(
            name=name,
            center=(x, 0),
            width=s_width,
            orientation=orientation,
            port_type="electrical",
            layer=m3,
        )
    return c


def test_cpw_on_imported_stack_meshes(stack, tmp_path):
    pytest.importorskip("gmsh")
    from gsim.palace import DrivenSim

    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "emx-cpw"))
    sim.set_geometry(_cpw_component())
    sim.set_stack(stack)
    sim.set_airbox(margin_x=30.0, margin_y=30.0, z_above=30.0)
    sim.add_cpw_port("o1", layer="M3", s_width=10, gap_width=6, length=5.0)
    sim.add_cpw_port("o2", layer="M3", s_width=10, gap_width=6, length=5.0)
    sim.set_driven(fmin=1e9, fmax=20e9, num_points=3)
    sim.mesh(preset="coarse")

    result = sim._last_mesh_result
    assert result.mesh_path.exists()
    groups = result.groups
    volumes = groups["volumes"]
    # Every dielectric material of the stack, the via and the airbox are meshed.
    for region in stack.dielectrics:
        assert region["material"] in volumes
    assert volumes["V2"]["is_via"]
    assert "air" in volumes
    assert {"M2_xy", "M2_z", "M3_xy", "M3_z"} <= set(groups["conductor_surfaces"])
    assert set(groups["port_surfaces"]) == {"P1", "P2"}

    # The Palace config carries the imported material values.
    config = json.loads(sim.write_config().read_text(encoding="utf-8"))
    by_group = {
        attr: entry
        for entry in config["Domains"]["Materials"]
        for attr in entry["Attributes"]
    }

    def domain(name):
        return by_group[volumes[name]["phys_group"]]

    assert domain("emx_dielectric_er_4p2")["Permittivity"] == 4.2
    assert domain("emx_dielectric_er_3p6")["Permittivity"] == 3.6
    assert domain("air")["Permittivity"] == 1.0
    assert domain("emx_substrate")["Conductivity"] == pytest.approx(10.0)
    assert domain("V2")["Conductivity"] == pytest.approx(8.0e6)
    thickness = {
        entry["Attributes"][0]: entry for entry in config["Boundaries"]["Conductivity"]
    }
    m3 = thickness[groups["conductor_surfaces"]["M3_xy"]["phys_group"]]
    assert m3["Conductivity"] == pytest.approx(1 / (0.02 * 1.5e-6), rel=1e-12)
    assert m3["Thickness"] == pytest.approx(1.5)
    m2 = thickness[groups["conductor_surfaces"]["M2_xy"]["phys_group"]]
    assert m2["Conductivity"] == pytest.approx(3.5e7)
    assert m2["Thickness"] == pytest.approx(1.0)
