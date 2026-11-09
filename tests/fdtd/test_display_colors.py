"""PDK colors reach 3D surfaces and labels without changing simulation inputs."""

import json
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from gdsfactory.technology import LayerView, LayerViews, LogicalLayer

from gsim import fdtd
from gsim.fdtd.display import layer_group_styles, normalize_group_styles
from gsim.fdtd.viz import plot_mesh
from tests.fdtd.test_display import _level, _resolved
from tests.fdtd.test_viz import _write_mesh


def test_layer_view_fill_color_is_exported():
    views = LayerViews(
        layer_views={
            "core": LayerView(layer=(1, 0), fill_color="#123456", frame_color="red")
        }
    )
    assert layer_group_styles(_resolved({"core": _level()}, views)) == {
        "core": {"visible": True, "color": "#123456"}
    }


def test_frame_color_is_used_when_fill_is_absent():
    views = LayerViews(
        layer_views={"core": LayerView(layer=(1, 0), frame_color="#abcdef")}
    )
    assert layer_group_styles(_resolved({"core": _level()}, views))["core"][
        "color"
    ] == ("#abcdef")


def test_stack_color_overrides_layer_view_and_preserves_opacity():
    views = LayerViews(layer_views={"core": LayerView(layer=(1, 0), color="red")})
    styles = layer_group_styles(
        _resolved({"core": _level(color="rgba(0, 128, 255, 0.1)", opacity=0.4)}, views)
    )
    assert styles == {"core": {"visible": True, "color": "#0080ff", "opacity": 0.4}}


def test_derived_layer_color_takes_precedence_over_source():
    views = LayerViews(
        layer_views={
            "drawing": LayerView(layer=(1, 0), color="red"),
            "physical": LayerView(layer=(100, 0), color="blue"),
        }
    )
    level = _level(derived_layer=LogicalLayer(layer=(100, 0)))
    assert layer_group_styles(_resolved({"core": level}, views))["core"]["color"] == (
        "#0000ff"
    )


def test_uncolored_derived_view_falls_back_to_source_color():
    views = LayerViews(
        layer_views={
            "drawing": LayerView(layer=(1, 0), color="red"),
            "physical": LayerView(layer=(100, 0), visible=False),
        }
    )
    level = _level(derived_layer=LogicalLayer(layer=(100, 0)))
    assert layer_group_styles(_resolved({"core": level}, views)) == {
        "core": {"visible": False, "color": "#ff0000"}
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_visible_colored_view_wins_over_hidden_duplicate(reverse):
    entries = [
        ("hidden", LayerView(layer=(1, 0), visible=False, color="red")),
        ("visible", LayerView(layer=(1, 0), color="blue")),
    ]
    views = LayerViews(layer_views=dict(reversed(entries) if reverse else entries))
    assert layer_group_styles(_resolved({"core": _level()}, views))["core"][
        "color"
    ] == ("#0000ff")


def test_hidden_parent_does_not_override_visible_color():
    views = LayerViews(
        layer_views={
            "hidden": LayerView(
                visible=False,
                group_members={"child": LayerView(layer=(1, 0), color="red")},
            ),
            "visible": LayerView(layer=(1, 0), color="blue"),
        }
    )
    assert layer_group_styles(_resolved({"core": _level()}, views))["core"][
        "color"
    ] == ("#0000ff")


def test_missing_colors_keep_the_viewer_palette():
    views = LayerViews(layer_views={"core": LayerView(layer=(1, 0))})
    assert layer_group_styles(_resolved({"core": _level()}, views)) == {
        "core": {"visible": True}
    }


def _background_level(material, *, name=None, derived_layer=None, **info):
    return SimpleNamespace(
        layer=LogicalLayer(layer=(9, 0)),
        derived_layer=derived_layer,
        name=name,
        material=material,
        background=True,
        info=info,
    )


def test_shared_background_drawing_color_is_not_a_material_color():
    views = LayerViews(layer_views={"drawing": LayerView(layer=(9, 0), color="red")})
    levels = {
        "lower": _background_level("medium_a"),
        "upper": _background_level("medium_b"),
    }
    assert layer_group_styles(_resolved(levels, views)) == {
        "lower": {"visible": True},
        "upper": {"visible": True},
    }


def test_shared_background_same_material_can_use_drawing_color():
    views = LayerViews(layer_views={"drawing": LayerView(layer=(9, 0), color="red")})
    levels = {
        "lower": _background_level("medium_a"),
        "upper": _background_level("medium_a"),
    }
    assert layer_group_styles(_resolved(levels, views)) == {
        "lower": {"visible": True, "color": "#ff0000"},
        "upper": {"visible": True, "color": "#ff0000"},
    }


def test_shared_background_uses_named_physical_views_and_stack_override():
    views = LayerViews(
        layer_views={
            "drawing": LayerView(layer=(9, 0), color="red"),
            "lower": LayerView(layer=(9, 0), fill_color="blue"),
            "declared_upper": LayerView(layer=(9, 0), frame_color="lime"),
        }
    )
    levels = {
        "lower": _background_level("medium_a"),
        "upper": _background_level("medium_b", name="declared_upper"),
    }
    assert layer_group_styles(_resolved(levels, views)) == {
        "lower": {"visible": True, "color": "#0000ff"},
        "upper": {"visible": True, "color": "#00ff00"},
    }
    levels["lower"].info["color"] = "yellow"
    assert layer_group_styles(_resolved(levels, views))["lower"]["color"] == "#ffff00"


def test_shared_background_can_use_distinct_derived_layer_color():
    views = LayerViews(
        layer_views={
            "drawing": LayerView(layer=(9, 0), color="red"),
            "physical": LayerView(layer=(100, 0), color="blue"),
        }
    )
    levels = {
        "lower": _background_level(
            "medium_a", derived_layer=LogicalLayer(layer=(100, 0))
        ),
        "upper": _background_level("medium_b"),
    }
    styles = layer_group_styles(_resolved(levels, views))
    assert styles["lower"]["color"] == "#0000ff"
    assert "color" not in styles["upper"]


def test_unused_stack_color_is_not_validated():
    resolved = _resolved({"core": _level(), "unused": _level((2, 0), color="invalid")})
    resolved.layers = {"core": resolved.layers["core"]}
    assert layer_group_styles(resolved) == {}


@pytest.mark.parametrize("value", ["not-a-color", "#zzzzzz", None])
def test_invalid_explicit_color_has_clear_error(value):
    with pytest.raises(
        ValueError, match="Group 'core' color must be a valid CSS color"
    ):
        normalize_group_styles({"core": {"color": value}})


def test_file_backed_color_uses_pdk_loader(tmp_path):
    path = tmp_path / "views.yaml"
    path.write_text(
        'LayerViews:\n  core:\n    layer: [1, 0]\n    fill_color: "#123456"\n'
    )
    resolved = _resolved({"core": _level()})
    resolved.pdk = SimpleNamespace(
        layer_views=path,
        get_layer_views=lambda: LayerViews(path),
    )
    assert layer_group_styles(resolved)["core"]["color"] == "#123456"


def test_explicit_color_is_embedded_and_precedes_fallbacks(tmp_path):
    viewer = plot_mesh(
        _write_mesh(tmp_path / "mesh.msh"),
        group_styles={"core": {"color": "red", "opacity": 0.3, "visible": False}},
        show_groups=["core"],
    )
    assert '"core":{"color":"#ff0000","visible":false,"opacity":0.3}' in viewer.html
    start = viewer.html.index("function colorFor(mesh, group)")
    explicit = viewer.html.index("if (color !== undefined)", start)
    port = viewer.html.index("if (isPortGroup(mesh, group))", start)
    largest = viewer.html.index("if (group === currentLargestGroup)", start)
    assert explicit < port < largest
    assert '"showGroups":["core"]' in viewer.html


def test_dark_and_light_group_colors_have_readable_labels(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to exercise the viewer's color logic")
    viewer = plot_mesh(_write_mesh(tmp_path / "mesh.msh"))
    helper = re.search(
        r"function groupTextColor\(color\) \{.*?\n      \}", viewer.html, re.DOTALL
    )
    assert helper is not None
    colors = [
        {"r": 0, "g": 0, "b": 0},
        {"r": 0, "g": 0, "b": 1},
        {"r": 1, "g": 1, "b": 1},
        {"r": 1, "g": 0, "b": 0},
    ]
    script = (
        f"const choose = ({helper.group()});"
        f"console.log(JSON.stringify({json.dumps(colors)}.map(choose)));"
    )
    completed = subprocess.run(  # noqa: S603 -- Fixed interpreter and repository-owned JS.
        [node, "-e", script], capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == ["#ffffff", "#ffffff", "#101216", "#101216"]
    compact_html = "".join(viewer.html.split())
    assert '"--group-text-color",groupTextColor(color)' in compact_html
    assert "color:var(--group-text-color,#101216)" in compact_html


def test_simulation_uses_pdk_color_without_changing_saved_inputs(
    tmp_path, fdtd_pdk_module
):
    fdtd_pdk_module.PDK.layer_views = LayerViews(
        layer_views={"core": LayerView(layer=(1, 0), fill_color="#123456")}
    )
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, default_port="o1")
    simulation.geometry("straight", mesh_size_nm=750)
    artifacts = simulation.write(tmp_path)
    config_before = artifacts.config_path.read_bytes()
    mesh_before = artifacts.mesh_path.read_bytes()
    viewer = simulation.plot_3d()
    assert '"core":{"color":"#123456","visible":true}' in viewer.html
    assert artifacts.config_path.read_bytes() == config_before
    assert artifacts.mesh_path.read_bytes() == mesh_before
