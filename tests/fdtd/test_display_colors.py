"""PDK colors reach 3D surfaces and labels without changing simulation inputs."""

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
