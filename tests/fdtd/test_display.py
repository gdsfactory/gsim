"""PDK display metadata reaches the viewer without changing run artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import gdsfactory as gf
import pytest
from gdsfactory.technology import LayerMap, LayerView, LayerViews, LogicalLayer

from gsim import fdtd
from gsim.fdtd.display import layer_group_styles
from gsim.fdtd.viz import plot_mesh
from tests.fdtd.test_viz import _write_mesh


def _resolved(levels, views=None):
    return cast(
        Any,
        SimpleNamespace(
            layer_stack=SimpleNamespace(layers=levels),
            layers=levels,
            pdk=SimpleNamespace(layer_views=views),
        ),
    )


def _level(layer=(1, 0), *, derived_layer=None, **info):
    return SimpleNamespace(
        layer=LogicalLayer(layer=layer), derived_layer=derived_layer, info=info
    )


def test_stack_opacity_and_visibility_use_physical_names():
    resolved = _resolved(
        {
            "core": _level(opacity=0.05),
            "oxide": _level((2, 0), opacity=0.5, visible=False),
        }
    )
    assert layer_group_styles(resolved) == {
        "core": {"opacity": 0.05},
        "oxide": {"opacity": 0.5, "visible": False},
    }


def test_hidden_layer_view_groups_and_stack_override():
    views = LayerViews(
        layer_views={
            "hidden": LayerView(
                visible=False,
                group_members={"child": LayerView(layer=(1, 0), visible=True)},
            )
        }
    )
    levels = {"core": _level()}
    assert layer_group_styles(_resolved(levels, views)) == {"core": {"visible": False}}
    levels["core"].info["visible"] = True
    assert layer_group_styles(_resolved(levels, views)) == {"core": {"visible": True}}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("visible", [False, True])
def test_duplicate_layer_views_are_visible_if_any_view_is_visible(
    reverse, grouped, visible
):
    hidden = LayerView(layer=(1, 0), visible=False)
    if grouped:
        hidden = LayerView(
            visible=False,
            group_members={"child": LayerView(layer=(1, 0), visible=True)},
        )
    entries = [
        ("hidden", hidden),
        ("other", LayerView(layer=(1, 0), visible=visible)),
    ]
    views = LayerViews(layer_views=dict(reversed(entries) if reverse else entries))

    assert layer_group_styles(_resolved({"core": _level()}, views)) == {
        "core": {"visible": visible}
    }


def test_derived_layer_view_takes_precedence():
    views = LayerViews(
        layer_views={
            "mask": LayerView(layer=(1, 0), visible=False),
            "solid": LayerView(layer=(100, 0), visible=True),
        }
    )
    resolved = _resolved(
        {"core": _level(derived_layer=LogicalLayer(layer=(100, 0)))}, views
    )
    assert layer_group_styles(resolved) == {"core": {"visible": True}}


def test_missing_derived_view_falls_back_to_source_view():
    views = LayerViews(layer_views={"mask": LayerView(layer=(1, 0), visible=False)})
    resolved = _resolved(
        {"core": _level(derived_layer=LogicalLayer(layer=(100, 0)))}, views
    )
    assert layer_group_styles(resolved) == {"core": {"visible": False}}


def test_named_layer_uses_owning_pdk():
    views = LayerViews(layer_views={"core": LayerView(layer=(1, 0), visible=False)})
    resolved = _resolved({"core": _level(layer="CORE")}, views)
    resolved.pdk.layers = SimpleNamespace(CORE=(1, 0))
    assert layer_group_styles(resolved) == {"core": {"visible": False}}


@pytest.mark.parametrize("grouped", [False, True])
def test_name_backed_layer_view_respects_group_visibility(grouped):
    class DisplayLayers(LayerMap):
        CORE = (1, 0)

    core_view = LayerView(visible=grouped)
    layer_views = (
        {"hidden": LayerView(visible=False, group_members={"CORE": core_view})}
        if grouped
        else {"CORE": core_view}
    )
    views = LayerViews(layers=DisplayLayers, layer_views=layer_views)
    resolved = _resolved({"core": _level()}, views)
    assert layer_group_styles(resolved) == {"core": {"visible": False}}


def test_layer_index_uses_component_layout():
    component = gf.Component()
    layer_index = component.kcl.layer(17, 3)
    views = LayerViews(layer_views={"core": LayerView(layer=(17, 3), visible=False)})
    resolved = _resolved({"core": _level(layer=layer_index)}, views)
    resolved.component = component
    assert layer_group_styles(resolved) == {"core": {"visible": False}}


def test_file_backed_layer_views_use_pdk_loader(tmp_path):
    view_path = tmp_path / "layer_views.yaml"
    view_path.write_text(
        "LayerViews:\n  core:\n    layer: [1, 0]\n    visible: false\n"
    )
    pdk = gf.Pdk(name="generic_display", layer_views=view_path)
    resolved = _resolved({"core": _level()})
    resolved.pdk = pdk
    assert layer_group_styles(resolved) == {"core": {"visible": False}}


@pytest.mark.parametrize("visible", [0, "false"])
def test_invalid_visibility_has_clear_error(visible):
    with pytest.raises(ValueError, match="visibility must be a bool"):
        layer_group_styles(_resolved({"core": _level(visible=visible)}))


def test_zero_opacity_can_be_switched_on():
    assert layer_group_styles(_resolved({"core": _level(opacity=0)})) == {
        "core": {"visible": False, "opacity": 1.0}
    }


def test_missing_metadata_retains_viewer_fallbacks():
    assert layer_group_styles(_resolved({"core": _level()})) == {}
    assert layer_group_styles(None) == {}


@pytest.mark.parametrize("opacity", [-0.1, 1.1, float("nan"), float("inf"), "invalid"])
def test_invalid_opacity_has_clear_error(opacity):
    with pytest.raises(ValueError, match="opacity must be between 0 and 1"):
        layer_group_styles(_resolved({"core": _level(opacity=opacity)}))


def test_viewer_embeds_metadata_and_explicit_overrides(tmp_path: Path):
    viewer = plot_mesh(
        _write_mesh(tmp_path / "mesh.msh"),
        group_styles={"core": {"opacity": 0.15, "visible": False}},
        show_groups=["core"],
        hide_groups=["oxide"],
    )
    assert '"groupStyles":{"core":{"visible":false,"opacity":0.15}}' in viewer.html
    assert '"showGroups":["core"]' in viewer.html
    assert '"hideGroups":["oxide"]' in viewer.html


def test_simulation_3d_passes_stack_metadata_and_preserves_run(
    tmp_path, fdtd_pdk_module
):
    fdtd_pdk_module.PDK.layer_stack.layers["core"].info.update(
        opacity=0.2, visible=False
    )
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module, mesh_size_nm=750, default_port="o1"
    )
    simulation.geometry("straight")
    artifacts = simulation.write(tmp_path)
    config_before = artifacts.config_path.read_bytes()
    mesh_before = artifacts.mesh_path.read_bytes()
    assert '"core":{"visible":false,"opacity":0.2}' in simulation.plot_3d().html
    assert (
        '"core":{"visible":false,"opacity":0.2}'
        in simulation.plot_3d(show_mesh=True).html
    )
    assert '"groupStyles":{}' in simulation.plot_2d_interactive().html
    assert artifacts.config_path.read_bytes() == config_before
    assert artifacts.mesh_path.read_bytes() == mesh_before


@pytest.mark.parametrize("saved_mesh", [False, True])
@pytest.mark.parametrize(
    "selection", [{}, {"hide_groups": ["unused"]}, {"show_groups": ["core"]}]
)
def test_simulation_3d_ignores_unused_display_metadata(
    tmp_path, fdtd_pdk_module, saved_mesh, selection
):
    levels = fdtd_pdk_module.PDK.layer_stack.layers
    levels["unused"] = levels.pop("thin_core")
    levels["unused"].info["opacity"] = 50
    levels["core"].info["opacity"] = 0.2
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module, mesh_size_nm=750, default_port="o1"
    )
    simulation.geometry("straight")
    if saved_mesh:
        simulation.write(tmp_path)

    viewer = simulation.plot_3d(**selection)

    assert '"groupStyles":{"core":{"opacity":0.2}}' in viewer.html
    assert (simulation._last_artifacts is not None) == saved_mesh


def test_simulation_3d_rejects_invalid_opacity_on_used_layer(fdtd_pdk_module):
    fdtd_pdk_module.PDK.layer_stack.layers["core"].info["opacity"] = 50
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module, mesh_size_nm=750, default_port="o1"
    )
    simulation.geometry("straight")

    with pytest.raises(
        ValueError, match="Group 'core' opacity must be between 0 and 1"
    ):
        simulation.plot_3d()
