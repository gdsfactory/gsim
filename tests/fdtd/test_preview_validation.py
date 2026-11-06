"""Compatibility and pre-mesh validation for public FDTD preview APIs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from gsim import fdtd
from gsim.fdtd.viz import MeshViewer
from tests.fdtd.test_viz import _write_mesh


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"height": 500},
        {"zoom_to_cursor": False},
        {"height": 500, "zoom_to_cursor": False},
    ],
)
def test_legacy_plot_2d_preserves_html_viewer(tmp_path, options, monkeypatch):
    simulation = fdtd.Simulation()
    cast(Any, simulation)._last_artifacts = SimpleNamespace(
        mesh_path=_write_mesh(tmp_path / "mesh.msh")
    )
    monkeypatch.setattr(
        simulation, "write", lambda _directory: simulation._last_artifacts
    )
    with pytest.warns(DeprecationWarning, match=r"plot_2d_interactive.*plot_2d_static"):
        viewer = simulation.plot_2d(
            axis="y",
            position_um=0.25,
            show_mesh=True,
            show_groups=["core"],
            hide_groups=["port_o2"],
            **options,
        )
    assert isinstance(viewer, MeshViewer)
    assert '"mode":"mesh_slice"' in viewer.html
    assert '"axis":"y"' in viewer.html
    assert '"positionUm":0.25' in viewer.html
    assert '"showGroups":["core"]' in viewer.html
    assert '"hideGroups":["port_o2"]' in viewer.html
    assert f'height="{options.get("height", 600)}"' in viewer._repr_html_()
    zoom = "true" if options.get("zoom_to_cursor", True) else "false"
    assert f'"zoomToCursor":{zoom}' in viewer.html
    output = viewer.save(tmp_path / "legacy.html")
    assert output.read_text(encoding="utf8") == viewer.html


def test_legacy_plot_2d_keeps_default_slice_mode(tmp_path, monkeypatch):
    simulation = fdtd.Simulation()
    cast(Any, simulation)._last_artifacts = SimpleNamespace(
        mesh_path=_write_mesh(tmp_path / "mesh.msh")
    )
    monkeypatch.setattr(
        simulation, "write", lambda _directory: simulation._last_artifacts
    )
    with pytest.warns(DeprecationWarning, match=r"plot_2d\(\) is deprecated"):
        viewer = simulation.plot_2d()
    assert isinstance(viewer, MeshViewer)
    assert '"mode":"slice"' in viewer.html


def test_static_preview_rejects_unknown_selected_port_before_meshing(
    fdtd_pdk_module, monkeypatch
):
    from gsim.fdtd import visualization

    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    simulation.source(port="nope")
    monkeypatch.setattr(
        visualization, "generate_mesh", lambda *_args, **_kwargs: pytest.fail("meshed")
    )
    with pytest.raises(fdtd.FDTDConfigError, match="source port 'nope' is not present"):
        simulation.plot_2d_static()


@pytest.mark.parametrize("region", ["dipole", "line_current", "gaussian"])
def test_static_preview_rejects_regions_outside_explicit_domain(
    fdtd_pdk_module, monkeypatch, region
):
    from gsim.fdtd import visualization

    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    simulation.domain(y_bounds=(-1, 1))
    if region == "dipole":
        simulation.source = fdtd.DipoleSource(
            position_um=(1, 1.1, 0.11), current_axis="z"
        )
    elif region == "line_current":
        simulation.source = fdtd.LineCurrentSource(
            position_um=(1, 0.9, 0.11), line_axis="y", current_axis="z", length_um=0.4
        )
    else:
        simulation.source = fdtd.GaussianBeamSource(
            center_um=(1, 1.1, 0.11),
            size_um=(1, 0.4, 0),
            aperture_normal="-z",
            propagation_direction=(0, 0, -1),
            e_polarization=(1, 0, 0),
            focal_point_um=(1, 0, 0),
            waist_radius_um=0.1,
        )
    monkeypatch.setattr(
        visualization, "generate_mesh", lambda *_args, **_kwargs: pytest.fail("meshed")
    )
    with pytest.raises(fdtd.FDTDConfigError, match=r"exceeds domain.y_bounds"):
        simulation.plot_2d_static()


def test_static_preview_shows_cropped_monitor_with_warning(fdtd_pdk_module):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    simulation.domain(y_bounds=(-1, 1))
    simulation.monitors.add_plane(
        "image", center_um=(1, 1.1, 0.11), size_um=(1, 0.4, 0), normal="+z"
    )
    with pytest.warns(UserWarning, match="Monitor 'image' cropped") as caught:
        axes = simulation.plot_2d_static()
    assert (
        sum("Monitor 'image' cropped" in str(warning.message) for warning in caught)
        == 1
    )
    patch = next(
        patch for patch in axes.patches if patch.get_label() == "Monitor image"
    )
    assert isinstance(patch, Rectangle)
    assert patch.get_bbox().bounds == pytest.approx((0.5, 0.9, 1, 0.1))
    assert simulation.monitors["image"].size_um == (1, 0.4, 0)
    plt.close(axes.figure)


@pytest.mark.parametrize("explicit", [False, True])
def test_static_preview_rejects_outside_plane_before_meshing(
    fdtd_pdk_module, monkeypatch, explicit
):
    from gsim.fdtd import visualization

    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    if explicit:
        simulation.domain(z_bounds=(-1, 1))
    simulation.monitors.add_plane(
        "outside", center_um=(1, 0, 100), size_um=(1, 1, 0), normal="+z"
    )
    monkeypatch.setattr(
        visualization, "generate_mesh", lambda *_args, **_kwargs: pytest.fail("meshed")
    )
    with pytest.raises(fdtd.FDTDConfigError, match=r"z-position.*outside"):
        simulation.plot_2d_static()


@pytest.mark.parametrize("source_port", [None, "o1", "o2"])
def test_preview_rejects_monitor_shadowing_vertical_port_before_meshing(
    fdtd_pdk_module, monkeypatch, source_port
):
    from gsim.fdtd import visualization

    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("vertical_coupler")
    if source_port is not None:
        simulation.source(port=source_port)
    simulation.monitors.add_plane(
        name="o2", center_um=(1, 0, 0.11), size_um=(1, 1, 0), normal="+z"
    )
    monkeypatch.setattr(
        visualization, "generate_mesh", lambda *_args, **_kwargs: pytest.fail("meshed")
    )
    with pytest.raises(fdtd.FDTDConfigError, match="cannot shadow vertical ports"):
        simulation.plot_2d_static()


def test_write_rejects_monitor_shadowing_vertical_port(fdtd_pdk_module, tmp_path):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("vertical_coupler")
    simulation.source(port="o1")
    simulation.monitors.add_plane(
        name="o2", center_um=(1, 0, 0.11), size_um=(1, 1, 0), normal="+z"
    )
    with pytest.raises(fdtd.FDTDConfigError, match="cannot shadow vertical ports"):
        simulation.write(tmp_path / "run")
