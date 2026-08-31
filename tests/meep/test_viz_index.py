"""Tests for refractive-index simulation previews."""

from __future__ import annotations

import math
from typing import Any, cast

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt


def _xz_sim_for_index_plot(angle_deg: float = -6.0):
    import gdsfactory as gf

    from gsim.common.stack import Layer, LayerStack
    from gsim.meep.simulation import Simulation

    component = gf.Component()
    component.add_polygon(
        [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
        layer=(1, 0),
    )
    component.add_port(
        name="o1",
        center=(5.0, 0.0),
        orientation=0.0,
        width=0.5,
        layer=(1, 0),
    )
    stack = LayerStack(
        pdk_name="test",
        units="um",
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=0.0,
                zmax=0.22,
                thickness=0.22,
                material="si",
                layer_type="dielectric",
            ),
        },
        dielectrics=[
            {"name": "box", "zmin": -2.0, "zmax": 0.0, "material": "SiO2"},
            {"name": "clad", "zmin": 0.22, "zmax": 1.0, "material": "SiO2"},
        ],
    )
    simulation = Simulation()
    simulation.geometry.component = component
    simulation.geometry.stack = stack
    simulation.materials = {"si": 12.0, "SiO2": 2.1}
    simulation.solver(mode="2d", y_cut="auto")
    simulation.source_fiber(x=0.0, z=1.22, angle_deg=angle_deg, waist=5.4)
    return simulation


def _xy_sim_for_index_plot():
    import gdsfactory as gf

    from gsim.common.stack import Layer, LayerStack
    from gsim.meep.simulation import Simulation

    component = gf.Component()
    component.add_polygon(
        [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
        layer=(1, 0),
    )
    component.add_port(
        name="o1",
        center=(-5.0, 0.0),
        orientation=180.0,
        width=0.5,
        layer=(1, 0),
    )
    component.add_port(
        name="o2",
        center=(5.0, 0.0),
        orientation=0.0,
        width=0.5,
        layer=(1, 0),
    )
    stack = LayerStack(
        pdk_name="test",
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=0.0,
                zmax=0.22,
                thickness=0.22,
                material="si",
                layer_type="dielectric",
            ),
        },
        dielectrics=[{"name": "clad", "zmin": -2.0, "zmax": 2.0, "material": "SiO2"}],
    )
    simulation = Simulation()
    simulation.geometry(component=component, stack=stack)
    simulation.materials = {"si": 12.0, "SiO2": 2.1}
    simulation.source(port="o1", wavelength=1.55, wavelength_span=0.01)
    simulation.monitors = ["o1", "o2"]
    simulation.solver(mode="2d", z_cut="auto")
    return simulation


def test_material_refractive_indices_support_tensor_components():
    from gsim.meep.index_viz import material_refractive_indices
    from gsim.meep.models.config import MaterialData

    materials = {"anisotropic": MaterialData(epsilon_diag=[1.0, 4.0, 9.0])}

    mean_index = material_refractive_indices(materials)["anisotropic"]
    x_index = material_refractive_indices(materials, "x")["anisotropic"]
    z_index = material_refractive_indices(materials, "z")["anisotropic"]

    assert mean_index == pytest.approx(math.sqrt(14.0 / 3.0))
    assert x_index == pytest.approx(1.0)
    assert z_index == pytest.approx(3.0)


def test_build_overlay_resolves_source_and_monitor_offsets():
    import numpy as np

    from gsim.common.geometry_model import GeometryModel
    from gsim.meep.models.config import DomainConfig, PortData
    from gsim.meep.overlay import build_sim_overlay

    geometry_model = GeometryModel(
        prisms={},
        bbox=((-2.0, -1.0, 0.0), (2.0, 1.0, 0.22)),
    )
    domain = DomainConfig(
        dpml=1.0,
        margin_x_low=0.5,
        margin_x_high=0.5,
        margin_y_low=0.5,
        margin_y_high=0.5,
        margin_z_low=0.0,
        margin_z_high=0.0,
        port_margin=0.5,
        extend_ports=0.0,
        source_port_offset=0.1,
        distance_source_to_monitors=0.2,
    )
    ports = [
        PortData(
            name="source",
            center=[-2.0, 0.0, 0.11],
            orientation=0.0,
            width=0.5,
            normal_axis=0,
            direction="+",
            is_source=True,
        ),
        PortData(
            name="output",
            center=[2.0, 0.0, 0.11],
            orientation=180.0,
            width=0.5,
            normal_axis=0,
            direction="-",
            is_source=False,
        ),
    ]

    overlay = build_sim_overlay(geometry_model, domain, ports)

    assert len(overlay.sources) == 1
    assert len(overlay.monitors) == 2
    assert overlay.sources[0].center[0] == pytest.approx(-1.9)
    assert overlay.monitors[0].center[0] == pytest.approx(-1.7)
    assert overlay.monitors[1].center[0] == pytest.approx(1.9)
    assert np.isclose(overlay.sources[0].width, 1.5)


def test_index_plot_is_default_with_material_map_colorbar_and_overlays():
    simulation = _xz_sim_for_index_plot()
    figure, ax = plt.subplots()

    result = simulation.plot_2d(ax=ax)

    assert result is ax
    assert len(figure.axes) == 2
    assert "Refractive index" in figure.axes[1].get_ylabel()
    material_ids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
    assert {"material:air", "material:SiO2", "material:si"} <= material_ids
    air_patch = next(patch for patch in ax.patches if patch.get_gid() == "material:air")
    silicon_patch = next(
        patch for patch in ax.patches if patch.get_gid() == "material:si"
    )
    assert sum(air_patch.get_facecolor()[:3]) > sum(silicon_patch.get_facecolor()[:3])
    assert air_patch.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert len({round(channel, 3) for channel in silicon_patch.get_facecolor()[:3]}) > 1
    _, labels = ax.get_legend_handles_labels()
    assert {"PML", "Source", "Monitor"} <= set(labels)
    assert "Sim cell" not in labels
    pml_patch = next(patch for patch in ax.patches if patch.get_label() == "PML")
    assert pml_patch.get_facecolor()[3] == 0.0
    assert pml_patch.get_edgecolor() == (0.0, 0.0, 0.0, 1.0)
    assert pml_patch.get_hatch() == "////"
    assert pml_patch.get_hatch_linewidth() == pytest.approx(0.6)
    plt.close(figure)


def test_index_plot_uses_requested_tensor_component():
    simulation = _xz_sim_for_index_plot()
    figure, ax = plt.subplots()

    simulation.plot_2d(kind="index", index_component="z", ax=ax)

    assert "n_z" in figure.axes[1].get_ylabel()
    plt.close(figure)


def test_xy_2d_index_plot_uses_resolved_background_material():
    simulation = _xy_sim_for_index_plot()
    figure, ax = plt.subplots()

    simulation.plot_2d(ax=ax)

    material_ids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
    assert "material:SiO2" in material_ids
    assert "material:air" not in material_ids
    plt.close(figure)


def test_index_plot_ignores_unpopulated_stack_materials():
    from gsim.common.stack import Layer

    simulation = _xy_sim_for_index_plot()
    assert simulation.geometry.stack is not None
    simulation.geometry.stack.layers["unused_metal"] = Layer(
        name="unused_metal",
        gds_layer=(99, 0),
        zmin=1.0,
        zmax=2.0,
        thickness=1.0,
        material="Aluminum",
        layer_type="conductor",
    )
    figure, ax = plt.subplots()

    simulation.plot_2d(ax=ax)

    material_ids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
    assert "material:Aluminum" not in material_ids
    plt.close(figure)


def test_multi_slice_index_plot_uses_separate_figures(monkeypatch):
    simulation = _xz_sim_for_index_plot()
    existing_figure_numbers = set(plt.get_fignums())
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(None))

    result = simulation.plot_2d(slices="xyz")

    new_figure_numbers = sorted(
        set(plt.get_fignums()).difference(existing_figure_numbers)
    )
    figures = [plt.figure(number) for number in new_figure_numbers]
    try:
        assert result is None
        assert len(show_calls) == 1
        assert len(figures) == 3
        assert [
            figure.axes[0].get_title().split(" · ")[1][:2] for figure in figures
        ] == [
            "YZ",
            "XZ",
            "XY",
        ]

        colorbar_bounds = set()
        for figure in figures:
            assert len(figure.axes) == 2
            plot_axis, colorbar_axis = figure.axes
            assert plot_axis.get_legend() is None
            assert "Refractive index" in colorbar_axis.get_ylabel()
            assert len(figure.legends) == 1
            legend = figure.legends[0]
            assert getattr(legend, "_outside_loc", None) == "lower"

            expected_labels = list(
                dict.fromkeys(plot_axis.get_legend_handles_labels()[1])
            )
            assert [text.get_text() for text in legend.get_texts()] == expected_labels

            colorbar_norm = colorbar_axis.collections[0].norm
            colorbar_bounds.add((colorbar_norm.vmin, colorbar_norm.vmax))

        assert len(colorbar_bounds) == 1
    finally:
        for figure in figures:
            plt.close(figure)


def test_multi_slice_index_plot_can_hide_bottom_legends(monkeypatch):
    simulation = _xz_sim_for_index_plot()
    existing_figure_numbers = set(plt.get_fignums())
    monkeypatch.setattr(plt, "show", lambda: None)

    simulation.plot_2d(slices="xy", legend=False)

    new_figure_numbers = set(plt.get_fignums()).difference(existing_figure_numbers)
    figures = [plt.figure(number) for number in new_figure_numbers]
    try:
        assert len(figures) == 2
        assert all(len(figure.axes) == 2 for figure in figures)
        assert all(not figure.legends for figure in figures)
    finally:
        for figure in figures:
            plt.close(figure)


@pytest.mark.parametrize("angle_deg", [-30.0, -6.0, 25.0])
def test_index_plot_draws_fiber_waist_at_configured_angle(angle_deg):
    simulation = _xz_sim_for_index_plot(angle_deg)
    figure, ax = plt.subplots()

    simulation.plot_2d(kind="index", ax=ax)

    source_line = next(line for line in ax.lines if line.get_label() == "Source")
    x_coordinates = np.asarray(source_line.get_xdata(), dtype=float)
    z_coordinates = np.asarray(source_line.get_ydata(), dtype=float)
    rendered_slope = (z_coordinates[1] - z_coordinates[0]) / (
        x_coordinates[1] - x_coordinates[0]
    )
    assert rendered_slope == pytest.approx(math.tan(math.radians(angle_deg)))
    assert not any(line.get_linestyle() == "--" for line in ax.lines)
    arrow = cast(
        Any,
        next(
            annotation
            for annotation in ax.texts
            if getattr(annotation, "arrow_patch", None) is not None
        ),
    )
    arrow_head = arrow.xy
    arrow_tail = arrow.xyann
    arrow_vector = (
        arrow_head[0] - arrow_tail[0],
        arrow_head[1] - arrow_tail[1],
    )
    theta = math.radians(angle_deg)
    assert arrow_vector == pytest.approx(
        (0.6 * math.sin(theta), -0.6 * math.cos(theta))
    )
    source_vector = (
        x_coordinates[1] - x_coordinates[0],
        z_coordinates[1] - z_coordinates[0],
    )
    assert math.hypot(*arrow_vector) == pytest.approx(0.6)
    assert sum(
        source_component * arrow_component
        for source_component, arrow_component in zip(
            source_vector, arrow_vector, strict=True
        )
    ) == pytest.approx(0.0, abs=1e-12)
    plt.close(figure)


def test_layer_plot_remains_available_explicitly():
    simulation = _xz_sim_for_index_plot()
    figure, ax = plt.subplots()

    simulation.plot_2d(kind="layers", ax=ax)

    assert len(figure.axes) == 1
    assert "Refractive index" not in ax.get_title()
    plt.close(figure)


def test_interactive_index_plot_is_default():
    simulation = _xz_sim_for_index_plot()

    figure = simulation.plot_2d_interactive()

    assert "Refractive index" in figure.layout.title.text
    assert next(trace for trace in figure.data if trace.name == "air").fillcolor == (
        "#ffffff"
    )
    assert any(trace.marker.showscale for trace in figure.data)
    pml_trace = next(trace for trace in figure.data if trace.name == "PML")
    assert pml_trace.fillpattern.shape == "/"
    assert pml_trace.fillpattern.bgcolor == "rgba(0,0,0,0)"
    legend_names = {trace.name for trace in figure.data if trace.showlegend}
    assert {"PML", "Source", "Monitor"} <= legend_names
    assert "Sim cell" not in legend_names
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.xanchor == "center"
    assert figure.layout.legend.y < 0
    assert figure.layout.margin.b >= 100
    colorbar_trace = next(trace for trace in figure.data if trace.marker.showscale)
    colorbar = colorbar_trace.marker.colorbar
    assert colorbar.x > 1
    assert colorbar.title.side == "right"


def test_xy_2d_interactive_plot_uses_resolved_background_material():
    simulation = _xy_sim_for_index_plot()

    figure = simulation.plot_2d_interactive()

    background_trace = figure.data[0]
    assert background_trace.name == "SiO2"
    assert background_trace.fillcolor != "#ffffff"


def test_extended_background_reaches_pml_in_both_index_plots():
    simulation = _xz_sim_for_index_plot()
    figure, ax = plt.subplots()

    simulation.plot_2d(ax=ax)
    silica_patches = cast(
        list[Any],
        [patch for patch in ax.patches if patch.get_gid() == "material:SiO2"],
    )
    assert min(patch.get_y() for patch in silica_patches) == pytest.approx(
        ax.get_ylim()[0]
    )
    plt.close(figure)

    interactive = simulation.plot_2d_interactive()
    silica_traces = [trace for trace in interactive.data if trace.name == "SiO2"]
    assert min(min(trace.y) for trace in silica_traces) == pytest.approx(
        interactive.layout.yaxis.range[0]
    )


@pytest.mark.parametrize("angle_deg", [-30.0, -6.0, 25.0])
def test_interactive_fiber_arrow_is_short_and_perpendicular(angle_deg):
    simulation = _xz_sim_for_index_plot(angle_deg)

    figure = simulation.plot_2d_interactive()

    source_trace = next(
        trace for trace in figure.data if trace.name == "Source" and trace.text
    )
    source_vector = (
        source_trace.x[1] - source_trace.x[0],
        source_trace.y[1] - source_trace.y[0],
    )
    arrow = figure.layout.annotations[0]
    arrow_vector = (arrow.x - arrow.ax, arrow.y - arrow.ay)
    theta = math.radians(angle_deg)
    assert arrow_vector == pytest.approx(
        (0.6 * math.sin(theta), -0.6 * math.cos(theta))
    )
    assert math.hypot(*arrow_vector) == pytest.approx(0.6)
    assert sum(
        source_component * arrow_component
        for source_component, arrow_component in zip(
            source_vector, arrow_vector, strict=True
        )
    ) == pytest.approx(0.0, abs=1e-12)


def test_interactive_layer_plot_remains_available_explicitly():
    simulation = _xz_sim_for_index_plot()

    figure = simulation.plot_2d_interactive(kind="layers")

    assert "Refractive index" not in figure.layout.title.text
    assert any(trace.name == "core" for trace in figure.data)
    assert not any(trace.marker.showscale for trace in figure.data)


def test_invalid_plot_kind_raises():
    simulation = _xz_sim_for_index_plot()

    with pytest.raises(ValueError, match="kind must be"):
        simulation.plot_2d(kind="epsilon")
