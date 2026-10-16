"""Check cross-section geometry compatibility and mode plot variants."""

from unittest.mock import Mock

import klayout.db as kdb
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gsim.meep import viz
from gsim.meep.models.config import DomainConfig
from tests.meep.test_mode_result_plots import _make_1d_result, _make_2d_result


@pytest.mark.parametrize("epsilon", [None, [], [4], [0, 0, 0], [np.nan] * 3])
def test_index_map_rejects_unusable_tensor_components(epsilon):
    from gsim.meep.index_viz import material_refractive_indices
    from gsim.meep.models.config import MaterialData

    data = {"material": MaterialData(epsilon_diag=epsilon)}
    if epsilon is None or epsilon == []:
        assert material_refractive_indices(data) == {}
    else:
        with pytest.raises(ValueError, match=r"component|permittivity"):
            material_refractive_indices(data, "z")


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("representation", ["array", "tuple", "klayout", "unavailable"])
def test_cross_section_polygon_compatibility(index_context, axis, representation):
    stack, _ = index_context
    points = [(-3, -2), (3, -2), (3, 2), (1, 2), (1, -1), (-1, -1), (-1, 2), (-3, 2)]
    if axis == "x":
        points = [(y, x) for x, y in points]
    polygon = (
        np.array(points)
        if representation == "array"
        else tuple(points)
        if representation == "tuple"
        else kdb.Polygon([kdb.Point(x, y) for x, y in points])
    )
    component = Mock()
    component.get_polygons_points.return_value = {
        (1, 0): [polygon, np.array([[0, 0]]), object()]
    }
    if representation == "unavailable":
        component.get_polygons_points.side_effect = RuntimeError("stale")
        component.dup.return_value.get_polygons_points.side_effect = RuntimeError(
            "stale"
        )
    rectangles = viz.build_cross_section_rectangles(component, stack, axis, 0)
    assert len(rectangles) == (0 if representation == "unavailable" else 4)
    if rectangles:
        assert sorted(
            (rectangle["h_min"], rectangle["h_max"]) for rectangle in rectangles
        ) == [(-3, -1), (-3, -1), (1, 3), (1, 3)]


@pytest.mark.parametrize("backend", ["pyvista", "open3d", "invalid"])
def test_three_dimensional_plot_builds_real_geometry(
    index_context, monkeypatch, backend
):
    import gsim.common.viz as renderers

    stack, component = index_context
    displayed = []
    monkeypatch.setattr(
        renderers, "plot_prisms_3d", lambda model, **_kw: displayed.append(model)
    )
    monkeypatch.setattr(
        renderers, "plot_prisms_3d_open3d", lambda model, **_kw: displayed.append(model)
    )
    if backend == "invalid":
        with pytest.raises(ValueError, match="Unsupported backend"):
            viz.plot_3d(
                component,
                stack,
                DomainConfig(
                    dpml=1,
                    margin_x_low=1,
                    margin_x_high=1,
                    margin_y_low=1,
                    margin_y_high=1,
                    margin_z_low=1,
                    margin_z_high=1,
                    port_margin=1,
                    extend_ports=0,
                    source_port_offset=0,
                    distance_source_to_monitors=0,
                ),
                backend=backend,
                extend_ports_length=0,
            )
    else:
        viz.plot_3d(
            component,
            stack,
            DomainConfig(
                dpml=1,
                margin_x_low=1,
                margin_x_high=1,
                margin_y_low=1,
                margin_y_high=1,
                margin_z_low=1,
                margin_z_high=1,
                port_margin=1,
                extend_ports=0,
                source_port_offset=0,
                distance_source_to_monitors=0,
            ),
            backend=backend,
            extend_ports_length=0,
        )
        assert displayed[0].all_prisms


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("norm", ["phase", "imag"])
def test_mode_plot_existing_axes_and_explicit_titles(monkeypatch, dimension, norm):
    result = (_make_1d_result if dimension == 1 else _make_2d_result)()
    show = Mock()
    monkeypatch.setattr(plt, "show", show)
    figure, axis = plt.subplots()
    try:
        returned, returned_axis = result.plot_mode(
            components="Ey",
            norm=norm,
            geometry=True,
            ax=axis,
            title=None,
            suptitle="test mode",
            show=True,
            cmap="twilight",
        )
        assert returned is figure
        assert returned_axis is axis
        assert figure._suptitle.get_text() == "test mode"
        show.assert_called_once()
    finally:
        plt.close("all")


@pytest.mark.parametrize("dimension", [1, 2])
def test_mode_plot_grid_and_component_validation(dimension):
    result = (_make_1d_result if dimension == 1 else _make_2d_result)()
    with pytest.raises(ValueError, match="Unknown component"):
        result.plot_mode(components=["absent"])
    result.z_grid = None
    with pytest.raises(ValueError, match="z_grid must be set"):
        result.plot_mode(components="Ey", show=False)
    result.x_grid = result.y_grid = None
    if dimension == 2:
        with pytest.raises(ValueError, match="No horizontal grid"):
            result._get_horizontal_grid()
    with pytest.raises(ValueError, match="No stack"):
        result._compute_index_profile_2d()
    plt.close("all")


@pytest.mark.parametrize("invalid", ["air", "missing", "zero", "no-permittivity"])
def test_index_profile_ignores_non_optical_layers(index_context, monkeypatch, invalid):
    from types import SimpleNamespace

    from gsim.meep import mode_solver

    stack, _ = index_context
    stack.layers["extra"] = stack.layers["core"].model_copy(
        update={"material": invalid}
    )
    properties = {
        "core": SimpleNamespace(epsilon_diag=4),
        "clad": SimpleNamespace(epsilon_diag=[2.25] * 3),
        "zero": SimpleNamespace(epsilon_diag=[0] * 3),
        "no-permittivity": SimpleNamespace(epsilon_diag=None),
    }
    monkeypatch.setattr(
        mode_solver, "_resolve_materials_cached", lambda *_args, **_kw: properties
    )
    profile = mode_solver.refractive_index_profile(
        stack, 1.55, z_grid=np.array([-0.5, 0.5, 2]), background_material="clad"
    )
    np.testing.assert_allclose(profile, [1.5, 2, 1.5])


@pytest.mark.parametrize("plane", ["xz", "yz", None])
@pytest.mark.parametrize("position", [None, "missing", (0, 0), (100, 100)])
def test_mode_geometry_overlay_uses_stored_slice(index_context, plane, position):
    stack, component = index_context
    result = _make_2d_result(
        stack=stack,
        component=component,
        cross_section_plane=plane,
        port_or_position=position,
    )
    figure, axis = plt.subplots()
    try:
        result._draw_geometry_overlay_2d(axis)
        expected = plane is not None and position == (0, 0)
        assert bool(axis.patches) == expected
    finally:
        plt.close(figure)
