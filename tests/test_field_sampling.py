"""Check field interpolation and plotting using real VTK meshes."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyvista as pv

from gsim import viz


@pytest.fixture
def field_grid(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    grid = pv.ImageData(
        dimensions=(5, 5, 5), spacing=(0.5, 0.5, 0.5), origin=(-1, -1, -1)
    )
    x, y, z = grid.points.T
    grid["scalar"] = 10 + x + 2 * y + 3 * z
    grid["E_real"] = np.column_stack((x + 2, y + 3, z + 4))
    yield grid
    plt.close("all")


@pytest.mark.parametrize("component", [None, 0, 1, 2])
def test_topview_interpolation_preserves_vector_components(field_grid, component):
    x, y, values = viz.sample_topview_field(
        field_grid,
        field="E_real",
        z=0,
        component=component,
        pad=(0, 0),
        grid_resolution=(7, 5),
    )
    expected = [x + 2, y + 3, np.full_like(x, 4)]
    np.testing.assert_allclose(
        values,
        np.linalg.norm(expected, axis=0) if component is None else expected[component],
    )


@pytest.mark.parametrize("z", [None, 0])
def test_surface_sampling_detects_top_plane_and_material_attribute(z):
    surface = pv.Plane(
        center=(0, 0, 1), i_size=2, j_size=2, i_resolution=4, j_resolution=4
    )
    surface["Q_s_real"] = 10 + surface.points[:, 0] + 2 * surface.points[:, 1]
    surface.cell_data["attribute"] = np.full(surface.n_cells, 7)
    x, y, values = viz.sample_topview_field(
        surface,
        field="Q_s_real",
        z=z,
        pad=(0, 0),
        grid_resolution=(7, 5),
    )
    np.testing.assert_allclose(values, 10 + x + 2 * y)


@pytest.mark.parametrize(
    "options",
    [
        {"z": None},
        {"component": 3},
        {"component": -1},
        {"field": "scalar", "component": 0},
        {"field": "missing"},
        {"attribute_values": [99]},
    ],
)
def test_sampling_rejects_missing_fields_invalid_components_and_filters(
    field_grid, options
):
    kwargs = {"field": "E_real", "z": 0, "grid_resolution": (3, 3)} | options
    with pytest.raises(ValueError):
        viz.sample_topview_field(field_grid, **kwargs)


@pytest.mark.parametrize(
    ("direct", "log", "symmetric"),
    [
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_topview_plots_keep_scalar_values_and_requested_color_scale(
    field_grid, direct, log, symmetric
):
    dataset = field_grid
    if direct:
        dataset = pv.Plane(center=(0, 0, 1), i_size=2, j_size=2)
        dataset["scalar"] = 13 + dataset.points[:, 0] + 2 * dataset.points[:, 1]
    viz.plot_topview(
        dataset,
        field="scalar",
        z=1,
        title="Known linear field",
        surface_direct=direct,
        log=log,
        symmetric=symmetric,
        pad=(0, 0),
        grid_resolution=(7, 5),
    )
    axes = plt.gcf().axes[0]
    image = axes.collections[0]
    array = image.get_array()
    assert array is not None
    values = np.ma.masked_invalid(array).compressed()
    assert values.min() >= 10
    assert values.max() <= 16
    assert axes.get_title() == "Known linear field"
    if log:
        assert isinstance(image.norm, plt.matplotlib.colors.LogNorm)
    elif symmetric:
        assert image.norm.vmax is not None
        assert image.norm.vmin == -image.norm.vmax


@pytest.mark.parametrize("normal", ["x", "y", "z"])
@pytest.mark.parametrize("log", [False, True])
def test_cross_section_plot_matches_analytical_vector_magnitude(
    field_grid, normal, log
):
    viz.plot_cross_section(
        field_grid,
        normal=normal,
        field="E_real",
        origin=0,
        log=log,
        yi_range=(-1, 1),
        zi_range=(-1, 1),
        grid_resolution=(5, 5),
    )
    axes = plt.gcf().axes[0]
    horizontal, vertical = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    components = [horizontal + 2, vertical + 3, np.full_like(horizontal, 4)]
    if normal == "x":
        components = [np.full_like(horizontal, 2), horizontal + 3, vertical + 4]
    elif normal == "y":
        components = [horizontal + 2, np.full_like(horizontal, 3), vertical + 4]
    sampled = axes.collections[0].get_array()
    assert sampled is not None
    np.testing.assert_allclose(sampled, np.linalg.norm(components, axis=0))
    assert tuple(axes.get_xlim()) == (-1, 1)
    assert tuple(axes.get_ylim()) == (-1, 1)


def test_cross_section_handles_empty_slice_and_rejects_scalar_fields(
    field_grid, caplog
):
    viz.plot_cross_section(field_grid, origin=100)
    assert "returned 0 points" in caplog.text
    with pytest.raises(ValueError, match="3-component"):
        viz.plot_cross_section(field_grid, field="scalar")
    with pytest.raises(ValueError, match="not found"):
        viz.plot_cross_section(field_grid, field="missing")
