"""Validate planar fallbacks, scalar fields, and sparse streamline seeds."""

from unittest.mock import Mock

import numpy as np
import pytest
import pyvista as pv

from gsim import viz
from gsim.palace import field_viz


@pytest.fixture
def planar_fields():
    grid = pv.ImageData(dimensions=(5, 5, 1), spacing=(0.5, 0.5, 1), origin=(-1, -1, 0))
    grid["E_real"] = grid.points + np.array([2, 2, 1])
    grid["scalar"] = grid.points[:, 0] + 2
    grid.cell_data["attribute"] = np.ones(grid.n_cells, dtype=int)
    return grid


@pytest.mark.parametrize("component", [None, 0, 1])
def test_direct_topview_preserves_selected_vector_field(
    planar_fields, monkeypatch, component
):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    viz.plot_topview(
        planar_fields,
        field="E_real",
        title="analytic",
        component=component,
        attribute_values=[1],
        surface_direct=True,
    )
    assert len(plt.gcf().axes[0].collections) == 1
    plt.close("all")
    for function in (viz.sample_topview_field, viz.plot_topview):
        options = (
            {"title": "analytic", "surface_direct": True}
            if function is viz.plot_topview
            else {}
        )
        with pytest.raises(ValueError, match="No cells found"):
            function(planar_fields, field="E_real", attribute_values=[9], **options)
        del planar_fields.cell_data["attribute"]
        with pytest.raises(ValueError, match="attribute"):
            function(planar_fields, field="E_real", attribute_values=[1], **options)
        planar_fields.cell_data["attribute"] = np.ones(planar_fields.n_cells, dtype=int)


@pytest.mark.parametrize(
    ("normal", "labels"), [("x", ("y", "z")), ("y", ("x", "z")), ("z", ("x", "y"))]
)
def test_planar_slice_fallback_and_labels(planar_fields, normal, labels):
    sliced, used_normal, _, _ = field_viz._slice_plane(
        planar_fields, normal="z", origin=100
    )
    assert sliced.n_points == planar_fields.n_points
    assert used_normal == "z"
    assert field_viz._plane_axis_labels(normal) == labels
    volume = pv.ImageData(dimensions=(2, 2, 2))
    with pytest.raises(ValueError, match="No points found"):
        field_viz._slice_plane(volume, normal=normal, origin=100)


@pytest.mark.parametrize("seed_from_field", [False, True])
@pytest.mark.parametrize("magnitude", [0, 1])
def test_streamline_zero_field_and_seed_filter(
    planar_fields, magnitude, seed_from_field
):
    planar_fields["E_real"] = np.full((planar_fields.n_points, 3), float(magnitude))
    stream = field_viz.extract_streamplot_inputs_2d(
        planar_fields,
        normal="z",
        grid_resolution=(8, 8),
        streamplot_seed_from_field=seed_from_field,
        streamplot_seed_stride=100,
        streamplot_mask_weak=False,
        streamplot_normalize=False,
    )
    horizontal, vertical = np.meshgrid(stream.x, stream.y)
    inside = (np.abs(horizontal) <= 1) & (np.abs(vertical) <= 1)
    np.testing.assert_allclose(np.hypot(stream.u, stream.v)[inside], magnitude)
    assert np.all(np.isfinite(stream.v))
    filtered = field_viz._filter_start_points_in_bounds(
        np.array([[0, 0], [4, 4]]), x=np.array([-1, 1]), y=np.array([-1, 1])
    )
    np.testing.assert_array_equal(filtered, [[0, 0]])
    assert field_viz._filter_start_points_in_bounds(
        np.array([]), x=np.array([-1, 1]), y=np.array([-1, 1])
    ).shape == (0, 2)


def test_scalar_and_empty_field_diagnostics(planar_fields, tmp_path, monkeypatch):
    values, name, signed = field_viz._resolve_scalar_values(
        planar_fields, "scalar", "magnitude", [0, 1]
    )
    np.testing.assert_array_equal(values, planar_fields["scalar"])
    assert name == "scalar"
    assert not signed
    with pytest.raises(ValueError, match="needs a vector field"):
        field_viz._resolve_scalar_values(planar_fields, "scalar", "real", [0, 1])
    for values in (np.array([np.nan]), np.zeros(2)):
        assert field_viz._auto_clim(values, signed=True) is None
    with pytest.raises(FileNotFoundError):
        field_viz.resolve_physical_groups(tmp_path, ["core"])
    with pytest.raises(TypeError, match="requires a file path"):
        field_viz.plot_fields_2d(
            planar_fields, physical_groups=["core"], normal="z", show=False
        )
    monkeypatch.setattr(pv, "Plotter", Mock(return_value=Mock()))
    field_viz.plot_fields_2d(
        planar_fields,
        cycles=[1, 2],
        title=["first", "second"],
        normal="z",
        screenshot=tmp_path / "fields.png",
        show=False,
    )
