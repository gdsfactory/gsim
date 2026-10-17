"""Exercise field selection and interpolation on real VTK datasets."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import pyvista as pv

from gsim.palace import fields
from tests.palace.test_eigenmode_fields import _write_parallel_cycle


@pytest.mark.parametrize(
    "selector",
    [{"attributes": [1]}, {"entity_names": "left"}, {"boundary_type": "PEC"}],
)
def test_parallel_field_selection(tmp_path, selector):
    for boundary in (False, True):
        name = "eigenmode_boundary" if boundary else "eigenmode"
        _write_parallel_cycle(
            tmp_path / "paraview" / name / "Cycle000001", mode=1, boundary=boundary
        )
    context = fields.build_selector_context(
        {
            "Boundaries": {
                "PEC": [{"Attributes": [1]}, None, {"Attributes": [2]}],
                "Empty": [],
            }
        },
        {"left": 1, "right": 2},
    )
    boundary = fields.load_boundary_field_data(
        tmp_path, context, step_index=1, **selector
    )
    volume = fields.load_volume_field_data(tmp_path, step_index=1)
    assert boundary.mesh.n_cells == (2 if "boundary_type" in selector else 1)
    assert volume.mesh.n_cells == 2
    for data in (boundary, volume):
        assert {"E_real", "E_imag"} <= set(data.point_arrays)
        assert "attribute" in data.cell_arrays
        assert data.timestep == 1
    with pytest.raises(ValueError, match="Provide one selector"):
        fields.load_boundary_field_data(tmp_path, context)


@pytest.fixture
def analytic_volume():
    grid = pv.ImageData(
        dimensions=(5, 5, 5), spacing=(0.5, 0.5, 0.5), origin=(-1, -1, -1)
    )
    grid["potential"] = grid.points @ [1, 2, 3]
    grid["vector"] = grid.points
    return fields.VolumeFieldData(grid, "analytic", 1, 1)


@pytest.mark.parametrize("kind", ["boundary", "slice", "contours"])
@pytest.mark.parametrize("screenshot", [False, True])
def test_render_preserves_analytic_fields(
    analytic_volume, monkeypatch, tmp_path, kind, screenshot
):
    plotter = Mock()
    monkeypatch.setattr(
        fields,
        "_require_pyvista",
        lambda: SimpleNamespace(Plotter=Mock(return_value=plotter)),
    )
    data = analytic_volume
    if kind == "boundary":
        data = fields.BoundaryFieldData(
            data.mesh.extract_surface(), "analytic", 1, 1, (1,)
        )
    render = {
        "boundary": fields.plot_boundary_field,
        "slice": fields.plot_volume_slice,
        "contours": fields.plot_volume_contours,
    }[kind]
    options = {"screenshot": tmp_path / "field.png"} if screenshot else {}
    assert render(data, scalar_field="potential", **options) is plotter
    displayed = plotter.add_mesh.call_args.args[0]
    assert displayed.n_points > 0
    np.testing.assert_allclose(
        displayed["potential"], displayed.points @ [1, 2, 3], atol=1e-6
    )
    if screenshot:
        plotter.screenshot.assert_called_once_with(str(tmp_path / "field.png"))
    else:
        plotter.screenshot.assert_not_called()


@pytest.mark.parametrize(("component", "index"), [("y", 1), ("z", 2)])
def test_plane_slice_and_vector_components(analytic_volume, component, index):
    sliced = fields.extract_plane_slice(analytic_volume, (0, 0, 0), (0, 0, 1))
    name = fields.activate_vector_component(sliced, "vector", component, "selected")
    np.testing.assert_allclose(sliced[name], sliced.points[:, index])
    contours = fields.extract_slice_contours(sliced, name, 5)
    assert (contours.n_points > 0) == (component == "y")
    with pytest.raises(KeyError, match="not found on slice"):
        fields.extract_slice_contours(sliced, "absent")
    with pytest.raises(ValueError, match="not a vector"):
        fields.activate_vector_component(sliced, "potential")
    with pytest.raises(KeyError, match="attribute"):
        fields.extract_boundary_cells(sliced, [1])


@pytest.mark.parametrize(
    ("values", "options", "expected"),
    [
        ([np.nan], {}, (0, 1)),
        ([0, 0], {"signed": True}, (-1, 1)),
        ([-2, -1], {}, (0, 1)),
        ([-1, 0], {"log_scale": True}, (1e-10, 1e-9)),
        ([2, 2], {"log_scale": True}, (2, 2.02)),
    ],
)
def test_degenerate_color_limits(values, options, expected):
    assert fields._auto_clim(np.array(values), **options) == pytest.approx(expected)


def test_missing_visualization_dependency(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(ImportError, match="pyvista is required"):
        fields._require_pyvista()


def test_logarithmic_field_requires_positive_values(analytic_volume):
    analytic_volume.mesh["negative"] = -np.ones(analytic_volume.mesh.n_points)
    with pytest.raises(ValueError, match="no positive values"):
        fields.plot_boundary_field(analytic_volume, "negative", log_scale=True)
