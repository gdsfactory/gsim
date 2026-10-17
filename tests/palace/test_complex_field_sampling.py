"""Check complex field sampling against constant analytical vector fields."""

import numpy as np
import pytest
import pyvista as pv

from gsim.palace import field_viz


@pytest.fixture
def complex_grid():
    grid = pv.ImageData(dimensions=(5, 5, 5), spacing=(1, 1, 1), origin=(-2, -2, -2))
    grid["E_real"] = np.tile([3.0, 4.0, 12.0], (grid.n_points, 1))
    grid["E_imag"] = grid["E_real"].copy()
    return grid


@pytest.mark.parametrize("normal", ["x", "y", "z"])
@pytest.mark.parametrize("components", [2, 3])
@pytest.mark.parametrize("field_seeds", [False, True])
def test_complex_streamlines_preserve_tangential_and_normal_magnitudes(
    complex_grid, normal, components, field_seeds
):
    if components == 2:
        complex_grid["E_real"] = complex_grid["E_real"][:, :2]
        complex_grid["E_imag"] = complex_grid["E_imag"][:, :2]
        complex_grid["En_real"] = np.full(complex_grid.n_points, 12.0)
        complex_grid["En_imag"] = np.full(complex_grid.n_points, 12.0)
    result = field_viz.extract_streamplot_inputs_2d(
        complex_grid,
        normal=normal,
        grid_resolution=(9, 9),
        streamplot_seed_from_field=field_seeds,
        streamplot_density=4,
    )
    normal_magnitude = 12 if components == 2 else {"x": 3, "y": 4, "z": 12}[normal]
    tangent_squared = 25 if components == 2 else 169 - normal_magnitude**2
    assert np.nanmax(result.et_mag) == pytest.approx(np.sqrt(2 * tangent_squared))
    assert np.nanmax(result.en_mag) == pytest.approx(normal_magnitude * np.sqrt(2))
    magnitude = np.hypot(result.u, result.v)
    np.testing.assert_allclose(magnitude[np.isfinite(magnitude) & (magnitude > 0)], 1)
    assert result.normal == normal
    assert result.start_points.shape[1] == 2


@pytest.mark.parametrize("normal", ["x", "y", "z"])
@pytest.mark.parametrize("scalar", ["real", "x", "y", "z"])
def test_field_plot_phase_lock_preserves_component_amplitude(
    monkeypatch, complex_grid, normal, scalar
):
    displayed = []

    def display(plotter, **_kwargs):
        displayed.append(plotter.mapper.dataset[plotter.mapper.array_name].copy())
        plotter.close()

    monkeypatch.setattr("gsim.viz._show_or_screenshot", display)
    field_viz.plot_fields_2d(complex_grid, normal=normal, scalar=scalar)
    component = ("y" if normal == "x" else "x") if scalar == "real" else scalar
    expected = {"x": 3, "y": 4, "z": 12}[component] * np.sqrt(2)
    np.testing.assert_allclose(displayed[0], expected)


@pytest.mark.parametrize("field", ["missing", "scalar", "vector"])
def test_streamlines_reject_missing_or_nonvector_fields(complex_grid, field):
    complex_grid["scalar"] = np.ones(complex_grid.n_points)
    complex_grid["vector"] = np.ones((complex_grid.n_points, 4))
    with pytest.raises(ValueError, match=r"not found|2- or 3-component"):
        field_viz.extract_streamplot_inputs_2d(complex_grid, field=field)
