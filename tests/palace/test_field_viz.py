"""Tests for 2D Palace field visualization utilities."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest
import pyvista as pv

import gsim.palace.field_viz as field_viz

mpl.use("Agg")


@pytest.fixture
def planar_vector_dataset() -> pv.DataSet:
    """Create a synthetic planar vector field dataset for plotting tests."""
    grid = pv.ImageData(
        dimensions=(41, 31, 1),
        spacing=(0.2, 0.2, 1.0),
        origin=(-4.0, -3.0, 0.0),
    )
    x = grid.points[:, 0]
    y = grid.points[:, 1]
    z = grid.points[:, 2]

    # Smooth rotational field with a weak out-of-plane component.
    e_real = np.column_stack([-y, x, 0.1 * z])
    grid.point_data["E_real"] = e_real
    return grid


def test_plot_fields_2d_dataset_smoke(planar_vector_dataset: pv.DataSet) -> None:
    """Matplotlib plotting utility should render from a dataset source."""
    fig, ax, out = field_viz.plot_fields_2d(
        planar_vector_dataset,
        normal="z",
        origin=0.0,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert out.u.shape == out.v.shape
    fig.clf()


def test_plot_fields_2d_loads_from_source(
    monkeypatch: pytest.MonkeyPatch,
    planar_vector_dataset: pv.DataSet,
) -> None:
    """Non-dataset source should call ``load_fields`` with requested args."""
    calls: dict[str, object] = {}

    def _fake_load_fields(
        source: str | dict,
        *,
        excitation: int,
        cycle: int | None,
        boundary: bool,
    ) -> pv.DataSet:
        calls["source"] = source
        calls["excitation"] = excitation
        calls["cycle"] = cycle
        calls["boundary"] = boundary
        return planar_vector_dataset

    monkeypatch.setattr(field_viz, "load_fields", _fake_load_fields)

    fig, _, _ = field_viz.plot_fields_2d(
        "dummy-output-dir",
        excitation=3,
        cycle=7,
        boundary=True,
        normal="z",
        origin=0.0,
        show=False,
    )

    assert calls == {
        "source": "dummy-output-dir",
        "excitation": 3,
        "cycle": 7,
        "boundary": True,
    }
    fig.clf()


def test_plot_fields_2d_missing_field_raises(planar_vector_dataset: pv.DataSet) -> None:
    """Missing field name should fail with a clear error."""
    with pytest.raises(ValueError, match="Field 'B_real' not found"):
        field_viz.plot_fields_2d(
            planar_vector_dataset,
            field="B_real",
            normal="z",
            origin=0.0,
            show=False,
        )


def test_extract_streamplot_inputs_2d(planar_vector_dataset: pv.DataSet) -> None:
    """Streamplot input extraction should return PalaceToolkit-style arrays."""
    out = field_viz.extract_streamplot_inputs_2d(
        planar_vector_dataset,
        field="E_real",
        normal="z",
        origin=0.0,
        grid_resolution=(40, 24),
    )

    assert out.x.ndim == 1
    assert out.y.ndim == 1
    assert out.u.ndim == 2
    assert out.v.ndim == 2
    assert out.u.shape == out.v.shape
    assert out.u.shape == (24, 40)
    assert out.start_points.ndim == 2
    assert out.start_points.shape[1] == 2


def test_axis_name_mapping() -> None:
    """Axis helper should map index to axis letter."""
    assert field_viz._axis_name(0) == "x"
    assert field_viz._axis_name(1) == "y"
    assert field_viz._axis_name(2) == "z"
