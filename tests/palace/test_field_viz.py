"""Tests for 2D Palace field visualization utilities."""

from __future__ import annotations

from typing import Any

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
    """PyVista plotting utility should render from a dataset source."""
    plotter = field_viz.plot_fields_2d(
        planar_vector_dataset,
        normal="z",
        origin=0.0,
        show=False,
    )

    assert isinstance(plotter, pv.Plotter)
    plotter.close()


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

    plotter = field_viz.plot_fields_2d(
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
    plotter.close()


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


def test_plot_fields_2d_forwards_interactive_flag(
    monkeypatch: pytest.MonkeyPatch,
    planar_vector_dataset: pv.DataSet,
) -> None:
    """The ``interactive`` flag is forwarded to the display helper."""
    import gsim.viz as viz

    captured: dict[str, object] = {}

    def _spy(plotter: Any, **_kwargs: object) -> None:
        captured["interactive"] = _kwargs.get("interactive")
        plotter.close()

    monkeypatch.setattr(viz, "_show_or_screenshot", _spy)

    field_viz.plot_fields_2d(
        planar_vector_dataset,
        normal="z",
        origin=0.0,
        show=True,
        interactive=False,
    )
    assert captured["interactive"] is False


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


def test_plot_fields_2d_cycles_subplot_grid(
    monkeypatch: pytest.MonkeyPatch,
    planar_vector_dataset: pv.DataSet,
) -> None:
    """A list of ``cycles`` should render one panel per cycle."""
    loaded: list[tuple[str | dict, int, int | None, bool]] = []

    def _fake_load_fields(
        source: str | dict,
        *,
        excitation: int,
        cycle: int | None,
        boundary: bool,
    ) -> pv.DataSet:
        loaded.append((source, excitation, cycle, boundary))
        return planar_vector_dataset

    monkeypatch.setattr(field_viz, "load_fields", _fake_load_fields)

    plotter = field_viz.plot_fields_2d(
        "dummy-output-dir",
        cycles=[1, 2],
        normal="z",
        origin=0.0,
        show=False,
    )

    assert [item[2] for item in loaded] == [1, 2]
    assert len(plotter.renderers) == 2
    plotter.close()


def test_plot_fields_2d_scalar_real_is_signed(
    planar_vector_dataset: pv.DataSet,
) -> None:
    """Scalar selectors should resolve without a paired imag array."""
    values, name, signed = field_viz._resolve_scalar_values(
        planar_vector_dataset, "E_real", "real", [0, 1]
    )
    assert signed is True
    assert name.startswith("Re(E_real")
    assert values.shape[0] == planar_vector_dataset.n_points

    for scalar in ("magnitude", "z"):
        _, _, is_signed = field_viz._resolve_scalar_values(
            planar_vector_dataset, "E_real", scalar, [0, 1]
        )
        assert is_signed is (scalar != "magnitude")

    plotter = field_viz.plot_fields_2d(
        planar_vector_dataset,
        scalar="real",
        normal="z",
        origin=0.0,
        show=False,
    )
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


def test_plot_fields_2d_unknown_scalar_raises(
    planar_vector_dataset: pv.DataSet,
) -> None:
    """An unsupported scalar selector should fail clearly."""
    with pytest.raises(ValueError, match="Unknown scalar"):
        field_viz.plot_fields_2d(
            planar_vector_dataset,
            scalar="phase",
            normal="z",
            origin=0.0,
            show=False,
        )


def test_plot_fields_2d_chip_bounds_crops(
    planar_vector_dataset: pv.DataSet,
) -> None:
    """``chip_bounds`` should crop the in-plane extent on flat data.

    Native-2D data is thin in one axis, so ``_slice_plane`` keeps that plane
    and ignores ``normal``.  The crop must use the slice's own in-plane axes:
    with the fixture flat in z, ``normal="x"`` used to crop the wrong axes
    (and raise on the zero-thickness box).
    """
    for normal in ("z", "x"):
        plotter = field_viz.plot_fields_2d(
            planar_vector_dataset,
            normal=normal,
            origin=0.0,
            chip_bounds=(-1.0, 1.0, -1.0, 1.0),
            show=False,
        )
        assert isinstance(plotter, pv.Plotter)
        plotter.close()

    with pytest.raises(ValueError, match="removed all data"):
        field_viz.plot_fields_2d(
            planar_vector_dataset,
            normal="z",
            origin=0.0,
            chip_bounds=(100.0, 101.0, 100.0, 101.0),
            show=False,
        )


def test_crop_inplane_uses_slice_axes(planar_vector_dataset: pv.DataSet) -> None:
    """The crop follows the slice axes, not the requested ``normal``."""
    sliced, _used_normal, _axis_idx, axes = field_viz._slice_plane(
        planar_vector_dataset,
        normal="x",  # mismatches the fixture's flat (z) axis
        origin=0.0,
    )
    assert axes == [0, 1]  # in-plane axes are x, y
    cropped = field_viz._crop_inplane(
        sliced, axes=axes, chip_bounds=(-1.0, 1.0, -1.0, 1.0)
    )
    assert 0 < cropped.n_points < sliced.n_points
    points = np.asarray(cropped.points)
    # Points are kept to within one cell of the box on the two in-plane axes.
    assert np.abs(points[:, 0]).max() <= 1.3
    assert np.abs(points[:, 1]).max() <= 1.3
