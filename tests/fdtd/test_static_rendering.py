"""Rendered material-boundary, hole, and vector-export regression checks."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.path import Path as MatplotlibPath

from gsim import fdtd
from gsim.fdtd.models import MeshGroup, MeshManifest
from gsim.fdtd.slice_viz import plot_mesh_slice
from tests.fdtd.test_static_viz import _tetrahedron_view
from tests.fdtd.test_viz import _write_mesh


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _ring_view(tmp_path, monkeypatch, *, show_mesh=False):
    from gsim.fdtd import slice_viz

    polygons = []
    # A 4-by-4 ring with a 2-by-2 hole, plus a disconnected unit square.
    cells = [
        (x, y) for x in range(4) for y in range(4) if not (1 <= x < 3 and 1 <= y < 3)
    ]
    cells.append((5, 0))
    for x, y in cells:
        polygons.extend(
            [
                np.asarray([(x, y), (x + 1, y), (x + 1, y + 1)]),
                np.asarray([(x, y), (x + 1, y + 1), (x, y + 1)]),
            ]
        )
    monkeypatch.setattr(slice_viz, "_mesh_sections", lambda *_args: {2: polygons})
    simulation = fdtd.Simulation(domain={"pml_cells": 0})
    cast(Any, simulation)._resolved = SimpleNamespace(ports={}, layers={})
    manifest = MeshManifest(
        {}, {"core": MeshGroup("core", 2, "Si", 1)}, {}, (0, 0, 0), (6000, 4000, 1000)
    )
    return plot_mesh_slice(
        _write_mesh(tmp_path / "mesh.msh"),
        manifest,
        {"Si": cast(Any, SimpleNamespace(refractive_index=3.5))},
        simulation,
        axis="z",
        position_um=0.5,
        show_mesh=show_mesh,
        legend=False,
    )


def _pixel_at(ax, x, y):
    pixels = np.asarray(ax.figure.canvas.buffer_rgba())
    column, row = np.rint(ax.transData.transform((x, y))).astype(int)
    return pixels[pixels.shape[0] - 1 - row, column, :3]


def test_material_regions_keep_holes_islands_and_shared_edges(tmp_path, monkeypatch):
    ax = _ring_view(tmp_path, monkeypatch)
    paths = ax.collections[0].get_paths()
    assert len(paths) == 2
    area = 0
    ring_count = 0
    for path in paths:
        starts = np.flatnonzero(path.codes == MatplotlibPath.MOVETO)
        for ring in np.split(path.vertices, starts[1:]):
            ring_count += 1
            area += (
                np.dot(ring[:, 0], np.roll(ring[:, 1], -1))
                - np.dot(ring[:, 1], np.roll(ring[:, 0], -1))
            ) / 2
    assert ring_count == 3
    assert area == pytest.approx(4 * 4 - 2 * 2 + 1)

    ax.figure.canvas.draw()
    # Actual rendered pixels retain the hole and the gap between the islands.
    np.testing.assert_array_equal(_pixel_at(ax, 2, 2), [255, 255, 255])
    np.testing.assert_array_equal(_pixel_at(ax, 4.5, 0.5), [255, 255, 255])
    material = _pixel_at(ax, 0.5, 0.25)
    assert np.all(material < 200)
    # Shared cell edges and the diagonal through a mesh cell have no white seams.
    for point in [(1, 0.5), (0.5, 0.5), (5.5, 0.5)]:
        np.testing.assert_array_equal(_pixel_at(ax, *point), material)


def test_sloping_material_boundary_has_antialiased_pixels(tmp_path):
    ax = _tetrahedron_view(
        tmp_path, axis="z", position_um=0.25, pml_cells=0, legend=False
    )
    canvas = ax.figure.canvas
    assert isinstance(canvas, FigureCanvasAgg)
    canvas.draw()
    # Inspect an interior crop crossing x+y=0.75, away from axes and overlays.
    pixels = np.asarray(canvas.buffer_rgba())
    lower, upper = ax.transData.transform([(0.2, 0.2), (0.55, 0.55)])
    x0, y0 = np.floor(lower).astype(int)
    x1, y1 = np.ceil(upper).astype(int)
    crop = pixels[pixels.shape[0] - y1 : pixels.shape[0] - y0, x0:x1, :3]
    fill = np.rint(np.asarray(ax.collections[0].get_facecolor())[0, :3] * 255)
    blended = np.all(crop > fill + 1, axis=2) & np.all(crop < 254, axis=2)
    assert np.count_nonzero(blended) > 10


def test_material_outlines_preserve_a_narrow_gap(tmp_path, monkeypatch):
    from gsim.fdtd import slice_viz

    gap = 2e-5
    left, right = 0.5 - gap / 2, 0.5 + gap / 2
    polygons = [
        np.asarray([(0, 0), (left, 0), (left, 1), (0, 1)]),
        np.asarray([(right, 0), (1, 0), (1, 1), (right, 1)]),
    ]
    monkeypatch.setattr(slice_viz, "_mesh_sections", lambda *_args: {2: polygons})
    ax = _tetrahedron_view(
        tmp_path, axis="z", position_um=0.25, pml_cells=0, legend=False
    )
    vertices = sorted(
        [np.asarray(path.vertices) for path in ax.collections[0].get_paths()],
        key=lambda coordinates: coordinates[:, 0].min(),
    )
    assert len(vertices) == 2
    assert vertices[0][:, 0].max() == pytest.approx(left, abs=1e-12)
    assert vertices[1][:, 0].min() == pytest.approx(right, abs=1e-12)
    assert vertices[1][:, 0].min() - vertices[0][:, 0].max() == pytest.approx(gap)


@pytest.mark.parametrize("show_mesh", [False, True])
def test_svg_keeps_material_outlines_vector_and_mesh_overlay_compact(
    tmp_path, monkeypatch, show_mesh
):
    ax = _ring_view(tmp_path, monkeypatch, show_mesh=show_mesh)
    ax.figure.axes[1].set_visible(False)  # Exclude the rasterized colorbar.
    output = StringIO()
    ax.figure.savefig(output, format="svg")
    assert ("<image" in output.getvalue()) is show_mesh
    assert len(ax.collections[0].get_paths()) == (26 if show_mesh else 2)


def test_plot_2d_accepts_matplotlib_options(fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, mesh_size_nm=750)
    simulation.geometry("straight")
    figure, ax = plt.subplots(figsize=(7, 3))
    result = simulation.plot_2d(
        ax=ax, axis="x", position_um=1, cmap="viridis", legend=False, aspect="auto"
    )
    assert result is ax
    assert ax.get_aspect() == "auto"
    assert not figure.legends
    np.testing.assert_allclose(
        np.asarray(ax.collections[-1].get_facecolor())[0], plt.colormaps["viridis"](1.0)
    )


def _group_view(tmp_path, monkeypatch, sections, groups, **kwargs):
    from gsim.fdtd import slice_viz

    monkeypatch.setattr(slice_viz, "_mesh_sections", lambda *_args: sections)
    simulation = fdtd.Simulation(domain={"pml_cells": 0})
    cast(Any, simulation)._resolved = SimpleNamespace(ports={}, layers={})
    manifest = MeshManifest({}, groups, {}, (0, 0, 0), (1000, 1000, 1000))
    return plot_mesh_slice(
        _write_mesh(tmp_path / "mesh.msh"),
        manifest,
        {
            "Si": cast(Any, SimpleNamespace(refractive_index=3.5)),
            "Oxide": cast(Any, SimpleNamespace(refractive_index=1.5)),
        },
        simulation,
        axis="z",
        position_um=0.5,
        **kwargs,
    )


def _rectangle(left, bottom, right, top):
    return np.asarray([(left, bottom), (right, bottom), (right, top), (left, top)])


@pytest.mark.parametrize("existing_axes", [False, True])
def test_overlapping_silicon_groups_have_uniform_edge_and_separate_legend(
    tmp_path, monkeypatch, existing_axes
):
    provided_ax = plt.subplots(figsize=(5, 3))[1] if existing_axes else None
    ax = _group_view(
        tmp_path,
        monkeypatch,
        {
            2: [_rectangle(0, 0.173, 1, 0.45)],
            3: [_rectangle(0.2, 0.173, 0.4, 0.85), _rectangle(0.6, 0.173, 0.8, 0.85)],
        },
        {
            "slab": MeshGroup("slab", 2, "Si", 1),
            "teeth": MeshGroup("teeth", 3, "Si", 2),
        },
        ax=provided_ax,
        figsize=(5, 3),
    )
    canvas = ax.figure.canvas
    assert isinstance(canvas, FigureCanvasAgg)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())
    lower, upper = ax.transData.transform([(0.1, 0.16), (0.9, 0.19)])
    x0, y0 = np.ceil(lower).astype(int)
    x1, y1 = np.floor(upper).astype(int)
    boundary = pixels[pixels.shape[0] - y1 : pixels.shape[0] - y0, x0:x1, :3]
    # The flat silicon bottom must have identical coverage under teeth and gaps.
    assert np.max(np.ptp(boundary.astype(int), axis=1)) == 0
    legend = ax.get_legend() if existing_axes else ax.figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "slab (Si)",
        "teeth (Si)",
        "Physical domain",
    ]


@pytest.mark.parametrize(("hidden", "top"), [("slab", 0.85), ("teeth", 0.45)])
def test_combined_fill_still_filters_each_physical_group(
    tmp_path, monkeypatch, hidden, top
):
    ax = _group_view(
        tmp_path,
        monkeypatch,
        {2: [_rectangle(0, 0.173, 1, 0.45)], 3: [_rectangle(0.2, 0.173, 0.4, 0.85)]},
        {
            "slab": MeshGroup("slab", 2, "Si", 1),
            "teeth": MeshGroup("teeth", 3, "Si", 2),
        },
        hide_groups=[hidden],
    )
    vertices = np.concatenate(
        [np.asarray(path.vertices) for path in ax.collections[0].get_paths()]
    )
    assert vertices[:, 1].max() == pytest.approx(top)
    names = [text.get_text() for text in ax.figure.legends[0].get_texts()]
    assert f"{hidden} (Si)" not in names


def test_equal_colors_do_not_merge_across_intervening_material_priority(
    tmp_path, monkeypatch
):
    ax = _group_view(
        tmp_path,
        monkeypatch,
        {
            2: [_rectangle(0.1, 0.1, 0.9, 0.9)],
            3: [_rectangle(0.1, 0.1, 0.9, 0.9)],
            4: [_rectangle(0.5, 0.1, 0.9, 0.9)],
        },
        {
            "lower": MeshGroup("lower", 2, "Si", 1),
            "oxide": MeshGroup("oxide", 3, "Oxide", 2),
            "upper": MeshGroup("upper", 4, "Si", 3),
        },
        legend=False,
    )
    ax.figure.canvas.draw()
    assert _pixel_at(ax, 0.25, 0.5)[0] > 100  # Oxide covers the lower silicon.
    assert _pixel_at(ax, 0.75, 0.5)[0] < 50  # Higher-priority silicon covers oxide.
