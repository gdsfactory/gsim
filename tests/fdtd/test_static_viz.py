"""Analytic geometry and public workflow checks for static FDTD previews."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from gsim import fdtd
from gsim.fdtd.models import MeshGroup, MeshManifest
from gsim.fdtd.slice_viz import plot_mesh_slice
from tests.fdtd.test_viz import _write_mesh


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _tetrahedron_view(tmp_path: Path, *, pml_cells: int = 2, **kwargs: Any) -> Axes:
    simulation = fdtd.Simulation(
        domain={"pml_cells": pml_cells}, solver={"cell_size_nm": 50}
    )
    cast(Any, simulation)._resolved = SimpleNamespace(ports={}, layers={})
    manifest = MeshManifest(
        volumes={},
        layers={"core": MeshGroup("core", 2, "Si", 1)},
        ports={},
        domain_min_nm=(0, 0, 0),
        domain_max_nm=(1000, 1000, 1000),
    )
    return plot_mesh_slice(
        _write_mesh(tmp_path / "mesh.msh"),
        manifest,
        {"Si": cast(Any, SimpleNamespace(refractive_index=3.5))},
        simulation,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("axis", "labels"), [("x", ("y", "z")), ("y", ("x", "z")), ("z", ("x", "y"))]
)
def test_slice_matches_analytic_tetrahedron(tmp_path, axis, labels):
    ax = _tetrahedron_view(tmp_path, axis=axis, position_um=0.25)
    collection = next(
        item for item in ax.collections if isinstance(item, PolyCollection)
    )
    vertices = np.asarray(collection.get_paths()[0].vertices)[:-1]
    # x+y+z <= 1 cut at 0.25 has legs 0.75 um and area 0.28125 um^2.
    np.testing.assert_allclose(
        sorted(map(tuple, vertices)), [(0, 0), (0, 0.75), (0.75, 0)]
    )
    area = (
        abs(
            np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1))
            - np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1))
        )
        / 2
    )
    assert area == pytest.approx(0.28125)
    assert ax.get_xlabel() == f"{labels[0]} (µm)"
    assert ax.get_ylabel() == f"{labels[1]} (µm)"
    np.testing.assert_allclose(
        np.asarray(collection.get_facecolor())[0], plt.colormaps["Blues"](1.0)
    )
    assert ax.figure.axes[1].get_ylabel() == "Refractive index n at \u03bb=1.55 µm"
    assert ax.get_xlim() == pytest.approx((-0.1, 1.1))
    assert ax.get_ylim() == pytest.approx((-0.1, 1.1) if axis == "z" else (-0.8, 1.8))
    assert [patch.get_label() for patch in ax.patches].count("PML") == 1


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("pml_cells", [0, 20])
def test_pml_rectangles_match_aligned_grid_thickness(tmp_path, axis, pml_cells):
    ax = _tetrahedron_view(tmp_path, axis=axis, position_um=0.25, pml_cells=pml_cells)
    # At 50 nm, 20 cells give 1 um in x/y and 32 aligned cells give 1.6 um in z.
    horizontal_pml = 1.0 if pml_cells else 0
    vertical_pml = (1.0 if axis == "z" else 1.6) if pml_cells else 0
    assert ax.get_xlim() == pytest.approx((-horizontal_pml, 1 + horizontal_pml))
    assert ax.get_ylim() == pytest.approx((-vertical_pml, 1 + vertical_pml))
    pml = [patch for patch in ax.patches if patch.get_hatch()]
    if not pml_cells:
        assert not pml
        return
    assert len(pml) == 4
    for patch, expected in zip(
        pml,
        [
            (-horizontal_pml, -vertical_pml, horizontal_pml, 1 + 2 * vertical_pml),
            (1, -vertical_pml, horizontal_pml, 1 + 2 * vertical_pml),
            (0, -vertical_pml, 1, vertical_pml),
            (0, 1, 1, vertical_pml),
        ],
        strict=True,
    ):
        assert isinstance(patch, Rectangle)
        assert (
            patch.get_x(),
            patch.get_y(),
            patch.get_width(),
            patch.get_height(),
        ) == pytest.approx(expected)


@pytest.mark.parametrize(("position", "paths"), [(0, 1), (1, 0)])
def test_slice_on_face_or_vertex(tmp_path, position, paths):
    ax = _tetrahedron_view(tmp_path, axis="z", position_um=position)
    assert sum(len(collection.get_paths()) for collection in ax.collections) == paths


def test_static_filter_mesh_edges_and_export(tmp_path):
    figure, ax = plt.subplots()
    result = _tetrahedron_view(
        tmp_path, axis="x", position_um=0.5, show_mesh=True, ax=ax, legend=False
    )
    assert result is ax
    assert ax.get_legend() is None
    assert not figure.legends
    assert np.asarray(ax.collections[0].get_edgecolor()).size
    output = tmp_path / "slice.png"
    figure.savefig(output)
    assert output.stat().st_size > 1000
    hidden = _tetrahedron_view(
        tmp_path, axis="z", position_um=0.5, hide_groups=["core"]
    )
    assert not hidden.collections
    filtered = _tetrahedron_view(tmp_path, axis="z", position_um=0.5, show_groups=[])
    assert not filtered.collections


@pytest.mark.parametrize("position", [-0.01, 1.01])
def test_outside_slice_has_clear_error(tmp_path, position):
    with pytest.raises(ValueError, match="outside"):
        _tetrahedron_view(tmp_path, axis="z", position_um=position)


@pytest.mark.parametrize("method", ["plot_2d_static", "plot_2d_interactive"])
@pytest.mark.parametrize(
    "kwargs",
    [{"axis": "bad"}, {"position_um": float("nan")}, {"position_um": float("inf")}],
)
def test_invalid_slice_is_rejected_before_geometry(method, kwargs):
    with pytest.raises(ValueError, match=r"axis must|position_um must be finite"):
        getattr(fdtd.Simulation(), method)(**kwargs)


def test_public_static_workflow_tracks_geometry_and_material_edits(
    tmp_path, fdtd_pdk_module
):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, mesh_size_nm=750)
    simulation.geometry("straight")
    # A preview should work before choosing a source port or writing a job.
    ax = simulation.plot_2d_static()
    assert isinstance(ax, Axes)
    assert "z = 0.11 µm" in ax.get_title()
    assert simulation._last_artifacts is None
    assert {text.get_text() for text in ax.texts} == {"Monitor o1", "Monitor o2"}
    simulation.source(port="o1")
    artifacts = simulation.write(tmp_path / "run")
    simulation.materials(overrides={"Si": {"refractive_index": 2.5}})
    simulation.geometry("straight", settings={"length": 3})
    simulation.add_monitors(xy_name="field_xy")
    updated = simulation.plot_2d_static(show_mesh=True)
    assert simulation._last_artifacts is artifacts
    assert updated.get_xlim()[1] == pytest.approx(3 + 32 * 0.06)
    assert {text.get_text() for text in updated.texts} >= {
        "Source o1",
        "Monitor o2",
        "Monitor field_xy",
    }
    assert updated.figure.axes[1].get_ylim()[1] == pytest.approx(2.5)


def test_sidewall_section_and_pml_units(fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, mesh_size_nm=750)
    simulation.geometry("straight")
    ax = simulation.plot_2d_static(axis="x", position_um=1, legend=False)
    core = next(
        collection
        for collection in ax.collections
        if collection.get_label() == "core (Si)"
    )
    vertices = np.concatenate([path.vertices for path in core.get_paths()])
    # 10 degree sidewalls, reference width 0.5 um at half the 0.22 um height.
    half_width_at_bottom = 0.25 + 0.11 * np.tan(np.deg2rad(10))
    assert vertices[:, 0].min() == pytest.approx(-half_width_at_bottom, abs=1e-6)
    assert vertices[:, 0].max() == pytest.approx(half_width_at_bottom, abs=1e-6)
    assert vertices[:, 1].min() == pytest.approx(0)
    assert vertices[:, 1].max() == pytest.approx(0.22)


def test_material_priority_controls_overlapping_sections(tmp_path, monkeypatch):
    from gsim.fdtd import slice_viz

    original_read = slice_viz.meshio.read

    def overlapping_mesh(path):
        mesh = original_read(path)
        mesh.cells.append(mesh.cells[0])
        mesh.cell_data["gmsh:physical"].append(np.asarray([3]))
        mesh.cell_data["gmsh:geometrical"].append(np.asarray([2]))
        return mesh

    monkeypatch.setattr(slice_viz.meshio, "read", overlapping_mesh)
    simulation = fdtd.Simulation()
    cast(Any, simulation)._resolved = SimpleNamespace(ports={}, layers={})
    high = MeshGroup("high", 2, "Si", 2)
    low = replace(high, name="low", physical_tag=3, priority=1)
    manifest = MeshManifest(
        {}, {"high": high, "low": low}, {}, (0, 0, 0), (1000, 1000, 1000)
    )
    ax = plot_mesh_slice(
        _write_mesh(tmp_path / "mesh.msh"),
        manifest,
        {"Si": cast(Any, SimpleNamespace(refractive_index=3.5))},
        simulation,
        axis="z",
        position_um=0.25,
    )
    assert [collection.get_label() for collection in ax.collections] == [
        "low (Si)",
        "high (Si)",
    ]


def test_quadrilateral_section_matches_analytic_area(tmp_path, monkeypatch):
    from gsim.fdtd import slice_viz

    original_read = slice_viz.meshio.read

    def rotated_tetrahedron(path):
        mesh = original_read(path)
        mesh.points = np.asarray(
            [(0, 0, 0), (1000, 0, 1000), (0, 1000, 1000), (1000, 1000, 0)]
        )
        return mesh

    monkeypatch.setattr(slice_viz.meshio, "read", rotated_tetrahedron)
    ax = _tetrahedron_view(tmp_path, axis="z", position_um=0.5)
    vertices = np.asarray(ax.collections[0].get_paths()[0].vertices)[:-1]
    np.testing.assert_allclose(
        sorted(map(tuple, vertices)), [(0, 0.5), (0.5, 0), (0.5, 1), (1, 0.5)]
    )
    area = (
        abs(
            np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1))
            - np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1))
        )
        / 2
    )
    assert area == pytest.approx(0.5)


@pytest.mark.parametrize(
    "source",
    [
        fdtd.DipoleSource(position_um=(1, 0, 0.11), current_axis="z"),
        fdtd.LineCurrentSource(
            position_um=(1, 0, 0.11), current_axis="z", line_axis="y", length_um=0.4
        ),
        fdtd.GaussianBeamSource(
            center_um=(1, 0, 0.11),
            size_um=(0.4, 0.4, 0),
            aperture_normal="-z",
            propagation_direction=(0, 0, -1),
            e_polarization=(1, 0, 0),
            focal_point_um=(1, 0, 0),
            waist_radius_um=0.1,
        ),
    ],
)
def test_non_port_source_only_appears_on_intersecting_slice(fdtd_pdk_module, source):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, source=source, mesh_size_nm=750)
    simulation.geometry("straight")
    simulation.monitors.add_plane(
        name="plane", center_um=(1, 0, 0.11), size_um=(0, 0.5, 0.22), normal="+x"
    )
    ax = simulation.plot_2d_static()
    assert {text.get_text() for text in ax.texts} >= {"Source", "Monitor plane"}
    off_plane = simulation.plot_2d_static(position_um=0.5)
    assert not off_plane.texts


def test_vertical_port_is_drawn_at_runtime_aperture(fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, mesh_size_nm=750)
    simulation.geometry("vertical_coupler")
    simulation.source(port="o2")
    ax = simulation.plot_2d_static(axis="y", position_um=0)
    labels = {text.get_text() for text in ax.texts}
    assert {"Monitor o1", "Source o2"} <= labels
    source_line = next(line for line in ax.lines if line.get_label() == "Source o2")
    np.testing.assert_allclose(np.asarray(source_line.get_ydata()), [1.22, 1.22])
    np.testing.assert_allclose(np.asarray(source_line.get_xdata()), [0.5, 1.5])
    filtered = simulation.plot_2d_static(
        axis="y", position_um=0, hide_groups=["port_o2"]
    )
    assert "Source o2" not in {text.get_text() for text in filtered.texts}


def test_failed_preview_removes_temporary_mesh_and_preserves_artifacts(
    fdtd_pdk_module, tmp_path, monkeypatch
):
    from gsim.fdtd import visualization

    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    previous_artifacts = SimpleNamespace(mesh_path=tmp_path / "old.msh")
    cast(Any, simulation)._last_artifacts = previous_artifacts
    mesh_paths = []

    def fail_meshing(_resolved, mesh_path, **_kwargs):
        mesh_paths.append(mesh_path)
        mesh_path.write_text("partial mesh")
        raise fdtd.FDTDGeometryError("meshing failed")

    monkeypatch.setattr(visualization, "generate_mesh", fail_meshing)
    with pytest.raises(fdtd.FDTDGeometryError, match="meshing failed"):
        simulation.plot_2d_static()
    assert simulation._last_artifacts is previous_artifacts
    assert len(mesh_paths) == 1
    assert not mesh_paths[0].parent.exists()


def test_non_port_source_does_not_reserve_vertical_monitor_names(fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, mesh_size_nm=750)
    simulation.geometry("vertical_coupler")
    simulation.source = fdtd.DipoleSource(position_um=(1, 0, 0.11), current_axis="z")
    simulation.monitors.add_plane(
        name="o2", center_um=(1, 0, 0.11), size_um=(1, 1, 0), normal="+z"
    )
    ax = simulation.plot_2d_static()
    assert {text.get_text() for text in ax.texts} >= {"Source", "Monitor o2"}
