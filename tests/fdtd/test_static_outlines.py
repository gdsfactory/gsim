"""Analytic interfaces and field-axis preservation for outline-only previews."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

from gsim import fdtd
from gsim.fdtd.models import MeshGroup, MeshManifest
from gsim.fdtd.slice_viz import plot_mesh_slice
from tests.fdtd.test_static_viz import _tetrahedron_view


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _rectangle(xmin, ymin, xmax, ymax):
    return np.asarray([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])


def _outline_view(monkeypatch, regions, **kwargs):
    from gsim.fdtd import slice_viz

    groups = {}
    sections = {}
    for tag, (name, material, priority, polygons) in enumerate(regions, start=1):
        groups[name] = MeshGroup(name, tag, material, priority)
        sections[tag] = polygons
    monkeypatch.setattr(slice_viz, "_mesh_sections", lambda *_args: sections)
    manifest = MeshManifest({}, groups, {}, (0, 0, 0), (1000, 1000, 1000))
    return plot_mesh_slice(
        Path("unused.msh"),
        manifest,
        {},
        fdtd.Simulation(),
        axis="z",
        position_um=0.5,
        outline_only=True,
        **kwargs,
    )


def _outline_length(ax):
    return sum(
        np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()
        for collection in ax.collections
        for segment in collection.get_segments()
    )


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_outline_uses_actual_tetrahedron_section_without_setup_marks(tmp_path, axis):
    ax = _tetrahedron_view(tmp_path, axis=axis, position_um=0.25, outline_only=True)
    assert len(ax.figure.axes) == 1
    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], LineCollection)
    assert np.asarray(ax.collections[0].get_facecolor()).size == 0
    assert not ax.patches
    assert not ax.texts
    assert ax.get_legend() is None
    assert not ax.figure.legends
    assert ax.get_xlim() == pytest.approx((0, 1))
    assert ax.get_ylim() == pytest.approx((0, 1))
    # The analytic cut has two legs of 0.75 um and a hypotenuse of 0.75 sqrt(2).
    assert _outline_length(ax) == pytest.approx(1.5 + 0.75 * np.sqrt(2))
    output = StringIO()
    figure = ax.figure
    assert isinstance(figure, Figure)
    figure.savefig(output, format="svg")
    assert "<image" not in output.getvalue()


@pytest.mark.parametrize("same_material", [False, True])
def test_touching_materials_have_unique_interfaces_without_mesh_diagonals(
    monkeypatch, same_material
):
    left = [
        np.asarray([(0, 0), (0.5, 0), (0.5, 1)]),
        np.asarray([(0, 0), (0.5, 1), (0, 1)]),
    ]
    ax = _outline_view(
        monkeypatch,
        [
            ("left", "Si", 1, left),
            (
                "right",
                "Si" if same_material else "Oxide",
                2,
                [_rectangle(0.5, 0, 1, 1)],
            ),
        ],
    )
    # Unit-square perimeter plus exactly one interface for different materials.
    assert _outline_length(ax) == pytest.approx(4 if same_material else 5)


@pytest.mark.parametrize("inner_priority", [0, 2])
def test_higher_priority_material_hides_lower_priority_boundaries(
    monkeypatch, inner_priority
):
    ax = _outline_view(
        monkeypatch,
        [
            ("inner", "Si", inner_priority, [_rectangle(0.25, 0.25, 0.75, 0.75)]),
            ("outer", "Oxide", 1, [_rectangle(0, 0, 1, 1)]),
        ],
    )
    # The inner perimeter is visible only when its material wins the overlap.
    assert _outline_length(ax) == pytest.approx(4 if inner_priority == 0 else 6)


def test_outline_preserves_a_material_hole(monkeypatch):
    ax = _outline_view(
        monkeypatch,
        [
            (
                "ring",
                "Si",
                1,
                [
                    _rectangle(0, 0, 1, 0.25),
                    _rectangle(0, 0.75, 1, 1),
                    _rectangle(0, 0.25, 0.25, 0.75),
                    _rectangle(0.75, 0.25, 1, 0.75),
                ],
            )
        ],
    )
    assert _outline_length(ax) == pytest.approx(4 + 2)
    assert len(ax.collections[0].get_segments()) == 2


def test_outline_keeps_a_narrow_gap_between_equal_materials(monkeypatch):
    gap = 2e-5
    ax = _outline_view(
        monkeypatch,
        [
            ("left", "Si", 1, [_rectangle(0, 0, 0.5 - gap / 2, 1)]),
            ("right", "Si", 2, [_rectangle(0.5 + gap / 2, 0, 1, 1)]),
        ],
    )
    assert _outline_length(ax) == pytest.approx(6 - 2 * gap)
    assert len(ax.collections[0].get_segments()) == 2


@pytest.mark.parametrize(
    ("options", "length"),
    [
        ({"hide_groups": ["outer"]}, 2),
        ({"show_groups": ["inner"]}, 2),
        ({"show_groups": []}, 0),
        ({"hide_groups": ["inner", "outer"]}, 0),
    ],
)
def test_outline_group_filters(monkeypatch, options, length):
    ax = _outline_view(
        monkeypatch,
        [
            ("inner", "Si", 0, [_rectangle(0.25, 0.25, 0.75, 0.75)]),
            ("outer", "Oxide", 1, [_rectangle(0, 0, 1, 1)]),
        ],
        **options,
    )
    assert _outline_length(ax) == pytest.approx(length)


@pytest.mark.parametrize("method", ["plot_2d", "plot_2d_static"])
def test_public_outline_preserves_field_axes_and_colorbar(fdtd_pdk_module, method):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, mesh_size_nm=750)
    simulation.geometry("straight")
    figure, ax = plt.subplots()
    norm = SymLogNorm(linthresh=0.03, vmin=-1, vmax=1)
    values = np.arange(12).reshape(3, 4) / 12
    image = ax.pcolormesh(
        np.linspace(0, 2, 4),
        np.linspace(-0.5, 0.75, 3),
        values,
        norm=norm,
        cmap="RdBu_r",
        shading="auto",
        zorder=8,
    )
    colorbar = figure.colorbar(image, ax=ax, label="Ey")
    ax.plot([0, 2], [0, 0], label="reference")
    legend = ax.legend()
    ax.set(
        xlim=(0.1, 1.9),
        ylim=(-0.4, 0.6),
        xlabel="X",
        ylabel="Z",
        title="Re(Ey)",
        aspect=2,
    )
    limits = ax.dataLim.get_points().copy()
    result = getattr(simulation, method)(
        axis="y", position_um=0, ax=ax, outline_only=True
    )
    assert result is ax
    assert len(figure.axes) == 2
    assert colorbar.ax.get_ylabel() == "Ey"
    assert ax.get_legend() is legend
    assert ax.get_xlim() == pytest.approx((0.1, 1.9))
    assert ax.get_ylim() == pytest.approx((-0.4, 0.6))
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title(), ax.get_aspect()) == (
        "X",
        "Z",
        "Re(Ey)",
        2,
    )
    np.testing.assert_array_equal(ax.dataLim.get_points(), limits)
    np.testing.assert_array_equal(image.get_array(), values)
    assert image.norm is norm
    assert image.get_cmap().name == "RdBu_r"
    assert len(ax.collections) == 2
    assert isinstance(ax.collections[1], LineCollection)
    assert ax.collections[1].get_zorder() > image.get_zorder()
    assert not ax.texts
    assert not ax.patches
    assert simulation._last_artifacts is None


@pytest.mark.parametrize("method", ["plot_2d", "plot_2d_static"])
def test_outline_rejects_mesh_edges_before_meshing(monkeypatch, method):
    from gsim.fdtd import visualization

    monkeypatch.setattr(
        visualization, "generate_mesh", lambda *_args, **_kwargs: pytest.fail("meshed")
    )
    with pytest.raises(ValueError, match=r"outline_only.*show_mesh"):
        getattr(fdtd.Simulation(), method)(outline_only=True, show_mesh=True)
