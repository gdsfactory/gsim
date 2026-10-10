"""Tests for solid mesh visualization cell-block normalization."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import meshio
import numpy as np
import pytest
import pyvista as pv

from gsim import viz
from gsim.viz import _aligned_block_tags, _normalize_solid_cell_block

_SKIP_RENDER_ON_WIN = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="PyVista off-screen rendering is unstable on Windows CI",
)


def test_normalize_quadratic_triangle_preserves_order() -> None:
    """triangle6 should map to QUADRATIC_TRIANGLE without linearization."""
    cells = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)

    normalized = _normalize_solid_cell_block("triangle6", cells)

    assert normalized is not None
    block, cell_type, topo_dim = normalized
    assert block.shape == (1, 6)
    assert cell_type == pv.CellType.QUADRATIC_TRIANGLE
    assert topo_dim == 2


def test_unknown_high_order_triangle_falls_back_to_linear() -> None:
    """Unsupported triangle variants should safely linearize for rendering."""
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)

    normalized = _normalize_solid_cell_block("triangle10", cells)

    assert normalized is not None
    block, cell_type, topo_dim = normalized
    assert block.shape == (1, 3)
    np.testing.assert_array_equal(block[0], np.array([0, 1, 2], dtype=np.int64))
    assert cell_type == pv.CellType.TRIANGLE
    assert topo_dim == 2


def test_normalize_quadratic_tetra_preserves_order() -> None:
    """tetra10 should map to QUADRATIC_TETRA for volume fallback plotting."""
    cells = np.array([list(range(10))], dtype=np.int64)

    normalized = _normalize_solid_cell_block("tetra10", cells)

    assert normalized is not None
    block, cell_type, topo_dim = normalized
    assert block.shape == (1, 10)
    assert cell_type == pv.CellType.QUADRATIC_TETRA
    assert topo_dim == 3


def test_normalize_empty_block_returns_none() -> None:
    """Empty or non-2D blocks should return ``None`` without raising."""
    assert (
        _normalize_solid_cell_block("triangle", np.empty((0, 3), dtype=np.int64))
        is None
    )
    assert (
        _normalize_solid_cell_block("triangle", np.array([0, 1, 2], dtype=np.int64))
        is None
    )


def test_normalize_known_type_with_wrong_node_count_returns_none() -> None:
    """Known cell type with mismatching node count should be skipped."""
    cells = np.array([[0, 1, 2, 3]], dtype=np.int64)  # triangle expects 3
    assert _normalize_solid_cell_block("triangle", cells) is None


def test_normalize_unknown_quad_linearizes_to_quad() -> None:
    """Unknown high-order quad variants should fall back to linear QUAD."""
    cells = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int64
    )
    normalized = _normalize_solid_cell_block("quad16", cells)
    assert normalized is not None
    block, cell_type, topo_dim = normalized
    assert block.shape == (1, 4)
    np.testing.assert_array_equal(block[0], np.array([0, 1, 2, 3], dtype=np.int64))
    assert cell_type == pv.CellType.QUAD
    assert topo_dim == 2


def test_normalize_completely_unknown_type_returns_none() -> None:
    """Cell types that are neither known nor triangle/quad-like return None."""
    cells = np.array([[0, 1]], dtype=np.int64)
    assert _normalize_solid_cell_block("line", cells) is None


def test_aligned_block_tags_handles_all_alignment_cases() -> None:
    """Padding, truncating, exact match, and missing index are all handled."""
    phys = [np.array([10, 20, 30], dtype=int)]

    # Exact size
    np.testing.assert_array_equal(_aligned_block_tags(phys, 0, 3), [10, 20, 30])

    # Pad with -1 when tags are too short
    padded = _aligned_block_tags(phys, 0, 5)
    np.testing.assert_array_equal(padded, [10, 20, 30, -1, -1])

    # Truncate when tags are longer than n_cells
    np.testing.assert_array_equal(_aligned_block_tags(phys, 0, 2), [10, 20])

    # Out-of-range idx -> all -1
    np.testing.assert_array_equal(_aligned_block_tags(phys, 5, 4), [-1, -1, -1, -1])


def _write_minimal_msh(
    path: Path,
    *,
    use_3d: bool = False,
    extra_cells: bool = False,
) -> None:
    """Write a minimal gmsh22 file for solid-renderer tests."""
    cells: list[tuple[str, np.ndarray] | meshio.CellBlock]
    if use_3d:
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = [("tetra", np.array([[0, 1, 2, 3]]))]
        cell_data: dict[str, list[np.ndarray]] = {
            "gmsh:physical": [np.array([1])],
            "gmsh:geometrical": [np.array([1])],
        }
        field_data = {"bulk": np.array([1, 3])}
    else:
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        cells = [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))]
        phys: list[np.ndarray] = [np.array([1, 2])]
        if extra_cells:
            # Add an unsupported cell block to exercise the skip-path.
            cells.append(("line", np.array([[0, 1]])))
            phys.append(np.array([99]))
        geom: list[np.ndarray] = [np.array([1, 1])]
        if extra_cells:
            geom.append(np.array([1]))
        cell_data = {
            "gmsh:physical": phys,
            "gmsh:geometrical": geom,
        }
        field_data = {
            "metal": np.array([1, 2]),
            "air_boundary": np.array([2, 2]),
        }

    mesh = meshio.Mesh(pts, cells, cell_data=cell_data, field_data=field_data)  # ty: ignore[invalid-argument-type]
    mesh.write(str(path), file_format="gmsh22")


@_SKIP_RENDER_ON_WIN
def test_plot_solid_renders_surface_groups(tmp_path: Path) -> None:
    """End-to-end smoke test for solid-mode rendering with transparency."""
    msh = tmp_path / "solid.msh"
    _write_minimal_msh(msh)
    out = tmp_path / "solid.png"

    viz.plot_mesh(
        msh,
        output=out,
        interactive=False,
        style="solid",
        transparent_groups=["air_boundary"],
    )

    assert out.exists()


@_SKIP_RENDER_ON_WIN
def test_plot_solid_falls_back_to_volume_cells(tmp_path: Path) -> None:
    """When only 3D cells exist they should still render via the volume path."""
    msh = tmp_path / "vol.msh"
    _write_minimal_msh(msh, use_3d=True)
    out = tmp_path / "vol.png"

    viz.plot_mesh(msh, output=out, interactive=False, style="solid")

    assert out.exists()


@_SKIP_RENDER_ON_WIN
def test_plot_solid_skips_unsupported_blocks(tmp_path: Path) -> None:
    """Unsupported cell blocks should be skipped without aborting the render."""
    msh = tmp_path / "mixed.msh"
    _write_minimal_msh(msh, extra_cells=True)
    out = tmp_path / "mixed.png"

    viz.plot_mesh(msh, output=out, interactive=False, style="solid")

    assert out.exists()


@_SKIP_RENDER_ON_WIN
def test_plot_mesh_2d_camera_faces_plane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plot_mesh on a 2D mesh points the camera straight at the plane."""
    msh = tmp_path / "flat.msh"
    _write_minimal_msh(msh)

    captured: dict[str, object] = {}

    def _spy(plotter: Any, **_kwargs: object) -> None:
        captured["camera"] = plotter.camera
        plotter.close()

    monkeypatch.setattr(viz, "_show_or_screenshot", _spy)

    viz.plot_mesh(msh, interactive=True, mode="live")

    camera = cast(pv.Camera, captured["camera"])
    view = np.array(camera.position) - np.array(camera.focal_point)
    view /= np.linalg.norm(view)
    # The fixture mesh is thin along z, so the camera must face +z/-z.
    assert np.allclose(np.abs(view), [0.0, 0.0, 1.0])


@_SKIP_RENDER_ON_WIN
def test_plot_mesh_3d_keeps_iso_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plot_mesh on a 3D mesh keeps the default isometric view."""
    msh = tmp_path / "vol.msh"
    _write_minimal_msh(msh, use_3d=True)

    captured: dict[str, object] = {}

    def _spy(plotter: Any, **_kwargs: object) -> None:
        captured["camera"] = plotter.camera
        plotter.close()

    monkeypatch.setattr(viz, "_show_or_screenshot", _spy)

    viz.plot_mesh(msh, interactive=True, mode="live")

    camera = cast(pv.Camera, captured["camera"])
    view = np.array(camera.position) - np.array(camera.focal_point)
    view /= np.linalg.norm(view)
    # Isometric view is not aligned to a single axis.
    assert sum(abs(d) > 0.5 for d in view) > 1


class _FakePlotter:
    """Minimal stand-in for a PyVista plotter in mode-resolution tests."""

    _id_counter = 0
    camera_position = None

    def __init__(self) -> None:
        self.shown = 0
        self.screenshotted = 0
        self.closed = 0
        self.notebook = False
        self._id_name = f"fake-{_FakePlotter._id_counter}"
        _FakePlotter._id_counter += 1

    def show_axes(self) -> None:
        return None

    def render(self) -> None:
        return None

    def show(self, **_kwargs: object) -> _FakePlotter:
        self.shown += 1
        return self

    def screenshot(self, _path: object) -> None:
        self.screenshotted += 1

    def close(self) -> None:
        self.closed += 1


def test_show_or_screenshot_static_when_interactive_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the session flag off, auto (interactive=None) renders a static PNG."""
    viz._LIVE_PLOTTERS.clear()
    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", False)
    monkeypatch.setattr(viz, "_in_notebook", lambda: False)

    plotter = _FakePlotter()
    viz._show_or_screenshot(plotter, interactive=None, mode="auto")
    assert plotter.shown == 0
    assert plotter.screenshotted == 1
    assert viz._LIVE_PLOTTERS == {}


def test_interactive_true_forces_live_even_when_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """interactive=True forces a live view even when the session flag is off."""
    viz._LIVE_PLOTTERS.clear()
    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", False)
    monkeypatch.setattr(viz, "_in_notebook", lambda: False)

    plotter = _FakePlotter()
    viz._show_or_screenshot(plotter, interactive=True, mode="auto")
    assert plotter.shown == 1
    assert plotter.screenshotted == 0
    assert {plotter._id_name: plotter} == viz._LIVE_PLOTTERS


def test_desktop_live_replaces_previous(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside a notebook, a live plot replaces any previously open live view."""
    viz._LIVE_PLOTTERS.clear()
    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", True)
    monkeypatch.setattr(viz, "_in_notebook", lambda: False)

    first = _FakePlotter()
    viz._show_or_screenshot(first, interactive=True, mode="auto")
    assert first.shown == 1
    assert first.screenshotted == 0
    assert {first._id_name: first} == viz._LIVE_PLOTTERS

    second = _FakePlotter()
    viz._show_or_screenshot(second, interactive=True, mode="auto")
    assert second.shown == 1
    assert first.closed == 1
    assert {second._id_name: second} == viz._LIVE_PLOTTERS


def test_notebook_keeps_multiple_views(monkeypatch: pytest.MonkeyPatch) -> None:
    """In a notebook each interactive plot adds a widget to the shared trame server."""
    viz._LIVE_PLOTTERS.clear()
    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", True)
    monkeypatch.setattr(viz, "_in_notebook", lambda: True)

    first = _FakePlotter()
    first.notebook = True
    viz._show_or_screenshot(first, interactive=True, mode="auto")
    assert first.shown == 1
    assert first.closed == 0

    second = _FakePlotter()
    second.notebook = True
    viz._show_or_screenshot(second, interactive=True, mode="auto")
    assert second.shown == 1
    assert second.closed == 0
    assert first.closed == 0  # previous view stays alive
    assert len(viz._LIVE_PLOTTERS) == 2


def test_notebook_uses_trame_backend_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notebook live views use server-side trame rendering by default."""
    viz._LIVE_PLOTTERS.clear()
    monkeypatch.setattr(viz, "_in_notebook", lambda: True)
    monkeypatch.setattr(viz, "_TRAME_BACKEND", "trame")

    seen: dict[str, object] = {}
    plotter = _FakePlotter()
    plotter.notebook = True

    def _show(**_kwargs: object) -> _FakePlotter:
        seen["backend"] = _kwargs.get("jupyter_backend")
        return plotter

    monkeypatch.setattr(plotter, "show", _show)

    viz._show_or_screenshot(plotter, interactive=True, mode="live")
    assert seen["backend"] == "trame"
    assert {plotter._id_name: plotter} == viz._LIVE_PLOTTERS


def test_notebook_live_view_silences_trame_debug_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trame's template/state debug loggers are quiet while a live view is built.

    The Vue-template tokenisation trame logs at INFO on every widget render
    floods notebook consoles.  ``_show_live`` must raise those loggers to
    WARNING for the duration of ``show()`` and restore them afterwards.
    """
    import logging

    noisy = ("trame", "trame_client", "trame_server", "trame_vtk", "trame_vuetify")
    for name in noisy:
        logging.getLogger(name).setLevel(logging.NOTSET)
    try:
        viz._LIVE_PLOTTERS.clear()
        monkeypatch.setattr(viz, "_in_notebook", lambda: True)
        monkeypatch.setattr(viz, "_TRAME_BACKEND", "trame")

        seen: dict[str, object] = {}
        plotter = _FakePlotter()
        plotter.notebook = True

        def _show(**_kwargs: object) -> _FakePlotter:
            seen["levels"] = {name: logging.getLogger(name).level for name in noisy}
            return plotter

        monkeypatch.setattr(plotter, "show", _show)

        viz._show_or_screenshot(plotter, interactive=True, mode="live")

        levels = cast(dict[str, int], seen["levels"])
        for name, level in levels.items():
            assert level >= logging.WARNING, f"{name} not silenced (level={level})"

        # Levels restored after the live view is created.
        for name in noisy:
            assert logging.getLogger(name).level == logging.NOTSET
        assert plotter._id_name in viz._LIVE_PLOTTERS
    finally:
        for name in noisy:
            logging.getLogger(name).setLevel(logging.NOTSET)


def test_set_trame_backend_validates() -> None:
    """set_trame_backend accepts trame/client and rejects anything else."""
    viz.set_trame_backend("client")
    assert viz._TRAME_BACKEND == "client"
    viz.set_trame_backend("trame")
    assert viz._TRAME_BACKEND == "trame"
    with pytest.raises(ValueError, match="backend"):
        viz.set_trame_backend("bogus")  # ty: ignore[invalid-argument-type]


def test_notebook_widget_failure_falls_back_to_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the interactive widget cannot be created, show a static image instead."""
    viz._LIVE_PLOTTERS.clear()
    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", True)
    monkeypatch.setattr(viz, "_in_notebook", lambda: True)

    plotter = _FakePlotter()
    plotter.notebook = True

    def _broken_show(**_kwargs: object) -> None:
        raise RuntimeError("trame widget failed")

    monkeypatch.setattr(plotter, "show", _broken_show)

    viz._show_or_screenshot(plotter, interactive=True, mode="auto")
    assert plotter.shown == 0
    assert plotter.screenshotted == 1
    assert plotter.closed == 1
    assert viz._LIVE_PLOTTERS == {}


def test_mode_live_forces_show_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    """mode='live' forces a live view, replacing any open view outside a notebook."""
    viz._LIVE_PLOTTERS.clear()
    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", False)
    monkeypatch.setattr(viz, "_in_notebook", lambda: False)

    prev = _FakePlotter()
    viz._LIVE_PLOTTERS[prev._id_name] = prev

    forced = _FakePlotter()
    viz._show_or_screenshot(forced, interactive=True, mode="live")
    assert forced.shown == 1
    assert prev.closed == 1


def test_mode_static_always_screenshots(monkeypatch: pytest.MonkeyPatch) -> None:
    """mode='static' renders a screenshot even before any live view."""
    viz._LIVE_PLOTTERS.clear()
    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", True)

    static = _FakePlotter()
    viz._show_or_screenshot(static, interactive=True, mode="static")
    assert static.shown == 0
    assert static.screenshotted == 1
    assert viz._LIVE_PLOTTERS == {}


def test_interactive_false_screenshots_even_in_live_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """interactive=False forces a static screenshot even when mode='live'."""
    viz._LIVE_PLOTTERS.clear()
    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", True)
    monkeypatch.setattr(viz, "_in_notebook", lambda: True)

    plotter = _FakePlotter()
    viz._show_or_screenshot(plotter, interactive=False, mode="live")
    assert plotter.shown == 0
    assert plotter.screenshotted == 1
    assert viz._LIVE_PLOTTERS == {}


def test_close_interactive_views() -> None:
    """close_interactive_views closes every registered view and clears the registry."""
    viz._LIVE_PLOTTERS.clear()
    a, b = _FakePlotter(), _FakePlotter()
    viz._LIVE_PLOTTERS[a._id_name] = a
    viz._LIVE_PLOTTERS[b._id_name] = b

    viz.close_interactive_views()
    assert a.closed == 1
    assert b.closed == 1
    assert viz._LIVE_PLOTTERS == {}


def test_close_interactive_view_single() -> None:
    """close_interactive_view closes one registered view by id."""
    viz._LIVE_PLOTTERS.clear()
    a, b = _FakePlotter(), _FakePlotter()
    viz._LIVE_PLOTTERS[a._id_name] = a
    viz._LIVE_PLOTTERS[b._id_name] = b

    viz.close_interactive_view(a._id_name)
    assert a.closed == 1
    assert b.closed == 0
    assert a._id_name not in viz._LIVE_PLOTTERS


def test_interactive_views_lists_ids() -> None:
    """interactive_views returns the ids of all open views."""
    viz._LIVE_PLOTTERS.clear()
    a = _FakePlotter()
    viz._LIVE_PLOTTERS[a._id_name] = a
    assert viz.interactive_views() == (a._id_name,)


def test_mode_for_respects_interactive_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto is static by default and live only when interactive mode is enabled."""
    monkeypatch.setattr(viz, "_VIZ_MODE", "auto")

    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", False)
    assert viz._mode_for("auto") == "static"

    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", True)
    assert viz._mode_for("auto") == "live"

    monkeypatch.setattr(viz, "_VIZ_MODE", "static")
    assert viz._mode_for("auto") == "static"

    monkeypatch.setattr(viz, "_VIZ_MODE", "live")
    assert viz._mode_for("auto") == "live"

    monkeypatch.setattr(viz, "_VIZ_MODE", "auto")
    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", False)
    assert viz._mode_for("auto") == "static"
    assert viz._mode_for("live") == "live"
    assert viz._mode_for("static") == "static"
    monkeypatch.setattr(viz, "_INTERACTIVE_MODE", False)


def test_set_interactive_mode_toggles_flag() -> None:
    """The interactive mode flag is off by default and toggles via the setter."""
    viz.set_interactive_mode(False)
    assert viz.interactive_mode() is False
    viz.set_interactive_mode(True)
    assert viz.interactive_mode() is True
    viz.set_interactive_mode(False)
    assert viz.interactive_mode() is False
