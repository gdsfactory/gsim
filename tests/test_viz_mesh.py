"""Tests for solid mesh visualization cell-block normalization."""

from __future__ import annotations

import sys
from pathlib import Path

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
