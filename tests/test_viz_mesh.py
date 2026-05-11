"""Tests for solid mesh visualization cell-block normalization."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from gsim.viz import _normalize_solid_cell_block


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
