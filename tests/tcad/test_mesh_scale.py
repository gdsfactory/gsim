"""Tests for the um -> cm mesh rescaling used to feed DEVSIM."""

from __future__ import annotations

import meshio
import numpy as np
import pytest

from gsim.tcad.mesh import UM_TO_CM, write_scaled_msh


def _write_toy_msh(path):
    """Minimal msh v2.2 mesh with a named physical group."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    cells = [("triangle", np.array([[0, 1, 2]]))]
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data={
            "gmsh:physical": [np.array([7])],
            "gmsh:geometrical": [np.array([1])],
        },
        field_data={"slab": np.array([7, 2])},
    )
    meshio.write(str(path), mesh, file_format="gmsh22", binary=False)
    return points


def test_points_scaled_groups_preserved(tmp_path):
    src = tmp_path / "src.msh"
    dst = tmp_path / "dst.msh"
    points = _write_toy_msh(src)

    out = write_scaled_msh(src, dst, scale=UM_TO_CM)
    assert out == dst

    scaled = meshio.read(str(dst))
    np.testing.assert_allclose(scaled.points, points * 1e-4)
    assert "slab" in scaled.field_data
    assert scaled.cells[0].type == "triangle"
    assert len(scaled.cells[0].data) == 1
    physical = np.concatenate(
        [
            arr
            for block, arr in zip(
                scaled.cells, scaled.cell_data["gmsh:physical"], strict=True
            )
            if block.type == "triangle"
        ]
    )
    assert 7 in physical


def test_rejects_nonpositive_scale(tmp_path):
    src = tmp_path / "src.msh"
    _write_toy_msh(src)
    with pytest.raises(ValueError, match="scale"):
        write_scaled_msh(src, tmp_path / "dst.msh", scale=0.0)
