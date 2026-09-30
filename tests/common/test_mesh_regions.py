"""Physical groups and cell blocks read off a gmsh mesh."""

from __future__ import annotations

import meshio
import numpy as np
import pytest

from gsim.common.mesh_regions import (
    cell_blocks,
    element_regions,
    group_names,
    group_tags,
    node_regions,
    region_elements,
)


def _mixed_dim_mesh(tmp_path, name="mixed.msh"):
    """Unit square as two triangle groups, with one named boundary curve.

    Points 0..3 are the square's corners counter-clockwise from the
    origin. Triangle 0 ('core') is the lower-right half, triangle 1
    ('clad') the upper-left one, and the dim-1 group 'contact' is the
    bottom edge.
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    mesh = meshio.Mesh(
        points=points,
        cells=[
            ("line", np.array([[0, 1]])),
            ("triangle", np.array([[0, 1, 2]])),
            ("triangle", np.array([[0, 2, 3]])),
        ],
        cell_data={
            "gmsh:physical": [np.array([3]), np.array([1]), np.array([2])],
            "gmsh:geometrical": [np.array([3]), np.array([1]), np.array([2])],
        },
        field_data={
            "core": np.array([1, 2]),
            "clad": np.array([2, 2]),
            "contact": np.array([3, 1]),
        },
    )
    path = tmp_path / name
    meshio.write(str(path), mesh, file_format="gmsh22", binary=False)
    return meshio.read(str(path))


class TestGroupMaps:
    def test_the_two_directions_are_inverses_of_one_another(self, tmp_path):
        mesh = _mixed_dim_mesh(tmp_path)

        tags = group_tags(mesh, dim=2)
        names = group_names(mesh, dim=2)

        assert tags == {"core": 1, "clad": 2}
        assert names == {1: "core", 2: "clad"}

    def test_a_group_of_another_dimension_is_not_returned(self, tmp_path):
        mesh = _mixed_dim_mesh(tmp_path)

        assert group_tags(mesh, dim=1) == {"contact": 3}
        assert "contact" not in group_tags(mesh, dim=2)

    def test_a_mesh_with_no_group_of_that_dimension_gives_an_empty_map(self, tmp_path):
        mesh = _mixed_dim_mesh(tmp_path)

        assert group_tags(mesh, dim=3) == {}
        assert group_names(mesh, dim=3) == {}


class TestCellBlocks:
    def test_blocks_of_one_type_concatenate_in_mesh_order(self, tmp_path):
        mesh = _mixed_dim_mesh(tmp_path)

        cells, tags = cell_blocks(mesh, "triangle")

        assert cells.tolist() == [[0, 1, 2], [0, 2, 3]]
        assert tags.tolist() == [1, 2]

    def test_the_cell_type_selects_the_dimension(self, tmp_path):
        mesh = _mixed_dim_mesh(tmp_path)

        cells, tags = cell_blocks(mesh, "line")

        assert cells.tolist() == [[0, 1]]
        assert tags.tolist() == [3]

    def test_cells_the_mesh_gives_no_tag_read_tag_zero(self):
        mesh = meshio.Mesh(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            cells=[("triangle", np.array([[0, 1, 2]]))],
        )

        cells, tags = cell_blocks(mesh, "triangle")

        assert cells.tolist() == [[0, 1, 2]]
        assert tags.tolist() == [0]
        # An untagged element names no region, and matches no group.
        assert element_regions(mesh)[1] == [""]

    def test_a_mesh_without_cells_of_that_type_is_rejected(self):
        mesh = meshio.Mesh(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            cells=[("line", np.array([[0, 1]]))],
            cell_data={"gmsh:physical": [np.array([1])]},
            field_data={"contact": np.array([1, 1])},
        )

        with pytest.raises(ValueError, match="triangle"):
            cell_blocks(mesh, "triangle")


class TestTriangleConveniences:
    def test_every_element_carries_its_group_name(self, tmp_path):
        mesh = _mixed_dim_mesh(tmp_path)

        tris, names = element_regions(mesh)

        assert tris.tolist() == [[0, 1, 2], [0, 2, 3]]
        assert names == ["core", "clad"]

    def test_an_element_of_no_named_group_gets_an_empty_name(self, tmp_path):
        mesh = _mixed_dim_mesh(tmp_path)
        mesh.field_data.pop("clad")

        _tris, names = element_regions(mesh)

        assert names == ["core", ""]

    def test_a_node_inherits_the_group_of_one_incident_element(self, tmp_path):
        mesh = _mixed_dim_mesh(tmp_path)

        # Nodes 0 and 2 sit on both triangles; the lower-numbered element
        # wins, so they read 'core'. Node 1 is core-only, node 3 clad-only.
        assert node_regions(mesh) == ["core", "core", "core", "clad"]

    def test_the_elements_of_a_region_come_back_in_mesh_order(self, tmp_path):
        mesh = _mixed_dim_mesh(tmp_path)

        assert region_elements(mesh, "core").tolist() == [0]
        assert region_elements(mesh, "clad").tolist() == [1]

    def test_a_region_that_is_not_on_the_mesh_is_reported(self, tmp_path):
        mesh = _mixed_dim_mesh(tmp_path)

        with pytest.raises(ValueError, match="electrode_high"):
            region_elements(mesh, "electrode_high")

    def test_a_mesh_is_read_from_a_path_too(self, tmp_path):
        _mixed_dim_mesh(tmp_path)

        assert region_elements(tmp_path / "mixed.msh", "core").tolist() == [0]
