"""Port-mode fields of a driven run's wave-port face, read off a boundary dataset.

A driven run saves, on the boundary mesh, the unit-power field ``E0_<index>`` of
every wave-port entry. :func:`port_mode_fields_from_grid` turns the cells of
one port face into the :class:`BoundaryModeField` the 2D integrals take, and
:func:`load_port_mode_fields` finds the dataset on disk. The datasets here are
small second-order triangulations written down by hand (generic geometry, no
solver), with fields linear in the coordinates, so every number that comes back
has a closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.palace import load_port_mode_fields as exported_loader
from gsim.palace.mode_fields import (
    BoundaryModeField,
    load_port_mode_fields,
    port_mode_fields_from_grid,
    voltage_line_integral,
)
from tests.palace.test_mode_fields import square_mesh

pv = pytest.importorskip("pyvista")

FACE = 3  # boundary attribute of the port face
OTHER = 9  # boundary attribute of a patch that is not the port face
COORD = 5.0  # position of the face along its normal (um)
SIZE = 10.0  # side of the square face (um)

OFFSET = np.array([0.5 + 0.25j, -1.0 + 0.0j, 2.0 - 0.5j])
GRADIENT = np.array(
    [
        [0.1, 0.2 - 0.1j, 0.0],
        [0.3j, -0.4, 0.05],
        [0.0, 0.7, -0.2 + 0.2j],
    ],
    dtype=np.complex128,
)


def e0_of(points, index=1):
    """``E0_<index>`` at 3D points: linear in the coordinates, scaled by index."""
    return index * (OFFSET[None, :] + points @ GRADIENT.T)


def face_points(normal, coord=COORD, size=SIZE, n=2):
    """The nodes and second-order cells of a square face normal to an axis."""
    unit, cells = square_mesh(n)
    plane = [axis for axis in range(3) if axis != normal]
    points = np.zeros((unit.shape[0], 3))
    points[:, plane] = unit * size
    points[:, normal] = coord
    return points, cells


def build_grid(
    normal=0,
    *,
    indices=(1,),
    n=2,
    coord=COORD,
    jitter=0.0,
    with_other=True,
    cell_type=22,
):
    """A boundary dataset: the port face plus (optionally) a patch off its plane."""
    points, cells = face_points(normal, coord=coord, n=n)
    if jitter:
        points[:, normal] += jitter * np.sin(np.arange(points.shape[0]))
    attribute = np.full(cells.shape[0], FACE)
    if with_other:
        # one triangle in a plane perpendicular to the face, off its extent
        plane = [axis for axis in range(3) if axis != normal]
        local = np.array(
            [(0, 0), (4, 0), (0, 4), (2, 0), (2, 2), (0, 2)], dtype=np.float64
        )
        patch = np.zeros((6, 3))
        patch[:, normal] = local[:, 0]
        patch[:, plane[0]] = local[:, 1]
        patch[:, plane[1]] = -7.0
        cells = np.vstack([cells, points.shape[0] + np.arange(6)])
        points = np.vstack([points, patch])
        attribute = np.append(attribute, OTHER)
    flat = np.hstack([np.full((cells.shape[0], 1), 6), cells]).ravel()
    grid = pv.UnstructuredGrid(
        flat, np.full(cells.shape[0], cell_type, dtype=np.uint8), points
    )
    grid.cell_data["attribute"] = attribute
    for index in indices:
        e = e0_of(points, index)
        grid.point_data[f"E0_{index}_real"] = np.ascontiguousarray(e.real)
        grid.point_data[f"E0_{index}_imag"] = np.ascontiguousarray(e.imag)
    return grid


def write_driven_boundary(root, grid, *, cycle=1, excitation=1):
    """Write one boundary cycle where a driven wave-port run puts it."""
    vtk = pytest.importorskip("vtk")
    cycle_dir = (
        root
        / "paraview"
        / "driven_boundary"
        / f"excitation_{excitation}"
        / f"Cycle{cycle:06d}"
    )
    cycle_dir.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPUnstructuredGridWriter()
    writer.SetFileName(str(cycle_dir / "data.pvtu"))
    writer.SetInputData(grid)
    writer.SetNumberOfPieces(1)
    writer.SetStartPiece(0)
    writer.SetEndPiece(0)
    writer.Write()


class TestFieldsFromAGrid:
    @pytest.mark.parametrize(
        ("normal", "plane"), [(0, (1, 2)), (1, (0, 2)), (2, (0, 1))]
    )
    def test_the_face_comes_back_in_its_own_coordinates(self, normal, plane):
        """``(y, z)`` for a face normal to x, ``(x, z)`` for y, ``(x, y)`` for z."""
        grid = build_grid(normal)
        (field,) = port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])

        points3, cells = face_points(normal)
        assert isinstance(field, BoundaryModeField)
        assert field.cells.shape == cells.shape
        assert field.points_um.shape == (points3.shape[0], 2)
        assert field.attribute.tolist() == [FACE] * cells.shape[0]
        # points may be renumbered by the extraction: compare the sets of nodes
        got = {tuple(np.round(p, 9)) for p in field.points_um}
        want = {tuple(np.round(p, 9)) for p in points3[:, list(plane)]}
        assert got == want

    @pytest.mark.parametrize(
        ("normal", "plane"), [(0, [1, 2]), (1, [0, 2]), (2, [0, 1])]
    )
    def test_the_in_plane_part_is_e_t_and_the_normal_part_is_e_n(self, normal, plane):
        grid = build_grid(normal)
        (field,) = port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])

        # rebuild the 3D node of every 2D node, then the field the grid carries
        points3 = np.zeros((field.points_um.shape[0], 3))
        points3[:, plane] = field.points_um
        points3[:, normal] = COORD
        expected = e0_of(points3, 1)
        np.testing.assert_allclose(field.e_t, expected[:, plane], atol=1e-12)
        np.testing.assert_allclose(field.e_n, expected[:, normal], atol=1e-12)

    def test_the_magnetic_field_is_absent(self):
        grid = build_grid(0)
        (field,) = port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])
        assert field.h_n is None
        assert field.h_t.shape == field.e_t.shape
        assert not np.any(field.h_t)

    def test_one_field_per_index_in_the_order_asked(self):
        grid = build_grid(0, indices=(1, 2, 5))
        fields = port_mode_fields_from_grid(grid, indices=[5, 1], attributes=[FACE])
        assert len(fields) == 2
        # E0_<index> is index * E0_1 in the builder
        np.testing.assert_allclose(fields[0].e_t, 5.0 * fields[1].e_t, atol=1e-12)
        np.testing.assert_allclose(fields[0].e_n, 5.0 * fields[1].e_n, atol=1e-12)
        assert np.array_equal(fields[0].cells, fields[1].cells)
        assert np.array_equal(fields[0].points_um, fields[1].points_um)

    def test_duplicated_attributes_count_once(self):
        grid = build_grid(0)
        (field,) = port_mode_fields_from_grid(
            grid, indices=[1], attributes=[FACE, FACE, np.int64(FACE)]
        )
        assert field.cells.shape[0] == face_points(0)[1].shape[0]

    def test_a_patch_of_another_attribute_is_left_out(self):
        with_patch = build_grid(0, with_other=True)
        without = build_grid(0, with_other=False)
        (a,) = port_mode_fields_from_grid(with_patch, indices=[1], attributes=[FACE])
        (b,) = port_mode_fields_from_grid(without, indices=[1], attributes=[FACE])
        assert a.cells.shape == b.cells.shape
        np.testing.assert_allclose(
            np.sort(a.points_um, axis=0), np.sort(b.points_um, axis=0)
        )

    def test_single_precision_flatness_is_tolerated(self):
        """Palace writes float32: a face a hair off its plane is still a face."""
        grid = build_grid(1, jitter=4e-4)
        (field,) = port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])
        assert field.points_um.shape[1] == 2

    def test_the_voltage_of_the_port_field_is_the_analytic_line_integral(self):
        """The face field feeds ``voltage_line_integral`` in the face's own (h, v)."""
        normal = 0
        grid = build_grid(normal)
        (field,) = port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])
        path = np.array([(1.0, 2.0), (8.0, 3.0), (6.0, 9.0)])

        nodes = np.zeros((path.shape[0], 3))
        nodes[:, 1:] = path
        nodes[:, 0] = COORD
        ends = e0_of(nodes, 1)[:, 1:]  # linear: the trapezoid rule is exact
        expected = complex(np.sum(0.5 * (ends[:-1] + ends[1:]) * np.diff(path, axis=0)))

        assert voltage_line_integral(field, path) == pytest.approx(expected, abs=1e-9)


class TestRefusals:
    def test_no_attribute_given(self):
        grid = build_grid(0)
        with pytest.raises(ValueError, match="No boundary attribute was given"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[])

    def test_a_dataset_without_attribute_data(self):
        grid = build_grid(0)
        del grid.cell_data["attribute"]
        with pytest.raises(ValueError, match="no 'attribute' cell data"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])

    def test_attributes_the_boundary_does_not_have(self):
        grid = build_grid(0)
        with pytest.raises(ValueError, match=r"no cell on the attributes \[4, 8\]"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[8, 4])
        with pytest.raises(ValueError, match=r"it has \[3, 9\]"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[4])

    def test_cells_that_are_not_second_order_triangles(self):
        points, _ = face_points(0)
        grid = pv.UnstructuredGrid(
            np.array([3, 0, 1, 2]), np.array([5], dtype=np.uint8), points
        )
        grid.cell_data["attribute"] = np.array([FACE])
        with pytest.raises(ValueError, match="not the second-order triangles"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])

    def test_a_face_that_spans_three_dimensions(self):
        """Both patches selected: the nodes fill no coordinate plane."""
        grid = build_grid(0)
        with pytest.raises(ValueError, match="does not lie in one coordinate plane"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE, OTHER])

    def test_a_face_that_is_not_flat_to_the_saved_precision(self):
        grid = build_grid(0, jitter=5e-2)
        with pytest.raises(ValueError, match="does not lie in one coordinate plane"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])

    def test_a_face_collapsed_onto_a_line(self):
        """Two flat axes leave the normal ambiguous."""
        points = np.zeros((6, 3))
        points[:, 0] = np.linspace(0.0, 4.0, 6)
        grid = pv.UnstructuredGrid(
            np.array([6, 0, 1, 2, 3, 4, 5]), np.array([22], dtype=np.uint8), points
        )
        grid.cell_data["attribute"] = np.array([FACE])
        with pytest.raises(ValueError, match="does not lie in one coordinate plane"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])

    def test_an_entry_the_run_did_not_save(self):
        grid = build_grid(0, indices=(1,))
        with pytest.raises(ValueError, match="no E0_2_real, E0_2_imag"):
            port_mode_fields_from_grid(grid, indices=[1, 2], attributes=[FACE])

    def test_an_entry_missing_only_its_imaginary_part(self):
        grid = build_grid(0, indices=(1,))
        del grid.point_data["E0_1_imag"]
        with pytest.raises(ValueError, match="no E0_1_imag"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])

    def test_a_field_that_is_not_a_three_vector(self):
        grid = build_grid(0)
        grid.point_data["E0_1_real"] = np.zeros((grid.n_points, 2))
        grid.point_data["E0_1_imag"] = np.zeros((grid.n_points, 2))
        with pytest.raises(ValueError, match=r"E0_1 has shape .*not \(N, 3\)"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[FACE])


class TestLoadingFromDisk:
    def test_it_reads_the_cycle_and_excitation_asked_for(self, tmp_path):
        pytest.importorskip("vtk")
        # excitation 1 holds E0_1 scaled 1; excitation 2 holds a different grid
        write_driven_boundary(tmp_path, build_grid(0), cycle=1, excitation=1)
        late = build_grid(0)
        for key in ("E0_1_real", "E0_1_imag"):
            late.point_data[key] = 3.0 * np.asarray(late.point_data[key])
        write_driven_boundary(tmp_path, late, cycle=2, excitation=1)

        (first,) = load_port_mode_fields(
            tmp_path, indices=[1], attributes=[FACE], cycle=1
        )
        (second,) = load_port_mode_fields(
            tmp_path, indices=[1], attributes=[FACE], cycle=2
        )

        assert first.cells.shape == second.cells.shape
        assert np.array_equal(first.points_um, second.points_um)
        np.testing.assert_allclose(second.e_t, 3.0 * first.e_t, atol=1e-6)
        np.testing.assert_allclose(second.e_n, 3.0 * first.e_n, atol=1e-6)

    def test_the_values_survive_the_round_trip(self, tmp_path):
        pytest.importorskip("vtk")
        write_driven_boundary(tmp_path, build_grid(2), cycle=1)
        (field,) = load_port_mode_fields(
            tmp_path, indices=[1], attributes=[FACE], cycle=1
        )
        points3 = np.zeros((field.points_um.shape[0], 3))
        points3[:, [0, 1]] = field.points_um
        points3[:, 2] = COORD
        np.testing.assert_allclose(field.e_t, e0_of(points3, 1)[:, [0, 1]], atol=1e-9)

    def test_the_excitation_is_selected(self, tmp_path):
        pytest.importorskip("vtk")
        write_driven_boundary(tmp_path, build_grid(0), cycle=1, excitation=1)
        other = build_grid(0)
        for key in ("E0_1_real", "E0_1_imag"):
            other.point_data[key] = -2.0 * np.asarray(other.point_data[key])
        write_driven_boundary(tmp_path, other, cycle=1, excitation=2)

        (one,) = load_port_mode_fields(
            tmp_path, indices=[1], attributes=[FACE], cycle=1, excitation=1
        )
        (two,) = load_port_mode_fields(
            tmp_path, indices=[1], attributes=[FACE], cycle=1, excitation=2
        )
        np.testing.assert_allclose(two.e_t, -2.0 * one.e_t, atol=1e-6)

    def test_a_cycle_the_run_never_saved(self, tmp_path):
        pytest.importorskip("vtk")
        write_driven_boundary(tmp_path, build_grid(0), cycle=1)
        with pytest.raises(FileNotFoundError):
            load_port_mode_fields(tmp_path, indices=[1], attributes=[FACE], cycle=4)

    def test_a_run_with_no_boundary_output(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_port_mode_fields(tmp_path, indices=[1], attributes=[FACE], cycle=1)

    def test_a_saved_boundary_without_the_entry_is_reported(self, tmp_path):
        pytest.importorskip("vtk")
        write_driven_boundary(tmp_path, build_grid(0, indices=(1,)), cycle=1)
        with pytest.raises(ValueError, match="no E0_7_real"):
            load_port_mode_fields(tmp_path, indices=[7], attributes=[FACE], cycle=1)

    def test_it_is_exported_from_the_package(self):
        import gsim.palace

        assert exported_loader is load_port_mode_fields
        assert "load_port_mode_fields" in gsim.palace.__all__


class TestADegenerateMesh:
    def test_a_mesh_without_a_triangle_of_area_is_refused(self):
        """Every triangle collapsed to a point: nothing can contain a sample."""
        field = BoundaryModeField(
            points_um=np.zeros((6, 2)),
            cells=np.arange(6).reshape(1, 6),
            attribute=np.array([1]),
            e_t=np.ones((6, 2), dtype=np.complex128),
            e_n=np.zeros(6, dtype=np.complex128),
            h_t=np.zeros((6, 2), dtype=np.complex128),
            h_n=None,
        )
        with pytest.raises(ValueError, match="no triangle with an area"):
            voltage_line_integral(field, [(0.0, 0.0), (1.0, 0.0)])
