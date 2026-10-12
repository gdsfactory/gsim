"""The Marks-Williams integrals over a saved Palace boundary Mode.

Palace writes a Mode's fields as nodal values on second-order triangles,
so the power flux and the contour current are quadrature problems rather
than solver problems, and both have closed-form answers on a field
written down by hand. The fields here are analytic; what is under test
is the quadrature, the edge bookkeeping of the contour, and the sign
consistency that keeps a closed contour from cancelling itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.palace.mode_fields import (
    BoundaryModeField,
    check_field_is_the_mode,
    contour_current,
    field_index_ratio,
    load_boundary_mode_field,
    power_flux,
    z0_power_current,
)

MU0 = 4.0e-7 * np.pi

SIDE = 4  # cells per axis of the unit square
AREA = 1.0


def square_mesh(n: int = SIDE):
    """A second-order triangulation of the unit square.

    Returns:
        ``(points, cells)`` with cells as three corners then the midside
        node of each edge, which is the layout Palace writes.
    """
    step = 1.0 / n
    corners = np.array(
        [(i * step, j * step) for j in range(n + 1) for i in range(n + 1)],
        dtype=np.float64,
    )

    def corner_id(i: int, j: int) -> int:
        return j * (n + 1) + i

    triangles = []
    for j in range(n):
        for i in range(n):
            a, b = corner_id(i, j), corner_id(i + 1, j)
            c, d = corner_id(i + 1, j + 1), corner_id(i, j + 1)
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    midside: dict[tuple[int, int], int] = {}
    points = list(corners)
    cells = []
    for tri in triangles:
        row = list(tri)
        for start, end in ((0, 1), (1, 2), (2, 0)):
            key = (min(tri[start], tri[end]), max(tri[start], tri[end]))
            if key not in midside:
                midside[key] = len(points)
                points.append((corners[key[0]] + corners[key[1]]) / 2.0)
            row.append(midside[key])
        cells.append(row)
    return np.asarray(points, dtype=np.float64), np.asarray(cells, dtype=np.int64)


def field_from(e_t, h_t, *, n: int = SIDE) -> BoundaryModeField:
    """A saved Mode carrying the transverse fields two callables give."""
    points, cells = square_mesh(n)
    return BoundaryModeField(
        points_um=points,
        cells=cells,
        attribute=np.ones(cells.shape[0], dtype=np.int64),
        e_t=np.asarray(e_t(points), dtype=np.complex128),
        e_n=np.zeros(points.shape[0], dtype=np.complex128),
        h_t=np.asarray(h_t(points), dtype=np.complex128),
        h_n=np.zeros(points.shape[0], dtype=np.complex128),
    )


def uniform(vector):
    """A transverse field with the same value at every node."""

    def build(points):
        return np.broadcast_to(np.asarray(vector), (points.shape[0], 2)).copy()

    return build


def rotational(rate: float):
    """``H = rate * (-y, x)``: a field whose curl is ``2 * rate`` z-hat."""

    def build(points):
        return rate * np.stack([-points[:, 1], points[:, 0]], axis=1)

    return build


class TestPowerFlux:
    def test_a_uniform_tem_field_integrates_to_its_area(self):
        """``P = (1/2) (E x H*) . z A`` when neither field varies."""
        e0, h0 = 2.0 + 0.0j, 0.0 + 3.0j
        field = field_from(uniform([0.0, e0]), uniform([h0, 0.0]))
        expected = 0.5 * (0.0 * np.conj(0.0) - e0 * np.conj(h0)) * AREA
        assert power_flux(field) == pytest.approx(expected, rel=1e-12)

    def test_the_quadrature_is_exact_for_a_quadratic_product(self):
        """Two linear fields multiply to a quadratic; degree 4 covers it."""

        def linear_e(points):
            return np.stack([np.zeros(points.shape[0]), points[:, 0]], axis=1)

        def linear_h(points):
            return np.stack([points[:, 1], np.zeros(points.shape[0])], axis=1)

        field = field_from(linear_e, linear_h)
        # -(1/2) integral x y over the unit square.
        assert power_flux(field) == pytest.approx(-0.5 * 0.25, rel=1e-12)


class TestContourCurrent:
    def test_a_curl_free_field_encloses_no_current(self):
        """A closed contour of a constant field cancels — if it is closed."""
        field = field_from(uniform([0.0, 0.0]), uniform([1.0, 2.0]))
        current = contour_current(field, h_span=(0.25, 0.75), v_span=(0.25, 0.75))
        assert abs(current) < 1e-12

    def test_a_rotational_field_encloses_the_curl_it_carries(self):
        """Ampere's law: the contour integral is the enclosed curl."""
        rate = 0.7
        field = field_from(uniform([0.0, 0.0]), rotational(rate))
        span = (0.25, 0.75)
        current = contour_current(field, h_span=span, v_span=span)
        enclosed = 2.0 * rate * (span[1] - span[0]) ** 2
        assert abs(current) == pytest.approx(enclosed, rel=1e-12)

    def test_the_contour_does_not_depend_on_where_it_is_drawn(self):
        """Any contour around the same enclosed curl reads the same."""
        field = field_from(uniform([0.0, 0.0]), rotational(0.7))
        inner = abs(contour_current(field, h_span=(0.25, 0.75), v_span=(0.25, 0.75)))
        outer = abs(contour_current(field, h_span=(0.0, 1.0), v_span=(0.0, 1.0)))
        # Four times the area, four times the enclosed curl.
        assert outer == pytest.approx(4.0 * inner, rel=1e-12)

    def test_a_rectangle_off_the_mesh_is_reported(self):
        field = field_from(uniform([0.0, 0.0]), uniform([1.0, 0.0]))
        with pytest.raises(ValueError, match="No mesh edge lies on the rectangle"):
            contour_current(field, h_span=(2.0, 3.0), v_span=(2.0, 3.0))


class TestZ0PowerCurrent:
    def test_it_divides_the_power_by_the_squared_current(self):
        rate = 0.7
        field = field_from(uniform([0.0, 1.0]), rotational(rate))
        span = (0.25, 0.75)
        z0 = z0_power_current(field, h_span=span, v_span=span)
        expected = (
            2.0
            * power_flux(field)
            / abs(contour_current(field, h_span=span, v_span=span)) ** 2
        )
        assert z0 == pytest.approx(expected, rel=1e-12)

    def test_a_mode_carrying_no_current_is_reported(self):
        field = field_from(uniform([0.0, 1.0]), uniform([1.0, 2.0]))
        with pytest.raises(ValueError, match="no current"):
            z0_power_current(field, h_span=(0.25, 0.75), v_span=(0.25, 0.75))


def write_saved_mode(root, *, cycle=1, arrays):
    """Write one ParaView cycle where a boundary-mode solve would put it.

    Palace writes a partitioned dataset — a ``.pvtu`` naming its pieces —
    so the reader is exercised through the same two files it meets in
    a real output directory rather than through a lone ``.vtu``.
    """
    import pyvista as pv
    import vtk

    points = np.array(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    )
    grid = pv.UnstructuredGrid(
        np.array([6, 0, 1, 2, 3, 4, 5]),
        np.array([69]),  # VTK_LAGRANGE_TRIANGLE
        np.column_stack([points, np.zeros(6)]),
    )
    for name, value in arrays.items():
        grid.point_data[name] = value
    grid.cell_data["attribute"] = np.array([7])

    cycle_dir = root / "paraview" / "boundarymode" / f"Cycle{cycle:06d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPUnstructuredGridWriter()
    writer.SetFileName(str(cycle_dir / "data.pvtu"))
    writer.SetInputData(grid)
    writer.SetNumberOfPieces(1)
    writer.SetStartPiece(0)
    writer.SetEndPiece(0)
    writer.Write()
    return points


def saved_mode_arrays(**overrides):
    """Field arrays whose two transverse components are told apart."""
    arrays = {
        "E_real": np.tile([1.0, 2.0], (6, 1)),
        "E_imag": np.tile([0.5, 0.25], (6, 1)),
        "En_real": np.full(6, 3.0),
        "En_imag": np.zeros(6),
        "Bt_real": np.tile([10.0, 20.0], (6, 1)),
        "Bt_imag": np.zeros((6, 2)),
        "B_real": np.full(6, 4.0),
        "B_imag": np.zeros(6),
    }
    arrays.update(overrides)
    return arrays


class TestLoadingASavedMode:
    """The path from Palace's own output directory to a field array."""

    def test_it_reads_the_cycle_the_mode_was_saved_in(self, tmp_path):
        pytest.importorskip("vtk")
        write_saved_mode(tmp_path, cycle=1, arrays=saved_mode_arrays())

        field = load_boundary_mode_field(tmp_path, mode_id=1)

        assert field.cells.shape == (1, 6)
        assert field.points_um.shape == (6, 2)
        assert field.attribute.tolist() == [7]

    def test_the_transverse_components_come_back_as_written(self, tmp_path):
        """Palace writes them in the same order as its point coordinates.

        An earlier reader swapped them; the fields that swap was
        diagnosed on came from meshes whose electrode outlines had lost
        their perfect-conductor groups, so what looked like a swapped
        write was a correct read of a wrongly conditioned solve.
        """
        pytest.importorskip("vtk")
        write_saved_mode(tmp_path, cycle=1, arrays=saved_mode_arrays())

        field = load_boundary_mode_field(tmp_path, mode_id=1)

        assert np.allclose(field.e_t[:, 0], complex(1.0, 0.5))
        assert np.allclose(field.e_t[:, 1], complex(2.0, 0.25))
        assert np.allclose(field.h_t[:, 0], 10.0 / MU0)
        assert np.allclose(field.h_t[:, 1], 20.0 / MU0)

    def test_the_longitudinal_components_are_not_swapped(self, tmp_path):
        pytest.importorskip("vtk")
        write_saved_mode(tmp_path, cycle=1, arrays=saved_mode_arrays())

        field = load_boundary_mode_field(tmp_path, mode_id=1)

        assert np.allclose(field.e_n, 3.0)
        assert field.h_n is not None
        assert np.allclose(field.h_n, 4.0 / MU0)

    def test_a_mode_saved_without_a_longitudinal_h_still_reads(self, tmp_path):
        """No integral wants it, so its absence is not a failure."""
        pytest.importorskip("vtk")
        arrays = saved_mode_arrays()
        del arrays["B_real"], arrays["B_imag"]
        write_saved_mode(tmp_path, cycle=1, arrays=arrays)

        field = load_boundary_mode_field(tmp_path, mode_id=1)

        assert field.h_n is None
        assert np.allclose(field.e_n, 3.0)

    def test_a_mode_the_solve_never_saved_is_reported(self, tmp_path):
        pytest.importorskip("vtk")
        write_saved_mode(tmp_path, cycle=1, arrays=saved_mode_arrays())

        with pytest.raises(FileNotFoundError):
            load_boundary_mode_field(tmp_path, mode_id=4)

    def test_a_cycle_carrying_no_fields_is_reported(self, tmp_path):
        """Palace writes a last cycle holding only the mesh partition."""
        pytest.importorskip("vtk")
        write_saved_mode(tmp_path, cycle=1, arrays={"Rank": np.zeros(6)})

        with pytest.raises(ValueError, match="carries no E_real"):
            load_boundary_mode_field(tmp_path, mode_id=1)


class TestFieldIndexRatio:
    def test_a_tem_field_recovers_its_own_index(self):
        """``H_t = (n / eta_0) z-hat x E_t`` is what the ratio inverts."""
        eta0 = 376.730313668
        n_eff = 2.4
        e_t = np.tile([3.0 + 0j, -1.0 + 0j], (6, 1))
        h_t = (n_eff / eta0) * np.stack([-e_t[:, 1], e_t[:, 0]], axis=1)
        field = BoundaryModeField(
            points_um=np.zeros((6, 2)),
            cells=np.arange(6).reshape(1, 6),
            attribute=np.array([1]),
            e_t=e_t,
            e_n=np.zeros(6, dtype=complex),
            h_t=h_t,
            h_n=None,
        )
        assert field_index_ratio(field) == pytest.approx(n_eff, rel=1e-9)

    def test_a_mode_without_an_electric_field_has_no_ratio(self):
        field = BoundaryModeField(
            points_um=np.zeros((6, 2)),
            cells=np.arange(6).reshape(1, 6),
            attribute=np.array([1]),
            e_t=np.zeros((6, 2), dtype=complex),
            e_n=np.zeros(6, dtype=complex),
            h_t=np.ones((6, 2), dtype=complex),
            h_n=None,
        )
        assert np.isnan(field_index_ratio(field))


class TestCheckFieldIsTheMode:
    """Asking a saved field whether it is the Mode it was fetched for.

    Which ParaView cycle holds which Mode is a convention, and a
    convention changes without an error; the fields carry their own
    index, so they can be asked.
    """

    @staticmethod
    def _field(*, n_from_fields: float) -> BoundaryModeField:
        """A TEM field whose own index is *n_from_fields*."""
        eta0 = 376.730313668
        e_t = np.tile([1.0 + 0j, 0.0 + 0j], (6, 1))
        h_t = (n_from_fields / eta0) * np.stack([-e_t[:, 1], e_t[:, 0]], axis=1)
        return BoundaryModeField(
            points_um=np.zeros((6, 2)),
            cells=np.arange(6).reshape(1, 6),
            attribute=np.array([1]),
            e_t=e_t,
            e_n=np.zeros(6, dtype=complex),
            h_t=h_t,
            h_n=None,
        )

    def test_fields_matching_the_mode_table_pass_quietly(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_field_is_the_mode(
                self._field(n_from_fields=2.4),
                mode_id=2,
                n_eff=complex(2.5, -0.01),
                context="The rf stage",
            )

    def test_fields_from_another_mode_are_reported(self):
        with pytest.warns(UserWarning, match="different mode's fields"):
            check_field_is_the_mode(
                self._field(n_from_fields=0.02),
                mode_id=2,
                n_eff=complex(2.5, -0.01),
                context="The rf stage",
            )

    def test_the_warning_opens_with_the_context_it_was_given(self):
        with pytest.warns(UserWarning, match="^The eigen read read back fields"):
            check_field_is_the_mode(
                self._field(n_from_fields=0.02),
                mode_id=2,
                n_eff=complex(2.5),
                context="The eigen read",
            )

    def test_a_mode_carrying_no_field_says_nothing(self):
        """Nothing to compare is not the same as a mismatch."""
        import warnings

        field = self._field(n_from_fields=2.4)
        field = type(field)(
            points_um=field.points_um,
            cells=field.cells,
            attribute=field.attribute,
            e_t=np.zeros_like(field.e_t),
            e_n=field.e_n,
            h_t=field.h_t,
            h_n=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_field_is_the_mode(
                field, mode_id=1, n_eff=complex(2.5), context="The rf stage"
            )

    def test_a_mode_table_claiming_no_index_says_nothing(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_field_is_the_mode(
                self._field(n_from_fields=2.4),
                mode_id=1,
                n_eff=complex(0.0),
                context="The rf stage",
            )
