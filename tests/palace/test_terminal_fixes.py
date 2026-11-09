"""Fixes before the validation runs.

Quadrature default, degeneracy tolerance, export and saved port modes.

Nothing here runs Palace. The saved boundary fields of a driven run are written
the way Palace writes them (a ``.pvtu`` of second-order triangles with
``E0_<index>_real`` / ``E0_<index>_imag`` point data and an ``attribute`` cell
array, in ``paraview/driven_boundary/excitation_N/CycleNNNNNN``) from the
polynomial synthetic modes of ``test_terminal_ports``, so the voltages read back
are known in closed form. Generic synthetic values only.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.constants import c as C0  # noqa: N812

import gsim.palace as palace
from gsim.palace import terminal_ports
from gsim.palace._validation_port_modes import FACE_MODE_RTOL
from gsim.palace.mode_fields import (
    VOLTAGE_NSAMPLES,
    BoundaryModeField,
    load_port_mode_fields,
    port_mode_fields_from_grid,
    voltage_line_integral,
)
from gsim.palace.ports.modes import PortModeEntry, write_port_modes
from gsim.palace.terminal_ports import (
    PortModes2D,
    align_gauge,
    build_terminal_s,
    modes2d,
    port_mode_reference_voltages,
    port_modes_2d,
    port_terminal_matrices,
    terminal_s,
)
from tests.palace.test_mode_fields import field_from, square_mesh
from tests.palace.test_terminal_assembly import three_conductor_specs, two_face_world
from tests.palace.test_terminal_ports import (
    DOMAIN,
    F_MID,
    GSSG,
    MESH_N,
    line,
    modes_from,
)

# -- quadrature default ----------------------------------------------------------------


def smooth_field() -> BoundaryModeField:
    """A field no second-order element holds exactly, so the quadrature order shows."""

    def e_t(points):
        x, y = points[:, 0], points[:, 1]
        return np.stack([np.sin(7 * x) * np.cos(5 * y), np.cos(3 * x + 2 * y)], axis=1)

    return field_from(e_t, np.zeros_like)


class TestQuadratureDefault:
    def test_the_default_is_800_everywhere(self):
        assert VOLTAGE_NSAMPLES == 800
        for function in (voltage_line_integral, port_terminal_matrices, port_modes_2d):
            default = inspect.signature(function).parameters["nsamples"].default
            assert default == 800, function.__name__

    def test_the_line_integral_uses_it_and_is_better_converged_than_200(self):
        field = smooth_field()
        path = [(0.05, 0.07), (0.93, 0.88), (0.2, 0.9)]
        by_default = voltage_line_integral(field, path)
        at = {
            n: voltage_line_integral(field, path, nsamples=n) for n in (200, 800, 12800)
        }
        assert by_default == at[800]
        assert at[200] != at[800]
        assert abs(at[800] - at[12800]) < abs(at[200] - at[12800])

    def test_the_matrices_are_read_with_it(self, monkeypatch):
        seen: list[int] = []
        # the calls are made in the module that holds port_terminal_matrices
        original = modes2d.voltage_line_integral

        def spy(field, path, *, nsamples, tol_um):
            seen.append(nsamples)
            return original(field, path, nsamples=nsamples, tol_um=tol_um)

        monkeypatch.setattr(modes2d, "voltage_line_integral", spy)
        _, t_v, t_i, _ = line()
        fields = modes_from(t_v[F_MID], t_i[F_MID])
        port_terminal_matrices(fields, GSSG, domain=DOMAIN)
        port_modes_2d(fields, GSSG, freq_ghz=1.0, n_eff=[3.0, 2.0, 1.0], domain=DOMAIN)
        assert set(seen) == {800}
        assert len(seen) == 18
        seen.clear()
        port_modes_2d(
            fields,
            GSSG,
            freq_ghz=1.0,
            n_eff=[3.0, 2.0, 1.0],
            domain=DOMAIN,
            nsamples=40,
        )
        assert set(seen) == {40}
        assert terminal_ports.VOLTAGE_NSAMPLES == 800


# -- degeneracy tolerance of align_gauge -----------------------------------------------

N_3D = np.array([1.5244, 1.4662, 1.1623])  # modes 1 and 2 are 3.8 % apart
N_2D = np.array([1.5837, 1.5197, 1.1766])  # the same modes, 3.9 % off (lossy 2D solve)
FREQ_GRID = np.array([1.0, 10.0, 50.0])


def three_mode_solves(n_eff_2d=N_2D):
    """Two 2D solves (1 and 50 GHz) of a three-terminal port and the 3D wave numbers."""
    t_v = np.eye(3) + 0.2 * (np.ones((3, 3)) - np.eye(3))
    solves = [PortModes2D(f, n_eff_2d, t_v, 0.02 * t_v) for f in (1.0, 50.0)]
    k0 = 2.0 * np.pi * FREQ_GRID * 1e9 / C0
    return solves, k0[:, None] * N_3D[None, :]


class TestDegenerateRtol:
    def test_the_default_is_the_face_mode_tolerance(self):
        for function in (align_gauge, build_terminal_s, terminal_s):
            default = inspect.signature(function).parameters["degenerate_rtol"].default
            assert default == FACE_MODE_RTOL == 0.005, function.__name__

    def test_modes_3p8_percent_apart_are_not_degenerate_when_rtol_is_opened(self):
        solves, kn = three_mode_solves()
        with pytest.raises(ValueError, match="cannot be paired"):
            align_gauge(solves, freq_ghz=FREQ_GRID, wave_numbers=kn)
        aligned = align_gauge(solves, freq_ghz=FREQ_GRID, wave_numbers=kn, rtol=0.15)
        assert aligned.degenerate == ()
        assert aligned.pairing == ((0, 1, 2), (0, 1, 2))
        assert aligned.mismatch == pytest.approx(1.5837 / 1.5244 - 1.0, rel=2e-3)
        # the gauge is the pilot rule's: every pilot voltage real and positive
        for k, row in enumerate(aligned.pilot):
            assert np.all(aligned.t_v[:, row, k].real > 0.0)
            np.testing.assert_allclose(aligned.t_v[:, row, k].imag, 0.0, atol=1e-12)

    def test_a_wider_degenerate_tolerance_flags_them(self):
        solves, kn = three_mode_solves()
        with pytest.raises(ValueError, match=r"same n_eff within 5\.00%"):
            align_gauge(
                solves,
                freq_ghz=FREQ_GRID,
                wave_numbers=kn,
                rtol=0.15,
                degenerate_rtol=0.05,
            )
        aligned = align_gauge(
            solves,
            freq_ghz=FREQ_GRID,
            wave_numbers=kn,
            rtol=0.15,
            degenerate_rtol=0.05,
            allow_degenerate=True,
        )
        assert aligned.degenerate == ((0, 1),)

    def test_pairing_tolerance_alone_does_not_set_it(self):
        """Closing ``rtol`` again does not touch the degeneracy test either."""
        solves, kn = three_mode_solves(n_eff_2d=N_3D * (1 + 1e-4))
        aligned = align_gauge(solves, freq_ghz=FREQ_GRID, wave_numbers=kn)
        assert aligned.degenerate == ()
        assert aligned.mismatch < 2e-4

    def test_modes_closer_than_the_tolerance_are_still_refused(self):
        n_3d = np.array([1.5244, 1.5200, 1.1623])  # 0.3 % apart
        solves, _ = three_mode_solves(n_eff_2d=n_3d)
        k0 = 2.0 * np.pi * FREQ_GRID * 1e9 / C0
        kn = k0[:, None] * n_3d[None, :]
        with pytest.raises(ValueError, match="reference_voltages"):
            align_gauge(solves, freq_ghz=FREQ_GRID, wave_numbers=kn, rtol=0.15)

    @pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
    def test_a_bad_value_is_refused(self, bad):
        solves, kn = three_mode_solves()
        with pytest.raises(ValueError, match="degenerate_rtol must be"):
            align_gauge(
                solves,
                freq_ghz=FREQ_GRID,
                wave_numbers=kn,
                rtol=0.15,
                degenerate_rtol=bad,
            )

    @pytest.mark.parametrize("function", [build_terminal_s, terminal_s])
    def test_the_terminal_s_entry_points_pass_it_on(self, tmp_path, function):
        _, runs, modes_2d, pilot, _ = two_face_world(tmp_path)
        with pytest.raises(ValueError, match="same n_eff"):
            function(
                runs,
                three_conductor_specs(),
                modes_2d,
                pilot=pilot,
                degenerate_rtol=10.0,
            )
        # the same call with the default tolerance gets through the alignment
        function(runs, three_conductor_specs(), modes_2d, pilot=pilot)


# -- the export ------------------------------------------------------------------------


class TestExports:
    @pytest.mark.parametrize(
        "name",
        ["build_terminal_s", "port_mode_reference_voltages", "load_port_mode_fields"],
    )
    def test_the_names_are_in_all_of_gsim_palace(self, name):
        assert name in palace.__all__

    def test_build_terminal_s_is_the_module_function(self):
        assert palace.build_terminal_s is terminal_ports.build_terminal_s
        assert palace.build_terminal_s is build_terminal_s

    def test_the_port_mode_readers_are_the_module_functions(self):
        assert palace.port_mode_reference_voltages is port_mode_reference_voltages
        assert palace.load_port_mode_fields is load_port_mode_fields

    def test_everything_in_all_exists(self):
        assert [name for name in palace.__all__ if not hasattr(palace, name)] == []


# -- saved port-mode fields of a driven run --------------------------------------------

FACE_ATTRIBUTE = 5
WALL_ATTRIBUTE = 9
NORMAL_COORDINATE = 40.0


def lagrange_grid(cells, points, attributes, arrays):
    """A pyvista grid of second-order Lagrange triangles with the given arrays."""
    pv = pytest.importorskip("pyvista")
    connectivity = np.hstack([np.full((len(cells), 1), 6), cells]).ravel()
    grid = pv.UnstructuredGrid(connectivity, np.full(len(cells), 69), points)
    for name, values in arrays.items():
        grid.point_data[name] = values
    grid.cell_data["attribute"] = np.asarray(attributes)
    return grid


def boundary_grid(fields_by_index, *, axis=0, n=4, wall=True):
    """Boundary dataset: the port face (attribute 5) in the plane ``axis = const``.

    ``fields_by_index`` maps a Palace entry index to an ``(N, 3)`` complex array
    on the face's points (components along x, y, z). With ``wall``, a triangle
    of another attribute lies elsewhere in 3D, as the outer boundary does.
    """
    points2, cells2 = square_mesh(n)
    npts = points2.shape[0]
    face_points = np.insert(points2, axis, NORMAL_COORDINATE, axis=1)
    wall_points = np.array(
        [(0, 0, 0), (9, 0, 0), (0, 9, 0), (4.5, 0, 0), (4.5, 4.5, 0), (0, 4.5, 0)],
        dtype=float,
    )
    cells = [cells2]
    points = [face_points]
    attributes = [np.full(len(cells2), FACE_ATTRIBUTE)]
    if wall:
        cells.append(np.arange(6)[None, :] + npts)
        points.append(wall_points)
        attributes.append(np.array([WALL_ATTRIBUTE]))
    arrays = {}
    for index, values in fields_by_index.items():
        padded = np.vstack([values, np.zeros((6, 3))]) if wall else values
        arrays[f"E0_{index}_real"] = padded.real
        arrays[f"E0_{index}_imag"] = padded.imag
    return lagrange_grid(
        np.vstack(cells), np.vstack(points), np.concatenate(attributes), arrays
    )


def embed(e_t, axis, normal=0.0):
    """Put in-plane components ``(N, 2)`` into 3-vectors normal to ``axis``."""
    plane = [a for a in range(3) if a != axis]
    out = np.zeros((e_t.shape[0], 3), dtype=np.complex128)
    out[:, plane] = e_t
    out[:, axis] = normal
    return out


def linear_e(points, scale=1.0):
    """A complex field linear in position: its line integral is exact on the mesh."""
    a, b = points[:, 0], points[:, 1]
    return scale * np.stack(
        [(1.0 + 0.5j) * (1.0 + 2.0 * a - b), (0.3 - 0.2j) * (0.5 + a + 3.0 * b)], axis=1
    )


class TestPortModeFieldsFromGrid:
    @pytest.mark.parametrize(("axis", "plane"), [(0, (1, 2)), (1, (0, 2)), (2, (0, 1))])
    def test_the_face_comes_out_in_its_own_in_plane_coordinates(self, axis, plane):
        points2, _ = square_mesh(4)
        e_t = linear_e(points2)
        grid = boundary_grid({3: embed(e_t, axis, normal=7.0)}, axis=axis)
        (field,) = port_mode_fields_from_grid(
            grid, indices=[3], attributes=[FACE_ATTRIBUTE]
        )
        assert plane == tuple(a for a in range(3) if a != axis)
        np.testing.assert_allclose(field.points_um, points2)
        np.testing.assert_allclose(field.e_t, e_t)
        np.testing.assert_allclose(field.e_n, 7.0)
        assert field.h_n is None
        assert not np.any(field.h_t)
        assert field.cells.shape == (square_mesh(4)[1].shape[0], 6)
        assert set(field.attribute.tolist()) == {FACE_ATTRIBUTE}

    def test_entries_come_back_in_the_order_asked_for(self):
        points2, _ = square_mesh(4)
        first, second = linear_e(points2), linear_e(points2, scale=2.0)
        grid = boundary_grid({1: embed(first, 0), 4: embed(second, 0)})
        fields = port_mode_fields_from_grid(
            grid, indices=[4, 1], attributes=[FACE_ATTRIBUTE]
        )
        np.testing.assert_allclose(fields[0].e_t, second)
        np.testing.assert_allclose(fields[1].e_t, first)

    def test_the_voltage_along_a_path_is_the_analytic_one(self):
        points2, _ = square_mesh(4)
        grid = boundary_grid({2: embed(linear_e(points2), 0)})
        (field,) = port_mode_fields_from_grid(
            grid, indices=[2], attributes=[FACE_ATTRIBUTE]
        )
        start, stop = np.array([0.2, 0.1]), np.array([0.7, 0.6])
        expected = linear_e(((start + stop) / 2)[None, :])[0] @ (stop - start)
        assert voltage_line_integral(field, [start, stop]) == pytest.approx(
            expected, rel=1e-12
        )

    def test_an_entry_without_a_field_is_reported(self):
        points2, _ = square_mesh(4)
        grid = boundary_grid({1: embed(linear_e(points2), 0)})
        with pytest.raises(ValueError, match=r"no E0_7_real, E0_7_imag"):
            port_mode_fields_from_grid(
                grid, indices=[1, 7], attributes=[FACE_ATTRIBUTE]
            )

    def test_an_attribute_without_cells_is_reported(self):
        points2, _ = square_mesh(4)
        grid = boundary_grid({1: embed(linear_e(points2), 0)})
        with pytest.raises(ValueError, match=r"no cell on the attributes \[6\]"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[6])
        with pytest.raises(ValueError, match="No boundary attribute"):
            port_mode_fields_from_grid(grid, indices=[1], attributes=[])

    def test_a_face_that_is_not_a_plane_is_refused(self):
        points2, _ = square_mesh(4)
        grid = boundary_grid({1: embed(linear_e(points2), 0)})
        with pytest.raises(ValueError, match="one coordinate plane"):
            port_mode_fields_from_grid(
                grid, indices=[1], attributes=[FACE_ATTRIBUTE, WALL_ATTRIBUTE]
            )


def write_cycle(root, cycle, fields_by_index, *, excitation=1, axis=0, n=4):
    """Write one boundary ParaView cycle where a driven run puts it."""
    vtk = pytest.importorskip("vtk")
    grid = boundary_grid(fields_by_index, axis=axis, n=n)
    folder = (
        root
        / "paraview"
        / "driven_boundary"
        / f"excitation_{excitation}"
        / f"Cycle{cycle:06d}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPUnstructuredGridWriter()
    writer.SetFileName(str(folder / "data.pvtu"))
    writer.SetInputData(grid)
    writer.SetNumberOfPieces(1)
    writer.SetStartPiece(0)
    writer.SetEndPiece(0)
    writer.Write()


class TestLoadPortModeFields:
    def test_it_reads_the_cycle_and_the_excitation_asked_for(self, tmp_path):
        points2, _ = square_mesh(4)
        e_t = linear_e(points2)
        write_cycle(tmp_path, 1, {1: embed(e_t, 0)})
        write_cycle(tmp_path, 2, {1: embed(3.0 * e_t, 0)})
        write_cycle(tmp_path, 1, {1: embed(5.0 * e_t, 0)}, excitation=2)
        kwargs = {"indices": [1], "attributes": [FACE_ATTRIBUTE]}
        (one,) = load_port_mode_fields(tmp_path, cycle=1, **kwargs)
        (two,) = load_port_mode_fields(tmp_path, cycle=2, **kwargs)
        (other,) = load_port_mode_fields(tmp_path, cycle=1, excitation=2, **kwargs)
        np.testing.assert_allclose(one.e_t, e_t, atol=1e-12)
        np.testing.assert_allclose(two.e_t, 3.0 * e_t, atol=1e-12)
        np.testing.assert_allclose(other.e_t, 5.0 * e_t, atol=1e-12)

    def test_a_cycle_that_was_not_saved_is_reported(self, tmp_path):
        points2, _ = square_mesh(4)
        write_cycle(tmp_path, 1, {1: embed(linear_e(points2), 0)})
        with pytest.raises(FileNotFoundError):
            load_port_mode_fields(
                tmp_path, indices=[1], attributes=[FACE_ATTRIBUTE], cycle=2
            )


# -- reference voltages for align_gauge ------------------------------------------------

# Palace entry index of modes 1, 2, 3 of port "gssg"; the active mode has the
# lowest index;
# the others come after every port, in no particular order of mode.
INDEX_OF_MODE = {1: 1, 2: 5, 3: 4}


def sidecar_entries(name="gssg"):
    """The ``port_modes.json`` entries of one three-mode port."""
    return [
        PortModeEntry(
            index=index,
            port=name,
            portnumber=1,
            mode=mode,
            active=mode == 1,
            excited=True,
            attributes=(FACE_ATTRIBUTE,),
        )
        for mode, index in INDEX_OF_MODE.items()
    ]


def saved_run(root, *, scales=(1.0, 2.0)):
    """A run folder with one saved cycle per scale: the synthetic modes, scaled."""
    _, t_v, t_i, _ = line()
    fields2d = modes_from(t_v[F_MID], t_i[F_MID])
    for cycle, scale in enumerate(scales, start=1):
        by_index = {
            INDEX_OF_MODE[k + 1]: embed(scale * field.e_t, 0, normal=0.25)
            for k, field in enumerate(fields2d)
        }
        write_cycle(root, cycle, by_index, n=MESH_N)
    write_port_modes(root, sidecar_entries())
    return fields2d


class TestReferenceVoltages:
    def test_each_column_is_the_voltage_of_its_mode_in_ascending_mode_order(
        self, tmp_path
    ):
        fields2d = saved_run(tmp_path)
        paths = terminal_ports.terminal_paths(GSSG, DOMAIN)
        expected = np.array(
            [
                [voltage_line_integral(f, paths[n]) for f in fields2d]
                for n in GSSG.terminal_names
            ]
        )
        refs = port_mode_reference_voltages(
            tmp_path, GSSG, tmp_path, cycles=[1, 2], domain=DOMAIN
        )
        assert len(refs) == 2
        np.testing.assert_allclose(refs[0], expected * 1e-6, rtol=1e-9, atol=1e-18)
        np.testing.assert_allclose(
            refs[1], 2.0 * expected * 1e-6, rtol=1e-9, atol=1e-18
        )

    def test_entries_may_be_given_as_a_list_and_paths_instead_of_a_domain(
        self, tmp_path
    ):
        saved_run(tmp_path, scales=(1.0,))
        paths = terminal_ports.terminal_paths(GSSG, DOMAIN)
        from_folder = port_mode_reference_voltages(
            tmp_path, GSSG, tmp_path, cycles=[1], domain=DOMAIN
        )
        from_list = port_mode_reference_voltages(
            tmp_path, GSSG, sidecar_entries(), cycles=[1], paths=paths
        )
        np.testing.assert_array_equal(from_folder[0], from_list[0])

    def test_they_feed_align_gauge_and_fix_its_gauge(self, tmp_path):
        """2D modes with a wrong phase and scale are fitted onto the saved voltages."""
        fields2d = saved_run(tmp_path)
        refs = port_mode_reference_voltages(
            tmp_path, GSSG, tmp_path, cycles=[1, 2], domain=DOMAIN
        )
        mats = port_terminal_matrices(fields2d, GSSG, domain=DOMAIN)
        rng = np.random.default_rng(3)
        gauge = rng.uniform(0.5, 2.0, 3) * np.exp(1j * rng.uniform(-3.0, 3.0, 3))
        n_eff = np.array([3.0, 2.0, 1.0])
        # one 2D solve per saved cycle (1 GHz, 50 GHz), each with its own wrong gauge
        solves = [
            PortModes2D(f, n_eff, mats.t_v * gauge, mats.t_i * gauge)
            for f in (1.0, 50.0)
        ]
        k0 = 2.0 * np.pi * FREQ_GRID * 1e9 / C0
        aligned = align_gauge(
            solves,
            freq_ghz=FREQ_GRID,
            wave_numbers=k0[:, None] * n_eff[None, :],
            reference_voltages=refs,
        )
        assert aligned.gauge_residual is not None
        assert aligned.gauge_residual.max() < 1e-9
        np.testing.assert_allclose(aligned.t_v[0], refs[0], rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(aligned.t_v[-1], refs[1], rtol=1e-9, atol=1e-15)
        assert aligned.pilot == ()

    def test_the_sidecar_must_hold_one_entry_per_terminal(self, tmp_path):
        saved_run(tmp_path, scales=(1.0,))
        with pytest.raises(ValueError, match="needs one saved mode per terminal"):
            port_mode_reference_voltages(
                tmp_path, GSSG, sidecar_entries("other"), cycles=[1], domain=DOMAIN
            )

    def test_missing_inputs_are_reported(self, tmp_path):
        saved_run(tmp_path, scales=(1.0,))
        with pytest.raises(ValueError, match=r"domain= .* or paths="):
            port_mode_reference_voltages(tmp_path, GSSG, tmp_path, cycles=[1])
        with pytest.raises(ValueError, match="cycles is empty"):
            port_mode_reference_voltages(
                tmp_path, GSSG, tmp_path, cycles=[], domain=DOMAIN
            )
        with pytest.raises(ValueError, match="No voltage path"):
            port_mode_reference_voltages(
                tmp_path,
                GSSG,
                tmp_path,
                cycles=[1],
                paths={"G1": [(0.1, 0.4), (0.2, 0.4)]},
            )
