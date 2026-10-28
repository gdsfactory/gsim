"""Terminal port specs, voltage paths, and ``T_V`` / ``T_I`` from saved 2D modes.

The geometry tests need no solver. The matrix tests build the saved fields of
a synthetic multiconductor line instead of solving one: for terminals ``i`` the
fields ``e_i`` and ``h_i`` are polynomials (second order, so the mesh holds them
exactly) chosen so that the voltage of ``e_j`` along terminal ``i``'s path is
``delta_ij``, the contour current of ``h_j`` around terminal ``i`` is
``delta_ij``, and ``integral e_i x h_j dA = delta_ij``. A mode of an RLGC line
with terminal voltages ``V`` and currents ``I`` is then ``E = sum_j V_j e_j`` and
``H = sum_j I_j h_j``, and everything the helpers read off it is known in
closed form. The end-to-end check runs those fields through
:func:`port_terminal_matrices` and :func:`modal_to_terminal_s` and compares with
the chain-matrix solution of the same line. Generic synthetic values only.
"""

from __future__ import annotations

from functools import cache
from itertools import pairwise

import numpy as np
import pytest

from gsim.common.modes import Conductor
from gsim.palace.boundarymode import BoundaryModeSim
from gsim.palace.line_impedance import line_impedance_paths
from gsim.palace.mode_fields import (
    BoundaryModeField,
    contour_current,
    reaction_matrix,
    voltage_line_integral,
)
from gsim.palace.terminal import modal_to_terminal_s, uniform_line_modal_s
from gsim.palace.terminal_ports import (
    PortTerminalMatrices,
    TerminalPortSpec,
    declare_terminal_paths,
    port_terminal_matrices,
    terminal_paths,
)
from tests.palace.test_mode_fields import square_mesh
from tests.palace.test_terminal import ELL, chain_terminal_s, modal_data, rlgc

MESH_N = 16
STEP = 1.0 / MESH_N
DOMAIN = ((0.0, 1.0), (0.0, 1.0))
# Every extent is a multiple of the mesh step, so conductor outlines are mesh
# edges (which the contour current needs). The conductors have different
# heights on purpose: the contour currents of three terminals are three
# independent functionals of H only when their centres are not collinear.
G1 = Conductor("G1", ((0.0625, 0.1875), (0.375, 0.5)))
S1 = Conductor("S1", ((0.3125, 0.375), (0.375, 0.5625)))
S2 = Conductor("S2", ((0.5, 0.5625), (0.3125, 0.5)))
G2 = Conductor("G2", ((0.6875, 0.9375), (0.375, 0.5)))
GSSG = TerminalPortSpec("gssg", terminals=[G1, S1, S2], reference=G2)

TOL_S = 1e-10


# -- spec ----------------------------------------------------------------------


class TestTerminalPortSpec:
    def test_terminals_are_kept_in_order_as_a_tuple(self):
        assert GSSG.terminals == (G1, S1, S2)
        assert GSSG.terminal_names == ("G1", "S1", "S2")

    def test_a_port_without_terminals_is_refused(self):
        with pytest.raises(ValueError, match="no terminals"):
            TerminalPortSpec("p", terminals=[], reference=G2)

    def test_a_port_needs_a_name(self):
        with pytest.raises(ValueError, match="needs a name"):
            TerminalPortSpec("", terminals=[S1], reference=G2)

    def test_repeated_terminal_names_are_refused(self):
        twin = Conductor("S1", ((0.5, 0.5625), (0.375, 0.5)))
        with pytest.raises(ValueError, match="unique"):
            TerminalPortSpec("p", terminals=[S1, twin], reference=G2)

    def test_the_reference_may_not_be_a_terminal(self):
        with pytest.raises(ValueError, match="reference may not also be a terminal"):
            TerminalPortSpec("p", terminals=[S1, G2], reference=G2)

    def test_overlapping_conductors_are_refused(self):
        bump = Conductor("bump", ((0.4, 0.45), (0.4, 0.45)))
        wide = Conductor("wide", ((0.3, 0.55), (0.4, 0.45)))
        with pytest.raises(ValueError, match="overlap or touch"):
            TerminalPortSpec("p", terminals=[S1, wide], reference=G2)
        # Disjoint from S1 and from the reference: fine.
        TerminalPortSpec("p", terminals=[S1, bump], reference=G2)

    def test_touching_conductors_are_refused(self):
        touching = Conductor("T", ((0.375, 0.5), (0.375, 0.5)))
        with pytest.raises(ValueError, match="overlap or touch"):
            TerminalPortSpec("p", terminals=[S1, touching], reference=G2)

    def test_the_reference_may_not_overlap_a_terminal(self):
        under = Conductor("under", ((0.3, 0.4), (0.4, 0.45)))
        with pytest.raises(ValueError, match="overlap or touch"):
            TerminalPortSpec("p", terminals=[S1], reference=under)

    @pytest.mark.parametrize(
        "extent",
        [
            ((0.2, 0.2), (0.3, 0.4)),
            ((0.3, 0.2), (0.3, 0.4)),
            ((0.1, 0.2), (0.4, 0.3)),
            ((0.1, np.nan), (0.3, 0.4)),
        ],
    )
    def test_a_degenerate_rectangle_is_refused(self, extent):
        with pytest.raises(ValueError, match="not a rectangle"):
            TerminalPortSpec("p", terminals=[Conductor("X", extent)], reference=G2)


# -- paths -----------------------------------------------------------------------


def clearance_ok(path, spec):
    """No sampled point of any segment lies in a conductor's closed rectangle."""
    boxes = [c.extent for c in (*spec.terminals, spec.reference)]
    for p, q in pairwise(np.asarray(path, dtype=float)):
        pts = p + np.linspace(0.0, 1.0, 4001)[:, None] * (q - p)
        for (h0, h1), (v0, v1) in boxes:
            inside = (
                (pts[:, 0] >= h0)
                & (pts[:, 0] <= h1)
                & (pts[:, 1] >= v0)
                & (pts[:, 1] <= v1)
            )
            if inside.any():
                return False
    return True


def distance_to_box(point, extent):
    """Euclidean distance from a point to a rectangle (0 inside)."""
    (h0, h1), (v0, v1) = extent
    dh = max(h0 - point[0], 0.0, point[0] - h1)
    dv = max(v0 - point[1], 0.0, point[1] - v1)
    return float(np.hypot(dh, dv))


class TestTerminalPaths:
    def test_there_is_one_path_per_terminal_in_terminal_order(self):
        paths = terminal_paths(GSSG, DOMAIN)
        assert list(paths) == ["G1", "S1", "S2"]
        assert all(len(p) >= 2 for p in paths.values())

    def test_paths_never_cross_a_conductor_and_stay_in_the_domain(self):
        for path in terminal_paths(GSSG, DOMAIN).values():
            assert clearance_ok(path, GSSG)
            assert all(0.0 < h < 1.0 and 0.0 < v < 1.0 for h, v in path)

    def test_paths_start_and_end_a_hair_inside_the_gap(self):
        gap = 0.125
        # PATH_CLEARANCE_FRACTION times the smaller of the gap and the thickness.
        hair = 1e-3 * 0.125
        for name, path in terminal_paths(GSSG, DOMAIN).items():
            own = GSSG.terminals[GSSG.terminal_names.index(name)].extent
            assert distance_to_box(path[0], own) == pytest.approx(hair, rel=1e-9)
            assert distance_to_box(path[-1], G2.extent) == pytest.approx(hair, rel=1e-9)
            assert hair < gap

    def test_a_terminal_facing_the_reference_gets_the_straight_path(self):
        path = terminal_paths(GSSG, DOMAIN)["S2"]
        assert len(path) == 2
        assert path[0][1] == pytest.approx(0.40625)  # S2's mid-height
        assert path[1][1] == pytest.approx(0.4375)  # G2's mid-height

    def test_a_terminal_behind_another_conductor_goes_over_the_metal(self):
        paths = terminal_paths(GSSG, DOMAIN)
        for name, mid_height in (("G1", 0.4375), ("S1", 0.46875)):
            path = paths[name]
            assert len(path) == 4
            assert max(v for _, v in path) > 0.5625  # above every conductor
            # leaves into the gap on its right, comes down on top of the reference
            assert path[0][1] == pytest.approx(mid_height)
            assert path[-1][1] == pytest.approx(0.5 + 1e-3 * 0.125)
            assert path[-1][0] == pytest.approx(0.8125)

    def test_the_straight_path_is_the_one_the_line_impedance_code_draws(self):
        domain = ((0.0, 100.0), (0.0, 50.0))
        signal = Conductor("S", ((40.0, 50.0), (20.0, 22.0)))
        ground = Conductor("G", ((56.0, 96.0), (20.0, 22.0)))
        spec = TerminalPortSpec("cpw", terminals=[signal], reference=ground)
        drawn = line_impedance_paths(
            signal=signal.extent, ground=ground.extent, domain=domain
        )
        got = terminal_paths(spec, domain)["S"]
        np.testing.assert_allclose(got, drawn.voltage, rtol=0, atol=1e-12)

    def test_a_terminal_above_a_ground_plane_gets_the_vertical_path(self):
        domain = ((-1.0, 2.0), (-1.0, 2.0))
        plane = Conductor("plane", ((0.0, 1.0), (0.0, 0.1)))
        strip = Conductor("strip", ((0.4, 0.6), (0.3, 0.4)))
        spec = TerminalPortSpec("ms", terminals=[strip], reference=plane)
        path = terminal_paths(spec, domain)["strip"]
        hair = 1e-3 * 0.2  # the smaller of the 0.2 gap and the strip's 0.2 width
        np.testing.assert_allclose(
            path, ((0.5, 0.3 - hair), (0.5, 0.1 + hair)), rtol=0, atol=1e-12
        )
        assert clearance_ok(path, spec)

    def test_a_terminal_to_the_right_of_the_reference_exits_to_its_left(self):
        left_ref = Conductor("ref", ((0.0625, 0.1875), (0.375, 0.5)))
        near = Conductor("near", ((0.3125, 0.375), (0.375, 0.5)))
        far = Conductor("far", ((0.5, 0.5625), (0.375, 0.5)))
        spec = TerminalPortSpec("p", terminals=[near, far], reference=left_ref)
        paths = terminal_paths(spec, DOMAIN)
        assert len(paths["near"]) == 2
        far_path = paths["far"]
        assert len(far_path) == 4
        assert far_path[0][0] < far.extent[0][0]  # leaves on the reference's side
        assert clearance_ok(far_path, spec)

    def test_a_path_that_would_cross_a_conductor_is_refused(self):
        # A lid spanning the domain closes the way over the metal; the terminal
        # 'a' is hidden behind 'block' from the reference and must go over.
        spec = TerminalPortSpec(
            "p",
            terminals=[
                Conductor("a", ((0.05, 0.15), (0.2, 0.3))),
                Conductor("block", ((0.25, 0.35), (0.2, 0.3))),
                Conductor("lid", ((0.0, 1.0), (0.8, 0.9))),
            ],
            reference=Conductor("ref", ((0.45, 0.55), (0.2, 0.3))),
        )
        with pytest.raises(ValueError, match="'a'"):
            terminal_paths(spec, DOMAIN)

    def test_a_domain_without_room_above_the_metal_is_refused(self):
        low = ((0.0, 1.0), (0.0, 0.5625))  # the highest conductor's top is at the wall
        with pytest.raises(ValueError, match="No voltage path"):
            terminal_paths(GSSG, low)

    def test_a_conductor_outside_the_domain_is_refused(self):
        small = ((0.0, 0.8), (0.0, 1.0))
        with pytest.raises(ValueError, match="not inside the domain"):
            terminal_paths(GSSG, small)

    def test_a_domain_that_is_not_a_rectangle_is_refused(self):
        with pytest.raises(ValueError, match="not a rectangle"):
            terminal_paths(GSSG, ((1.0, 0.0), (0.0, 1.0)))


class TestDeclareTerminalPaths:
    def test_every_terminal_is_declared_under_its_own_index(self):
        sim = BoundaryModeSim()
        indices = declare_terminal_paths(sim, GSSG, DOMAIN)
        assert indices == [1, 2, 3]
        assert [p.name for p in sim.mode_paths] == ["gssg:G1", "gssg:S1", "gssg:S2"]
        voltage = sim.mode_postprocessing()["Voltage"]
        paths = terminal_paths(GSSG, DOMAIN)
        for entry, name in zip(voltage, GSSG.terminal_names, strict=True):
            written = np.asarray(entry["VoltagePath"], dtype=float)
            np.testing.assert_allclose(written, paths[name])

    def test_declaring_again_replaces_the_paths_and_keeps_the_indices(self):
        sim = BoundaryModeSim()
        sim.add_impedance_path("probe", voltage=[[0.0, 0.1], [0.0, 0.2]])
        first = declare_terminal_paths(sim, GSSG, DOMAIN)
        second = declare_terminal_paths(sim, GSSG, DOMAIN)
        assert first == second == [2, 3, 4]
        assert len(sim.mode_paths) == 4

    def test_the_sample_count_reaches_palace(self):
        sim = BoundaryModeSim()
        declare_terminal_paths(sim, GSSG, DOMAIN, nsamples=37)
        assert {p["NSamples"] for p in sim.mode_postprocessing()["Voltage"]} == {37}


# -- synthetic saved modes -------------------------------------------------------


def mono(points):
    """The six monomials 1, x, y, x^2, xy, y^2 at ``(P, 2)`` points."""
    x, y = points[:, 0], points[:, 1]
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(3)


def path_functional(path):
    """Row ``c`` with ``c . [ex(6), ey(6)] = integral of E . dl`` along a polyline."""
    row = np.zeros(12)
    t = 0.5 * (_NODES + 1.0)
    w = 0.5 * _WEIGHTS
    for p, q in pairwise(np.asarray(path, dtype=float)):
        step = q - p
        m = mono(p[None, :] + t[:, None] * step[None, :])
        row[:6] += step[0] * (w @ m)
        row[6:] += step[1] * (w @ m)
    return row


def contour_functional(extent):
    """The same for the counter-clockwise loop around a rectangle."""
    (h0, h1), (v0, v1) = extent
    return path_functional([(h0, v0), (h1, v0), (h1, v1), (h0, v1), (h0, v0)])


def mass_matrix():
    """``M_ab = integral m_a m_b dA`` over the unit square."""
    t = 0.5 * (_NODES + 1.0)
    grid_x, grid_y = np.meshgrid(t, t)
    weight = np.outer(0.5 * _WEIGHTS, 0.5 * _WEIGHTS).ravel()
    m = mono(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1))
    return m.T @ (weight[:, None] * m)


@cache
def basis(spec=GSSG):
    """Real polynomial fields ``e_j``, ``h_j`` with the dual normalisation."""
    count = len(spec.terminals)
    paths = terminal_paths(spec, DOMAIN)
    v_rows = np.array([path_functional(paths[n]) for n in spec.terminal_names])
    e_coef = np.linalg.lstsq(v_rows, np.eye(count), rcond=None)[0]
    mass = mass_matrix()
    i_rows = np.array([contour_functional(t.extent) for t in spec.terminals])
    p_rows = np.array(
        [
            np.concatenate([-(mass @ e_coef[6:, i]), mass @ e_coef[:6, i]])
            for i in range(count)
        ]
    )
    h_coef = np.linalg.lstsq(
        np.vstack([i_rows, p_rows]),
        np.vstack([np.eye(count), np.eye(count)]),
        rcond=None,
    )[0]
    return e_coef, h_coef


@cache
def mesh():
    """The saved-mode mesh: second-order triangles on the unit square."""
    return square_mesh(MESH_N)


def modes_from(t_v, t_i, *, spec=GSSG):
    """Saved fields of the modes whose terminal voltages and currents are given."""
    e_coef, h_coef = basis(spec)
    points, cells = mesh()
    m = mono(points)
    zeros = np.zeros(points.shape[0], dtype=np.complex128)
    out = []
    for k in range(t_v.shape[1]):
        e = e_coef @ t_v[:, k]
        h = h_coef @ t_i[:, k]
        out.append(
            BoundaryModeField(
                points_um=points,
                cells=cells,
                attribute=np.ones(cells.shape[0], dtype=np.int64),
                e_t=np.stack([m @ e[:6], m @ e[6:]], axis=1).astype(np.complex128),
                e_n=zeros,
                h_t=np.stack([m @ h[:6], m @ h[6:]], axis=1).astype(np.complex128),
                h_n=zeros,
            )
        )
    return out


@cache
def line():
    """Modes of the three-conductor lossy RLGC line at every frequency."""
    rlgc_pul = rlgc("three")
    t_v, t_i, gamma = modal_data(*rlgc_pul)
    return rlgc_pul, t_v, t_i, gamma


F_MID = 4


def matrices_at(index, **kwargs):
    """``port_terminal_matrices`` of the synthetic modes at one frequency."""
    _, t_v, t_i, _ = line()
    kwargs.setdefault("domain", DOMAIN)
    return port_terminal_matrices(modes_from(t_v[index], t_i[index]), GSSG, **kwargs)


class TestSyntheticBasis:
    """The synthetic fields do what the module docstring says they do."""

    def test_voltage_current_and_reaction_duals_are_exact(self):
        eye = np.eye(3)
        fields = modes_from(eye, eye)
        paths = terminal_paths(GSSG, DOMAIN)
        v = np.array(
            [
                [voltage_line_integral(f, paths[n]) for f in fields]
                for n in GSSG.terminal_names
            ]
        )
        i = np.array(
            [
                [
                    contour_current(f, h_span=t.extent[0], v_span=t.extent[1])
                    for f in fields
                ]
                for t in GSSG.terminals
            ]
        )
        np.testing.assert_allclose(v, eye, atol=1e-10)
        np.testing.assert_allclose(i, eye, atol=1e-10)
        np.testing.assert_allclose(reaction_matrix(fields), eye, atol=1e-10)


# -- T_V and T_I -------------------------------------------------------------------


@pytest.mark.parametrize("current", ["contour", "reaction"])
def test_the_terminal_s_matches_the_chain_matrix_of_the_line(current):
    """Fields -> T_V, T_I -> modal_to_terminal_s equals the telegrapher solution."""
    rlgc_pul, t_v, _, gamma = line()
    n_freq = t_v.shape[0]
    mats = [matrices_at(f, current=current) for f in range(n_freq)]
    tv = np.array([m.t_v for m in mats])
    ti = np.array([m.t_i for m in mats])
    s_modal = uniform_line_modal_s(gamma, ELL)
    s_term = modal_to_terminal_s(s_modal, [tv, tv], [ti, ti], z_ref=50.0)
    reference = chain_terminal_s(*rlgc_pul)
    assert np.abs(s_term - reference).max() < TOL_S


def test_the_contour_and_the_reaction_route_agree():
    contour = matrices_at(F_MID, current="contour")
    reaction = matrices_at(F_MID, current="reaction")
    np.testing.assert_allclose(contour.t_v, reaction.t_v, rtol=1e-9, atol=1e-12)
    scale = np.abs(contour.t_i).max()
    np.testing.assert_allclose(contour.t_i, reaction.t_i, rtol=0, atol=1e-10 * scale)


@pytest.mark.parametrize("current", ["contour", "reaction"])
class TestColumnConventions:
    def test_every_column_carries_unit_power_into_the_port(self, current):
        mats = matrices_at(F_MID, current=current)
        for k in range(3):
            reaction = np.vdot(mats.t_i[:, k], mats.t_v[:, k])  # sum V conj(I)
            assert abs(reaction) == pytest.approx(1.0, rel=1e-9)
            assert np.real(np.vdot(mats.t_v[:, k], mats.t_i[:, k])) > 0.0

    def test_the_pilot_voltage_is_real_and_positive(self, current):
        mats = matrices_at(F_MID, current=current)
        for k, row in enumerate(mats.pilot):
            assert mats.t_v[row, k].imag == pytest.approx(0.0, abs=1e-12)
            assert mats.t_v[row, k].real > 0.0
            assert abs(mats.t_v[row, k]) == pytest.approx(np.abs(mats.t_v[:, k]).max())

    def test_the_mode_order_only_orders_the_columns(self, current):
        _, t_v, t_i, _ = line()
        fields = modes_from(t_v[F_MID], t_i[F_MID])
        base = port_terminal_matrices(fields, GSSG, domain=DOMAIN, current=current)
        flipped = port_terminal_matrices(
            fields[::-1], GSSG, domain=DOMAIN, current=current
        )
        np.testing.assert_allclose(flipped.t_v, base.t_v[:, ::-1], atol=1e-9)
        np.testing.assert_allclose(flipped.t_i, base.t_i[:, ::-1], atol=1e-9)

    def test_the_saved_amplitude_and_phase_do_not_matter(self, current):
        _, t_v, t_i, _ = line()
        rng = np.random.default_rng(7)
        factors = rng.uniform(0.2, 5.0, 3) * np.exp(1j * rng.uniform(-3, 3, 3))
        base = port_terminal_matrices(
            modes_from(t_v[F_MID], t_i[F_MID]), GSSG, domain=DOMAIN, current=current
        )
        scaled = port_terminal_matrices(
            modes_from(t_v[F_MID] * factors, t_i[F_MID] * factors),
            GSSG,
            domain=DOMAIN,
            current=current,
        )
        np.testing.assert_allclose(scaled.t_v, base.t_v, atol=1e-9)
        np.testing.assert_allclose(scaled.t_i, base.t_i, atol=1e-9)
        np.testing.assert_allclose(scaled.power, base.power * np.abs(factors) ** 2)

    def test_a_port_facing_the_other_way_gives_the_same_matrices(self, current):
        """Reversing H (the face points the other way) is undone by the orientation."""
        _, t_v, t_i, _ = line()
        base = port_terminal_matrices(
            modes_from(t_v[F_MID], t_i[F_MID]), GSSG, domain=DOMAIN, current=current
        )
        reversed_h = port_terminal_matrices(
            modes_from(t_v[F_MID], -t_i[F_MID]), GSSG, domain=DOMAIN, current=current
        )
        np.testing.assert_allclose(reversed_h.t_v, base.t_v, atol=1e-9)
        np.testing.assert_allclose(reversed_h.t_i, base.t_i, atol=1e-9)

    def test_a_named_pilot_sets_the_phase_reference(self, current):
        mats = matrices_at(F_MID, current=current, pilot=["S2", "S2", "S2"])
        assert mats.pilot == (2, 2, 2)
        np.testing.assert_allclose(mats.t_v[2].imag, 0.0, atol=1e-12)
        assert np.all(mats.t_v[2].real > 0.0)
        as_index = matrices_at(F_MID, current=current, pilot=[2, 2, 2])
        np.testing.assert_allclose(as_index.t_v, mats.t_v, atol=1e-12)

    def test_the_paths_can_be_given_instead_of_the_domain(self, current):
        paths = terminal_paths(GSSG, DOMAIN)
        by_domain = matrices_at(F_MID, current=current)
        by_paths = matrices_at(F_MID, current=current, domain=None, paths=paths)
        np.testing.assert_allclose(by_paths.t_v, by_domain.t_v, atol=1e-12)


def test_the_result_carries_its_audit_numbers():
    mats = matrices_at(F_MID)
    assert isinstance(mats, PortTerminalMatrices)
    assert mats.t_v.shape == mats.t_i.shape == (3, 3)
    assert mats.cond == pytest.approx(np.linalg.cond(mats.t_v))
    assert mats.power.shape == (3,)
    assert np.all(mats.power > 0.0)


class TestRefusals:
    def _fields(self):
        _, t_v, t_i, _ = line()
        return modes_from(t_v[F_MID], t_i[F_MID])

    def test_fewer_modes_than_terminals_is_refused(self):
        with pytest.raises(ValueError, match="one saved mode per terminal"):
            port_terminal_matrices(self._fields()[:2], GSSG, domain=DOMAIN)

    def test_two_copies_of_one_mode_do_not_span_the_terminals(self):
        first = self._fields()
        fields = [first[0], first[0], first[2]]
        for current in ("contour", "reaction"):
            with pytest.raises(ValueError, match="ill-conditioned"):
                port_terminal_matrices(fields, GSSG, domain=DOMAIN, current=current)

    def test_modes_that_load_the_terminals_alike_do_not_span_them(self):
        """Three modes that all put the same voltage pattern on the terminals."""
        _, t_v, t_i, _ = line()
        pattern = t_v[F_MID][:, :1]
        v = pattern @ np.array([[1.0, 1.0 + 1e-14, 1.0 + 2e-14]])
        i = t_i[F_MID][:, :1] @ np.array([[1.0, 1.0 + 1e-14, 1.0 + 2e-14]])
        with pytest.raises(ValueError, match="ill-conditioned"):
            port_terminal_matrices(modes_from(v, i), GSSG, domain=DOMAIN)

    def test_a_mode_without_power_is_refused(self):
        fields = self._fields()
        zero = BoundaryModeField(
            points_um=fields[0].points_um,
            cells=fields[0].cells,
            attribute=fields[0].attribute,
            e_t=np.zeros_like(fields[0].e_t),
            e_n=fields[0].e_n,
            h_t=np.zeros_like(fields[0].h_t),
            h_n=fields[0].h_n,
        )
        with pytest.raises(ValueError, match=r"Mode\(s\) \[2\].*carry no power"):
            port_terminal_matrices([fields[0], zero, fields[2]], GSSG, domain=DOMAIN)

    def test_a_reactive_mode_cannot_be_oriented(self):
        _, t_v, _, _ = line()
        with pytest.raises(ValueError, match="no real power"):
            port_terminal_matrices(
                modes_from(t_v[F_MID], 1j * t_v[F_MID] * 1e-2), GSSG, domain=DOMAIN
            )

    def test_the_paths_or_the_domain_are_needed(self):
        with pytest.raises(ValueError, match=r"domain= .* or paths="):
            port_terminal_matrices(self._fields(), GSSG)

    def test_a_terminal_without_a_path_is_refused(self):
        paths = dict(terminal_paths(GSSG, DOMAIN))
        del paths["S1"]
        with pytest.raises(ValueError, match="No voltage path was given"):
            port_terminal_matrices(self._fields(), GSSG, paths=paths)

    def test_an_unknown_current_route_is_refused(self):
        with pytest.raises(ValueError, match="'contour' or 'reaction'"):
            port_terminal_matrices(
                self._fields(),
                GSSG,
                domain=DOMAIN,
                current="power",  # ty: ignore[invalid-argument-type]
            )

    @pytest.mark.parametrize("pilot", [["G1", "G1"], ["G1", "G1", "nope"], [0, 1, 3]])
    def test_a_bad_pilot_is_refused(self, pilot):
        with pytest.raises(ValueError, match="pilot"):
            port_terminal_matrices(self._fields(), GSSG, domain=DOMAIN, pilot=pilot)
