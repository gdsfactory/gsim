"""Voltage line integrals and the reaction matrix of saved Palace Modes.

The two helpers are what a terminal port needs beyond the power and the
contour current :mod:`gsim.palace.mode_fields` already integrates: the voltage
along a path from a terminal to the reference, and the unconjugated reaction
``R_jk`` of every pair of Modes. Both are quadrature problems on a saved
field, so both have closed-form answers on fields written down by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.palace.mode_fields import (
    BoundaryModeField,
    power_flux,
    reaction_matrix,
    voltage_line_integral,
)
from tests.palace.test_mode_fields import field_from, square_mesh, uniform

PATH = [(0.1, 0.2), (0.8, 0.3), (0.6, 0.9), (0.2, 0.7)]


def linear_field(offset, gradient):
    """``E(p) = offset + gradient @ p``: a field the mesh holds exactly."""
    offset = np.asarray(offset, dtype=np.complex128)
    gradient = np.asarray(gradient, dtype=np.complex128)

    def build(points):
        return offset[None, :] + points @ gradient.T

    return build


def polyline_integral_of_linear(e_of, path):
    """Exact integral of a linear field: the trapezoid rule is exact for it."""
    pts = np.asarray(path, dtype=float)
    ends = e_of(pts)
    steps = np.diff(pts, axis=0)
    return complex(np.sum(0.5 * (ends[:-1] + ends[1:]) * steps))


def tem_field(e_of, admittance=1.0, *, n=4):
    """A TEM-like Mode: ``H = y z-hat x E``, so that ``E x H`` is ``y |E|^2``."""

    def h_of(points):
        e = e_of(points)
        return admittance * np.stack([-e[:, 1], e[:, 0]], axis=1)

    return field_from(e_of, h_of, n=n)


def square_with_a_hole(n=8, lo=0.3, hi=0.7) -> BoundaryModeField:
    """The unit square with the triangles centred in ``[lo, hi]^2`` removed."""
    points, cells = square_mesh(n)
    centre = points[cells[:, :3]].mean(axis=1)
    keep = ~(
        (centre[:, 0] > lo)
        & (centre[:, 0] < hi)
        & (centre[:, 1] > lo)
        & (centre[:, 1] < hi)
    )
    assert not keep.all()
    cells = cells[keep]
    zeros = np.zeros(points.shape[0], dtype=np.complex128)
    return BoundaryModeField(
        points_um=points,
        cells=cells,
        attribute=np.ones(cells.shape[0], dtype=np.int64),
        e_t=np.stack([points[:, 0] + 0j, points[:, 1] + 0j], axis=1),
        e_n=zeros,
        h_t=np.zeros((points.shape[0], 2), dtype=np.complex128),
        h_n=zeros,
    )


class TestVoltageLineIntegral:
    def test_a_linear_field_integrates_to_the_analytic_value(self):
        e_of = linear_field([0.3 + 0.1j, -0.2j], [[1.0, 2.0 - 1j], [0.5j, -1.5]])
        field = field_from(e_of, uniform([0.0, 0.0]))
        got = voltage_line_integral(field, PATH)
        assert got == pytest.approx(polyline_integral_of_linear(e_of, PATH), abs=1e-12)

    def test_a_quadratic_field_is_integrated_exactly_across_elements(self):
        """Second-order elements hold a quadratic exactly; so must the integral."""

        def e_of(points):
            x, y = points[:, 0], points[:, 1]
            return np.stack([(1 + 2j) * (x**2 + x * y), 0.0 * x], axis=1)

        field = field_from(e_of, uniform([0.0, 0.0]))
        path = [(0.1, 0.3), (0.9, 0.3)]
        analytic = (1 + 2j) * ((0.9**3 - 0.1**3) / 3.0 + 0.3 * (0.9**2 - 0.1**2) / 2.0)
        assert voltage_line_integral(field, path) == pytest.approx(analytic, abs=1e-12)

    def test_the_sign_follows_the_direction_of_the_path(self):
        e_of = linear_field([1.0, 0.5], [[0.2, 0.1], [0.0, 0.3]])
        field = field_from(e_of, uniform([0.0, 0.0]))
        forward = voltage_line_integral(field, PATH)
        backward = voltage_line_integral(field, PATH[::-1])
        assert backward == pytest.approx(-forward, abs=1e-12)

    def test_the_sample_count_does_not_change_a_linear_field(self):
        e_of = linear_field([0.3, 0.1], [[1.0, 2.0], [0.5, -1.5]])
        field = field_from(e_of, uniform([0.0, 0.0]))
        coarse = voltage_line_integral(field, PATH, nsamples=1)
        fine = voltage_line_integral(field, PATH, nsamples=400)
        assert coarse == pytest.approx(fine, abs=1e-12)

    def test_a_path_outside_the_mesh_is_reported(self):
        field = field_from(uniform([1.0, 0.0]), uniform([0.0, 0.0]))
        with pytest.raises(ValueError, match="outside the meshed cross-section"):
            voltage_line_integral(field, [(0.5, 0.5), (1.5, 0.5)])

    def test_a_path_through_a_conductor_hole_is_reported(self):
        field = square_with_a_hole()
        with pytest.raises(ValueError, match="perfect conductor"):
            voltage_line_integral(field, [(0.1, 0.5), (0.9, 0.5)])

    def test_a_path_around_the_hole_is_integrated(self):
        field = square_with_a_hole()
        path = [(0.1, 0.5), (0.1, 0.9), (0.9, 0.9), (0.9, 0.5)]
        # E = (x, y): the integral is the potential difference of x^2/2 + y^2/2.
        expected = 0.5 * (0.9**2 + 0.5**2) - 0.5 * (0.1**2 + 0.5**2)
        assert voltage_line_integral(field, path) == pytest.approx(expected, abs=1e-12)

    def test_a_point_a_hair_off_the_mesh_is_tolerated_and_a_gap_is_not(self):
        e_of = linear_field([1.0, 0.0], [[0.5, 0.0], [0.0, 0.0]])
        field = field_from(e_of, uniform([0.0, 0.0]))
        near = [(0.5, 0.5), (1.0 + 5e-5, 0.5)]
        assert voltage_line_integral(field, near) == pytest.approx(
            polyline_integral_of_linear(e_of, near), abs=1e-12
        )
        with pytest.raises(ValueError, match="outside the meshed cross-section"):
            voltage_line_integral(field, [(0.5, 0.5), (1.0 + 1e-3, 0.5)])
        assert voltage_line_integral(
            field, [(0.5, 0.5), (1.0 + 1e-3, 0.5)], tol_um=1e-2
        ) == pytest.approx(
            polyline_integral_of_linear(e_of, [(0.5, 0.5), (1.0 + 1e-3, 0.5)]),
            abs=1e-12,
        )

    @pytest.mark.parametrize(
        ("path", "message"),
        [
            ([(0.5, 0.5)], "M >= 2"),
            ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "M >= 2"),
            ([(0.2, 0.2), (0.2, 0.2), (0.5, 0.5)], "repeats a point"),
            ([(0.2, 0.2), (np.nan, 0.5)], "non-finite"),
        ],
    )
    def test_a_malformed_path_is_refused(self, path, message):
        field = field_from(uniform([1.0, 0.0]), uniform([0.0, 0.0]))
        with pytest.raises(ValueError, match=message):
            voltage_line_integral(field, path)

    def test_a_sample_count_below_one_is_refused(self):
        field = field_from(uniform([1.0, 0.0]), uniform([0.0, 0.0]))
        with pytest.raises(ValueError, match="nsamples"):
            voltage_line_integral(field, PATH, nsamples=0)


class TestReactionMatrix:
    def test_reciprocal_modes_give_a_symmetric_reaction(self):
        """``H = y z x E`` with one ``y``: ``R_jk = y integral E_j . E_k``."""

        def e_radial(points):
            return np.stack([points[:, 0], points[:, 1]], axis=1)

        def e_swirl(points):
            return np.stack([-points[:, 1], points[:, 0]], axis=1)

        admittance = 0.5
        modes = [
            tem_field(e_radial, admittance),
            tem_field(e_swirl, admittance),
        ]
        reaction = reaction_matrix(modes)
        expected = admittance * (2.0 / 3.0) * np.eye(2)
        assert reaction.shape == (2, 2)
        np.testing.assert_allclose(reaction, expected, atol=1e-12)
        np.testing.assert_allclose(reaction, reaction.T, atol=1e-12)

    def test_overlapping_modes_stay_symmetric(self):
        e_one = uniform([1.0, 0.5])
        e_two = uniform([-0.3, 2.0])
        admittance = 0.8
        modes = [tem_field(e_one, admittance), tem_field(e_two, admittance)]
        reaction = reaction_matrix(modes)
        overlap = admittance * (1.0 * -0.3 + 0.5 * 2.0)
        assert reaction[0, 1] == pytest.approx(overlap, abs=1e-12)
        np.testing.assert_allclose(reaction, reaction.T, atol=1e-12)

    def test_one_real_mode_reacts_with_itself_as_twice_its_power(self):
        def e_of(points):
            return np.stack([points[:, 0] + 0.2, 0.5 * points[:, 1]], axis=1)

        mode = tem_field(e_of, 1.7)
        assert reaction_matrix([mode])[0, 0] == pytest.approx(
            2.0 * power_flux(mode), rel=1e-12
        )

    def test_the_reaction_does_not_conjugate_the_magnetic_field(self):
        e0, h0 = 2.0 + 0.0j, 0.0 + 3.0j
        mode = field_from(uniform([0.0, e0]), uniform([h0, 0.0]))
        # (E x H) . z = Ex Hy - Ey Hx = -e0 h0 over the unit area.
        assert reaction_matrix([mode])[0, 0] == pytest.approx(-e0 * h0, rel=1e-12)
        assert reaction_matrix([mode])[0, 0] != pytest.approx(
            2.0 * power_flux(mode), rel=1e-6
        )

    def test_modes_on_different_meshes_are_refused(self):
        coarse = field_from(uniform([1.0, 0.0]), uniform([0.0, 1.0]), n=2)
        fine = field_from(uniform([1.0, 0.0]), uniform([0.0, 1.0]), n=4)
        with pytest.raises(ValueError, match="not on the mesh of Mode 0"):
            reaction_matrix([coarse, fine])

    def test_no_modes_is_refused(self):
        with pytest.raises(ValueError, match="at least one Mode"):
            reaction_matrix([])
