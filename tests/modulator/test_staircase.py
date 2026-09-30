"""Hermetic tests for the Strip averages and the node-to-strip transfer.

Everything that reduces a Carrier map to per-Strip averages is under
test here: the one-dimensional Strip averages, the two-dimensional node cloud
reduced onto it, and the Strips a Staircase draws from the result.
"""

from __future__ import annotations

import warnings
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from gsim.modulator.staircase import (
    OpticalStripMaterial,
    RFStripMaterial,
    StaircaseDrawing,
    build_staircase_cross_section,
    staircase_profile,
    strip_averages_from_nodes,
)
from tests._helpers import fake_coupling


class TestStaircaseProfile:
    def test_single_strip_recovers_average(self):
        # N=1 must recover the exact average of the piecewise-linear profile.
        h = np.array([0.0, 1.0, 2.0])
        v = np.array([0.0, 2.0, 0.0])  # triangle, mean = 1.0
        edges, means = staircase_profile(h, v, n_bins=1)
        assert edges == pytest.approx([0.0, 2.0])
        assert means == pytest.approx([1.0])

    def test_constant_profile_any_n(self):
        h = np.linspace(0.0, 4.0, 9)
        v = np.full(9, 7.5)
        edges, means = staircase_profile(h, v, n_bins=5)
        assert len(edges) == 6
        assert means == pytest.approx(np.full(5, 7.5))

    def test_strips_partition_window(self):
        h = np.linspace(-1.0, 1.0, 21)
        v = h**2
        edges, _means = staircase_profile(h, v, n_bins=4, h_min=-0.5, h_max=0.5)
        assert edges[0] == pytest.approx(-0.5)
        assert edges[-1] == pytest.approx(0.5)
        assert np.all(np.diff(edges) > 0)

    def test_convergence_with_n(self):
        # Staircase approximation of a smooth profile converges in L2 as N grows.
        h = np.linspace(0.0, 1.0, 401)
        v = np.exp(-((h - 0.5) ** 2) / 0.01)

        def l2_error(n_bins: int) -> float:
            edges, means = staircase_profile(h, v, n_bins=n_bins)
            approx = np.interp(h, edges[:-1], means, left=means[0], right=means[-1])
            # Evaluate staircase exactly: index of Strip per sample.
            idx = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, n_bins - 1)
            approx = means[idx]
            return float(np.sqrt(np.trapezoid((approx - v) ** 2, h)))

        errors = [l2_error(n) for n in (2, 8, 32)]
        assert errors[0] > errors[1] > errors[2]

    def test_unsorted_input_sorted_internally(self):
        h = np.array([2.0, 0.0, 1.0])
        v = np.array([0.0, 0.0, 2.0])
        _edges, means = staircase_profile(h, v, n_bins=1)
        assert means == pytest.approx([1.0])

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            staircase_profile(np.array([0.0]), np.array([1.0]), n_bins=1)
        with pytest.raises(ValueError):
            staircase_profile(np.array([0.0, 1.0]), np.array([1.0, 1.0]), n_bins=0)
        with pytest.raises(ValueError):
            staircase_profile(
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
                n_bins=2,
                h_min=1.0,
                h_max=0.0,
            )


class TestStripAveragesFromNodes:
    def test_single_strip_recovers_mean_of_linear_profile(self):
        h = np.linspace(0.0, 1.0, 101)
        values = 2.0 * h  # mean 1.0
        edges, means = strip_averages_from_nodes(h, values, n_strips=1)
        np.testing.assert_allclose(edges, [0.0, 1.0])
        assert means[0] == pytest.approx(1.0)

    def test_vertical_band_filters_2d_node_cloud(self):
        # Two rows of nodes; only the y ~ 0 row carries the profile.
        h = np.concatenate([np.linspace(0.0, 1.0, 51), np.linspace(0.0, 1.0, 51)])
        v = np.concatenate([np.zeros(51), np.ones(51)])
        values = np.concatenate([np.linspace(0.0, 2.0, 51), np.full(51, 100.0)])
        _edges, means = strip_averages_from_nodes(
            h, values, n_strips=1, v_um=v, v_range=(-0.1, 0.1)
        )
        assert means[0] == pytest.approx(1.0)

    def test_error_decreases_with_strip_count(self):
        h = np.linspace(-1.0, 1.0, 401)
        values = np.tanh(5.0 * h)

        def reconstruction_error(n_strips):
            edges, means = strip_averages_from_nodes(h, values, n_strips=n_strips)
            idx = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, n_strips - 1)
            return float(np.sqrt(np.mean((means[idx] - values) ** 2)))

        errors = [reconstruction_error(n) for n in (1, 4, 16, 64)]
        assert errors == sorted(errors, reverse=True)
        assert errors[-1] < 0.05 * errors[0]

    def test_values_sharing_a_coordinate_are_averaged(self):
        # Two rows inside the band carrying different fields: the strip value
        # is the average over the band, not whichever node survives dedup.
        h_row = np.linspace(0.0, 1.0, 51)
        h = np.concatenate([h_row, h_row])
        v = np.concatenate([np.zeros(51), np.full(51, 0.05)])
        values = np.concatenate([np.zeros(51), 2.0 * h_row])
        _edges, means = strip_averages_from_nodes(
            h, values, n_strips=1, v_um=v, v_range=(-0.1, 0.1)
        )
        # The field is bilinear, the interpolant linear on triangles: which
        # diagonal splits a cell is the triangulation's choice.
        assert means[0] == pytest.approx(0.5, rel=1e-3)

    def test_strips_track_the_band_averaged_profile(self):
        # A field varying across the band as well as along it: each strip
        # must land on the band average, not on one row.
        h_row = np.linspace(0.0, 1.0, 41)
        rows = np.linspace(0.0, 0.1, 5)
        h = np.tile(h_row, rows.size)
        v = np.repeat(rows, h_row.size)
        values = h + 10.0 * v
        _edges, means = strip_averages_from_nodes(
            h, values, n_strips=4, v_um=v, v_range=(-0.01, 0.11)
        )
        expected = np.array([0.125, 0.375, 0.625, 0.875]) + 10.0 * rows.mean()
        np.testing.assert_allclose(means, expected, rtol=1e-6)

    def test_an_irregular_cloud_lands_on_the_analytic_average(self):
        # What a charge-solve mesh actually hands over: columns of unequal
        # height, at coordinates agreeing only to rounding, carrying a
        # field that varies along the band as well as across it.
        rng = np.random.default_rng(0)
        columns = np.linspace(-0.5, 0.5, 240)
        h_parts, v_parts, value_parts = [], [], []
        for h in columns:
            rows = rng.integers(3, 12)
            z = rng.uniform(0.0, 0.22, rows)
            jitter = rng.normal(scale=1e-12, size=rows)
            h_parts.append(np.full(rows, h) + jitter)
            v_parts.append(z)
            value_parts.append(np.exp(-10.0 * h**2) + 4.0 * z)
        h = np.concatenate(h_parts)
        v = np.concatenate(v_parts)
        values = np.concatenate(value_parts)

        edges, means = strip_averages_from_nodes(
            h, values, n_strips=6, v_um=v, v_range=(0.0, 0.22)
        )

        # The band average of the field is exp(-10 h^2) + 4 * mean(z),
        # integrated over each strip.
        dense = np.linspace(-0.5, 0.5, 20001)
        profile = np.exp(-10.0 * dense**2) + 4.0 * 0.11
        expected = [
            profile[(dense >= lo) & (dense <= hi)].mean() for lo, hi in pairwise(edges)
        ]
        np.testing.assert_allclose(means, expected, rtol=0.05)

    def test_a_strip_is_averaged_over_its_area_not_over_its_nodes(self):
        # A charge-solve mesh refines the silicon's surfaces: a third of
        # its nodes sit on the top and bottom lines, which have no area.
        # Once the carriers vary with depth — the surfaces deplete first
        # when the oxide takes part in the electrostatics — an average
        # that counts nodes reads the surfaces' value for the Strip.
        height = 0.22
        surface_h = np.linspace(0.0, 1.0, 201)
        interior_h = np.linspace(0.0, 1.0, 21)
        interior_z = np.linspace(0.0, height, 9)[1:-1]
        h = np.concatenate([surface_h, surface_h, np.tile(interior_h, interior_z.size)])
        v = np.concatenate(
            [
                np.zeros(surface_h.size),
                np.full(surface_h.size, height),
                np.repeat(interior_z, interior_h.size),
            ]
        )
        # Depleted at both surfaces, full at mid-height: the area mean is 0.5.
        values = 1.0 - np.abs(2.0 * v / height - 1.0)

        _edges, means = strip_averages_from_nodes(
            h, values, n_strips=4, v_um=v, v_range=(0.0, height)
        )

        np.testing.assert_allclose(means, 0.5, rtol=0.02)

    def test_a_surface_row_a_rounding_above_the_band_is_part_of_it(self):
        # A charge mesh hands the silicon's top surface over at
        # 0.22000000000000003: the row where the depletion peaks once the
        # oxide is in the solve, and not one to lose to a comparison.
        height = 0.22
        rows = np.array([0.0, 0.11, height + 3e-17 + np.spacing(height)])
        columns = np.linspace(0.0, 1.0, 11)
        h = np.tile(columns, rows.size)
        v = np.repeat(rows, columns.size)
        values = np.where(v > 0.2, 0.0, 1.0)  # depleted at the top surface

        _edges, means = strip_averages_from_nodes(
            h, values, n_strips=2, v_um=v, v_range=(0.0, height)
        )

        # Full to mid-height, falling to zero at the top: 3/4.
        np.testing.assert_allclose(means, 0.75, rtol=1e-6)

    def test_a_zero_column_tolerance_groups_exact_coordinates(self):
        columns = np.linspace(0.0, 1.0, 11)
        h = np.tile(columns, 3)
        v = np.repeat([0.0, 0.11, 0.22], columns.size)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a division by the tolerance
            _edges, means = strip_averages_from_nodes(
                h, 2.0 * h, n_strips=1, v_um=v, v_range=(0.0, 0.22), column_tol_um=0.0
            )

        assert means[0] == pytest.approx(1.0)

    def test_band_without_nodes_raises(self):
        h = np.linspace(0.0, 1.0, 11)
        with pytest.raises(ValueError, match="v_range"):
            strip_averages_from_nodes(
                h, h, n_strips=1, v_um=np.zeros(11), v_range=(5.0, 6.0)
            )

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            strip_averages_from_nodes([0.0, 1.0], [1.0], n_strips=1)


def _carriers(edges, n_of_h, p_of_h, *, samples=201):
    """A one-row Carrier map across ``edges``, in the mesh frame."""
    h = np.linspace(edges[0], edges[-1], samples)
    return SimpleNamespace(
        x_um=h, y_um=np.full(h.size, 0.11), electrons_cm3=n_of_h(h), holes_cm3=p_of_h(h)
    )


def _build(carriers, n_strips, **kwargs):
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    params = dict(
        n_strips=n_strips,
        junction=(float(carriers.x_um[0]), float(carriers.x_um[-1])),
        zmin=0.0,
        zmax=0.22,
        electrodes=None,
        response=fake_coupling,
        material=RFStripMaterial(),
        drawing=StaircaseDrawing(base_layer=(40, 0)),
    )
    params.update(kwargs)
    return build_staircase_cross_section(carriers, **params)


def _material(stack, name):
    from gsim.common.stack.materials import MaterialProperties

    props = stack.materials[stack.layers[name].material]
    return (
        props
        if isinstance(props, MaterialProperties)
        else MaterialProperties.model_validate(props)
    )


class TestDrawnStripsRF:
    def test_single_strip_recovers_uniform_model(self):
        carriers = _carriers(
            [-0.2, 0.2], lambda h: np.full(h.size, 1e18), np.zeros_like
        )
        staircase = _build(carriers, 1)
        stack = staircase.stack()

        # One region spanning the window with the uniform-model conductivity.
        assert staircase.strip_names == ["strip_0"]
        spec = stack.layers["strip_0"]
        assert spec.gds_layer == (40, 0)
        assert spec.zmin == 0.0
        assert spec.zmax == pytest.approx(0.22)
        expected_sigma = float(fake_coupling(1e18, 0.0).conductivity_s_per_m)
        assert staircase.strips.conductivity_s_per_m[0] == pytest.approx(expected_sigma)
        assert _material(stack, "strip_0").conductivity == pytest.approx(expected_sigma)

    def test_arbitrary_strip_count_layers_and_materials(self):
        n = 7
        edges = np.linspace(-0.35, 0.35, n + 1)
        rise = lambda h: 1e18 * (h - edges[0]) / (edges[-1] - edges[0])  # noqa: E731
        fall = lambda h: 1e18 - rise(h)  # noqa: E731
        staircase = _build(_carriers(edges, rise, fall), n)
        stack = staircase.stack()

        assert len(staircase.strip_names) == n
        centres = 0.5 * (edges[1:] + edges[:-1])
        for i, name in enumerate(staircase.strip_names):
            assert stack.layers[name].gds_layer == (40, i)
            assert stack.layers[name].material in stack.materials
        # Linear profiles: each strip average is the value at its centre.
        np.testing.assert_allclose(
            staircase.strips.conductivity_s_per_m,
            fake_coupling(rise(centres), fall(centres)).conductivity_s_per_m,
            rtol=1e-6,
        )

    def test_rejects_a_descending_extent(self):
        carriers = _carriers(
            [-0.2, 0.2], lambda h: np.full(h.size, 1e18), np.zeros_like
        )
        with pytest.raises(ValueError, match="ascending"):
            _build(carriers, 1, junction=(0.2, -0.2))


class TestDrawnStripsOptical:
    def test_perturbed_permittivity_and_loss(self):
        n0 = 3.4757
        carriers = _carriers(
            [-0.1, 0.1],
            lambda h: np.full(h.size, 1e18),
            lambda h: np.full(h.size, 1e18),
        )
        staircase = _build(
            carriers, 1, material=OpticalStripMaterial(wavelength_um=1.55, index=n0)
        )
        material = _material(staircase.stack(), "strip_0")
        coupled = fake_coupling(1e18, 1e18)
        dn = float(coupled.index_shift)
        # Carrier-depressed index: eps_re < n0^2, loss tangent positive.
        assert material.permittivity == pytest.approx((n0 + dn) ** 2, rel=1e-3)
        assert material.permittivity < n0**2
        assert material.loss_tangent > 0.0
        assert float(coupled.absorption_cm) > 0.0
        np.testing.assert_allclose(staircase.strips.index_shift, [dn])
