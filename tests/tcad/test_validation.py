"""Hermetic tests for the TCAD vs analytic depletion-model comparison."""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.stack.pn_junction import PNJunctionConfig
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap
from gsim.tcad.validation import (
    analytic_capacitance_f_per_cm,
    compare_capacitance,
    estimate_depletion_width_um,
)


def _empty_carriers():
    zeros = np.zeros(1)
    return CarrierMap(
        x_um=zeros,
        y_um=zeros,
        region=["r"],
        electrons_cm3=zeros,
        holes_cm3=zeros,
    )


def _synthetic_sweep(voltages, capacitances):
    points = [
        BiasPoint(bias_v=v, carriers=_empty_carriers(), capacitance_f_per_cm=c)
        for v, c in zip(voltages, capacitances, strict=True)
    ]
    return BiasSweepResult(contact="cathode", points=points)


class TestAnalyticCapacitance:
    def test_matches_parallel_plate_hand_value(self):
        junction = PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18)
        height_um = 0.22
        [c] = analytic_capacitance_f_per_cm(junction, 0.0, height_um=height_um)
        # C per cm depth = (eps_s / W) * (height in m) * (1 cm in m).
        expected = junction.c_per_area * height_um * 1e-6 * 1e-2
        assert c == pytest.approx(expected, rel=1e-12)

    def test_decreases_with_reverse_bias(self):
        junction = PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18)
        c = analytic_capacitance_f_per_cm(junction, [0.0, 1.0, 2.0], height_um=0.22)
        assert c[2] < c[1] < c[0]

    def test_rejects_nonpositive_height(self):
        junction = PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18)
        with pytest.raises(ValueError, match="height_um"):
            analytic_capacitance_f_per_cm(junction, 0.0, height_um=0.0)


class TestCompareCapacitance:
    def test_reports_both_curves_and_deviation(self):
        junction = PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18)
        voltages = [0.0, 0.5, 1.0]
        analytic = analytic_capacitance_f_per_cm(junction, voltages, height_um=0.22)
        sweep = _synthetic_sweep(voltages, analytic * 1.05)

        comparison = compare_capacitance(junction, sweep, height_um=0.22)
        np.testing.assert_allclose(comparison.v_reverse, voltages)
        np.testing.assert_allclose(comparison.c_analytic_f_per_cm, analytic)
        np.testing.assert_allclose(comparison.c_tcad_f_per_cm, analytic * 1.05)
        assert comparison.max_relative_deviation == pytest.approx(0.05, rel=1e-6)
        assert comparison.within(0.06)
        assert not comparison.within(0.04)

    def test_anode_sweep_sign_convention(self):
        junction = PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18)
        # Anode swept negative = reverse bias.
        voltages = [0.0, -1.0]
        analytic = analytic_capacitance_f_per_cm(junction, [0.0, 1.0], height_um=0.22)
        sweep = _synthetic_sweep(voltages, analytic)
        comparison = compare_capacitance(
            junction, sweep, height_um=0.22, reverse_bias_sign=-1.0
        )
        np.testing.assert_allclose(comparison.v_reverse, [0.0, 1.0])
        assert comparison.within(1e-9)


class TestDepletionWidthEstimate:
    def test_recovers_synthetic_depleted_span(self):
        pos = np.linspace(-1.0, 1.0, 2001)
        na = nd = 1e18
        # Depleted strip |x| < 0.15 um: both carriers collapse there.
        n = np.where(pos > 0.15, nd, 1e5)
        p = np.where(pos < -0.15, na, 1e5)
        width = estimate_depletion_width_um(pos, n, p, na_cm3=na, nd_cm3=nd)
        assert width == pytest.approx(0.3, abs=0.005)

    def test_no_depletion_returns_zero(self):
        pos = np.linspace(-1.0, 1.0, 11)
        n = np.full_like(pos, 1e18)
        p = np.full_like(pos, 1e18)
        assert estimate_depletion_width_um(pos, n, p, na_cm3=1e18, nd_cm3=1e18) == 0.0

    def test_rejects_bad_fraction(self):
        pos = np.zeros(3)
        with pytest.raises(ValueError, match="fraction"):
            estimate_depletion_width_um(
                pos, pos, pos, na_cm3=1e18, nd_cm3=1e18, fraction=1.5
            )
