"""The junction branch read off the charge solve's result containers.

Hermetic: the bias points carry analytic series-RC admittances, so what
is under test is the containers' unit re-expression (per cm of DEVSIM
depth to per meter of electrode) and the fit wiring, not DEVSIM.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap
from tests._helpers import series_rc_admittance

FREQ_HZ = 1.0


def tiny_map() -> CarrierMap:
    """The smallest carrier map a bias point will hold."""
    zeros = np.zeros(1)
    return CarrierMap(
        x_um=zeros,
        y_um=zeros,
        region=["n_rib"],
        electrons_cm3=zeros,
        holes_cm3=zeros,
    )


def rc_point(bias_v: float, r_s_ohm_m: float, c_j_f_per_m: float) -> BiasPoint:
    """A bias point whose admittance is exactly a series RC per meter."""
    y_per_m = series_rc_admittance(FREQ_HZ, r_s_ohm_m, c_j_f_per_m)
    return BiasPoint(
        bias_v=bias_v,
        carriers=tiny_map(),
        admittance_s_per_cm=complex(y_per_m) / 1e2,
        admittance_freq_hz=FREQ_HZ,
    )


class TestJunctionBranch:
    def test_the_fit_recovers_the_branch_per_meter(self):
        point = rc_point(0.0, 8e-4, 2.4e-10)

        r_s, c_j = point.junction_branch()

        assert r_s == pytest.approx(8e-4, rel=1e-9)
        assert c_j == pytest.approx(2.4e-10, rel=1e-9)

    def test_the_admittance_is_reexpressed_per_meter(self):
        point = rc_point(0.0, 8e-4, 2.4e-10)

        assert point.admittance_s_per_m == pytest.approx(
            point.admittance_s_per_cm * 1e2
        )

    def test_a_point_without_an_admittance_refuses_the_fit(self):
        point = BiasPoint(bias_v=1.5, carriers=tiny_map())

        with pytest.raises(ValueError, match=r"1\.5"):
            point.junction_branch()

    def test_the_sweep_fits_every_point_in_order(self):
        sweep = BiasSweepResult(
            contact="cathode",
            points=[
                rc_point(0.0, 1e-3, 3e-10),
                rc_point(1.0, 2e-3, 2e-10),
                rc_point(2.0, 3e-3, 1e-10),
            ],
        )

        r_s, c_j = sweep.junction_branch()

        assert r_s == pytest.approx([1e-3, 2e-3, 3e-3], rel=1e-9)
        assert c_j == pytest.approx([3e-10, 2e-10, 1e-10], rel=1e-9)

    def test_the_sweep_reports_its_admittances(self):
        point = rc_point(0.0, 1e-3, 3e-10)
        sweep = BiasSweepResult(contact="cathode", points=[point])

        assert sweep.admittance_s_per_cm == pytest.approx([point.admittance_s_per_cm])
