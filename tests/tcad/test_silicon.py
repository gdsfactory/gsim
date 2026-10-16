"""The silicon constants the charge solve and its analytic check share."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.constants import Boltzmann, elementary_charge

from gsim.common.stack.pn_junction import PNJunctionConfig
from gsim.tcad.silicon import (
    INTRINSIC_DENSITY_300K_CM3,
    band_gap_ev,
    intrinsic_density_cm3,
)


def test_the_density_is_devsims_at_300_k():
    assert intrinsic_density_cm3(300.0) == INTRINSIC_DENSITY_300K_CM3


def test_the_band_gap_is_the_room_temperature_one():
    # Sze, table of silicon properties: 1.12 eV at 300 K.
    assert band_gap_ev(300.0) == pytest.approx(1.12, abs=0.005)
    assert band_gap_ev(350.0) < band_gap_ev(300.0) < band_gap_ev(250.0)


def test_the_density_follows_the_band_gap():
    # ln(n_i / T^1.5) against 1/T has slope -E_g / 2k, E_g taken where
    # it is evaluated: the Arrhenius form the density is built from.
    k_ev = Boltzmann / elementary_charge
    for t in (250.0, 300.0, 350.0, 400.0):
        expected = (
            INTRINSIC_DENSITY_300K_CM3
            * (t / 300.0) ** 1.5
            * math.exp(
                band_gap_ev(300.0) / (600.0 * k_ev) - band_gap_ev(t) / (2 * k_ev * t)
            )
        )
        assert intrinsic_density_cm3(t) == pytest.approx(expected, rel=1e-12)
    temperatures = np.linspace(200.0, 450.0, 11)
    assert np.all(np.diff([intrinsic_density_cm3(t) for t in temperatures]) > 0)


def test_the_built_in_voltage_falls_with_temperature():
    def v_bi(t):
        return PNJunctionConfig(
            na_cm3=1e18, nd_cm3=1e18, temperature_k=t, ni_cm3=intrinsic_density_cm3(t)
        ).v_bi

    assert v_bi(350.0) < v_bi(300.0) < v_bi(250.0)


@pytest.mark.parametrize("temperature_k", [0.0, -10.0, math.nan, math.inf])
def test_a_temperature_must_be_positive_and_finite(temperature_k):
    with pytest.raises(ValueError, match="temperature_k"):
        intrinsic_density_cm3(temperature_k)
