"""Physical input limits of the Sze depletion model."""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common.stack import pn_junction as pn


@pytest.fixture(autouse=True)
def activate_pdk():
    gf.gpdk.PDK.activate()


@pytest.mark.parametrize("intrinsic_density", [0.0, -1.0])
def test_built_in_voltage_rejects_invalid_intrinsic_density(intrinsic_density):
    with pytest.raises(ValueError, match="ni_cm3 must be positive"):
        pn.built_in_voltage(1e18, 1e18, ni_cm3=intrinsic_density)


@pytest.mark.parametrize("permittivity", [0.0, 0.99])
def test_depletion_requires_physical_permittivity(permittivity):
    with pytest.raises(ValueError, match="permittivity must be >= 1"):
        pn.depletion_width(1e18, 1e18, permittivity=permittivity)


@pytest.mark.parametrize(("acceptors", "donors"), [(0, 1e18), (1e18, -1)])
def test_depletion_extents_reject_invalid_doping(acceptors, donors):
    with pytest.raises(ValueError, match="Doping concentrations must be positive"):
        pn.depletion_extents(acceptors, donors, w_um=0.1)


def test_depletion_extents_allow_zero_but_reject_negative_width():
    assert pn.depletion_extents(1e18, 2e18, w_um=0) == (0, 0)
    with pytest.raises(ValueError, match="w_um must be non-negative"):
        pn.depletion_extents(1e18, 2e18, w_um=-0.1)


@pytest.mark.parametrize("width", [0.0, -0.1])
def test_capacitance_requires_positive_depletion_width(width):
    with pytest.raises(ValueError, match="w_um must be positive"):
        pn.junction_capacitance_per_area(11.9, width)


@pytest.mark.parametrize(
    ("length", "height"), [(0, 0.22), (10, 0), (-1, 0.22), (10, -1)]
)
def test_junction_capacitance_rejects_nonpositive_area(length, height):
    junction = pn.PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18)
    with pytest.raises(ValueError, match="length_um and height_um must be positive"):
        junction.capacitance(length, height)
