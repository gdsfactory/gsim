"""Physical input limits and optional geometry controls for PN junctions."""

from __future__ import annotations

import gdsfactory as gf
import numpy as np
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


def test_config_selects_mode_at_width_threshold():
    junction = pn.PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18)
    assert junction.select_mode(4 * junction.w_um, 10 * junction.w_um) == "high_res"
    assert junction.select_mode(10 * junction.w_um, 20 * junction.w_um) == "capacitance"
    assert (
        junction.select_mode(2 * junction.w_um, 4 * junction.w_um, fraction=0.5)
        == "high_res"
    )


def test_unsnapped_doping_regions_remain_contiguous_on_both_sides():
    component = gf.Component()
    result = pn.make_doping_profile(
        component,
        length=2,
        rib_center_y=1,
        rib_width=0.4,
        profile={"upper": [(0.1, 100), (0.2, 200)], "lower": [(0.1, 100), (0.2, 200)]},
        sides={
            "upper": {"base_layer": (21, 1), "name_prefix": "p", "sign": 1},
            "lower": {"base_layer": (20, 1), "name_prefix": "n", "sign": -1},
        },
        zmin=0,
        zmax=0.22,
        snap_grid_um=None,
    )
    assert result["centres"]["upper"] == pytest.approx([1.25, 1.4])
    assert result["centres"]["lower"] == pytest.approx([0.75, 0.6])
    polygons = component.get_polygons_points(by="tuple")
    for layer, expected_bounds in [
        ((21, 1), (1.2, 1.3)),
        ((21, 2), (1.3, 1.5)),
        ((20, 1), (0.7, 0.8)),
        ((20, 2), (0.5, 0.7)),
    ]:
        y = polygons[layer][0][:, 1]
        assert (y.min(), y.max()) == pytest.approx(expected_bounds)


def test_pn_profile_rejects_nonpositive_length():
    with pytest.raises(ValueError, match="length must be positive"):
        pn.make_pn_junction_profile(
            gf.Component(),
            length=0,
            center_y=0,
            rib_width=0.4,
            junction={"na_cm3": 1e18, "nd_cm3": 1e18},
            p_region=("p", (21, 0), 100),
            n_region=("n", (20, 0), 100),
        )


@pytest.mark.parametrize("wavelength", [0, -1.55])
def test_optical_conversion_rejects_nonpositive_wavelength(wavelength):
    with pytest.raises(ValueError, match="wavelength_um must be positive"):
        pn.optical_params(11.9 + 0.01j, wavelength)


def test_optical_permittivity_rejects_invalid_background():
    with pytest.raises(ValueError, match="eps_bg_rel must be >= 1"):
        pn.epsilon_eff_relative(1e18, 0, wavelength_um=1.55, eps_bg_rel=0.5)


@pytest.mark.parametrize(("electron_time", "hole_time"), [(0, 1e-13), (1e-13, -1e-13)])
def test_optical_permittivity_rejects_invalid_relaxation_times(
    electron_time, hole_time
):
    with pytest.raises(ValueError, match="Relaxation times must be positive"):
        pn.epsilon_eff_relative(
            1e18,
            1e18,
            wavelength_um=1.55,
            eps_bg_rel=11.9,
            tau_e_s=electron_time,
            tau_h_s=hole_time,
        )


def test_explicit_relaxation_times_control_plasma_shift_and_absorption():
    electron_time, hole_time = pn.drude_relaxation_times()
    default = pn.epsilon_eff_relative(1e18, 0, wavelength_um=1.55, eps_bg_rel=11.9)
    explicit = pn.epsilon_eff_relative(
        1e18,
        0,
        wavelength_um=1.55,
        eps_bg_rel=11.9,
        tau_e_s=electron_time,
        tau_h_s=hole_time,
    )
    slower = pn.epsilon_eff_relative(
        1e18,
        0,
        wavelength_um=1.55,
        eps_bg_rel=11.9,
        tau_e_s=2 * electron_time,
    )
    assert explicit == pytest.approx(default)
    assert 11.9 - slower.real == pytest.approx((11.9 - default.real) / 2)
    assert slower.imag == pytest.approx(default.imag / 4)


def test_unknown_optical_material_uses_silicon_background():
    with pytest.warns(UserWarning, match="not found in database"):
        background = pn.default_eps_bg_rel(1.55, material="unknown-test-material")
    assert background == pn.DEFAULT_SI_PERMITTIVITY


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"zmin": 0.22, "zmax": 0.22}, "zmax must exceed zmin"),
        ({"length": 0}, "length must be positive"),
        ({"rib_width": 0.001}, "does not fit"),
        ({"snap_grid_um": 0}, "snap_grid_um must be positive"),
        ({"snap_grid_um": -0.01}, "snap_grid_um must be positive"),
    ],
)
def test_segmented_profile_rejects_invalid_geometry(overrides, message):
    parameters = dict(length=2, center_y=0, rib_width=0.4, n_p=2, n_n=2)
    parameters.update(overrides)
    with pytest.raises(ValueError, match=message):
        pn.make_segmented_junction_profile(
            gf.Component(),
            junction={"na_cm3": 1e18, "nd_cm3": 1e18},
            **parameters,
        )


def test_unsnapped_segment_edges_preserve_requested_widths():
    result = pn.make_segmented_junction_profile(
        gf.Component(),
        length=2,
        center_y=0.1234,
        rib_width=0.4006,
        junction={"na_cm3": 1e18, "nd_cm3": 1e18},
        n_p=2,
        n_n=3,
        snap_grid_um=None,
    )
    segments = sorted(result["segments"].values(), key=lambda segment: segment["y0_um"])
    bounds = np.array([[segment["y0_um"], segment["y1_um"]] for segment in segments])
    assert bounds[0, 0] == pytest.approx(-0.0769)
    assert bounds[-1, 1] == pytest.approx(0.3237)
    np.testing.assert_array_equal(bounds[:-1, 1], bounds[1:, 0])
    np.testing.assert_allclose(
        np.diff(bounds, axis=1).ravel(), [0.2003 / 3] * 3 + [0.2003 / 2] * 2
    )
