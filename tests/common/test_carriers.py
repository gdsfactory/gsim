"""Tests for the carrier-to-material coupling layer (gsim.common.carriers).

Reference values come from the published plasma-dispersion fits:

- R. Soref and B. Bennett, "Electrooptical effects in silicon," IEEE JQE 23,
  123-129 (1987): linearized fits at 1.55 um.
- M. Nedeljkovic, R. Soref, G. Z. Mashanovich, "Free-Carrier Electrorefraction
  and Electroabsorption Modulation Predictions for Silicon Over the
  1-14 um Infrared Wavelength Range," IEEE Photonics J. 3, 1171-1180
  (2011): power-law fits at 1.3 / 1.55 um.

Conductivity checks are hand calculations of sigma = q (mu_n n + mu_p p).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import elementary_charge as Q  # noqa: N812

from gsim.common.carriers import (
    MobilityModel,
    PlasmaDispersionModel,
    carrier_absorption_cm,
    carrier_conductivity,
    carrier_index_shift,
    permittivity_perturbation,
)


class TestPlasmaDispersionPresets:
    def test_nedeljkovic_1550_index_shift(self):
        model = PlasmaDispersionModel.nedeljkovic_1550()
        # Delta n = -(5.4e-22 N^1.011 + 1.53e-18 P^0.838)
        n = 1e18
        p = 1e18
        expected = -(5.4e-22 * n**1.011 + 1.53e-18 * p**0.838)
        assert carrier_index_shift(n, p, model=model) == pytest.approx(
            expected, rel=1e-12
        )
        # Order of magnitude sanity: ~ -3e-3 at 1e18/1e18
        assert -6e-3 < expected < -1e-3

    def test_nedeljkovic_1550_absorption(self):
        model = PlasmaDispersionModel.nedeljkovic_1550()
        # Delta alpha = 8.88e-21 N^1.167 + 5.84e-20 P^1.109 [cm^-1]
        n = 1e17
        p = 5e17
        expected = 8.88e-21 * n**1.167 + 5.84e-20 * p**1.109
        assert carrier_absorption_cm(n, p, model=model) == pytest.approx(
            expected, rel=1e-12
        )

    def test_nedeljkovic_1310_differs_from_1550(self):
        m13 = PlasmaDispersionModel.nedeljkovic_1310()
        m15 = PlasmaDispersionModel.nedeljkovic_1550()
        assert carrier_index_shift(1e18, 1e18, model=m13) != carrier_index_shift(
            1e18, 1e18, model=m15
        )
        assert m13.wavelength_um == pytest.approx(1.31)
        assert m15.wavelength_um == pytest.approx(1.55)

    def test_soref_1550_linear_electron_term(self):
        model = PlasmaDispersionModel.soref_1550()
        # Soref-Bennett 1987: Delta n = -(8.8e-22 N + 8.5e-18 P^0.8)
        n = 2e18
        expected = -(8.8e-22 * n)
        assert carrier_index_shift(n, 0.0, model=model) == pytest.approx(
            expected, rel=1e-12
        )

    def test_custom_coefficients_override(self):
        model = PlasmaDispersionModel(
            wavelength_um=1.55,
            dn_electron_coeff=1e-21,
            dn_electron_exp=1.0,
            dn_hole_coeff=0.0,
            dn_hole_exp=1.0,
            dalpha_electron_coeff=0.0,
            dalpha_electron_exp=1.0,
            dalpha_hole_coeff=0.0,
            dalpha_hole_exp=1.0,
        )
        assert carrier_index_shift(1e18, 1e18, model=model) == pytest.approx(-1e-3)

    def test_array_input(self):
        model = PlasmaDispersionModel.nedeljkovic_1550()
        n = np.array([1e16, 1e17, 1e18])
        p = np.zeros(3)
        dn = carrier_index_shift(n, p, model=model)
        assert dn.shape == (3,)
        # Monotonically more negative with concentration.
        assert dn[0] > dn[1] > dn[2]

    def test_zero_carriers_zero_shift(self):
        model = PlasmaDispersionModel.nedeljkovic_1550()
        assert carrier_index_shift(0.0, 0.0, model=model) == 0.0
        assert carrier_absorption_cm(0.0, 0.0, model=model) == 0.0

    def test_negative_carriers_rejected(self):
        model = PlasmaDispersionModel.nedeljkovic_1550()
        with pytest.raises(ValueError):
            carrier_index_shift(-1e17, 0.0, model=model)


class TestCarrierConductivity:
    def test_hand_calculation(self):
        # sigma [S/m] = q * (mu_n n + mu_p p), mu in cm^2/Vs, n in cm^-3:
        # sigma [S/cm] = q mu n -> *100 for S/m.
        sigma = carrier_conductivity(1e18, 0.0, mu_n_cm2=1000.0, mu_p_cm2=400.0)
        expected = Q * 1000.0 * 1e18 * 100.0
        assert sigma == pytest.approx(expected, rel=1e-12)

    def test_holes_and_electrons_add(self):
        both = carrier_conductivity(1e17, 1e17, mu_n_cm2=1400.0, mu_p_cm2=450.0)
        only_n = carrier_conductivity(1e17, 0.0, mu_n_cm2=1400.0, mu_p_cm2=450.0)
        only_p = carrier_conductivity(0.0, 1e17, mu_n_cm2=1400.0, mu_p_cm2=450.0)
        assert both == pytest.approx(only_n + only_p, rel=1e-12)

    def test_default_silicon_mobilities(self):
        # Defaults are low-field silicon values (1417 / 470.5 cm^2/Vs).
        sigma = carrier_conductivity(1e18, 0.0)
        expected = Q * 1417.0 * 1e18 * 100.0
        assert sigma == pytest.approx(expected, rel=1e-12)

    def test_array_input(self):
        n = np.array([1e16, 1e18])
        sigma = carrier_conductivity(n, np.zeros(2))
        assert sigma.shape == (2,)
        assert sigma[1] == pytest.approx(sigma[0] * 100.0, rel=1e-9)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            carrier_conductivity(-1.0, 0.0)


class TestPermittivityPerturbation:
    def test_real_part_from_index_shift(self):
        # eps = (n0 + dn - i k)^2; for dalpha=0: d(eps_re) ~ 2 n0 dn + dn^2.
        eps = permittivity_perturbation(
            n0=3.48, dn=-1e-3, dalpha_cm=0.0, wavelength_um=1.55
        )
        expected = (3.48 - 1e-3) ** 2
        assert eps.real == pytest.approx(expected, rel=1e-12)
        assert eps.imag == 0.0

    def test_absorption_gives_negative_imag(self):
        # kappa = alpha * lambda / (4 pi), alpha in m^-1.
        alpha_cm = 10.0
        kappa = (alpha_cm * 100.0) * 1.55e-6 / (4.0 * np.pi)
        eps = permittivity_perturbation(
            n0=3.48, dn=0.0, dalpha_cm=alpha_cm, wavelength_um=1.55
        )
        expected = (3.48 - 1j * kappa) ** 2
        assert eps.real == pytest.approx(expected.real, rel=1e-12)
        assert eps.imag == pytest.approx(expected.imag, rel=1e-12)
        assert eps.imag < 0.0


class TestMobilityModel:
    def test_undoped_silicon_keeps_the_lattice_mobility(self):
        model = MobilityModel.masetti_silicon()
        assert model.electrons_cm2(0.0) == pytest.approx(1417.0)
        assert model.holes_cm2(0.0) == pytest.approx(470.5)

    def test_published_values_are_reproduced(self):
        # Masetti et al. (1983): hand evaluation of the fit at 1e20 cm^-3.
        model = MobilityModel.masetti_silicon()
        assert model.electrons_cm2(1e20) == pytest.approx(60.9, abs=0.1)
        assert model.holes_cm2(1e20) == pytest.approx(49.9, abs=0.1)

    def test_mobility_falls_with_doping(self):
        model = MobilityModel.masetti_silicon()
        doping = np.logspace(14, 20.5, 40)
        assert np.all(np.diff(model.electrons_cm2(doping)) < 0)
        assert np.all(np.diff(model.holes_cm2(doping)) < 0)

    def test_constant_ignores_the_doping(self):
        model = MobilityModel.constant(mu_n_cm2=100.0, mu_p_cm2=50.0)
        doping = np.array([0.0, 1e15, 1e20])
        np.testing.assert_allclose(model.electrons_cm2(doping), 100.0)
        np.testing.assert_allclose(model.holes_cm2(doping), 50.0)

    def test_negative_impurity_rejected(self):
        with pytest.raises(ValueError):
            MobilityModel.masetti_silicon().electrons_cm2(-1.0)

    def test_the_model_reaches_the_conductivity(self):
        # A neutral n Region: its electrons number its donors.
        model = MobilityModel.masetti_silicon()
        sigma = carrier_conductivity(1e20, 0.0, mobility=model)
        expected = Q * float(model.electrons_cm2(1e20)) * 1e20 * 100.0
        assert sigma == pytest.approx(expected, rel=1e-12)
        # A contact Region conducts far less than the lattice mobility says.
        assert sigma < 0.1 * carrier_conductivity(1e20, 0.0)

    def test_a_constant_model_is_the_two_constants(self):
        model = MobilityModel.constant(mu_n_cm2=1000.0, mu_p_cm2=400.0)
        n, p = np.array([1e16, 1e18]), np.array([1e17, 0.0])
        np.testing.assert_allclose(
            carrier_conductivity(n, p, mobility=model),
            carrier_conductivity(n, p, mu_n_cm2=1000.0, mu_p_cm2=400.0),
        )
