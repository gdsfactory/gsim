"""Tests for the low-field mobility model (gsim.common.carriers)."""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.carriers import MobilityModel


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
