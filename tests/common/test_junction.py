"""Tests for the series-RC junction branch fit (gsim.common.junction)."""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.junction import series_rc_from_admittance
from tests._helpers import series_rc_admittance


class TestSeriesRCFromAdmittance:
    def test_recovers_the_branch_it_came_from(self):
        r_s, c_j = 8e-4, 2.4e-10  # 0.8 ohm mm, 0.24 fF/um
        y = series_rc_admittance(1e6, r_s, c_j)
        r_fit, c_fit = series_rc_from_admittance(y, freq_hz=1e6)
        assert r_fit == pytest.approx(r_s, rel=1e-12)
        assert c_fit == pytest.approx(c_j, rel=1e-12)

    def test_pure_capacitor_has_zero_resistance(self):
        c_j = 1e-10
        omega = 2 * np.pi * 1e6
        r_fit, c_fit = series_rc_from_admittance(1j * omega * c_j, freq_hz=1e6)
        assert r_fit == pytest.approx(0.0, abs=1e-15)
        assert c_fit == pytest.approx(c_j, rel=1e-12)

    def test_a_sweep_of_admittances_fits_pointwise(self):
        r_s = np.array([1e-3, 2e-3, 3e-3])
        c_j = np.array([3e-10, 2e-10, 1e-10])
        y = series_rc_admittance(1e6, r_s, c_j)
        r_fit, c_fit = series_rc_from_admittance(y, freq_hz=1e6)
        assert r_fit == pytest.approx(r_s, rel=1e-12)
        assert c_fit == pytest.approx(c_j, rel=1e-12)

    def test_rejects_an_inductive_admittance(self):
        with pytest.raises(ValueError, match="capacitive"):
            series_rc_from_admittance(1e-3 - 1e-4j, freq_hz=1e6)

    def test_rejects_a_negative_conductance(self):
        y = series_rc_admittance(1e6, 1e-3, 2e-10)
        with pytest.raises(ValueError, match="series RC"):
            series_rc_from_admittance(-y.real + 1j * y.imag, freq_hz=1e6)

    def test_rejects_a_nonpositive_frequency(self):
        with pytest.raises(ValueError, match="frequency"):
            series_rc_from_admittance(1e-3 + 1e-4j, freq_hz=0.0)
