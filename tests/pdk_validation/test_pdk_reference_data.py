"""Reference-data smoke tests for the open-PDK validation cases.

These download pinned data (see :mod:`tests.pdk_validation._fetch`) and are
deselected by default; run with ``pytest -m pdk_data``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.pdk_validation import _validation as validation

pytestmark = pytest.mark.pdk_data


def test_ubc_spectra_are_physical(
    ubc_ring_spectrum, ubc_mzi_spectrum, ubc_bragg_spectrum
):
    for wavelength_nm, power_dbm in (
        ubc_ring_spectrum,
        ubc_mzi_spectrum,
        ubc_bragg_spectrum,
    ):
        assert wavelength_nm.size > 1000
        assert np.all(np.isfinite(wavelength_nm))
        assert 1400.0 < wavelength_nm.min() < wavelength_nm.max() < 1700.0
        assert np.isfinite(power_dbm).any()


def test_siepic_coupler_has_full_scattering_matrix(siepic_coupler):
    assert len(siepic_coupler) == 16
    for values in siepic_coupler.values():
        assert values.shape[1] == 3
    frequencies = next(iter(siepic_coupler.values()))[:, 0]
    assert frequencies.min() > 1.5e14
    assert frequencies.max() < 2.5e14


def test_cspdk_luts_open_as_dataarrays(cspdk_dc_lut, cspdk_racetrack_lut):
    assert set(cspdk_dc_lut.dims) == {
        "wavelength",
        "gap",
        "radius",
        "length_x",
        "v_offset",
    }
    assert set(cspdk_racetrack_lut.dims) == {"wavelength", "gap", "radius", "length_x"}
    assert cspdk_dc_lut.wavelength.min() >= 1.5
    assert cspdk_dc_lut.wavelength.max() <= 1.6


def test_cspdk_references_build_finite_scattering_matrices(
    cspdk_dc_lut, cspdk_racetrack_lut
):
    assert cspdk_dc_lut is not None
    assert cspdk_racetrack_lut is not None
    wavelengths = np.linspace(1.5, 1.6, 21)
    for builder in (
        validation._ref_cspdk_coupler,
        validation._ref_cspdk_ring_coupling,
    ):
        reference = builder(wavelengths)
        assert set(reference.s_params) == {"S31", "S41"}
        for values in reference.s_params.values():
            assert values.shape == wavelengths.shape
            assert np.all(np.isfinite(values))


def test_siepic_reference_parses_measured_scattering_matrix(siepic_coupler):
    assert siepic_coupler
    wavelengths = np.linspace(1.5, 1.6, 21)
    reference = validation._ref_siepic_coupler(wavelengths)
    assert set(reference.s_params) == {"S31", "S41"}
    for values in reference.s_params.values():
        assert values.shape == wavelengths.shape
        assert np.all(np.isfinite(values))
        assert np.all(np.abs(values) > 0)
