"""Fixtures for the open-PDK FDTD validation cases.

Reference data is fetched through :mod:`tests.pdk_validation._fetch` from a
pinned git revision and cached (see that module for the ``GSIM_PDK_DATA`` and
``GSIM_PDK_OFFLINE`` controls).  Tests are guarded by the ``pdk_data`` marker
and skip cleanly when the data cannot be retrieved.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.pdk_validation import _fetch

__all__ = [
    "cspdk_dc_lut",
    "cspdk_racetrack_lut",
    "siepic_coupler",
    "ubc_bragg_spectrum",
    "ubc_mzi_spectrum",
    "ubc_ring_spectrum",
]


def _load_ubc(relative_path: str) -> tuple[Any, Any]:
    """Fetch and load a UBC measured spectrum, skipping when unavailable."""
    try:
        path = _fetch.fetch_ubc(relative_path)
    except _fetch.FetchError as error:
        pytest.skip(f"UBC reference data unavailable: {error}")
    return _fetch.load_ubc_spectrum(path)


@pytest.fixture(scope="session")
def ubc_ring_spectrum() -> tuple[Any, Any]:
    """Measured UBC TE double-ring spectrum ``(wavelength_nm, power_dbm)``."""
    return _load_ubc(_fetch.UBC_RING_MAT)


@pytest.fixture(scope="session")
def ubc_mzi_spectrum() -> tuple[Any, Any]:
    """Measured UBC MZI spectrum ``(wavelength_nm, power_dbm)``."""
    return _load_ubc(_fetch.UBC_MZI_MAT)


@pytest.fixture(scope="session")
def ubc_bragg_spectrum() -> tuple[Any, Any]:
    """Measured UBC Bragg-grating spectrum ``(wavelength_nm, power_dbm)``."""
    return _load_ubc(_fetch.UBC_BRAGG_MAT)


@pytest.fixture(scope="session")
def siepic_coupler() -> dict[tuple[int, int], Any]:
    """Measured SiEPIC ``opics_ebeam`` coupler S-matrix ``(freq, |S|, phase)``."""
    try:
        path = _fetch.fetch_siepic_coupler()
    except _fetch.FetchError as error:
        pytest.skip(f"SiEPIC coupler reference data unavailable: {error}")
    return _fetch.load_siepic_coupler(path)


@pytest.fixture(scope="session")
def cspdk_dc_lut() -> Any:
    """Cornerstone Si220 strip directional-coupler coupling LUT."""
    from tests.pdk_validation import _cspdk

    try:
        return _cspdk.load_lut(model=_cspdk.DIRECTIONAL_COUPLER_STRIP)
    except (_fetch.FetchError, _cspdk.CSPDKError) as error:
        pytest.skip(f"cspdk directional-coupler LUT unavailable: {error}")


@pytest.fixture(scope="session")
def cspdk_racetrack_lut() -> Any:
    """Cornerstone Si220 strip racetrack coupling LUT."""
    from tests.pdk_validation import _cspdk

    try:
        return _cspdk.load_lut(model=_cspdk.RACETRACK_STRIP)
    except (_fetch.FetchError, _cspdk.CSPDKError) as error:
        pytest.skip(f"cspdk racetrack LUT unavailable: {error}")
