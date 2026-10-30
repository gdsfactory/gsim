"""Read Cornerstone (``cspdk``) coupling LUTs without importing ``cspdk``.

Importing the ``cspdk`` package pulls in its PDK cells, which currently fail
against the gdsfactory release pinned by gsim (they call
``gdsfactory.add_pins.add_electrical_pins``, removed in the 9.4x series).  The
coupling LUTs are plain netCDF files, so this module opens them directly with
xarray: first from an installed ``cspdk`` distribution if present, otherwise
from the pinned ``gdsfactory/cspdk`` revision.  This keeps the Cornerstone
data available to the validation suite regardless of the gdsfactory version.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import xarray as xr

from tests.pdk_validation import _fetch

CSPDK_REPO = "gdsfactory/cspdk"
CSPDK_COMMIT = "09ca1a1e23c3a00f4f51e89ef2fa83b3f352ccbb"  # v2.1.0

# LUT filenames shipped under ``cspdk/<band>/models``.
DIRECTIONAL_COUPLER_STRIP = "directional_coupler_strip"
DIRECTIONAL_COUPLER_RIB = "directional_coupler_rib"
RACETRACK_STRIP = "coupler_racetrack_strip"
RACETRACK_RIB = "coupler_racetrack_rib"


class CSPDKError(RuntimeError):
    """Raised when a Cornerstone LUT cannot be located or opened."""


def _installed_lut(relative_path: str) -> Path | None:
    """Return a LUT path inside an installed but unimported ``cspdk``."""
    spec = importlib.util.find_spec("cspdk")
    if spec is None or not spec.submodule_search_locations:
        return None
    site_packages = Path(next(iter(spec.submodule_search_locations))).parent
    candidate = site_packages / relative_path
    return candidate if candidate.is_file() else None


def lut_path(*, band: str = "si220", model: str = DIRECTIONAL_COUPLER_STRIP) -> Path:
    """Locate a ``cspdk`` coupling LUT, installed or pinned-remote."""
    relative = f"cspdk/{band}/models/{model}.nc"
    local = _installed_lut(relative)
    if local is not None:
        return local
    return _fetch.fetch(CSPDK_REPO, CSPDK_COMMIT, relative)


def load_lut(*, band: str = "si220", model: str = DIRECTIONAL_COUPLER_STRIP) -> Any:
    """Open and load a ``cspdk`` coupling LUT as a DataArray."""
    path = lut_path(band=band, model=model)
    try:
        return xr.open_dataarray(path).load()
    except ValueError as error:
        raise CSPDKError(
            f"could not open {path}; install 'h5netcdf' (or 'netCDF4') to read "
            "the Cornerstone netCDF4 LUTs"
        ) from error


def interpolate_kappa(lut: Any, **coords: Any) -> Any:
    """Interpolate the coupling amplitude ``kappa`` at the given coordinates.

    Mirrors the SAX lookup used by the PDK models; the coupling amplitude is
    returned so ``tau = sqrt(1 - kappa**2)``.
    """
    import sax

    expanded = lut if "kappa" in lut.dims else lut.expand_dims({"kappa": ["kappa"]}, -1)
    return sax.interpolate_xarray(expanded, **coords)["kappa"]


__all__ = [
    "CSPDK_COMMIT",
    "CSPDK_REPO",
    "DIRECTIONAL_COUPLER_RIB",
    "DIRECTIONAL_COUPLER_STRIP",
    "RACETRACK_RIB",
    "RACETRACK_STRIP",
    "CSPDKError",
    "interpolate_kappa",
    "load_lut",
    "lut_path",
]
