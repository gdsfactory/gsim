"""Pinned, cached retrieval of open-PDK validation data.

Two sources back the FDTD-vs-PDK validation cases, and neither ships in the
published wheels:

* **UBC measured spectra** (MZI, ring, Bragg) live in the ``gdsfactory/ubc``
  git tree under ``ubcpdk/samples/data``.  The wheel and sdist both omit them
  because ``ubcpdk/samples`` is not a Python package and ``*.mat`` is not
  declared in the distribution's ``package-data``.
* **SiEPIC ``opics_ebeam`` measured coupler S-parameters** live in the
  ``SiEPIC/SiEPIC_EBeam_PDK`` git tree; the loader is a KLayout pymacro that
  is not published on PyPI.

Both are fetched on demand from a *pinned* git commit and cached on disk, so
test runs are reproducible and offline-friendly after the first download.

Environment variables
---------------------
``GSIM_PDK_DATA``
    Override the cache root (default: ``$XDG_CACHE_HOME/gsim/pdk-data``).
``GSIM_PDK_OFFLINE``
    When set, never touch the network; a cache miss raises :class:`FetchError`.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Pinned source revisions ---------------------------------------------------

UBC_REPO = "gdsfactory/ubc"
UBC_COMMIT = "23a845acbcb03fccad77f96fdb85a0aa27306601"  # ubcpdk v3.3.4
SIEPIC_REPO = "SiEPIC/SiEPIC_EBeam_PDK"
SIEPIC_COMMIT = "eac593d3deb87432a11d6b2bd5c5902b23a33ee5"  # v0.4.53

# Canonical files used by the validation cases ------------------------------

UBC_MZI_MAT = "ubcpdk/samples/data/mzi/ZiheGao_MZI3_270_Scan1.mat"
UBC_RING_MAT = "ubcpdk/samples/data/ring/LukasC_RingDoubleTER10g100_1495.mat"
UBC_BRAGG_MAT = "ubcpdk/samples/data/bragg/ELEC_413_lukasc_BraggSet1Num10_1272.mat"
SIEPIC_COUPLER_DIR = "Lumerical_EBeam_CML/EBeam/source_data/ebeam_dc_halfring_straight"

_RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
_TIMEOUT_SECONDS = 60.0
_COUPLER_PATTERN = re.compile(
    r"te_ebeam_dc_halfring_straight_gap=(?P<gap>\d+)nm_"
    r"radius=(?P<radius>\d+)um_width=(?P<width>\d+)nm_"
    r"thickness=(?P<thickness>\d+)nm_CoupleLength=(?P<length>[\d.]+)um\.dat"
)


class FetchError(RuntimeError):
    """Raised when a pinned PDK data file cannot be retrieved or parsed."""


def data_root() -> Path:
    """Return the on-disk cache root, honouring ``GSIM_PDK_DATA``."""
    override = os.environ.get("GSIM_PDK_DATA")
    if override:
        root = Path(override)
    else:
        base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
        root = Path(base) / "gsim" / "pdk-data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_path(repo: str, commit: str, path: str) -> Path:
    """Map a pinned source to a short, Windows-compatible cache filename."""
    source = f"{repo}\0{commit}\0{path}"
    digest = hashlib.sha256(source.encode()).hexdigest()[:24]
    filename = Path(path).name
    suffix = Path(filename).suffix
    stem = filename[: -len(suffix)] if suffix else filename
    return data_root() / f"{stem[:32]}-{digest}{suffix}"


def _legacy_cache_path(repo: str, commit: str, path: str) -> Path:
    owner, name = repo.split("/", 1)
    return data_root() / f"{owner}-{name}@{commit}" / path


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return _sha256_bytes(handle.read())


def fetch(
    repo: str,
    commit: str,
    path: str,
    *,
    sha256: str | None = None,
    force: bool = False,
    timeout: float = _TIMEOUT_SECONDS,
) -> Path:
    """Download ``path`` from a pinned ``repo``/``commit`` and cache it.

    A cached copy is reused unless ``force`` is set.  When ``sha256`` is given
    the digest is verified (for cached and freshly downloaded files).  With
    ``GSIM_PDK_OFFLINE`` set, a cache miss raises :class:`FetchError` instead
    of touching the network.
    """
    destination = _cache_path(repo, commit, path)
    if (
        destination.is_file()
        and not force
        and (sha256 is None or _sha256_file(destination) == sha256)
    ):
        return destination

    if not force:
        legacy = _legacy_cache_path(repo, commit, path)
        # Older versions used the full source path, which may exceed MAX_PATH.
        if (
            (os.name != "nt" or len(str(legacy)) < 240)
            and legacy.is_file()
            and (sha256 is None or _sha256_file(legacy) == sha256)
        ):
            shutil.copyfile(legacy, destination)
            return destination

    url = _RAW_URL.format(repo=repo, commit=commit, path=path)
    if os.environ.get("GSIM_PDK_OFFLINE"):
        raise FetchError(
            f"{path!r} is not cached at {destination} and GSIM_PDK_OFFLINE "
            f"is set; unset it or pre-populate the cache from {url}"
        )

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise FetchError(f"could not download {url}: {error}") from error

    if sha256 is not None:
        actual = _sha256_bytes(payload)
        if actual != sha256:
            raise FetchError(
                f"checksum mismatch for {path!r}: expected {sha256}, got {actual}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination


def fetch_ubc(path: str, **kwargs: Any) -> Path:
    """Fetch a file from the pinned UBC SiEPIC EBeam data tree (gap 1)."""
    return fetch(UBC_REPO, UBC_COMMIT, path, **kwargs)


def coupler_filename(
    *,
    gap_nm: int = 100,
    radius_um: int = 10,
    width_nm: int = 500,
    thickness_nm: int = 220,
    couple_length_um: float = 0,
) -> str:
    """Return the SiEPIC coupler ``.dat`` filename for a geometry."""
    return (
        f"te_ebeam_dc_halfring_straight_gap={gap_nm}nm_radius={radius_um}um_"
        f"width={width_nm}nm_thickness={thickness_nm}nm_"
        f"CoupleLength={couple_length_um:g}um.dat"
    )


def fetch_siepic_coupler(
    *,
    gap_nm: int = 100,
    radius_um: int = 10,
    width_nm: int = 500,
    thickness_nm: int = 220,
    couple_length_um: float = 0,
    **kwargs: Any,
) -> Path:
    """Fetch a measured SiEPIC coupler ``.dat`` (gap 2)."""
    filename = coupler_filename(
        gap_nm=gap_nm,
        radius_um=radius_um,
        width_nm=width_nm,
        thickness_nm=thickness_nm,
        couple_length_um=couple_length_um,
    )
    return fetch(
        SIEPIC_REPO, SIEPIC_COMMIT, f"{SIEPIC_COUPLER_DIR}/{filename}", **kwargs
    )


def load_ubc_spectrum(path: str | os.PathLike[str], port: int = 0) -> tuple[Any, Any]:
    """Load a UBC measured spectrum as ``(wavelength_nm, power_dbm)``.

    Handles the three on-disk layouts used by ``ubcpdk.data.read_mat``
    (``scanResults``, ``scandata`` and ``wavelength``) and normalizes the
    wavelength axis, which is stored in metres or nanometres depending on the
    acquisition tool.
    """
    import numpy as np
    from scipy.io import loadmat

    data = loadmat(path)
    if "scanResults" in data:
        rows = data["scanResults"][0][port][0]
        wavelength, power = rows[:, 0], rows[:, 1]
    elif "scandata" in data:
        wavelength = data["scandata"][0][0][0][:][0]
        power = data["scandata"][0][0][1][:, port]
    elif "wavelength" in data:
        wavelength = data["wavelength"][0][:]
        power = data["power"][:, port][:]
    else:
        raise FetchError(f"{path!r} has no recognised spectrum arrays")

    wavelength_nm = np.asarray(wavelength, dtype=float).ravel()
    if np.nanmax(np.abs(wavelength_nm)) < 1e-2:  # stored in metres
        wavelength_nm = wavelength_nm * 1e9
    return wavelength_nm, np.asarray(power, dtype=float).ravel()


def load_siepic_coupler(path: str | os.PathLike[str]) -> dict[tuple[int, int], Any]:
    """Parse a SiEPIC coupler ``.dat`` into an S-matrix dictionary.

    Returns ``{(receive_port, source_port): array}`` where each array holds
    the ``(frequency_hz, |S|, phase)`` columns.
    """
    import numpy as np

    header = re.compile(
        r'\("port (?P<receive>\d+)","mode \d+",\d+,"port (?P<source>\d+)",'
        r'\d+,"transmission"\)'
    )
    blocks: dict[tuple[int, int], Any] = {}
    key: tuple[int, int] | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        if key is not None and rows:
            blocks[key] = np.asarray(rows, dtype=float)

    for line in Path(path).read_text(encoding="utf8", errors="ignore").splitlines():
        match = header.match(line.strip())
        if match:
            flush()
            key = (int(match.group("receive")), int(match.group("source")))
            rows = []
            continue
        parts = line.split()
        if key is None or len(parts) != 3:
            continue
        try:
            rows.append([float(part) for part in parts])
        except ValueError:
            continue
    flush()

    if not blocks:
        raise FetchError(f"{path!r} contains no coupler S-parameter blocks")
    return blocks


__all__ = [
    "SIEPIC_COMMIT",
    "SIEPIC_COUPLER_DIR",
    "SIEPIC_REPO",
    "UBC_BRAGG_MAT",
    "UBC_COMMIT",
    "UBC_MZI_MAT",
    "UBC_REPO",
    "UBC_RING_MAT",
    "FetchError",
    "coupler_filename",
    "data_root",
    "fetch",
    "fetch_siepic_coupler",
    "fetch_ubc",
    "load_siepic_coupler",
    "load_ubc_spectrum",
]
