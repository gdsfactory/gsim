"""Unit tests for the pinned open-PDK data fetcher (no network required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from tests.pdk_validation import _fetch


def test_fetch_reuses_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GSIM_PDK_DATA", str(tmp_path))
    destination = _fetch._cache_path("owner/repo", "deadbeef", "dir/file.mat")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"cached")

    assert _fetch.fetch("owner/repo", "deadbeef", "dir/file.mat") == destination


def test_fetch_offline_cache_miss_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GSIM_PDK_DATA", str(tmp_path))
    monkeypatch.setenv("GSIM_PDK_OFFLINE", "1")

    with pytest.raises(_fetch.FetchError, match="GSIM_PDK_OFFLINE"):
        _fetch.fetch("owner/repo", "deadbeef", "dir/missing.mat")


def test_offline_ignores_stale_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GSIM_PDK_DATA", str(tmp_path))
    monkeypatch.setenv("GSIM_PDK_OFFLINE", "1")
    destination = _fetch._cache_path("owner/repo", "deadbeef", "dir/file.mat")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stale")

    with pytest.raises(_fetch.FetchError, match="GSIM_PDK_OFFLINE"):
        _fetch.fetch("owner/repo", "deadbeef", "dir/file.mat", sha256="0" * 64)


def test_coupler_filename_round_trip() -> None:
    name = _fetch.coupler_filename(gap_nm=100, radius_um=10)
    assert name == (
        "te_ebeam_dc_halfring_straight_gap=100nm_radius=10um_"
        "width=500nm_thickness=220nm_CoupleLength=0um.dat"
    )


def test_load_ubc_spectrum_normalises_units(tmp_path: Path) -> None:
    path = tmp_path / "spectrum.mat"
    wavelength_m = np.linspace(1.5e-6, 1.6e-6, 11).reshape(1, -1)
    power = np.linspace(-30.0, -20.0, 11).reshape(-1, 1)
    savemat(path, {"wavelength": wavelength_m, "power": power})

    wavelength_nm, loaded_power = _fetch.load_ubc_spectrum(path)

    assert wavelength_nm.shape == (11,)
    assert loaded_power.shape == (11,)
    assert wavelength_nm[0] == pytest.approx(1500.0)
    assert wavelength_nm[-1] == pytest.approx(1600.0)


def test_load_siepic_coupler_parses_blocks(tmp_path: Path) -> None:
    path = tmp_path / "coupler.dat"
    lines = ['["port 1",""]', '["port 2",""]']
    for source in (1, 2):
        lines.append(f'("port 1","mode 1",1,"port {source}",1,"transmission")')
        lines.append("(2, 3)")
        lines.append("1.9e14 0.1 0.0")
        lines.append("2.0e14 0.2 1.0")
    path.write_text("\n".join(lines), encoding="utf8")

    blocks = _fetch.load_siepic_coupler(path)

    assert set(blocks) == {(1, 1), (1, 2)}
    assert blocks[(1, 1)].shape == (2, 3)
    assert blocks[(1, 2)][-1, 0] == pytest.approx(2.0e14)
