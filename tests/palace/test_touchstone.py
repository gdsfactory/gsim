"""Touchstone export/import round trips (issue #272 validation item).

Acceptance criterion: round-trip 2- and 4-port networks through Touchstone
with a maximum complex-S error below 1e-9, preserving frequency units, port
order and reference impedance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gsim.palace.results import SParam, SParams, load_sparams

TWO_PORT_DATA = (
    Path(__file__).resolve().parents[2] / "nbs/data/inductor/circuit_synthesis"
)


def _max_abs_s_error(a: SParams, b: SParams) -> float:
    assert a.port_names == b.port_names, "port order changed in round trip"
    assert np.allclose(a.freq, b.freq, rtol=0, atol=1e-12), (
        "frequency units/grid changed"
    )
    return max(
        float(np.max(np.abs(a[(i, j)].complex - b[(i, j)].complex)))
        for i in a.port_names
        for j in a.port_names
    )


def _synthetic_sparams(n_ports: int, freq_ghz: np.ndarray, seed: int) -> SParams:
    """Passive-looking synthetic N-port with distinct, well-conditioned rows."""
    rng = np.random.default_rng(seed)
    f_hz = np.asarray(freq_ghz, dtype=float) * 1e9
    tau = 40e-12
    data = {}
    for i in range(n_ports):
        for j in range(n_ports):
            mag = (0.85 if i == j else 0.07) * np.exp(-f_hz * tau * (0.1 + 0.03 * i))
            ph = 40 * np.sin(2 * np.pi * f_hz / f_hz[-1] * (j + 1)) + (
                120 if i == j else -30
            )
            db = 20 * np.log10(np.clip(mag, 1e-9, None))
            deg = np.mod(ph + rng.normal(0, 0.2, size=len(f_hz)), 360) - 180
            data[(f"o{i + 1}", f"o{j + 1}")] = SParam(db=db, deg=deg)
    return SParams(
        freq=freq_ghz,
        data=data,
        port_names=[f"o{i + 1}" for i in range(n_ports)],
        z0=50.0,
    )


def test_touchstone_round_trip_2port_real_data(tmp_path: Path):
    """2-port: committed Palace data round-trips far below 1e-9."""
    if not (TWO_PORT_DATA / "port-S.csv").exists():  # pragma: no cover
        pytest.skip("circuit-synthesis data not present")
    sp = load_sparams(TWO_PORT_DATA)

    path = sp.write_touchstone(tmp_path / "inductor")
    assert path.suffix == ".s2p"

    back = SParams.from_touchstone(path)
    assert back.port_names == sp.port_names, "port order must be preserved"
    assert back.z0 == pytest.approx(sp.z0), "reference impedance must be preserved"
    assert np.allclose(back.freq, sp.freq), (
        "frequency units must be preserved (Hz <-> GHz)"
    )

    err = _max_abs_s_error(sp, back)
    assert err < 1e-9, f"complex-S round-trip error {err:.2e} exceeds 1e-9"


def test_touchstone_round_trip_4port(tmp_path: Path):
    """4-port: full matrix round-trips below 1e-9 with port order intact."""
    freq_ghz = np.linspace(5, 40, 37)
    sp = _synthetic_sparams(4, freq_ghz, seed=11)

    path = sp.write_touchstone(tmp_path / "four_port")
    assert path.suffix == ".s4p"

    back = SParams.from_touchstone(path)
    assert back.port_names == sp.port_names
    assert back.z0 == pytest.approx(sp.z0)

    err = _max_abs_s_error(sp, back)
    assert err < 1e-9, f"complex-S round-trip error {err:.2e} exceeds 1e-9"


def test_touchstone_round_trip_different_z0(tmp_path: Path):
    """A non-default reference impedance is written into the header and restored."""
    freq_ghz = np.linspace(1, 20, 13)
    sp = _synthetic_sparams(2, freq_ghz, seed=5)
    sp = SParams(freq=sp.freq, data=sp._data, port_names=sp.port_names, z0=25.0)

    path = sp.write_touchstone(tmp_path / "z0_25")
    back = SParams.from_touchstone(path)
    assert back.z0 == pytest.approx(25.0)
    assert _max_abs_s_error(sp, back) < 1e-9


def test_touchstone_port_names_restored(tmp_path: Path):
    """Port names embedded as comments come back through the reader."""
    freq_ghz = np.linspace(1, 10, 7)
    sp = _synthetic_sparams(3, freq_ghz, seed=2)
    sp = SParams(freq=sp.freq, data=sp._data, port_names=["P1", "P2", "S1"], z0=50.0)

    back = SParams.from_touchstone(sp.write_touchstone(tmp_path / "named"))
    assert back.port_names == ["P1", "P2", "S1"]
    # names, not port order, drive the (to, from) lookup after import
    assert ("P1", "S1") in back._data
