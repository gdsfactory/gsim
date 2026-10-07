"""Tests for symmetry-aware S-parameter post-processing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gsim.palace.results import SParam, SParams, load_sparams
from gsim.palace.symmetry import (
    combine_even_odd,
    full_model_impedance,
    mixed_mode_from_halves,
)

FREQ = np.array([1.0, 2.0, 3.0])
NAMES = ["a", "b"]


def _symmetry(kind: str) -> dict:
    """Symmetry record as written to ``port_information.json``."""
    return {
        "axis": "y",
        "position": 0.0,
        "kind": kind,
        "keep": "positive",
        "mode": "even" if kind == "pmc" else "odd",
    }


def _sparams(
    matrix: np.ndarray,
    kind: str | None,
    *,
    z0: float = 50.0,
    ptype: str = "lumped",
    freq: np.ndarray = FREQ,
    names: list[str] = NAMES,
) -> SParams:
    """SParams from a complex ``(n_freq, n, n)`` matrix; ``S[to, from]``."""
    data = {}
    for i, to in enumerate(names):
        for j, frm in enumerate(names):
            s = matrix[:, i, j]
            data[(to, frm)] = SParam(
                db=20 * np.log10(np.abs(s)), deg=np.rad2deg(np.angle(s))
            )
    return SParams(
        freq=freq,
        data=data,
        port_names=list(names),
        symmetry=_symmetry(kind) if kind else None,
        port_meta={n: {"type": ptype, "Z0": z0} for n in names},
    )


def _full_blocks(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric blocks A, B of a mirror-symmetric reciprocal 4-port."""
    rng = np.random.default_rng(seed)
    a = 0.3 * (rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2)))
    b = 0.3 * (rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2)))
    return a + a.transpose(0, 2, 1), b + b.transpose(0, 2, 1)


def _halves(seed: int = 0) -> tuple[SParams, SParams, np.ndarray, np.ndarray]:
    """Even (cc = A + B) and odd (dd = A - B) half-model results."""
    a, b = _full_blocks(seed)
    return _sparams(a + b, "pmc"), _sparams(a - b, "pec"), a, b


def _complex(sp: SParams, to: str, frm: str) -> np.ndarray:
    """Complex values of one S-parameter."""
    return sp[to, frm].complex


def test_full_model_impedance():
    """Z_diff = 2 Z_odd for a PEC plane, Z_cm = Z_even / 2 for PMC."""
    assert full_model_impedance(40.0, "pec") == pytest.approx(80.0)
    assert full_model_impedance(40.0, "pmc") == pytest.approx(20.0)
    with pytest.raises(ValueError, match="kind"):
        full_model_impedance(40.0, "mirror")


def test_mixed_mode_references():
    """Lumped references: R/2 for the even and 2R for the odd half."""
    even, odd, _, _ = _halves()
    mm = mixed_mode_from_halves(even, odd)
    assert mm["cc"] is even
    assert mm["dd"] is odd
    assert mm["z_ref_cc"] == {"a": 25.0, "b": 25.0}
    assert mm["z_ref_dd"] == {"a": 100.0, "b": 100.0}


def test_mixed_mode_wave_ports_are_modal():
    """Wave ports have a modal reference."""
    a, b = _full_blocks()
    even = _sparams(a + b, "pmc", ptype="waveport")
    odd = _sparams(a - b, "pec", ptype="waveport")
    mm = mixed_mode_from_halves(even, odd)
    assert mm["z_ref_cc"] == "modal"
    assert mm["z_ref_dd"] == "modal"


def test_mixed_mode_order_of_halves_does_not_matter_for_kind_check():
    """Swapping the halves is an error: the even half must be PMC."""
    even, odd, _, _ = _halves()
    with pytest.raises(ValueError, match="PMC"):
        mixed_mode_from_halves(odd, even)


def test_mixed_mode_rejects_same_kinds():
    """Two PMC halves cannot be combined."""
    a, b = _full_blocks()
    with pytest.raises(ValueError, match="opposite"):
        mixed_mode_from_halves(_sparams(a, "pmc"), _sparams(b, "pmc"))


def test_mixed_mode_rejects_missing_symmetry():
    """A result without a symmetry record is not a half model."""
    a, b = _full_blocks()
    with pytest.raises(ValueError, match="symmetry"):
        mixed_mode_from_halves(_sparams(a, None), _sparams(b, "pec"))


def test_mixed_mode_rejects_mismatched_reference_impedance():
    """Different port R is an error."""
    a, b = _full_blocks()
    with pytest.raises(ValueError, match="impedance"):
        mixed_mode_from_halves(_sparams(a, "pmc"), _sparams(b, "pec", z0=75.0))


def test_mixed_mode_rejects_mismatched_ports_and_freq():
    """Port names and frequencies must agree."""
    a, b = _full_blocks()
    even = _sparams(a, "pmc")
    with pytest.raises(ValueError, match="ports"):
        mixed_mode_from_halves(even, _sparams(b, "pec", names=["a", "c"]))
    with pytest.raises(ValueError, match="frequenc"):
        mixed_mode_from_halves(even, _sparams(b, "pec", freq=FREQ * 2))


def test_combine_round_trip_through_mixed_mode():
    """Combining the halves and converting back gives cc and dd again."""
    even, odd, a, b = _halves()
    full = combine_even_odd(even, odd)
    assert full.port_names == ["a", "b", "a_mirror", "b_mirror"]
    for i, to in enumerate(NAMES):
        for j, frm in enumerate(NAMES):
            direct = _complex(full, to, frm)
            cross = _complex(full, to, f"{frm}_mirror")
            np.testing.assert_allclose(direct, a[:, i, j], atol=1e-9)
            np.testing.assert_allclose(cross, b[:, i, j], atol=1e-9)
            np.testing.assert_allclose(direct + cross, _complex(even, to, frm))
            np.testing.assert_allclose(direct - cross, _complex(odd, to, frm))


def test_combine_is_reciprocal_and_mirror_symmetric():
    """S_ij = S_ji and S_i'j' = S_ij for reciprocal halves."""
    even, odd, _, _ = _halves(seed=3)
    full = combine_even_odd(even, odd)
    names = full.port_names
    for p in names:
        for q in names:
            np.testing.assert_allclose(_complex(full, p, q), _complex(full, q, p))
    for p in NAMES:
        for q in NAMES:
            np.testing.assert_allclose(
                _complex(full, f"{p}_mirror", f"{q}_mirror"), _complex(full, p, q)
            )
            np.testing.assert_allclose(
                _complex(full, f"{p}_mirror", q), _complex(full, p, f"{q}_mirror")
            )


def test_combine_custom_mirror_names():
    """Mirror names can be given per port."""
    even, odd, _, _ = _halves()
    full = combine_even_odd(even, odd, mirror_names={"a": "c", "b": "d"})
    assert full.port_names == ["a", "b", "c", "d"]


def test_combine_rejects_wave_ports():
    """A full model with wave ports is modal; combining is refused."""
    a, b = _full_blocks()
    even = _sparams(a + b, "pmc", ptype="waveport")
    odd = _sparams(a - b, "pec", ptype="waveport")
    with pytest.raises(ValueError, match="wave"):
        combine_even_odd(even, odd)


def test_combine_rejects_mismatched_inputs():
    """The same checks as ``mixed_mode_from_halves`` apply."""
    a, b = _full_blocks()
    with pytest.raises(ValueError, match="impedance"):
        combine_even_odd(_sparams(a, "pmc"), _sparams(b, "pec", z0=75.0))


def test_combine_rejects_mirror_name_clash():
    """Mirror names must not collide with existing ports."""
    even, odd, _, _ = _halves()
    with pytest.raises(ValueError, match="mirror"):
        combine_even_odd(even, odd, mirror_names={"a": "b", "b": "b2"})


def test_sparams_without_symmetry_has_none_and_plain_repr():
    """No plane: symmetry is None and the repr is unchanged."""
    a, _ = _full_blocks()
    sp = _sparams(a, None)
    assert sp.symmetry is None
    assert "symmetry" not in repr(sp)


def test_repr_shows_symmetry_line():
    """A half model's repr names the plane and mode."""
    even, odd, _, _ = _halves()
    assert "symmetry: y=0.0 PMC (even/common mode" in repr(even)
    assert "symmetry: y=0.0 PEC (odd/differential mode" in repr(odd)


def test_load_sparams_reads_symmetry(tmp_path: Path):
    """``load_sparams`` takes symmetry and port metadata from the JSON."""
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)
    port_info = {
        "ports": [
            {"portnumber": 1, "name": "o1", "Z0": 50.0, "type": "lumped"},
            {"portnumber": 2, "name": "o2", "Z0": 50.0, "type": "lumped"},
        ],
        "symmetry": _symmetry("pec"),
    }
    (tmp_path / "port_information.json").write_text(json.dumps(port_info))
    (palace_dir / "port-S.csv").write_text(
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.),"
        " |S[2][1]| (dB), arg(S[2][1]) (deg.)\n"
        "1.0, -20.0, -45.0, -3.0, -90.0\n"
    )
    sp = load_sparams(tmp_path)
    assert sp.symmetry == _symmetry("pec")
    assert sp.port_meta["o1"] == {"type": "lumped", "Z0": 50.0}
    assert "odd/differential" in repr(sp)


def test_load_sparams_without_symmetry(tmp_path: Path):
    """Old port_information.json files still load, with symmetry None."""
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)
    (tmp_path / "port_information.json").write_text(
        json.dumps({"ports": [{"portnumber": 1, "name": "o1"}]})
    )
    (palace_dir / "port-S.csv").write_text(
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.)\n1.0, -20.0, -45.0\n"
    )
    sp = load_sparams(tmp_path)
    assert sp.symmetry is None
