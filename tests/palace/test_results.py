"""Tests for gsim.palace.results — S-parameter loading with port-name mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gsim.palace.results import SParams, get_port_map, load_sparams


@pytest.fixture
def sim_dir(tmp_path: Path) -> Path:
    """Create a minimal Palace output directory."""
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)

    port_info = {
        "ports": [
            {"portnumber": 1, "name": "o1", "Z0": 50.0, "type": "cpw"},
            {"portnumber": 2, "name": "o2", "Z0": 50.0, "type": "cpw"},
            {"portnumber": 3, "name": "o3", "Z0": 50.0, "type": "lumped"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (tmp_path / "port_information.json").write_text(json.dumps(port_info))

    csv_content = (
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.),"
        " |S[2][1]| (dB), arg(S[2][1]) (deg.),"
        " |S[3][1]| (dB), arg(S[3][1]) (deg.)\n"
        "1.0, -20.0, -45.0, -3.0, -90.0, -30.0, -120.0\n"
        "2.0, -18.0, -50.0, -2.5, -85.0, -28.0, -115.0\n"
    )
    (palace_dir / "port-S.csv").write_text(csv_content)

    return tmp_path


@pytest.fixture
def sim_dir_no_names(tmp_path: Path) -> Path:
    """Sim dir with port_information.json without name fields."""
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)

    port_info = {
        "ports": [
            {"portnumber": 1, "Z0": 50.0, "type": "cpw"},
            {"portnumber": 2, "Z0": 50.0, "type": "cpw"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (tmp_path / "port_information.json").write_text(json.dumps(port_info))

    csv_content = (
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.),"
        " |S[2][1]| (dB), arg(S[2][1]) (deg.)\n"
        "1.0, -20.0, -45.0, -3.0, -90.0\n"
    )
    (palace_dir / "port-S.csv").write_text(csv_content)

    return tmp_path


class TestSParams:
    """Tests for the SParams result object."""

    def test_returns_sparams_object(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert isinstance(sp, SParams)

    def test_freq(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.freq[0] == pytest.approx(1.0)
        assert sp.freq[1] == pytest.approx(2.0)
        assert len(sp.freq) == 2

    def test_port_names(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.port_names == ["o1", "o2", "o3"]

    def test_bracket_access(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s11 = sp["o1", "o1"]
        assert s11.db[0] == pytest.approx(-20.0)
        assert s11.deg[0] == pytest.approx(-45.0)

    def test_bracket_access_cross(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s21 = sp["o2", "o1"]
        assert s21.db[0] == pytest.approx(-3.0)
        assert s21.deg[0] == pytest.approx(-90.0)

    def test_mag_property(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s11 = sp["o1", "o1"]
        expected_mag = 10 ** (-20.0 / 20)
        assert s11.mag[0] == pytest.approx(expected_mag)

    def test_complex_property(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s11 = sp["o1", "o1"]
        c = s11.complex[0]
        assert abs(c) == pytest.approx(10 ** (-20.0 / 20))
        assert np.rad2deg(np.angle(c)) == pytest.approx(-45.0)

    def test_rf_shorthand_s11(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.s11.db[0] == pytest.approx(-20.0)

    def test_rf_shorthand_s21(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.s21.db[0] == pytest.approx(-3.0)

    def test_rf_shorthand_s31(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.s31.db[0] == pytest.approx(-30.0)

    def test_invalid_shorthand_raises(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        with pytest.raises(AttributeError):
            _ = sp.s99

    def test_invalid_bracket_raises(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        with pytest.raises(KeyError, match="not found"):
            _ = sp["o1", "o99"]

    def test_keys(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        keys = sp.keys()
        assert ("o1", "o1") in keys
        assert ("o2", "o1") in keys
        assert ("o3", "o1") in keys

    def test_repr(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        r = repr(sp)
        assert "3 ports" in r
        assert "o1" in r

    def test_to_dataframe(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        df = sp.to_dataframe()
        assert "freq_ghz" in df.columns
        assert "S_o1_o1_db" in df.columns

    def test_plot_runs(self, sim_dir: Path) -> None:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        sp = load_sparams(sim_dir)
        sp.plot()
        plt.close("all")


class TestLoadSparamsSource:
    """Tests for source resolution (dir, subdir, dict)."""

    def test_accepts_dir(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert len(sp.freq) == 2

    def test_accepts_palace_subdir(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir / "output" / "palace")
        assert len(sp.freq) == 2

    def test_accepts_results_dict(self, sim_dir: Path) -> None:
        results = {
            "port-S.csv": sim_dir / "output" / "palace" / "port-S.csv",
        }
        sp = load_sparams(results)
        assert sp["o1", "o1"].db[0] == pytest.approx(-20.0)

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"port-S\.csv"):
            load_sparams(tmp_path)

    def test_results_dict_missing_csv_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="port-S"):
            load_sparams({"other.csv": Path("/nonexistent")})

    def test_fallback_numeric_names(self, sim_dir_no_names: Path) -> None:
        sp = load_sparams(sim_dir_no_names)
        assert sp.port_names == ["p1", "p2"]
        assert sp["p1", "p1"].db[0] == pytest.approx(-20.0)


class TestGetPortMap:
    """Tests for get_port_map."""

    def test_returns_mapping(self, sim_dir: Path) -> None:
        pm = get_port_map(sim_dir)
        assert pm == {1: "o1", 2: "o2", 3: "o3"}

    def test_legacy_numeric_fallback(self, sim_dir_no_names: Path) -> None:
        pm = get_port_map(sim_dir_no_names)
        assert pm == {1: "p1", 2: "p2"}


class TestSParamsSaveLoad:
    """Tests for SParams save_npz/from_file round-trip."""

    def test_round_trip(self, sim_dir: Path, tmp_path: Path) -> None:
        sp = load_sparams(sim_dir)
        out = sp.save_npz(tmp_path / "cached")
        assert out.suffix == ".npz"
        assert out.exists()

        loaded = SParams.from_file(out)
        assert loaded.port_names == sp.port_names
        assert len(loaded.freq) == len(sp.freq)
        np.testing.assert_allclose(loaded.freq, sp.freq)
        for key in sp._data:
            np.testing.assert_allclose(loaded[key].db, sp[key].db)
            np.testing.assert_allclose(loaded[key].deg, sp[key].deg)

    def test_adds_npz_suffix(self, sim_dir: Path, tmp_path: Path) -> None:
        sp = load_sparams(sim_dir)
        out = sp.save_npz(tmp_path / "no_ext")
        assert out.name == "no_ext.npz"

    def test_from_file_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SParams.from_file(tmp_path / "nonexistent.npz")
