"""Tests for gsim.palace.results — S-parameter loading with port-name mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gsim.palace.results import get_port_map, load_sparams


@pytest.fixture
def sim_dir(tmp_path: Path) -> Path:
    """Create a minimal Palace output directory."""
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)

    # Write port_information.json at the sim root (common location)
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

    # Write a minimal port-S.csv
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
    """Sim dir with port_information.json that has no name fields (legacy)."""
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


class TestLoadSparams:
    """Tests for load_sparams."""

    def test_columns_renamed_with_port_names(self, sim_dir: Path) -> None:
        df = load_sparams(sim_dir)
        assert "freq_ghz" in df.columns
        assert "S_o1_o1_db" in df.columns
        assert "S_o2_o1_db" in df.columns
        assert "S_o3_o1_db" in df.columns
        assert "S_o1_o1_deg" in df.columns

    def test_data_preserved(self, sim_dir: Path) -> None:
        df = load_sparams(sim_dir)
        assert len(df) == 2
        assert df["freq_ghz"].iloc[0] == pytest.approx(1.0)
        assert df["S_o1_o1_db"].iloc[0] == pytest.approx(-20.0)
        assert df["S_o2_o1_deg"].iloc[0] == pytest.approx(-90.0)

    def test_fallback_numeric_names(self, sim_dir_no_names: Path) -> None:
        df = load_sparams(sim_dir_no_names)
        assert "S_p1_p1_db" in df.columns
        assert "S_p2_p1_db" in df.columns

    def test_explicit_port_info_path(self, sim_dir: Path) -> None:
        df = load_sparams(sim_dir, port_info_path=sim_dir / "port_information.json")
        assert "S_o1_o1_db" in df.columns

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"port-S\.csv"):
            load_sparams(tmp_path)

    def test_accepts_palace_subdir(self, sim_dir: Path) -> None:
        """Can pass the output/palace/ dir directly."""
        df = load_sparams(sim_dir / "output" / "palace")
        assert "freq_ghz" in df.columns
        assert "S_o1_o1_db" in df.columns

    def test_accepts_results_dict(self, sim_dir: Path) -> None:
        """Can pass a results dict like sim.run() returns."""
        results = {
            "port-S.csv": sim_dir / "output" / "palace" / "port-S.csv",
            "port-V.csv": sim_dir / "output" / "palace" / "port-V.csv",
        }
        df = load_sparams(results)
        assert "freq_ghz" in df.columns
        assert "S_o1_o1_db" in df.columns
        assert "S_o2_o1_db" in df.columns

    def test_results_dict_missing_csv_raises(self) -> None:
        """Results dict without port-S.csv raises."""
        with pytest.raises(FileNotFoundError, match="port-S"):
            load_sparams({"other.csv": Path("/nonexistent")})


class TestGetPortMap:
    """Tests for get_port_map."""

    def test_returns_mapping(self, sim_dir: Path) -> None:
        pm = get_port_map(sim_dir)
        assert pm == {1: "o1", 2: "o2", 3: "o3"}

    def test_legacy_numeric_fallback(self, sim_dir_no_names: Path) -> None:
        pm = get_port_map(sim_dir_no_names)
        assert pm == {1: "p1", 2: "p2"}
