"""Palace mesh-contact and domain-energy helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gsim.palace import add_domain_energy_postprocessing, check_lumped_port_contact


@pytest.fixture
def sim_dir(tmp_path: Path) -> Path:
    (tmp_path / "palace.msh").write_text(
        """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
5
2 1 "M1_pec"
2 2 "P1"
2 3 "P2"
3 4 "VACUUM"
3 5 "SUBSTRATE"
$EndPhysicalNames
$Nodes
6
1 0 0 0
2 1 0 0
3 0 1 0
4 1 1 0
5 2 0 0
6 2 1 0
$EndNodes
$Elements
3
1 2 2 1 1 1 2 3
2 2 2 2 2 2 4 3
3 2 2 3 3 4 5 6
$EndElements
"""
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "Domains": {
                    "Materials": [{"Attributes": [4]}],
                    "Postprocessing": {"Energy": [{"Index": 7, "Attributes": [5]}]},
                },
                "Boundaries": {
                    "PEC": {"Attributes": [1]},
                    "LumpedPort": [{"Index": 1, "Attributes": [2]}],
                },
            }
        )
    )
    return tmp_path


def test_domain_energy_requests_preserve_indices(sim_dir: Path) -> None:
    expected = {"VACUUM": 8, "SUBSTRATE": 7}
    assert add_domain_energy_postprocessing(sim_dir, list(expected)) == expected
    assert add_domain_energy_postprocessing(sim_dir, list(expected)) == expected
    config = json.loads((sim_dir / "config.json").read_text())
    assert config["Domains"]["Postprocessing"]["Energy"] == [
        {"Index": 7, "Attributes": [5]},
        {"Index": 8, "Attributes": [4]},
    ]


def test_domain_energy_rejects_surface_group(sim_dir: Path) -> None:
    config_path = sim_dir / "config.json"
    original = config_path.read_text()
    with pytest.raises(ValueError, match="dimension 3"):
        add_domain_energy_postprocessing(sim_dir, ["M1_pec"])
    assert config_path.read_text() == original


def test_lumped_port_contact_rejects_orphan(sim_dir: Path) -> None:
    assert check_lumped_port_contact(sim_dir) == {1: 2}
    path = sim_dir / "config.json"
    config = json.loads(path.read_text())
    config["Boundaries"]["LumpedPort"].append({"Index": 2, "Attributes": [3]})
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=r"lumped ports \[2\].*share no nodes"):
        check_lumped_port_contact(sim_dir)


def test_lumped_port_contact_uses_configured_attributes(sim_dir: Path) -> None:
    path = sim_dir / "config.json"
    config = json.loads(path.read_text())
    config["Boundaries"]["LumpedPort"].extend(
        [
            {"Index": 99, "Attributes": [2], "R": 50.0},
            {"Index": 100, "Elements": [{"Attributes": [2]}]},
        ]
    )
    path.write_text(json.dumps(config))
    assert check_lumped_port_contact(sim_dir) == {1: 2, 99: 2, 100: 2}


def test_lumped_port_contact_preserves_active_gmsh_model(sim_dir: Path) -> None:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.model.add("caller")
        gmsh.model.geo.addPoint(0, 0, 0, 1, 42)
        gmsh.model.geo.synchronize()
        models = gmsh.model.list()

        assert check_lumped_port_contact(sim_dir) == {1: 2}
        assert gmsh.model.getCurrent() == "caller"
        assert gmsh.model.list() == models
        assert gmsh.model.getEntities(0) == [(0, 42)]

        config_path = sim_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["Boundaries"]["LumpedPort"].append({"Index": 2, "Attributes": [3]})
        config_path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="share no nodes"):
            check_lumped_port_contact(sim_dir)
        assert gmsh.model.getCurrent() == "caller"
        assert gmsh.model.list() == models
    finally:
        gmsh.finalize()
