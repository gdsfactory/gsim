"""Input topology estimates and JSON metadata preserve the generated mesh."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import gmsh
import numpy as np
import pytest

from gsim.palace import DrivenSim
from gsim.palace.base import PalaceSimMixin
from gsim.palace.mesh.config_generator import collect_mesh_stats
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.mesh.metadata import write_metadata
from gsim.palace.models.results import SimulationResult


@pytest.fixture(autouse=True)
def _two_tetrahedra():
    """Two tets sharing a face: five vertices, nine edges and seven faces."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("shared_face")
    entity = gmsh.model.addDiscreteEntity(3)
    gmsh.model.mesh.addNodes(
        3,
        entity,
        [3, 9, 42, 101, 205],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1],
    )
    gmsh.model.mesh.addElementsByType(
        entity, 4, [7, 19], [3, 9, 42, 101, 9, 3, 42, 205]
    )
    try:
        yield
    finally:
        gmsh.finalize()


@pytest.mark.parametrize(("order", "expected"), [(1, 9), (2, 32), (3, 75), (4, 144)])
def test_shared_entities_are_counted_once(order, expected):
    stats = collect_mesh_stats(field_order=order)

    assert stats["topology"] == {"edges": 9, "triangular_faces": 7}
    assert stats["field_dofs"]["estimated_field_dofs"] == expected
    assert stats["field_dofs"]["field_order"] == order


@pytest.mark.parametrize("geometry_order", [1, 2])
def test_statistics_preserve_mesh_and_separate_geometry_order(geometry_order, tmp_path):
    # Use a generated volume: Gmsh does not elevate hand-built discrete tets.
    gmsh.clear()
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
    gmsh.model.mesh.generate(3)
    linear_stats = collect_mesh_stats(field_order=3)
    gmsh.model.mesh.setOrder(geometry_order)
    before, after = tmp_path / "before.msh", tmp_path / "after.msh"
    gmsh.write(str(before))

    stats = collect_mesh_stats(field_order=3)
    gmsh.write(str(after))

    assert stats["geometry_orders"] == [geometry_order]
    assert stats["field_dofs"] == linear_stats["field_dofs"]
    assert before.read_bytes() == after.read_bytes()


def test_detached_surface_does_not_add_volume_dofs():
    surface = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, surface, [301, 302, 303], [2, 0, 0, 3, 0, 0, 2, 1, 0])
    gmsh.model.mesh.addElementsByType(surface, 2, [21], [301, 302, 303])

    stats = collect_mesh_stats()

    assert stats["field_dofs"]["estimated_field_dofs"] == 32
    assert stats["dimension"] == 3


def test_mixed_volume_types_have_no_partial_estimate():
    volume = gmsh.model.addDiscreteEntity(3)
    nodes = np.arange(301, 309)
    gmsh.model.mesh.addNodes(
        3,
        volume,
        nodes,
        [2, 0, 0, 3, 0, 0, 3, 1, 0, 2, 1, 0, 2, 0, 1, 3, 0, 1, 3, 1, 1, 2, 1, 1],
    )
    gmsh.model.mesh.addElementsByType(volume, 5, [21], nodes)

    estimate = collect_mesh_stats()["field_dofs"]

    assert estimate["estimated_field_dofs"] is None
    assert "unavailable_reason" in estimate


@pytest.mark.parametrize("problem_type", ["electrostatic", "boundarymode"])
def test_other_field_spaces_are_not_reported_as_nd(problem_type):
    assert (
        collect_mesh_stats(problem_type=problem_type)["field_dofs"][
            "estimated_field_dofs"
        ]
        is None
    )


def test_empty_mesh_has_no_estimate():
    gmsh.model.mesh.clear()
    assert collect_mesh_stats()["field_dofs"]["estimated_field_dofs"] is None


def test_metadata_tracks_effective_config_and_remains_extensible(tmp_path):
    stats = collect_mesh_stats(field_order=1)
    stats["future_measurement"] = {"value": 12}
    config = {
        "Problem": {"Type": "Driven"},
        "Solver": {"Order": 3},
        "Model": {"Refinement": {"MaxIts": 2}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    metadata = write_metadata(stats, tmp_path, config_path)

    assert metadata == json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["schema_version"] == 1
    assert metadata["mesh"]["field_dofs"]["estimated_field_dofs"] == 75
    assert metadata["simulation"]["refinement"] == {"MaxIts": 2}
    assert metadata["mesh"]["field_dofs"]["scope"] == "before_palace_preprocessing"
    assert metadata["mesh"]["future_measurement"] == {"value": 12}
    assert metadata["mesh"]["kappa"]["max"] > 1
    assert isinstance(metadata["mesh"]["kappa"]["max"], float)
    assert json.loads(config_path.read_text()) == config
    json.dumps(metadata, allow_nan=False)


def test_singular_kappa_is_valid_json(tmp_path):
    stats = collect_mesh_stats()
    stats["kappa"] = {"max": None, "singular_elements": 1}
    metadata = write_metadata(stats, tmp_path)
    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata
    assert metadata["mesh"]["kappa"]["max"] is None


def test_dofs_appear_in_printed_and_result_summaries(capsys):
    stats = collect_mesh_stats(field_order=3)
    sim = SimpleNamespace(
        _last_mesh_result=MeshResult(mesh_path=Path("mesh.msh"), mesh_stats=stats)
    )
    result = SimulationResult(
        mesh_path=Path("mesh.msh"), output_dir=Path("output"), mesh_stats=stats
    )
    PalaceSimMixin.print_mesh_stats(sim)

    expected = "Estimated Field DOFs: 75 (order 3; before Palace preprocessing)"
    assert expected in capsys.readouterr().out
    assert expected in str(result)


def test_upload_directory_includes_metadata(tmp_path, monkeypatch):
    stats = collect_mesh_stats()
    gmsh.write(str(tmp_path / "palace.msh"))
    sim = DrivenSim()
    sim.set_output_dir(tmp_path)

    def write_config(_self):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"Problem": {"Type": "Driven"}}))
        write_metadata(stats, tmp_path, config_path)
        return config_path

    monkeypatch.setattr(DrivenSim, "write_config", write_config)
    upload_dir = sim._prepare_upload_dir()
    try:
        metadata = json.loads((upload_dir / "metadata.json").read_text())
        assert metadata["mesh"]["field_dofs"]["estimated_field_dofs"] == 9
        assert metadata["mesh"]["kappa"] == stats["kappa"]
        assert (upload_dir / "palace.msh").read_bytes() == (
            tmp_path / "palace.msh"
        ).read_bytes()
    finally:
        shutil.rmtree(upload_dir)
