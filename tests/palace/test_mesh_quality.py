"""Mesh distortion agrees with the Palace/MFEM normalized Jacobian metric."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gmsh
import numpy as np
import pytest

from gsim.palace import DrivenSim
from gsim.palace.base import PalaceSimMixin
from gsim.palace.mesh import quality
from gsim.palace.mesh.config_generator import collect_mesh_stats
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.models.results import SimulationResult, format_mesh_distortion

IDEAL_VERTICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)],
    ]
)


@pytest.fixture(autouse=True)
def _gmsh_model():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("distortion")
    try:
        yield
    finally:
        gmsh.finalize()


def _add_element(points, element_type=4, tag=17, node_start=1):
    dimension = gmsh.model.mesh.getElementProperties(element_type)[1]
    entity = gmsh.model.addDiscreteEntity(dimension)
    # Sparse node tags exercise the mapping from connectivity to coordinates.
    node_tags = np.arange(len(points)) * 7 + node_start
    gmsh.model.mesh.addNodes(dimension, entity, node_tags, points.ravel())
    gmsh.model.mesh.addElementsByType(entity, element_type, [tag], node_tags)


@pytest.mark.parametrize("stretch", [1.0, 10.0, 1000.0])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_distortion_is_normalized_and_scale_invariant(stretch, scale):
    rotation = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])
    points = IDEAL_VERTICES * [stretch, 1, 1]
    points = (points @ rotation.T + [5, -7, 11]) * scale
    _add_element(points)

    stats = collect_mesh_stats()

    assert stats["tetrahedra"] == 1
    assert stats["kappa"]["max"] == pytest.approx(stretch)
    assert stats["kappa"]["worst_element_tag"] == 17
    assert stats["kappa"]["singular_elements"] == 0
    assert stats["kappa"]["sample_location"] == "tetrahedron center"


def test_reduction_across_chunks_and_unsorted_nodes(monkeypatch):
    monkeypatch.setattr(quality, "_BATCH_SIZE", 2)
    vertices = np.concatenate([IDEAL_VERTICES * [1, 1, k] for k in (3, 1, 2, 9, 5)])
    node_tags = np.arange(len(vertices), dtype=np.uint64) * 7 + 1
    element_tags = np.array([11, 12, 13, 14, 15])
    blocks = [(4, element_tags, node_tags.copy())]
    shuffled = np.random.default_rng(42).permutation(len(vertices))

    distortion = quality.tetrahedron_distortion(
        blocks, node_tags[shuffled], vertices[shuffled]
    )

    assert distortion["max"] == pytest.approx(9)
    assert distortion["worst_element_tag"] == 14


def test_curved_geometry_uses_all_nodes_at_center():
    reference_nodes = gmsh.model.mesh.getElementProperties(11)[4].reshape(-1, 3)
    points = reference_nodes @ IDEAL_VERTICES[1:]
    # This quadratic deformation has unit derivative at the tet center, but
    # changes its corner geometry. A corner-only Jacobian would be incorrect.
    points[:, 0] += 0.5 * (points[:, 0] - 0.5) ** 2
    _add_element(points, element_type=11)

    stats = collect_mesh_stats()

    assert stats["tetrahedra"] == 1
    assert stats["nodes"] == 10
    assert stats["kappa"]["max"] == pytest.approx(1)


def test_mixed_orders_are_counted_and_reduced():
    _add_element(IDEAL_VERTICES * [3, 1, 1])
    reference_nodes = gmsh.model.mesh.getElementProperties(11)[4].reshape(-1, 3)
    points = reference_nodes @ IDEAL_VERTICES[1:]
    _add_element(points * [7, 1, 1], element_type=11, tag=23, node_start=100)

    stats = collect_mesh_stats()

    assert stats["tetrahedra"] == 2
    assert stats["kappa"]["max"] == pytest.approx(7)
    assert stats["kappa"]["worst_element_tag"] == 23


def test_singular_tet_is_reported_without_nonfinite_json():
    _add_element(IDEAL_VERTICES * [1, 1, 0])

    distortion = collect_mesh_stats()["kappa"]

    assert distortion["max"] is None
    assert distortion["singular_elements"] == 1
    assert distortion["worst_element_tag"] == 17
    json.dumps(distortion, allow_nan=False)
    formatted = format_mesh_distortion({"kappa": distortion})
    assert formatted is not None
    assert "infinite (1 singular tet centers)" in formatted


def test_distortion_does_not_replace_signed_validity():
    _add_element(IDEAL_VERTICES[[1, 0, 2, 3]])

    stats = collect_mesh_stats()

    assert stats["kappa"]["max"] == pytest.approx(1)
    assert stats["sicn"]["invalid"] == 1


def test_no_tet_distortion_for_2d_mesh():
    _add_element(IDEAL_VERTICES[:3], element_type=2)

    stats = collect_mesh_stats()

    assert "kappa" not in stats
    assert "tetrahedra" not in stats
    assert format_mesh_distortion(stats) is None


def test_distortion_appears_in_both_mesh_summaries(capsys):
    _add_element(IDEAL_VERTICES * [42, 1, 1])
    stats = collect_mesh_stats()
    sim = SimpleNamespace(
        _last_mesh_result=MeshResult(mesh_path=Path("mesh.msh"), mesh_stats=stats)
    )
    result = SimulationResult(
        mesh_path=Path("mesh.msh"), output_dir=Path("output"), mesh_stats=stats
    )

    PalaceSimMixin.print_mesh_stats(sim)

    expected = "Worst element distortion, \u03ba: 42 (tet centers; 1 is ideal)"
    assert expected in str(result)
    assert expected in capsys.readouterr().out


def test_sim_mesh_logs_metrics_automatically(monkeypatch, tmp_path, caplog):
    sim = DrivenSim()
    sim.set_output_dir(tmp_path)
    monkeypatch.setattr(
        DrivenSim,
        "validate_config",
        lambda _self: SimpleNamespace(valid=True, errors=[]),
    )
    monkeypatch.setattr(DrivenSim, "_resolve_stack", lambda _self: object())
    monkeypatch.setattr(
        DrivenSim, "_configure_ports_on_component", lambda _self, _stack: None
    )
    monkeypatch.setattr(
        "gsim.palace.ports.extract_ports", lambda _component, _stack: []
    )
    result = SimulationResult(
        mesh_path=tmp_path / "mesh.msh",
        output_dir=tmp_path,
        mesh_stats={
            "nodes": 4,
            "tetrahedra": 1,
            "kappa": {"max": 42.0},
            "field_dofs": {"estimated_field_dofs": 20, "field_order": 2},
        },
    )
    monkeypatch.setattr(
        DrivenSim, "_generate_mesh_internal", lambda _self, **_kwargs: result
    )

    with caplog.at_level("INFO", logger="gsim.palace.base"):
        sim.mesh()

    assert "Worst element distortion, \u03ba: 42" in caplog.text
    assert (
        "Estimated Field DOFs: 20 (order 2; before Palace preprocessing)" in caplog.text
    )
