"""Tests for the mesh hash and the recorded Gmsh settings.

The hash identifies a mesh by what Palace reads from it: the node coordinates,
the connectivity of the elements and the physical groups they belong to. It
does not depend on how nodes and elements happen to be numbered, which is the
one thing Gmsh does not keep fixed across runs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import gmsh
import numpy as np
import pytest

import gsim
from gsim.palace.base import PalaceSimMixin
from gsim.palace.mesh import config_generator
from gsim.palace.mesh.config_generator import collect_mesh_stats
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.mesh.gmsh_utils import mesh_hash
from gsim.palace.mesh.metadata import write_metadata
from gsim.palace.models.results import SimulationResult, mesh_identity_lines


@pytest.fixture
def _gmsh_session() -> Iterator[None]:
    """An initialized Gmsh session, finalized after the test."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield
    gmsh.finalize()


def _mesh_two_boxes(
    size: float = 0.3,
    names: tuple[str, str] = ("left", "right"),
    tags: tuple[int, int, int] = (1, 2, 3),
) -> None:
    """Mesh two unit boxes that share a face, with volume and interface groups."""
    gmsh.model.add("two_boxes")
    left = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    right = gmsh.model.occ.addBox(1, 0, 0, 1, 1, 1)
    gmsh.model.occ.fragment([(3, left)], [(3, right)])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size)

    first, second = (tag for _, tag in gmsh.model.getEntities(3))
    boundary = {
        volume: {
            tag for _, tag in gmsh.model.getBoundary([(3, volume)], oriented=False)
        }
        for volume in (first, second)
    }
    gmsh.model.addPhysicalGroup(3, [first], tags[0], name=names[0])
    gmsh.model.addPhysicalGroup(3, [second], tags[1], name=names[1])
    gmsh.model.addPhysicalGroup(
        2, sorted(boundary[first] & boundary[second]), tags[2], name="interface"
    )
    gmsh.model.mesh.generate(3)


def _hash_of_fresh_mesh(**kwargs) -> str | None:
    """Mesh the two boxes in a new Gmsh session and return the mesh hash."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        _mesh_two_boxes(**kwargs)
        return mesh_hash()
    finally:
        gmsh.finalize()


def _renumber_randomly(seed: int = 0) -> None:
    """Give every node and element a new ID, keeping the mesh itself intact."""
    rng = np.random.default_rng(seed)
    nodes, _, _ = gmsh.model.mesh.getNodes()
    gmsh.model.mesh.renumberNodes(nodes, rng.permutation(nodes))
    elements = np.concatenate(gmsh.model.mesh.getElements()[1])
    gmsh.model.mesh.renumberElements(elements, rng.permutation(elements))


def _rewrite_elements(seed: int, *, shuffle_nodes: bool = False) -> None:
    """Store every entity's elements in a random order, and optionally list the
    nodes of each element in a random order too."""
    rng = np.random.default_rng(seed)
    for dim, entity in gmsh.model.getEntities():
        types, tags, nodes = gmsh.model.mesh.getElements(dim, entity)
        if len(types) == 0:
            continue
        gmsh.model.mesh.removeElements(dim, entity)
        new_tags, new_nodes = [], []
        for element_type, element_tags, element_nodes in zip(
            types, tags, nodes, strict=True
        ):
            per_element = gmsh.model.mesh.getElementProperties(element_type)[3]
            rows = element_nodes.reshape(-1, per_element)
            order = rng.permutation(len(rows))
            rows = rows[order]
            if shuffle_nodes:
                rows = rng.permuted(rows, axis=1)
            new_tags.append(element_tags[order])
            new_nodes.append(rows.ravel())
        gmsh.model.mesh.addElements(dim, entity, types, new_tags, new_nodes)


def test_the_same_mesh_gives_the_same_hash() -> None:
    first = _hash_of_fresh_mesh()
    assert first is not None
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64
    assert _hash_of_fresh_mesh() == first


@pytest.mark.usefixtures("_gmsh_session")
def test_numbering_does_not_change_the_hash() -> None:
    _mesh_two_boxes()
    before = mesh_hash()
    _renumber_randomly()
    assert mesh_hash() == before


@pytest.mark.usefixtures("_gmsh_session")
def test_the_order_of_the_elements_does_not_change_the_hash() -> None:
    _mesh_two_boxes()
    before = mesh_hash()
    _rewrite_elements(seed=1)
    assert mesh_hash() == before


@pytest.mark.usefixtures("_gmsh_session")
def test_the_order_of_the_nodes_inside_an_element_is_ignored() -> None:
    """The hash identifies the set of nodes of an element, not its orientation."""
    _mesh_two_boxes()
    before = mesh_hash()
    _rewrite_elements(seed=2, shuffle_nodes=True)
    assert mesh_hash() == before


@pytest.mark.parametrize("binary", [0, 1])
def test_the_written_file_gives_the_recorded_hash(tmp_path: Path, binary: int) -> None:
    """Opening the .msh gsim writes gives back the hash recorded when meshing."""
    path = tmp_path / "mesh.msh"
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        _mesh_two_boxes()
        recorded = mesh_hash()
        gmsh.option.setNumber("Mesh.Binary", binary)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(str(path))
        assert mesh_hash() == recorded
    finally:
        gmsh.finalize()


def test_a_different_mesh_gives_a_different_hash() -> None:
    assert _hash_of_fresh_mesh(size=0.3) != _hash_of_fresh_mesh(size=0.25)


@pytest.mark.usefixtures("_gmsh_session")
def test_moving_one_node_changes_the_hash() -> None:
    _mesh_two_boxes()
    before = mesh_hash()
    tags, coords, _ = gmsh.model.mesh.getNodes()
    gmsh.model.mesh.setNode(int(tags[0]), (coords[:3] + 1e-3).tolist(), [])
    assert mesh_hash() != before


@pytest.mark.usefixtures("_gmsh_session")
def test_a_move_far_below_the_resolution_does_not_change_the_hash() -> None:
    """Nodes are compared to 0.1 nm, so floating-point noise cannot flip the hash.

    The corner node sits exactly on the resolution grid, far from a rounding
    boundary, so a shift of 1e-6 um must round to the same position.
    """
    _mesh_two_boxes()
    before = mesh_hash()
    corner_tags, corner_coords, _ = gmsh.model.mesh.getNodes(0, 1)
    assert np.allclose(corner_coords, np.round(corner_coords)), (
        "box corners are integers"
    )
    gmsh.model.mesh.setNode(int(corner_tags[0]), (corner_coords + 1e-6).tolist(), [])
    assert mesh_hash() == before


def test_which_elements_belong_to_which_group_matters() -> None:
    """Swapping the names of the two volumes moves elements between groups."""
    assert _hash_of_fresh_mesh(names=("left", "right")) != _hash_of_fresh_mesh(
        names=("right", "left")
    )


def test_group_names_matter() -> None:
    assert _hash_of_fresh_mesh(names=("left", "right")) != _hash_of_fresh_mesh(
        names=("west", "right")
    )


def test_group_tag_numbers_do_not_matter() -> None:
    """The tag is an arbitrary label; the name is what identifies the group."""
    assert _hash_of_fresh_mesh(tags=(1, 2, 3)) == _hash_of_fresh_mesh(tags=(10, 20, 30))


@pytest.mark.usefixtures("_gmsh_session")
def test_higher_order_elements_are_hashed() -> None:
    _mesh_two_boxes()
    linear = mesh_hash()
    gmsh.model.mesh.setOrder(2)
    quadratic = mesh_hash()
    assert quadratic is not None
    assert quadratic != linear
    _renumber_randomly()
    assert mesh_hash() == quadratic


@pytest.mark.usefixtures("_gmsh_session")
def test_no_mesh_and_no_groups_give_no_hash() -> None:
    gmsh.model.add("empty")
    assert mesh_hash() is None

    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.5)
    gmsh.model.mesh.generate(3)
    assert box == 1
    assert mesh_hash() is None, "a mesh with no physical groups has nothing to hash"


@pytest.mark.usefixtures("_gmsh_session")
def test_hashing_leaves_the_model_untouched() -> None:
    _mesh_two_boxes()
    tags, coords, _ = gmsh.model.mesh.getNodes()
    elements = np.concatenate(gmsh.model.mesh.getElements()[1])
    mesh_hash()
    tags_after, coords_after, _ = gmsh.model.mesh.getNodes()
    elements_after = np.concatenate(gmsh.model.mesh.getElements()[1])
    assert np.array_equal(tags, tags_after)
    assert np.array_equal(coords, coords_after)
    assert np.array_equal(elements, elements_after)


@pytest.mark.usefixtures("_gmsh_session")
def test_collect_mesh_stats_records_hash_versions_and_settings() -> None:
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 2)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    _mesh_two_boxes()
    stats = collect_mesh_stats()

    assert stats["mesh_hash"] == mesh_hash()
    assert stats["versions"] == {"gsim": gsim.__version__, "gmsh": gmsh.__version__}
    options = stats["gmsh_options"]
    assert options["Mesh.MaxNumThreads3D"] == 2
    assert options["Mesh.Algorithm3D"] == 1
    for name in (
        "General.NumThreads",
        "Mesh.Algorithm",
        "Mesh.Reproducible",
        "Mesh.RandomSeed",
    ):
        assert name in options


@pytest.mark.usefixtures("_gmsh_session")
def test_a_failing_hash_keeps_the_versions_and_settings_and_warns(
    monkeypatch, caplog
) -> None:
    def broken() -> str:
        msg = "out of memory"
        raise MemoryError(msg)

    monkeypatch.setattr(config_generator, "mesh_hash", broken)
    _mesh_two_boxes()
    with caplog.at_level(logging.WARNING, logger=config_generator.__name__):
        stats = collect_mesh_stats()

    assert "mesh_hash" not in stats
    assert stats["versions"]["gmsh"] == gmsh.__version__
    assert "Mesh.Algorithm3D" in stats["gmsh_options"]
    assert "Mesh.MeshSizeFactor" in stats["gmsh_options"]
    assert "Could not hash the mesh: out of memory" in caplog.text


@pytest.mark.usefixtures("_gmsh_session")
def test_failing_settings_do_not_lose_the_hash(monkeypatch) -> None:
    def broken() -> dict:
        msg = "no options"
        raise RuntimeError(msg)

    monkeypatch.setattr(config_generator, "gmsh_options", broken)
    _mesh_two_boxes()
    stats = collect_mesh_stats()
    assert stats["mesh_hash"] == mesh_hash()
    assert "gmsh_options" not in stats


@pytest.mark.usefixtures("_gmsh_session")
def test_a_mesh_without_groups_has_its_settings_but_no_hash() -> None:
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
    gmsh.model.mesh.generate(3)
    stats = collect_mesh_stats()
    assert "mesh_hash" not in stats
    assert "Mesh.Algorithm3D" in stats["gmsh_options"]


def test_the_identity_lines_leave_out_options_the_gmsh_does_not_have() -> None:
    digest = "sha256:" + "ab" * 32
    assert len(mesh_identity_lines({"mesh_hash": digest})) == 1
    cases = {
        "Mesh.Algorithm": ("Mesher:     2D algorithm 6", 6.0),
        "Mesh.Algorithm3D": ("Mesher:     3D algorithm 10", 10.0),
        "General.NumThreads": ("Mesher:     threads 4 (1D/2D/3D limits 0/0/0)", 4.0),
    }
    for option, (expected, value) in cases.items():
        lines = mesh_identity_lines(
            {"mesh_hash": digest, "gmsh_options": {option: value}}
        )
        assert lines[1:] == [expected]


@pytest.mark.usefixtures("_gmsh_session")
def test_the_hash_and_settings_reach_the_metadata_file(tmp_path) -> None:
    """metadata.json exports the mesh stats, so the hash travels with the inputs."""
    _mesh_two_boxes()
    metadata = write_metadata(collect_mesh_stats(), tmp_path)

    mesh = metadata["mesh"]
    assert mesh["mesh_hash"] == mesh_hash()
    assert mesh["versions"] == {"gsim": gsim.__version__, "gmsh": gmsh.__version__}
    assert "Mesh.Algorithm3D" in mesh["gmsh_options"]
    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata


def _print_stats(mesh_stats: dict, capsys) -> str:
    sim = SimpleNamespace(
        _last_mesh_result=MeshResult(mesh_path=Path("mesh.msh"), mesh_stats=mesh_stats)
    )
    PalaceSimMixin.print_mesh_stats(sim)
    return capsys.readouterr().out


def test_print_mesh_stats_shows_the_hash_and_the_mesher(capsys) -> None:
    out = _print_stats(
        {
            "elements": 10,
            "tetrahedra": 8,
            "mesh_hash": "sha256:" + "ab" * 32,
            "versions": {"gsim": "0.5.0", "gmsh": "4.15.2"},
            "gmsh_options": {
                "Mesh.Algorithm": 5.0,
                "Mesh.Algorithm3D": 10.0,
                "General.NumThreads": 4.0,
                "Mesh.MaxNumThreads1D": 0.0,
                "Mesh.MaxNumThreads2D": 1.0,
                "Mesh.MaxNumThreads3D": 4.0,
            },
        },
        capsys,
    )
    assert "sha256:" + "ab" * 8 in out
    assert "ab" * 9 not in out, "only the first 16 hex digits are shown"
    assert "gmsh 4.15.2" in out
    assert "gsim 0.5.0" in out
    assert "3D algorithm 10" in out
    assert "threads 4" in out


def test_print_mesh_stats_without_a_hash_prints_nothing_extra(capsys) -> None:
    out = _print_stats({"elements": 10, "tetrahedra": 8}, capsys)
    assert "Mesh hash" not in out
    assert "Mesher" not in out


def _result(mesh_stats: dict) -> SimulationResult:
    return SimulationResult(
        mesh_path=Path("mesh.msh"), output_dir=Path("out"), mesh_stats=mesh_stats
    )


def test_simulation_result_summary_shows_the_hash_and_the_mesher() -> None:
    """``sim.mesh()`` returns a SimulationResult, whose summary notebooks display."""
    summary = str(
        _result(
            {
                "nodes": 5,
                "elements": 10,
                "tetrahedra": 8,
                "mesh_hash": "sha256:" + "cd" * 32,
                "versions": {"gsim": "0.5.0", "gmsh": "4.15.2"},
                "gmsh_options": {"Mesh.Algorithm3D": 1.0, "General.NumThreads": 1.0},
            }
        )
    )
    assert "Mesh hash:" in summary
    assert "sha256:" + "cd" * 8 in summary
    assert "3D algorithm 1" in summary
    assert "Mesh hash" not in str(_result({"nodes": 5, "elements": 10}))
