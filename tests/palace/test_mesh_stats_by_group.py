"""Tests for the per-physical-group breakdown in collect_mesh_stats."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gmsh
import pytest

from gsim.palace.base import PalaceSimMixin
from gsim.palace.mesh.config_generator import collect_mesh_stats
from gsim.palace.mesh.generator import MeshResult

FINE = 0.05
COARSE = 0.1


def _initialize_gmsh() -> None:
    """Start gmsh with the mesh-size options these tests rely on.

    ``MeshSizeFromPoints`` and ``MeshSizeExtendFromBoundary`` are both gmsh
    defaults, and are set explicitly because a developer's ``~/.gmsh-options``
    can turn them off. With ``MeshSizeFromPoints`` at 0 the ``setSize`` calls
    below are ignored outright, every box meshes to the same default size, and
    the element-count assertions fail on a 1:1 ratio for a reason that has
    nothing to do with the code under test.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)


@pytest.fixture(scope="module")
def two_box_mesh() -> dict:
    """Two equal, disjoint unit boxes meshed with element sizes in a 1:2 ratio.

    The coarse box is 10 elements across, so both boxes are past the
    pre-asymptotic regime where a box only a few elements wide cannot follow
    the volumetric scaling.
    """
    _initialize_gmsh()
    try:
        gmsh.model.add("two_boxes")
        fine = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        coarse = gmsh.model.occ.addBox(3, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()

        for box, size in ((fine, FINE), (coarse, COARSE)):
            points = gmsh.model.getBoundary([(3, box)], recursive=True)
            gmsh.model.mesh.setSize(points, size)

        gmsh.model.addPhysicalGroup(3, [fine], name="fine")
        gmsh.model.addPhysicalGroup(3, [coarse], name="coarse")
        fine_faces = [t for _, t in gmsh.model.getBoundary([(3, fine)], oriented=False)]
        gmsh.model.addPhysicalGroup(2, fine_faces, name="fine_skin")

        gmsh.model.mesh.generate(3)
        stats = collect_mesh_stats()
    finally:
        gmsh.finalize()
    return stats


def _group(stats: dict, kind: str, name: str) -> dict:
    return next(g for g in stats["groups"][kind] if g["name"] == name)


def test_volume_groups_account_for_every_tetrahedron(two_box_mesh) -> None:
    stats = two_box_mesh
    counted = sum(g["elements"] for g in stats["groups"]["volumes"])
    assert counted == stats["tetrahedra"]


def test_finer_region_holds_more_elements(two_box_mesh) -> None:
    """Element count scales as V / h**3, so halving h gives ~8x more tets.

    A bound of 6 separates that volumetric scaling from a surface-like h**-2
    one, which would give 4, and fails if the counts land on the wrong group.
    """
    fine = _group(two_box_mesh, "volumes", "fine")
    coarse = _group(two_box_mesh, "volumes", "coarse")
    assert fine["elements"] > 6 * coarse["elements"]
    assert fine["edge_length"]["min"] < coarse["edge_length"]["min"]
    assert 0 < fine["edge_length"]["min"] <= fine["edge_length"]["max"]


def test_surface_groups_report_their_triangles(two_box_mesh) -> None:
    skin = _group(two_box_mesh, "surfaces", "fine_skin")
    assert skin["elements"] > 0
    assert 0 < skin["edge_length"]["min"] <= skin["edge_length"]["max"]


def _print_stats(mesh_stats: dict, capsys) -> str:
    """Run PalaceSimMixin.print_mesh_stats on a mesh result holding mesh_stats."""
    sim = SimpleNamespace(
        _last_mesh_result=MeshResult(mesh_path=Path("mesh.msh"), mesh_stats=mesh_stats)
    )
    PalaceSimMixin.print_mesh_stats(sim)
    return capsys.readouterr().out


def test_print_mesh_stats_lists_volume_regions_largest_first(capsys) -> None:
    out = _print_stats(
        {
            "elements": 100,
            "tetrahedra": 80,
            "groups": {
                "volumes": [
                    {
                        "name": "sio2",
                        "tag": 1,
                        "elements": 20,
                        "edge_length": {"min": 0.012, "max": 1.5},
                    },
                    {
                        "name": "air",
                        "tag": 2,
                        "elements": 60,
                        "edge_length": {"min": 0.5, "max": 40.0},
                    },
                ],
                "surfaces": [{"name": "metal_xy", "tag": 3, "elements": 20}],
            },
        },
        capsys,
    )
    assert "Elements by region" in out
    assert out.index("air") < out.index("sio2")
    assert "75.0%" in out
    assert "25.0%" in out
    assert "0.012" in out
    assert "metal_xy" not in out


def test_print_mesh_stats_falls_back_to_surfaces_for_2d_meshes(capsys) -> None:
    out = _print_stats(
        {
            "elements": 30,
            "groups": {
                "volumes": [],
                "surfaces": [
                    {"name": "substrate", "tag": 1, "elements": 10},
                    {"name": "oxide", "tag": 2, "elements": 20},
                ],
            },
        },
        capsys,
    )
    assert out.index("oxide") < out.index("substrate")
    assert "66.7%" in out


def test_print_mesh_stats_skips_the_region_table_when_no_group_has_elements(
    capsys,
) -> None:
    """Empty groups give nothing to tabulate, and no share of a zero total."""
    out = _print_stats(
        {
            "elements": 10,
            "tetrahedra": 8,
            "groups": {
                "volumes": [{"name": "air", "tag": 1, "elements": 0}],
                "surfaces": [],
            },
        },
        capsys,
    )
    assert "Tetrahedra: 8" in out
    assert "Elements by region" not in out


def test_empty_group_does_not_hide_the_others() -> None:
    """A group with no elements reports 0 and the other groups still appear."""
    _initialize_gmsh()
    try:
        meshed = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(
            gmsh.model.getBoundary([(3, meshed)], recursive=True), 0.5
        )
        gmsh.model.addPhysicalGroup(3, [meshed], name="meshed")
        gmsh.model.mesh.generate(3)
        # An entity added after meshing has no elements.
        unmeshed = gmsh.model.occ.addBox(3, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [unmeshed], name="unmeshed")
        stats = collect_mesh_stats()
    finally:
        gmsh.finalize()
    assert _group(stats, "volumes", "unmeshed")["elements"] == 0
    assert _group(stats, "volumes", "meshed")["elements"] == stats["tetrahedra"]
