"""A graded mesh grows away from the features it grades from at a set rate."""

from __future__ import annotations

import meshio
import numpy as np
import pytest

from gsim.tcad import MeshGrading, Simulation

from .conftest import build_contacted_diode

#: Size near the graded features and the cap far from them (um).
REFINED = 0.02
MAX = 1.0
#: Growth of the element size per um of distance from the graded features.
RATE = 0.25

#: The diode's metallurgical junction, a vertical segment (h, z0, z1) in um.
JUNCTION = (-20.0, 0.0, 0.22)
#: The two electrodes, (h_min, h_max, z_min, z_max) in um.
ELECTRODES = ((-20.6, -20.3, 0.22, 0.72), (-19.7, -19.4, 0.22, 0.72))


def _mesh(tmp_path, **solver_settings):
    comp, stack = build_contacted_diode()
    sim = Simulation()
    sim.geometry(comp, stack=stack)
    sim.domain(
        plane="x=0",
        window=(-21.1, -18.9),
        margin_x=2.0,
        margin_y=2.0,
        z_above=1.5,
        z_below=1.0,
    )
    sim.contacts.add(name="cathode", semiconductor="n_pad", electrode="cathode_metal")
    sim.contacts.add(name="anode", semiconductor="p_pad", electrode="anode_metal")
    sim.interfaces.add(name="junction", region_a="n_rib", region_b="p_rib")
    sim.solver(
        mesh_preset="coarse",
        refined_mesh_size_um=REFINED,
        max_mesh_size_um=MAX,
        **solver_settings,
    )
    mesh = sim.mesh(output_dir=tmp_path)
    return meshio.read(str(mesh.mesh_path))


def _triangles(mesh, region: str | None = None):
    """Triangle corners ``(n, 3, 2)``, optionally of one named region."""
    tag = mesh.field_data[region][0] if region else None
    blocks = []
    for block, phys in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True):
        if block.type != "triangle":
            continue
        keep = np.ones(len(block.data), bool) if tag is None else phys == tag
        blocks.append(block.data[keep])
    return mesh.points[np.vstack(blocks)][..., :2]


def _distance_to_rectangle(p, h0, h1, z0, z1):
    dh = np.maximum(np.maximum(h0 - p[..., 0], p[..., 0] - h1), 0.0)
    dz = np.maximum(np.maximum(z0 - p[..., 1], p[..., 1] - z1), 0.0)
    return np.hypot(dh, dz)


def _distance_to_features(p):
    """Distance (um) to the junction or the nearest electrode."""
    h, z0, z1 = JUNCTION
    distance = _distance_to_rectangle(p, h, h, z0, z1)
    for box in ELECTRODES:
        distance = np.minimum(distance, _distance_to_rectangle(p, *box))
    return distance


def _worst_size_ratio(corners) -> float:
    """Largest ratio of an element's longest edge to the graded size.

    The graded size is ``REFINED + RATE * d`` capped at ``MAX``, with ``d``
    the distance of the element's centroid from the graded features.
    """
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=-1)
    distance = _distance_to_features(corners.mean(axis=1))
    target = np.minimum(MAX, REFINED + RATE * distance)
    return float((edges.max(axis=1) / target).max())


def _obtuse_fraction(corners) -> float:
    a = corners[:, 1] - corners[:, 0]
    b = corners[:, 2] - corners[:, 1]
    c = corners[:, 0] - corners[:, 2]
    dots = np.stack(
        [
            -np.einsum("ij,ij->i", a, c),
            -np.einsum("ij,ij->i", b, a),
            -np.einsum("ij,ij->i", c, b),
        ]
    )
    return float((dots < 0).any(axis=0).mean())


GRADING = MeshGrading(growth_per_um=RATE, sources=["conductors", "junctions"])


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    root = tmp_path_factory.mktemp("grading")
    return {
        "lines": _mesh(root / "lines"),
        "graded": _mesh(root / "graded", mesh_grading=GRADING),
    }


def test_element_size_grows_at_most_at_the_set_rate(meshes):
    # gmsh takes a size as a target: an edge runs up to about twice it.
    assert _worst_size_ratio(_triangles(meshes["graded"])) < 2.0
    # Without grading the size grows about 2 per um from every internal
    # curve, eight times the rate, so elements a few tenths of a micron
    # from the junction are several times the graded size.
    assert _worst_size_ratio(_triangles(meshes["lines"])) > 3.0


def test_grading_lowers_the_obtuse_fraction_in_the_oxide(meshes):
    lines = _obtuse_fraction(_triangles(meshes["lines"], "sio2"))
    graded = _obtuse_fraction(_triangles(meshes["graded"], "sio2"))

    assert graded < 0.5 * lines


def test_grading_is_a_mesh_setting():
    """Grading changes the mesh, so it is one of the solver's mesh fields."""
    from gsim.tcad.api import MESH_FIELDS

    assert "mesh_grading" in MESH_FIELDS


#: A refinement box around the junction: (h_min, h_max, z_min, z_max, size).
BOX = (-20.15, -19.85, 0.0, 0.22, 0.01)


def _worst_ratio_outside_box(corners) -> float:
    """Largest edge over ``size + RATE * d``, ``d`` the distance to BOX."""
    h0, h1, z0, z1, size = BOX
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=-1)
    distance = _distance_to_rectangle(corners.mean(axis=1), h0, h1, z0, z1)
    target = np.minimum(MAX, size + RATE * distance)
    return float((edges.max(axis=1) / target).max())


def test_a_graded_refinement_box_grows_at_the_set_rate(tmp_path):
    # A box alone is a step: its size inside, then at once the size of the
    # nearest curve. Graded, it holds its size inside and grows outside.
    step = _mesh(
        tmp_path / "step",
        refinement_boxes=[BOX],
        mesh_grading=MeshGrading(growth_per_um=RATE, sources=["conductors"]),
    )
    graded = _mesh(
        tmp_path / "graded",
        refinement_boxes=[BOX],
        mesh_grading=MeshGrading(growth_per_um=RATE, sources=["fine_regions"]),
    )

    assert _worst_ratio_outside_box(_triangles(graded)) < 2.0
    assert _worst_ratio_outside_box(_triangles(step)) > 3.0


class TestTheGradingSetting:
    def test_needs_a_positive_rate_and_a_source(self):
        with pytest.raises(ValueError, match="growth_per_um"):
            MeshGrading(growth_per_um=0.0)
        with pytest.raises(ValueError, match="sources"):
            MeshGrading(growth_per_um=RATE, sources=[])
        with pytest.raises(ValueError, match="sources"):
            MeshGrading(growth_per_um=RATE, sources=["vias"])
        with pytest.raises(ValueError, match="repeat"):
            MeshGrading(growth_per_um=RATE, sources=["junctions", "junctions"])

    def test_grades_from_every_feature_by_default(self):
        assert set(MeshGrading(growth_per_um=RATE).sources) == {
            "conductors",
            "junctions",
            "fine_regions",
        }

    def test_is_off_by_default(self):
        from gsim.palace.models import MeshConfig
        from gsim.tcad import Solver

        assert Solver().mesh_grading is None
        assert MeshConfig().mesh_grading is None
