"""Internal ``*__None`` faces must not become absorbing (#417).

Two touching layers that are both conductor shells (a via layer with
``layer_type="conductor"``) leave the face they share in a ``<dielectric>__None``
group, with a volume on both sides. It must stay out of ``Absorbing``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import gmsh
import numpy as np
import pytest

from gsim.palace.mesh import generator, gmsh_utils
from gsim.palace.mesh.config_generator import write_config
from tests._electrode_layouts import MESH, component, physical_names, stack

LOGGER = "gsim.palace.mesh.gmsh_utils"
TOL = 1e-6

# m1 and m2 bars with a via on top of m1 and below m2 (um)
BOXES = [
    ("m1", 0, 0, 10, 4),
    ("v1", 4, 1, 6, 3),
    ("m2", 0, 0, 10, 4),
]


def _flat_at(tag: int, axis: int, value: float) -> bool:
    bb = gmsh.model.getBoundingBox(2, tag)
    return abs(bb[axis] - value) < TOL and abs(bb[axis + 3] - value) < TOL


def test_same_material_internal_face_is_not_labelled_none(kernel):
    lower = kernel.addBox(0, 0, 0, 1, 1, 1)
    upper = kernel.addBox(0, 0, 1, 1, 1, 1)
    pg_map = gmsh_utils.run_boolean_pipeline(
        [gmsh_utils.Entity("sio2", 3, 0, [lower, upper])]
    )
    shared = next(
        int(tag)
        for _, tag in gmsh.model.getEntities(2)
        if len(gmsh.model.getAdjacencies(2, tag)[0]) == 2
    )

    assert "sio2__None" in pg_map
    assert gmsh.model.getPhysicalGroupsForEntity(2, shared).size == 0


def _stacked_cubes(kernel) -> dict[str, list[int]]:
    """A big box with two unit cubes stacked inside it, fragmented.

    Returns the tags of the faces: ``outer`` (6, one volume), ``shared`` (the
    face between the cubes, two volumes) and ``bottom`` (the cube-to-big
    interface at z = 0, two volumes).
    """
    big = kernel.addBox(-5, -5, -5, 10, 10, 10)
    cube1 = kernel.addBox(0, 0, 0, 1, 1, 1)
    cube2 = kernel.addBox(0, 0, 1, 1, 1, 1)
    kernel.fragment([(3, big)], [(3, cube1), (3, cube2)])
    kernel.synchronize()

    faces: dict[str, list[int]] = {"outer": [], "shared": [], "bottom": []}
    for _, tag in gmsh.model.getEntities(2):
        if len(gmsh.model.getAdjacencies(2, tag)[0]) == 1:
            faces["outer"].append(tag)
        elif _flat_at(tag, 2, 1.0):
            faces["shared"].append(tag)
        elif _flat_at(tag, 2, 0.0):
            faces["bottom"].append(tag)
    assert len(faces["outer"]) == 6
    assert len(faces["shared"]) == 1
    assert len(faces["bottom"]) == 1
    return faces


def test_drops_only_internal_faces(kernel, caplog):
    """Internal faces leave their group; outer faces and other groups stay."""
    faces = _stacked_cubes(kernel)
    outer, shared, bottom = faces["outer"], faces["shared"][0], faces["bottom"][0]
    volumes = [tag for _, tag in gmsh.model.getEntities(3)]

    gmsh.model.addPhysicalGroup(2, outer[:5], tag=1, name="air__None")
    gmsh.model.addPhysicalGroup(2, [outer[5], bottom], tag=2, name="mix__None")
    gmsh.model.addPhysicalGroup(2, [shared], tag=3, name="sio2__None")
    # 3D groups whose tags coincide with 2D tags must not be mistaken for them
    gmsh.model.addPhysicalGroup(3, volumes, tag=3, name="sio2")
    gmsh.model.addPhysicalGroup(3, volumes, tag=7, name="ghost__None")
    pg_map = {
        "air__None": 1,
        "mix__None": 2,
        "sio2__None": 3,
        "sio2": 3,
        "other__None": 1,  # stale: tag 1 is the 2D group "air__None"
        "ghost__None": 7,  # tag 7 exists only in 3D
    }

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        dropped = gmsh_utils.drop_internal_none_surfaces(pg_map)

    assert dropped == {"sio2__None": [shared], "mix__None": [bottom]}
    assert "sio2__None" not in pg_map
    assert (2, 3) not in gmsh.model.getPhysicalGroups(2)
    assert (3, 3) in gmsh.model.getPhysicalGroups(3)

    assert pg_map["air__None"] == 1
    assert {int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(2, 1)} == set(
        outer[:5]
    )
    assert {
        int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(2, pg_map["mix__None"])
    } == {outer[5]}
    assert gmsh.model.getPhysicalName(2, pg_map["mix__None"]) == "mix__None"

    assert pg_map["sio2"] == 3
    assert pg_map["other__None"] == 1
    assert pg_map["ghost__None"] == 7

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    text = "\n".join(r.getMessage() for r in warnings)
    assert "1 of 1 surfaces of 'sio2__None'" in text
    assert "1 of 2 surfaces of 'mix__None'" in text
    assert "air__None" not in text


def test_outer_faces_only_change_nothing(kernel, caplog):
    """A group of outer faces is left alone and nothing is logged."""
    faces = _stacked_cubes(kernel)
    gmsh.model.addPhysicalGroup(2, faces["outer"], tag=1, name="air__None")
    pg_map = {"air__None": 1}

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        dropped = gmsh_utils.drop_internal_none_surfaces(pg_map)

    assert dropped == {}
    assert pg_map == {"air__None": 1}
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_face_on_the_domain_box_is_kept(kernel):
    """A face with no volume that lies on a box side stays; one inside goes."""
    kernel.addBox(-5, -5, -5, 10, 10, 10)
    on_box = kernel.addRectangle(-1, -1, 5, 2, 2)  # at z-max, no volume
    inside = kernel.addRectangle(-1, -1, 2, 2, 2)  # free, inside the box
    kernel.synchronize()
    assert gmsh.model.getAdjacencies(2, on_box)[0].size == 0

    gmsh.model.addPhysicalGroup(2, [on_box, inside], tag=1, name="air__None")
    pg_map = {"air__None": 1}

    dropped = gmsh_utils.drop_internal_none_surfaces(pg_map)

    assert dropped == {"air__None": [inside]}
    kept = gmsh.model.getEntitiesForPhysicalGroup(2, pg_map["air__None"])
    assert [int(t) for t in kept] == [on_box]


def test_shared_face_near_the_domain_box_is_dropped(kernel):
    """A face between two volumes is internal even within tol of a box side."""
    kernel.addBox(0, 0, 0, 1, 1, 1)
    kernel.addBox(0, 0, 1, 1, 1, 5e-4)  # thin top layer: z = 1 is 5e-4 from z-max
    kernel.fragment([(3, 1)], [(3, 2)])
    kernel.synchronize()
    shared = next(
        int(t)
        for _, t in gmsh.model.getEntities(2)
        if gmsh.model.getAdjacencies(2, t)[0].size == 2
    )
    top = next(
        int(t)
        for _, t in gmsh.model.getEntities(2)
        if gmsh.model.getAdjacencies(2, t)[0].size == 1
        and abs(gmsh.model.getBoundingBox(2, t)[2] - (1 + 5e-4)) < 1e-6
    )

    gmsh.model.addPhysicalGroup(2, [shared, top], tag=1, name="air__None")
    pg_map = {"air__None": 1}

    dropped = gmsh_utils.drop_internal_none_surfaces(pg_map)

    assert dropped == {"air__None": [shared]}
    kept = gmsh.model.getEntitiesForPhysicalGroup(2, pg_map["air__None"])
    assert [int(t) for t in kept] == [top]


def _msh_triangles(msh_path: Path, attributes: list[int]):
    """Node coordinates and triangles of the given tags (text MSH 2.2)."""
    lines = Path(msh_path).read_text().splitlines()
    start = lines.index("$Nodes")
    count = int(lines[start + 1])
    xyz = {}
    for line in lines[start + 2 : start + 2 + count]:
        node, x, y, z = line.split()
        xyz[int(node)] = (float(x), float(y), float(z))
    start = lines.index("$Elements")
    count = int(lines[start + 1])
    triangles = []
    for line in lines[start + 2 : start + 2 + count]:
        fields = [int(v) for v in line.split()]
        _, kind, ntags = fields[:3]
        if kind == 2 and fields[3] in attributes:
            triangles.append([xyz[n] for n in fields[3 + ntags :]])
    return np.array(list(xyz.values())), triangles


@pytest.mark.parametrize("via_kind", ["conductor", "via"])
def test_absorbing_only_on_the_outer_boundary(tmp_path, caplog, via_kind):
    """No Absorbing triangle lies inside the domain, also for shell vias."""
    st = stack()
    st.layers["v1"].layer_type = via_kind

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = generator.generate_mesh(
            component(BOXES, ports=[]),
            st,
            [],
            tmp_path,
            write_config=False,
            margin_x=5,
            margin_y=5,
            air_margin=0,
            airbox_margin_x=0,
            airbox_margin_y=0,
            airbox_z_above=5,
            airbox_z_below=0,
            **MESH,
        )
    config = json.loads(
        write_config(result, st, [], simulation_type="eigenmode").read_text()
    )

    attributes = config["Boundaries"]["Absorbing"]["Attributes"]
    assert attributes == result.groups["boundary_surfaces"]["absorbing"]["phys_group"]
    names = physical_names(result.mesh_path)
    assert {names[a] for a in attributes} == {"air__None", "sio2__None"}

    nodes, triangles = _msh_triangles(result.mesh_path, attributes)
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    assert triangles
    inside = [
        tri
        for tri in triangles
        if not any(
            all(abs(p[axis] - side) < TOL for p in tri)
            for axis in range(3)
            for side in (low[axis], high[axis])
        )
    ]
    assert inside == []

    assert not [r for r in caplog.records if "are internal" in r.getMessage()]
