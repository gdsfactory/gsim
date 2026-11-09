"""Which Contact claims a node of another Region at a Contact's end.

Two Contacts meeting at one point, on two doped Regions, with an oxide
wedge reaching the same point between their electrodes. The oxide's node
there may answer to one Contact only, or DEVSIM would get two boundary
conditions for it.
"""

from __future__ import annotations

import meshio
import numpy as np

from gsim.tcad.device import _write_contact_ends

POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),  # where both Contacts and the oxide meet
        (2.0, 1.0, 0.0),
        (0.5, 2.0, 0.0),
        (1.5, 2.0, 0.0),
    ]
)
SHARED = 4
TAGS = {"n_pad": 1, "p_pad": 2, "sio2": 3, "cathode": 4, "anode": 5}


def _write(path):
    triangles = [(0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4), (4, 7, 6)]
    regions = [TAGS["n_pad"]] * 2 + [TAGS["p_pad"]] * 2 + [TAGS["sio2"]]
    lines = [(3, 4), (4, 5)]
    line_tags = [TAGS["cathode"], TAGS["anode"]]
    physical = [np.array(line_tags), np.array(regions)]
    meshio.write(
        str(path),
        meshio.Mesh(
            POINTS,
            [("line", np.array(lines)), ("triangle", np.array(triangles))],
            cell_data={"gmsh:physical": physical, "gmsh:geometrical": physical},
            field_data={
                name: np.array([tag, 2 if tag <= 3 else 1])
                for name, tag in TAGS.items()
            },
        ),
        file_format="gmsh22",
        binary=False,
    )
    return path


def _claims(tmp_path, contact_regions):
    src = _write(tmp_path / "wedge.msh")
    ends = _write_contact_ends(
        src, tmp_path / "ends.msh", contact_regions, ["n_pad", "p_pad", "sio2"]
    )
    return {
        (contact, end.region, end.point)
        for contact, contact_ends in ends.items()
        for end in contact_ends
    }


def test_a_node_two_contacts_reach_answers_to_the_first_declared(tmp_path):
    point = tuple(POINTS[SHARED, :2])
    # Each Contact keeps its own Region's node; the oxide's goes to one.
    assert _claims(tmp_path, {"cathode": "n_pad", "anode": "p_pad"}) == {
        ("cathode", "sio2", point)
    }
    assert _claims(tmp_path, {"anode": "p_pad", "cathode": "n_pad"}) == {
        ("anode", "sio2", point)
    }


def test_the_written_mesh_has_one_point_group_at_the_node(tmp_path):
    _claims(tmp_path, {"cathode": "n_pad", "anode": "p_pad"})
    mesh = meshio.read(str(tmp_path / "ends.msh"))
    vertices = [block.data[:, 0] for block in mesh.cells if block.type == "vertex"]
    assert np.concatenate(vertices).tolist() == [SHARED]
