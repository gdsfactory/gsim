"""Execute the generated viewer's geometry code without a browser or solver."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from gsim.fdtd.viz import plot_mesh

_PORT_MESH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
3 2 "core"
2 3 "port_o1"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1000 0 0
3 0 1000 0
4 0 0 1000
$EndNodes
$Elements
2
1 2 2 3 1 1 3 2
2 4 2 2 1 1 2 3 4
$EndElements
"""


def _run_viewer_code(html: str, check_script: str) -> dict:
    """Run functions and embedded inputs from the generated viewer artifact."""
    node_command = shutil.which("node")
    if node_command is None:
        pytest.skip("Viewer JavaScript geometry checks require Node.js")
    functions = re.findall(
        r"^      function .*?^      }", html, re.MULTILINE | re.DOTALL
    )
    embedded_data = re.findall(
        r"^      const (?:meshText|options) = .*;$", html, re.MULTILINE
    )
    script = "\n".join(
        [
            "const NM_TO_UM = 1e-3;",
            *embedded_data,
            *functions,
            check_script,
        ]
    )
    result = subprocess.run(  # noqa: S603 - fixed local JavaScript from viewer artifact
        [node_command, "-"], input=script, capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def _viewer_surfaces(html: str) -> dict:
    return _run_viewer_code(
        html,
        """
const mesh = parseGmsh(meshText);
const surfaces = surfaceGroups(mesh);
console.log(JSON.stringify(Object.fromEntries(
    [...surfaces].map(([group, nodes]) => [group, {
        nodes,
        visible: shouldShow(mesh, group),
    }]),
)));
""",
    )


@pytest.mark.parametrize("mode", ["surface", "mesh"])
@pytest.mark.parametrize(
    "visibility",
    [
        {"hide_groups": ["port_o1"]},
        {"hide_groups": ["2_3"]},
        {"show_groups": ["core"]},
    ],
)
def test_hidden_port_keeps_material_closed(tmp_path: Path, mode, visibility) -> None:
    mesh_path = tmp_path / "port.msh"
    mesh_path.write_text(_PORT_MESH, encoding="utf8")
    surfaces = _viewer_surfaces(plot_mesh(mesh_path, mode=mode, **visibility).html)

    assert surfaces["3_2"]["visible"]
    assert not surfaces["2_3"]["visible"]
    nodes = surfaces["3_2"]["nodes"]
    # Each tetrahedron face appears once, including the face under the hidden port.
    faces = [nodes[index : index + 3] for index in range(0, len(nodes), 3)]
    assert Counter(tuple(sorted(face)) for face in faces) == Counter(
        {(1, 2, 3): 1, (1, 2, 4): 1, (1, 3, 4): 1, (2, 3, 4): 1}
    )
    # Independently check outward normals against this tetrahedron's centroid.
    coordinates = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1)}
    for face in faces:
        a, b, c = (coordinates[tag] for tag in face)
        ab = [b[index] - a[index] for index in range(3)]
        ac = [c[index] - a[index] for index in range(3)]
        normal = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        assert sum(normal[index] * (0.25 - a[index]) for index in range(3)) < 0
    assert surfaces["2_3"]["nodes"] == [1, 3, 2]


def test_visible_port_keeps_material_face_underneath(tmp_path: Path) -> None:
    mesh_path = tmp_path / "port.msh"
    mesh_path.write_text(_PORT_MESH, encoding="utf8")
    surfaces = _viewer_surfaces(plot_mesh(mesh_path).html)

    assert surfaces["2_3"]["visible"]
    assert surfaces["3_2"]["visible"]
    assert len(surfaces["3_2"]["nodes"]) == 12
    assert surfaces["2_3"]["nodes"] == [1, 3, 2]


def test_port_does_not_expose_internal_material_faces(tmp_path: Path) -> None:
    mesh_path = tmp_path / "port.msh"
    mesh_text = _PORT_MESH.replace("$Nodes\n4", "$Nodes\n5").replace(
        "$EndNodes", "5 0 0 -1000\n$EndNodes"
    )
    mesh_text = mesh_text.replace("$Elements\n2", "$Elements\n3").replace(
        "$EndElements", "3 4 2 2 1 1 3 2 5\n$EndElements"
    )
    mesh_path.write_text(mesh_text, encoding="utf8")
    surfaces = _viewer_surfaces(plot_mesh(mesh_path, hide_groups=["port_o1"]).html)
    nodes = surfaces["3_2"]["nodes"]

    assert len(nodes) == 18
    faces = [set(nodes[index : index + 3]) for index in range(0, len(nodes), 3)]
    assert {1, 2, 3} not in faces


def test_non_port_surface_keeps_existing_ownership(tmp_path: Path) -> None:
    mesh_path = tmp_path / "wall.msh"
    mesh_path.write_text(_PORT_MESH.replace("port_o1", "wall"), encoding="utf8")
    surfaces = _viewer_surfaces(plot_mesh(mesh_path).html)

    assert len(surfaces["3_2"]["nodes"]) == 9
    assert surfaces["2_3"]["nodes"] == [1, 3, 2]


def test_coplanar_groups_have_distinct_depths_and_keep_coordinates(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "coplanar.msh"
    mesh_text = _PORT_MESH.replace("$PhysicalNames\n2", "$PhysicalNames\n3").replace(
        "$EndPhysicalNames", '3 4 "slab"\n$EndPhysicalNames'
    )
    mesh_text = mesh_text.replace("$Nodes\n4", "$Nodes\n5").replace(
        "$EndNodes", "5 0 0 2000\n$EndNodes"
    )
    mesh_text = mesh_text.replace("$Elements\n2", "$Elements\n3").replace(
        "$EndElements", "3 4 2 4 1 1 2 3 5\n$EndElements"
    )
    mesh_path.write_text(mesh_text, encoding="utf8")
    result = _run_viewer_code(
        plot_mesh(mesh_path).html,
        """
// Capture the generated scene's geometry and materials without a WebGL context.
class Group {
    children = [];
    add(child) { this.children.push(child); }
}
class BufferGeometry {
    attributes = {};
    setAttribute(name, attribute) { this.attributes[name] = attribute; }
    dispose() {}
}
class BufferAttribute {
    constructor(array) { this.array = array; }
}
class MeshStandardMaterial {
    constructor(parameters) { Object.assign(this, parameters); }
}
class Mesh {
    constructor(geometry, material) {
        this.geometry = geometry;
        this.material = material;
    }
}
const THREE = {
    Group, BufferGeometry, BufferAttribute, MeshStandardMaterial, Mesh,
    Color: class {}, FrontSide: 0, DoubleSide: 2,
};
const toCreasedNormals = (geometry) => geometry;
const isSlice = false;
const showEdges = false;
const physicalGroupColors = new Map();
const physicalGroupPalette = [0x58c7a3, 0xff7a7a];
const largestGroupColor = 0xb8b2b0;
const groupRoots = new Map();
const mesh = parseGmsh(meshText);
const surfaces = surfaceGroups(mesh);
const currentLargestGroup = largestMaterialGroup(mesh, surfaces);
const root = new Group();
for (const [group, tags] of orderedGroupEntries(mesh, surfaces)) {
    addRenderable(root, mesh, group, positionsForNodeTags(tags, mesh.nodes),
        null, group === currentLargestGroup);
}
console.log(JSON.stringify(Object.fromEntries([...groupRoots].map(([group, item]) => {
    const face = item.children[0];
    return [group, {
        offsetEnabled: face.material.polygonOffset,
        factor: face.material.polygonOffsetFactor,
        units: face.material.polygonOffsetUnits,
        coordinates: [...face.geometry.attributes.position.array],
    }];
}))));
""",
    )

    core, slab, port = (result[group] for group in ("3_2", "3_4", "2_3"))
    for parameter in ("factor", "units"):
        # Unbiased mesh edges stay in front of every filled surface.
        assert 0 < port[parameter] < core[parameter] < slab[parameter]
    assert all(item["offsetEnabled"] for item in result.values())
    # The common z=0 face stays at its physical coordinates in both materials.
    port_coordinates = port["coordinates"]
    assert port_coordinates == [0, 0, 0, 0, 1, 0, 1, 0, 0]
    for material in (core, slab):
        triangles = [
            material["coordinates"][index : index + 9]
            for index in range(0, len(material["coordinates"]), 9)
        ]
        assert port_coordinates in triangles
