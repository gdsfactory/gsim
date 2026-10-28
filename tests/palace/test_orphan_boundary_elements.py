"""A boundary element must be a face of a meshed element.

Palace (through MFEM) looks up the element behind every boundary element.
A boundary element that is no edge of any triangle (2D) or no face of any
tetrahedron (3D) has no element to find, and Palace dies with a segmentation
fault (exit code 139) before it writes anything.

In a native-2D boundary-mode mesh a conductor is a hole. When a conductor
reaches the outer wall of the domain, the piece of the wall it covers has no
triangle next to it, so it must not be written as part of the outer (absorbing
or PEC) boundary. The tests mesh one generic cross-section (a signal strip
and a ground strip on one metal of a three-layer generic stack, no solver)
with the strips on the walls and a few micrometres off them.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import gdsfactory as gf
import meshio
import numpy as np
import pytest

from gsim.palace import BoundaryModeSim, DrivenSim
from tests._electrode_layouts import LAYERS, MESH, component, stack

# Signal and ground strip on m1 (um), long along x. The section x = 0 cuts
# both; the domain wall in y sits on their outer faces when the margin is 0.
SIGNAL = ("m1", -20, 0, 20, 4)
GROUND = ("m1", -20, 8, 20, 12)
M1_THICKNESS = LAYERS["m1"][2] - LAYERS["m1"][1]


def boundary_elements_without_parent(
    msh_path: Path,
) -> list[tuple[int, tuple[int, ...]]]:
    """``(physical tag, node tuple)`` of boundary elements that bound no element.

    In a 2D mesh the boundary elements are lines and the elements triangles; in a
    3D mesh triangles and tetrahedra. Reads the file Palace gets.
    """
    mesh = meshio.read(str(msh_path), file_format="gmsh")
    kinds = {block.type for block in mesh.cells}
    boundary, parent = (
        ("triangle", "tetra") if "tetra" in kinds else ("line", "triangle")
    )
    size = 3 if parent == "tetra" else 2

    faces = {
        tuple(sorted(face))
        for block in mesh.cells
        if block.type == parent
        for element in block.data
        for face in itertools.combinations(element, size)
    }
    orphans = []
    for block, tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True):
        if block.type != boundary:
            continue
        for nodes, tag in zip(block.data, tags, strict=True):
            if tuple(sorted(nodes)) not in faces:
                orphans.append((int(tag), tuple(int(n) for n in nodes)))
    return orphans


def write_msh(path: Path, points, lines, triangles) -> Path:
    """A tiny 2D mesh with physical tags ``1`` (triangles) and ``7`` (lines)."""
    cells = [("triangle", np.array(triangles)), ("line", np.array(lines))]
    mesh = meshio.Mesh(
        np.asarray(points, dtype=float),
        cells,
        cell_data={
            "gmsh:physical": [
                np.full(len(c), t)
                for c, t in zip((triangles, lines), (1, 7), strict=True)
            ],
            "gmsh:geometrical": [
                np.full(len(c), t)
                for c, t in zip((triangles, lines), (1, 7), strict=True)
            ],
        },
    )
    meshio.write(path, mesh, file_format="gmsh22", binary=False)
    return path


POINTS = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0)]
TRIANGLES = [(0, 1, 2), (0, 2, 3)]


class TestTheDetector:
    """The check itself, on meshes written by hand."""

    def test_a_line_on_a_triangle_edge_is_fine(self, tmp_path):
        path = write_msh(tmp_path / "ok.msh", POINTS, [(0, 1), (2, 3)], TRIANGLES)

        assert boundary_elements_without_parent(path) == []

    def test_a_line_that_is_no_triangle_edge_is_found(self, tmp_path):
        # node 4 is not part of any triangle; (1, 4) borders none
        path = write_msh(tmp_path / "orphan.msh", POINTS, [(0, 1), (1, 4)], TRIANGLES)

        assert boundary_elements_without_parent(path) == [(7, (1, 4))]


def mesh_section(tmp_path: Path, margin_y: float) -> BoundaryModeSim:
    """Mesh the strips' cross-section with *margin_y* um of dielectric to the walls."""
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path / f"section-{margin_y:g}"))
    sim.set_geometry(component([SIGNAL, GROUND], ports=[]))
    sim.set_stack(stack())
    sim.set_airbox(margin_x=0.0, margin_y=margin_y, z_above=5.0, z_below=2.0)
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=10e9, num_modes=2, save=0)
    sim.metallic_boundaries = True
    sim.mesh(preset="coarse", verbose=False, **MESH)
    return sim


def outer_wall_length(sim: BoundaryModeSim, side: str) -> float:
    """Total length (um) of the outer-boundary edges on one wall of the section."""
    mesh = meshio.read(str(sim.mesh_path), file_format="gmsh")
    ((h0, h1), (v0, v1)) = sim.mesh_extent
    outer = sim.mesh_groups["boundary_surfaces"]["absorbing"]["phys_group"]
    outer = outer if isinstance(outer, list) else [outer]
    wall = {"left": (0, h0), "right": (0, h1), "bottom": (1, v0), "top": (1, v1)}[side]
    total = 0.0
    for block, tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True):
        if block.type != "line":
            continue
        for nodes, tag in zip(block.data, tags, strict=True):
            if int(tag) not in outer:
                continue
            p, q = mesh.points[nodes][:, :2]
            if abs(p[wall[0]] - wall[1]) < 1e-6 and abs(q[wall[0]] - wall[1]) < 1e-6:
                total += float(np.hypot(*(q - p)))
    return total


@pytest.fixture(scope="module")
def on_wall(tmp_path_factory):
    """Both strips' outer faces lie on the domain walls."""
    return mesh_section(tmp_path_factory.mktemp("orphans"), margin_y=0.0)


@pytest.fixture(scope="module")
def off_wall(tmp_path_factory):
    """The same section with 5 um of dielectric between the strips and the walls."""
    return mesh_section(tmp_path_factory.mktemp("orphans"), margin_y=5.0)


class TestNative2DSection:
    def test_the_domain_walls_sit_on_the_strips(self, on_wall):
        ((h0, h1), _) = on_wall.mesh_extent

        assert (h0, h1) == pytest.approx((0.0, 12.0))

    def test_no_boundary_element_is_left_without_a_triangle(self, on_wall):
        assert boundary_elements_without_parent(on_wall.mesh_path) == []

    def test_the_control_with_a_margin_has_none_either(self, off_wall):
        assert boundary_elements_without_parent(off_wall.mesh_path) == []

    def test_the_wall_under_a_strip_is_not_part_of_the_outer_boundary(
        self, on_wall, off_wall
    ):
        ((_, _), (v0, v1)) = on_wall.mesh_extent
        full_height = v1 - v0

        for side in ("left", "right"):
            assert outer_wall_length(on_wall, side) == pytest.approx(
                full_height - M1_THICKNESS
            )
        # with a margin the whole wall is dielectric
        ((_, _), (w0, w1)) = off_wall.mesh_extent
        for side in ("left", "right"):
            assert outer_wall_length(off_wall, side) == pytest.approx(w1 - w0)

    def test_the_other_walls_are_untouched(self, on_wall):
        ((h0, h1), _) = on_wall.mesh_extent

        for side in ("top", "bottom"):
            assert outer_wall_length(on_wall, side) == pytest.approx(h1 - h0)

    def test_the_conductor_edges_are_all_still_there(self, on_wall):
        """Only the wall piece goes: the strips keep their boundary condition."""
        conductors = (
            on_wall.mesh_groups["pec_surfaces"]
            | on_wall.mesh_groups["conductor_surfaces"]
        )
        assert set(conductors) == {"m1"}

        mesh = meshio.read(str(on_wall.mesh_path), file_format="gmsh")
        tag = conductors["m1"]["phys_group"]
        length = 0.0
        for block, tags in zip(
            mesh.cells, mesh.cell_data["gmsh:physical"], strict=True
        ):
            if block.type != "line":
                continue
            for nodes, t in zip(block.data, tags, strict=True):
                if int(t) == tag:
                    p, q = mesh.points[nodes][:, :2]
                    length += float(np.hypot(*(q - p)))
        # two strips of 4 x M1_THICKNESS (um); each one's face on a wall is not
        # an edge of any triangle and is not part of its boundary
        strip_width = SIGNAL[4] - SIGNAL[2]
        perimeter = 2 * (strip_width + M1_THICKNESS)
        assert length == pytest.approx(2 * (perimeter - M1_THICKNESS))


@pytest.mark.parametrize("planar_conductors", [False, True])
def test_a_driven_3d_mesh_has_no_orphan_either(tmp_path, planar_conductors):
    """A ground strip on the side wall of the 3D domain (lumped ports)."""
    gf.gpdk.PDK.activate()
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "driven"))
    sim.set_geometry(
        component([SIGNAL, GROUND], ports=[("P1", -19, 2, "m1"), ("P2", 19, 2, "m1")])
    )
    sim.set_stack(stack())
    sim.set_airbox(margin_x=0.0, margin_y=0.0, z_above=5.0, z_below=2.0)
    sim.add_port("P1", layer="m1", length=2.0)
    sim.add_port("P2", layer="m1", length=2.0, excited=False)
    sim.set_driven(fmin=1e9, fmax=10e9, num_points=2)
    sim.mesh(
        preset="coarse",
        planar_conductors=planar_conductors,
        margin_x=0,
        margin_y=0,
        **MESH,
    )

    assert boundary_elements_without_parent(sim.mesh_path) == []
