"""Check geometric invariants against the real OpenCASCADE kernel."""

import gmsh
import numpy as np
import pytest

from gsim.palace.mesh import gmsh_utils as geometry


@pytest.fixture
def kernel():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("primitives")
    try:
        yield gmsh.model.occ
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("meshseed", [0, 0.5])
def test_box_volume_surface_queries_and_groups_preserve_dimensions(kernel, meshseed):
    tag = geometry.create_box(kernel, 1, 2, 3, 3, 5, 7, meshseed=meshseed)
    kernel.synchronize()
    assert kernel.getMass(3, tag) == pytest.approx(24)
    assert geometry.get_volumes_at_z_range(3, 7) == [(3, tag)]
    assert geometry.get_volumes_at_z_range(0, 1) == []
    bottom = geometry.get_surfaces_at_z(3)
    assert len(bottom) == 1
    assert len(geometry.get_boundary_lines(bottom[0][1], kernel)) == 4
    faces = gmsh.model.getBoundary([(3, tag)], oriented=False)
    assert sum(geometry.is_vertical_surface(face) for _, face in faces) == 4
    group = geometry.assign_physical_group(3, [tag], "dielectric")
    assert gmsh.model.getPhysicalName(3, group) == "dielectric"
    assert list(gmsh.model.getEntitiesForPhysicalGroup(3, group)) == [tag]
    assert geometry.assign_physical_group(3, [], "empty") == -1


@pytest.mark.parametrize("height", [0, 2])
@pytest.mark.parametrize("hole", [False, True])
def test_polygon_extrusion_preserves_area_and_holes(kernel, height, hole):
    tag = geometry.extrude_polygon(
        kernel,
        [0, 4, 4, 0],
        [0, 0, 3, 3],
        zmin=1,
        thickness=height,
        holes=[([1, 2, 2, 1], [1, 1, 2, 2])] if hole else None,
    )
    kernel.synchronize()
    assert kernel.getMass(3 if height else 2, tag) == pytest.approx(
        (12 - int(hole)) * (height or 1)
    )
    assert geometry.extrude_polygon(kernel, [0, 1], [0, 0], 0, height) is None


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_port_rectangles_keep_area_and_orientation(kernel, axis):
    extent = [2, 2, 2]
    extent[axis] = 0
    tag = geometry.create_port_rectangle(kernel, 0, 0, 0, *extent)
    kernel.synchronize()
    assert kernel.getMass(2, tag) == pytest.approx(4)
    normal = np.zeros(3)
    normal[axis] = 1
    np.testing.assert_allclose(np.abs(geometry.get_surface_normal(tag)), normal)
    assert geometry.is_vertical_surface(tag) == (axis != 2)


def test_fragment_mapping_retains_each_original_volume(kernel):
    left = kernel.addBox(0, 0, 0, 2, 1, 1)
    right = kernel.addBox(1, 0, 0, 2, 1, 1)
    original, mapping = geometry.fragment_all(kernel)
    for tag in [left, right]:
        pieces = geometry.get_tags_after_fragment([tag], original, mapping, dimension=3)
        assert sum(kernel.getMass(3, piece) for piece in pieces) == pytest.approx(2)
    assert sum(
        kernel.getMass(dim, tag) for dim, tag in gmsh.model.getEntities(3)
    ) == pytest.approx(3)
