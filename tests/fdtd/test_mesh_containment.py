"""Bounding-box roundoff must not trigger expensive redundant CAD clipping."""

import gmsh
import pytest

from gsim.fdtd.mesh import _clip_layer_to_background
from gsim.fdtd.mesh_validation import MESH_BOUNDS_TOLERANCE_NM


@pytest.fixture
def kernel():
    """Own a Gmsh kernel without changing any existing session."""
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        yield gmsh.model.occ
    finally:
        gmsh.finalize()


def test_exactly_contained_solid_does_not_clip_occ_bounding_box_padding(
    kernel, monkeypatch
):
    bounds = (0, 0, 0, 10, 10, 10)
    background = kernel.addBox(0, 0, 0, 10, 10, 10)
    solid = kernel.addBox(0, 2, 2, 10, 2, 2)
    bbox = kernel.getBoundingBox(3, solid)
    assert -MESH_BOUNDS_TOLERANCE_NM <= bbox[0] < 0
    assert 10 < bbox[3] <= 10 + MESH_BOUNDS_TOLERANCE_NM
    monkeypatch.setattr(
        kernel, "intersect", lambda *_args, **_kwargs: pytest.fail("Redundant clipping")
    )
    assert _clip_layer_to_background(kernel, [solid], background, bounds) == [solid]
    assert kernel.getMass(3, solid) == pytest.approx(40)


def test_real_protrusion_is_still_clipped(kernel):
    bounds = (0, 0, 0, 10, 10, 10)
    background = kernel.addBox(0, 0, 0, 10, 10, 10)
    overhang = 1000 * MESH_BOUNDS_TOLERANCE_NM
    solid = kernel.addBox(-overhang, 2, 2, 10 + overhang, 2, 2)
    clipped = _clip_layer_to_background(kernel, [solid], background, bounds)
    assert len(clipped) == 1
    assert kernel.getMass(3, clipped[0]) == pytest.approx(40, rel=1e-12, abs=1e-10)
    kernel.synchronize()
    vertices = gmsh.model.getBoundary(
        [(3, clipped[0])], combined=False, oriented=False, recursive=True
    )
    points = [
        gmsh.model.getValue(0, tag, []) for dimension, tag in vertices if dimension == 0
    ]
    assert min(point[0] for point in points) >= -MESH_BOUNDS_TOLERANCE_NM
    assert max(point[0] for point in points) <= 10 + MESH_BOUNDS_TOLERANCE_NM
