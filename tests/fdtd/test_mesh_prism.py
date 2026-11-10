"""Native holed faces must preserve the exact solid and fallback safely."""

from __future__ import annotations

from contextlib import contextmanager

import gmsh
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from gsim.fdtd.mesh_prism import extrude_polygon


@contextmanager
def _model():
    """Own an isolated Gmsh model for each construction check."""
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("holed_prism")
    try:
        yield gmsh.model.occ
    finally:
        gmsh.finalize()


def _extrude(kernel, polygon, *, height=2):
    """Pass a Shapely reference footprint to the FDTD prism constructor."""
    coordinates = list(polygon.exterior.coords)[:-1]
    holes = [list(ring.coords)[:-1] for ring in polygon.interiors]
    return extrude_polygon(
        kernel,
        [x for x, _ in coordinates],
        [y for _, y in coordinates],
        0,
        height,
        holes=[([x for x, _ in ring], [y for _, y in ring]) for ring in holes],
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [1, 1000])
def test_direct_holes_preserve_analytic_volume_and_exclude_voids(
    monkeypatch, reverse, scale
):
    outer = [(0, 0), (10, 0), (10, 8), (6, 8), (6, 6), (0, 6)]
    holes = [
        list(box(1, 1, 3, 3).exterior.coords),
        list(box(7, 1, 9, 4).exterior.coords),
    ]
    if reverse:
        outer.reverse()
        holes = [ring[::-1] for ring in holes]
    polygon = Polygon(
        [(x * scale, y * scale) for x, y in outer],
        holes=[[(x * scale, y * scale) for x, y in ring] for ring in holes],
    )
    # Independent rectangle arithmetic: 60 + 8 - 4 - 6, extruded by 2.
    expected_volume = 116 * scale**3
    with _model() as kernel:
        monkeypatch.setattr(
            kernel,
            "cut",
            lambda *_args, **_kwargs: pytest.fail("Unexpected boolean cut"),
        )
        tag = _extrude(kernel, polygon, height=2 * scale)
        assert kernel.getMass(3, tag) == pytest.approx(expected_volume)
        kernel.synchronize()
        assert gmsh.model.isInside(3, tag, [5 * scale, 3 * scale, scale]) == 1
        assert gmsh.model.isInside(3, tag, [2 * scale, 2 * scale, scale]) == 0
        assert gmsh.model.isInside(3, tag, [8 * scale, 2 * scale, scale]) == 0
        assert gmsh.model.isInside(3, tag, [2 * scale, 7 * scale, scale]) == 0
        gmsh.option.setNumber("Mesh.MeshSizeMax", scale)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.model.mesh.generate(3)
        types, _, nodes = gmsh.model.mesh.getElements(3, tag)
        assert 4 in types
        indices = list(types).index(4)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        points = dict(
            zip(node_tags, np.asarray(coordinates).reshape(-1, 3), strict=True)
        )
        tetrahedra = np.array([points[node] for node in nodes[indices]]).reshape(
            -1, 4, 3
        )
        volumes = abs(np.linalg.det(tetrahedra[:, 1:] - tetrahedra[:, :1])) / 6
        assert np.all(volumes > 0)
        assert volumes.sum() == pytest.approx(expected_volume)


@pytest.mark.parametrize("failure", ["plane_error", "filled_holes"])
def test_failed_direct_face_falls_back_without_removing_existing_entities(
    monkeypatch, failure
):
    polygon = Polygon(
        list(box(0, 0, 5, 5).exterior.coords),
        holes=[list(box(1, 1, 2, 2).exterior.coords)],
    )
    with _model() as kernel:
        existing = kernel.addBox(20, 20, 0, 1, 1, 1)
        original_plane = kernel.addPlaneSurface
        original_cut = kernel.cut
        attempts = []
        cuts = []

        def faulty_plane(loops, *args, **kwargs):
            attempts.append(list(loops))
            if len(attempts) == 1:
                if failure == "plane_error":
                    raise ValueError("Synthetic direct-face failure")
                return original_plane(loops[:1], *args, **kwargs)
            return original_plane(loops, *args, **kwargs)

        def record_cut(*args, **kwargs):
            cuts.append(True)
            return original_cut(*args, **kwargs)

        monkeypatch.setattr(kernel, "addPlaneSurface", faulty_plane)
        monkeypatch.setattr(kernel, "cut", record_cut)
        tag = _extrude(kernel, polygon)
        assert cuts == [True]
        assert kernel.getMass(3, tag) == pytest.approx(48)
        assert kernel.getMass(3, existing) == pytest.approx(1)
        assert set(kernel.getEntities(3)) == {(3, existing), (3, tag)}
        assert len(kernel.getEntities(2)) == 16


def test_surface_only_and_simple_polygon_paths_preserve_dimensions():
    polygon = Polygon(
        list(box(0, 0, 5, 5).exterior.coords),
        holes=[list(box(1, 1, 2, 2).exterior.coords)],
    )
    with _model() as kernel:
        tag = _extrude(kernel, polygon, height=0)
        assert kernel.getMass(2, tag) == pytest.approx(24)
        assert kernel.getEntities(3) == []
        solid = _extrude(kernel, box(10, 10, 13, 14), height=3)
        assert kernel.getMass(3, solid) == pytest.approx(36)
