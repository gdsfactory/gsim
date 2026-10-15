"""Tests for solver-specific mesh semantics."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Polygon, box

import gsim.fdtd.mesh_geometry as mesh_geometry
import gsim.fdtd.mesh_loft as mesh_loft
from gsim.common.pdk import ResolvedLayer, ResolvedPort
from gsim.fdtd.mesh import _priority_by_mesh_order
from gsim.fdtd.mesh_geometry import iter_polygons, sidewall_slice_count
from gsim.fdtd.mesh_loft import LoftIncompatibleError, loft_section_polygons


def test_lower_pdk_mesh_order_becomes_higher_fdtd_priority() -> None:
    layers = {
        "core": SimpleNamespace(mesh_order=1),
        "same_order": SimpleNamespace(mesh_order=1),
        "slab": SimpleNamespace(mesh_order=4),
        "cladding": SimpleNamespace(mesh_order=7),
    }

    priorities = _priority_by_mesh_order(layers)

    assert priorities == {
        "core": 3,
        "same_order": 3,
        "slab": 2,
        "cladding": 1,
    }


def _resolved_layer(geometry: Polygon | MultiPolygon) -> ResolvedLayer:
    return ResolvedLayer(
        key="core",
        declared_name="core",
        layer=(1, 0),
        derived_layer=None,
        geometry=geometry,
        material="Si",
        zmin=0,
        thickness=0.22,
        zmax=0.22,
        sidewall_angle=10,
        width_to_z=0.5,
        bias=None,
        z_to_bias=None,
        mesh_order=1,
    )


def _resolved_port(name: str, x_um: float, normal_x: int) -> ResolvedPort:
    return ResolvedPort(
        name=name,
        center=(x_um, 0, 0.11),
        width=0.5,
        orientation=180.0 if normal_x < 0 else 0.0,
        normal=(normal_x, 0, 0),
        port_type="optical",
        layer_key="core",
        material="Si",
    )


def test_sidewall_slices_bound_geometry_error() -> None:
    layer = _resolved_layer(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

    slice_count = sidewall_slice_count(layer, geometry_tolerance_nm=10)

    assert slice_count == 2
    total_displacement_nm = 38.7919357558623
    assert total_displacement_nm / (2 * slice_count) < 10


def test_disconnected_polygons_retain_holes() -> None:
    ring = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    bus = Polygon([(5, 0), (7, 0), (7, 1), (5, 1)])

    polygons = iter_polygons(MultiPolygon([ring, bus]), layer_key="core")

    assert len(polygons) == 2
    assert sum(len(polygon.interiors) for polygon in polygons) == 1
    assert sum(polygon.area for polygon in polygons) == pytest.approx(14)


def test_loft_sections_follow_exact_sidewall_and_port_planes() -> None:
    layer = _resolved_layer(Polygon([(0, -0.25), (1, -0.25), (1, 0.25), (0, 0.25)]))
    ports = [_resolved_port("o1", 0, -1), _resolved_port("o2", 1, 1)]

    bottom, top = loft_section_polygons(layer, ports)

    sidewall_offset_um = 0.01939596787793115
    assert bottom.bounds == pytest.approx(
        (0, -0.25 - sidewall_offset_um, 1, 0.25 + sidewall_offset_um)
    )
    assert top.bounds == pytest.approx(
        (0, -0.25 + sidewall_offset_um, 1, 0.25 - sidewall_offset_um)
    )


def test_loft_preflight_rejects_topology_without_ring_correspondence() -> None:
    ring = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )

    with pytest.raises(LoftIncompatibleError, match="without holes"):
        loft_section_polygons(_resolved_layer(ring), [])


def test_incompatible_loft_uses_stepped_fallback(monkeypatch) -> None:
    ring = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    fallback_calls = []

    def fake_stepped_builder(
        kernel,
        layer,
        ports,
        *,
        geometry_tolerance_nm,
    ) -> list[int]:
        fallback_calls.append((kernel, layer, ports, geometry_tolerance_nm))
        return [42]

    monkeypatch.setattr(
        mesh_loft,
        "_add_stepped_layer_volumes",
        fake_stepped_builder,
    )
    kernel = object()
    layer = _resolved_layer(ring)

    volume_tags = mesh_loft.add_layer_volumes(
        kernel,
        layer,
        [],
        geometry_tolerance_nm=10,
    )

    assert volume_tags == [42]
    assert fallback_calls == [(kernel, layer, [], 10)]


def test_disconnected_polygons_are_lofted_independently(monkeypatch) -> None:
    first = box(0, 0, 1, 1)
    second = box(2, 0, 3, 1)
    layer = _resolved_layer(MultiPolygon([first, second]))
    kernel = object()
    lofted_geometries = []

    monkeypatch.setattr(
        mesh_loft,
        "_add_lofted_layer_volumes",
        lambda _kernel, polygon_layer, _sections: lofted_geometries.append(
            polygon_layer.geometry
        )
        or [len(lofted_geometries)],
    )

    volume_tags = mesh_loft.add_layer_volumes(
        kernel,
        layer,
        [],
        geometry_tolerance_nm=10,
    )

    assert volume_tags == [1, 2]
    assert lofted_geometries == [first, second]


@pytest.mark.parametrize("axis", [0, 1])
def test_stepped_profiles_preserve_port_planes_and_volume_in_nanometers(
    monkeypatch, axis
):
    geometry = box(0, -0.25, 1, 0.25)
    ports = [_resolved_port("o1", 0, -1), _resolved_port("o2", 1, 1)]
    if axis == 1:
        geometry = rotate(geometry, 90, origin=(0, 0))
        ports = [
            replace(
                port, center=(0, port.center[0], 0.11), normal=(0, port.normal[0], 0)
            )
            for port in ports
        ]
    extrusions = []

    def record_extrusion(_kernel, x, y, z, height, *, holes):
        extrusions.append((Polygon(zip(x, y, strict=True)), z, height))
        assert holes == []
        return len(extrusions)

    monkeypatch.setattr(mesh_geometry, "extrude_polygon", record_extrusion)
    tags = mesh_geometry._add_stepped_layer_volumes(
        None,
        _resolved_layer(geometry),
        ports,
        geometry_tolerance_nm=10,
    )
    assert tags == [1, 2]
    assert sum(
        polygon.area * height for polygon, _, height in extrusions
    ) == pytest.approx(110_000_000)
    for polygon, z, height in extrusions:
        assert polygon.bounds[axis] == pytest.approx(0)
        assert polygon.bounds[axis + 2] == pytest.approx(1000)
        assert z >= 0
        assert z + height <= 220


@pytest.mark.parametrize(
    "overrides",
    [
        {"bias": 0.1},
        {"z_to_bias": ([0, 1], [0, 0.1])},
        {"width_to_z": -0.1},
        {"width_to_z": 1.1},
        {"sidewall_angle": 80},
    ],
)
def test_unsupported_layer_profiles_fail_before_extrusion(overrides):
    layer = replace(_resolved_layer(box(0, 0, 1, 1)), **overrides)
    with pytest.raises(mesh_geometry.FDTDGeometryError, match="core"):
        mesh_geometry._validate_layer(layer)
