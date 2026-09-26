"""Regression tests for matching wave-port bounds to dielectric volumes."""

from __future__ import annotations

import pytest

from gsim.common.stack import LayerStack
from gsim.common.stack.extractor import Layer
from gsim.palace.mesh.geometry import (
    GeometryData,
    resolve_dielectric_regions,
    resolve_mesh_domain_bounds,
)


def _assert_airbox_bounds(stack: LayerStack, expected_z: tuple[float, float]):
    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    margins = {
        "margin_x": 1.0,
        "margin_y": 2.0,
        "airbox_z_below": 3.0,
        "airbox_z_above": 7.0,
    }
    bounds = resolve_mesh_domain_bounds(geometry, stack, **margins)
    regions = resolve_dielectric_regions(geometry, stack, **margins)
    airbox = next(region for region in regions if region.name == "airbox")
    expected_bounds = (-1.0, -2.0, expected_z[0], 11.0, 22.0, expected_z[1])

    assert bounds == pytest.approx(expected_bounds)
    assert (
        airbox.xmin,
        airbox.ymin,
        airbox.zmin,
        airbox.xmax,
        airbox.ymax,
        airbox.zmax,
    ) == pytest.approx(expected_bounds)
    return regions


@pytest.mark.parametrize(
    ("zmin", "zmax"),
    [(20.0, 20.0), (20.0, 10.0), (-20.0, -20.0), (-10.0, -20.0)],
    ids=["zero-above", "reversed-above", "zero-below", "reversed-below"],
)
def test_nonpositive_dielectric_thickness_does_not_expand_port_bounds(zmin, zmax):
    stack = LayerStack(
        dielectrics=[
            {"name": "oxide", "zmin": -2.0, "zmax": 5.2, "material": "sio2"},
            {"name": "invalid", "zmin": zmin, "zmax": zmax, "material": "sio2"},
        ],
        materials={"sio2": {"type": "dielectric", "permittivity": 3.9}},
    )

    regions = _assert_airbox_bounds(stack, expected_z=(-5.0, 12.2))

    assert [region.name for region in regions] == ["oxide", "airbox"]


def test_stack_without_dielectrics_uses_all_layer_extents():
    stack = LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=(index, 0),
                zmin=zmin,
                zmax=zmax,
                thickness=zmax - zmin,
                material="aluminum",
                layer_type="conductor",
            )
            for index, (name, zmin, zmax) in enumerate(
                [("lower_metal", -4.0, -1.0), ("upper_metal", 2.0, 6.0)], start=1
            )
        }
    )

    regions = _assert_airbox_bounds(stack, expected_z=(-7.0, 13.0))

    assert [region.name for region in regions] == ["airbox"]


def test_empty_stack_uses_default_z_range_for_airbox_and_ports():
    regions = _assert_airbox_bounds(LayerStack(), expected_z=(-3.0, 7.0))

    assert [region.name for region in regions] == ["airbox"]


@pytest.mark.parametrize(
    "resolve", [resolve_mesh_domain_bounds, resolve_dielectric_regions]
)
@pytest.mark.parametrize(
    ("zmin", "zmax"), [(float("-inf"), 5.2), (-2.0, float("inf"))]
)
def test_nonfinite_stack_extents_are_rejected(resolve, zmin, zmax):
    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        dielectrics=[
            {"name": "oxide", "zmin": zmin, "zmax": zmax, "material": "sio2"}
        ],
        materials={"sio2": {"type": "dielectric", "permittivity": 3.9}},
    )

    with pytest.raises(ValueError, match="z extents"):
        resolve(
            geometry,
            stack,
            margin_x=0.0,
            airbox_z_above=7.0,
            airbox_z_below=3.0,
        )
