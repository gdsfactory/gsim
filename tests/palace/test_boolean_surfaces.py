"""Check priority cuts, smooth loops, and decimation with actual geometry."""

import gdsfactory as gf
import numpy as np
import pytest

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh.geometry import extract_geometry
from gsim.palace.mesh.gmsh_utils import (
    Entity,
    create_polygon_surface,
    run_boolean_pipeline,
)


@pytest.mark.parametrize("coincident", [False, True])
def test_surface_priority_retains_only_unclaimed_area(kernel, coincident):
    left = kernel.addRectangle(0, 0, 0, 2, 2)
    right = kernel.addRectangle(0 if coincident else 1, 0, 0, 2, 2)
    entities = [Entity("left", 2, 1, [left]), Entity("right", 2, 2, [right])]
    groups = run_boolean_pipeline(entities)
    areas = {
        entity.name: sum(
            kernel.getMass(dim, tag) for dim, tag in entity.dimtags if dim == 2
        )
        for entity in entities[:2]
    }
    assert areas == pytest.approx({"left": 4, "right": 0 if coincident else 2})
    assert any("left" in name for name in groups)
    assert all(tag > 0 for tag in groups.values())


@pytest.mark.parametrize("mode", ["spline", "bspline"])
@pytest.mark.parametrize("corners", [5, 180])
def test_smooth_closed_loops_preserve_area(kernel, mode, corners):
    angle = np.linspace(0, 2 * np.pi, 33)
    xs, ys = np.cos(angle).tolist(), np.sin(angle).tolist()
    tag = create_polygon_surface(
        kernel, xs, ys, 0, loop_mode=mode, corner_turn_threshold_deg=corners
    )
    kernel.synchronize()
    assert tag is not None
    assert kernel.getMass(2, tag) == pytest.approx(np.pi, rel=0.02)
    assert (
        create_polygon_surface(kernel, [0, 0, 1], [0, 0, 0], 0, loop_mode=mode) is None
    )


def test_decimation_preserves_layer_bounds_and_area():
    gf.gpdk.PDK.activate()
    component = gf.Component()
    angle = np.linspace(0, 2 * np.pi, 201)
    component.add_polygon(np.column_stack([np.cos(angle), np.sin(angle)]), layer=(1, 0))
    stack = LayerStack(
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=0,
                zmax=1,
                thickness=1,
                material="silicon",
                layer_type="dielectric",
            )
        }
    )
    original = extract_geometry(component, stack)
    simplified = extract_geometry(component, stack, decimate_tolerance=0.01)
    assert simplified.bbox == pytest.approx(original.bbox, abs=0.02)
    assert len(simplified.polygons[0][1]) < len(original.polygons[0][1])
