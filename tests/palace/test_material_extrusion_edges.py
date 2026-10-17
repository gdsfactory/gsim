"""Verify thin layers, overlapping polygons, and holed conductor sheets."""

import pytest

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh.geometry import GeometryData, add_metals


@pytest.mark.parametrize("kind", ["dielectric", "conductor", "via"])
@pytest.mark.parametrize("thickness", [0.01, 1])
def test_material_extrusion_preserves_union_and_holes(kernel, kind, thickness):
    layer = Layer(
        name="material",
        gds_layer=(1, 0),
        zmin=0,
        zmax=thickness,
        thickness=thickness,
        material="material",
        layer_type=kind,
    )
    stack = LayerStack(
        layers={"material": layer},
        materials={"material": {"permittivity": 4, "conductivity": 0}},
    )
    holes = [([0.5, 1, 1, 0.5], [0.5, 0.5, 1, 1])]
    polygons = [(1, [0, 2, 2, 0], [0, 0, 2, 2], holes)]
    if kind == "dielectric":
        polygons.append((1, [1.5, 3, 3, 1.5], [0, 0, 2, 2], []))
    tags = add_metals(
        kernel,
        GeometryData(polygons=polygons, bbox=(0, 0, 3, 2), layer_bboxes={}),
        stack,
        planar_conductors=kind == "conductor",
    )
    if kind == "dielectric":
        volumes = tags["material"]["volumes"]
        assert sum(kernel.getMass(3, tag) for tag in volumes) == pytest.approx(
            0 if thickness < 0.02 else 5.75
        )
    else:
        sheets = tags["material"]["surfaces_xy"]
        assert sum(kernel.getMass(2, tag) for tag in sheets) == pytest.approx(3.75)
