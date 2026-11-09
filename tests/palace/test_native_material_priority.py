"""Check native 2D material ownership with overlapping doping and metals."""

import gdsfactory as gf
import pytest

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh.generator import _generate_native_2d_groups
from gsim.palace.mesh.geometry import extract_geometry
from gsim.palace.models import CrossSectionPlaneConfig


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("conductivity", [0, [1e6, 2e6, 3e6]])
def test_native_material_priority_preserves_domain_area(kernel, axis, conductivity):
    gf.gpdk.PDK.activate()
    component = gf.Component()
    specs = [
        ("core", 1, 0, 1, 0, 3, "dielectric"),
        ("doped", 2, 0, 1, 0.8, 2.2, "dielectric"),
        ("metal", 3, 0.2, 0.8, 0.5, 1.5, "conductor"),
    ]
    layers = {}
    for name, index, zmin, zmax, low, high, kind in specs:
        component.add_polygon(
            [(low, low), (high, low), (high, high), (low, high)], layer=(index, 0)
        )
        layers[name] = Layer(
            name=name,
            gds_layer=(index, 0),
            zmin=zmin,
            zmax=zmax,
            thickness=zmax - zmin,
            material=name,
            layer_type=kind,
        )
    stack = LayerStack(
        layers=layers,
        materials={
            "core": {"permittivity": 4},
            "doped": {"permittivity": 4},
            "metal": {"conductivity": conductivity},
            "oxide": {"permittivity": 2},
            "air": {"permittivity": 1},
        },
        dielectrics=[{"material": "oxide", "zmin": -1, "zmax": 0}],
    )
    options = dict(
        kernel=kernel,
        component=component,
        geometry=extract_geometry(component, stack),
        stack=stack,
        margin_x=1,
        margin_y=1,
        air_margin=1,
        airbox_margin_x=None,
        airbox_margin_y=None,
        airbox_z_above=None,
        airbox_z_below=None,
    )
    groups = _generate_native_2d_groups(
        cross_section=CrossSectionPlaneConfig(
            axis=axis, value=1, window=(-1, 4), window_z=(-1, 2)
        ),
        **options,
    )
    assert {"core", "doped", "oxide", "air"} <= groups["volumes"].keys()
    surfaces = [tag for info in groups["volumes"].values() for tag in info["tags"]]
    assert len(surfaces) == len(set(surfaces))
    assert sum(kernel.getMass(2, tag) for tag in surfaces) == pytest.approx(14.4)
    assert "metal" in groups["conductor_surfaces" if conductivity else "pec_surfaces"]
    assert groups["refinement_lines"]
    with pytest.raises(ValueError, match="does not intersect"):
        _generate_native_2d_groups(
            cross_section=CrossSectionPlaneConfig(axis=axis, value=100), **options
        )
