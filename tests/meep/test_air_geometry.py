"""Patterned air regions must survive mode setup in a dielectric background."""

from types import SimpleNamespace

import gdsfactory as gf
import numpy as np
import pytest

from gsim.common.stack.extractor import Layer
from gsim.meep.mode_solver import refractive_index_profile
from gsim.meep.simulation import _precompute_cross_section_geometry


@pytest.mark.parametrize("air", ["Air", "air"])
def test_patterned_air_is_exported_and_plotted_inside_oxide(air):
    component = gf.Component()
    component.add_polygon([(0, -0.2), (2, -0.2), (2, 0.2), (0, 0.2)], layer=(2, 0))
    component.add_port(
        name="o1", center=(0, 0), width=0.4, layer=(2, 0), orientation=180
    )
    layer = Layer(
        name="void",
        material=air,
        gds_layer=(2, 0),
        zmin=0,
        zmax=0.2,
        thickness=0.2,
        layer_type="dielectric",
    )
    stack = SimpleNamespace(layers={"void": layer}, dielectrics=[])
    geometry = _precompute_cross_section_geometry(
        component, stack, port="o1", y_span=2, background_material="SiO2"
    )
    assert any(block.material == air for block in geometry.blocks)
    profile = refractive_index_profile(
        stack,
        wavelength=1.55,
        z_grid=np.array([0.1]),
        y_grid=np.array([0, 0.8]),
        component=component,
        port="o1",
        background_material="SiO2",
    )
    assert profile[0, 0] == pytest.approx(1)
    assert profile[0, 1] > 1.4


@pytest.mark.parametrize("air", ["Air", "air"])
def test_undrawn_air_declaration_does_not_create_a_blanket_void(air):
    component = gf.c.straight(length=2)
    layer = Layer(
        name="undercut",
        material=air,
        gds_layer=(2, 0),
        zmin=0,
        zmax=0.2,
        thickness=0.2,
        layer_type="dielectric",
    )
    stack = SimpleNamespace(layers={"undercut": layer}, dielectrics=[])
    geometry = _precompute_cross_section_geometry(
        component, stack, port="o1", y_span=2, background_material="SiO2"
    )
    assert not geometry.blocks
    profile = refractive_index_profile(
        stack,
        wavelength=1.55,
        z_grid=np.array([0.1]),
        y_grid=np.array([0, 0.8]),
        component=component,
        port="o1",
        background_material="SiO2",
    )
    assert np.all(profile > 1.4)
