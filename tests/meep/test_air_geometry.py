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
        component,
        stack,
        port="o1",
        y_span=2,
        background_material="SiO2",
        z_margin=0.5,
        pml_thickness=0.2,
    )
    assert any(block.material == air for block in geometry.blocks)
    assert len(geometry.blocks) == 1
    block = geometry.blocks[0]
    assert block.z_center - block.z_size / 2 == pytest.approx(-0.1)
    assert block.z_center + block.z_size / 2 == pytest.approx(0.1)
    profile = refractive_index_profile(
        stack,
        wavelength=1.55,
        z_grid=np.array([-0.3, 0.1, 0.4]),
        y_grid=np.array([0, 0.8]),
        component=component,
        port="o1",
        background_material="SiO2",
    )
    assert profile[1, 0] == pytest.approx(1)
    assert profile[1, 1] > 1.4
    assert np.all(profile[[0, 2], :] > 1.4)


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


@pytest.mark.parametrize("air", ["Air", "air"])
@pytest.mark.parametrize("plane", ["slab", "xz", "yz"])
def test_local_mode_air_geometry_keeps_declared_z_bounds(monkeypatch, air, plane):
    from gsim.meep import mode_solver

    fake_meep = SimpleNamespace(
        Vector3=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
        Block=lambda **kwargs: SimpleNamespace(**kwargs),
        Simulation=lambda **kwargs: SimpleNamespace(**kwargs),
        PML=lambda **kwargs: SimpleNamespace(**kwargs),
        X="x",
        Y="y",
        Z="z",
        inf=np.inf,
    )
    monkeypatch.setattr(mode_solver, "_import_meep", lambda: fake_meep)
    monkeypatch.setattr(mode_solver, "_meep_medium", lambda material: material)
    component = gf.Component()
    component.add_polygon([(0, -0.2), (2, -0.2), (2, 0.2), (0, 0.2)], layer=(2, 0))
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
    kwargs = dict(
        stack=stack,
        materials={air: object(), "SiO2": object()},
        background_material="SiO2",
        z_margin=0.5,
        pml_thickness=0.2,
        resolution=32,
    )
    if plane == "slab":
        simulation, _ = mode_solver._build_slab_xz_cell(**kwargs)
    else:
        builder = getattr(mode_solver, f"_build_component_{plane}_cell")
        kwargs["x_span" if plane == "xz" else "y_span"] = 2
        kwargs["y_cut" if plane == "xz" else "x_cut"] = 0 if plane == "xz" else 1
        simulation, _ = builder(component=component, **kwargs)
    assert len(simulation.geometry) == 1
    block = simulation.geometry[0]
    assert block.center.z - block.size.z / 2 == pytest.approx(-0.1)
    assert block.center.z + block.size.z / 2 == pytest.approx(0.1)


@pytest.mark.parametrize("air", ["Air", "air"])
def test_patterned_void_is_preserved_inside_silicon_with_air_background(air):
    component = gf.Component()
    component.add_polygon([(0, -1), (2, -1), (2, 1), (0, 1)], layer=(1, 0))
    component.add_polygon([(0, -0.2), (2, -0.2), (2, 0.2), (0, 0.2)], layer=(2, 0))
    component.add_port(name="o1", center=(0, 0), width=2, layer=(1, 0), orientation=180)
    stack = SimpleNamespace(
        layers={
            "core": Layer(
                name="core",
                material="Si",
                gds_layer=(1, 0),
                zmin=0,
                zmax=0.2,
                thickness=0.2,
                layer_type="dielectric",
            ),
            "void": Layer(
                name="void",
                material=air,
                gds_layer=(2, 0),
                zmin=0,
                zmax=0.2,
                thickness=0.2,
                layer_type="dielectric",
            ),
        },
        dielectrics=[],
    )
    geometry = _precompute_cross_section_geometry(
        component, stack, port="o1", y_span=2, background_material="Air"
    )
    assert any(block.material == air for block in geometry.blocks)
    profile = refractive_index_profile(
        stack,
        wavelength=1.55,
        z_grid=np.array([0.1]),
        y_grid=np.array([0, 0.8]),
        component=component,
        port="o1",
        background_material="Air",
    )
    assert profile[0, 0] == pytest.approx(1)
    assert profile[0, 1] > 3
