"""Blanket films fill the FDTD domain independently of display visibility."""

import json

import meshio
import numpy as np
import pytest
from gdsfactory.technology import LayerLevel

from gsim import fdtd
from gsim.common.pdk import resolve_passive_pcell
from gsim.fdtd.domain import background_bounds_nm


@pytest.fixture
def blanket_pdk(fdtd_pdk_module):
    module = fdtd_pdk_module
    levels = module.PDK.layer_stack.layers
    for name, material, zmin, thickness in (
        ("substrate", "Si", -2, 1),
        ("box", "SiO2", -1, 1),
        ("clad", "SiO2", 0, 1),
        ("outside", "Unmodeled", 2, 1),
    ):
        levels[name] = LayerLevel(
            layer=(99, 0),
            background=True,
            zmin=zmin,
            thickness=thickness,
            material=material,
            mesh_order=10,
            info={"visible": False},
        )
    return module


def _simulation(module, z_bounds=(-1.5, 0.5), **domain):
    simulation = fdtd.Simulation(pdk=module)
    simulation.domain(
        padding_y=(0.4, 0.7), z_bounds=z_bounds, clip_geometry=True, **domain
    )
    simulation.source(port="o1")
    simulation.geometry("straight", mesh_size_nm=700)
    return simulation


def _group_points(mesh, name):
    tags = mesh.cell_data_dict["gmsh:physical"]["tetra"]
    cells = mesh.cells_dict["tetra"][tags == mesh.field_data[name][0]]
    assert len(cells) > 0
    vertices = mesh.points[cells]
    signed_volumes = np.linalg.det(vertices[:, 1:] - vertices[:, :1]) / 6
    assert np.all(signed_volumes > 0)
    return mesh.points[np.unique(cells)]


@pytest.mark.parametrize("z_bounds", [(-1.5, 0.5), (-0.5, 0.1), (-2.5, 1.2)])
def test_blanket_mesh_fills_padding_and_clips_each_vertical_band(
    tmp_path, blanket_pdk, z_bounds
):
    original = blanket_pdk.PDK.layer_stack.model_dump()
    simulation = _simulation(blanket_pdk, z_bounds)
    artifacts = simulation.write(tmp_path)
    mesh = meshio.read(artifacts.mesh_path)
    for name in ("substrate", "box", "clad"):
        level = blanket_pdk.PDK.layer_stack.layers[name]
        lower = max(level.zmin, z_bounds[0])
        upper = min(level.zmin + level.thickness, z_bounds[1])
        if lower >= upper:
            assert name not in mesh.field_data
            continue
        points = _group_points(mesh, name)
        assert points.min(axis=0) == pytest.approx([0, -650, lower * 1000])
        assert points.max(axis=0) == pytest.approx([2000, 950, upper * 1000])
        assert simulation.resolved.layers[name].geometry.bounds == pytest.approx(
            (0, -0.65, 2, 0.95)
        )
    config = json.loads(artifacts.config_path.read_text())
    assert "Unmodeled" not in config["materials"]
    assert blanket_pdk.PDK.layer_stack.model_dump() == original
    assert not simulation.resolved.component.get_polygons(by="tuple").get((99, 0))


def test_auto_padding_does_not_accumulate_and_geometry_tracks_domain_changes(
    tmp_path, blanket_pdk
):
    simulation = _simulation(blanket_pdk)
    expected = (0, -650, -1500, 2000, 950, 500)
    for index in range(2):
        artifacts = simulation.write(tmp_path / str(index))
        mesh = meshio.read(artifacts.mesh_path)
        assert mesh.points.min(axis=0) == pytest.approx(expected[:3])
        assert mesh.points.max(axis=0) == pytest.approx(expected[3:])
        assert background_bounds_nm(
            simulation.resolved,
            simulation.materials.background,
            simulation.domain.padding,
            padding_y=simulation.domain.padding_y,
            z_bounds=simulation.domain.z_bounds,
            clip_geometry=True,
        ) == pytest.approx(expected)
    simulation.domain(padding_y=None, y_bounds=(-1.1, 1.3), z_bounds=(-0.5, 0.4))
    artifacts = simulation.write(tmp_path / "changed")
    mesh = meshio.read(artifacts.mesh_path)
    assert mesh.points.min(axis=0) == pytest.approx([0, -1100, -500])
    assert mesh.points.max(axis=0) == pytest.approx([2000, 1300, 400])
    assert "substrate" not in mesh.field_data


def test_excluded_film_can_reappear_only_in_lateral_padding(tmp_path, blanket_pdk):
    component = blanket_pdk.PDK.get_component("straight").copy()
    component.add_polygon([(0, -0.25), (2, -0.25), (2, 0.25), (0, 0.25)], layer=(9, 0))
    blanket_pdk.PDK.layer_stack.layers["box"].background_exclude_layers = [(9, 0)]
    native = resolve_passive_pcell(component, pdk=blanket_pdk, z_bounds=(-1.5, 0.5))
    assert "box" not in native.layers
    simulation = _simulation(blanket_pdk)
    simulation.geometry(component)
    assert simulation.resolved.layers["box"].geometry.area == pytest.approx(2.2)
    assert not simulation.resolved.layers["box"].geometry.contains(
        simulation.resolved.layers["core"].geometry.centroid
    )
    mesh = meshio.read(simulation.write(tmp_path).mesh_path)
    _group_points(mesh, "box")


def test_newly_exposed_film_contributes_to_automatic_domain_height(
    tmp_path, fdtd_pdk_module
):
    module = fdtd_pdk_module
    levels = module.PDK.layer_stack.layers
    levels["buried_oxide"].background = True
    levels["buried_oxide"].background_exclude_layers = [(9, 0)]
    levels["buried_oxide"].material = "Si"
    levels["buried_oxide"].zmin = -2
    levels["buried_oxide"].thickness = 1
    component = module.PDK.get_component("straight").copy()
    component.add_polygon([(0, -0.25), (2, -0.25), (2, 0.25), (0, 0.25)], layer=(9, 0))
    simulation = fdtd.Simulation(pdk=module, domain={"padding": 0.25})
    simulation.source(port="o1")
    simulation.geometry(component, mesh_size_nm=700)
    mesh = meshio.read(simulation.write(tmp_path).mesh_path)
    assert mesh.points[:, 2].min() == pytest.approx(-2250)
    assert mesh.points[:, 2].max() == pytest.approx(470)
    points = _group_points(mesh, "buried_oxide")
    assert points[:, 2].min() == pytest.approx(-2000)
    assert points[:, 2].max() == pytest.approx(-1000)
