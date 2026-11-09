"""Blanket films fill the FDTD domain independently of display visibility."""

import json

import gdsfactory as gf
import meshio
import numpy as np
import pytest
from gdsfactory.technology import LayerLevel

from gsim import fdtd
from gsim.common.pdk import resolve_passive_pcell
from gsim.fdtd.domain import background_bounds_nm


@pytest.fixture(autouse=True)
def active_test_pdk(fdtd_pdk_module):
    """Activate this test's PDK and restore the previous process state."""
    try:
        previous_pdk = gf.get_active_pdk()
    except ValueError:
        previous_pdk = gf.gpdk.PDK
    fdtd_pdk_module.PDK.activate(force=True)
    yield
    previous_pdk.activate(force=True)


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


def _group_volume(mesh, name):
    """Integrate tetrahedra independently of the layer's profile builder."""
    tags = mesh.cell_data_dict["gmsh:physical"]["tetra"]
    cells = mesh.cells_dict["tetra"][tags == mesh.field_data[name][0]]
    points = mesh.points[cells]
    return np.abs(np.linalg.det(points[:, 1:] - points[:, :1])).sum() / 6


@pytest.mark.parametrize("angle", [-10, 10])
@pytest.mark.parametrize("reference", [0, 0.5, 1])
def test_tapered_blankets_fill_the_domain_at_every_height(
    tmp_path, blanket_pdk, angle, reference
):
    levels = blanket_pdk.PDK.layer_stack.layers
    levels["core"].sidewall_angle = 0
    for name in ("substrate", "box", "clad"):
        levels[name].sidewall_angle = angle
        levels[name].width_to_z = reference
    simulation = _simulation(blanket_pdk)
    mesh = meshio.read(simulation.write(tmp_path).mesh_path)
    # Domain is 2 x 1.6 um. Substrate and clad retain half a micron.
    # Physical groups retain overlap; material priority is applied by the solver.
    assert _group_volume(mesh, "substrate") == pytest.approx(2 * 1.6 * 0.5 * 1e9)
    assert _group_volume(mesh, "box") == pytest.approx(2 * 1.6 * 1e9)
    assert _group_volume(mesh, "clad") == pytest.approx(2 * 1.6 * 0.5 * 1e9)


def test_tapered_background_keeps_exclusion_mask_reference(blanket_pdk):
    from math import radians, tan

    from shapely.geometry import box

    from gsim.common.polygon import fuse_polygons

    component = blanket_pdk.PDK.get_component("straight").copy()
    component.add_polygon(
        [(0.7, -0.1), (1.3, -0.1), (1.3, 0.1), (0.7, 0.1)], layer=(9, 0)
    )
    level = blanket_pdk.PDK.layer_stack.layers["box"]
    level.background_exclude_layers = [(9, 0)]
    level.sidewall_angle = 10
    level.width_to_z = 0.5
    domain = box(0, -0.65, 2, 0.95)
    geometry = fuse_polygons(
        component, level, background_bounds=domain.bounds, simplify_tol=0
    )
    for fraction in (0, 0.5, 1):
        offset = (0.5 - fraction) * tan(radians(10))
        profile = geometry.buffer(offset, join_style=2).intersection(domain)
        hole_width = max(0, 0.6 - 2 * offset)
        hole_height = max(0, 0.2 - 2 * offset)
        assert profile.area == pytest.approx(3.2 - hole_width * hole_height)


def test_default_slice_stays_at_layout_midpoint_with_asymmetric_padding(
    fdtd_pdk_module, monkeypatch
):
    from gsim.fdtd import slice_viz

    fdtd_pdk_module.PDK.layer_stack.layers["buried_oxide"].background = True
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.domain(padding_y=(0.4, 0.7))
    simulation.geometry("straight", mesh_size_nm=750)
    positions = []
    monkeypatch.setattr(
        slice_viz,
        "plot_mesh_slice",
        lambda *_args, **kwargs: positions.append(kwargs["position_um"]),
    )
    simulation.plot_2d_static(axis="y")
    simulation.plot_2d_static(axis="y", position_um=0.3)
    assert positions == pytest.approx([0, 0.3])


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


def test_merged_fiber_medium_samples_film_in_lateral_padding(tmp_path, blanket_pdk):
    simulation = _simulation(blanket_pdk)
    simulation.materials(background="ambient", overrides={"ambient": 1})
    center = (1, 0.6, 0.35)
    simulation.monitors.add_plane(
        "padding_receiver",
        center_um=center,
        size_um=(0.2, 0.2, 0),
        normal="+z",
        fiber_mode=fdtd.FiberMode(
            propagation_direction=(0, 0, 1),
            e_polarization=(1, 0, 0),
            focal_point_um=center,
            waist_radius_um=0.2,
        ),
    )
    artifacts = simulation.write(tmp_path)
    config = json.loads(artifacts.config_path.read_text())
    receiver = next(
        monitor
        for monitor in config["monitors"]
        if monitor["name"] == "padding_receiver"
    )
    assert receiver["fiber_mode"]["refractive_index"] == pytest.approx(
        1.444023621703261
    )


@pytest.mark.parametrize("angle", [-10, 0, 10])
@pytest.mark.parametrize("reference", [0, 0.5, 1])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_one_explicit_axis_keeps_automatic_layout_bounds(
    tmp_path,
    blanket_pdk,
    angle,
    reference,
    axis,
):
    levels = blanket_pdk.PDK.layer_stack.layers
    levels["core"].sidewall_angle = 0
    for name in ("substrate", "box", "clad"):
        levels[name].sidewall_angle = angle
        levels[name].width_to_z = reference
    bounds = (
        {"x_bounds": (0, 2)}
        if axis == "x"
        else {
            "padding_y": None,
            "y_bounds": (-0.8, 1.2),
        }
    )
    simulation = fdtd.Simulation(pdk=blanket_pdk)
    simulation.domain(padding_y=(0.4, 0.7), z_bounds=(-1.5, 0.5), clip_geometry=True)
    simulation.domain(**bounds)
    simulation.source(port="o1")
    simulation.geometry("straight", mesh_size_nm=700)
    expected_low = [0, -650 if axis == "x" else -800, -1500]
    expected_high = [2000, 950 if axis == "x" else 1200, 500]
    for index in range(2):
        artifacts = simulation.write(tmp_path / str(index))
        mesh = meshio.read(artifacts.mesh_path)
        assert mesh.points.min(axis=0) == pytest.approx(expected_low)
        assert mesh.points.max(axis=0) == pytest.approx(expected_high)
        for name, thickness in (("substrate", 0.5), ("box", 1), ("clad", 0.5)):
            width = (expected_high[1] - expected_low[1]) / 1000
            assert _group_volume(mesh, name) == pytest.approx(
                2 * width * thickness * 1e9
            )
