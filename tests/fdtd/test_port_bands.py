"""Guided ports join disjoint bands and respect bounded diagnostic windows."""

import json
import warnings

import gdsfactory as gf
import meshio
import numpy as np
import pytest

from gsim import fdtd


@pytest.fixture
def band_stack(fdtd_pdk_module):
    """Return two same-material bands sharing one drawn port layer."""
    module = fdtd_pdk_module
    module.PDK.activate()
    core = module.PDK.layer_stack.layers["core"].model_copy(
        update={"zmin": 0, "thickness": 0.1, "sidewall_angle": 0, "mesh_order": 1}
    )
    upper = core.model_copy(update={"zmin": 0.1, "thickness": 0.2, "mesh_order": 2})
    module.PDK.layer_stack.layers = {"core": core, "upper": upper}
    component = gf.Component()
    component.add_polygon([(0, -0.3), (2, -0.3), (2, 0.3), (0, 0.3)], layer=(1, 0))
    for name, x, angle in (("o1", 0, 180), ("o2", 2, 0)):
        component.add_port(
            name, center=(x, 0), width=0.6, orientation=angle, layer=(1, 0)
        )
    return module, component


def _simulation(band_stack, **bounds):
    module, component = band_stack
    simulation = fdtd.Simulation(pdk=module)
    simulation.domain(clip_geometry=True, **bounds)
    simulation.source(port="o1")
    simulation.geometry(component, mesh_size_nm=700)
    return simulation


def _port_vertices(artifacts):
    mesh = meshio.read(artifacts.mesh_path)
    triangles = mesh.cells_dict["triangle"]
    tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    return mesh.points[triangles[tags == mesh.field_data["port_o1"][0]]]


def _area(vertices):
    return (
        np.linalg.norm(
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=1,
        ).sum()
        / 2
    )


@pytest.mark.parametrize("reverse_extrusion", [False, True])
@pytest.mark.parametrize("upper_has_priority", [False, True])
def test_overlapping_port_bands_keep_the_priority_owner(
    tmp_path, band_stack, reverse_extrusion, upper_has_priority
):
    module, _ = band_stack
    upper = module.PDK.layer_stack.layers["upper"]
    upper.zmin = 0.1 if reverse_extrusion else 0.01
    upper.thickness = -0.09 if reverse_extrusion else 0.09
    upper.mesh_order = 0 if upper_has_priority else 2
    simulation = _simulation(band_stack)
    port = simulation.resolved.ports["o1"]
    assert port.physical_layer_keys == (("upper",) if upper_has_priority else ("core",))
    vertices = _port_vertices(simulation.write(tmp_path))
    # In the contained-overlap case, summing both faces would incorrectly give 114000.
    assert _area(vertices) == pytest.approx(54000 if upper_has_priority else 60000)


def test_competing_abutting_bands_do_not_both_join(tmp_path, band_stack):
    module, _ = band_stack
    upper = module.PDK.layer_stack.layers["upper"]
    upper.thickness = 0.1
    module.PDK.layer_stack.layers["overlapping_branch"] = upper.model_copy(
        update={"thickness": 0.2, "mesh_order": 3}
    )
    simulation = _simulation(band_stack)
    assert simulation.resolved.ports["o1"].physical_layer_keys == ("core", "upper")
    assert _area(_port_vertices(simulation.write(tmp_path))) == pytest.approx(600 * 200)


def test_band_chain_is_joined_regardless_of_priority_order(tmp_path, band_stack):
    module, _ = band_stack
    upper = module.PDK.layer_stack.layers["upper"]
    upper.zmin = 0.2
    upper.thickness = 0.1
    module.PDK.layer_stack.layers["middle"] = upper.model_copy(
        update={"zmin": 0.2, "thickness": -0.1, "mesh_order": 3}
    )
    simulation = _simulation(band_stack)
    assert simulation.resolved.ports["o1"].physical_layer_keys == (
        "core",
        "middle",
        "upper",
    )
    assert _area(_port_vertices(simulation.write(tmp_path))) == pytest.approx(600 * 300)


@pytest.mark.parametrize("margin_um", [0, 0.25, 0.75])
@pytest.mark.parametrize("z_bounds", [(0.04, 0.23), (0.02, 0.04)])
@pytest.mark.parametrize("reverse_extrusion", [False, True])
def test_bounded_port_monitors_match_the_clipped_mesh(
    tmp_path, band_stack, margin_um, z_bounds, reverse_extrusion
):
    module, _ = band_stack
    if reverse_extrusion:
        upper = module.PDK.layer_stack.layers["upper"]
        upper.zmin, upper.thickness = 0.3, -0.2
    simulation = _simulation(band_stack, y_bounds=(-0.2, 0.2), z_bounds=z_bounds)
    low, high = z_bounds
    assert simulation.resolved.ports["o1"].center[2] == pytest.approx((low + high) / 2)
    assert simulation.resolved.ports["o1"].z_bounds == pytest.approx(z_bounds)
    simulation.add_monitors(xy_name="field_xy", margin_um=margin_um)
    monitor = simulation.monitors["port_o1"]
    assert monitor.center_um == pytest.approx((0.18, 0, (low + high) / 2))
    assert monitor.size_um == pytest.approx(
        (0, 0.6 + 2 * margin_um, high - low + 2 * margin_um)
    )
    overview = simulation.monitors["field_xy"]
    assert overview.center_um[2] == pytest.approx((low + high) / 2)
    assert overview.size_um == pytest.approx((2, 0.6, 0))
    with pytest.warns(UserWarning, match="cropped to the physical"):
        artifacts = simulation.write(tmp_path)
    vertices = _port_vertices(artifacts)
    assert vertices[:, :, 2].min() == pytest.approx(low * 1000)
    assert vertices[:, :, 2].max() == pytest.approx(high * 1000)
    assert _area(vertices) == pytest.approx(400 * (high - low) * 1000)
    config = json.loads(artifacts.config_path.read_text())
    saved = next(plane for plane in config["monitors"] if plane["name"] == "port_o1")
    assert saved["region_min"][2] == pytest.approx(low * 1000)
    assert saved["region_max"][2] == pytest.approx(high * 1000)
    assert saved["region_min"][1] == pytest.approx(-200)
    assert saved["region_max"][1] == pytest.approx(200)
    saved_xy = next(
        plane for plane in config["monitors"] if plane["name"] == "field_xy"
    )
    assert saved_xy["region_min"] == pytest.approx((0, -200, (low + high) * 500))
    assert saved_xy["region_max"] == pytest.approx((2000, 200, (low + high) * 500))
    # Clipping ports must not flatten the original sidewall/extrusion profiles.
    assert module.PDK.layer_stack.layers["core"].thickness == 0.1
    assert simulation.resolved.layers["core"].z_bounds == (0, 0.1)


def test_static_source_overlay_uses_clipped_port_height(band_stack):
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    simulation = _simulation(band_stack, z_bounds=(0.04, 0.23))
    axis = simulation.plot_2d_static(axis="y", position_um=0)
    source = next(line for line in axis.lines if line.get_label() == "Source o1")
    assert source.get_ydata() == pytest.approx((0.04, 0.23))


def test_crop_missing_diagnostic_aperture_fails_before_mutation(band_stack):
    _, component = band_stack
    component.add_polygon([(0, 1), (2, 1), (2, 2), (0, 2)], layer=(1, 0))
    simulation = _simulation(band_stack, y_bounds=(1.1, 1.3))
    with pytest.raises(fdtd.FDTDConfigError, match=r"Monitor 'port_o1'.*no area"):
        simulation.add_monitors(xy=False, margin_um=0)
    assert len(simulation.monitors) == 0


def test_bounded_diagnostics_recover_requested_margins_when_domain_grows(
    tmp_path, band_stack
):
    simulation = _simulation(band_stack, y_bounds=(-0.2, 0.2), z_bounds=(0.04, 0.23))
    simulation.geometry(band_stack[1], port_layer_map={(1, 0): ("core", "upper")})
    simulation.add_monitors(xy_name="field_xy", margin_um=0.25)
    requested = simulation.monitors.model_dump()
    with pytest.warns(UserWarning, match="cropped to the physical"):
        simulation.write(tmp_path / "small")
    simulation.domain(y_bounds=(-1, 1), z_bounds=(-1, 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifacts = simulation.write(tmp_path / "large")
    assert not any("cropped to the physical" in str(item.message) for item in caught)
    assert simulation.monitors.model_dump() == requested
    config = json.loads(artifacts.config_path.read_text())
    port = next(plane for plane in config["monitors"] if plane["name"] == "port_o1")
    assert port["region_min"] == pytest.approx((180, -550, -210))
    assert port["region_max"] == pytest.approx((180, 550, 480))
    assert config["geometry"]["ports"]["o1"]["layer"] == "core"
    assert simulation.resolved.ports["o1"].physical_layer_keys == ("core", "upper")
    field = next(plane for plane in config["monitors"] if plane["name"] == "field_xy")
    assert field["region_min"] == pytest.approx((0, -300, 135))
    assert field["region_max"] == pytest.approx((2000, 300, 135))
