"""Notebook diagnostic planes follow the resolved PDK geometry and source."""

from __future__ import annotations

import json

import pytest

from gsim import fdtd, gcloud


@pytest.fixture(autouse=True)
def block_cloud(monkeypatch):
    """Configuration tests must never perform cloud operations."""

    def blocked(*_args, **_kwargs):
        pytest.fail("Monitor configuration attempted a cloud operation")

    for target in (fdtd.Simulation, gcloud):
        for name in ("run", "upload", "start", "wait_for_results"):
            if hasattr(target, name):
                monkeypatch.setattr(target, name, blocked)


def test_default_diagnostic_planes_serialize(tmp_path, fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.source(port="o1", wavelength_um=1.55)
    simulation.geometry("straight", settings={"length": 4})
    assert simulation.add_monitors() is simulation
    planes = {plane.name: plane for plane in simulation.monitors}
    assert set(planes) == {"port_o1", "port_o2", "overview_xy"}
    assert planes["port_o1"].center_um == pytest.approx((0.18, 0, 0.11))
    assert planes["port_o2"].center_um == pytest.approx((3.82, 0, 0.11))
    assert planes["port_o1"].normal == "-x"
    assert planes["port_o2"].normal == "+x"
    assert planes["port_o1"].size_um == pytest.approx((0, 2, 1.72))
    overview = planes["overview_xy"]
    assert overview.center_um == pytest.approx((2, 0, 0.11))
    assert overview.size_um == pytest.approx((4, 0.5, 0))
    for plane in planes.values():
        assert plane.flux is False
        assert plane.wavelengths_um == [1.55]
        assert plane.heatmap is not None
        assert plane.heatmap.quantity == "intensity"
        assert plane.heatmap.wavelengths_um == [1.55]
    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text(encoding="utf8"))
    assert {plane["name"] for plane in document["monitors"]} == set(planes)
    assert all(plane["wavelengths"] == [1550] for plane in document["monitors"])
    assert set(document["geometry"]["ports"]) == {"o1", "o2"}
    assert document["excitation"]["default_port"] == "o1"


@pytest.mark.parametrize(("xy", "z_um"), [(False, None), (False, 100), (True, 0.0)])
def test_optional_xy_and_explicit_zero_height(fdtd_pdk_module, xy, z_um):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    simulation.add_monitors(xy=xy, z_um=z_um)
    names = {plane.name for plane in simulation.monitors}
    assert ("overview_xy" in names) is xy
    if xy:
        assert simulation.monitors["overview_xy"].center_um[2] == 0


def test_y_ports_use_current_source_and_solver(fdtd_pdk_module):
    component = fdtd_pdk_module.PDK.get_component("straight").copy()
    component.rotate(90)
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry(component)
    simulation.source(wavelength_um=1.31)
    simulation.solver(cell_size_nm=40)
    simulation.add_monitors(xy=False)
    assert simulation.monitors["port_o1"].normal == "-y"
    assert simulation.monitors["port_o2"].normal == "+y"
    assert simulation.monitors["port_o1"].size_um == pytest.approx((2, 0, 1.72))
    assert simulation.monitors["port_o1"].center_um == pytest.approx((0, 0.12, 0.11))
    assert simulation.monitors["port_o2"].center_um == pytest.approx((0, 1.88, 0.11))
    heatmap = simulation.monitors["port_o1"].heatmap
    assert heatmap is not None
    assert heatmap.wavelengths_um == [1.31]


def test_custom_monitors_and_collision_are_atomic(fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    custom = simulation.monitors.add_plane(
        "overview_xy", center_um=(1, 0, 0.11), size_um=(2, 0.5, 0), normal="+z"
    )
    before = simulation.monitors.model_dump()
    with pytest.raises(ValueError, match="already exists"):
        simulation.add_monitors()
    assert simulation.monitors.model_dump() == before
    simulation.add_monitors(xy=False)
    assert simulation.monitors["overview_xy"] is custom
    before = simulation.monitors.model_dump()
    with pytest.raises(ValueError, match="already exists"):
        simulation.add_monitors(xy=False)
    assert simulation.monitors.model_dump() == before


def test_vertical_ports_keep_their_existing_workflow(fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("vertical_coupler")
    simulation.add_monitors()
    assert {plane.name for plane in simulation.monitors} == {"port_o1", "overview_xy"}


def test_missing_geometry_fails_before_monitor_mutation():
    simulation = fdtd.Simulation()
    with pytest.raises(fdtd.FDTDGeometryError, match="No geometry"):
        simulation.add_monitors()
    assert len(simulation.monitors) == 0


@pytest.mark.parametrize("margin_um", [0.0, 0.25, 1.0])
def test_window_tracks_each_port_and_owning_layer(fdtd_pdk_module, margin_um):
    import gdsfactory as gf

    layer = fdtd_pdk_module.PDK.layer_stack.layers["core"]
    layer.zmin = 1.2
    layer.thickness = -0.6
    component = gf.Component()
    component.add_polygon([(0, -0.6), (4, -0.9), (4, 0.9), (0, 0.6)], layer=(1, 0))
    component.add_port("in", center=(0, 0), width=1.2, orientation=180, layer=(1, 0))
    component.add_port("out", center=(4, 0), width=1.8, orientation=0, layer=(1, 0))
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry(component)
    simulation.add_monitors(xy=False, margin_um=margin_um)
    for name, width in (("in", 1.2), ("out", 1.8)):
        monitor = simulation.monitors[f"port_{name}"]
        assert monitor.size_um == pytest.approx(
            (0, width + 2 * margin_um, 0.6 + 2 * margin_um)
        )
        assert monitor.center_um[2] == pytest.approx(0.9)


@pytest.mark.parametrize("margin_um", [-1.0, float("nan"), float("inf")])
def test_invalid_margin_does_not_mutate_monitors(fdtd_pdk_module, margin_um):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        simulation.add_monitors(margin_um=margin_um)
    assert len(simulation.monitors) == 0
