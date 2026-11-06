"""XY field-image lookups remain compatible with saved monitor names."""

import json
import warnings

import pytest

from gsim import fdtd
from tests.fdtd.test_results import _result_document


@pytest.mark.parametrize("stored_name", ["field_xy", "overview_xy"])
def test_monitor_lookup_and_removal_accept_both_names(stored_name):
    monitors = fdtd.Monitors()
    plane = monitors.add_plane(
        stored_name, center_um=(0, 0, 0), size_um=(1, 1, 0), normal="+z"
    )
    alternate = "overview_xy" if stored_name == "field_xy" else "field_xy"
    assert monitors[alternate] is plane
    assert [monitor.name for monitor in monitors] == [stored_name]
    assert monitors.remove(alternate) is plane
    assert len(monitors) == 0


@pytest.mark.parametrize("stored_name", ["field_xy", "overview_xy"])
def test_result_alias_loads_saved_files_without_duplicate_monitors(
    tmp_path, stored_name
):
    document = _result_document()
    document["plane_monitors"][stored_name] = document["plane_monitors"].pop("top")
    path = tmp_path / "sparams.json"
    path.write_text(json.dumps(document))
    monitors = fdtd.FDTDResult.from_file(path).monitors
    alternate = "overview_xy" if stored_name == "field_xy" else "field_xy"
    assert monitors[alternate] is monitors[stored_name]
    assert monitors.get(alternate) is monitors[stored_name]
    assert alternate in monitors
    assert list(monitors) == [stored_name]
    assert monitors.get("missing", "fallback") == "fallback"
    with pytest.raises(KeyError, match="missing"):
        _ = monitors["missing"]


def test_explicit_names_take_precedence_over_aliases(tmp_path):
    monitors = fdtd.Monitors()
    for name in ("field_xy", "overview_xy"):
        monitors.add_plane(name, center_um=(0, 0, 0), size_um=(1, 1, 0), normal="+z")
    assert monitors["field_xy"].name == "field_xy"
    assert monitors["overview_xy"].name == "overview_xy"
    document = _result_document()
    plane = document["plane_monitors"].pop("top")
    document["plane_monitors"] = {"field_xy": plane, "overview_xy": plane}
    path = tmp_path / "sparams.json"
    path.write_text(json.dumps(document))
    results = fdtd.FDTDResult.from_file(path).monitors
    assert results["field_xy"] is not results["overview_xy"]
    assert results["field_xy"].name == "field_xy"
    assert results["overview_xy"].name == "overview_xy"


@pytest.mark.parametrize("name", ["field_xy", "overview_xy"])
def test_field_alias_collision_does_not_add_diagnostic_planes(fdtd_pdk_module, name):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    simulation.monitors.add_plane(
        name, center_um=(1, 0, 0.11), size_um=(2, 0.5, 0), normal="+z"
    )
    before = simulation.monitors.model_dump()
    with pytest.raises(ValueError, match="already exists"):
        simulation.add_monitors()
    assert simulation.monitors.model_dump() == before


@pytest.mark.parametrize("name", ["field_xy", "overview_xy", "device_image"])
def test_explicit_xy_name_serializes_without_naming_warning(
    tmp_path, fdtd_pdk_module, name
):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    simulation.source(port="o1", wavelength_span_um=0, num_wavelengths=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation.add_monitors(xy_name=name)
        artifacts = simulation.write(tmp_path)
    assert not any(
        "default XY monitor name" in str(warning.message) for warning in caught
    )
    expected_names = {"port_o1", "port_o2", name}
    assert {monitor.name for monitor in simulation.monitors} == expected_names
    assert {
        monitor["name"] for monitor in simulation.monitors.model_dump()
    } == expected_names
    document = json.loads(artifacts.config_path.read_text())
    assert {monitor["name"] for monitor in document["monitors"]} == expected_names


def test_xy_false_ignores_name_without_deprecation_warning(fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation.add_monitors(xy=False)
    assert not any(
        "default XY monitor name" in str(warning.message) for warning in caught
    )
    assert {monitor.name for monitor in simulation.monitors} == {"port_o1", "port_o2"}


@pytest.mark.parametrize("name", ["port_o1", "port_o2"])
def test_xy_name_cannot_shadow_a_generated_port_image(fdtd_pdk_module, name):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight")
    with pytest.raises(ValueError, match="already exists"):
        simulation.add_monitors(xy_name=name)
    assert len(simulation.monitors) == 0
