"""Diagnostic windows on a crop edge stay inside the domain to the last bit."""

import json
import warnings

import pytest

from tests.fdtd.test_port_bands import _simulation, band_stack  # noqa: F401

CROPS = [
    {"y_bounds": (-0.1, 0.3)},
    {"z_bounds": (0.02, 0.12)},
    {"y_bounds": (-0.1, 0.3), "z_bounds": (0.02, 0.12)},
    {"y_bounds": (-0.07, 0.13), "z_bounds": (0.01, 0.07)},
    {"y_bounds": (-0.15, 0.25), "z_bounds": (0.03, 0.29)},
]


@pytest.mark.parametrize("margin_um", [0, 0.25])
@pytest.mark.parametrize("crop", CROPS, ids=str)
def test_windows_on_crop_edges_are_written_inside_the_domain(
    tmp_path,
    band_stack,  # noqa: F811
    margin_um,
    crop,
):
    simulation = _simulation(band_stack, **crop)
    simulation.add_monitors(xy_name="field_xy", margin_um=margin_um)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        artifacts = simulation.write(tmp_path)
    config = json.loads(artifacts.config_path.read_text())
    bounds = config["grid"]["non_pml_bounds"]
    assert {monitor["name"] for monitor in config["monitors"]} >= {"port_o1"}
    for monitor in config["monitors"]:
        for axis in range(3):
            # Exact comparison: one ulp outside the domain must not be written.
            assert monitor["region_min"][axis] >= bounds["min"][axis]
            assert monitor["region_max"][axis] <= bounds["max"][axis]
    port = next(item for item in config["monitors"] if item["name"] == "port_o1")
    for axis, name in ((1, "y_bounds"), (2, "z_bounds")):
        if name in crop:
            low, high = crop[name]
            # The window is as wide as or wider than the crop, so it lands on it
            # (a window one ulp inside the edge is harmless).
            assert port["region_min"][axis] == pytest.approx(
                bounds["min"][axis], abs=1e-7, rel=0
            )
            assert port["region_max"][axis] == pytest.approx(
                bounds["max"][axis], abs=1e-7, rel=0
            )
            assert bounds["min"][axis] == pytest.approx(low * 1000)
            assert bounds["max"][axis] == pytest.approx(high * 1000)


def test_requested_window_overshoots_the_crop_by_an_ulp(band_stack):  # noqa: F811
    simulation = _simulation(band_stack, z_bounds=(0.02, 0.12))
    simulation.add_monitors(xy=False, margin_um=0)
    monitor = simulation.monitors["port_o1"]
    lower_nm = monitor.center_um[2] * 1000 - monitor.size_um[2] * 500
    # Guards the premise: the unclipped request is not exactly on the crop edge.
    assert lower_nm < 20
    assert lower_nm == pytest.approx(20, abs=1e-9)
