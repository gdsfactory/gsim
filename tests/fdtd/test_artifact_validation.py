"""Exercise artifact validation and vertical source/monitor conventions."""

from types import SimpleNamespace

import numpy as np
import pytest

from gsim.fdtd.models import FDTDConfigError, FDTDGeometryError
from gsim.fdtd.simulation import ArtifactSimulation


@pytest.mark.parametrize(
    "options",
    [
        {"wavelength_um": 0},
        {"mesh_size_nm": np.nan},
        {"geometry_tolerance_nm": 31},
        {"background_material": ""},
        {"pml_cells": -1},
        {"wavelength_halfspan_um": 2},
        {"num_wavelengths": 0},
        {"max_timesteps": 0},
        {"energy_decay_fraction": 1},
        {"max_wall_seconds": -1},
        {"vertical_port_axis": "x"},
        {"vertical_port_aperture_width_um": 0},
        {"vertical_port_waist_radius_um": np.inf},
    ],
)
def test_invalid_artifact_settings(options):
    with pytest.raises(ValueError):
        ArtifactSimulation(**options)


@pytest.mark.parametrize("axis", ["+z", "-z"])
@pytest.mark.parametrize("polarization", ["vertical_te", "vertical_tm"])
@pytest.mark.parametrize("explicit_aperture", [False, True])
def test_vertical_source_and_monitor_are_oppositely_directed(
    fdtd_pdk_module, axis, polarization, explicit_aperture
):
    fdtd_pdk_module.PDK.activate()
    component = fdtd_pdk_module.PDK.cells["vertical_coupler"]()
    component.ports["o2"].port_type = polarization
    simulation = ArtifactSimulation(
        pdk=fdtd_pdk_module,
        vertical_port_axis=axis,
        vertical_port_aperture_width_um=0.8 if explicit_aperture else None,
        vertical_port_waist_radius_um=0.3 if explicit_aperture else None,
    )
    simulation.geometry(component)
    source, monitor = simulation._vertical_port_configs(
        simulation._material_snapshots()
    )["o2"]
    sign = 1 if axis == "+z" else -1
    assert source.propagation_direction == (0, 0, -sign)
    assert monitor.fiber_mode is not None
    assert monitor.fiber_mode.propagation_direction == (0, 0, sign)
    assert source.region_min == monitor.region_min
    assert source.region_max == monitor.region_max
    assert source.e_polarization == (
        (0, 1, 0) if polarization == "vertical_te" else (1, 0, 0)
    )
    assert source.waist_radius == pytest.approx(300 if explicit_aperture else 500)
    simulation.vertical_port_aperture_width_um = 100
    with pytest.raises(FDTDConfigError, match="aperture exceeds"):
        simulation._vertical_port_configs(simulation._material_snapshots())


def test_unconfigured_or_unknown_artifact_inputs(fdtd_pdk_module):
    simulation = ArtifactSimulation(pdk=fdtd_pdk_module)
    with pytest.raises(FDTDGeometryError, match="No geometry"):
        _ = simulation.resolved
    with pytest.raises(FDTDConfigError, match="default_port is required"):
        simulation._selected_port_name()
    simulation.geometry("straight")
    simulation.default_port = "absent"
    with pytest.raises(FDTDConfigError, match="not present"):
        simulation._selected_port_name()
    simulation.background_material = "Si"
    assert set(simulation._material_snapshots()) == {"Si"}
    simulation.background_material = "unknown-material"
    with pytest.raises(FDTDConfigError, match="Could not resolve background"):
        simulation._material_snapshots()
    with pytest.raises(FDTDConfigError, match="Unsupported vertical"):
        simulation._vertical_polarization(
            SimpleNamespace(orientation=0, port_type="optical")
        )
