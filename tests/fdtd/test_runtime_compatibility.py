"""Explicit beam serialization and the historical FDTD import path."""

from __future__ import annotations

import json

from gsim import fdtd


def test_historical_simulation_import_exposes_public_workflow(fdtd_pdk_module):
    from gsim.fdtd.simulation import ArtifactSimulation, Simulation

    assert Simulation is fdtd.Simulation
    simulation = Simulation(pdk=fdtd_pdk_module)
    simulation.source(port="o1")
    assert simulation.source.port == "o1"
    legacy = ArtifactSimulation(pdk=fdtd_pdk_module, default_port="o1")
    assert legacy.geometry("straight", settings={"length": 2}).component is not None


def test_explicit_gaussian_beam_serializes_position_and_size(tmp_path, fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight", settings={"length": 2}, mesh_size_nm=750)
    simulation.source = fdtd.GaussianBeamSource(
        center_um=(1, -0.2, 0.8),
        size_um=(1.5, 0.6, 0),
        aperture_normal="-z",
        propagation_direction=(0, 0, -1),
        e_polarization=(1, 0, 0),
        focal_point_um=(1, -0.1, 0.11),
        waist_radius_um=0.4,
        refractive_index=1.4,
    )
    simulation.monitors.add_plane(
        "output",
        center_um=(1.5, 0.2, 0.1),
        size_um=(0, 0.8, 0.6),
        normal="+x",
    )
    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text())
    assert document["excitation"]["type"] == "gaussian_beam"
    assert document["excitation"]["gaussian_beam"] == {
        "region_min": [250, -500, 800],
        "region_max": [1750, 100, 800],
        "aperture_normal": "-z",
        "propagation_direction": [0, 0, -1],
        "e_polarization": [1, 0, 0],
        "focal_point": [1000, -100, 110],
        "waist_radius": 400,
        "refractive_index": 1.4,
    }
    assert document["monitors"][0]["region_min"] == [1500, -200, -200]
    assert document["monitors"][0]["region_max"] == [1500, 600, 400]
