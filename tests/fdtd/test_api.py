"""Tests for the concern-based public FDTD API and cloud lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import pytest
from pydantic import ValidationError

from gsim import fdtd, gcloud


def test_gpdk_material_resolves_through_builtin_material_card() -> None:
    gf.gpdk.PDK.activate()
    simulation = fdtd.Simulation(pdk=gf.gpdk.PDK)

    resolved = simulation.geometry(gf.gpdk.PDK.get_component("mmi1x2"))

    assert resolved.materials["si"].card.name == "si"
    assert resolved.materials["si"].card.optical is not None


def test_material_override_participates_in_gpdk_resolution() -> None:
    gf.gpdk.PDK.activate()
    simulation = fdtd.Simulation(pdk=gf.gpdk.PDK)
    simulation.materials(overrides={"si": 3.47})

    resolved = simulation.geometry(gf.gpdk.PDK.get_component("mmi1x2"))

    assert resolved.materials["si"].refractive_index == 3.47


def test_concerns_batch_validate_without_partial_updates() -> None:
    simulation = fdtd.Simulation()

    assert simulation.source.num_wavelengths == 101
    assert simulation.solver.cell_size_nm == 60
    assert simulation.geometry.mesh_size_nm == 1000
    assert simulation.geometry.geometry_tolerance_nm == 10

    simulation.source(port="o2", wavelength_um=1.31, wavelength_span_um=0.04)
    simulation.domain(
        padding_um=0.75,
        pml_cells=16,
        x_bounds=(-0.25, 2.25),
        y_bounds=(-1.0, 1.0),
        z_bounds=(-0.5, 0.75),
    )
    simulation.solver(cell_size_nm=40, energy_decay_fraction=1e-5)
    simulation.geometry.mesh_size_nm = 750
    simulation.geometry.geometry_tolerance_nm = 20

    assert isinstance(simulation.source, fdtd.PortSource)
    assert simulation.source.port == "o2"
    assert simulation.domain.padding_um == 0.75
    assert simulation.domain.x_bounds == (-0.25, 2.25)
    assert simulation.domain.y_bounds == (-1.0, 1.0)
    assert simulation.domain.z_bounds == (-0.5, 0.75)
    assert simulation.solver.cell_size_nm == 40
    assert simulation.geometry.mesh_size_nm == 750
    assert simulation.geometry.geometry_tolerance_nm == 20

    with pytest.raises(ValidationError):
        simulation.solver(cell_size_nm=-1, max_wall_seconds=20)
    assert simulation.solver.cell_size_nm == 40
    assert simulation.solver.max_wall_seconds == 3600

    with pytest.raises(ValidationError):
        simulation.geometry.geometry_tolerance_nm = 31
    assert simulation.geometry.geometry_tolerance_nm == 20

    with pytest.raises(ValidationError, match="lower bound"):
        simulation.domain(z_bounds=(1.0, -1.0))
    assert simulation.domain.z_bounds == (-0.5, 0.75)


def test_dipole_material_override_and_plane_monitor_serialize(
    tmp_path: Path,
    fdtd_pdk_module,
) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight", settings={"length": 2}, mesh_size_nm=750)
    simulation.materials(overrides={"Si": 3.4})
    simulation.source = fdtd.DipoleSource(
        position_um=(1, 0, 0.11),
        current_axis="z",
        wavelength_um=1.55,
        wavelength_span_um=0.02,
        num_wavelengths=3,
    )
    simulation.monitors.add_plane(
        "top",
        center_um=(1, 0, 0.5),
        size_um=(2, 1, 0),
        normal="+z",
        heatmap=fdtd.Heatmap(quantity="intensity", wavelengths_um=[1.55]),
    )

    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text(encoding="utf8"))

    assert document["materials"]["Si"]["refractive_index"] == 3.4
    assert document["excitation"]["type"] == "dipole"
    assert document["excitation"]["dipole"] == {
        "position": [1000.0, 0.0, 110.0],
        "current_axis": "z",
    }
    assert document["monitors"][0]["region_min"] == [0.0, -500.0, 500.0]
    assert document["monitors"][0]["heatmap"] == {
        "quantity": "intensity",
        "wavelengths": [1550.0],
    }


def test_line_current_and_gaussian_aperture_validation() -> None:
    source = fdtd.LineCurrentSource(
        position_um=(0, 0, 0),
        line_axis="y",
        current_axis="z",
        length_um=0.5,
    )
    assert source.type == "line_current"

    with pytest.raises(ValidationError, match="aperture_normal"):
        fdtd.GaussianBeamSource(
            center_um=(0, 0, 1),
            size_um=(2, 2, 1),
            aperture_normal="-z",
            propagation_direction=(0, 0, -1),
            e_polarization=(1, 0, 0),
            focal_point_um=(0, 0, 0),
            waist_radius_um=0.5,
        )


def test_monitor_collection_rejects_duplicates_and_supports_removal() -> None:
    monitors = fdtd.Monitors()
    monitor = monitors.add_plane(
        "top", center_um=(0, 0, 1), size_um=(2, 2, 0), normal="+z"
    )
    with pytest.raises(ValueError, match="already exists"):
        monitors.add(monitor)

    assert monitors.remove("top") is monitor
    monitors.add(monitor)
    monitors.clear()
    assert len(monitors) == 0


def test_run_without_wait_uses_fine_grained_cloud_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = fdtd.Simulation()
    calls: list[tuple] = []

    def write_files(directory: Path) -> None:
        directory.joinpath("config.json").write_text("{}", encoding="utf8")

    monkeypatch.setattr(simulation, "write", write_files)
    monkeypatch.setattr(
        gcloud,
        "upload",
        lambda _directory, job_type, **kwargs: calls.append(
            ("upload", job_type, kwargs["input_hash"])
        )
        or "job-123",
    )
    monkeypatch.setattr(
        gcloud,
        "start",
        lambda job_id, **_kwargs: calls.append(("start", job_id)),
    )

    job_id = simulation.run(wait=False, verbose="quiet")

    assert job_id == "job-123"
    assert simulation.job_id == "job-123"
    assert calls[0][0:2] == ("upload", "fdtd")
    assert calls[0][2].startswith("sha256:")
    assert calls[1] == ("start", "job-123")
