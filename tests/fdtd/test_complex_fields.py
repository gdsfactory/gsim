"""Complex-field requests, phase fidelity, metadata, and lazy sidecar failures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from gsim import fdtd
from gsim.fdtd.config import FieldsConfig
from gsim.gcloud import RunResult
from gsim.hashing import compute_input_hash
from tests.fdtd.test_results import _result_document


@pytest.mark.parametrize(
    ("model", "wavelength_key"),
    [(fdtd.Fields, "wavelengths_um"), (FieldsConfig, "wavelengths")],
)
def test_field_selections_are_explicit_and_case_insensitive(model, wavelength_key):
    request = model(components=["EX", "hY"], **{wavelength_key: [1.55]})
    assert request.components == ["ex", "hy"]
    assert model().model_dump(exclude_none=True) == {}


@pytest.mark.parametrize(
    "settings",
    [
        {"components": []},
        {"components": ["ex", "EX"]},
        {"components": ["intensity"]},
        {"components": [1]},
        {"wavelengths_um": []},
        {"wavelengths_um": [0]},
        {"wavelengths_um": [-1]},
        {"wavelengths_um": [float("nan")]},
        {"wavelengths_um": [float("inf")]},
        {"unknown": 1},
    ],
)
def test_invalid_field_requests_fail_early(settings):
    with pytest.raises(ValidationError):
        fdtd.Fields(**settings)


def test_fields_only_monitor_writes_solver_request_and_preserves_default_hash(
    tmp_path, fdtd_pdk_module
):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight", settings={"length": 2}, mesh_size_nm=750)
    simulation.materials(overrides={"Si": 3.4})
    simulation.source(
        port="o1", wavelength_um=1.55, wavelength_span_um=0.02, num_wavelengths=3
    )
    monitor = simulation.monitors.add_plane(
        "capture",
        center_um=(1, 0, 0.5),
        size_um=(2, 1, 0),
        normal="+z",
    )
    baseline = simulation.write(tmp_path / "baseline")
    assert "fields" not in json.loads(baseline.config_path.read_text())["monitors"][0]
    monitor.fields = fdtd.Fields(
        components=["EX", "ey", "hx", "hy"], wavelengths_um=[1.54, 1.56]
    )
    monitor.flux = False
    exported = simulation.write(tmp_path / "exported")
    config = fdtd.FDTDConfig.model_validate_json(exported.config_path.read_text())
    assert config.monitors[0].fields is not None
    assert config.monitors[0].fields.model_dump(exclude_none=True) == {
        "components": ["ex", "ey", "hx", "hy"],
        "wavelengths": [1540.0, 1560.0],
    }
    monitor.fields = fdtd.Fields()
    all_fields = simulation.write(tmp_path / "all-fields")
    assert json.loads(all_fields.config_path.read_text())["monitors"][0]["fields"] == {}
    monitor.flux = True
    monitor.fields = None
    restored = simulation.write(tmp_path / "restored")
    assert baseline.config_path.read_bytes() == restored.config_path.read_bytes()
    assert compute_input_hash(
        baseline.config_path.parent, "fdtd"
    ) == compute_input_hash(restored.config_path.parent, "fdtd")


def _write_fields(tmp_path: Path):
    document = _result_document()
    monitor = document["plane_monitors"]["top"]
    monitor["field_coordinates"] = {
        "u": [-980, 20, 1020],
        "v": [-480, 520],
        "normal_position": 520,
        "units": "nm",
        "axis_order": ["v", "u"],
        "grid_location": "primal",
    }
    # An analytic unit-amplitude plane phasor verifies phase sign and axis order.
    x, y = np.meshgrid(np.array([-0.98, 0.02, 1.02]), np.array([-0.48, 0.52]))
    expected = np.exp(-1j * (0.4 * x + 0.3 * y))
    np.save(tmp_path / "ey_0_real.npy", expected.real)
    np.save(tmp_path / "ey_0_imag.npy", expected.imag)
    monitor["fields"] = [
        {
            "component": "ey",
            "wavelength_nm": 1500.0,
            "frequency_hz": 2e14,
            "shape": [2, 3],
            "real_file": "ey_0_real.npy",
            "imag_file": "ey_0_imag.npy",
            "dtype": "float64",
            "normalization": "raw_dft",
            "dft_convention": "exp(-i*omega*t)",
            "phase_reference": "simulation_t0",
            "colocated": True,
            "field_unit_system": "solver_normalized",
            "solver_time_unit_seconds": 3.33564095198152e-18,
            "boundary_interpolation": "one_sided_clamp",
        }
    ]
    path = tmp_path / "sparams.json"
    path.write_text(json.dumps(document))
    return path, document, expected


def test_complex_fields_preserve_raw_phase_and_actual_coordinates(tmp_path):
    path, document, expected = _write_fields(tmp_path)
    result = fdtd.FDTDResult.from_file(path)
    monitor = result.monitors["top"]
    selected = monitor.field("EY", wavelength_um=1.5001)
    assert monitor.shape == (3, 2)
    assert selected.shape == (2, 3)
    np.testing.assert_allclose(selected.load(), expected)
    np.testing.assert_allclose(abs(selected.load()), 1)
    # A below-noise source sample must not mask or normalize raw fields.
    assert not result.valid[0]
    assert selected.metadata == document["plane_monitors"]["top"]["fields"][0]
    assert selected.frequency_hz == 2e14
    assert selected.coordinates is monitor.field_coordinates
    assert monitor.plane_position_um == 0.5
    assert monitor.field_coordinates.normal_position_um == 0.52
    np.testing.assert_allclose(monitor.field_coordinates.u_um, [-0.98, 0.02, 1.02])
    assert monitor.field_coordinates.axis_order == ("v", "u")
    with pytest.raises(KeyError, match="complex field"):
        monitor.field("ex", wavelength_um=1.5)


def test_complex_sidecars_load_lazily_through_download_file_mapping(tmp_path):
    path, _, expected = _write_fields(tmp_path)
    actual_real = tmp_path / "downloaded_real.npy"
    actual_imag = tmp_path / "downloaded_imag.npy"
    (tmp_path / "ey_0_real.npy").rename(actual_real)
    (tmp_path / "ey_0_imag.npy").rename(actual_imag)
    result = fdtd.FDTDResult.from_run_result(
        RunResult(
            sim_dir=tmp_path,
            job_name="fdtd-field-test",
            files={
                path.name: path,
                "ey_0_real.npy": actual_real,
                "ey_0_imag.npy": actual_imag,
            },
        )
    )
    selected = result.monitors["top"].field("ey", wavelength_um=1.5)
    np.testing.assert_allclose(selected.load(), expected)
    actual_real.unlink()
    assert selected.component == "ey"
    with pytest.raises(FileNotFoundError):
        selected.load()


def test_legacy_results_do_not_require_complex_field_metadata(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(_result_document()))
    monitor = fdtd.FDTDResult.from_file(path).monitors["top"]
    assert monitor.fields == []
    assert monitor.field_coordinates is None


@pytest.mark.parametrize("convention", [None, "", "unknown", "exp(+i*omega*t)"])
def test_unsupported_or_missing_dft_convention_is_rejected(tmp_path, convention):
    path, document, _ = _write_fields(tmp_path)
    entry = document["plane_monitors"]["top"]["fields"][0]
    if convention is None:
        entry.pop("dft_convention")
    else:
        entry["dft_convention"] = convention
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="dft_convention"):
        fdtd.FDTDResult.from_file(path)


def test_nearby_field_wavelengths_keep_distinct_sidecars(tmp_path):
    path, document, expected = _write_fields(tmp_path)
    monitor = document["plane_monitors"]["top"]
    first = monitor["fields"][0]
    first["wavelength_nm"] = 1550.00001
    second = {
        **first,
        "wavelength_nm": 1550.00002,
        "real_file": "ey_1_real.npy",
        "imag_file": "ey_1_imag.npy",
    }
    monitor["fields"].append(second)
    np.save(tmp_path / "ey_1_real.npy", 2 * expected.real)
    np.save(tmp_path / "ey_1_imag.npy", 2 * expected.imag)
    path.write_text(json.dumps(document))
    selected = (
        fdtd.FDTDResult.from_file(path)
        .monitors["top"]
        .field(
            "ey",
            wavelength_um=1.550000019,
        )
    )
    assert selected.real_file.name == "ey_1_real.npy"
    np.testing.assert_allclose(selected.load(), 2 * expected)


def test_complex_fields_accept_float64_sidecars_with_explicit_byte_order(tmp_path):
    path, _, expected = _write_fields(tmp_path)
    np.save(tmp_path / "ey_0_real.npy", expected.real.astype(">f8"))
    np.testing.assert_allclose(
        fdtd.FDTDResult.from_file(path).monitors["top"].fields[0].load(),
        expected,
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_coordinates",
        "coordinate_count",
        "axis_order",
        "units",
        "shape",
        "component",
        "path",
        "nonfinite_coordinate",
    ],
)
def test_invalid_complex_field_metadata_is_rejected(tmp_path, change):
    path, document, _ = _write_fields(tmp_path)
    monitor = document["plane_monitors"]["top"]
    if change == "missing_coordinates":
        del monitor["field_coordinates"]
    elif change == "coordinate_count":
        monitor["field_coordinates"]["u"] = [0]
    elif change == "axis_order":
        monitor["field_coordinates"]["axis_order"] = ["u", "v"]
    elif change == "units":
        monitor["field_coordinates"]["units"] = "unknown"
    elif change == "shape":
        monitor["fields"][0]["shape"] = [3, 2]
    elif change == "component":
        monitor["fields"][0]["component"] = "power"
    elif change == "path":
        monitor["fields"][0]["real_file"] = "../outside.npy"
    else:
        monitor["field_coordinates"]["u"][0] = float("nan")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        fdtd.FDTDResult.from_file(path)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((3, 2)),
        np.zeros((2, 3), dtype=int),
        np.full((2, 3), np.nan),
        np.zeros((2, 3), dtype=np.float32),
    ],
)
def test_invalid_complex_field_samples_fail_when_loaded(tmp_path, values):
    path, _, _ = _write_fields(tmp_path)
    np.save(tmp_path / "ey_0_imag.npy", values)
    selected = fdtd.FDTDResult.from_file(path).monitors["top"].fields[0]
    with pytest.raises(ValueError):
        selected.load()
