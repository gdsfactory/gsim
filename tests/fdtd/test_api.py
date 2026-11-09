"""Tests for the concern-based public FDTD API and cloud lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import pytest
from pydantic import ValidationError

from gsim import fdtd, gcloud


@pytest.mark.parametrize(
    ("background", "expected_name", "expected_index"),
    [(None, "Air", 1.0), ("SiO2", "SiO2", 1.4440236217)],
)
@pytest.mark.parametrize("explicit_bounds", [False, True])
def test_background_default_and_explicit_silica_export(
    tmp_path,
    fdtd_pdk_module,
    background,
    expected_name,
    expected_index,
    explicit_bounds,
) -> None:
    """Default space is air; explicit silica preserves the former environment."""
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    if background is not None:
        simulation.materials(background=background)
    if explicit_bounds:
        simulation.domain(z_bounds=(-0.5, 0.8))
    simulation.geometry("straight", mesh_size_nm=750)
    simulation.source(port="o1")

    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text())

    assert simulation.materials.background == expected_name
    assert document["geometry"]["volumes"]["background"]["material"] == expected_name
    assert document["background_refractive_index"] == pytest.approx(expected_index)
    assert document["materials"][expected_name]["refractive_index"] == pytest.approx(
        expected_index
    )
    bounds = document["grid"]["non_pml_bounds"]
    assert (bounds["min"][2], bounds["max"][2]) == (
        (-500, 800)
        if explicit_bounds
        else (-1000, 1220 if background is None else 1000)
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            {"kind": "mode_order", "mode_order": 2},
            {"kind": "mode_order", "mode_order": 2},
        ),
        (
            {
                "kind": "te_like",
                "te_axis": "z",
                "num_candidate_modes": 6,
                "min_polarization_fraction": 0.7,
            },
            {
                "kind": "te_like",
                "te_axis": "z",
                "num_candidate_modes": 6,
                "min_polarization_fraction": 0.7,
            },
        ),
    ],
)
def test_guided_mode_selection_serializes(
    tmp_path: Path, fdtd_pdk_module, mode: dict, expected: dict
) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight", mesh_size_nm=750)
    simulation.source(port="o1", mode=mode)

    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text(encoding="utf8"))

    source = simulation.source
    assert isinstance(source, fdtd.PortSource)
    assert isinstance(source.mode, fdtd.ModeSelection)
    assert document["excitation"]["mode"] == expected


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ({"kind": "mode_order"}, "mode_order is required"),
        ({"kind": "mode_order", "mode_order": 0}, "greater than or equal to 1"),
        ({"kind": "mode_order", "mode_order": 2, "te_axis": "z"}, "polarization"),
        ({"kind": "te_like"}, "te_axis is required"),
        ({"kind": "fundamental", "mode_order": 2}, "takes no additional"),
    ],
)
def test_invalid_guided_mode_selection(mode: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        fdtd.ModeSelection.model_validate(mode)


def test_mode_selection_requires_guided_port(tmp_path: Path, fdtd_pdk_module) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("vertical_coupler", mesh_size_nm=750)
    simulation.source(port="o2", mode={"kind": "mode_order", "mode_order": 2})

    with pytest.raises(fdtd.FDTDConfigError, match="guided PDK port"):
        simulation.write(tmp_path)


def test_mode_selection_axis_must_be_transverse(
    tmp_path: Path, fdtd_pdk_module
) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("straight", mesh_size_nm=750)
    simulation.source(port="o1", mode={"kind": "te_like", "te_axis": "x"})

    with pytest.raises(fdtd.FDTDConfigError, match="te_axis must be transverse"):
        simulation.write(tmp_path)


@pytest.mark.parametrize("kind", ["te_like", "tm_like"])
@pytest.mark.parametrize(
    ("source_port", "te_axis", "invalid_port"),
    [("o1", "y", "o2"), ("o2", "x", "o1")],
)
def test_mode_selection_axis_must_be_transverse_to_every_guided_port(
    tmp_path: Path,
    fdtd_pdk_module,
    kind: str,
    source_port: str,
    te_axis: str,
    invalid_port: str,
) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("bend", mesh_size_nm=750)
    simulation.source(port=source_port, mode={"kind": kind, "te_axis": te_axis})

    with pytest.raises(
        fdtd.FDTDConfigError, match=f"te_axis.*{te_axis!r}.*{invalid_port!r}"
    ):
        simulation.write(tmp_path)
    assert not (tmp_path / "config.json").exists()


@pytest.mark.parametrize("kind", ["te_like", "tm_like"])
@pytest.mark.parametrize(
    ("component", "te_axis"),
    [("straight", "y"), ("bend", "z"), ("vertical_coupler", "z")],
)
def test_mode_selection_axis_transverse_to_guided_ports_serializes(
    tmp_path: Path,
    fdtd_pdk_module,
    kind: str,
    component: str,
    te_axis: str,
) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry(component, mesh_size_nm=750)
    mode = {"kind": kind, "te_axis": te_axis}
    simulation.source(port="o1", mode=mode)

    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text(encoding="utf8"))

    assert document["excitation"]["mode"] == mode
    if component == "vertical_coupler":
        assert set(document["geometry"]["ports"]) == {"o1"}
        assert document["monitors"][0]["name"] == "o2"


@pytest.mark.parametrize(
    "mode", [None, {"kind": "fundamental"}, {"kind": "mode_order", "mode_order": 2}]
)
def test_bend_mode_selection_without_polarization_axis_serializes(
    tmp_path: Path,
    fdtd_pdk_module,
    mode: dict | None,
) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.geometry("bend", mesh_size_nm=750)
    simulation.source(port="o1", mode=mode)

    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text(encoding="utf8"))

    assert document["excitation"].get("mode") == mode
    assert set(document["geometry"]["ports"]) == {"o1", "o2"}


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
        padding=0.75,
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
    assert simulation.domain.padding == 0.75
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


_PLANE = {"name": "out", "size_um": (0, 1, 1), "normal": "+x"}


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (fdtd.Solver, lambda value: {"max_wall_seconds": value}),
        (fdtd.PortSource, lambda value: {"amplitude": value}),
        (fdtd.Material, lambda value: {"refractive_index": value}),
        (
            fdtd.PlaneMonitor,
            lambda value: {**_PLANE, "center_um": (value, 0, 0)},
        ),
        (fdtd.Domain, lambda value: {"padding": value}),
    ],
    ids=["solver", "source", "material", "plane_monitor", "domain"],
)
def test_concern_models_reject_non_finite_numbers(model, kwargs, value) -> None:
    with pytest.raises(ValidationError, match="finite number"):
        model(**kwargs(value))


def test_concern_models_reject_non_finite_assignment() -> None:
    simulation = fdtd.Simulation()

    with pytest.raises(ValidationError, match="finite number"):
        simulation.solver.max_wall_seconds = float("inf")
    with pytest.raises(ValidationError, match="finite number"):
        simulation.source(amplitude=float("nan"))

    assert simulation.solver.max_wall_seconds == 3600.0
    assert simulation.source.amplitude == 1.0


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
