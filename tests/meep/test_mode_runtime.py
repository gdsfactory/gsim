"""Exercise local mode orchestration with a deterministic MEEP boundary."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gsim.meep import Simulation as MeepSimulation
from gsim.meep import mode_solver
from gsim.meep.models.results import ModeResult
from gsim.meep.simulation import _precompute_cross_section_geometry


@pytest.mark.parametrize("z_margin", [0.5, (0, 1)])
def test_slab_solver_converts_coordinates_and_releases_runtime(
    index_context, backend, z_margin
):
    stack, _ = index_context
    z = np.array([-0.5, 0.5])
    result = mode_solver.solve_slab_mode(
        stack,
        1.55,
        field_z_grid=z,
        z_margin=z_margin,
        pml_thickness=0.5,
        background_material="clad",
        parity="EVEN_Y",
    )
    center = 0 if isinstance(z_margin, float) else 0.5
    np.testing.assert_allclose(result.fields["Ey"][:, 0], 3 * (z - center) + 1j)
    np.testing.assert_array_equal(result.z_grid, z)
    assert result.n_eff == pytest.approx(1.8)
    assert result.n_group == 4
    simulation = backend.instances[0]
    assert simulation.initialized == simulation.resets == 1
    assert simulation.default_material.epsilon == 2.25
    assert simulation.calls[0]["parity"] == 1
    assert [pml.direction for pml in simulation.boundary_layers] == ["X", "Z"]


@pytest.mark.parametrize("orientation", [0, 90])
@pytest.mark.parametrize("sidewall_angle", [0, 30])
@pytest.mark.parametrize("use_position", [False, True])
def test_cross_section_runtime_and_cloud_geometry_agree_on_core_dimensions(
    index_context, backend, orientation, sidewall_angle, use_position
):
    stack, component = index_context
    stack.layers["core"].sidewall_angle = sidewall_angle
    component.add_port(
        "cut", center=(0, 0), orientation=orientation, width=0.5, layer=(1, 0)
    )
    plane = "yz" if orientation == 0 else "xz"
    selection = {"position": (0, 0)} if use_position else {"port": "cut"}
    horizontal = np.array([-0.5, 0.5])
    result = mode_solver.solve_cross_section_mode(
        component,
        stack,
        wavelength=1.55,
        x_span=4,
        y_span=4,
        cross_section_plane=plane,
        **selection,
        z_margin=(0, 1),
        pml_thickness=0.5,
        **{"field_y_grid" if orientation == 0 else "field_x_grid": horizontal},
        field_z_grid=np.array([0.0, 0.5]),
    )
    simulation = backend.instances[0]
    assert simulation.initialized == simulation.resets == 1
    assert result.cross_section_plane == plane
    scale = 2 if plane == "yz" else 1
    np.testing.assert_allclose(
        result.fields["Ex"],
        scale * horizontal[None, :] + np.array([-1.5, 0])[:, None] + 1j,
    )
    core = next(shape for shape in simulation.geometry if shape.material.epsilon == 4)
    if sidewall_angle:
        assert core.height == 1
        assert core.sidewall_angle == pytest.approx(np.pi / 6)
    else:
        assert core.size.z == 1
        assert (core.size.y if plane == "yz" else core.size.x) == 2
    cloud = _precompute_cross_section_geometry(
        component,
        stack,
        port="cut",
        x_span=4,
        y_span=4,
        z_margin=(0, 1),
        pml_thickness=0.5,
    )
    block = next(block for block in cloud.blocks if block.material == "core")
    assert (block.horizontal_size, block.z_size, block.z_center) == (2, 1, 0)
    assert cloud.plane == plane
    assert cloud.cell_horizontal_span == 5


@pytest.mark.parametrize("bands", [None, [1, 2, 3]])
def test_band_sweep_reuses_simulation_and_skips_failed_bands(
    index_context, backend, bands
):
    backend.failed_bands.add(2)
    results = mode_solver.solve_slab_modes(index_context[0], 1.55, band_nums=bands)
    assert list(results) == ([1] if bands is None else [1, 3])
    assert len(backend.instances) == 1
    assert backend.instances[0].resets == 1


def test_wavelength_sweep_resolves_each_wavelength_and_resets_failed_simulations(
    index_context, backend, caplog
):
    backend.failed_frequencies.add(1 / 1.6)
    results = mode_solver.solve_slab_wavelength_sweep(
        index_context[0],
        [1.5, 1.6],
        field_z_grid=np.array([0.0, 0.5]),
        z_margin=(0, 1),
        pml_thickness=0.5,
        background_material="clad",
    )
    assert list(results) == [1.5]
    assert results[1.5].n_eff == pytest.approx(1.8)
    assert [simulation.resets for simulation in backend.instances] == [1, 1]
    assert "failed at lambda=1.6000" in caplog.text


def test_failed_single_solve_still_releases_simulation(index_context, backend):
    backend.failed_bands.add(1)
    with pytest.raises(RuntimeError, match="get_eigenmode returned None"):
        mode_solver.solve_slab_mode(index_context[0], 1.55)
    assert backend.instances[0].resets == 1


def test_unphysical_mode_reports_all_relevant_diagnostics(index_context, caplog):
    result = ModeResult(
        n_eff=3, wavelength=1.55, frequency=1 / 1.55, n_group=-1, kdom=[1, 0, 1]
    )

    with caplog.at_level("INFO", logger=mode_solver.__name__):
        mode_solver._validate_mode(
            result,
            mode_solver._resolve_materials_cached(set()),
            index_context[0],
            is_slab=True,
        )
    assert all(
        message in caplog.text
        for message in ["exceeds maximum core index", "significant kz", "is negative"]
    )


@pytest.mark.parametrize(
    ("wavelengths", "bands", "cross_section"),
    [
        ([1.55], 1, False),
        ([1.5, 1.6], 1, False),
        ([1.55], 2, False),
        ([1.5, 1.6], 2, False),
        ([1.55], 1, True),
    ],
)
def test_simulation_dispatch_and_serialized_mode_config(
    index_context, backend, tmp_path, wavelengths, bands, cross_section
):
    stack, component = index_context
    component.add_port("cut", center=(0, 0), orientation=0, width=0.5, layer=(1, 0))
    simulation = MeepSimulation()
    simulation.geometry.stack = stack
    simulation.geometry.component = component
    simulation.materials = {"core": 4, "clad": 2.25}
    simulation.mode_solver(wavelengths=wavelengths, num_bands=bands, n_field_z=3)
    if cross_section:
        simulation.mode_solver(port="cut", y_span=4, n_field_y=3)

    sweep = simulation.solve_modes_local()
    assert len(sweep.results) == len(wavelengths) * bands
    assert all(mode.n_eff == pytest.approx(1.8) for mode in sweep.results)
    assert all(mode.domain_config is not None for mode in sweep.results)
    mode = sweep.results[0]
    mode.stack = mode.component = mode.port_or_position = None
    simulation._enrich_mode_results(sweep)
    assert mode.stack is stack
    assert mode.component is component
    assert all(runtime.resets == 1 for runtime in backend.instances)
    simulation.write_mode_solver_config(tmp_path)
    config = json.loads((tmp_path / "mode_solver_config.json").read_text())
    assert config["wavelengths"] == wavelengths
    assert config["bands"] == list(range(1, bands + 1))
    assert bool(config.get("cross_section_geometry")) == cross_section
    compile((tmp_path / "run_meep.py").read_text(), "run_meep.py", "exec")


@pytest.mark.parametrize(
    "options",
    [
        {"port": "missing"},
        {"cross_section_plane": "invalid"},
        {"y_span": None},
        {"component": None},
        {"stack": None},
        {"port": None},
        {"port": None, "position": (0, 0), "x_span": None},
    ],
)
def test_cross_section_rejects_incomplete_configuration(
    index_context, backend, options
):
    stack, component = index_context
    component.add_port("cut", center=(0, 0), orientation=0, width=0.5, layer=(1, 0))
    kwargs = {
        "component": component,
        "stack": stack,
        "port": "cut",
        "y_span": 4,
        "wavelength": 1.55,
    } | options
    with pytest.raises(ValueError):
        mode_solver.solve_cross_section_mode(**kwargs)
    assert backend.instances == []


@pytest.mark.parametrize(
    "outcome", ["csv", "preview", "failed", "missing-executable", "missing-mpi"]
)
def test_local_execution_collects_results_and_reports_process_failures(
    index_context, monkeypatch, tmp_path, outcome
):
    stack, component = index_context
    component.add_port("cut", center=(0, 0), orientation=0, width=0.5, layer=(1, 0))
    simulation = MeepSimulation()
    simulation.geometry.stack = stack
    simulation.geometry.component = component
    simulation.materials = {"core": 4, "clad": 2.25}
    simulation.source.port = "cut"

    def execute(command, **kwargs):
        assert kwargs["cwd"] == tmp_path
        assert command[-2:] == ["meep-python", str(tmp_path / "run_meep.py")]
        if outcome == "failed":
            raise subprocess.CalledProcessError(
                2, command, output="solver output", stderr="solver error"
            )
        if outcome == "missing-executable":
            raise FileNotFoundError("missing interpreter")
        if outcome == "csv":
            (tmp_path / "s_parameters.csv").write_text(
                "wavelength,S21_mag,S21_phase\n1.55,0.5,90\n"
            )
        return SimpleNamespace(
            stdout="finished", stderr="warning: diagnostic\nsolver information"
        )

    runner = Mock(side_effect=execute)
    monkeypatch.setattr(subprocess, "run", runner)
    monkeypatch.setattr(
        "shutil.which", lambda _name: None if outcome == "missing-mpi" else "mpirun"
    )
    if outcome in {"failed", "missing-executable", "missing-mpi"}:
        expected = {
            "failed": "solver error",
            "missing-executable": "Python executable not found",
            "missing-mpi": "mpirun not found",
        }[outcome]
        with pytest.raises(RuntimeError, match=expected):
            simulation.run_local(
                tmp_path, python_executable="meep-python", num_processes=2
            )
    else:
        result = simulation.run_local(
            tmp_path,
            python_executable="meep-python",
            num_processes=1 if outcome == "csv" else 2,
        )
        assert result.wavelengths == ([1.55] if outcome == "csv" else [])
        if outcome == "csv":
            np.testing.assert_allclose(result.s_params["S21"], [0.5j], atol=1e-15)
