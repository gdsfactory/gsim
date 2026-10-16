"""Exercise local runtime selection without launching Palace or MPI."""

import io
import shutil
import subprocess
from unittest.mock import Mock, mock_open

import pytest

from gsim.palace import DrivenSim, base, runtime
from tests.palace.test_run_local import _setup_sim


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "physical id : 0\ncore id : 0\nphysical id : 0\ncore id : 0\n"
            "physical id : 1\ncore id : 0",
            2,
        ),
        ("core id : 0\n", 6),
    ],
)
def test_physical_cpu_count(content, expected, monkeypatch):
    monkeypatch.setattr("builtins.open", mock_open(read_data=content))
    monkeypatch.setattr(base.os, "cpu_count", lambda: 6)
    assert base._count_physical_cpus() == expected


@pytest.mark.parametrize(
    "missing",
    ["output", "config", "mesh", "executable", "permission", "sif", "apptainer"],
)
def test_missing_local_inputs(tmp_path, monkeypatch, missing):
    sim = DrivenSim()
    monkeypatch.setattr(base, "_count_physical_cpus", lambda: 2)
    monkeypatch.delenv("PALACE_SIF", raising=False)
    monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)
    options = {"palace_executable": tmp_path / "palace", "verbose": False}
    if missing != "output":
        _setup_sim(sim, tmp_path / "simulation")
    if missing in {"config", "mesh"}:
        assert sim.output_dir is not None
        (
            sim.output_dir / ("config.json" if missing == "config" else "palace.msh")
        ).unlink()
    elif missing == "permission":
        options["palace_executable"].write_text("not executable")
        monkeypatch.setattr(base.os, "access", lambda *_args: False)
    elif missing in {"sif", "apptainer"}:
        options = {
            "use_apptainer": True,
            "palace_sif_path": tmp_path / "Palace.sif",
            "verbose": False,
        }
        if missing == "apptainer":
            options["palace_sif_path"].touch()
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)) as error:
        sim.run_local(**options)
    expected = {
        "output": "Output directory",
        "config": "Config file",
        "mesh": "Mesh file",
        "executable": "executable not found",
        "permission": "not executable",
        "sif": "SIF file",
        "apptainer": "Apptainer not found",
    }[missing]
    assert expected in str(error.value)


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("prior", ["", "/existing"])
def test_bundled_runtime_receives_libraries(tmp_path, monkeypatch, verbose, prior):
    sim = DrivenSim()
    _setup_sim(sim, tmp_path / "simulation")
    binary = tmp_path / "palace"
    binary.write_text("launcher")
    binary.chmod(0o755)
    library = tmp_path / "lib"
    library.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LD_LIBRARY_PATH", prior)
    monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)
    monkeypatch.delenv("PALACE_SIF", raising=False)
    monkeypatch.setattr(base, "_count_physical_cpus", lambda: 2)
    monkeypatch.setattr(runtime, "resolve_palace_binary", lambda: binary)
    monkeypatch.setattr(runtime, "resolve_palace_library_dir", lambda: library)
    monkeypatch.setattr(runtime, "_palace_cpu_available", lambda: False)
    monkeypatch.setattr(runtime, "_palace_toolkit_available", lambda: False)
    monkeypatch.setattr(runtime, "_cached_binary", lambda: binary)
    process = Mock()
    process.stdout = io.StringIO("Palace finished\n\n")
    process.wait.return_value = 0
    popen = Mock()
    popen.return_value.__enter__ = Mock(return_value=process)
    popen.return_value.__exit__ = Mock(return_value=False)
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "Palace 1.0", ""))
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "run", run)
    sim.run_local(use_apptainer=False, verbose=verbose)
    invocation = popen.call_args if verbose else run.call_args
    assert invocation.kwargs["env"]["LD_LIBRARY_PATH"] == str(library) + (
        f":{prior}" if prior else ""
    )
    assert invocation.args[0][0] == str(binary)
