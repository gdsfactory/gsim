"""Decide which Palace a local run starts, without starting one."""

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from gsim.palace import runtime
from gsim.palace.runtime import PalaceLaunch, PalaceRuntime, resolve_launch, run_palace


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("PALACE_SIF", "PALACE_EXECUTABLE", "LD_LIBRARY_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "resolve_palace_runtime", lambda: None)


def test_an_explicit_executable_wins_over_a_sif(tmp_path):
    exe = _executable(tmp_path / "mine" / "palace")
    (tmp_path / "Palace.sif").touch()
    launch = resolve_launch(executable=exe, use_apptainer=True, search_roots=[tmp_path])
    assert launch == PalaceLaunch((str(exe),), launch.env)
    assert not launch.apptainer


def test_palace_executable_env_beats_a_local_bin(tmp_path, monkeypatch):
    env_exe = _executable(tmp_path / "env" / "palace")
    _executable(tmp_path / "bin" / "palace")
    monkeypatch.setenv("PALACE_EXECUTABLE", str(env_exe))
    assert resolve_launch(search_roots=[tmp_path]).binary == str(env_exe)


def test_a_local_bin_beats_the_resolved_runtime(tmp_path, monkeypatch):
    local = _executable(tmp_path / "bin" / "palace")
    bundled = _executable(tmp_path / "bundled" / "palace")
    monkeypatch.setattr(
        runtime, "resolve_palace_runtime", lambda: PalaceRuntime(bundled, "PATH")
    )
    assert resolve_launch(search_roots=[tmp_path]).binary == str(local)


def test_apptainer_runs_a_discovered_sif(tmp_path, monkeypatch):
    sif = tmp_path / "bin" / "Palace.sif"
    sif.parent.mkdir()
    sif.touch()
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    launch = resolve_launch(use_apptainer=True, search_roots=[tmp_path])
    assert launch.command == ("apptainer", "run", str(sif))
    assert launch.apptainer


@pytest.mark.parametrize(
    ("found", "expected"),
    [
        (PalaceRuntime(Path("bundled"), "gsim cached runtime", Path("/rt/lib")), True),
        (PalaceRuntime(Path("bundled"), "PALACE_BIN"), False),
    ],
)
def test_libraries_travel_with_the_binary_they_belong_to(
    tmp_path, monkeypatch, found, expected
):
    binary = _executable(tmp_path / "bundled")
    found = PalaceRuntime(binary, found.source, found.library_dir)
    monkeypatch.setattr(runtime, "resolve_palace_runtime", lambda: found)
    launch = resolve_launch(search_roots=[tmp_path / "nowhere"])
    assert launch.binary == str(binary)
    assert ("LD_LIBRARY_PATH" in launch.env) is expected
    if expected:
        assert launch.env["LD_LIBRARY_PATH"] == str(found.library_dir)


def test_nothing_found_falls_back_to_palace_on_path(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="executable not found: palace"):
        resolve_launch(search_roots=[tmp_path])


@pytest.mark.parametrize("verbose", [False, True])
def test_run_palace_builds_the_command_and_raises_on_failure(
    tmp_path, monkeypatch, verbose
):
    launch = PalaceLaunch(("palace",), {"A": "1"})
    failure = subprocess.CalledProcessError(134, ["palace"])
    process = Mock()
    process.stdout = iter(["solving\n"])
    process.wait.return_value = 134
    popen = Mock()
    popen.return_value.__enter__ = Mock(return_value=process)
    popen.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=failure))
    emitted: list[str] = []

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_palace(
            launch,
            cwd=tmp_path,
            num_processes=2,
            num_threads=4,
            verbose=verbose,
            emit=emitted.append,
        )

    assert error.value.returncode == 134
    if verbose:
        assert popen.call_args.args[0] == [
            "palace",
            "-np",
            "2",
            "-nt",
            "4",
            "config.json",
        ]
        assert popen.call_args.kwargs["env"] == {"A": "1"}
        assert popen.call_args.kwargs["encoding"] == "utf-8"
        assert popen.call_args.kwargs["errors"] == "replace"
        assert "solving" in emitted
        assert error.value.stderr == "solving"


def test_missing_launcher_leaves_no_empty_log(tmp_path):
    launch = PalaceLaunch((str(tmp_path / "missing-palace"),), {})
    target = tmp_path / "logs" / "solver.log"

    with pytest.raises(RuntimeError, match="Palace executable not found"):
        run_palace(
            launch, cwd=tmp_path, num_processes=1, verbose=False, log_path=target
        )

    assert not target.exists()


@pytest.mark.parametrize("failure", ["launcher", "cwd"])
def test_existing_log_survives_a_launch_that_never_starts(tmp_path, failure):
    target = tmp_path / "solver.log"
    target.write_text("previous run\n", encoding="utf-8")
    if failure == "launcher":
        launch = PalaceLaunch((str(tmp_path / "missing-palace"),), {})
        cwd = tmp_path
    else:
        launch = PalaceLaunch((sys.executable,), {})
        cwd = tmp_path / "missing"

    with pytest.raises(RuntimeError, match="Palace executable not found"):
        run_palace(launch, cwd=cwd, num_processes=1, verbose=True, log_path=target)

    assert target.read_text(encoding="utf-8") == "previous run\n"
