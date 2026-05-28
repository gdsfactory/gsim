"""Tests for local Palace execution wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from gsim.palace import DrivenSim


def test_run_local_accepts_relative_local_executable(monkeypatch, tmp_path):
    """A relative executable should work without passing use_apptainer=False."""
    output_dir = tmp_path / "sim"
    postpro_dir = output_dir / "output" / "palace"
    output_dir.mkdir(parents=True)
    postpro_dir.mkdir(parents=True)

    # Required inputs checked by run_local before launching Palace.
    (output_dir / "config.json").write_text("{}")
    (output_dir / "palace.msh").write_text("mesh")

    # Simulate a locally built Palace binary under the current working directory.
    local_bin_dir = tmp_path / "bin"
    local_bin_dir.mkdir()
    local_palace = local_bin_dir / "palace"
    local_palace.write_text("#!/bin/sh\nexit 0\n")
    local_palace.chmod(0o755)

    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd, _check, _capture_output, _text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        assert _check is True
        assert _capture_output is True
        assert _text is True
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    sim = DrivenSim()
    sim.set_output_dir(output_dir)

    result = sim.run_local(
        palace_executable="./bin/palace",
        num_processes=1,
        verbose=False,
    )

    assert isinstance(result, dict)
    cmd = cast(list[str], captured["cmd"])
    assert isinstance(cmd, list)
    assert Path(cmd[0]) == local_palace.resolve()
    assert Path(cmd[0]).is_absolute()
    assert captured["cwd"] == output_dir


def test_run_local_no_args_discovers_bin_palace(monkeypatch, tmp_path):
    """run_local() without options should discover ./bin/palace."""
    output_dir = tmp_path / "sim"
    postpro_dir = output_dir / "output" / "palace"
    output_dir.mkdir(parents=True)
    postpro_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text("{}")
    (output_dir / "palace.msh").write_text("mesh")

    local_bin_dir = tmp_path / "bin"
    local_bin_dir.mkdir()
    local_palace = local_bin_dir / "palace"
    local_palace.write_text("#!/bin/sh\nexit 0\n")
    local_palace.chmod(0o755)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PALACE_SIF", raising=False)
    monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd, _check, _capture_output, _text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    sim = DrivenSim()
    sim.set_output_dir(output_dir)
    result = sim.run_local(num_processes=1, verbose=False)

    assert isinstance(result, dict)
    cmd = cast(list[str], captured["cmd"])
    assert isinstance(cmd, list)
    assert Path(cmd[0]) == local_palace.resolve()
    assert captured["cwd"] == output_dir


def test_run_local_no_args_prefers_local_sif(monkeypatch, tmp_path):
    """run_local() should use Apptainer when a local SIF is auto-discovered."""
    output_dir = tmp_path / "sim"
    postpro_dir = output_dir / "output" / "palace"
    output_dir.mkdir(parents=True)
    postpro_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text("{}")
    (output_dir / "palace.msh").write_text("mesh")

    local_sif = tmp_path / "Palace.sif"
    local_sif.write_text("fake")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PALACE_SIF", raising=False)
    monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/apptainer")

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd, _check, _capture_output, _text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    sim = DrivenSim()
    sim.set_output_dir(output_dir)
    result = sim.run_local(num_processes=1, verbose=False)

    assert isinstance(result, dict)
    cmd = cast(list[str], captured["cmd"])
    assert isinstance(cmd, list)
    assert cmd[0] == "apptainer"
    assert cmd[1] == "run"
    assert Path(cmd[2]) == local_sif.resolve()
    assert captured["cwd"] == output_dir
