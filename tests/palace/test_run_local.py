"""Tests for local Palace execution wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from gsim.palace import BoundaryModeSim, DrivenSim
from gsim.palace.base import _recommend_parallel


def _mesh_result(elements: int, tetrahedra: int = 0, bbox: dict | None = None):
    """A minimal MeshResult-like object exposing mesh_stats."""
    stats: dict = {
        "elements": elements,
        "nodes": elements,
    }
    if tetrahedra:
        stats["tetrahedra"] = tetrahedra
    if bbox is not None:
        stats["bbox"] = bbox
    return SimpleNamespace(mesh_stats=stats, groups={})


def _setup_sim(sim, output_dir: Path) -> None:
    postpro_dir = output_dir / "output" / "palace"
    output_dir.mkdir(parents=True)
    postpro_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text("{}")
    (output_dir / "palace.msh").write_text("mesh")
    sim.set_output_dir(output_dir)


def _setup_local_palace(monkeypatch, tmp_path: Path) -> None:
    """Provide a fake ./bin/palace and resolve it from cwd."""
    local_bin_dir = tmp_path / "bin"
    local_bin_dir.mkdir()
    local_palace = local_bin_dir / "palace"
    local_palace.write_text("#!/bin/sh\nexit 0\n")
    local_palace.chmod(0o755)
    monkeypatch.chdir(tmp_path)


def test_recommend_parallel_2d_uses_single_rank_openmp(monkeypatch):
    """2D mode analysis defaults to 1 MPI rank + OpenMP threads."""
    monkeypatch.setattr("gsim.palace.base._count_physical_cpus", lambda: 16)
    procs, threads = _recommend_parallel(
        {"elements": 50_000}, "boundarymode", None, None
    )
    assert procs == 1
    assert threads == 16


def test_recommend_parallel_3d_capped_by_dofs(monkeypatch):
    """3D runs are capped by problem size and never exceed 4 ranks."""
    monkeypatch.setattr("gsim.palace.base._count_physical_cpus", lambda: 16)
    # ~2.5M DOFs, enough for several ranks, but still capped at 4.
    procs, threads = _recommend_parallel(
        {"elements": 500_000, "tetrahedra": 500_000}, "driven", None, None
    )
    assert procs == 4
    assert threads is None

    # Tiny 3D problem: only 1 rank.
    procs, _ = _recommend_parallel(
        {"elements": 1_000, "tetrahedra": 1_000}, "driven", None, None
    )
    assert procs == 1


def test_recommend_parallel_respects_explicit_processes(monkeypatch):
    """An explicit num_processes is kept unchanged."""
    monkeypatch.setattr("gsim.palace.base._count_physical_cpus", lambda: 16)
    procs, threads = _recommend_parallel({"elements": 50_000}, "boundarymode", 2, None)
    assert procs == 2
    assert threads is None

    procs, threads = _recommend_parallel({"elements": 50_000}, "boundarymode", 1, None)
    assert procs == 1
    assert threads == 16  # serial run defaults threads to cores


def test_run_local_boundarymode_defaults_to_single_rank(monkeypatch, tmp_path):
    """BoundaryModeSim.run_local() without args uses -np 1 and -nt <cpus>."""
    _setup_local_palace(monkeypatch, tmp_path)
    output_dir = tmp_path / "sim"

    monkeypatch.delenv("PALACE_SIF", raising=False)
    monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)
    monkeypatch.setattr("gsim.palace.base._count_physical_cpus", lambda: 8)

    captured: dict[str, object] = {}

    def _fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("subprocess.run", _fake_run)

    sim = BoundaryModeSim()
    sim._last_mesh_result = _mesh_result(50_000)
    _setup_sim(sim, output_dir)

    result = sim.run_local(verbose=False)

    assert isinstance(result, dict)
    cmd = cast(list[str], captured["cmd"])
    assert "-np" in cmd
    assert "1" in cmd[cmd.index("-np") + 1 :][:1]
    assert "-nt" in cmd
    assert "8" in cmd[cmd.index("-nt") + 1 :][:1]


def test_run_local_explicit_large_processes_warns(monkeypatch, tmp_path, caplog):
    """Requesting too many ranks for a 2D problem logs a warning."""
    _setup_local_palace(monkeypatch, tmp_path)
    output_dir = tmp_path / "sim"
    monkeypatch.delenv("PALACE_SIF", raising=False)
    monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)
    monkeypatch.setattr("gsim.palace.base._count_physical_cpus", lambda: 16)

    captured: dict[str, object] = {}

    def _fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("subprocess.run", _fake_run)

    sim = BoundaryModeSim()
    sim._last_mesh_result = _mesh_result(50_000)
    _setup_sim(sim, output_dir)

    with caplog.at_level("WARNING", logger="gsim.palace.base"):
        sim.run_local(num_processes=8, verbose=False)

    assert any("does not scale beyond 1 rank" in r.message for r in caplog.records)
    # The explicit request is still respected.
    cmd = cast(list[str], captured["cmd"])
    assert "8" in cmd[cmd.index("-np") + 1 :][:1]


def test_run_local_3d_caps_default_processes(monkeypatch, tmp_path):
    """A 3D DrivenSim default is capped to 4 ranks regardless of cores."""
    _setup_local_palace(monkeypatch, tmp_path)
    output_dir = tmp_path / "sim"
    monkeypatch.delenv("PALACE_SIF", raising=False)
    monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)
    monkeypatch.setattr("gsim.palace.base._count_physical_cpus", lambda: 32)

    captured: dict[str, object] = {}

    def _fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("subprocess.run", _fake_run)

    sim = DrivenSim()
    sim._last_mesh_result = _mesh_result(1_000_000, tetrahedra=1_000_000)
    _setup_sim(sim, output_dir)

    result = sim.run_local(verbose=False)

    assert isinstance(result, dict)
    cmd = cast(list[str], captured["cmd"])
    assert "-np" in cmd
    assert "4" in cmd[cmd.index("-np") + 1 :][:1]


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

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
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

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
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


def test_run_local_no_args_prefers_local_bin_over_sif(monkeypatch, tmp_path):
    """run_local() should prefer local ./bin/palace over local SIF."""
    output_dir = tmp_path / "sim"
    postpro_dir = output_dir / "output" / "palace"
    output_dir.mkdir(parents=True)
    postpro_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text("{}")
    (output_dir / "palace.msh").write_text("mesh")

    local_sif = tmp_path / "Palace.sif"
    local_sif.write_text("fake")

    local_bin_dir = tmp_path / "bin"
    local_bin_dir.mkdir()
    local_palace = local_bin_dir / "palace"
    local_palace.write_text("#!/bin/sh\nexit 0\n")
    local_palace.chmod(0o755)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PALACE_SIF", raising=False)
    monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    sim = DrivenSim()
    sim.set_output_dir(output_dir)
    result = sim.run_local(num_processes=1, verbose=False)

    assert isinstance(result, dict)
    cmd = cast(list[str], captured["cmd"])
    assert isinstance(cmd, list)
    assert Path(cmd[0]) == local_palace.resolve()
    assert Path(cmd[0]).is_absolute()
    assert captured["cwd"] == output_dir


def test_run_local_no_args_uses_local_sif_when_no_executable(monkeypatch, tmp_path):
    """run_local() should use Apptainer if only a local SIF is available."""
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

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    sim = DrivenSim()
    sim.set_output_dir(output_dir)
    result = sim.run_local(num_processes=1, verbose=False, use_apptainer=True)

    assert isinstance(result, dict)
    cmd = cast(list[str], captured["cmd"])
    assert isinstance(cmd, list)
    assert cmd[0] == "apptainer"
    assert cmd[1] == "run"
    assert Path(cmd[2]) == local_sif.resolve()
    assert captured["cwd"] == output_dir


def test_run_local_uses_bundled_resolver_before_path_fallback(monkeypatch, tmp_path):
    """No explicit executable should still discover the installed runtime."""
    output_dir = tmp_path / "sim"
    sim = BoundaryModeSim()
    _setup_sim(sim, output_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PALACE_SIF", raising=False)
    monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)
    bundled = tmp_path / "bundled-palace"
    bundled.write_text("#!/bin/sh\nexit 0\n")
    bundled.chmod(0o755)
    monkeypatch.setattr("gsim.palace.runtime.resolve_palace_binary", lambda: bundled)
    monkeypatch.setattr("gsim.palace.runtime.resolve_palace_library_dir", lambda: None)
    captured = {}

    def run(cmd, **kwargs):
        captured.update(cmd=cmd, cwd=kwargs["cwd"])
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("subprocess.run", run)
    result = sim.run_local(num_processes=1, verbose=False)
    assert isinstance(result, dict)
    assert Path(captured["cmd"][0]) == bundled
    assert captured["cwd"] == output_dir
