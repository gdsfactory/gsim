"""Tests for local Palace execution wiring."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

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


def _write_mode_table(output_dir) -> None:
    """What a boundary-mode Palace leaves behind: one mode table."""
    postpro_dir = output_dir / "output" / "palace"
    postpro_dir.mkdir(parents=True, exist_ok=True)
    (postpro_dir / "mode-kn.csv").write_text(
        "m, Re{kn} (1/m), Im{kn} (1/m), Re{n_eff}, Im{n_eff}\n"
        "1, 4.2e7, -1.0e2, 2.1, -1.0e-5\n"
    )


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
        _write_mode_table(output_dir)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("subprocess.run", _fake_run)

    sim = BoundaryModeSim()
    sim._last_mesh_result = _mesh_result(50_000)
    _setup_sim(sim, output_dir)

    result = sim.run_local(verbose=False)

    # A boundary-mode run hands back its own mode tables.
    assert result.modes[1]["n_eff"].real == pytest.approx(2.1)
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
        _write_mode_table(output_dir)
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
        _write_mode_table(output_dir)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("subprocess.run", run)
    result = sim.run_local(num_processes=1, verbose=False)
    # A boundary-mode run hands back its own mode tables.
    assert result.modes[1]["n_eff"].real == pytest.approx(2.1)
    assert Path(captured["cmd"][0]) == bundled
    assert captured["cwd"] == output_dir


class TestAbortedBinaryIsReported:
    """Every local Palace run diagnoses a binary that died on a signal.

    A broken Palace runtime — typically a bundled MPI that cannot start —
    kills the binary before the solver writes anything, and the raw
    ``CalledProcessError`` that surfaces carries an exit status and
    nothing a user can act on. Every simulation turns that into a report
    naming the binary that ran and saying whether the runtime or the
    model is at fault, so a caller catches one exception type whichever
    solve it asked for.
    """

    @pytest.fixture
    def aborting_palace(self, monkeypatch, tmp_path):
        """A Palace binary that dies, having written *tables* mode tables."""
        _setup_local_palace(monkeypatch, tmp_path)
        monkeypatch.delenv("PALACE_SIF", raising=False)
        monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)
        script: dict[str, object] = {
            "tables": 0,
            "returncode": 134,
            "stderr": "",
            "output_dir": tmp_path / "sim",
        }

        def _fake_run(_cmd, **_kwargs):
            if script["tables"]:
                _write_mode_table(script["output_dir"])
            raise subprocess.CalledProcessError(
                cast(int, script["returncode"]),
                ["/opt/somewhere/palace", "-np", "1", "config.json"],
                output="",
                stderr=cast(str, script["stderr"]),
            )

        monkeypatch.setattr("subprocess.run", _fake_run)
        return script

    @staticmethod
    def _boundary_mode_run(tmp_path, **kwargs):
        sim = BoundaryModeSim()
        sim._last_mesh_result = _mesh_result(50_000)
        _setup_sim(sim, tmp_path / "sim")
        sim.set_boundary_mode(freq=10e9, num_modes=4)
        return sim.run_local(verbose=False, **kwargs)

    @staticmethod
    def _driven_run(tmp_path, **kwargs):
        sim = DrivenSim()
        sim._last_mesh_result = _mesh_result(50_000)
        _setup_sim(sim, tmp_path / "sim")
        return sim.run_local(verbose=False, **kwargs)

    @pytest.mark.usefixtures("aborting_palace")
    def test_the_report_names_the_binary_and_blames_the_runtime(self, tmp_path):
        with pytest.raises(RuntimeError) as excinfo:
            self._boundary_mode_run(tmp_path)
        message = str(excinfo.value)
        # Resolved by the run itself, and named by the command that died.
        assert str(tmp_path / "bin" / "palace") in message
        assert "exit status 134" in message
        assert "SIGABRT" in message
        assert "any solver output" in message
        assert "runtime" in message
        assert "PALACE_BIN" in message

    @pytest.mark.usefixtures("aborting_palace")
    def test_a_driven_solve_is_diagnosed_the_same_way(self, tmp_path):
        """Not only the boundary-mode path: the report is on the mixin."""
        with pytest.raises(RuntimeError) as excinfo:
            self._driven_run(tmp_path)
        message = str(excinfo.value)
        assert "SIGABRT" in message
        assert "driven" in message
        assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)

    @pytest.mark.usefixtures("aborting_palace")
    def test_a_boundary_mode_solve_is_named_by_its_frequency(self, tmp_path):
        with pytest.raises(RuntimeError, match="at f = 1e\\+10 Hz"):
            self._boundary_mode_run(tmp_path)

    @pytest.mark.usefixtures("aborting_palace")
    def test_the_caller_names_the_binary_when_it_asked_for_one(self, tmp_path):
        elsewhere = tmp_path / "elsewhere" / "palace"
        elsewhere.parent.mkdir()
        elsewhere.write_text("#!/bin/sh\nexit 0\n")
        elsewhere.chmod(0o755)

        with pytest.raises(RuntimeError, match=str(elsewhere)):
            self._boundary_mode_run(tmp_path, palace_executable=str(elsewhere))

    @pytest.mark.usefixtures("aborting_palace")
    def test_a_caller_supplied_remedy_closes_the_report(self, tmp_path):
        """The one Route-shaped sentence is handed down, not written here."""
        with pytest.raises(RuntimeError, match="route='femwell'"):
            self._boundary_mode_run(
                tmp_path, remedy="re-solve on the default route with route='femwell'"
            )

    @pytest.mark.usefixtures("aborting_palace")
    def test_without_a_remedy_only_palace_bin_is_offered(self, tmp_path):
        with pytest.raises(RuntimeError) as excinfo:
            self._driven_run(tmp_path)
        assert str(excinfo.value).endswith("runtime works here.")

    def test_a_plain_exit_is_not_blamed_on_the_runtime(self, tmp_path, aborting_palace):
        """Exit 1 is Palace refusing the run itself; its stderr says why."""
        aborting_palace["returncode"] = 1
        aborting_palace["stderr"] = "Invalid configuration\n"

        with pytest.raises(RuntimeError) as excinfo:
            self._boundary_mode_run(tmp_path)
        message = str(excinfo.value)
        assert "exit status 1." in message
        assert "runtime" not in message.split("Point PALACE_BIN", maxsplit=1)[0]
        assert "Invalid configuration" in message

    @pytest.mark.usefixtures("aborting_palace")
    def test_the_raw_error_is_chained_not_lost(self, tmp_path):
        with pytest.raises(RuntimeError) as excinfo:
            self._boundary_mode_run(tmp_path)
        assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)
        assert excinfo.value.__cause__.returncode == 134

    def test_a_segfault_is_named_as_one(self, tmp_path, aborting_palace):
        aborting_palace["returncode"] = 139

        with pytest.raises(RuntimeError, match="SIGSEGV"):
            self._boundary_mode_run(tmp_path)

    def test_the_last_worded_stderr_line_is_quoted(self, tmp_path, aborting_palace):
        """MPI ends its error blocks with a dashed rule; quote past it."""
        aborting_palace["stderr"] = (
            "noise\nopal_shmem_base_select failed\n" + "-" * 40 + "\n"
        )

        with pytest.raises(RuntimeError, match="opal_shmem_base_select failed"):
            self._boundary_mode_run(tmp_path)

    def test_a_streaming_run_is_diagnosed_too(self, monkeypatch, tmp_path):
        """`verbose=True` is the default, and it streams rather than checks."""
        _setup_local_palace(monkeypatch, tmp_path)
        monkeypatch.delenv("PALACE_SIF", raising=False)
        monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)

        class _DeadProcess:
            stdout = iter(["opal_shmem_base_select failed\n"])

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def wait(self):
                return -6

        monkeypatch.setattr("subprocess.Popen", lambda *_a, **_kw: _DeadProcess())

        sim = DrivenSim()
        sim._last_mesh_result = _mesh_result(50_000)
        _setup_sim(sim, tmp_path / "sim")

        with pytest.raises(RuntimeError) as excinfo:
            sim.run_local(verbose=True)
        message = str(excinfo.value)
        assert "SIGABRT" in message
        assert "any solver output" in message
        assert "opal_shmem_base_select failed" in message

    def test_partial_output_is_not_blamed_on_the_runtime(
        self, tmp_path, aborting_palace
    ):
        """Output left behind means the solver ran; the runtime did start."""
        aborting_palace["tables"] = 1

        with pytest.raises(RuntimeError) as excinfo:
            self._driven_run(tmp_path)
        message = str(excinfo.value)
        assert "any solver output" not in message
        assert "partial solver output" in message
        assert "exit status 134" in message
        assert str(tmp_path) in message
