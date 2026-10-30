"""MagnetostaticSim.run_local keeps Palace's log for the convergence check.

A fake Palace binary replays the not-converged ``rings`` run: it prints that
run's log and writes its tables, and exits 0, as Palace does.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsim.palace import MagnetostaticSim

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the fake Palace is a POSIX script"
)

NOT_CONVERGED = (
    Path(__file__).parent / "fixtures" / "magnetostatic" / "rings-not-converged"
)


@pytest.fixture
def fake_palace(tmp_path) -> Path:
    """A Palace stand-in that prints the run's log and writes its tables."""
    script = tmp_path / "palace"
    script.write_text(
        f"#!{sys.executable}\n"
        "import shutil, sys\n"
        "from pathlib import Path\n"
        f"source = Path({str(NOT_CONVERGED)!r})\n"
        "out = Path('output') / 'palace'\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "for name in ('terminal-M.csv', 'terminal-I.csv', 'domain-E.csv'):\n"
        "    shutil.copy(source / name, out / name)\n"
        "sys.stdout.write((source / 'palace.log').read_text(encoding='utf-8'))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def sim(tmp_path) -> MagnetostaticSim:
    """A magnetostatic sim with a written config and mesh, and two sources."""
    output_dir = tmp_path / "sim"
    output_dir.mkdir()
    (output_dir / "config.json").write_text("{}")
    (output_dir / "palace.msh").write_text("mesh")
    sim = MagnetostaticSim()
    sim._last_mesh_result = SimpleNamespace(
        mesh_stats={"elements": 1_000, "tetrahedra": 1_000},
        groups={},
        model_name="palace",
    )
    sim.set_output_dir(output_dir)
    sim.add_current_source("inner", plus_port="P1", minus_port="P2")
    sim.add_current_source("outer", plus_port="Q1", minus_port="Q2")
    return sim


@pytest.mark.parametrize("verbose", [False, True])
def test_the_log_is_kept_with_the_results(sim, fake_palace, verbose) -> None:
    results = sim.run_local(
        palace_executable=fake_palace, num_processes=1, verbose=verbose
    )
    log = sim._postpro_dir() / "palace.log"
    assert results.files["palace.log"] == log
    assert "PCG solver did NOT converge" in log.read_text(encoding="utf-8")


def test_a_not_converged_run_is_refused_on_load(sim, fake_palace) -> None:
    results = sim.run_local(
        palace_executable=fake_palace, num_processes=1, verbose=False
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        sim.load_inductance(results)


def test_without_the_check_the_matrix_is_labelled_by_source(sim, fake_palace) -> None:
    results = sim.run_local(
        palace_executable=fake_palace, num_processes=1, verbose=False
    )
    inductance = sim.load_inductance(results, check_convergence=False)
    assert inductance.sources == ("inner", "outer")
    assert inductance.mutual_inductance("inner", "outer") < 0  # the wrong sign


def test_a_log_kept_elsewhere_leaves_convergence_unchecked(
    sim, fake_palace, tmp_path
) -> None:
    elsewhere = tmp_path / "logs" / "x.log"
    results = sim.run_local(
        palace_executable=fake_palace,
        num_processes=1,
        verbose=False,
        log_path=elsewhere,
    )
    assert "PCG solver did NOT converge" in elsewhere.read_text(encoding="utf-8")
    assert "palace.log" not in results.files
    with pytest.warns(UserWarning, match="convergence not checked") as record:
        sim.load_inductance(results)
    assert [w.filename for w in record] == [__file__]  # points at this call


def test_the_check_flag_is_passed_through(sim, monkeypatch) -> None:
    calls = []

    def fake_load(results, **kwargs):
        calls.append((results, kwargs))

    monkeypatch.setattr("gsim.palace.magnetostatic.load_inductance", fake_load)
    sim.load_inductance("dir", check_convergence=False)
    sim.load_inductance("dir")
    assert calls == [
        ("dir", {"source_names": ["inner", "outer"], "check_convergence": False}),
        ("dir", {"source_names": ["inner", "outer"], "check_convergence": True}),
    ]
