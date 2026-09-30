"""A boundary-mode simulation owns its run.

Where Palace's tables land, that the previous run's are gone before the
next one, that what is read back is this run's — and what a run that
ends badly means: Palace 0.17 crashes on shutdown *after* answering, so
a complete table left behind is the answer, while a binary that died
before writing anything is a broken runtime and is reported as one. None
of it is something a caller should have to spell.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

import meshio
import numpy as np
import pytest

from gsim.palace import BoundaryModeSim
from gsim.palace.base import PalaceSimMixin
from gsim.palace.boundarymode import RUN_SUBDIR

MODE_TABLE = (
    "m, Re{kn} (1/m), Im{kn} (1/m), Re{n_eff}, Im{n_eff}\n"
    "1, 4.2e7, -1.0e2, 2.1, -1.0e-5\n"
)


def mode_table_text(n_modes: int) -> str:
    """A ``mode-kn.csv`` carrying *n_modes* Modes."""
    header = "m, Re{kn} (1/m), Im{kn} (1/m), Re{n_eff}, Im{n_eff}\n"
    rows = "".join(
        f"{m}, {4.2e7 + m:.6e}, -1.0e2, {2.0 + 0.1 * m:.6e}, -1.0e-5\n"
        for m in range(1, n_modes + 1)
    )
    return header + rows


def sim_at(tmp_path) -> BoundaryModeSim:
    sim = BoundaryModeSim()
    sim.set_output_dir(tmp_path)
    return sim


def write_table(directory, name="mode-kn.csv", text=MODE_TABLE) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text)


class TestMeshExtent:
    """The rectangle the mesh covers, read off the mesh itself."""

    @staticmethod
    def _meshed(tmp_path, points) -> BoundaryModeSim:
        """A simulation standing on a hand-written two-triangle mesh."""
        path = tmp_path / "palace.msh"
        meshio.write(
            path,
            meshio.Mesh(
                np.asarray(points, dtype=float),
                [("triangle", np.array([[0, 1, 2], [1, 2, 3]]))],
            ),
            file_format="gmsh",
        )
        sim = sim_at(tmp_path)
        sim._last_mesh_result = SimpleNamespace(mesh_path=path)
        return sim

    def test_it_is_the_bounding_box_of_the_mesh_nodes(self, tmp_path):
        sim = self._meshed(
            tmp_path,
            [(-2.5, -1.0, 0.0), (3.5, -1.0, 0.0), (-2.5, 4.0, 0.0), (3.5, 4.0, 0.0)],
        )

        assert sim.mesh_extent == ((-2.5, 3.5), (-1.0, 4.0))

    def test_without_a_mesh_it_is_an_error(self):
        with pytest.raises(ValueError, match="Call mesh"):
            _ = BoundaryModeSim().mesh_extent


class TestRunDirectory:
    def test_it_sits_under_the_output_directory(self, tmp_path):
        assert sim_at(tmp_path).run_dir == tmp_path / RUN_SUBDIR

    def test_without_an_output_directory_it_is_an_error(self):
        with pytest.raises(ValueError, match="set_output_dir"):
            _ = BoundaryModeSim().run_dir

    def test_no_run_leaves_no_files_and_no_results(self, tmp_path):
        sim = sim_at(tmp_path)
        assert sim.last_run_files == {}
        assert sim.read_results() is None

    def test_this_runs_tables_are_read_back(self, tmp_path):
        sim = sim_at(tmp_path)
        write_table(sim.run_dir)

        assert set(sim.last_run_files) == {"mode-kn.csv"}
        results = sim.read_results()
        assert results is not None
        assert results.modes[1]["n_eff"].real == pytest.approx(2.1)

    def test_a_table_left_beside_the_run_is_not_this_runs(self, tmp_path):
        """Only the run directory is read, so nothing else can shadow it."""
        sim = sim_at(tmp_path)
        write_table(tmp_path)

        assert sim.last_run_files == {}
        assert sim.read_results() is None


class TestRunningLocally:
    @pytest.fixture
    def scripted_palace(self, monkeypatch):
        """Stand the mixin's run in for a Palace that writes what it is told."""
        script = {"writes": True}

        def fake_run_local(self, **_kwargs):
            if script["writes"]:
                write_table(self.run_dir)
            return {}

        monkeypatch.setattr(PalaceSimMixin, "run_local", fake_run_local)
        return script

    @pytest.mark.usefixtures("scripted_palace")
    def test_the_previous_runs_tables_are_cleared_first(self, tmp_path):
        sim = sim_at(tmp_path)
        write_table(sim.run_dir, name="mode-Z.csv", text="m, Z_PV[1] (Ohm)\n1, 50\n")

        results = sim.run_local(palace_executable="palace", verbose=False)

        assert set(sim.last_run_files) == {"mode-kn.csv"}
        assert results.characteristic_impedance(index=1, mode=1) is None
        assert results.modes[1]["n_eff"].real == pytest.approx(2.1)

    def test_a_run_that_leaves_nothing_is_an_error(self, tmp_path, scripted_palace):
        scripted_palace["writes"] = False
        with pytest.raises(RuntimeError, match="no text results"):
            sim_at(tmp_path).run_local(palace_executable="palace", verbose=False)

    @pytest.mark.usefixtures("scripted_palace")
    def test_the_returned_results_are_what_read_results_reads(self, tmp_path):
        sim = sim_at(tmp_path)
        results = sim.run_local(palace_executable="palace", verbose=False)
        again = sim.read_results()
        assert again is not None
        assert again.modes == results.modes


class TestCrashedRunSalvage:
    """Palace 0.17 can corrupt its heap on shutdown, after answering.

    A run that exits abnormally with its complete mode table on disk is
    an answer, not a failure; a truncated or absent table stays one.
    """

    @pytest.fixture
    def crashing_palace(self, monkeypatch):
        """A Palace that writes *modes* Modes and then dies."""
        script: dict[str, Any] = {
            "modes": 0,
            "raises": RuntimeError("free(): corrupted chunks"),
        }

        def fake_run_local(self, **_kwargs):
            if script["modes"]:
                write_table(self.run_dir, text=mode_table_text(script["modes"]))
            raise script["raises"]

        monkeypatch.setattr(PalaceSimMixin, "run_local", fake_run_local)
        return script

    @staticmethod
    def _asking_for(tmp_path, num_modes: int) -> BoundaryModeSim:
        sim = sim_at(tmp_path)
        sim.set_boundary_mode(freq=10e9, num_modes=num_modes)
        return sim

    def test_a_complete_table_is_used_and_the_crash_reported(
        self, tmp_path, crashing_palace
    ):
        crashing_palace["modes"] = 4

        with pytest.warns(UserWarning, match="exited abnormally"):
            results = self._asking_for(tmp_path, 4).run_local(verbose=False)

        assert len(results.modes) == 4
        assert results.modes[1]["n_eff"].real == pytest.approx(2.1)

    def test_the_warning_names_the_frequency_and_the_palace_bug(
        self, tmp_path, crashing_palace
    ):
        crashing_palace["modes"] = 1

        with pytest.warns(UserWarning, match="f = 1e\\+10 Hz"):
            self._asking_for(tmp_path, 1).run_local(verbose=False)

    def test_a_truncated_table_is_not_an_answer(self, tmp_path, crashing_palace):
        crashing_palace["modes"] = 2

        with pytest.raises(RuntimeError, match="corrupted chunks"):
            self._asking_for(tmp_path, 4).run_local(verbose=False)

    @pytest.mark.usefixtures("crashing_palace")
    def test_no_output_at_all_is_not_an_answer(self, tmp_path):
        with pytest.raises(RuntimeError, match="corrupted chunks"):
            self._asking_for(tmp_path, 4).run_local(verbose=False)

    @pytest.mark.usefixtures("crashing_palace")
    def test_a_previous_runs_table_is_not_salvaged_as_this_ones(self, tmp_path):
        """The run directory is cleared before running, so a stale table is gone."""
        sim = self._asking_for(tmp_path, 4)
        write_table(sim.run_dir, text=mode_table_text(4))

        with pytest.raises(RuntimeError, match="corrupted chunks"):
            sim.run_local(verbose=False)

    def test_salvage_can_be_turned_off(self, tmp_path, crashing_palace):
        """A caller who wants the raise back keeps it reachable."""
        crashing_palace["modes"] = 4

        with pytest.raises(RuntimeError, match="corrupted chunks"):
            self._asking_for(tmp_path, 4).run_local(verbose=False, salvage=False)


class TestAnAbortedBinaryStillSalvages:
    """An abort is asked for its answer before it is diagnosed.

    Where the diagnosis itself is written — and that a Palace binary
    that died on a signal is reported as a runtime failure rather than
    as a raw exit status — is
    :mod:`tests.palace.test_run_local`; what is here is that the
    salvage runs first, on a solve that answered before it died.
    """

    @pytest.fixture
    def aborting_palace(self, monkeypatch):
        """A Palace binary that dies, having written *modes* Modes."""
        script: dict[str, Any] = {"modes": 0}

        def fake_run_local(self, **_kwargs):
            if script["modes"]:
                write_table(self.run_dir, text=mode_table_text(script["modes"]))
            raise subprocess.CalledProcessError(
                134,
                ["/opt/somewhere/palace", "-np", "1", "config.json"],
                output="",
                stderr="",
            )

        monkeypatch.setattr(PalaceSimMixin, "run_local", fake_run_local)
        return script

    def test_a_complete_table_left_by_an_abort_is_still_salvaged(
        self, tmp_path, aborting_palace
    ):
        """The salvage runs before the report: an answer beats a diagnosis."""
        aborting_palace["modes"] = 4
        sim = sim_at(tmp_path)
        sim.set_boundary_mode(freq=10e9, num_modes=4)

        with pytest.warns(UserWarning, match="exited abnormally"):
            results = sim.run_local(verbose=False)

        assert len(results.modes) == 4
