"""Tests for reading and checking the inductance matrix of a Palace run.

The fixtures in ``fixtures/magnetostatic/rings`` are the outputs of Palace's
``examples/rings`` magnetostatic example in awslabs/palace (Apache-2.0), run
locally with Palace v0.18.1; see the README there.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gsim.palace import InductanceMatrix, load_inductance
from gsim.palace.results import load_text_results

MAGNETOSTATIC = Path(__file__).parent / "fixtures" / "magnetostatic"
RINGS = MAGNETOSTATIC / "rings"
NOT_CONVERGED = MAGNETOSTATIC / "rings-not-converged"

PICO = 1e-12


def test_reads_the_rings_inductances() -> None:
    rings = load_inductance(RINGS)
    assert rings.sources == ("S1", "S2")
    assert rings.self_inductance("S1") == pytest.approx(42.739 * PICO, rel=1e-4)
    assert rings.self_inductance("S2") == pytest.approx(714.103 * PICO, rel=1e-4)
    assert rings.mutual_inductance("S1", "S2") == pytest.approx(1.960 * PICO, rel=1e-4)
    assert rings.coupling("S1", "S2") == pytest.approx(0.01122, rel=1e-3)
    assert isinstance(rings, InductanceMatrix)


def test_the_rings_run_has_no_problems() -> None:
    assert load_inductance(RINGS).problems() == []


def _from_matrix(matrix: np.ndarray) -> InductanceMatrix:
    """Every table Palace writes, consistent with an inductance matrix."""
    current = np.full(len(matrix), 0.05152105132651)
    return InductanceMatrix(
        sources=tuple(f"S{i}" for i in range(1, len(matrix) + 1)),
        matrix=matrix,
        inverse=np.linalg.inv(matrix),
        excitation_current=current,
        stored_energy=0.5 * np.diag(matrix) * current**2,
    )


def _rings_with(**entries: float) -> InductanceMatrix:
    """The rings matrix with some entries replaced, e.g. ``m01=...``."""
    matrix = load_inductance(RINGS).matrix.copy()
    for key, value in entries.items():
        matrix[int(key[1]), int(key[2])] = value
    return _from_matrix(matrix)


def _all_coupled(k: float) -> InductanceMatrix:
    """Three sources, every pair coupled by ``k``."""
    diagonal = np.array([42.739, 714.103, 100.0]) * PICO
    coupling = np.full((3, 3), k) + (1 - k) * np.eye(3)
    root = np.sqrt(diagonal)
    return _from_matrix(coupling * np.outer(root, root))


def _rings_scaled(field: str, factor: float | np.ndarray) -> InductanceMatrix:
    """The rings run with one of the tables Palace wrote scaled."""
    rings = load_inductance(RINGS)
    values = getattr(rings, field)
    assert values is not None
    return replace(rings, **{field: values * factor})


CHECKS = {
    "not finite": "not finite",
    "self inductance": "no positive self inductance",
    "reciprocity": "not reciprocal",
    "energy sign": "magnetic energy is negative",
    "coupling": "coupling coefficient above 1",
    "inverse": "does not invert",
    "stored energy": "stored energy does not match",
    "zero current": "zero excitation current",
}


@pytest.mark.parametrize(
    ("broken", "check"),
    [
        (lambda: _rings_with(m01=1.5 * 1.960237936695e-12), "reciprocity"),
        (lambda: _all_coupled(-0.6), "energy sign"),
        (
            lambda: _rings_with(
                m01=1.2 * np.sqrt(42.739 * 714.103) * PICO,
                m10=1.2 * np.sqrt(42.739 * 714.103) * PICO,
            ),
            "coupling",
        ),
        (lambda: _rings_scaled("inverse", 1.01), "inverse"),
        (lambda: _rings_scaled("stored_energy", np.array([1.0, 1.2])), "stored energy"),
        (lambda: _rings_with(m01=np.nan), "not finite"),
        (lambda: _rings_with(m11=0.0), "self inductance"),
        (
            lambda: _rings_scaled("excitation_current", np.array([1.0, 0.0])),
            "zero current",
        ),
    ],
    ids=[
        "asymmetric",
        "non-psd",
        "k-above-1",
        "bad-inverse",
        "bad-energy",
        "nan",
        "no-self-inductance",
        "zero-current",
    ],
)
def test_each_broken_matrix_trips_exactly_its_check(broken, check) -> None:
    problems = broken().problems()
    tripped = [
        name for name, text in CHECKS.items() if any(text in p for p in problems)
    ]
    assert tripped == [check], problems
    assert len(problems) == 1, problems


def _copy(tmp_path: Path, *names: str) -> Path:
    """A run directory holding the rings log and the named rings tables.

    All the tables by default.
    """
    run = tmp_path / "run"
    run.mkdir()
    shutil.copy(RINGS / "palace.log", run / "palace.log")
    for path in RINGS.glob("*.csv"):
        if not names or path.name in names:
            shutil.copy(path, run / path.name)
    return run


def test_reads_the_excitation_currents_and_stored_energies() -> None:
    rings = load_inductance(RINGS)
    assert rings.excitation_current == pytest.approx([0.05152105132651] * 2)
    assert rings.stored_energy == pytest.approx(
        [5.672327652641e-14, 9.477640572111e-13]
    )
    assert rings.inverse is not None
    assert rings.inverse[0, 1] == pytest.approx(-6.423634223590e07)


def test_directory_dict_and_local_results_load_the_same(tmp_path) -> None:
    """Cloud ``run()`` gives a files dict, ``run_local()`` a ``PalaceTextResults``."""
    run = _copy(tmp_path)
    loads = [
        load_inductance(run),
        load_inductance({path.name: path for path in run.iterdir()}),
        load_inductance(load_text_results(run)),
    ]
    first = loads[0]
    for other in loads[1:]:
        assert other.sources == first.sources == ("S1", "S2")
        for field in ("matrix", "inverse", "excitation_current", "stored_energy"):
            assert np.array_equal(getattr(other, field), getattr(first, field))


def test_finds_the_files_in_the_palace_output_subdirectory(tmp_path) -> None:
    shutil.copytree(RINGS, tmp_path / "sim" / "output" / "palace")
    assert load_inductance(tmp_path / "sim").matrix.shape == (2, 2)


def test_source_names_label_the_matrix() -> None:
    rings = load_inductance(RINGS, source_names=["inner", "outer"])
    assert rings.sources == ("inner", "outer")
    assert rings.self_inductance("outer") == pytest.approx(714.103 * PICO, rel=1e-4)
    with pytest.raises(KeyError, match="S1"):
        rings.self_inductance("S1")


def test_the_optional_files_can_be_missing(tmp_path) -> None:
    rings = load_inductance(_copy(tmp_path, "terminal-M.csv"))
    assert rings.inverse is None
    assert rings.excitation_current is None
    assert rings.stored_energy is None
    assert rings.problems() == []


def test_a_missing_matrix_names_the_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=r"terminal-M\.csv"):
        load_inductance(_copy(tmp_path, "terminal-I.csv"))


def test_an_energy_table_without_the_magnetic_column_is_ignored(tmp_path) -> None:
    run = _copy(tmp_path)
    energy = run / "domain-E.csv"
    energy.write_text(
        energy.read_text(encoding="utf-8").replace("E_mag (J)", "E_total (J)"),
        encoding="utf-8",
    )
    assert load_inductance(run).stored_energy is None


def test_a_matrix_that_is_not_square_is_rejected(tmp_path) -> None:
    run = _copy(tmp_path)
    (run / "terminal-M.csv").write_text(
        "i, M[i][1] (H), M[i][2] (H), M[i][3] (H)\n"
        "1, +1.0e-12, +1.0e-13, +1.0e-13\n"
        "2, +1.0e-13, +2.0e-12, +1.0e-13\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not square"):
        load_inductance(run)


def test_a_table_with_the_wrong_number_of_rows_is_rejected(tmp_path) -> None:
    run = _copy(tmp_path)
    current = run / "terminal-I.csv"
    current.write_text(
        "".join(current.read_text(encoding="utf-8").splitlines(keepends=True)[:2]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"terminal-I\.csv"):
        load_inductance(run)


def test_an_inverse_of_the_wrong_shape_is_rejected(tmp_path) -> None:
    run = _copy(tmp_path)
    (run / "terminal-Minv.csv").write_text(
        "i, Minv[i][1] (1/H)\n1, +1.0e10\n2, +1.0e9\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"terminal-Minv\.csv"):
        load_inductance(run)


@pytest.mark.parametrize(
    ("names", "match"),
    [(["only-one"], "2 sources"), (["same", "same"], "different")],
)
def test_bad_source_names_are_rejected(names, match) -> None:
    with pytest.raises(ValueError, match=match):
        load_inductance(RINGS, source_names=names)


def test_the_frames_are_labelled_by_source() -> None:
    rings = load_inductance(RINGS, source_names=["inner", "outer"])
    inductance = rings.inductance_frame()
    assert list(inductance.index) == list(inductance.columns) == ["inner", "outer"]
    assert inductance.loc["inner", "outer"] == pytest.approx(1.960237936695e-12)
    coupling = rings.coupling_frame()
    assert list(coupling.index) == list(coupling.columns) == ["inner", "outer"]
    assert coupling.loc["inner", "inner"] == 1.0
    assert coupling.loc["outer", "inner"] == pytest.approx(0.01122, rel=1e-3)


def test_pairs_lists_self_rows_first_then_each_pair() -> None:
    pairs = _all_coupled(0.25).pairs()
    assert list(pairs.columns) == ["first", "second", "inductance_H", "coupling"]
    assert list(zip(pairs["first"], pairs["second"], strict=True)) == [
        ("S1", "S1"),
        ("S2", "S2"),
        ("S3", "S3"),
        ("S1", "S2"),
        ("S1", "S3"),
        ("S2", "S3"),
    ]
    assert list(pairs["coupling"]) == pytest.approx([1, 1, 1, 0.25, 0.25, 0.25])
    assert pairs["inductance_H"][1] == pytest.approx(714.103 * PICO)
    assert pairs["inductance_H"][5] == pytest.approx(
        0.25 * np.sqrt(714.103 * 100.0) * PICO
    )


def test_a_mutual_inductance_needs_two_different_sources() -> None:
    with pytest.raises(ValueError, match="self_inductance"):
        load_inductance(RINGS).mutual_inductance("S1", "S1")


def test_a_mutual_keeps_its_sign() -> None:
    reversed_outer = _rings_with(m01=-1.960237936695e-12, m10=-1.960237936695e-12)
    assert reversed_outer.mutual_inductance("S2", "S1") == pytest.approx(
        -1.96 * PICO, rel=1e-3
    )
    assert reversed_outer.coupling("S1", "S2") == pytest.approx(-0.01122, rel=1e-3)
    assert reversed_outer.problems() == []


def test_coupling_is_nan_without_a_positive_self_inductance() -> None:
    assert np.isnan(_rings_with(m11=0.0).coupling("S1", "S2"))


def test_a_weak_mutual_is_judged_on_its_own_scale() -> None:
    """20 % asymmetry on a 2 pH mutual is 0.05 % of the 714 pH self inductance."""
    lopsided = _rings_with(m01=1.2 * 1.960237936695e-12)
    assert any("not reciprocal" in text for text in lopsided.problems())


def test_a_run_that_did_not_converge_is_refused() -> None:
    """Its matrix looks physical; only the log shows the solve failed."""
    unchecked = load_inductance(NOT_CONVERGED, check_convergence=False)
    assert unchecked.problems() == []
    with pytest.raises(RuntimeError, match="did not converge") as error:
        load_inductance(NOT_CONVERGED, source_names=["inner", "outer"])
    message = str(error.value)
    assert "'inner'" in message
    assert "'outer'" in message
    assert "PCG solver did NOT converge in 10 iterations" in message
    assert "max_iterations" in message
    assert "check_convergence=False" in message


def test_a_converged_run_loads_silently(tmp_path) -> None:
    run = _copy(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_inductance(RINGS)
        load_inductance({path.name: path for path in run.iterdir()})
        load_inductance(load_text_results(run))


def test_a_run_without_its_log_loads_with_a_warning(tmp_path) -> None:
    run = _copy(tmp_path)
    (run / "palace.log").unlink()
    with pytest.warns(UserWarning, match="convergence not checked") as record:
        rings = load_inductance(run)
    assert rings.matrix.shape == (2, 2)
    assert [w.filename for w in record] == [__file__]  # points at this call


def test_a_log_without_solver_messages_loads_with_a_warning(tmp_path) -> None:
    run = _copy(tmp_path)
    (run / "palace.log").write_text("Git changeset ID: v0.18.1\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="convergence not confirmed") as record:
        load_inductance(run)
    assert [w.filename for w in record] == [__file__]


def test_the_convergence_check_can_be_skipped(tmp_path) -> None:
    run = _copy(tmp_path)
    (run / "palace.log").unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_inductance(run, check_convergence=False)
        load_inductance(NOT_CONVERGED, check_convergence=False)


def test_the_cloud_build_failure_line_names_its_source(tmp_path) -> None:
    run = _copy(tmp_path)
    (run / "palace.log").write_text(
        "PCG solver converged in 49 iterations\n"
        "Current source solve did not converge for port 2!\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=r"source 'outer' \(index 2\)"):
        load_inductance(run, source_names=["inner", "outer"])
