"""Tests for reading the per-pass output of Palace's adaptive mesh refinement.

With ``SaveAdaptIterations``, Palace keeps the postprocessing output of pass X
in an ``iterationX`` subdirectory and writes the last pass at the top level.
The files below are verbatim from Palace's own regression reference
``test/data/regression/ref/transmon/transmon_amr`` (awslabs/palace), an
eigenmode run with two refinement passes after the initial solve.
"""

from __future__ import annotations

import math

import pytest

from gsim.palace import load_refinement_history
from gsim.palace.results import refinement_convergence

ERROR_HEADER = (
    "                       Norm,                    Minimum,"
    "                    Maximum,                       Mean\n"
)
EIG_HEADER = (
    "        m,                Re{f} (GHz),                Im{f} (GHz),"
    "                          Q,              Error (Bkwd.),"
    "               Error (Abs.)\n"
)

# Pass directory -> (error-indicators.csv row, eig.csv first row)
TRANSMON_AMR = {
    "iteration1": (
        "        +3.455421238025e-01,        +4.882044825618e-08,"
        "        +4.191766435607e-02,        +5.090410784829e-04\n",
        " 1.00e+00,        +4.291377978227e+00,        +1.199243788462e-04,"
        "        +1.789201670749e+04,        +1.498599904688e-14,"
        "        +4.961598405864e-09\n",
    ),
    "iteration2": (
        "        +2.761992653477e-01,        +4.876511804807e-08,"
        "        +1.997973183467e-02,        +5.039794334944e-04\n",
        " 1.00e+00,        +4.352933425564e+00,        +1.222836763303e-04,"
        "        +1.779850572828e+04,        +1.125808861424e-14,"
        "        +3.727353649785e-09\n",
    ),
    ".": (
        "        +2.122785931511e-01,        +4.806347984241e-08,"
        "        +9.849547968700e-03,        +3.088679913680e-04\n",
        " 1.00e+00,        +4.398274617639e+00,        +1.242970341517e-04,"
        "        +1.769259680818e+04,        +4.783413462643e-15,"
        "        +3.045742429387e-09\n",
    ),
}


def _write_pass(directory, error_row: str, eig_row: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "error-indicators.csv").write_text(ERROR_HEADER + error_row)
    (directory / "eig.csv").write_text(EIG_HEADER + eig_row)


@pytest.fixture
def transmon_amr(tmp_path):
    for name, (error_row, eig_row) in TRANSMON_AMR.items():
        _write_pass(tmp_path / name, error_row, eig_row)
    return tmp_path


def _norms(history) -> list[float]:
    norms = []
    for results in history:
        indicators = results.error_indicators
        assert indicators is not None
        norms.append(indicators["norm"])
    return norms


def _mode1_frequency(results) -> float:
    row = results.csv_tables["eig.csv"][0]
    return float(next(v for k, v in row.items() if k.strip() == "Re{f} (GHz)"))


def test_passes_are_read_oldest_first(transmon_amr) -> None:
    norms = _norms(load_refinement_history(transmon_amr))
    assert norms == pytest.approx([0.3455421238025, 0.2761992653477, 0.2122785931511])


def test_error_indicator_columns(transmon_amr) -> None:
    final = load_refinement_history(transmon_amr)[-1]
    assert final.error_indicators == pytest.approx(
        {
            "norm": 0.2122785931511,
            "min": 4.806347984241e-08,
            "max": 9.849547968700e-03,
            "mean": 3.088679913680e-04,
        }
    )


def test_passes_are_ordered_by_number_not_by_name(tmp_path) -> None:
    """iteration10 must come after iteration2, which a string sort gets wrong."""
    for number, norm in ((2, "+2.0e-01"), (10, "+1.0e-01")):
        row = f"{norm:>27},{norm:>27},{norm:>27},{norm:>27}\n"
        _write_pass(tmp_path / f"iteration{number}", row, TRANSMON_AMR["."][1])
    _write_pass(tmp_path, *TRANSMON_AMR["."])
    norms = _norms(load_refinement_history(tmp_path))
    assert norms == pytest.approx([0.2, 0.1, 0.2122785931511])


def test_run_without_refinement_is_a_single_pass(tmp_path) -> None:
    _write_pass(tmp_path, *TRANSMON_AMR["."])
    assert len(load_refinement_history(tmp_path)) == 1


def test_convergence_separates_the_estimator_from_the_metric(transmon_amr) -> None:
    """The estimator keeps falling while the mode frequency still moves ~1 % a pass."""
    table = refinement_convergence(
        load_refinement_history(transmon_amr), _mode1_frequency
    )
    assert [row["pass"] for row in table] == [1, 2, 3]
    assert [row["error_norm"] for row in table] == pytest.approx(
        [0.3455421238025, 0.2761992653477, 0.2122785931511]
    )
    assert [row["value"] for row in table] == pytest.approx(
        [4.291377978227, 4.352933425564, 4.398274617639]
    )
    assert table[0]["relative_change"] is None
    assert [row["relative_change"] for row in table[1:]] == pytest.approx(
        [0.014344, 0.010416], abs=1e-6
    )


def test_simulation_directory_layout(tmp_path) -> None:
    """gsim runs write Palace's output under output/palace in the sim directory."""
    for name, (error_row, eig_row) in TRANSMON_AMR.items():
        _write_pass(tmp_path / "output" / "palace" / name, error_row, eig_row)
    assert _norms(load_refinement_history(tmp_path)) == pytest.approx(
        [0.3455421238025, 0.2761992653477, 0.2122785931511]
    )


def test_a_metric_crossing_zero_does_not_fail(transmon_amr) -> None:
    """A change relative to zero is undefined: NaN, not a ZeroDivisionError."""
    values = iter([0.0, 1.0, 2.0])
    table = refinement_convergence(
        load_refinement_history(transmon_amr), lambda _: next(values)
    )
    assert math.isnan(table[1]["relative_change"])
    assert table[2]["relative_change"] == pytest.approx(1.0)


def test_a_pass_without_error_indicators_reads_as_none(tmp_path) -> None:
    """Palace only writes error-indicators.csv when it estimated an error."""
    (tmp_path / "eig.csv").write_text(EIG_HEADER + TRANSMON_AMR["."][1])
    assert load_refinement_history(tmp_path)[-1].error_indicators is None


def test_convergence_lists_a_pass_that_has_no_error_estimate(tmp_path) -> None:
    _write_pass(tmp_path / "iteration1", *TRANSMON_AMR["iteration1"])
    (tmp_path / "eig.csv").write_text(EIG_HEADER + TRANSMON_AMR["."][1])
    table = refinement_convergence(load_refinement_history(tmp_path), _mode1_frequency)
    assert table[0]["error_norm"] == pytest.approx(0.3455421238025)
    assert table[1]["error_norm"] is None
    assert table[1]["value"] == pytest.approx(4.398274617639)


def test_a_field_beyond_the_header_is_ignored(tmp_path) -> None:
    """A trailing comma adds a field with no column name."""
    error_row = TRANSMON_AMR["."][0].rstrip("\n") + ",\n"
    _write_pass(tmp_path, error_row, TRANSMON_AMR["."][1])
    final = load_refinement_history(tmp_path)[-1]
    assert final.error_indicators == pytest.approx(
        {
            "norm": 0.2122785931511,
            "min": 4.806347984241e-08,
            "max": 9.849547968700e-03,
            "mean": 3.088679913680e-04,
        }
    )
