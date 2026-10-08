"""Tests for reading and checking the capacitance matrices of a Palace run.

The CSV files below are verbatim from Palace's own regression references,
``test/data/regression/ref/spheres`` (two terminals) and
``test/data/regression/ref/cavity2d/electrostatic`` (one terminal), in
awslabs/palace (Apache-2.0).
"""

from __future__ import annotations

import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gsim.palace import CapacitanceMatrices, ElectrostaticSim, load_capacitance
from gsim.palace.models import TerminalConfig

SPHERES = {
    "terminal-C.csv": """\
        i,                C[i][1] (F),                C[i][2] (F)
 1.00e+00,        +1.237445610357e-12,        -4.770975738888e-13
 2.00e+00,        -4.770975738888e-13,        +2.478413459856e-12
""",
    "terminal-Cm.csv": """\
        i,              C_m[i][1] (F),              C_m[i][2] (F)
 1.00e+00,        +7.603480364677e-13,        +4.770975738888e-13
 2.00e+00,        +4.770975738888e-13,        +2.001315885967e-12
""",
    "terminal-Cinv.csv": """\
        i,            C\u207b\u00b9[i][1] (1/F),            C\u207b\u00b9[i][2] (1/F)
 1.00e+00,        +8.729021681592e+11,        +1.680347179422e+11
 2.00e+00,        +1.680347179422e+11,        +4.358308142509e+11
""",
    "terminal-V.csv": """\
        i,               V_inc[i] (V)
 1.00e+00,        +1.940954181484e+01
 2.00e+00,        +1.940954181484e+01
""",
    "domain-E.csv": """\
        i,                 E_elec (J),                  E_mag (J),                  E_cap (J),                  E_ind (J),              E_elec[1] (J),                  p_elec[1],               E_mag[1] (J),                   p_mag[1]
 1.00e+00,        +2.330916363410e-10,        +0.000000000000e-09,        +0.000000000000e-09,        +0.000000000000e-09,        +2.330916363410e-10,        +1.000000000000e+00,        +0.000000000000e-09,        +0.000000000000e+00
 2.00e+00,        +4.668467398102e-10,        +0.000000000000e-09,        +0.000000000000e-09,        +0.000000000000e-09,        +4.668467398102e-10,        +1.000000000000e+00,        +0.000000000000e-09,        +0.000000000000e+00
""",  # noqa: E501
}

CAVITY = {
    "terminal-C.csv": """\
        i,                C[i][1] (F)
 1.00e+00,        +1.502328996030e-10
""",
    "terminal-Cm.csv": """\
        i,              C_m[i][1] (F)
 1.00e+00,        +1.502328996030e-10
""",
    "terminal-Cinv.csv": """\
        i,            C\u207b\u00b9[i][1] (1/F)
 1.00e+00,        +6.656331620054e+09
""",
    "terminal-V.csv": """\
        i,               V_inc[i] (V)
 1.00e+00,        +1.940954181355e+01
""",
    "domain-E.csv": """\
        i,                 E_elec (J),                  E_mag (J),                  E_cap (J),                  E_ind (J),              E_elec[1] (J),                  p_elec[1],               E_mag[1] (J),                   p_mag[1]
 1.00e+00,        +2.829864367612e-08,        +0.000000000000e+00,        +0.000000000000e+00,        +0.000000000000e+00,        +2.829864367612e-08,        +1.000000000000e+00,        +0.000000000000e+00,        +0.000000000000e+00
""",  # noqa: E501
}


def _write(directory: Path, files: dict[str, str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


def _spheres(tmp_path: Path, names: list[str] | None = None) -> CapacitanceMatrices:
    return load_capacitance(_write(tmp_path / "run", SPHERES), terminal_names=names)


def _from_maxwell(maxwell: np.ndarray) -> CapacitanceMatrices:
    """Every matrix Palace writes, derived from a Maxwell matrix."""
    voltage = np.ones(len(maxwell)) * 2.0
    ground = maxwell.sum(axis=1)
    return CapacitanceMatrices(
        terminals=tuple(f"T{i}" for i in range(1, len(maxwell) + 1)),
        maxwell=maxwell,
        mutual=-(maxwell - np.diag(np.diag(maxwell))) + np.diag(ground),
        inverse=np.linalg.inv(maxwell),
        excitation_voltage=voltage,
        stored_energy=0.5 * np.diag(maxwell) * voltage**2,
    )


def _clean() -> CapacitanceMatrices:
    """A consistent two-terminal set, to be broken one property at a time."""
    maxwell = np.array([[3.0, -1.0], [-1.0, 2.0]]) * 1e-12
    voltage = np.array([2.0, 3.0])
    return CapacitanceMatrices(
        terminals=("A", "B"),
        maxwell=maxwell,
        mutual=np.array([[2.0, 1.0], [1.0, 1.0]]) * 1e-12,
        inverse=np.linalg.inv(maxwell),
        excitation_voltage=voltage,
        stored_energy=0.5 * np.diag(maxwell) * voltage**2,
    )


def test_reads_the_two_sphere_matrices(tmp_path) -> None:
    cap = _spheres(tmp_path)
    assert cap.terminals == ("T1", "T2")
    assert cap.maxwell == pytest.approx(
        np.array(
            [
                [1.237445610357e-12, -4.770975738888e-13],
                [-4.770975738888e-13, 2.478413459856e-12],
            ]
        )
    )
    assert cap.mutual == pytest.approx(
        np.array(
            [
                [7.603480364677e-13, 4.770975738888e-13],
                [4.770975738888e-13, 2.001315885967e-12],
            ]
        )
    )
    assert cap.excitation_voltage == pytest.approx([19.40954181484, 19.40954181484])
    assert cap.stored_energy == pytest.approx([2.330916363410e-10, 4.668467398102e-10])


def test_terminal_names_label_the_matrices(tmp_path) -> None:
    cap = _spheres(tmp_path, names=["plus", "minus"])
    frame = cap.maxwell_frame()
    assert list(frame.index) == ["plus", "minus"]
    assert list(frame.columns) == ["plus", "minus"]
    assert frame.loc["plus", "minus"] == pytest.approx(-4.770975738888e-13)
    assert cap.mutual_frame().loc["minus", "minus"] == pytest.approx(2.001315885967e-12)


def test_electrode_to_electrode_and_to_ground_are_reported_separately(tmp_path) -> None:
    cap = _spheres(tmp_path, names=["plus", "minus"])
    assert cap.between("plus", "minus") == pytest.approx(4.770975738888e-13)
    assert cap.between("minus", "plus") == cap.between("plus", "minus")
    assert cap.to_ground("plus") == pytest.approx(7.603480364677e-13)
    assert cap.to_ground("minus") == pytest.approx(2.001315885967e-12)
    with pytest.raises(KeyError, match="gnd"):
        cap.to_ground("gnd")


def test_reads_the_results_dict_that_run_returns(tmp_path) -> None:
    run = _write(tmp_path / "run", SPHERES)
    results = {name: run / name for name in SPHERES}
    assert load_capacitance(results).maxwell == pytest.approx(
        _spheres(tmp_path).maxwell
    )


def test_finds_the_files_in_the_palace_output_subdirectory(tmp_path) -> None:
    _write(tmp_path / "sim" / "output" / "palace", SPHERES)
    assert load_capacitance(tmp_path / "sim").maxwell.shape == (2, 2)


def test_a_single_terminal_run_loads_as_a_1x1_matrix(tmp_path) -> None:
    cap = load_capacitance(_write(tmp_path / "run", CAVITY))
    assert cap.maxwell.shape == (1, 1)
    assert cap.maxwell[0, 0] == pytest.approx(1.502328996030e-10)
    assert cap.to_ground("T1") == pytest.approx(1.502328996030e-10)


def test_the_optional_files_can_be_missing(tmp_path) -> None:
    core = {
        k: v for k, v in SPHERES.items() if k in ("terminal-C.csv", "terminal-Cm.csv")
    }
    cap = load_capacitance(_write(tmp_path / "run", core))
    assert cap.inverse is None
    assert cap.excitation_voltage is None
    assert cap.stored_energy is None
    assert cap.problems() == []


def test_a_missing_required_file_names_it(tmp_path) -> None:
    only_maxwell = {"terminal-C.csv": SPHERES["terminal-C.csv"]}
    with pytest.raises(FileNotFoundError, match=r"terminal-Cm\.csv"):
        load_capacitance(_write(tmp_path / "run", only_maxwell))


def test_an_energy_table_without_the_expected_column_is_ignored(tmp_path) -> None:
    """Another Palace version may name the column differently."""
    files = dict(SPHERES)
    files["domain-E.csv"] = files["domain-E.csv"].replace("E_elec (J)", "E_total (J)")
    cap = load_capacitance(_write(tmp_path / "run", files))
    assert cap.stored_energy is None
    assert cap.problems() == []


def test_a_matrix_that_is_not_square_is_rejected(tmp_path) -> None:
    files = dict(SPHERES)
    files["terminal-C.csv"] = textwrap.dedent("""
        i,  C[i][1] (F),  C[i][2] (F),  C[i][3] (F)
        1,  +1.0e-12,  -1.0e-13,  -1.0e-13
        2,  -1.0e-13,  +2.0e-12,  -1.0e-13
    """).lstrip()
    with pytest.raises(ValueError, match="not square"):
        load_capacitance(_write(tmp_path / "run", files))


def test_the_wrong_number_of_names_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="2 terminals"):
        _spheres(tmp_path, names=["only-one"])


@pytest.mark.parametrize("files", [SPHERES, CAVITY], ids=["spheres", "cavity"])
def test_real_palace_output_has_no_problems(tmp_path, files) -> None:
    cap = load_capacitance(_write(tmp_path / "run", files))
    assert cap.problems() == []


def test_a_consistent_set_has_no_problems() -> None:
    assert _clean().problems() == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda c: replace(c, maxwell=c.maxwell + np.array([[0, 5e-13], [0, 0]])),
            "is not symmetric",
        ),
        (
            lambda c: replace(c, maxwell=np.array([[1.0, -2.0], [-2.0, 1.0]]) * 1e-12),
            "eigenvalue",
        ),
        (
            lambda c: replace(c, maxwell=np.array([[3.0, 1.0], [1.0, 2.0]]) * 1e-12),
            "entry is positive",
        ),
        (
            lambda c: replace(c, maxwell=np.array([[1.0, -2.0], [-2.0, 4.0]]) * 1e-12),
            "negative capacitance to ground",
        ),
        (
            lambda c: replace(c, mutual=-c.mutual),
            "mutual matrix does not follow",
        ),
        (
            lambda c: replace(c, mutual=c.mutual * np.array([[1.5, 1.0], [1.0, 1.0]])),
            "mutual matrix does not follow",
        ),
        (
            lambda c: replace(c, inverse=np.eye(2) * 1e12),
            "does not invert",
        ),
        (
            lambda c: replace(c, stored_energy=c.stored_energy * np.array([1.0, 1.2])),
            "stored energy does not match",
        ),
        (
            lambda c: replace(c, maxwell=np.zeros((2, 2)), mutual=np.zeros((2, 2))),
            "self capacitance",
        ),
    ],
    ids=[
        "asymmetric",
        "negative-energy",
        "positive-off-diagonal",
        "negative-ground-capacitance",
        "mutual-sign",
        "mutual-diagonal",
        "inverse",
        "stored-energy",
        "no-self-capacitance",
    ],
)
def test_each_problem_is_reported(change, expected) -> None:
    problems = change(_clean()).problems()
    assert any(expected in text for text in problems), problems


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("maxwell", "the Maxwell matrix"),
        ("mutual", "the mutual matrix"),
        ("inverse", "the inverse"),
        ("excitation_voltage", "the excitation voltages"),
        ("stored_energy", "the stored energies"),
    ],
)
def test_a_value_that_is_not_finite_is_reported_by_name(name, label) -> None:
    """A NaN fails every comparison, so no other check would notice it."""
    clean = _clean()
    broken = replace(clean, **{name: np.full_like(getattr(clean, name), np.nan)})
    assert broken.problems() == [f"these are not finite (NaN or infinity): {label}"]


def test_infinity_is_not_finite_either() -> None:
    clean = _clean()
    broken = replace(clean, stored_energy=np.array([np.inf, 1e-12]))
    assert broken.problems() == [
        "these are not finite (NaN or infinity): the stored energies"
    ]


def test_a_nan_in_the_files_is_reported(tmp_path) -> None:
    """np.loadtxt reads the text nan as NaN, which no comparison would flag."""
    broken = dict(SPHERES)
    broken["terminal-C.csv"] = broken["terminal-C.csv"].replace(
        "-4.770975738888e-13,", "nan,", 1
    )
    cap = load_capacitance(_write(tmp_path / "run", broken))
    assert any("not finite" in text for text in cap.problems())


def test_a_slightly_positive_coupling_is_numerical_noise_not_a_problem() -> None:
    """A weakly coupled pair can come out just above zero on a discretization."""
    noisy = np.array([[3.0, 1e-4], [1e-4, 2.0]]) * 1e-12
    assert _from_maxwell(noisy).problems() == []


def test_a_clearly_positive_coupling_is_still_a_problem() -> None:
    wrong = np.array([[3.0, 0.1], [0.1, 2.0]]) * 1e-12
    assert any("entry is positive" in text for text in _from_maxwell(wrong).problems())


def test_electrostatic_sim_labels_the_matrices_with_its_terminals(tmp_path) -> None:
    sim = ElectrostaticSim(
        terminals=[
            TerminalConfig(name="plus", layer="m5"),
            TerminalConfig(name="minus", layer="m4"),
        ]
    )
    run = _write(tmp_path / "run", SPHERES)
    cap = sim.load_capacitance(run)
    assert cap.terminals == ("plus", "minus")
    assert cap.between("plus", "minus") == pytest.approx(4.770975738888e-13)


def test_duplicate_terminal_names_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="different"):
        _spheres(tmp_path, names=["same", "same"])


def test_between_needs_two_different_terminals(tmp_path) -> None:
    with pytest.raises(ValueError, match="to_ground"):
        _spheres(tmp_path).between("T1", "T1")
