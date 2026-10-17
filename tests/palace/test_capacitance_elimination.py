"""Tests for eliminating floating terminals and listing capacitance pairs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gsim.palace import CapacitanceMatrices, ElectrostaticSim
from gsim.palace.models import TerminalConfig
from gsim.palace.terminals import ResolvedTerminal

# three plates in a row: T1 - T2 - T3, with capacitances to ground (F)
C12, C23 = 2e-15, 3e-15
G1, G2, G3 = 1e-15, 0.5e-15, 1.5e-15


def _from_maxwell(maxwell: np.ndarray, names=("T1", "T2", "T3")) -> CapacitanceMatrices:
    voltage = np.ones(len(maxwell)) * 2.0
    ground = maxwell.sum(axis=1)
    return CapacitanceMatrices(
        terminals=tuple(names),
        maxwell=maxwell,
        mutual=-(maxwell - np.diag(np.diag(maxwell))) + np.diag(ground),
        inverse=np.linalg.inv(maxwell),
        excitation_voltage=voltage,
        stored_energy=0.5 * np.diag(maxwell) * voltage**2,
    )


def _three_plates() -> CapacitanceMatrices:
    return _from_maxwell(
        np.array(
            [
                [C12 + G1, -C12, 0.0],
                [-C12, C12 + C23 + G2, -C23],
                [0.0, -C23, C23 + G3],
            ]
        )
    )


def test_eliminating_the_middle_plate_gives_the_series_capacitance() -> None:
    reduced = _three_plates().eliminate(["T2"])
    assert reduced.terminals == ("T1", "T3")
    expected = C12 * C23 / (C12 + C23 + G2)
    assert reduced.between("T1", "T3") == pytest.approx(expected, rel=1e-12)


def test_the_ground_capacitances_follow_the_floating_plate() -> None:
    reduced = _three_plates().eliminate(["T2"])
    # the floating plate sits at V2 = (C12 V1 + C23 V3) / total; with V1 = V3 = 1
    # the charge on plate 1 is G1 + C12 (1 - V2) = G1 + C12 G2 / total
    total = C12 + C23 + G2
    assert reduced.to_ground("T1") == pytest.approx(G1 + C12 * G2 / total)
    assert reduced.to_ground("T3") == pytest.approx(G3 + C23 * G2 / total)


def test_the_reduced_inverse_is_the_inverse_of_the_reduced_matrix() -> None:
    reduced = _three_plates().eliminate(["T2"])
    assert reduced.inverse is not None
    np.testing.assert_allclose(
        reduced.inverse, np.linalg.inv(reduced.maxwell), rtol=1e-10
    )


def test_the_reduced_matrices_are_consistent() -> None:
    assert _three_plates().eliminate(["T2"]).problems() == []
    assert _three_plates().eliminate(["T1", "T3"]).problems() == []


def test_voltages_and_energies_are_dropped() -> None:
    reduced = _three_plates().eliminate(["T2"])
    assert reduced.excitation_voltage is None
    assert reduced.stored_energy is None


def test_without_a_stored_inverse_there_is_none_after_elimination() -> None:
    cap = _three_plates()
    bare = CapacitanceMatrices(cap.terminals, cap.maxwell, cap.mutual)
    assert bare.eliminate(["T2"]).inverse is None


def test_eliminating_the_end_plates_leaves_the_middle_one() -> None:
    reduced = _three_plates().eliminate(["T1", "T3"])
    assert reduced.terminals == ("T2",)
    c = _three_plates().maxwell
    # one plate left: the Schur complement is its capacitance to everything else
    ends = [0, 2]
    expected = c[1, 1] - c[1, ends] @ np.linalg.solve(c[np.ix_(ends, ends)], c[ends, 1])
    assert reduced.maxwell[0, 0] == pytest.approx(expected)


def test_eliminating_nothing_returns_the_same_object() -> None:
    cap = _three_plates()
    assert cap.eliminate([]) is cap


def test_eliminate_rejects_unknown_repeated_and_all_terminals() -> None:
    cap = _three_plates()
    with pytest.raises(KeyError, match="unknown terminal 'X'"):
        cap.eliminate(["X"])
    with pytest.raises(ValueError, match="must be different"):
        cap.eliminate(["T2", "T2"])
    with pytest.raises(ValueError, match="cannot eliminate every terminal"):
        cap.eliminate(["T1", "T2", "T3"])


def test_a_floating_plate_without_any_coupling_is_a_singular_block() -> None:
    maxwell = np.array([[3.0, 0.0], [0.0, 0.0]]) * 1e-15
    cap = CapacitanceMatrices(("T1", "T2"), maxwell, maxwell.copy())
    # a zero block cannot be inverted; numpy raises, we explain
    with pytest.raises(ValueError, match="singular"):
        cap.eliminate(["T2"])


# ----- pairs ---------------------------------------------------------------------


def test_pairs_lists_electrode_pairs_and_ground_separately() -> None:
    cap = _three_plates()
    table = cap.pairs()
    assert list(table.columns) == ["first", "second", "capacitance_F"]
    assert len(table) == 3 + 3  # three pairs, three grounds
    rows = {
        (r["first"], r["second"]): r["capacitance_F"] for r in table.to_dict("records")
    }
    for a, b in [("T1", "T2"), ("T1", "T3"), ("T2", "T3")]:
        assert rows[(a, b)] == cap.between(a, b)
    for name in cap.terminals:
        assert rows[(name, "ground")] == cap.to_ground(name)
    assert rows[("T1", "T2")] == pytest.approx(C12)
    assert rows[("T1", "ground")] == pytest.approx(G1)


def test_pairs_of_one_terminal_is_only_its_ground() -> None:
    table = _three_plates().eliminate(["T1", "T3"]).pairs()
    assert [(r["first"], r["second"]) for r in table.to_dict("records")] == [
        ("T2", "ground")
    ]


# ----- ElectrostaticSim.load_capacitance -----------------------------------------


def _write_run(path, cap: CapacitanceMatrices):
    path.mkdir(parents=True, exist_ok=True)
    n = len(cap.maxwell)

    def table(name, matrix, label):
        header = ",".join(["i"] + [f"{label}[i][{j + 1}] (F)" for j in range(n)])
        lines = [header] + [
            ",".join([f"{i + 1}"] + [f"{v:+.12e}" for v in row])
            for i, row in enumerate(matrix)
        ]
        (path / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    table("terminal-C.csv", cap.maxwell, "C")
    table("terminal-Cm.csv", cap.mutual, "C_m")
    return path


def _net_sim(out_dir=None):
    sim = ElectrostaticSim(
        terminals=[
            TerminalConfig(name="TOP", net="PLUS"),
            TerminalConfig(name="LEFT", net="MINUS"),
            TerminalConfig(name="SHIELD", net="net3", floating=True),
        ]
    )
    if out_dir is not None:
        sim.set_output_dir(out_dir)
    return sim


def test_load_capacitance_eliminates_the_floating_terminal(tmp_path) -> None:
    run = _write_run(tmp_path / "run", _three_plates())
    sim = _net_sim()
    sim._resolved_terminals = [
        ResolvedTerminal("TOP", (0,), False),
        ResolvedTerminal("LEFT", (1,), False),
        ResolvedTerminal("SHIELD", (2,), True),
    ]
    cap = sim.load_capacitance(run)
    assert cap.terminals == ("TOP", "LEFT")
    expected = _three_plates().eliminate(["T3"]).between("T1", "T2")
    assert cap.between("TOP", "LEFT") == pytest.approx(expected, rel=1e-9)
    assert cap.problems() == []


def test_load_capacitance_can_keep_the_floating_terminal(tmp_path) -> None:
    run = _write_run(tmp_path / "run", _three_plates())
    sim = _net_sim()
    sim._resolved_terminals = [
        ResolvedTerminal("TOP", (0,), False),
        ResolvedTerminal("LEFT", (1,), False),
        ResolvedTerminal("SHIELD", (2,), True),
    ]
    cap = sim.load_capacitance(run, eliminate_floating=False)
    assert cap.terminals == ("TOP", "LEFT", "SHIELD")


def test_load_capacitance_reads_terminals_json_without_a_config_written(
    tmp_path,
) -> None:
    out = tmp_path / "es"
    out.mkdir()
    (out / "terminals.json").write_text(
        json.dumps(
            [
                {"index": 1, "name": "TOP", "nets": [0], "floating": False},
                {"index": 2, "name": "LEFT", "nets": [1], "floating": False},
                {"index": 3, "name": "net3", "nets": [2], "floating": True},
            ]
        )
    )
    run = _write_run(tmp_path / "run", _three_plates())
    sim = ElectrostaticSim(terminals=[TerminalConfig(name="TOP", net="PLUS")])
    sim.set_output_dir(out)
    sim.set_unassigned_conductors("floating")
    cap = sim.load_capacitance(run)
    assert cap.terminals == ("TOP", "LEFT")


def test_net_terminals_without_terminals_json_explain_what_is_missing(tmp_path) -> None:
    run = _write_run(tmp_path / "run", _three_plates())
    with pytest.raises(ValueError, match="write_config"):
        _net_sim().load_capacitance(run)


def test_layer_terminals_load_exactly_as_before(tmp_path) -> None:
    sim = ElectrostaticSim(
        terminals=[
            TerminalConfig(name="plus", layer="m5"),
            TerminalConfig(name="minus", layer="m4"),
        ]
    )
    two = _from_maxwell(np.array([[3.0, -1.0], [-1.0, 2.0]]) * 1e-12, ("a", "b"))
    cap = sim.load_capacitance(_write_run(tmp_path / "run", two))
    assert cap.terminals == ("plus", "minus")
    assert cap.between("plus", "minus") == pytest.approx(1e-12)
