"""Preserve exact zero coupling when reading Palace's padded CSV numbers."""

from pathlib import Path

import numpy as np
import pytest

from gsim.palace.results import SParam, SParams, load_sparams


@pytest.fixture
def zero_transmission_output(tmp_path: Path) -> Path:
    (tmp_path / "port-S.csv").write_text(
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.),"
        " |S[2][1]| (dB), arg(S[2][1]) (deg.),"
        " |S[1][2]| (dB), arg(S[1][2]) (deg.),"
        " |S[2][2]| (dB), arg(S[2][2]) (deg.)\n"
        " 1.0 , 0.0 , 0.0 ,   -inf   , -90.0 , -inf , 90.0 , 0.0 , 0.0\n"
        " 2.0 , -20.0 , 45.0 , -400.0 , -90.0 , -400.0 , 90.0 , -20.0 , 45.0\n"
    )
    return tmp_path


def test_zero_and_tiny_transmission(zero_transmission_output: Path) -> None:
    parameters = load_sparams(zero_transmission_output)
    assert parameters.freq.dtype.kind == "f"
    for pair in parameters.keys():  # noqa: SIM118 - SParams is not iterable
        assert parameters[pair].db.dtype.kind == "f"
        assert parameters[pair].deg.dtype.kind == "f"
    assert np.isneginf(parameters.s21.db[0])
    assert parameters.s21.mag[0] == parameters.s21.complex[0] == 0
    assert parameters.s21.db[1] == -400
    assert parameters.s21.mag[1] == pytest.approx(1e-20, abs=0)
    network = parameters.to_skrf(z0=1)
    np.testing.assert_array_equal(network.s[0], np.eye(2, dtype=complex))
    np.testing.assert_allclose(network.s[1, 1, 0], -1e-20j, atol=0)


def test_zero_round_trip(zero_transmission_output: Path, tmp_path: Path) -> None:
    parameters = load_sparams(zero_transmission_output)
    restored = SParams.from_file(parameters.save_npz(tmp_path / "zero.npz"))
    np.testing.assert_array_equal(restored.s21.db, parameters.s21.db)
    np.testing.assert_array_equal(restored.s21.complex, parameters.s21.complex)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("f (GHz)", "bad-frequency"),
        ("f (GHz)", "inf"),
        ("|S[2][1]| (dB)", "bad-magnitude"),
        ("|S[2][1]| (dB)", "nan"),
        ("|S[2][1]| (dB)", "inf"),
        ("arg(S[2][1]) (deg.)", "bad-phase"),
        ("arg(S[2][1]) (deg.)", "-inf"),
        ("arg(S[2][1]) (deg.)", ""),
    ],
)
def test_malformed_numeric_cells_report_context(
    tmp_path: Path, column: str, value: str
) -> None:
    columns = ["f (GHz)", "|S[2][1]| (dB)", "arg(S[2][1]) (deg.)"]
    cells = ["1.0", "-inf", "0.0"]
    cells[columns.index(column)] = value
    csv_path = tmp_path / "port-S.csv"
    csv_path.write_text(
        ",".join(columns) + "\n1.0,-20.0,0.0\n" + ",".join(cells) + "\n"
    )
    with pytest.raises(ValueError, match="Invalid numeric value") as error:
        load_sparams(tmp_path)
    message = str(error.value)
    assert str(csv_path) in message
    assert "row 3" in message
    assert column in message
    assert repr(value) in message


def test_sparam_coerces_numeric_arrays() -> None:
    parameter = SParam(db=np.array([" -inf "]), deg=np.array([" 90.0 "]))
    assert parameter.db.dtype.kind == parameter.deg.dtype.kind == "f"
    assert parameter.mag[0] == parameter.complex[0] == 0
