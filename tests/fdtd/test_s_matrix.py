"""Tests for building an S-matrix from GDSFactory FDTD results.

Results go through ``FDTDResult.from_file`` so the solver's JSON layout, including
descending wavelengths, is what the builder sees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gsim import fdtd
from gsim.fdtd import FDTDResult

# Descending wavelength with ascending frequency, as a frequency sweep reports it.
WAVELENGTH_NM = [1600.0, 1575.0, 1550.0, 1525.0, 1500.0]
LENGTH_UM = 2.0  # Keeps the phase step per sample under pi, so it unwraps.
N_GROUP = 3.4


def _delay_samples(scale: float = 1.0) -> list[dict[str, float]]:
    """Return exp(-j beta L) on WAVELENGTH_NM as FDTD JSON re/im records."""
    phase = 2 * np.pi * N_GROUP * LENGTH_UM / (np.array(WAVELENGTH_NM) / 1000)
    values = scale * np.exp(-1j * phase)
    return [{"re": float(v.real), "im": float(v.imag)} for v in values]


def _write_run(
    tmp_path: Path,
    name: str,
    s_parameters: dict[str, list[dict[str, float]]],
    *,
    excitation_type: str = "eigenmode",
    excited_port: str | None = "o1",
    below_noise_floor: list[bool] | None = None,
    wavelength_nm: list[float] | None = None,
    ports: list[str] | None = None,
) -> FDTDResult:
    wavelengths = WAVELENGTH_NM if wavelength_nm is None else wavelength_nm
    document: dict[str, Any] = {
        "schema_version": 1,
        "excitation_type": excitation_type,
        "excited_port": excited_port,
        "ports": ["o1", "o2"] if ports is None else ports,
        "frequencies": {
            "wavelength_nm": wavelengths,
            "hz": [299792458.0 / (wl * 1e-9) for wl in wavelengths],
            "below_noise_floor": (
                [False] * len(wavelengths)
                if below_noise_floor is None
                else below_noise_floor
            ),
        },
        "s_parameters": s_parameters,
    }
    path = tmp_path / f"sparams_{name}.json"
    path.write_text(json.dumps(document), encoding="utf8")
    return FDTDResult.from_file(path)


def _two_runs(tmp_path: Path) -> list[FDTDResult]:
    reflection = _delay_samples(0.0)
    return [
        _write_run(
            tmp_path,
            "o1",
            {"S(o1,o1)": reflection, "S(o2,o1)": _delay_samples()},
        ),
        _write_run(
            tmp_path,
            "o2",
            {"S(o2,o2)": reflection, "S(o1,o2)": _delay_samples(0.5)},
            excited_port="o2",
        ),
    ]


def test_two_runs_build_the_full_matrix_on_ascending_wavelength(
    tmp_path: Path,
) -> None:
    matrix = fdtd.s_matrix(_two_runs(tmp_path))

    expected_um = np.array([1.5, 1.525, 1.55, 1.575, 1.6])
    np.testing.assert_allclose(matrix.wavelength_um, expected_um)
    assert matrix.ports == ("o1", "o2")
    assert matrix.measured.all()
    # Passed through unconjugated: stored as the solver wrote it.
    through = np.exp(-2j * np.pi * N_GROUP * LENGTH_UM / expected_um)
    np.testing.assert_allclose(matrix.s[:, 1, 0], through, atol=1e-12)
    np.testing.assert_allclose(matrix.s[:, 0, 1], 0.5 * through, atol=1e-12)


def test_descending_json_reaches_sax_as_the_conjugate_at_the_right_wavelength(
    tmp_path: Path,
) -> None:
    model = fdtd.s_matrix(_two_runs(tmp_path)).sax_model()

    actual = model(wl=1.525)["o1", "o2"]

    np.testing.assert_allclose(
        actual, np.exp(2j * np.pi * N_GROUP * LENGTH_UM / 1.525), atol=1e-12
    )


def test_descending_json_writes_ascending_touchstone(tmp_path: Path) -> None:
    skrf = pytest.importorskip("skrf")
    matrix = fdtd.s_matrix(_two_runs(tmp_path))

    network = skrf.Network(str(matrix.write_touchstone(tmp_path / "device")))

    # Ascending frequency runs from the longest wavelength down, the JSON's order.
    expected_um = np.array(WAVELENGTH_NM) / 1000
    np.testing.assert_allclose(network.f, 299792458.0 / (expected_um * 1e-6))
    np.testing.assert_allclose(
        network.s[:, 1, 0],
        np.exp(-2j * np.pi * N_GROUP * LENGTH_UM / expected_um),
        atol=1e-12,
    )


def test_one_run_refuses_then_honours_each_flag(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path, "o1", {"S(o1,o1)": _delay_samples(0.0), "S(o2,o1)": _delay_samples()}
    )

    with pytest.raises(ValueError, match=r"S\(o1,o2\).*S\(o2,o2\)"):
        fdtd.s_matrix(run)

    matrix = fdtd.s_matrix(run, assume_reciprocal=True, assume_unmeasured_zero=True)

    np.testing.assert_array_equal(matrix.s[:, 0, 1], matrix.s[:, 1, 0])
    np.testing.assert_array_equal(matrix.s[:, 1, 1], np.zeros(5))
    assert matrix.measured.tolist() == [[True, False], [True, False]]


def test_no_results_refused() -> None:
    with pytest.raises(ValueError, match="no results"):
        fdtd.s_matrix([])


def test_non_eigenmode_result_refused_naming_the_source(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path, "dipole", {}, excitation_type="dipole", excited_port=None
    )

    with pytest.raises(ValueError, match=r"sparams_dipole\.json.*dipole.*no S-param"):
        fdtd.s_matrix(run, assume_unmeasured_zero=True)


def test_different_port_lists_refused(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, "a", {"S(o2,o1)": _delay_samples()}),
        _write_run(
            tmp_path, "b", {"S(o1,o2)": _delay_samples()}, ports=["o1", "o2", "o3"]
        ),
    ]

    with pytest.raises(ValueError, match="port list"):
        fdtd.s_matrix(runs, assume_unmeasured_zero=True)


def test_different_wavelength_grids_refused(tmp_path: Path) -> None:
    shifted = [wl + 1.0 for wl in WAVELENGTH_NM]
    runs = [
        _write_run(tmp_path, "a", {"S(o2,o1)": _delay_samples()}),
        _write_run(
            tmp_path, "b", {"S(o1,o2)": _delay_samples()}, wavelength_nm=shifted
        ),
    ]

    with pytest.raises(ValueError, match="wavelength grid"):
        fdtd.s_matrix(runs, assume_unmeasured_zero=True)


def test_same_entry_in_two_runs_refused(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, "a", {"S(o2,o1)": _delay_samples()}),
        _write_run(tmp_path, "b", {"S(o2,o1)": _delay_samples()}),
    ]

    with pytest.raises(ValueError, match=r"S\(o2,o1\)"):
        fdtd.s_matrix(runs, assume_unmeasured_zero=True)


def test_key_with_unknown_port_refused(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "a", {"S(o9,o1)": _delay_samples()})

    with pytest.raises(ValueError, match="o9"):
        fdtd.s_matrix(run, assume_unmeasured_zero=True)


def test_below_noise_floor_samples_are_dropped_and_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run = _write_run(
        tmp_path,
        "a",
        {"S(o2,o1)": _delay_samples()},
        below_noise_floor=[False, True, False, False, False],
    )

    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        matrix = fdtd.s_matrix(run, assume_unmeasured_zero=True)

    # The flagged sample is 1575 nm, the second of the descending JSON.
    np.testing.assert_allclose(matrix.wavelength_um, [1.5, 1.525, 1.55, 1.6])
    assert any("1 of 5" in record.getMessage() for record in caplog.records)


def test_all_below_noise_floor_refused(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path, "a", {"S(o2,o1)": _delay_samples()}, below_noise_floor=[True] * 5
    )

    with pytest.raises(ValueError, match="invalid"):
        fdtd.s_matrix(run, assume_unmeasured_zero=True)


def test_inverted_phase_warns_because_the_convention_is_inferred(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    conjugated = [{"re": s["re"], "im": -s["im"]} for s in _delay_samples()]
    run = _write_run(tmp_path, "a", {"S(o2,o1)": conjugated})

    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        fdtd.s_matrix(run, assume_unmeasured_zero=True)

    assert any("convention" in record.getMessage() for record in caplog.records)
