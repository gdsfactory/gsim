"""Tests for building an S-matrix from Meep results.

Results go through ``SParameterResult.from_csv``, so the runner's CSV layout (magnitude,
phase in degrees, descending wavelength) is what the builder sees. Expected values come
from an independent reference: the analytic line, ``sax.models.straight`` or
``skrf.Network`` reading the written file.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gsim.meep import SParameterResult, s_matrix

# Descending, as Meep's ascending frequencies give it. Fine enough to unwrap the phase.
WAVELENGTH_UM = np.linspace(1.6, 1.5, 51)
OFF_GRID_UM = np.array([1.5013, 1.5177, 1.5333, 1.5491, 1.5719, 1.5987])

# Parameters of the dispersive line that sax.models.straight also evaluates.
WL0_UM = 1.55
NEFF = 2.34
NG = 3.4
LENGTH_UM = 100.0


def _line_phase(wavelength_um: np.ndarray) -> np.ndarray:
    """Return beta * L of the first-order dispersive line, in radians."""
    neff = NEFF - (wavelength_um - WL0_UM) * (NG - NEFF) / WL0_UM
    return 2 * np.pi * neff * LENGTH_UM / wavelength_um


def _meep_line(wavelength_um: np.ndarray = WAVELENGTH_UM) -> np.ndarray:
    """Return the line's transmission as Meep reports it, exp(+j beta L)."""
    return np.exp(1j * _line_phase(wavelength_um))


def _load_csv(
    tmp_path: Path,
    columns: dict[str, np.ndarray],
    *,
    wavelength_um: np.ndarray = WAVELENGTH_UM,
) -> SParameterResult:
    """Write a runner-style CSV in its own directory and load it."""
    names = list(columns)
    header = ["wavelength"] + [
        f"{n}_{part}" for n in names for part in ("mag", "phase")
    ]
    rows = [",".join(header)]
    for k, wavelength in enumerate(wavelength_um):
        row = [f"{wavelength:.12f}"]
        for name in names:
            value = columns[name][k]
            row += [f"{abs(value):.15f}", f"{np.angle(value, deg=True):.12f}"]
        rows.append(",".join(row))
    directory = tmp_path / "-".join(names)
    directory.mkdir(parents=True)
    path = directory / "s_parameters.csv"
    path.write_text("\n".join(rows) + "\n")
    return SParameterResult.from_csv(path)


def _line_column(tmp_path: Path) -> SParameterResult:
    """Return the source-1 column of a matched line, one run of a Meep simulation."""
    return _load_csv(
        tmp_path, {"S11": np.zeros(WAVELENGTH_UM.size), "S21": _meep_line()}
    )


def test_csv_line_reaches_sax_matching_its_straight_model(tmp_path: Path) -> None:
    sax = pytest.importorskip("sax")
    matrix = s_matrix(
        _line_column(tmp_path), assume_reciprocal=True, assume_unmeasured_zero=True
    )

    actual = matrix.sax_model()(wl=OFF_GRID_UM)
    reference = sax.models.straight(
        wl=OFF_GRID_UM, wl0=WL0_UM, neff=NEFF, ng=NG, length=LENGTH_UM
    )
    named = sax.PortNamer(1, 1)

    np.testing.assert_array_less(
        np.abs(
            actual["port_1", "port_2"] - np.asarray(reference[named.in0, named.out0])
        ),
        1e-3,
    )


def test_csv_line_reaches_touchstone_as_the_conjugate(tmp_path: Path) -> None:
    skrf = pytest.importorskip("skrf")
    matrix = s_matrix(
        _line_column(tmp_path), assume_reciprocal=True, assume_unmeasured_zero=True
    )

    network = skrf.Network(str(matrix.write_touchstone(tmp_path / "device")))

    # Ascending frequency runs from the longest wavelength down, the CSV's order.
    np.testing.assert_allclose(network.f, 299792458.0 / (WAVELENGTH_UM * 1e-6))
    np.testing.assert_allclose(
        network.s[:, 1, 0], np.exp(-1j * _line_phase(WAVELENGTH_UM)), atol=1e-9
    )


def test_two_columns_merge_without_flags_and_keep_receive_source_order(
    tmp_path: Path,
) -> None:
    skrf = pytest.importorskip("skrf")
    wavelength = WAVELENGTH_UM
    # Non-reciprocal on purpose: only a swapped index could mix these two up.
    forward = 0.9 * _meep_line()
    backward = 0.5 * _meep_line()
    runs = [
        _load_csv(tmp_path, {"S11": np.full(wavelength.size, 0.1), "S21": forward}),
        _load_csv(tmp_path, {"S12": backward, "S22": np.full(wavelength.size, 0.2)}),
    ]

    matrix = s_matrix(runs)
    model = matrix.sax_model()(wl=wavelength)
    network = skrf.Network(str(matrix.write_touchstone(tmp_path / "device")))

    assert matrix.ports == ("port_1", "port_2")
    np.testing.assert_array_equal(matrix.measured, np.ones((2, 2), dtype=bool))
    # SAX keys (in, out) and keeps Meep's sign, S21 being from port 1 to port 2.
    np.testing.assert_allclose(model["port_1", "port_2"], forward, atol=1e-9)
    np.testing.assert_allclose(model["port_2", "port_1"], backward, atol=1e-9)
    np.testing.assert_allclose(model["port_1", "port_1"], 0.1, atol=1e-9)
    np.testing.assert_allclose(model["port_2", "port_2"], 0.2, atol=1e-9)
    np.testing.assert_allclose(network.s[:, 1, 0], np.conj(forward), atol=1e-9)
    np.testing.assert_allclose(network.s[:, 0, 1], np.conj(backward), atol=1e-9)


def test_port_names_rename_by_one_based_index(tmp_path: Path) -> None:
    result = _load_csv(
        tmp_path, {"S11": np.full(WAVELENGTH_UM.size, 0.1), "S21": _meep_line()}
    )

    matrix = s_matrix(
        result,
        port_names=["o1", "o2"],
        assume_reciprocal=True,
        assume_unmeasured_zero=True,
    )
    model = matrix.sax_model()(wl=WAVELENGTH_UM)

    assert matrix.ports == ("o1", "o2")
    np.testing.assert_allclose(model["o1", "o2"], _meep_line(), atol=1e-9)
    np.testing.assert_allclose(model["o1", "o1"], 0.1, atol=1e-9)


def test_default_port_names_are_the_ones_the_result_reports(tmp_path: Path) -> None:
    result = _line_column(tmp_path)

    matrix = s_matrix(result, assume_reciprocal=True, assume_unmeasured_zero=True)

    assert list(matrix.ports) == result.port_names == ["port_1", "port_2"]


def test_a_gap_in_the_indices_is_a_port_nothing_measured(tmp_path: Path) -> None:
    result = _load_csv(
        tmp_path, {"S11": np.zeros(WAVELENGTH_UM.size), "S31": _meep_line()}
    )

    with pytest.raises(ValueError, match=r"S\(port_2,port_1\)"):
        s_matrix(result)


def test_port_names_that_miss_the_highest_index_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"port 2.*port_names"):
        s_matrix(_line_column(tmp_path), port_names=["o1"])


def test_port_names_can_name_a_port_no_result_measured(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"S\(o3,o1\)"):
        s_matrix(
            _line_column(tmp_path),
            port_names=["o1", "o2", "o3"],
            assume_reciprocal=True,
        )

    matrix = s_matrix(
        _line_column(tmp_path / "again"),
        port_names=["o1", "o2", "o3"],
        assume_reciprocal=True,
        assume_unmeasured_zero=True,
    )

    assert matrix.ports == ("o1", "o2", "o3")
    np.testing.assert_array_equal(matrix.s[:, 2, :], 0)
    assert not matrix.measured[2, 0]


@pytest.mark.parametrize("name", ["S10", "S20", "S00"])
def test_fiber_source_results_are_refused(tmp_path: Path, name: str) -> None:
    result = _load_csv(tmp_path, {name: np.full(WAVELENGTH_UM.size, 0.5)})

    with pytest.raises(ValueError, match=rf"{name}.*fiber"):
        s_matrix(result, assume_reciprocal=True, assume_unmeasured_zero=True)


@pytest.mark.parametrize("name", ["S1", "S111", "s21", "S2x", "T21", "S01"])
def test_malformed_names_are_refused(name: str) -> None:
    result = SParameterResult(wavelengths=[1.55], s_params={name: [0.5 + 0j]})

    with pytest.raises(ValueError, match=rf"{name}.*S<receive><source>"):
        s_matrix(result, assume_reciprocal=True, assume_unmeasured_zero=True)


def test_preview_result_without_csv_is_refused(tmp_path: Path) -> None:
    preview = SParameterResult.from_directory(tmp_path)

    with pytest.raises(ValueError, match="no S-parameters"):
        s_matrix(preview)
    with pytest.raises(ValueError, match="no S-parameters"):
        s_matrix(preview, port_names=["o1", "o2"])


def test_csv_without_samples_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "s_parameters.csv"
    path.write_text("wavelength,S11_mag,S11_phase,S21_mag,S21_phase\n")

    with pytest.raises(ValueError, match="invalid"):
        s_matrix(SParameterResult.from_csv(path), assume_unmeasured_zero=True)


def test_no_results_are_refused() -> None:
    with pytest.raises(ValueError, match="no results"):
        s_matrix([])


def test_one_column_refuses_then_names_both_remedies(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match=r"S\(port_1,port_2\).*S\(port_2,port_2\).*assume_reciprocal.*"
        r"assume_unmeasured_zero",
    ):
        s_matrix(_line_column(tmp_path))


def test_columns_with_different_ports_are_refused(tmp_path: Path) -> None:
    zeros = np.zeros(WAVELENGTH_UM.size)
    runs = [
        _load_csv(tmp_path, {"S11": zeros, "S21": _meep_line()}),
        _load_csv(tmp_path, {"S12": _meep_line(), "S22": zeros, "S32": zeros}),
    ]

    with pytest.raises(ValueError, match="port list"):
        s_matrix(runs)


def test_columns_on_different_grids_are_refused(tmp_path: Path) -> None:
    other_um = WAVELENGTH_UM + 1e-3
    runs = [
        _line_column(tmp_path),
        _load_csv(
            tmp_path,
            {"S12": _meep_line(other_um), "S22": np.zeros(other_um.size)},
            wavelength_um=other_um,
        ),
    ]

    with pytest.raises(ValueError, match="wavelength grid"):
        s_matrix(runs)


def test_the_same_entry_in_two_results_is_refused(tmp_path: Path) -> None:
    runs = [_line_column(tmp_path), _line_column(tmp_path / "again")]

    with pytest.raises(ValueError, match=r"S\(port_1,port_1\).*more than one"):
        s_matrix(runs)


def test_meep_sign_is_causal_after_conversion_and_the_opposite_sign_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    zeros = np.zeros(WAVELENGTH_UM.size)
    flags = {"assume_reciprocal": True, "assume_unmeasured_zero": True}

    with caplog.at_level(logging.WARNING, logger="gsim.common.s_matrix"):
        s_matrix(_load_csv(tmp_path, {"S11": zeros, "S21": _meep_line()}), **flags)
    assert caplog.records == []

    backwards = np.conj(_meep_line())
    with caplog.at_level(logging.WARNING, logger="gsim.common.s_matrix"):
        s_matrix(_load_csv(tmp_path, {"S12": backwards, "S22": zeros}), **flags)
    assert "negative group delay" in caplog.text


def _plane_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if "reference planes" in record.getMessage()
    ]


def test_merged_reflections_warn_about_reference_planes_and_name_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    reflection = np.full(WAVELENGTH_UM.size, 0.1 + 0j)
    runs = [
        _load_csv(tmp_path, {"S11": reflection, "S21": _meep_line()}),
        _load_csv(tmp_path, {"S12": _meep_line(), "S22": 0 * reflection}),
    ]

    with caplog.at_level(logging.WARNING, logger="gsim.meep.models.results"):
        s_matrix(runs, port_names=["o1", "o2"])

    (message,) = _plane_warnings(caplog)
    assert "S(o1,o1)" in message
    assert "distance_source_to_monitors" in message
    # S(o2,o2) is exactly zero, so it has no phase to misplace.
    assert "S(o2,o2)" not in message


def test_reciprocal_copy_in_a_merged_matrix_warns_naming_the_copy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    zeros = np.zeros(WAVELENGTH_UM.size)
    runs = [
        _load_csv(tmp_path, {"S11": zeros, "S21": _meep_line(), "S31": zeros}),
        _load_csv(tmp_path, {"S12": _meep_line(), "S22": zeros, "S32": _meep_line()}),
    ]

    with caplog.at_level(logging.WARNING, logger="gsim.meep.models.results"):
        matrix = s_matrix(
            runs,
            port_names=["o1", "o2", "o3"],
            assume_reciprocal=True,
            assume_unmeasured_zero=True,
        )

    (message,) = _plane_warnings(caplog)
    assert not matrix.measured[1, 2]
    assert "S(o2,o3)" in message
    assert "S(o1,o3)" in message
    assert "S(o3,o3)" not in message


def test_one_run_has_one_plane_per_port_and_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    result = _load_csv(
        tmp_path, {"S11": np.full(WAVELENGTH_UM.size, 0.1), "S21": _meep_line()}
    )

    with caplog.at_level(logging.WARNING, logger="gsim.meep.models.results"):
        s_matrix(result, assume_reciprocal=True, assume_unmeasured_zero=True)

    assert _plane_warnings(caplog) == []


def test_merged_transmission_without_reflection_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    zeros = np.zeros(WAVELENGTH_UM.size)
    runs = [
        _load_csv(tmp_path, {"S11": zeros, "S21": _meep_line()}),
        _load_csv(tmp_path, {"S12": _meep_line(), "S22": zeros}),
    ]

    with caplog.at_level(logging.WARNING, logger="gsim.meep.models.results"):
        s_matrix(runs)

    assert _plane_warnings(caplog) == []


def test_builder_works_without_meep_sax_or_jax_installed() -> None:
    code = (
        "import sys\n"
        "for name in ('meep', 'sax', 'jax'):\n"
        "    sys.modules[name] = None\n"
        "from gsim.meep import SParameterResult, s_matrix\n"
        "result = SParameterResult(wavelengths=[1.55], s_params={'S11': [0.5j]})\n"
        "print(s_matrix(result).s[0, 0, 0])\n"
    )

    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert complex(completed.stdout.strip()) == -0.5j
