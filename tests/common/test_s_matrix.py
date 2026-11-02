"""Tests for the solver-independent S-matrix export (ADR 0013).

Numerical expectations come from closed forms or independent libraries
(``sax.models.straight``, ``skrf.Network``), never from this module's own output.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gsim.common.s_matrix import SMatrix, SMatrixRun, assemble_s_matrix

PORTS = ("o1", "o2")
WAVELENGTH_UM = np.linspace(1.5, 1.6, 51)
OFF_GRID_UM = np.array([1.5013, 1.5177, 1.5333, 1.5491, 1.5719, 1.5987])

# Parameters of the dispersive line that sax.models.straight also evaluates.
WL0_UM = 1.55
NEFF = 2.34
NG = 3.4
LENGTH_UM = 100.0


def _line_phase(wavelength_um: np.ndarray, length_um: float = LENGTH_UM) -> np.ndarray:
    """Return beta * L of the first-order dispersive line, in radians."""
    neff = NEFF - (wavelength_um - WL0_UM) * (NG - NEFF) / WL0_UM
    return 2 * np.pi * neff * length_um / wavelength_um


def _delay_line_runs(wavelength_um: np.ndarray = WAVELENGTH_UM) -> list[SMatrixRun]:
    """Return one run per port of a matched delay line, stored as exp(+jwt)."""
    through = np.exp(-1j * _line_phase(wavelength_um))
    reflection = np.zeros_like(through)
    return [
        SMatrixRun(
            label="run_o1",
            wavelength_um=wavelength_um,
            ports=PORTS,
            entries={("o1", "o1"): reflection, ("o2", "o1"): through},
            valid=np.ones(wavelength_um.size, dtype=bool),
        ),
        SMatrixRun(
            label="run_o2",
            wavelength_um=wavelength_um,
            ports=PORTS,
            entries={("o2", "o2"): reflection, ("o1", "o2"): through},
            valid=np.ones(wavelength_um.size, dtype=bool),
        ),
    ]


def test_delay_line_model_matches_sax_straight_between_samples() -> None:
    sax = pytest.importorskip("sax")
    matrix = assemble_s_matrix(_delay_line_runs())

    actual = matrix.sax_model()(wl=OFF_GRID_UM)
    reference = sax.models.straight(
        wl=OFF_GRID_UM, wl0=WL0_UM, neff=NEFF, ng=NG, length=LENGTH_UM
    )
    named = sax.PortNamer(1, 1)

    np.testing.assert_array_less(
        np.abs(actual["o1", "o2"] - np.asarray(reference[named.in0, named.out0])), 1e-3
    )
    np.testing.assert_array_less(
        np.abs(actual["o2", "o1"] - np.asarray(reference[named.out0, named.in0])), 1e-3
    )


def test_model_keeps_magnitude_between_samples_of_a_fast_phase() -> None:
    """Re/im interpolation would collapse |S| between samples ~1.8 rad apart."""
    matrix = assemble_s_matrix(_delay_line_runs())

    through = matrix.sax_model()(wl=OFF_GRID_UM)["o1", "o2"]

    np.testing.assert_allclose(np.abs(through), 1.0, atol=1e-12)


def test_circuit_cascade_with_sax_straight_equals_analytic_total_phase() -> None:
    sax = pytest.importorskip("sax")
    model = assemble_s_matrix(_delay_line_runs()).sax_model()
    named = sax.PortNamer(1, 1)
    circuit, _ = sax.circuit(
        {
            "instances": {"dut": "dut", "tail": "tail"},
            "connections": {"dut,o2": f"tail,{named.in0}"},
            "ports": {"in": "dut,o1", "out": f"tail,{named.out0}"},
        },
        models={"dut": model, "tail": sax.models.straight},
    )
    tail_um = 37.0

    actual = circuit(
        wl=OFF_GRID_UM,
        tail={"wl0": WL0_UM, "neff": NEFF, "ng": NG, "length": tail_um},
    )["in", "out"]

    expected = np.exp(
        1j * (_line_phase(OFF_GRID_UM) + _line_phase(OFF_GRID_UM, tail_um))
    )
    np.testing.assert_array_less(np.abs(np.asarray(actual) - expected), 1e-3)


def test_sax_key_order_is_in_out_on_a_non_reciprocal_matrix() -> None:
    forward = 0.9 * np.exp(-1j * _line_phase(WAVELENGTH_UM))
    backward = 0.1 * np.exp(-1j * _line_phase(WAVELENGTH_UM, 50.0))
    zeros = np.zeros_like(forward)
    valid = np.ones(WAVELENGTH_UM.size, dtype=bool)
    matrix = assemble_s_matrix(
        [
            SMatrixRun(
                "run_o1",
                WAVELENGTH_UM,
                PORTS,
                {("o1", "o1"): zeros, ("o2", "o1"): forward},
                valid,
            ),
            SMatrixRun(
                "run_o2",
                WAVELENGTH_UM,
                PORTS,
                {("o2", "o2"): zeros, ("o1", "o2"): backward},
                valid,
            ),
        ]
    )

    model = matrix.sax_model()(wl=WAVELENGTH_UM)

    # ("o1", "o2") reads "in at o1, out at o2", which gsim names S(o2,o1).
    np.testing.assert_allclose(model["o1", "o2"], np.conj(forward), atol=1e-12)
    np.testing.assert_allclose(model["o2", "o1"], np.conj(backward), atol=1e-12)
    assert set(model) == {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}


def test_matrix_stores_receive_source_order_and_ascending_wavelength() -> None:
    matrix = assemble_s_matrix(_delay_line_runs(WAVELENGTH_UM[::-1]))

    np.testing.assert_allclose(matrix.wavelength_um, WAVELENGTH_UM)
    np.testing.assert_allclose(
        matrix.frequency_hz, 299792458.0 / (WAVELENGTH_UM * 1e-6)
    )
    np.testing.assert_allclose(
        matrix.s[:, 1, 0], np.exp(-1j * _line_phase(WAVELENGTH_UM)), atol=1e-12
    )
    assert matrix.ports == PORTS
    assert matrix.measured.tolist() == [[True, True], [True, True]]


def test_scalar_wl_gives_0d_and_array_wl_keeps_shape() -> None:
    model = assemble_s_matrix(_delay_line_runs()).sax_model()

    assert np.shape(model(wl=1.55)["o1", "o2"]) == ()
    assert np.shape(model()["o1", "o2"]) == ()
    assert model(wl=OFF_GRID_UM.reshape(2, 3))["o1", "o2"].shape == (2, 3)


def test_default_wl_is_band_midpoint_and_visible_to_sax_settings() -> None:
    sax = pytest.importorskip("sax")
    model = assemble_s_matrix(_delay_line_runs()).sax_model()

    assert sax.get_settings(model)["wl"] == pytest.approx(1.55)
    np.testing.assert_allclose(
        model()["o1", "o2"], np.exp(1j * _line_phase(np.array(1.55))), atol=1e-3
    )


@pytest.mark.parametrize("wl", [1.4999, 1.6001, [1.55, 1.7], float("nan")])
def test_wavelength_outside_the_band_is_refused(wl: object) -> None:
    model = assemble_s_matrix(_delay_line_runs()).sax_model()

    with pytest.raises(ValueError, match=r"1\.5.*1\.6"):
        model(wl=wl)


def test_band_edges_accept_a_rounding_error_and_reject_beyond() -> None:
    model = assemble_s_matrix(_delay_line_runs()).sax_model()

    edge = np.exp(1j * _line_phase(np.array(1.6)))
    np.testing.assert_allclose(model(wl=1.6 + 5e-10)["o1", "o2"], edge, atol=1e-6)
    with pytest.raises(ValueError):
        model(wl=1.6 + 5e-9)


def test_model_does_not_follow_later_changes_to_the_matrix_arrays() -> None:
    matrix = assemble_s_matrix(_delay_line_runs())
    model = matrix.sax_model()
    before = model(wl=1.55)["o1", "o2"]

    matrix.s[:] = 0.0
    matrix.wavelength_um[:] = 1.0

    np.testing.assert_allclose(model(wl=1.55)["o1", "o2"], before)


def _non_reciprocal_matrix() -> tuple[SMatrix, dict[tuple[str, str], np.ndarray]]:
    """Return a 2-port whose four entries are all different, stored as exp(+jwt)."""
    wavelength_um = np.array([1.6, 1.5, 1.55])
    table = {
        ("o1", "o1"): np.array([0.1 + 0.2j, 0.3 - 0.1j, 0.2 + 0.0j]),
        ("o2", "o1"): np.array([0.7 + 0.1j, 0.6 - 0.2j, 0.5 + 0.4j]),
        ("o1", "o2"): np.array([0.05 - 0.01j, 0.04 + 0.02j, 0.03 + 0.03j]),
        ("o2", "o2"): np.array([-0.2 + 0.1j, -0.1 + 0.3j, -0.3 - 0.2j]),
    }
    run = SMatrixRun("run", wavelength_um, PORTS, table, np.ones(3, dtype=bool))
    return assemble_s_matrix([run]), table


def test_touchstone_read_by_skrf_has_ascending_f_values_names_and_z0(
    tmp_path: Path,
) -> None:
    skrf = pytest.importorskip("skrf")
    matrix, table = _non_reciprocal_matrix()

    written = matrix.write_touchstone(tmp_path / "device", z0=75.0)

    assert written == tmp_path / "device.s2p"
    network = skrf.Network(str(written))
    # table holds the samples for wavelengths [1.6, 1.5, 1.55] um; ascending
    # frequency is wavelengths 1.6, 1.55, 1.5, that is indices 0, 2, 1.
    ascending_f = [0, 2, 1]
    np.testing.assert_allclose(
        network.f, 299792458.0 / (np.array([1.6, 1.55, 1.5]) * 1e-6), rtol=1e-12
    )
    assert np.all(np.diff(network.f) > 0)
    for (receive, source), (i, j) in {
        ("o1", "o1"): (0, 0),
        ("o2", "o1"): (1, 0),
        ("o1", "o2"): (0, 1),
        ("o2", "o2"): (1, 1),
    }.items():
        np.testing.assert_allclose(
            network.s[:, i, j], table[receive, source][ascending_f], atol=1e-12
        )
    assert list(network.port_names) == list(PORTS)
    np.testing.assert_allclose(network.z0, 75.0)


def test_touchstone_z0_defaults_to_50_ohm(tmp_path: Path) -> None:
    skrf = pytest.importorskip("skrf")
    matrix, _ = _non_reciprocal_matrix()

    network = skrf.Network(str(matrix.write_touchstone(tmp_path / "device")))

    np.testing.assert_allclose(network.z0, 50.0)


@pytest.mark.parametrize(
    "z0", [0, 0.0, -50.0, 50 + 1j, float("inf"), float("-inf"), float("nan")]
)
def test_touchstone_refuses_z0_that_is_not_positive_finite_real(
    tmp_path: Path, z0: complex
) -> None:
    matrix, _ = _non_reciprocal_matrix()

    with pytest.raises(ValueError, match="reference impedance"):
        matrix.write_touchstone(tmp_path / "device", z0=z0)
    assert list(tmp_path.iterdir()) == []


def test_touchstone_accepts_z0_with_zero_imaginary_part(tmp_path: Path) -> None:
    skrf = pytest.importorskip("skrf")
    matrix, _ = _non_reciprocal_matrix()

    network = skrf.Network(
        str(matrix.write_touchstone(tmp_path / "device", z0=75 + 0j))
    )

    np.testing.assert_allclose(network.z0, 75.0)


def test_touchstone_three_ports_keep_matrix_order(tmp_path: Path) -> None:
    skrf = pytest.importorskip("skrf")
    ports = ("a", "b", "c")
    wavelength_um = np.array([1.5, 1.6])
    table = {
        (receive, source): np.array([1.0, 2.0]) * (0.01 * (10 * i + j + 11) + 0.02j)
        for i, receive in enumerate(ports)
        for j, source in enumerate(ports)
    }
    run = SMatrixRun("run", wavelength_um, ports, table, np.ones(2, dtype=bool))

    written = assemble_s_matrix([run]).write_touchstone(tmp_path / "device")

    assert written.name == "device.s3p"
    network = skrf.Network(str(written))
    assert list(network.port_names) == list(ports)
    for i, receive in enumerate(ports):
        for j, source in enumerate(ports):
            np.testing.assert_allclose(
                network.s[:, i, j], table[receive, source][::-1], atol=1e-12
            )


def test_touchstone_appends_suffix_without_eating_a_dotted_name(
    tmp_path: Path,
) -> None:
    matrix, _ = _non_reciprocal_matrix()

    written = matrix.write_touchstone(tmp_path / "run_1.5")

    assert written == tmp_path / "run_1.5.s2p"
    assert written.is_file()


@pytest.mark.parametrize("name", ["device.s2p", "device.S2P"])
def test_touchstone_keeps_a_matching_suffix(tmp_path: Path, name: str) -> None:
    matrix, _ = _non_reciprocal_matrix()

    written = matrix.write_touchstone(str(tmp_path / name))

    assert written == tmp_path / name
    assert written.is_file()
    assert [path.name for path in tmp_path.iterdir()] == [name]


@pytest.mark.parametrize("name", ["device.s4p", "device.S1P"])
def test_touchstone_refuses_a_suffix_with_the_wrong_port_count(
    tmp_path: Path, name: str
) -> None:
    matrix, _ = _non_reciprocal_matrix()

    with pytest.raises(ValueError, match=r"2 ports"):
        matrix.write_touchstone(tmp_path / name)
    assert not (tmp_path / name).exists()


GRID_UM = np.array([1.5, 1.55, 1.6])
FORWARD = np.array([0.8 + 0.1j, 0.7 - 0.3j, 0.1 - 0.75j])
REFLECTION = np.array([0.02j, 0.03 + 0j, -0.01 + 0.01j])


def _single_run(
    entries: dict[tuple[str, str], np.ndarray] | None = None, **overrides: object
) -> SMatrixRun:
    """Return one source-at-o1 run, overridable field by field."""
    fields: dict = {
        "label": "run_o1",
        "wavelength_um": GRID_UM,
        "ports": PORTS,
        "entries": (
            {("o1", "o1"): REFLECTION, ("o2", "o1"): FORWARD}
            if entries is None
            else entries
        ),
        "valid": np.ones(GRID_UM.size, dtype=bool),
    }
    fields.update(overrides)
    return SMatrixRun(**fields)


def test_one_run_without_flags_lists_missing_entries_and_both_remedies() -> None:
    with pytest.raises(ValueError) as error:
        assemble_s_matrix([_single_run()])

    message = str(error.value)
    assert "S(o1,o2)" in message
    assert "S(o2,o2)" in message
    assert "S(o2,o1)" not in message
    assert "assume_reciprocal" in message
    assert "assume_unmeasured_zero" in message
    assert "source" in message


def test_reciprocal_flag_copies_only_the_mirrored_entry() -> None:
    with pytest.raises(ValueError, match=r"S\(o2,o2\)") as error:
        assemble_s_matrix([_single_run()], assume_reciprocal=True)

    assert "S(o1,o2)" not in str(error.value)


def test_zero_flag_fills_everything_missing_with_zero() -> None:
    matrix = assemble_s_matrix([_single_run()], assume_unmeasured_zero=True)

    np.testing.assert_array_equal(matrix.s[:, 0, 1], np.zeros(3))
    np.testing.assert_array_equal(matrix.s[:, 1, 1], np.zeros(3))
    np.testing.assert_array_equal(matrix.s[:, 1, 0], FORWARD)
    assert matrix.measured.tolist() == [[True, False], [True, False]]


def test_both_flags_copy_first_and_mark_every_filled_entry() -> None:
    matrix = assemble_s_matrix(
        [_single_run()], assume_reciprocal=True, assume_unmeasured_zero=True
    )

    np.testing.assert_array_equal(matrix.s[:, 0, 1], FORWARD)
    np.testing.assert_array_equal(matrix.s[:, 1, 1], np.zeros(3))
    assert matrix.measured.tolist() == [[True, False], [True, False]]


def test_reciprocal_flag_keeps_a_measured_entry_that_differs() -> None:
    backward = 0.5 * FORWARD
    runs = [
        _single_run(),
        _single_run({("o2", "o2"): REFLECTION, ("o1", "o2"): backward}, label="run_o2"),
    ]

    matrix = assemble_s_matrix(runs, assume_reciprocal=True)

    np.testing.assert_array_equal(matrix.s[:, 0, 1], backward)
    assert matrix.measured.all()


def test_assumed_entries_are_logged_at_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="gsim.common.s_matrix"):
        assemble_s_matrix(
            [_single_run()], assume_reciprocal=True, assume_unmeasured_zero=True
        )

    infos = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert any("S(o1,o2)" in message and "S(o2,o1)" in message for message in infos)
    assert any("S(o2,o2)" in message for message in infos)


def test_no_runs_refused() -> None:
    with pytest.raises(ValueError, match="no results"):
        assemble_s_matrix([])


def test_run_without_s_parameters_refused_naming_it() -> None:
    with pytest.raises(ValueError, match=r"run_empty.*no S-parameters"):
        assemble_s_matrix([_single_run({}, label="run_empty")])


def test_different_port_lists_refused() -> None:
    other = _single_run({("o2", "o2"): REFLECTION}, label="b", ports=("o1", "o3"))

    with pytest.raises(ValueError, match="port"):
        assemble_s_matrix([_single_run(), other])


def test_reordered_ports_in_a_later_run_give_the_in_order_matrix() -> None:
    backward = 0.5 * FORWARD
    entries = {("o2", "o2"): REFLECTION, ("o1", "o2"): backward}
    in_order = assemble_s_matrix(
        [_single_run(), _single_run(entries, label="b", ports=("o1", "o2"))]
    )

    reordered = assemble_s_matrix(
        [_single_run(), _single_run(entries, label="b", ports=("o2", "o1"))]
    )

    assert reordered.ports == PORTS
    np.testing.assert_array_equal(reordered.s, in_order.s)
    np.testing.assert_array_equal(reordered.measured, in_order.measured)
    np.testing.assert_array_equal(reordered.s[:, 1, 0], FORWARD)
    np.testing.assert_array_equal(reordered.s[:, 0, 1], backward)


def test_matrix_keeps_the_port_order_of_the_first_run() -> None:
    backward = 0.5 * FORWARD
    runs = [
        _single_run(ports=("o2", "o1")),
        _single_run({("o2", "o2"): REFLECTION, ("o1", "o2"): backward}, label="b"),
    ]

    matrix = assemble_s_matrix(runs)

    assert matrix.ports == ("o2", "o1")
    np.testing.assert_array_equal(matrix.s[:, 0, 1], FORWARD)  # S(o2,o1)
    np.testing.assert_array_equal(matrix.s[:, 1, 0], backward)  # S(o1,o2)


def test_later_run_repeating_a_port_refused() -> None:
    other = _single_run({("o2", "o2"): REFLECTION}, label="b", ports=("o1", "o2", "o2"))

    with pytest.raises(ValueError, match="duplicate"):
        assemble_s_matrix([_single_run(), other], assume_unmeasured_zero=True)


def test_different_wavelength_grids_refused() -> None:
    other = _single_run(
        {("o2", "o2"): REFLECTION},
        label="b",
        wavelength_um=GRID_UM * (1 + 1e-6),
    )

    with pytest.raises(ValueError, match="wavelength"):
        assemble_s_matrix([_single_run(), other], assume_unmeasured_zero=True)


def test_different_grid_sizes_refused() -> None:
    other = _single_run(
        {("o2", "o2"): np.ones(2, dtype=complex)},
        label="b",
        wavelength_um=GRID_UM[:2],
        valid=np.ones(2, dtype=bool),
    )

    with pytest.raises(ValueError, match="wavelength"):
        assemble_s_matrix([_single_run(), other], assume_unmeasured_zero=True)


def test_grids_equal_after_sorting_within_tolerance_align_by_wavelength() -> None:
    backward = np.array([0.1 + 0j, 0.2 + 0j, 0.3 + 0j])
    other = _single_run(
        {("o2", "o2"): REFLECTION[::-1], ("o1", "o2"): backward[::-1]},
        label="b",
        wavelength_um=GRID_UM[::-1] * (1 + 1e-10),
    )

    matrix = assemble_s_matrix([_single_run(), other])

    np.testing.assert_array_equal(matrix.s[:, 1, 0], FORWARD)
    np.testing.assert_array_equal(matrix.s[:, 0, 1], backward)


def test_same_entry_in_two_runs_refused_naming_it() -> None:
    other = _single_run({("o2", "o1"): FORWARD}, label="b")

    with pytest.raises(ValueError, match=r"S\(o2,o1\)"):
        assemble_s_matrix([_single_run(), other], assume_unmeasured_zero=True)


@pytest.mark.parametrize("key", [("o9", "o1"), ("o1", "o9")])
def test_key_naming_an_unknown_port_refused(key: tuple[str, str]) -> None:
    with pytest.raises(ValueError, match="o9"):
        assemble_s_matrix([_single_run({key: FORWARD})], assume_unmeasured_zero=True)


def test_entry_of_the_wrong_length_refused() -> None:
    with pytest.raises(ValueError, match=r"S\(o2,o1\)"):
        assemble_s_matrix(
            [_single_run({("o2", "o1"): FORWARD[:2]})], assume_unmeasured_zero=True
        )


def test_duplicate_port_names_refused() -> None:
    run = _single_run({("o1", "o1"): FORWARD}, ports=("o1", "o1"))

    with pytest.raises(ValueError, match="duplicate"):
        assemble_s_matrix([run], assume_unmeasured_zero=True)


def test_repeated_wavelength_refused() -> None:
    """Interpolation over a repeated abscissa has no defined answer."""
    with pytest.raises(ValueError, match="ascending"):
        assemble_s_matrix(
            [_single_run(wavelength_um=np.array([1.5, 1.5, 1.6]))],
            assume_unmeasured_zero=True,
        )


def test_flagged_and_non_finite_samples_are_dropped_and_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wavelength_um = np.linspace(1.5, 1.6, 6)
    through = np.exp(-1j * _line_phase(wavelength_um))
    with_nan = through.copy()
    with_nan[2] = np.nan
    run_o1 = _single_run(
        {("o2", "o1"): with_nan},
        wavelength_um=wavelength_um,
        valid=np.array([True, False, True, True, True, True]),
    )
    run_o2 = _single_run(
        {("o1", "o2"): through},
        label="b",
        wavelength_um=wavelength_um,
        valid=np.array([True, True, True, True, True, False]),
    )

    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        matrix = assemble_s_matrix([run_o1, run_o2], assume_unmeasured_zero=True)

    np.testing.assert_allclose(matrix.wavelength_um, wavelength_um[[0, 3, 4]])
    np.testing.assert_array_equal(matrix.s[:, 1, 0], through[[0, 3, 4]])
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("3 of 6" in message for message in warnings)


def test_no_warning_when_every_sample_is_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        assemble_s_matrix(_delay_line_runs())

    assert caplog.records == []


@pytest.mark.parametrize(
    ("values", "valid"),
    [
        (np.full(3, np.nan, dtype=complex), np.ones(3, dtype=bool)),
        (FORWARD, np.zeros(3, dtype=bool)),
    ],
    ids=["non-finite", "flagged"],
)
def test_all_samples_invalid_refused(values: np.ndarray, valid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="invalid"):
        assemble_s_matrix(
            [_single_run({("o2", "o1"): values}, valid=valid)],
            assume_unmeasured_zero=True,
        )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def _transmission_run(entries: dict[tuple[str, str], np.ndarray]) -> SMatrixRun:
    wavelength_um = next(iter(entries.values())).size
    return _single_run(
        entries,
        wavelength_um=np.linspace(1.5, 1.6, wavelength_um),
        valid=np.ones(wavelength_um, dtype=bool),
    )


def test_wrong_sign_transmission_warns_naming_entry_and_both_causes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wrong_sign = np.exp(+1j * _line_phase(WAVELENGTH_UM))

    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        assemble_s_matrix(
            [_transmission_run({("o2", "o1"): wrong_sign})],
            assume_unmeasured_zero=True,
        )

    (message,) = _warnings(caplog)
    assert "S(o2,o1)" in message
    assert "convention" in message
    assert "coarse" in message


def test_physical_delay_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    through = np.exp(-1j * _line_phase(WAVELENGTH_UM))

    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        assemble_s_matrix(
            [_transmission_run({("o2", "o1"): through})],
            assume_unmeasured_zero=True,
        )

    assert _warnings(caplog) == []


def test_fewer_than_three_samples_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wavelength_um = np.linspace(1.5, 1.6, 2)
    wrong_sign = np.exp(+1j * _line_phase(wavelength_um, 1.0))

    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        assemble_s_matrix(
            [_transmission_run({("o2", "o1"): wrong_sign})],
            assume_unmeasured_zero=True,
        )

    assert _warnings(caplog) == []


def test_sample_count_is_taken_after_dropping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wrong_sign = np.exp(+1j * _line_phase(GRID_UM, 1.0))
    run = _single_run({("o2", "o1"): wrong_sign}, valid=np.array([True, False, True]))

    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        assemble_s_matrix([run], assume_unmeasured_zero=True)

    (message,) = _warnings(caplog)
    assert "Dropped" in message


def test_no_measured_off_diagonal_entry_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wrong_sign = np.exp(+1j * _line_phase(WAVELENGTH_UM))

    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        matrix = assemble_s_matrix(
            [_transmission_run({("o1", "o1"): wrong_sign})],
            assume_unmeasured_zero=True,
        )

    assert _warnings(caplog) == []
    assert matrix.measured.tolist() == [[True, False], [False, False]]


def test_check_reads_the_strongest_entry_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    strong = np.exp(-1j * _line_phase(WAVELENGTH_UM))
    weak_wrong_sign = 0.01 * np.exp(+1j * _line_phase(WAVELENGTH_UM))
    runs = [
        _transmission_run({("o2", "o1"): strong}),
        _transmission_run({("o1", "o2"): weak_wrong_sign}),
    ]

    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        assemble_s_matrix(runs, assume_unmeasured_zero=True)
    assert _warnings(caplog) == []

    caplog.clear()
    runs = [
        _transmission_run({("o2", "o1"): weak_wrong_sign}),
        _transmission_run({("o1", "o2"): -1j * np.conj(strong)}),
    ]
    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        assemble_s_matrix(runs, assume_unmeasured_zero=True)
    (message,) = _warnings(caplog)
    assert "S(o1,o2)" in message


def test_all_zero_transmission_gives_no_delay_to_judge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="gsim.common.s_matrix"):
        assemble_s_matrix(
            [_transmission_run({("o2", "o1"): np.zeros(5, dtype=complex)})],
            assume_unmeasured_zero=True,
        )

    assert _warnings(caplog) == []


def test_importing_the_module_loads_neither_sax_nor_jax_nor_skrf() -> None:
    code = (
        "import sys; import gsim.common.s_matrix; "
        "print([m for m in ('sax', 'jax', 'skrf') if m in sys.modules])"
    )

    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "[]"


def test_one_surviving_sample_still_gives_a_model_and_a_file(tmp_path: Path) -> None:
    skrf = pytest.importorskip("skrf")
    value = np.array([0.6 - 0.2j, np.nan, np.nan], dtype=complex)
    matrix = assemble_s_matrix(
        [_single_run({("o2", "o1"): value})], assume_unmeasured_zero=True
    )

    model = matrix.sax_model()
    network = skrf.Network(str(matrix.write_touchstone(tmp_path / "device")))

    np.testing.assert_allclose(model()["o1", "o2"], 0.6 + 0.2j, atol=1e-12)
    np.testing.assert_allclose(network.s[:, 1, 0], [0.6 - 0.2j], atol=1e-12)
    with pytest.raises(ValueError):
        model(wl=1.5001)


def test_one_port_device_exports_to_both_targets(tmp_path: Path) -> None:
    skrf = pytest.importorskip("skrf")
    run = _single_run({("o1", "o1"): REFLECTION}, ports=("o1",))

    matrix = assemble_s_matrix([run])
    network = skrf.Network(str(matrix.write_touchstone(tmp_path / "device")))

    assert set(matrix.sax_model()()) == {("o1", "o1")}
    assert network.nports == 1
    np.testing.assert_allclose(network.s[:, 0, 0], REFLECTION[::-1], atol=1e-12)
