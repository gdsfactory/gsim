"""Tests for the solver-independent S-matrix export (ADR 0013).

Numerical expectations come from closed forms or independent libraries
(``sax.models.straight``, ``skrf.Network``), never from this module's own output.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.s_matrix import SMatrixRun, assemble_s_matrix

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
