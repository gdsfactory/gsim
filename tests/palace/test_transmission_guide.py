"""Execute the scikit-rf guide and check its Palace/data conventions."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gsim.palace import load_sparams


@pytest.fixture(scope="module")
def guide():
    """Run the actual documentation, including its independent RLGC checks."""
    pytest.importorskip("skrf", minversion="2.1")
    path = Path(__file__).parents[2] / "docs" / "transmission_line_analysis.md"
    source = path.read_text()
    blocks = list(re.finditer(r"```python\n(.*?)```", source, re.DOTALL))
    assert len(blocks) == 5
    namespace = {}
    for match in blocks:
        padding = "\n" * source[: match.start(1)].count("\n")
        exec(compile(padding + match[1], str(path), "exec"), namespace)  # noqa: S102
    return namespace


def test_executable_guide(guide):
    """Both upstream calibrators recover independently generated propagation."""
    for calibration in (guide["calibration"], guide["tug"]):
        np.testing.assert_allclose(calibration.gamma, guide["gamma_exact"], rtol=1e-9)
    np.testing.assert_allclose(
        guide["calibration"].z0[-1], 70.0042 - 0.7646j, atol=1e-4
    )


def test_palace_csv_to_calibrated_dut(guide, tmp_path):
    """Exercise GHz/dB/degrees ingestion and both excitation columns."""
    measured = guide["measured_dut"]
    columns = {"f (GHz)": measured.f / 1e9}
    for i in range(2):
        for j in range(2):
            entry = measured.s[:, i, j]
            columns[f"|S[{i + 1}][{j + 1}]| (dB)"] = 20 * np.log10(abs(entry))
            columns[f"arg(S[{i + 1}][{j + 1}]) (deg.)"] = np.angle(entry, deg=True)
    pd.DataFrame(columns).to_csv(tmp_path / "port-S.csv", index=False)
    converted = load_sparams(tmp_path).to_skrf(z0=50)
    np.testing.assert_allclose(converted.f, measured.f)
    np.testing.assert_allclose(converted.s, measured.s, atol=1e-14)
    np.testing.assert_allclose(
        guide["calibration"].apply_cal(converted).s,
        guide["line"](700e-6).s,
        atol=1e-9,
    )


@pytest.mark.parametrize("shifts_um", [(20, 80), (-30, 90), (0, 0)])
def test_shifted_mismatched_dut(guide, shifts_um):
    """Reference shifts preserve a held-out discontinuity and reciprocity."""
    line = guide["line"]
    # Neither this resistor nor these lengths belong to the standards.
    resistor = guide["ports"].resistor(37)
    dut = line(350e-6) ** resistor ** line(550e-6)
    measured = guide["left"] ** dut ** guide["right"]
    corrected = guide["calibration"].apply_cal(measured)
    shifts = np.asarray(shifts_um) * 1e-6
    medium = guide["extracted_medium"]
    shifted = corrected
    if shifts[0] != 0:
        shifted = medium.line(-shifts[0], unit="m") ** shifted
    if shifts[1] != 0:
        shifted = shifted ** medium.line(-shifts[1], unit="m")
    expected = line(350e-6 - shifts[0]) ** resistor ** line(550e-6 - shifts[1])
    np.testing.assert_allclose(shifted.s, expected.s, atol=1e-9)
    np.testing.assert_allclose(shifted.s[:, 1, 0], shifted.s[:, 0, 1], atol=1e-9)
    if np.any(shifts):
        assert np.max(abs(shifted.s - dut.s)) > 0.01


def test_phase_conditioning(guide):
    """Distinguish singular pairs from a useful additional line."""
    separation = guide["phase_separation_deg"]
    np.testing.assert_allclose(separation([np.pi], [0, 1, 2]), 0, atol=1e-12)
    np.testing.assert_allclose(separation([np.pi], [0, 0.5, 1]), 90)
    np.testing.assert_allclose(separation([np.pi / 18], [0, 1]), 10)


def test_perfectly_matched_standards(guide):
    """TUG handles the exact zero-reflection case without a custom rejection."""
    rf = guide["rf"]
    gamma = 1j * guide["omega"] * guide["index_estimate"] / guide["c"]
    medium = rf.media.DefinedGammaZ0(guide["frequency"], gamma=gamma, z0=50)
    lengths = guide["lengths_m"]
    lines = [medium.line(length, unit="m") for length in lengths]
    termination = medium.delay_short(50e-6, unit="m")
    reflect = guide["two_port_reflect"](termination, termination)
    calibration = guide["TUGMultilineTRL"](
        line_meas=lines,
        line_lengths=lengths.tolist(),
        reflect_meas=[reflect],
        reflect_est=[-1],
        er_est=guide["index_estimate"] ** 2,
        reflect_offset=50e-6,
        ref_plane=0,
        switch_terms=(guide["zero_switch"], guide["zero_switch"].copy()),
    )
    calibration.run()
    # The independently known line impedance already equals the 50-ohm ports.
    held_out = medium.line(700e-6, unit="m")
    np.testing.assert_allclose(calibration.gamma, gamma, rtol=1e-9)
    np.testing.assert_allclose(calibration.apply_cal(held_out).s, held_out.s, atol=1e-9)


def test_capacitance_does_not_determine_impedance_with_shunt_loss(guide):
    """c0's G=0 assumption is material; independent Zc handles nonzero G."""
    rf = guide["rf"]
    omega = guide["omega"]
    capacitance = guide["capacitance_per_m"]
    series = guide["resistance_per_m"] + 1j * omega * guide["inductance_per_m"]
    shunt = 0.5 + 1j * omega * capacitance
    gamma = np.sqrt(series * shunt)
    impedance = np.sqrt(series / shunt)
    medium = rf.media.DefinedGammaZ0(
        guide["frequency"], gamma=gamma, z0=impedance, z0_port=50
    )
    left, right = guide["left"], guide["right"]
    measured_lines = [
        left ** medium.line(length, unit="m") ** right for length in guide["lengths_m"]
    ]
    termination = medium.line(50e-6, unit="m") ** guide["ports"].short()
    reflect = guide["two_port_reflect"](
        left**termination, right.flipped() ** termination
    )
    kwargs = {
        "measured": [measured_lines[0], reflect, *measured_lines[1:]],
        "Grefls": [-1],
        "l": guide["lengths_m"].tolist(),
        "er_est": guide["index_estimate"] ** 2,
        "gamma_root_choice": "estimate",
        "refl_offset": 50e-6,
        "ref_plane": 0,
        "switch_terms": (guide["zero_switch"], guide["zero_switch"].copy()),
        "z0_ref": 50,
    }
    assumed = guide["NISTMultilineTRL"](**kwargs, c0=capacitance)
    physical = guide["NISTMultilineTRL"](**kwargs, z0_line=impedance)
    assumed.run()
    physical.run()
    np.testing.assert_allclose(assumed.gamma, gamma, rtol=1e-9)
    assert np.max(abs(assumed.z0 / impedance - 1)) > 0.1
    held_out = medium.line(700e-6, unit="m")
    measured = left**held_out**right
    np.testing.assert_allclose(physical.apply_cal(measured).s, held_out.s, atol=1e-9)
    assert np.max(abs(assumed.apply_cal(measured).s - held_out.s)) > 0.01
