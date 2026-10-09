"""Check the CPW de-embedding tutorial against lines with known parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import skrf as rf
from scipy.constants import speed_of_light

from gsim.palace.results import SParam, SParams

# Local workspace copy; in the repository this is parents[2] / "nbs".
NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "nbs"


@pytest.fixture
def notebook_sources():
    """Read the tutorial cells without launching cloud simulations."""
    notebook_path = NOTEBOOK_DIR / "palace_cpw_deembedding.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]]


@pytest.fixture
def extraction_functions(notebook_sources):
    """Load the actual extraction cell without launching cloud simulations."""
    source = next(
        source
        for source in notebook_sources
        if "def extract_modal_parameters(" in source
    )
    namespace = {}
    exec(compile(source, "extraction_cell", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def fit_functions(notebook_sources):
    """Load the actual line-fit cell without launching cloud simulations."""
    source = next(
        source for source in notebook_sources if "def fit_line_parameters(" in source
    )
    namespace = {}
    exec(compile(source, "fit_cell", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def transmission_line():
    """A lossy line whose 800 um phase crosses pi within the notebook sweep."""
    frequency = rf.Frequency(1, 100, 300, unit="GHz")
    impedance = 45 - 1j
    effective_index = 2.3
    gamma = 20 + 1j * frequency.w * effective_index / speed_of_light
    medium = rf.media.DefinedGammaZ0(
        frequency=frequency, z0_port=50, z0=impedance, gamma=gamma
    )
    return medium, impedance, gamma, effective_index


@pytest.mark.parametrize("length_um", [100, 300, 400, 700, 800, 1600])
def test_extracts_known_line_past_phase_wrap(
    extraction_functions, transmission_line, length_um
):
    """Recover impedance and propagation through pi and 2*pi phase crossings."""
    medium, impedance, gamma, effective_index = transmission_line
    network = medium.line(length_um, unit="um")
    parameters = extraction_functions["extract_modal_parameters"](
        network, length_m=length_um * 1e-6
    )

    np.testing.assert_allclose(parameters["zc"], impedance, rtol=1e-10)
    np.testing.assert_allclose(parameters["gamma"], gamma, rtol=1e-10)
    np.testing.assert_allclose(parameters["neff"], effective_index, rtol=1e-10)
    np.testing.assert_allclose(
        parameters["gamma"] * parameters["zc"], gamma * impedance, rtol=1e-10
    )
    np.testing.assert_allclose(
        parameters["gamma"] / parameters["zc"], gamma / impedance, rtol=1e-10
    )


@pytest.mark.parametrize("long_length_um", [400, 800])
def test_deembedding_recovers_propagation_of_added_length(
    extraction_functions, transmission_line, long_length_um
):
    """The Z-parameter split preserves propagation and reconstructs both lines."""
    medium, _, gamma, effective_index = transmission_line
    extra_length_um = long_length_um - 100
    extracted = extraction_functions["p370_extract"](
        medium.line(100, unit="um"),
        medium.line(long_length_um, unit="um"),
        delta_length_m=extra_length_um * 1e-6,
    )
    np.testing.assert_allclose(extracted["gamma"], gamma, rtol=1e-10)
    np.testing.assert_allclose(extracted["neff"], effective_index, rtol=1e-10)
    assert extracted["split_error"] < 1e-10
    assert extracted["reembed_error"] < 1e-10


@pytest.mark.parametrize("port_names", [("o1", "o2"), ("p1", "p2")])
@pytest.mark.parametrize("complete", [True, False])
def test_checks_full_matrix_with_layout_or_numeric_port_names(
    notebook_sources, transmission_line, port_names, complete
):
    """Accept cloud numeric names but reject an uncomputed output reflection."""
    medium, _, _, _ = transmission_line
    network = medium.line(100, unit="um")
    data = {
        (to_port, from_port): SParam(network.s_db[:, i, j], network.s_deg[:, i, j])
        for i, to_port in enumerate(port_names)
        for j, from_port in enumerate(port_names)
    }
    if not complete:
        del data[port_names[1], port_names[1]]
    result = SParams(network.f / 1e9, data, list(port_names))
    source = next(
        source
        for source in notebook_sources
        if source.startswith("# De-embedding requires")
    )
    case = ("lumped" if port_names[0] == "o1" else "wave", 100)
    namespace: dict[str, Any] = {"results_by_case": {case: result}}
    if not complete:
        with pytest.raises(AssertionError, match="both port excitations"):
            exec(compile(source, "network_cell", "exec"), namespace)  # noqa: S102
    else:
        exec(compile(source, "network_cell", "exec"), namespace)  # noqa: S102
        np.testing.assert_allclose(namespace["networks"][case].s, network.s, atol=1e-12)


# --- Physical RLGC from de-embedded gamma plus impedance anchors (gsim#341) ---
#
# Synthetic line with known parameters: constant L and C, skin-effect resistance
# Rs*sqrt(f/1 GHz) with the matching internal reactance, and an optional
# constant loss tangent. The 100, 400 and 800 um lines are built with launches
# (series L, shunt C) on both ends, as the notebook de-embeds them.

FREQUENCY_HZ = np.linspace(1e9, 100e9, 300)
ANCHOR_HZ = np.array([10, 25, 50, 75, 100]) * 1e9
TRUE_L = 0.30e-6
TRUE_C = 115e-12
TRUE_RS = 400.0
LAUNCH = (5e-12, 5e-15)
LENGTHS_M = (100e-6, 400e-6, 800e-6)
EXTRA_LENGTHS_M = (300e-6, 700e-6)


def _true_line(tan_delta):
    """Return gamma, Zc and G per metre of the synthetic line on FREQUENCY_HZ."""
    omega = 2 * np.pi * FREQUENCY_HZ
    skin = TRUE_RS * np.sqrt(FREQUENCY_HZ / 1e9)
    series = skin + 1j * skin + 1j * omega * TRUE_L
    shunt = omega * TRUE_C * (tan_delta + 1j)
    return np.sqrt(series * shunt), np.sqrt(series / shunt), shunt.real


def _anchors(zc, scale=1.0):
    """Return |Zc| at the anchor frequencies, as a 2D mode solve would give."""
    interpolated = np.interp(ANCHOR_HZ, FREQUENCY_HZ, zc.real) + 1j * np.interp(
        ANCHOR_HZ, FREQUENCY_HZ, zc.imag
    )
    return scale * np.abs(interpolated)


def _deembedded_sections(
    extraction_functions, gamma, zc, *, reference, noise=0.0, seed=341
):
    """Build 100, 400 and 800 um networks and de-embed the 300 and 700 um sections.

    ``reference`` is the real impedance in which S is physically defined; the
    networks are always labelled 50 ohm, as the notebook does with
    ``to_skrf(z0=50)``. reference=51 therefore mimics a wave port whose
    power-normalized S is only assigned 50 ohm.
    """
    frequency = rf.Frequency.from_f(FREQUENCY_HZ, unit="Hz")
    port = rf.media.DefinedGammaZ0(frequency=frequency, z0=reference)
    medium = rf.media.DefinedGammaZ0(
        frequency=frequency, gamma=gamma, z0=zc, z0_port=reference
    )
    launch = port.inductor(LAUNCH[0]) ** port.shunt_capacitor(LAUNCH[1])
    rng = np.random.default_rng(seed)
    networks = []
    for length in LENGTHS_M:
        s = (launch ** medium.line(length, unit="m") ** launch.flipped()).s.copy()
        if noise:
            s = s + noise * (
                rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape)
            )
            s = 0.5 * (s + np.transpose(s, (0, 2, 1)))
        networks.append(rf.Network(frequency=frequency, s=s, z0=50.0))
    thru, long_400, long_800 = networks
    return {
        length: extraction_functions["p370_extract"](thru, network, length)
        for network, length in zip((long_400, long_800), EXTRA_LENGTHS_M, strict=True)
    }


def _old_conductance(sections):
    """Return G = Re(gamma / Zc) per section, as the notebook used to plot it."""
    return {length: (r["gamma"] / r["zc"]).real for length, r in sections.items()}


def _fit(
    fit_functions, sections, anchors, *, frequency=FREQUENCY_HZ, anchor_hz=ANCHOR_HZ
):
    return fit_functions["fit_line_parameters"](
        frequency,
        [(length, r["gamma"]) for length, r in sections.items()],
        anchor_hz,
        anchors,
    )


def test_assigned_reference_reproduces_negative_conductance(extraction_functions):
    """A lossless-dielectric line shows G < 0 although gamma is extracted exactly."""
    gamma, zc, _ = _true_line(0.0)
    sections = _deembedded_sections(extraction_functions, gamma, zc, reference=51.0)

    old_g = _old_conductance(sections)
    assert min(g.min() for g in old_g.values()) < -0.05
    for result in sections.values():
        np.testing.assert_allclose(result["gamma"], gamma, rtol=1e-9)


@pytest.mark.parametrize("reference", [51.0, 50.0])
@pytest.mark.parametrize("tan_delta", [0.0, 2e-3])
def test_joint_fit_recovers_passive_line(
    extraction_functions, fit_functions, reference, tan_delta
):
    """Gamma plus |Zc| anchors recover the line, whatever the port reference."""
    gamma, zc, g_true = _true_line(tan_delta)
    sections = _deembedded_sections(
        extraction_functions, gamma, zc, reference=reference
    )
    best, fits = _fit(fit_functions, sections, _anchors(zc))

    assert best["model"] == "skin"
    assert fits["skin"]["chi2_red"] < 1e-3 * fits["sqrt_f"]["chi2_red"]
    params = best["params"]
    if tan_delta == 0.0:
        assert abs(params["tan_delta"]) < 1e-6
    else:
        assert params["tan_delta"] == pytest.approx(tan_delta, rel=1e-3)
    assert params["L"] == pytest.approx(TRUE_L, rel=1e-3)
    assert params["C"] == pytest.approx(TRUE_C, rel=1e-3)
    assert params["Rs"] == pytest.approx(TRUE_RS, rel=1e-3)
    for result in sections.values():
        rlgc = fit_functions["physical_rlgc"](FREQUENCY_HZ, result["gamma"], best)
        assert np.max(np.abs(rlgc["G"] - g_true)) < 1e-4


def test_anchor_sets_scale_only(extraction_functions, fit_functions):
    """Anchors 2 % high scale R and L up and G and C down; tan_delta is unchanged."""
    gamma, zc, _ = _true_line(2e-3)
    sections = _deembedded_sections(extraction_functions, gamma, zc, reference=50.0)
    reference_fit, _ = _fit(fit_functions, sections, _anchors(zc))
    scaled_fit, _ = _fit(fit_functions, sections, _anchors(zc, scale=1.02))

    ref, scaled = reference_fit["params"], scaled_fit["params"]
    assert scaled["L"] / ref["L"] == pytest.approx(1.02, abs=1e-6)
    assert scaled["C"] / ref["C"] == pytest.approx(1 / 1.02, abs=1e-6)
    assert scaled["tan_delta"] / ref["tan_delta"] == pytest.approx(1.0, abs=1e-6)


def test_noisy_sections(extraction_functions, fit_functions):
    """With S noise the old per-section G disagrees; the joint fit stays unbiased."""
    tan_delta = 2e-3
    gamma, zc, _ = _true_line(tan_delta)
    sections = _deembedded_sections(
        extraction_functions, gamma, zc, reference=51.0, noise=1e-4
    )
    old_g = _old_conductance(sections)
    assert abs(old_g[300e-6][-1] - old_g[700e-6][-1]) > 0.005

    best, _ = _fit(fit_functions, sections, _anchors(zc))
    fitted, sigma = best["params"]["tan_delta"], best["sigma"]["tan_delta"]
    assert fitted == pytest.approx(tan_delta, rel=0.05)
    assert abs(fitted - tan_delta) < 4 * sigma


def test_low_frequency_lumped_anchor(extraction_functions, fit_functions):
    """Lumped ports: the de-embedded |Zc| at 5-20 GHz is a good enough anchor."""
    gamma, zc, _ = _true_line(0.0)
    sections = _deembedded_sections(extraction_functions, gamma, zc, reference=50.0)
    band = (FREQUENCY_HZ >= 5e9) & (FREQUENCY_HZ <= 20e9)
    anchors = np.mean([np.abs(r["zc"][band]) for r in sections.values()], axis=0)
    best, _ = _fit(fit_functions, sections, anchors, anchor_hz=FREQUENCY_HZ[band])

    assert best["params"]["L"] == pytest.approx(TRUE_L, rel=0.01)
    assert abs(best["params"]["tan_delta"]) < 5e-5
