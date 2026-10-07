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
