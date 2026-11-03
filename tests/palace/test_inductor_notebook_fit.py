"""The inductor tutorial's fitting settings preserve measured FEM behavior."""

from pathlib import Path

import numpy as np
import pytest

from gsim.palace import (
    differential_impedance,
    fit_rlc,
    load_circuit_synthesis,
    load_sparams,
)
from gsim.palace.circuit import VectorFit


def test_notebook_fit_order_matches_fem_after_passivity_enforcement():
    """The compact model stays passive and tracks the independent FEM reference."""
    pytest.importorskip("skrf")
    data_dir = (
        Path(__file__).resolve().parents[2] / "nbs/data/inductor/circuit_synthesis"
    )
    circuit = load_circuit_synthesis(data_dir)
    parameters = load_sparams(data_dir)
    frequency = parameters.freq * 1e9
    reference = differential_impedance(parameters.to_skrf(z0=50).z)
    fit = circuit.fit_rlc(
        frequency,
        model="vector_fit",
        n_poles_real=2,
        n_poles_cmplx=2,
        enforce_passivity=True,
    )
    assert isinstance(fit, VectorFit)
    assert fit.is_stable
    assert fit.is_passive()
    relative_error = np.abs(
        differential_impedance(fit.z(frequency)) - reference
    ) / np.abs(reference)
    assert relative_error.max() < 0.05


def test_notebook_fit_order_matches_fresh_fem_after_passivity_enforcement():
    """Current process inputs retain accuracy after global passivity enforcement."""
    pytest.importorskip("skrf")
    data = np.loadtxt(
        Path(__file__).parent / "data/inductor_fresh_fit.csv", delimiter=","
    )
    frequency = data[:, 0]
    circuit_s = (data[:, 1:9:2] + 1j * data[:, 2:9:2]).reshape(-1, 2, 2)
    reference = data[:, 9] + 1j * data[:, 10]
    fit = fit_rlc(
        frequency,
        s=circuit_s,
        model="vector_fit",
        n_poles_real=2,
        n_poles_cmplx=2,
        enforce_passivity=True,
    )
    assert isinstance(fit, VectorFit)
    assert fit.is_stable
    assert fit.is_passive()
    relative_error = np.abs(
        differential_impedance(fit.z(frequency)) - reference
    ) / np.abs(reference)
    assert relative_error.max() < 0.05
