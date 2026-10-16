"""Check power-wave conversion with arbitrary reference impedances."""

import numpy as np
import pytest

from gsim.palace import circuit


@pytest.mark.parametrize(
    "impedance", [50.0, np.array([25.0, 75.0]), np.array([[25.0, 75.0], [50.0, 100.0]])]
)
def test_impedance_normalization_matches_explicit_power_waves(impedance):
    z = np.array([[[80, 10], [10, 90]], [[100, 20], [20, 120]]], dtype=complex)
    s = circuit._to_s_data(np.array([1e9, 2e9]), z, None, impedance)
    np.testing.assert_allclose(circuit.s_to_z(s, impedance), z)
    normalized = circuit._network_z0(impedance, 2, 2)
    np.testing.assert_allclose(
        np.broadcast_to(normalized, (2, 2)), np.broadcast_to(impedance, (2, 2))
    )


@pytest.mark.parametrize(
    ("z", "s", "z0"),
    [
        (None, None, 50),
        (None, np.ones(2), 50),
        (np.ones(2), None, np.ones(2)),
        (np.ones((2, 3)), None, 50),
    ],
)
def test_invalid_network_inputs(z, s, z0):
    with pytest.raises(ValueError):
        circuit._to_s_data(np.array([1e9, 2e9]), z, s, z0)


@pytest.mark.parametrize("z0", [np.ones(3), np.ones((2, 2, 2)), -1])
def test_invalid_reference_impedance_shapes(z0):
    with pytest.raises(ValueError):
        circuit._z0_sqrt_matrices(z0, 2, 2)
    if np.ndim(z0):
        with pytest.raises(ValueError):
            circuit._network_z0(z0, 2, 2)


def test_missing_optional_optimizer(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "optax", None)
    assert not circuit._jax_available()
