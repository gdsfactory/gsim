"""Independent telegrapher networks and unequal launches for RF tests."""

from dataclasses import dataclass

import numpy as np
import pytest


def line_matrix(gamma, length_m, impedance_ratio=1):
    argument = gamma * length_m
    matrix = np.zeros((len(gamma), 2, 2), dtype=complex)
    matrix[:, 0, 0] = matrix[:, 1, 1] = np.cosh(argument)
    matrix[:, 0, 1] = impedance_ratio * np.sinh(argument)
    matrix[:, 1, 0] = np.sinh(argument) / impedance_ratio
    return matrix


def lumped_matrix(series, shunt):
    series, shunt = np.broadcast_arrays(series, shunt)
    matrix = np.zeros((series.size, 2, 2), dtype=complex)
    matrix[:, 0, 0] = 1 + series * shunt
    matrix[:, 0, 1] = series
    matrix[:, 1, 0] = shunt
    matrix[:, 1, 1] = 1
    return matrix


def network_from_matrix(frequency_hz, matrix):
    """Use explicit ABCD-to-S equations, independent of skrf matrix conversion."""
    import skrf as rf

    a, b = matrix[:, 0, 0], matrix[:, 0, 1]
    c, d = matrix[:, 1, 0], matrix[:, 1, 1]
    denominator = a + b + c + d
    scattering = np.zeros_like(matrix)
    scattering[:, 0, 0] = (a + b - c - d) / denominator
    scattering[:, 1, 1] = (-a + b - c + d) / denominator
    scattering[:, 1, 0] = 2 / denominator
    scattering[:, 0, 1] = 2 * (a * d - b * c) / denominator
    return rf.Network(f=frequency_hz, s=scattering, z0=1)


@dataclass
class AnalyticalLines:
    frequency_hz: np.ndarray
    gamma_per_m: np.ndarray
    left: np.ndarray
    right: np.ndarray

    def line(self, length_m):
        return network_from_matrix(
            self.frequency_hz,
            self.left @ line_matrix(self.gamma_per_m, length_m) @ self.right,
        )

    def isolated_reflect(self, offset_m=50e-6):
        import skrf as rf

        left = network_from_matrix(self.frequency_hz, self.left)
        right = network_from_matrix(self.frequency_hz, self.right)
        # Imperfect unknown common termination; launches intentionally differ.
        reflection = -0.91 * np.exp(0.12j) * np.exp(-2 * self.gamma_per_m * offset_m)
        scattering = np.zeros_like(left.s)
        scattering[:, 0, 0] = left.s[:, 0, 0] + (
            left.s[:, 0, 1]
            * left.s[:, 1, 0]
            * reflection
            / (1 - left.s[:, 1, 1] * reflection)
        )
        scattering[:, 1, 1] = right.s[:, 1, 1] + (
            right.s[:, 0, 1]
            * right.s[:, 1, 0]
            * reflection
            / (1 - right.s[:, 0, 0] * reflection)
        )
        return rf.Network(f=self.frequency_hz, s=scattering, z0=1)


@pytest.fixture
def analytical_lines():
    pytest.importorskip("skrf")
    frequency_hz = np.linspace(35e9, 100e9, 66)
    omega = 2 * np.pi * frequency_hz
    inductance = 400e-9
    capacitance = 160e-12
    resistance = 400 * np.sqrt(frequency_hz / 1e9)
    conductance = resistance * capacitance / inductance
    # Independent lossy telegrapher solution with real 50-ohm Zc. S matrices
    # below express waves normalized to this line, not an inferred ohm value.
    gamma = np.sqrt(
        (resistance + 1j * omega * inductance)
        * (conductance + 1j * omega * capacitance)
    )
    f_ghz = frequency_hz / 1e9
    left = lumped_matrix(0.08 + 1j * f_ghz / 400, 0.015 + 1j * f_ghz / 800)
    right = lumped_matrix(0.04 + 1j * f_ghz / 700, 0.025 + 1j * f_ghz / 350)
    return AnalyticalLines(frequency_hz, gamma, left, right)
