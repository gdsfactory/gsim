"""The references the magnetostatic notebook checks Palace against.

None of these use finite elements or Palace output. The closed forms are checked
against limits known independently of them, and the ribbon bundles of
``scripts/magnetostatic_pec_bundle.py`` against the closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.magnetostatic_pec_bundle import (
    THICKNESS,
    WIDTH,
    _log_mean_distance,
    _ribbons,
    equivalent_radius,
    maxwell_mutual,
    ring_bundle,
    ring_references,
    self_and_mutual,
)

MU0 = 4e-7 * np.pi


def _neumann(first: float, second: float, points: int = 2000) -> float:
    """Neumann's integral over two coplanar coaxial circles, in pH for um."""
    angle = 2 * np.pi * (np.arange(points) + 0.5) / points
    distance = np.sqrt(first**2 + second**2 - 2 * first * second * np.cos(angle))
    step = 2 * np.pi / points
    return float(
        0.1 * first * second * 2 * np.pi * step * np.sum(np.cos(angle) / distance)
    )


def _ring_self(radius: float) -> float:
    """The slender-ring formula for a perfect conductor, in pH for um."""
    radial = equivalent_radius(WIDTH, THICKNESS)
    return MU0 * 1e6 * radius * (np.log(8 * radius / radial) - 2)


def test_the_equivalent_radius_has_the_known_limits() -> None:
    assert equivalent_radius(1.0, 1.0) == pytest.approx(0.59017, abs=1e-5)
    assert equivalent_radius(4.0, 1e-5) == pytest.approx(1.0, abs=1e-3)
    assert equivalent_radius(2.0, 3.0) == equivalent_radius(3.0, 2.0)


def test_maxwell_matches_neumann_on_two_circles() -> None:
    assert maxwell_mutual(25.0, 50.0) == pytest.approx(_neumann(25.0, 50.0), rel=1e-6)


def test_the_ribbons_converge_to_the_equivalent_radius_of_the_trace() -> None:
    """In the cross-section, the ribbons give the bar's logarithmic capacity."""
    ribbons = _ribbons(0.1)
    logs = np.empty((len(ribbons), len(ribbons)))
    for i, first in enumerate(ribbons):
        for j, second in enumerate(ribbons):
            d = np.hypot(first.u - second.u, first.v - second.v)
            if i == j:
                logs[i, j] = np.log(first.width) - 1.5
            elif first.face == second.face:
                logs[i, j] = _log_mean_distance(d, first.width)
            else:
                logs[i, j] = np.log(d)
    ones = np.ones(len(ribbons))
    radius = np.exp(1 / (ones @ np.linalg.solve(logs, ones)))
    assert radius == pytest.approx(equivalent_radius(WIDTH, THICKNESS), rel=2e-3)


def test_a_slender_ring_bundle_reaches_the_closed_forms() -> None:
    """Far from the conductor size, the section no longer matters."""
    matrix = ring_bundle(2500.0, 5000.0, 0.25)
    assert matrix[0, 0] == pytest.approx(_ring_self(2500.0), rel=2e-3)
    assert matrix[1, 1] == pytest.approx(_ring_self(5000.0), rel=2e-3)
    assert matrix[0, 1] == pytest.approx(maxwell_mutual(2500.0, 5000.0), rel=1e-4)


def test_the_notebook_quotes_the_ring_references() -> None:
    matrix = ring_references()
    assert matrix[0, 0] == pytest.approx(91.25, abs=5e-3)
    assert matrix[1, 1] == pytest.approx(226.08, abs=5e-3)
    assert matrix[0, 1] == pytest.approx(27.106, abs=5e-4)


def test_the_bundle_reproduces_its_published_series() -> None:
    """One width of the series behind the notebook's quoted pair references."""
    self_inductance, mutual = self_and_mutual(1.0)
    assert self_inductance == pytest.approx(52.827, abs=1e-3)
    assert mutual == pytest.approx(-0.95518, abs=1e-5)
