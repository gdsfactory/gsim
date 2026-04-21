"""Smoke tests that the generated MEEP runner script is valid Python."""

from __future__ import annotations

import ast

from gsim.meep.script import _MEEP_RUNNER_TEMPLATE


def test_runner_template_parses():
    """Generated runner must be valid Python at import time."""
    ast.parse(_MEEP_RUNNER_TEMPLATE)


def test_runner_template_contains_xz_branch():
    """XZ 2D branch should be present in the runner."""
    assert 'plane = config.get("plane", "xy")' in _MEEP_RUNNER_TEMPLATE
    assert 'is_xz = plane == "xz"' in _MEEP_RUNNER_TEMPLATE


def test_runner_has_xz_geometry_path():
    """Inlined cross-section cutter and XZ geometry builder exist in the runner."""
    assert "_build_geometry_xz" in _MEEP_RUNNER_TEMPLATE
    assert "extract_xz_rectangles_runner" in _MEEP_RUNNER_TEMPLATE


def test_runner_has_fiber_source_path():
    """Gaussian-beam fiber source builder is present in the runner."""
    assert "_build_fiber_source" in _MEEP_RUNNER_TEMPLATE
    assert "GaussianBeamSource" in _MEEP_RUNNER_TEMPLATE
