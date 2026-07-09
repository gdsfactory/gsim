"""Tests for BoundaryMode and cross-section model helpers."""

from __future__ import annotations

import math

import pytest

from gsim.palace.models import BoundaryModeConfig, CrossSectionPlaneConfig


class TestCrossSectionPlaneConfig:
    """Plane parsing and validation tests."""

    def test_parse_string_spec(self):
        """from_spec parses axis/value with whitespace and uppercase axis."""
        plane = CrossSectionPlaneConfig.from_spec(" Y = 100.5 ")
        assert plane.axis == "y"
        assert plane.value == pytest.approx(100.5)

    def test_parse_invalid_spec(self):
        """Invalid plane specs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid plane spec"):
            CrossSectionPlaneConfig.from_spec("invalid")

    def test_reject_non_finite_value(self):
        """Non-finite coordinates are rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            CrossSectionPlaneConfig(axis="x", value=math.inf)


class TestBoundaryModeConfig:
    """BoundaryMode config serialization and validation tests."""

    def test_to_palace_config(self):
        """BoundaryModeConfig serializes to Palace JSON format."""
        cfg = BoundaryModeConfig(
            freq=12.5e9,
            num_modes=2,
            save=1,
            target=1.8,
            tolerance=1e-7,
            max_size=80,
            solver_type="SLEPc",
        )

        result = cfg.to_palace_config()
        assert result["Freq"] == pytest.approx(12.5)
        assert result["N"] == 2
        assert result["Save"] == 1
        assert result["Target"] == pytest.approx(1.8)
        assert result["Tol"] == pytest.approx(1e-7)
        assert result["MaxSize"] == 80
        assert result["Type"] == "SLEPc"
        assert "Attributes" not in result
