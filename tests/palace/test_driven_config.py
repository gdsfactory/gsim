"""Regression tests for Palace driven frequency sample specifications."""

from __future__ import annotations

import pytest

from gsim.palace.models import DrivenConfig


@pytest.mark.parametrize("num_points", [2, 101])
def test_log_sweep_exports_sample_count(num_points):
    """Palace log samples require NSample and reject FreqStep."""
    config = DrivenConfig(
        fmin=10e6,
        fmax=60e9,
        num_points=num_points,
        scale="log",
        save_step=5,
        adaptive_tol=1e-3,
        adaptive_max_samples=20,
    ).to_palace_config()

    assert config == {
        "Samples": [
            {
                "Type": "Log",
                "MinFreq": 0.01,
                "MaxFreq": 60.0,
                "NSample": num_points,
                "SaveStep": 5,
            }
        ],
        "AdaptiveTol": 1e-3,
        "AdaptiveMaxSamples": 20,
    }


def test_linear_sweep_keeps_frequency_step():
    """Linear sweeps retain their GHz step and optional field output."""
    config = DrivenConfig(
        fmin=1e9,
        fmax=3e9,
        num_points=5,
        adaptive_tol=0,
        save_fields_at=[2e9],
    ).to_palace_config()

    assert config == {
        "Samples": [
            {
                "Type": "Linear",
                "MinFreq": 1.0,
                "MaxFreq": 3.0,
                "FreqStep": 0.5,
                "SaveStep": 0,
            }
        ],
        "AdaptiveTol": 0,
        "Save": [2.0],
    }


def test_single_frequency_linear_sweep():
    """A one-point linear request exports an explicit frequency."""
    config = DrivenConfig(fmin=50e9, fmax=50e9, num_points=1).to_palace_config()

    assert config["Samples"] == [
        {
            "Type": "Point",
            "Freq": [50.0],
            "SaveStep": 0,
        }
    ]


@pytest.mark.parametrize("fmax", [1e9, 100e9])
def test_single_point_log_sweep(fmax):
    """Avoid Palace's division by zero for logarithmic NSample=1."""
    config = DrivenConfig(
        fmin=1e9, fmax=fmax, num_points=1, scale="log", save_step=1
    ).to_palace_config()

    assert config["Samples"] == [{"Type": "Point", "Freq": [1.0], "SaveStep": 1}]
