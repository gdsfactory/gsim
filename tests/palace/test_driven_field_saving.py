"""Field output must refer to actual Palace frequency samples."""

from __future__ import annotations

import pytest

from gsim.palace import DrivenSim
from gsim.palace.models import DrivenConfig


def test_log_sweep_center_save_uses_log_grid():
    """The nearest sample to 50.5 GHz on [1, 10, 100] is 10 GHz."""
    sim = DrivenSim()
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=3, scale="log", save_freq="center")

    assert sim.driven.to_palace_config()["Save"] == pytest.approx([10.0])


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_single_frequency_center_save(scale):
    """The public convenience API saves a single-frequency solution."""
    sim = DrivenSim()
    sim.set_driven(f=5e9, scale=scale, save_freq="center")

    config = sim.driven.to_palace_config()
    assert config["Save"] == [5.0]
    assert config["Samples"] == [{"Type": "Point", "Freq": [5.0], "SaveStep": 0}]


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_one_point_range_samples_only_lower_bound(scale):
    """A one-point request has one sample even with distinct frequency bounds."""
    config = DrivenConfig(
        fmin=1e9,
        fmax=100e9,
        num_points=1,
        scale=scale,
        save_fields_at=[50e9],
    ).to_palace_config()

    assert config["Save"] == [1.0]
    assert config["Samples"] == [{"Type": "Point", "Freq": [1.0], "SaveStep": 0}]


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_zero_width_sweep_with_multiple_requested_points(scale):
    """Degenerate ranges still have a single frequency to save."""
    config = DrivenConfig(
        fmin=5e9, fmax=5e9, num_points=40, scale=scale, save_fields_at=[6e9]
    ).to_palace_config()

    assert config["Save"] == [5.0]


def test_log_saves_clamp_deduplicate_and_preserve_order():
    """Repeated and out-of-range requests map to unique real samples."""
    config = DrivenConfig(
        fmin=1e9,
        fmax=100e9,
        num_points=3,
        scale="log",
        save_fields_at=[90e9, 10.1e9, 10.3e9, 0.5e9, 200e9],
    ).to_palace_config()

    assert config["Save"] == pytest.approx([100.0, 10.0, 1.0])


def test_linear_saves_keep_nearest_sample_behavior():
    config = DrivenConfig(
        fmin=1e9,
        fmax=3e9,
        num_points=5,
        save_fields_at=[2.1e9, 2e9, 0.5e9, 4e9],
    ).to_palace_config()

    assert config["Save"] == [2.0, 1.0, 3.0]


def test_exact_log_saves_do_not_warn(caplog):
    config = DrivenConfig(
        fmin=1e9,
        fmax=100e9,
        num_points=3,
        scale="log",
        save_fields_at=[1e9, 10e9, 100e9],
    ).to_palace_config()

    assert config["Save"] == pytest.approx([1.0, 10.0, 100.0])
    assert not caplog.records


@pytest.mark.parametrize("frequency", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_save_frequency_is_rejected(frequency):
    with pytest.raises(ValueError, match="finite"):
        DrivenConfig(save_fields_at=[frequency])
