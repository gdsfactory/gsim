"""Transient config exports must follow Palace's waveform and time contract."""

from __future__ import annotations

import pytest

from gsim.palace.models import TransientConfig


@pytest.mark.parametrize(
    ("excitation", "palace_excitation"),
    [
        ("sinusoidal", "Sinusoidal"),
        ("gaussian", "Gaussian"),
        ("ramp", "Ramp"),
        ("smoothstep", "SmoothStep"),
    ],
)
def test_transient_waveform_export(excitation, palace_excitation):
    config = TransientConfig(
        max_time=10,
        time_step=0.1,
        excitation=excitation,
        excitation_freq=5e9,
        excitation_width=0.25,
    ).to_palace_config()

    assert config == {
        "Excitation": palace_excitation,
        "MaxTime": 10.0,
        "TimeStep": 0.1,
        "ExcitationFreq": 5.0,
        "ExcitationWidth": 0.25,
    }


def test_optional_pulse_parameters_are_omitted():
    assert TransientConfig(max_time=10, time_step=0.1).to_palace_config() == {
        "Excitation": "Sinusoidal",
        "MaxTime": 10.0,
        "TimeStep": 0.1,
    }


def test_time_step_is_required():
    with pytest.raises(ValueError, match="time_step"):
        TransientConfig(max_time=10)


@pytest.mark.parametrize("field", ["max_time", "time_step"])
@pytest.mark.parametrize("value", [0, -0.1, float("inf"), float("nan")])
def test_times_must_be_positive_and_finite(field, value):
    settings = {"max_time": 10, "time_step": 0.1, field: value}
    with pytest.raises(ValueError, match=field):
        TransientConfig(**settings)


@pytest.mark.parametrize("field", ["excitation_freq", "excitation_width"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_pulse_parameters_must_be_finite(field, value):
    with pytest.raises(ValueError, match=field):
        TransientConfig(max_time=10, time_step=0.1, **{field: value})


def test_explicit_zero_pulse_parameters_preserve_palace_defaults():
    config = TransientConfig(
        max_time=10, time_step=0.1, excitation_freq=0, excitation_width=0
    ).to_palace_config()

    assert config["ExcitationFreq"] == 0
    assert config["ExcitationWidth"] == 0
