"""Check monitor allocation budgets without meshing or running a solver."""

import math

import pytest

from gsim.fdtd.config import FDTDConfig
from gsim.fdtd.metadata import _cell_counts
from gsim.fdtd.monitor_memory import estimate_monitor_memory


def _config(*, monitors=(), count=3, cell=100, guided=False, cw=False):
    source = {
        "type": "eigenmode" if guided else "dipole",
        "center_wavelength": 1500,
        "wavelength_halfspan": 0 if cw else 50,
        "num_wavelengths": 1 if cw else count,
        "waveform": "continuous_wave" if cw else "pulse",
    }
    source.update(
        {"default_port": "o1"}
        if guided
        else {"dipole": {"position": (0, 0, 0), "current_axis": "x"}}
    )
    return FDTDConfig.model_validate(
        {
            "background_refractive_index": 1.5,
            "materials": {"core": {"refractive_index": 3.5}},
            "geometry": {
                "volumes": {
                    "background": {"phys_group": 3, "material": "core", "priority": 0}
                },
                "layers": {
                    "core": {"phys_group": 1, "material": "core", "priority": 1}
                },
                "ports": {"o1": {"phys_group": 2, "layer": "core", "normal": (1, 0, 0)}}
                if guided
                else {},
            },
            "excitation": source,
            "grid": {
                "nanometers_per_cell": cell,
                "pml_cells": 32,
                "non_pml_bounds": {
                    "min": (-10000, -10000, -1000),
                    "max": (10000, 10000, 1000),
                },
            },
            "run": {"energy_decay_fraction": 1e-6, "max_wall_seconds": 3600},
            "monitors": list(monitors),
        }
    )


def _plane(**overrides):
    return {
        "name": "top",
        "region_min": (0, 0, 100),
        "region_max": (1000, 2000, 100),
        "normal": "+z",
        **overrides,
    }


def _estimate(config, port_bounds=None):
    return estimate_monitor_memory(config, port_bounds or {}, _cell_counts(config))


def test_no_monitors_or_ports_needs_no_monitor_memory():
    estimate = _estimate(_config())
    assert estimate["estimated_monitor_memory_bytes"] == 0
    assert estimate["monitors"] == []


def test_plane_allocation_uses_two_cells_six_complex_fields_and_sample_slots():
    config = _config(monitors=[_plane(wavelengths=[1500])], cw=True)
    estimate = _estimate(config)
    pixels = 11 * 21  # Conservative cells intersecting a 1 by 2 um plane.
    stride = math.floor(1500 / (2 * 0.95 * 100 / math.sqrt(3)))
    samples = 256 // stride + 2
    assert estimate["accumulator_bytes"] == pixels * 2 * 6 * 16
    assert estimate["sampling_buffer_bytes"] == pixels * 2 * 6 * 4 * samples
    assert estimate["reference_profile_bytes"] == 0
    assert estimate["estimated_monitor_memory_bytes"] == (pixels * (192 + 48 * samples))


def test_heatmap_union_keeps_explicit_duplicate_frequency_allocations():
    monitor = _plane(
        wavelengths=[1450, 1500, 1500], heatmap={"wavelengths": [1500, 1550]}
    )
    estimate = _estimate(_config(monitors=[monitor]))
    assert estimate["monitors"][0]["frequency_count"] == 4
    assert estimate["accumulator_bytes"] == 11 * 21 * 192 * 4


def test_fiber_profiles_add_four_complex_components_per_pixel_and_frequency():
    fiber = {
        "propagation_direction": (0, 0, 1),
        "e_polarization": (0, 1, 0),
        "focal_point": (0, 0, 0),
        "waist_radius": 4600,
        "refractive_index": 1.5,
    }
    plain = _estimate(_config(monitors=[_plane()]))
    receiver = _estimate(_config(monitors=[_plane(fiber_mode=fiber)]))
    expected = 11 * 21 * 3 * 64
    assert receiver["reference_profile_bytes"] == expected
    assert (
        receiver["estimated_monitor_memory_bytes"]
        - plain["estimated_monitor_memory_bytes"]
        == expected
    )


@pytest.mark.parametrize("count", [1, 51, 501])
def test_guided_ports_are_counted_without_explicit_monitors(count):
    config = _config(guided=True, cell=50, count=count)
    bounds = {"o1": {"min": (0, -200, 0), "max": (0, 200, 200)}}
    estimate = _estimate(config, bounds)
    entry = estimate["monitors"][0]
    assert entry["kind"] == "guided_port"
    assert entry["frequency_count"] == count
    assert entry["reference_profile_bytes"] == entry["pixels_upper_bound"] * count * 65
    assert entry["retained_port_mode_bytes"] > 0
    assert entry["accumulator_bytes"] == entry["pixels_upper_bound"] * count * 192


def test_monitor_budgets_add_and_do_not_change_grid_counts():
    first = _config(monitors=[_plane()])
    both = _config(monitors=[_plane(), _plane(name="second")])
    assert _cell_counts(first) == _cell_counts(both)
    assert (
        _estimate(both)["estimated_monitor_memory_bytes"]
        == 2 * _estimate(first)["estimated_monitor_memory_bytes"]
    )


def test_explicit_monitor_spectrum_does_not_inherit_source_point_count():
    one = _config(monitors=[_plane(wavelengths=[1500])], count=51)
    many = _config(monitors=[_plane(wavelengths=[1500])], count=501)
    assert _estimate(one) == _estimate(many)


def test_zero_span_spectrum_records_only_one_frequency():
    config = _config(count=501)
    config.excitation.wavelength_halfspan = 0
    config.monitors = _config(monitors=[_plane()]).monitors
    assert _estimate(config)["monitors"][0]["frequency_count"] == 1


def test_translated_non_grid_aligned_plane_is_not_underestimated():
    config = _config(
        monitors=[_plane(region_min=(17, 23, 100), region_max=(1017, 2023, 100))]
    )
    assert _estimate(config)["monitors"][0]["pixels_upper_bound"] >= 10 * 20
