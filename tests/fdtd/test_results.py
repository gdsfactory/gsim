"""Tests for typed GDSFactory FDTD results, tables, and monitor plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gsim.fdtd import FDTDResult
from gsim.gcloud import RunResult


def _result_document(source_type: str = "eigenmode") -> dict[str, Any]:
    common: dict[str, Any] = {
        "schema_version": 1,
        "excitation_type": source_type,
        "excited_port": "o1" if source_type == "eigenmode" else None,
        "ports": ["o1", "o2"],
        "frequencies": {
            "wavelength_nm": [1500, 1550, 1600],
            "hz": [2.0e14, 1.93e14, 1.87e14],
            "below_noise_floor": [True, False, False],
        },
        "plane_monitors": {
            "top": {
                "normal_axis": "z",
                "normal_sign": 1,
                "u_axis": "x",
                "v_axis": "y",
                "plane_position_nm": 500,
                "u_extent_nm": [-1000, 1000],
                "v_extent_nm": [-500, 500],
                "shape": [2, 3],
                "wavelength_nm": [1500, 1550, 1600],
                "flux": [0.1, 0.2, 0.3],
                "heatmaps": [
                    {
                        "file": "top_intensity_1550nm.npy",
                        "quantity": "intensity",
                        "wavelength_nm": 1550,
                        "shape": [2, 3],
                    }
                ],
            }
        },
        "convergence": {"converged": True},
        "grid": {"total_cells": 100},
        "timing": {"run_seconds": 2},
        "config_resolved": {"source_type": source_type},
    }
    samples = [
        {"re": 0, "im": 0},
        {"re": 0.5, "im": 0},
        {"re": 0, "im": 0.25},
    ]
    if source_type == "eigenmode":
        common["s_parameters"] = {"S(o2,o1)": samples}
    else:
        common["port_outputs"] = {
            "o2": [
                {**sample, "modal_power": power, "power_fraction": 1.0}
                for sample, power in zip(samples, [0, 0.25, 0.0625], strict=True)
            ]
        }
    return common


def _write_result(tmp_path: Path, source_type: str = "eigenmode") -> Path:
    result_path = tmp_path / "sparams_o1.json"
    result_path.write_text(json.dumps(_result_document(source_type)), encoding="utf8")
    np.save(tmp_path / "top_intensity_1550nm.npy", np.arange(6).reshape(2, 3))
    return result_path


def test_eigenmode_result_masks_noise_and_builds_dataframe(tmp_path: Path) -> None:
    result = FDTDResult.from_file(_write_result(tmp_path))

    trace = result.s_parameters[("o2", "o1")]
    frame = result.s_parameters.to_dataframe()

    assert result.wavelength_um.tolist() == [1.5, 1.55, 1.6]
    assert trace.valid.tolist() == [False, True, True]
    assert frame["magnitude"].tolist() == [0, 0.5, 0.25]
    figure, axes = result.plot()
    assert np.isnan(axes.lines[0].get_ydata()[0])
    plt.close(figure)
    plotly_figure = result.plot_plotly()
    assert plotly_figure.data[0].name == "S(o2,o1)"
    assert tuple(plotly_figure.layout.xaxis.range) == (1.55, 1.6)


def test_monitor_heatmap_is_lazy_and_uses_physical_axes(tmp_path: Path) -> None:
    result_path = _write_result(tmp_path)
    result = FDTDResult.from_file(result_path)
    monitor = result.monitors["top"]

    selected = monitor.heatmap(wavelength_um=1.551)
    assert selected.load().shape == (2, 3)
    figure, axes = monitor.plot_heatmap(wavelength_um=1.55)
    assert axes.get_xlabel() == "x (µm)"
    assert axes.get_ylabel() == "y (µm)"
    plt.close(figure)


def test_non_eigenmode_result_and_run_result_parser(tmp_path: Path) -> None:
    result_path = _write_result(tmp_path, "dipole")
    raw_result = RunResult(
        sim_dir=tmp_path,
        files={result_path.name: result_path},
        job_name="fdtd-test",
    )

    result = FDTDResult.from_run_result(raw_result)
    frame = result.port_outputs.to_dataframe()

    assert result.job_name == "fdtd-test"
    assert frame["modal_power"].tolist() == [0, 0.25, 0.0625]
    figure, axes = result.plot()
    assert axes.get_ylabel() == "Modal power"
    plt.close(figure)
    plotly_figure = result.plot_plotly()
    assert plotly_figure.layout.yaxis.title.text == "Modal power"

    normalized_figure = result.plot_plotly(normalize_to="top")
    assert normalized_figure.layout.yaxis.title.text == "Power / |top flux|"
    np.testing.assert_allclose(
        normalized_figure.data[0].y,
        [np.nan, 1.25, 0.0625 / 0.3],
        equal_nan=True,
    )
