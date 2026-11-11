"""Check the notebook helper against an independent native solver projection."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest

from gsim import fdtd


def _notebook_helper():
    path = (
        Path(__file__).resolve().parents[2]
        / "nbs"
        / "fdtd_focusing_grating_coupler_gpdk.ipynb"
    )
    notebook = nbformat.read(path, as_version=4)
    function = next(
        node
        for cell in notebook.cells
        if cell.cell_type == "code"
        for node in ast.parse(cell.source).body
        if isinstance(node, ast.FunctionDef) and node.name == "fiber_coupling"
    )
    namespace = {"np": np}
    exec(  # noqa: S102
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["fiber_coupling"]


def test_recorded_notebook_overlap_matches_native_solver():
    directory = Path(__file__).parent / "data" / "fiber_overlap"
    result = fdtd.FDTDResult.from_file(directory / "result.json")
    mode = json.loads((directory / "fiber_mode.json").read_text())
    native = fdtd.fiber_coupling_efficiency(result.monitors["native_diagnostic"])
    # Postprocessing has no dependency on the independently recorded projection.
    result.monitors.pop("native_diagnostic")
    reflection = result.s_parameters[("o1", "o1")].values[0]
    input_flux = result.monitors["input_power"].flux
    assert input_flux is not None
    incident = input_flux[0] / (1 - abs(reflection) ** 2)
    tilt = np.rad2deg(
        np.arctan2(mode["propagation_direction"][0], mode["propagation_direction"][2])
    )
    amplitude, efficiency = _notebook_helper()(
        result.monitors["radiation"],
        1.554,
        incident,
        mode["focal_point_um"],
        tilt_deg=tilt,
        waist_um=mode["waist_radius_um"],
        refractive_index=mode["refractive_index"],
    )
    assert np.isfinite(amplitude)
    assert efficiency == pytest.approx(native.efficiency[0], rel=5e-4)


def test_notebook_overlap_rejects_an_uncaptured_wavelength():
    directory = Path(__file__).parent / "data" / "fiber_overlap"
    result = fdtd.FDTDResult.from_file(directory / "result.json")
    with pytest.raises(ValueError, match="not captured"):
        _notebook_helper()(result.monitors["radiation"], 1.55, 1.0, (18.5, 0, 0.22))
