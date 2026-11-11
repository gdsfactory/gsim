"""Check the grating notebook's exported source and fiber receiver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import meshio
import nbformat
import numpy as np
import pytest

from gsim import fdtd


def test_grating_notebook_launches_guided_mode_and_captures_fiber_fields(tmp_path):
    notebook_path = (
        Path(__file__).resolve().parents[2]
        / "nbs"
        / "fdtd_focusing_grating_coupler_gpdk.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    preview_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and cell.source.strip() == "simulation.plot_3d()"
    )
    namespace: dict[str, Any] = {"__name__": "__grating_notebook_setup__"}
    # Stop before plotting or running; only the checked-in setup is executed.
    for cell in notebook.cells[:preview_index]:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(notebook_path), "exec"), namespace)  # noqa: S102

    simulation = namespace["simulation"]
    assert isinstance(simulation, fdtd.Simulation)
    assert isinstance(simulation.source, fdtd.PortSource)
    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text())

    assert document["excitation"]["type"] == "eigenmode"
    assert document["excitation"]["default_port"] == "o1"
    assert document["excitation"]["center_wavelength"] == pytest.approx(1554)
    assert document["excitation"]["num_wavelengths"] == 101
    assert document["grid"]["nanometers_per_cell"] == 50
    assert namespace["grating"].cell.settings["fiber_angle"] == 10
    assert namespace["fiber_center_x_um"] == 20.5
    assert namespace["fiber_y_um"] == 0.0
    assert [monitor["name"] for monitor in document["monitors"]] == [
        "radiation",
        "input_power",
        "grating_side",
    ]
    receiver = document["monitors"][0]
    assert receiver["normal"] == "+z"
    assert receiver["region_min"][2] == receiver["region_max"][2]
    assert receiver["region_min"][2] == pytest.approx(1050)
    assert "fiber_mode" not in receiver
    assert receiver["fields"] == {
        "components": ["ex", "ey", "hx", "hy"],
    }
    # Preserve the complete capture aperture when exporting the notebook.
    assert [
        upper - lower
        for lower, upper in zip(
            receiver["region_min"][:2], receiver["region_max"][:2], strict=True
        )
    ] == pytest.approx([26000, 26000])
    power_monitor = document["monitors"][1]
    assert power_monitor["normal"] == "+x"
    assert power_monitor["flux"]
    assert "fields" not in power_monitor
    assert "fiber_mode" not in power_monitor
    assert power_monitor["region_min"][0] == power_monitor["region_max"][0]
    assert power_monitor["region_min"][0] - document["grid"]["non_pml_bounds"]["min"][
        0
    ] == pytest.approx(1500)
    assert power_monitor["region_min"][1:] == pytest.approx([-750, -500])
    assert power_monitor["region_max"][1:] == pytest.approx([750, 720])
    side_monitor = document["monitors"][2]
    assert side_monitor["normal"] == "+y"
    assert side_monitor["region_min"][1] == 0
    assert side_monitor["region_max"][1] == 0
    assert not side_monitor["flux"]
    assert side_monitor["fields"] == {
        "components": ["ey"],
        "wavelengths": [1550],
    }
    assert side_monitor["region_min"][0] == pytest.approx(
        namespace["grating_x_bounds"][0] * 1000
    )
    assert side_monitor["region_max"][0] == pytest.approx(
        namespace["grating_x_bounds"][1] * 1000
    )
    assert side_monitor["region_min"][2] == pytest.approx(-4000)
    assert side_monitor["region_max"][2] == pytest.approx(1550)

    # Extending the oxide domain alone does not create a silicon substrate.
    # Check the generated material volume, including its physical interface.
    substrate = document["geometry"]["layers"]["substrate"]
    assert substrate["material"] == "si"
    mesh = meshio.read(artifacts.mesh_path)
    tags = mesh.cell_data_dict["gmsh:physical"]["tetra"]
    substrate_cells = mesh.cells_dict["tetra"][tags == substrate["phys_group"]]
    assert len(substrate_cells)
    substrate_points = mesh.points[np.unique(substrate_cells)]
    bounds = document["grid"]["non_pml_bounds"]
    assert substrate_points.min(axis=0) == pytest.approx(
        [bounds["min"][0], bounds["min"][1], -4000]
    )
    assert substrate_points.max(axis=0) == pytest.approx(
        [bounds["max"][0], bounds["max"][1], -3000]
    )
