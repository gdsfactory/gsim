"""Check the grating notebook's exported source and fiber receiver."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import meshio
import nbformat
import numpy as np
import pytest
from matplotlib.colors import SymLogNorm

from gsim import fdtd


@pytest.mark.parametrize("u_axis", ["x", "z"])
def test_grating_field_plot_uses_recorded_physical_axes(u_axis, monkeypatch):
    """A nonsquare analytic field must align with physical X/Z in either order."""
    notebook_path = (
        Path(__file__).resolve().parents[2]
        / "nbs"
        / "fdtd_focusing_grating_coupler_gpdk.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    source = next(
        cell.source
        for cell in notebook.cells
        if "field_norm = SymLogNorm" in cell.source
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    x = np.array([2.0, 3.0, 4.0, 5.0])
    z = np.array([-1.0, 0.0, 1.0])
    physical = x[None, :] + 10 * z[:, None]
    transposed = u_axis == "z"
    coordinates = SimpleNamespace(
        u_um=z if transposed else x,
        v_um=x if transposed else z,
        normal_position_um=0.0,
    )
    sample = SimpleNamespace(
        wavelength_um=1.55,
        coordinates=coordinates,
        load=lambda: (physical.T if transposed else physical).astype(complex),
    )
    plane = SimpleNamespace(u_axis=u_axis, field=lambda *_args, **_kwargs: sample)
    overlays = []
    namespace: dict[str, Any] = {
        "np": np,
        "plt": plt,
        "SymLogNorm": SymLogNorm,
        "result": SimpleNamespace(monitors={"grating_side": plane}),
        "simulation": SimpleNamespace(
            plot_2d_static=lambda **kwargs: overlays.append(kwargs)
        ),
    }
    try:
        exec(compile(source, str(notebook_path), "exec"), namespace)  # noqa: S102
        np.testing.assert_array_equal(namespace["x"], x)
        np.testing.assert_array_equal(namespace["z"], z)
        np.testing.assert_allclose(
            namespace["image"].get_array(), physical / np.abs(physical).max()
        )
        assert namespace["axis"].get_xlim() == (x[0], x[-1])
        assert namespace["axis"].get_ylim() == (z[0], z[-1])
        assert overlays == [
            {
                "axis": "y",
                "position_um": 0.0,
                "ax": namespace["axis"],
                "outline_only": True,
            }
        ]
    finally:
        plt.close(namespace.get("figure"))


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
