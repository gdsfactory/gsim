"""Load and plot local and downloaded Palace outputs with their input meshes."""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import pytest
import pyvista as pv
from vtkmodules.vtkIOParallelXML import vtkXMLPUnstructuredGridWriter

from gsim.palace.field_viz import plot_fields_2d, resolve_physical_groups
from gsim.palace.results import load_fields, load_text_results


def _write_cycle(directory: Path, dataset: pv.UnstructuredGrid) -> None:
    directory.mkdir(parents=True)
    writer = vtkXMLPUnstructuredGridWriter()
    writer.SetFileName(str(directory / "data.pvtu"))
    writer.SetInputData(dataset)
    assert writer.Write() == 1


@pytest.fixture(params=["output/palace", "output"], ids=["local", "cloud"])
def simulation_output(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    output_dir = tmp_path / request.param
    output_dir.mkdir(parents=True)
    mesh_dir = tmp_path if request.param == "output/palace" else tmp_path / "input"
    mesh_dir.mkdir(exist_ok=True)

    grid = (
        pv.Plane(i_resolution=1, j_resolution=1)
        .triangulate()
        .cast_to_unstructured_grid()
    )
    grid.cell_data["attribute"] = np.ones(grid.n_cells, dtype=np.int32)
    mesh = meshio.Mesh(
        points=grid.points,
        cells=[("triangle", grid.cells.reshape(-1, 4)[:, 1:])],
        field_data={"rib": np.array([1, 2])},
        cell_data={
            "gmsh:physical": [np.ones(grid.n_cells, dtype=int)],
            "gmsh:geometrical": [np.ones(grid.n_cells, dtype=int)],
        },
    )
    mesh.write(mesh_dir / "palace.msh", file_format="gmsh22", binary=False)
    (output_dir / "mode-kn.csv").write_text(
        "m,Re{kn} (1/m),Im{kn} (1/m),Re{n_eff},Im{n_eff}\n1,5,0,2,0\n"
    )
    for solver in ("boundarymode", "boundarymode_boundary"):
        for cycle in (1, 2):
            grid.point_data["E_real"] = np.full((grid.n_points, 3), float(cycle))
            _write_cycle(output_dir / "paraview" / solver / f"Cycle{cycle:06d}", grid)

    # Palace's final volume cycle contains partition metadata in cell data.
    grid.point_data.clear()
    grid.cell_data["Indicator"] = np.zeros(grid.n_cells)
    grid.cell_data["Rank"] = np.zeros(grid.n_cells)
    _write_cycle(output_dir / "paraview/boundarymode/Cycle000003", grid)
    return output_dir


@pytest.mark.parametrize("use_output_dir", [False, True], ids=["root", "output-dir"])
@pytest.mark.parametrize("boundary", [False, True], ids=["volume", "boundary"])
def test_load_fields_from_simulation_directory(
    simulation_output: Path, tmp_path: Path, use_output_dir: bool, boundary: bool
) -> None:
    source = simulation_output if use_output_dir else tmp_path
    dataset = load_fields(source, boundary=boundary)
    np.testing.assert_array_equal(dataset.point_data["E_real"], 2.0)
    first_mode = load_fields(source, cycle=1, boundary=boundary)
    np.testing.assert_array_equal(first_mode.point_data["E_real"], 1.0)


def test_load_text_results_from_simulation_directory(
    simulation_output: Path, tmp_path: Path
) -> None:
    results = load_text_results(tmp_path)
    assert results.files["mode-kn.csv"] == simulation_output / "mode-kn.csv"
    assert results.modes[1]["n_eff"] == 2.0
    dataset = load_fields(results.files)
    np.testing.assert_array_equal(dataset.point_data["E_real"], 2.0)


@pytest.mark.parametrize("use_output_dir", [False, True], ids=["root", "output-dir"])
def test_plot_fields_with_physical_groups(
    simulation_output: Path, tmp_path: Path, use_output_dir: bool
) -> None:
    source = simulation_output if use_output_dir else tmp_path
    assert resolve_physical_groups(source, ["rib"]) == [1]
    plotter = plot_fields_2d(source, physical_groups=["rib"], show=False)
    assert isinstance(plotter, pv.Plotter)
    plotter.close()
