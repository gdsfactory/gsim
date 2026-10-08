"""Read real parallel VTK eigenmode outputs, including metadata-only cycles."""

from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np
import pytest
import pyvista as pv

from gsim.palace.results import load_fields


def _write_parallel_cycle(
    directory: Path,
    *,
    mode: int | None,
    boundary: bool = False,
    metadata: tuple[str, ...] = ("Indicator", "Rank", "attribute"),
) -> None:
    """Write two disjoint VTK partitions and their parallel manifest."""
    directory.mkdir(parents=True)
    root = Element("VTKFile", type="PUnstructuredGrid", version="0.1")
    parallel = SubElement(root, "PUnstructuredGrid", GhostLevel="0")
    point_data = SubElement(parallel, "PPointData")
    cell_data = SubElement(parallel, "PCellData")
    if mode is not None or "attribute" in metadata:
        SubElement(cell_data, "PDataArray", type="Int32", Name="attribute")
    if mode is None:
        for name in metadata:
            if name != "attribute":
                SubElement(cell_data, "PDataArray", type="Float64", Name=name)
    else:
        for name in ("E_real", "E_imag"):
            SubElement(
                point_data,
                "PDataArray",
                type="Float64",
                Name=name,
                NumberOfComponents="3",
            )
    points = SubElement(parallel, "PPoints")
    SubElement(points, "PDataArray", type="Float64", NumberOfComponents="3")

    for partition in range(2):
        coordinates = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        coordinates[:, 0] += 2 * partition
        if boundary:
            grid = pv.UnstructuredGrid(
                [3, 0, 1, 2], [pv.CellType.TRIANGLE], coordinates[:3]
            )
        else:
            grid = pv.UnstructuredGrid(
                [4, 0, 1, 2, 3], [pv.CellType.TETRA], coordinates
            )
        if mode is not None or "attribute" in metadata:
            grid.cell_data["attribute"] = np.array([partition + 1], dtype=np.int32)
        if mode is None:
            for name in metadata:
                if name != "attribute":
                    grid.cell_data[name] = np.array([float(partition)])
        else:
            grid.point_data["E_real"] = np.full(
                (grid.n_points, 3), float(mode + 10 * partition)
            )
            grid.point_data["E_imag"] = np.full((grid.n_points, 3), -float(mode))
        filename = f"data.{partition:06d}.vtu"
        grid.save(directory / filename)
        SubElement(parallel, "Piece", Source=filename)
    ElementTree(root).write(directory / "data.pvtu")


@pytest.fixture(params=[".", "output/palace", "output"])
def eigenmode_output(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    output = tmp_path / request.param
    output.mkdir(parents=True, exist_ok=True)
    (output / "eig.csv").write_text("m,f (GHz)\n1,1.0\n2,2.0\n")
    for boundary in (False, True):
        solver = "eigenmode_boundary" if boundary else "eigenmode"
        for cycle in (1, 2, 3):
            _write_parallel_cycle(
                output / "paraview" / solver / f"Cycle{cycle:06d}",
                mode=cycle if cycle < 3 else None,
                boundary=boundary,
            )
    return output


@pytest.mark.parametrize("source_kind", ["simulation", "output", "results"])
@pytest.mark.parametrize("boundary", [False, True])
def test_load_eigenmode_partitions(
    eigenmode_output: Path, tmp_path: Path, source_kind: str, boundary: bool
) -> None:
    sources = {
        "simulation": tmp_path,
        "output": eigenmode_output,
        "results": {"eig.csv": eigenmode_output / "eig.csv"},
    }
    source = sources[source_kind]
    latest = load_fields(source, boundary=boundary)
    first = load_fields(source, mode=1, boundary=boundary)
    by_cycle = load_fields(source, cycle=1, boundary=boundary)
    assert latest.n_cells == first.n_cells == 2
    assert first.n_points == (6 if boundary else 8)
    np.testing.assert_array_equal(np.unique(latest["E_real"]), [2.0, 12.0])
    np.testing.assert_array_equal(np.unique(first["E_real"]), [1.0, 11.0])
    np.testing.assert_array_equal(first["E_imag"], -1.0)
    np.testing.assert_array_equal(first["E_real"], by_cycle["E_real"])
    np.testing.assert_array_equal(first.cell_data["attribute"], [1, 2])


def test_explicit_metadata_cycle_is_available(eigenmode_output: Path) -> None:
    dataset = load_fields(eigenmode_output, cycle=3)
    assert set(dataset.cell_data) == {"attribute", "Indicator", "Rank"}
    with pytest.raises(ValueError, match=r"Mode 3 has no solution fields.*Cycle000003"):
        load_fields(eigenmode_output, mode=3)


def test_missing_mode_reports_exact_cycle(eigenmode_output: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Cycle000004"):
        load_fields(eigenmode_output, mode=4)


@pytest.mark.parametrize("metadata", ["Indicator", "Rank", "attribute"])
def test_metadata_only_output_fails_clearly(tmp_path: Path, metadata: str) -> None:
    directory = tmp_path / "paraview/eigenmode/Cycle000001"
    _write_parallel_cycle(directory, mode=None, metadata=(metadata,))
    with pytest.raises(ValueError, match=r"No solution fields found.*eigenmode"):
        load_fields(tmp_path)


@pytest.mark.parametrize("mode", [0, -1])
def test_invalid_mode(tmp_path: Path, mode: int) -> None:
    with pytest.raises(ValueError, match="one-based positive integer"):
        load_fields(tmp_path, mode=mode)


def test_conflicting_selectors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode or cycle, not both"):
        load_fields(tmp_path, cycle=1, mode=1)


@pytest.mark.parametrize("solver", ["driven", "driven/excitation_1", "boundarymode"])
def test_existing_solver_layouts(tmp_path: Path, solver: str) -> None:
    _write_parallel_cycle(tmp_path / "paraview" / solver / "Cycle000001", mode=1)
    dataset = load_fields(tmp_path, cycle=1)
    np.testing.assert_array_equal(np.unique(dataset["E_real"]), [1.0, 11.0])
    if solver == "boundarymode":
        selected = load_fields(tmp_path, mode=1)
        np.testing.assert_array_equal(dataset["E_real"], selected["E_real"])
    else:
        with pytest.raises(FileNotFoundError, match="ParaView output not found"):
            load_fields(tmp_path, mode=1)


def test_missing_excitation_does_not_read_another(tmp_path: Path) -> None:
    _write_parallel_cycle(tmp_path / "paraview/driven/excitation_2/Cycle000001", mode=2)
    with pytest.raises(FileNotFoundError, match="excitation 1"):
        load_fields(tmp_path, excitation=1)


def test_mode_selection_ignores_driven_output(tmp_path: Path) -> None:
    for solver, value in (("driven", 9), ("eigenmode", 1)):
        _write_parallel_cycle(
            tmp_path / "paraview" / solver / "Cycle000001", mode=value
        )
    dataset = load_fields(tmp_path, mode=1)
    np.testing.assert_array_equal(np.unique(dataset["E_real"]), [1.0, 11.0])
