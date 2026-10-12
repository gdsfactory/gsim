"""Select solution fields when Palace appends mesh diagnostic cycles."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from gsim.palace.results import load_fields


def _write_cycle(directory: Path, association: str, arrays: dict[str, float]) -> None:
    """Write a small parallel VTK dataset with real point or cell arrays."""
    directory.mkdir(parents=True)
    mesh = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid()
    mesh.points = mesh.points.astype(float)
    mesh.cell_data["attribute"] = np.ones(mesh.n_cells)
    data = mesh.point_data if association == "point" else mesh.cell_data
    count = mesh.n_points if association == "point" else mesh.n_cells
    for name, value in arrays.items():
        data[name] = np.full(count, value, dtype=float)
    mesh.save(directory / "proc000000.vtu")

    def declarations(names) -> str:
        return "".join(
            f'<PDataArray type="Float64" Name="{name}" NumberOfComponents="1"/>'
            for name in names
        )

    (directory / "data.pvtu").write_text(
        '<?xml version="1.0"?>'
        '<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">'
        '<PUnstructuredGrid GhostLevel="0">'
        '<PPoints><PDataArray type="Float64" NumberOfComponents="3"/></PPoints>'
        f"<PPointData>{declarations(mesh.point_data)}</PPointData>"
        f"<PCellData>{declarations(mesh.cell_data)}</PCellData>"
        '<Piece Source="proc000000.vtu"/>'
        "</PUnstructuredGrid></VTKFile>"
    )


@pytest.mark.parametrize("diagnostic_association", ["point", "cell"])
@pytest.mark.parametrize("field_association", ["point", "cell"])
@pytest.mark.parametrize("boundary", [False, True])
def test_load_fields_skips_diagnostic_cycle(
    tmp_path: Path,
    diagnostic_association: str,
    field_association: str,
    boundary: bool,
) -> None:
    """Both old and new diagnostic layouts must select the latest solution."""
    solver = "driven_boundary" if boundary else "driven"
    directory = tmp_path / "paraview" / solver / "excitation_2"
    _write_cycle(directory / "Cycle000001", field_association, {"E_real": 1.0})
    _write_cycle(directory / "Cycle000002", field_association, {"E_real": 2.0})
    _write_cycle(
        directory / "Cycle000003",
        diagnostic_association,
        {"Indicator": 0.01, "Rank": 0.0},
    )

    fields = load_fields(tmp_path, excitation=2, boundary=boundary)
    np.testing.assert_array_equal(fields["E_real"], 2.0)

    # An explicit request must still allow inspecting a diagnostic cycle.
    diagnostic = load_fields(tmp_path, excitation=2, cycle=3, boundary=boundary)
    assert "Indicator" in diagnostic.array_names
    assert "E_real" not in diagnostic.array_names
