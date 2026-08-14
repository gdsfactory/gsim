"""Behavioral tests for the public FDTD artifact workflow."""

from __future__ import annotations

import json

import meshio
import numpy as np
import pytest

from gsim import fdtd
from gsim.fdtd.models import FDTDGeometryError


def _physical_group_points(mesh: meshio.Mesh, name: str, cell_type: str) -> np.ndarray:
    """Return mesh points used by one named physical group."""
    physical_tag = mesh.field_data[name][0]
    cells = mesh.cells_dict[cell_type]
    tags = mesh.cell_data_dict["gmsh:physical"][cell_type]
    point_indices = np.unique(cells[tags == physical_tag].ravel())
    return mesh.points[point_indices]


def test_write_uses_project_material_then_fallback_and_valid_mesh(
    tmp_path,
    fdtd_pdk_module,
) -> None:
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module,
        mesh_size_nm=750,
        background_padding_um=0.25,
        pml_cells=16,
        num_wavelengths=3,
    )
    resolved = simulation.geometry("straight", settings={"length": 2.0})

    assert resolved.materials["Si"].source == "project"
    assert resolved.materials["Si"].refractive_index == pytest.approx(3.4757)

    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text(encoding="utf8"))
    mesh = meshio.read(artifacts.mesh_path)

    assert document["schema_version"] == 1
    assert document["mesh_file"] == "mesh.msh"
    assert document["length_scale_meters"] == 1e-9
    assert document["materials"]["Si"]["refractive_index"] == pytest.approx(3.4757)
    assert document["materials"]["SiO2"]["refractive_index"] == pytest.approx(
        1.4440236217
    )
    assert document["geometry"]["ports"]["o1"]["normal"] == [-1, 0, 0]
    assert document["geometry"]["ports"]["o2"]["normal"] == [1, 0, 0]

    expected_names = {"background", "core", "port_o1", "port_o2"}
    assert set(mesh.field_data) == expected_names
    assert {block.type for block in mesh.cells} == {"triangle", "tetra"}
    for group in artifacts.manifest.volumes.values():
        assert mesh.field_data[group.name].tolist() == [group.physical_tag, 3]
    for group in artifacts.manifest.layers.values():
        assert mesh.field_data[group.name].tolist() == [group.physical_tag, 3]
    for port in artifacts.manifest.ports.values():
        assert mesh.field_data[port.physical_name].tolist() == [
            port.physical_tag,
            2,
        ]

    port_points = _physical_group_points(mesh, "port_o1", "triangle")
    assert port_points[:, 0] == pytest.approx(0)
    assert port_points[:, 1].min() == pytest.approx(-269.39597, abs=1e-3)
    assert port_points[:, 1].max() == pytest.approx(269.39597, abs=1e-3)
    assert port_points[:, 2].min() == pytest.approx(0)
    assert port_points[:, 2].max() == pytest.approx(220)

    mesh_header = artifacts.mesh_path.read_text(encoding="utf8").splitlines()
    assert mesh_header[mesh_header.index("$MeshFormat") + 1] == "2.2 0 8"


def test_write_requires_geometry(tmp_path) -> None:
    with pytest.raises(FDTDGeometryError, match=r"Call Simulation\.geometry"):
        fdtd.Simulation().write(tmp_path)
