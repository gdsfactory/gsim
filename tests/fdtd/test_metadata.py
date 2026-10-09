"""Measured mesh bounds and requested grid settings for FDTD uploads."""

from __future__ import annotations

import json
from dataclasses import replace

import meshio
import numpy as np
import pytest

from gsim import fdtd
from gsim.fdtd import metadata as fdtd_metadata
from gsim.fdtd.config import FDTDConfig
from gsim.fdtd.models import FDTDConfigError, MeshGroup, MeshManifest, PortMeshGroup


def _manifest():
    return MeshManifest(
        volumes={"background": MeshGroup("background", 11, "SiO2", 0)},
        layers={"core": MeshGroup("core", 17, "Si", 1)},
        ports={"o1": PortMeshGroup("o1", "port_o1", 23, "core", (-1, 0, 0))},
    )


def _config(*, pml=17, source="eigenmode", halfspan=50, center=1500, cell=100):
    excitation = {
        "type": source,
        "center_wavelength": center,
        "wavelength_halfspan": halfspan,
        "num_wavelengths": 3,
    }
    if source == "eigenmode":
        excitation["default_port"] = "o1"
    else:
        excitation["dipole"] = {"position": [0, 0, 0], "current_axis": "x"}
    return FDTDConfig.model_validate(
        {
            "background_refractive_index": 1.5,
            "materials": {
                "SiO2": {"refractive_index": 1.5},
                "Si": {"refractive_index": 3.5},
            },
            "geometry": {
                "volumes": {
                    "background": {"phys_group": 11, "material": "SiO2", "priority": 0}
                },
                "layers": {"core": {"phys_group": 17, "material": "Si", "priority": 1}},
                "ports": {
                    "o1": {"phys_group": 23, "layer": "core", "normal": [-1, 0, 0]}
                },
            },
            "excitation": excitation,
            "grid": {"nanometers_per_cell": cell, "pml_cells": pml},
            "run": {"energy_decay_fraction": 1e-6, "max_wall_seconds": 3600},
        }
    )


def _mesh(tmp_path, extents=(200, 300, 400)):
    x, y, z = extents
    points = np.array(
        [
            [0, 0, 0],
            [x, 0, 0],
            [0, y, 0],
            [0, 0, z],
            [x / 2, 0, 0],
            [0, y / 2, 0],
            [0, 0, z / 2],
            [1e9, 1e9, 1e9],
        ],
        dtype=float,
    )
    mesh = meshio.Mesh(
        points,
        [("tetra", [[0, 1, 2, 3], [0, 4, 5, 6]]), ("triangle", [[0, 5, 6]])],
        cell_data={
            "gmsh:physical": [np.array([11, 17]), np.array([23])],
            "gmsh:geometrical": [np.array([1, 2]), np.array([3])],
        },
        field_data={"background": [11, 3], "core": [17, 3], "port_o1": [23, 2]},
    )
    path = tmp_path / "mesh.msh"
    meshio.write(path, mesh, file_format="gmsh22", binary=False)
    return path


@pytest.mark.parametrize(
    ("extents", "pml", "cell"),
    [((200, 300, 400), 17, 100), ((201, 301, 1601), 0, 20)],
)
def test_metadata_contains_measured_bounds_and_requested_grid(
    tmp_path, extents, pml, cell
):
    """Unused mesh vertices must not inflate the submitted physical domain."""
    metadata = fdtd_metadata.generate_metadata(
        _manifest(), _config(pml=pml, cell=cell), _mesh(tmp_path, extents)
    )
    assert metadata == {
        "schema_version": 1,
        "solver": "fdtd",
        "mesh": {"bounds_nm": {"min": [0, 0, 0], "max": list(extents)}},
        "grid": {"nanometers_per_cell": cell, "pml_cells": pml},
    }
    json.dumps(metadata, allow_nan=False)


def test_source_material_and_monitor_changes_do_not_expand_metadata(tmp_path):
    """Physics settings remain in the configuration rather than diagnostics."""
    mesh_path = _mesh(tmp_path)
    original = fdtd_metadata.generate_metadata(_manifest(), _config(), mesh_path)
    config = _config(source="dipole", halfspan=0).model_dump()
    config["materials"]["Si"]["refractive_index"] = 3.4
    config["monitors"] = [
        {
            "name": "top",
            "region_min": [0, 0, 100],
            "region_max": [200, 300, 100],
            "normal": "+z",
            "heatmap": {"wavelengths": [1500, 1560]},
        }
    ]
    metadata = fdtd_metadata.generate_metadata(
        _manifest(), FDTDConfig.model_validate(config), mesh_path
    )
    assert metadata == original


def test_metadata_rejects_config_and_manifest_group_mismatch(tmp_path):
    manifest = replace(
        _manifest(),
        layers={"core": replace(_manifest().layers["core"], physical_tag=99)},
    )
    with pytest.raises(FDTDConfigError, match="manifest"):
        fdtd_metadata.generate_metadata(manifest, _config(), _mesh(tmp_path))


def test_metadata_rejects_missing_physical_group(tmp_path):
    mesh_path = _mesh(tmp_path)
    mesh = meshio.read(mesh_path)
    del mesh.field_data["core"]
    meshio.write(mesh_path, mesh, file_format="gmsh22", binary=False)
    with pytest.raises(FDTDConfigError, match="physical group"):
        fdtd_metadata.generate_metadata(_manifest(), _config(), mesh_path)


def test_metadata_rejects_nonfinite_referenced_vertices(tmp_path):
    mesh_path = _mesh(tmp_path)
    mesh = meshio.read(mesh_path)
    mesh.points[1, 0] = np.nan
    meshio.write(mesh_path, mesh, file_format="gmsh22", binary=False)
    with pytest.raises(FDTDConfigError, match="finite"):
        fdtd_metadata.generate_metadata(_manifest(), _config(), mesh_path)


@pytest.mark.parametrize("historical", [False, True])
def test_write_emits_metadata_from_finalized_inputs(
    tmp_path, fdtd_pdk_module, historical
):
    if historical:
        from gsim.fdtd.simulation import ArtifactSimulation

        simulation = ArtifactSimulation(
            pdk=fdtd_pdk_module,
            default_port="o1",
            nanometers_per_cell=100,
            pml_cells=17,
        )
        simulation.geometry("straight", settings={"length": 2})
    else:
        simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
        simulation.source(port="o1")
        simulation.geometry("straight", settings={"length": 2})
        simulation.domain(
            x_bounds=(0, 2), y_bounds=(-1, 1), z_bounds=(-1, 1), pml_cells=17
        )
        simulation.solver(cell_size_nm=100)

    simulation.write(tmp_path)
    metadata = json.loads(tmp_path.joinpath("metadata.json").read_text())
    config = json.loads(tmp_path.joinpath("config.json").read_text())
    assert metadata["schema_version"] == 1
    assert metadata["solver"] == "fdtd"
    assert metadata["grid"] == config["grid"]
    assert set(metadata) == {"schema_version", "solver", "mesh", "grid"}
    if not historical:
        assert metadata["mesh"]["bounds_nm"] == {
            "min": [0, -1000, -1000],
            "max": [2000, 1000, 1000],
        }
