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
        domain_min_nm=(0, 0, 0),
        domain_max_nm=(200, 300, 400),
    )


def _config(
    *,
    pml=17,
    source="eigenmode",
    halfspan=50,
    center=1500,
    cell=100,
    bounds=((0, 0, 0), (200, 300, 400)),
):
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
            "grid": {
                "nanometers_per_cell": cell,
                "pml_cells": pml,
                "non_pml_bounds": {"min": bounds[0], "max": bounds[1]},
            },
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
    ("extents", "pml", "cell", "shape", "cell_count"),
    [
        ((200, 300, 400), 17, 100, [38, 39, 112], 165984),
        ((201, 301, 1601), 0, 20, [13, 18, 128], 29952),
    ],
)
def test_metadata_contains_measured_bounds_and_requested_grid(
    tmp_path, extents, pml, cell, shape, cell_count
):
    """Unused mesh vertices must not inflate the submitted physical domain."""
    metadata = fdtd_metadata.generate_metadata(
        _manifest(),
        _config(pml=pml, cell=cell, bounds=((0, 0, 0), extents)),
        _mesh(tmp_path, extents),
    )
    assert metadata["schema_version"] == 1
    assert metadata["solver"] == "fdtd"
    assert metadata["mesh"] == {"bounds_nm": {"min": [0, 0, 0], "max": list(extents)}}
    assert metadata["grid"]["nanometers_per_cell"] == cell
    assert metadata["grid"]["pml_cells"] == pml
    assert metadata["grid"]["cell_counts"]["total_cells_xyz"] == shape
    assert metadata["grid"]["cell_counts"]["total_cells"] == cell_count
    json.dumps(metadata, allow_nan=False)


def test_cell_counts_use_non_pml_domain_and_include_aligned_pml_and_guards(tmp_path):
    metadata = fdtd_metadata.generate_metadata(
        _manifest(),
        _config(pml=1, bounds=((-100, -200, -300), (300, 400, 500))),
        _mesh(tmp_path),
    )
    assert metadata["mesh"]["bounds_nm"]["max"] == [200, 300, 400]
    assert metadata["grid"]["cell_counts"] == {
        "scope": "client_estimate",
        "layout": "fdtd-z16-v1",
        "interior_cells_xyz": [4, 6, 8],
        "alignment_cells_xyz": [0, 0, 8],
        "pml_cells_per_face_xyz": [1, 1, 16],
        "guard_cells_per_face_xyz": [1, 1, 16],
        "total_cells_xyz": [8, 10, 80],
        "total_cells": 6400,
    }


@pytest.mark.parametrize(
    ("extents", "interior", "alignment", "total", "cell_count"),
    [
        (
            (0.1 * 3 * 1000, 0.1 * 3 * 1000, 0.1 * 3 * 1600),
            [10, 10, 16],
            [0, 0, 0],
            [46, 46, 112],
            236992,
        ),
        (
            (np.nextafter(300.0, 0), 300.0, np.nextafter(480.0, 0)),
            [10, 10, 16],
            [0, 0, 0],
            [46, 46, 112],
            236992,
        ),
        (
            (300.001, 300.1, 480.001),
            [11, 11, 17],
            [0, 0, 15],
            [47, 47, 128],
            282752,
        ),
    ],
)
def test_cell_counts_normalize_integer_noise_and_preserve_fractional_extents(
    tmp_path, extents, interior, alignment, total, cell_count
):
    metadata = fdtd_metadata.generate_metadata(
        _manifest(),
        _config(cell=30, pml=17, bounds=((0, 0, 0), extents)),
        _mesh(tmp_path),
    )
    assert metadata["grid"]["cell_counts"] == {
        "scope": "client_estimate",
        "layout": "fdtd-z16-v1",
        "interior_cells_xyz": interior,
        "alignment_cells_xyz": alignment,
        "pml_cells_per_face_xyz": [17, 17, 32],
        "guard_cells_per_face_xyz": [1, 1, 16],
        "total_cells_xyz": total,
        "total_cells": cell_count,
    }


@pytest.mark.parametrize(
    ("domain_size", "cell_count"),
    [
        ((998, 998, 64), 96_000_000),
        ((998, 998, 64.1), 112_000_000),
        ((2498, 2498, 80), 700_000_000),
        ((2498, 2498, 80.1), 800_000_000),
    ],
)
def test_cell_counts_rounding_near_admission_boundaries(
    tmp_path, domain_size, cell_count
):
    metadata = fdtd_metadata.generate_metadata(
        _manifest(),
        _config(cell=1, pml=0, bounds=((0, 0, 0), domain_size)),
        _mesh(tmp_path, extents=(1, 1, 1)),
    )
    assert metadata["grid"]["cell_counts"]["total_cells"] == cell_count


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
    assert {key: metadata["grid"][key] for key in config["grid"]} == config["grid"]
    assert metadata["grid"]["cell_counts"]["total_cells"] > 0
    assert set(metadata) == {"schema_version", "solver", "mesh", "grid"}
    if not historical:
        assert metadata["mesh"]["bounds_nm"] == {
            "min": [0, -1000, -1000],
            "max": [2000, 1000, 1000],
        }
