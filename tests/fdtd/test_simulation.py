"""Behavioral tests for the public GDSFactory FDTD artifact workflow."""

from __future__ import annotations

import json

import meshio
import numpy as np
import pytest

from gsim import fdtd
from gsim.fdtd.models import FDTDConfigError, FDTDGeometryError


def _physical_group_points(mesh: meshio.Mesh, name: str, cell_type: str) -> np.ndarray:
    """Return mesh points used by one named physical group."""
    physical_tag = mesh.field_data[name][0]
    cells = mesh.cells_dict[cell_type]
    tags = mesh.cell_data_dict["gmsh:physical"][cell_type]
    point_indices = np.unique(cells[tags == physical_tag].ravel())
    return mesh.points[point_indices]


def _internal_horizontal_boundary_faces(mesh: meshio.Mesh, name: str) -> np.ndarray:
    """Return horizontal material-boundary faces away from its z extrema."""
    physical_tag = mesh.field_data[name][0]
    tetrahedra = mesh.cells_dict["tetra"]
    tags = mesh.cell_data_dict["gmsh:physical"]["tetra"]
    tetrahedra = tetrahedra[tags == physical_tag]
    faces = np.concatenate(
        [
            tetrahedra[:, [0, 1, 2]],
            tetrahedra[:, [0, 1, 3]],
            tetrahedra[:, [0, 2, 3]],
            tetrahedra[:, [1, 2, 3]],
        ]
    )
    sorted_faces = np.sort(faces, axis=1)
    _, unique_indices, counts = np.unique(
        sorted_faces,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    boundary_faces = faces[unique_indices[counts == 1]]
    face_z = mesh.points[boundary_faces, 2]
    material_z = mesh.points[np.unique(tetrahedra), 2]
    is_horizontal = np.ptp(face_z, axis=1) < 1e-6
    is_internal = (face_z[:, 0] > material_z.min() + 1e-6) & (
        face_z[:, 0] < material_z.max() - 1e-6
    )
    return boundary_faces[is_horizontal & is_internal]


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
        default_port="o1",
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
    silicon_table = document["materials"]["Si"]["dispersion"]["table"]
    silicon_index = silicon_table["wavelength_nm"].index(1550.0)
    assert silicon_table["n"][silicon_index] == pytest.approx(3.4757)
    assert set(silicon_table["k"]) == {0.0}
    silica = document["materials"]["SiO2"]["dispersion"]
    assert silica["wavelength_range_nm"] == [1500.0, 1600.0]
    assert silica["drude_lorentz"]["eps_inf"] == pytest.approx(1.2648942847)
    assert len(silica["drude_lorentz"]["lorentz"]) == 2
    assert document["background_refractive_index"] == pytest.approx(1.4440235342)
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
    assert port_points[:, 1].min() == pytest.approx(-269.395968, abs=1e-3)
    assert port_points[:, 1].max() == pytest.approx(269.395968, abs=1e-3)
    assert port_points[:, 2].min() == pytest.approx(0)
    assert port_points[:, 2].max() == pytest.approx(220)
    assert len(_internal_horizontal_boundary_faces(mesh, "core")) == 0

    mesh_header = artifacts.mesh_path.read_text(encoding="utf8").splitlines()
    assert mesh_header[mesh_header.index("$MeshFormat") + 1] == "2.2 0 8"


def test_write_requires_geometry(tmp_path) -> None:
    with pytest.raises(FDTDGeometryError, match=r"Call Simulation\.geometry"):
        fdtd.Simulation().write(tmp_path)


def test_explicit_domain_bounds_set_background_box(
    tmp_path,
    fdtd_pdk_module,
) -> None:
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module,
        default_port="o1",
        mesh_size_nm=750,
    )
    simulation.domain(
        x_bounds=(0.0, 2.0),
        y_bounds=(-0.5, 0.5),
        z_bounds=(-0.4, 0.6),
    )
    simulation.geometry("straight", settings={"length": 2.0})

    artifacts = simulation.write(tmp_path)
    mesh = meshio.read(artifacts.mesh_path)
    background_points = _physical_group_points(mesh, "background", "tetra")

    assert background_points.min(axis=0) == pytest.approx([0.0, -500.0, -400.0])
    assert background_points.max(axis=0) == pytest.approx([2000.0, 500.0, 600.0])


def test_explicit_domain_bounds_must_contain_geometry(
    tmp_path,
    fdtd_pdk_module,
) -> None:
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module,
        default_port="o1",
        mesh_size_nm=750,
    )
    simulation.domain(x_bounds=(0.1, 2.0))
    simulation.geometry("straight", settings={"length": 2.0})

    with pytest.raises(FDTDGeometryError, match=r"domain\.x_bounds.*geometry"):
        simulation.write(tmp_path)


def test_explicit_domain_bounds_must_contain_source(
    tmp_path,
    fdtd_pdk_module,
) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, mesh_size_nm=750)
    simulation.domain(z_bounds=(-0.4, 0.6))
    simulation.geometry("straight", settings={"length": 2.0})
    simulation.source = fdtd.GaussianBeamSource(
        center_um=(1.0, 0.0, 0.8),
        size_um=(1.0, 1.0, 0.0),
        aperture_normal="-z",
        propagation_direction=(0.0, 0.0, -1.0),
        e_polarization=(0.0, 1.0, 0.0),
        focal_point_um=(1.0, 0.0, 0.11),
        waist_radius_um=0.5,
    )

    with pytest.raises(FDTDConfigError, match=r"source aperture.*z-extent"):
        simulation.write(tmp_path)


def test_explicit_domain_bounds_must_contain_monitor(
    tmp_path,
    fdtd_pdk_module,
) -> None:
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module,
        default_port="o1",
        mesh_size_nm=750,
    )
    simulation.domain(z_bounds=(-0.4, 0.6))
    simulation.geometry("straight", settings={"length": 2.0})
    simulation.monitors.add_plane(
        "cross",
        center_um=(1.0, 0.0, 0.1),
        size_um=(2.0, 0.0, 2.0),
        normal="+y",
        flux=False,
        heatmap=fdtd.Heatmap(quantity="abs_e", wavelengths_um=[1.55]),
    )

    with pytest.raises(FDTDConfigError, match=r"Monitor 'cross'.*z-extent"):
        simulation.write(tmp_path)


def test_coarse_transfer_mesh_preserves_ten_nm_cad_feature(
    tmp_path,
    fdtd_pdk_module,
) -> None:
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module,
        default_port="o1",
        mesh_size_nm=1000,
    )
    simulation.geometry("ten_nm_feature")

    artifacts = simulation.write(tmp_path)
    mesh = meshio.read(artifacts.mesh_path)
    material_points = _physical_group_points(mesh, "thin_core", "tetra")
    feature_points = material_points[material_points[:, 1] > 300]

    assert len(feature_points) >= 8
    assert feature_points[:, 0].min() == pytest.approx(500)
    assert feature_points[:, 0].max() == pytest.approx(510)
    assert feature_points[:, 1].min() == pytest.approx(400)
    assert feature_points[:, 1].max() == pytest.approx(500)


def test_write_requires_explicit_source_port(tmp_path, fdtd_pdk_module) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, mesh_size_nm=750)
    simulation.geometry("straight", settings={"length": 2.0})

    with pytest.raises(FDTDConfigError, match="requires an explicit port"):
        simulation.write(tmp_path)

    assert not (tmp_path / "mesh.msh").exists()


def test_vertical_port_becomes_fiber_monitor_not_material_port(
    tmp_path,
    fdtd_pdk_module,
) -> None:
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module,
        default_port="o1",
        mesh_size_nm=750,
        background_padding_um=0.25,
    )
    simulation.geometry("vertical_coupler")

    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text(encoding="utf8"))
    mesh = meshio.read(artifacts.mesh_path)

    assert document["excitation"]["type"] == "eigenmode"
    assert document["excitation"]["default_port"] == "o1"
    assert set(document["geometry"]["ports"]) == {"o1"}
    assert "port_o2" not in mesh.field_data
    assert document["monitors"][0]["name"] == "o2"
    assert document["monitors"][0]["normal"] == "+z"
    assert document["monitors"][0]["fiber_mode"]["e_polarization"] == [
        -0.0,
        1.0,
        0.0,
    ]


def test_vertical_default_port_uses_gaussian_beam(
    tmp_path,
    fdtd_pdk_module,
) -> None:
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module,
        default_port="o2",
        mesh_size_nm=750,
        background_padding_um=0.25,
    )
    simulation.source(
        vertical_monitor_heatmap=fdtd.Heatmap(
            quantity="abs_e",
            wavelengths_um=[1.55],
        )
    )
    simulation.geometry("vertical_coupler")

    artifacts = simulation.write(tmp_path)
    document = json.loads(artifacts.config_path.read_text(encoding="utf8"))

    assert document["excitation"]["type"] == "gaussian_beam"
    assert "default_port" not in document["excitation"]
    assert document["excitation"]["gaussian_beam"]["aperture_normal"] == "-z"
    assert document["excitation"]["gaussian_beam"]["propagation_direction"] == [
        0.0,
        0.0,
        -1.0,
    ]
    assert document["monitors"][0]["heatmap"] == {
        "quantity": "abs_e",
        "wavelengths": [1550.0],
    }


def test_write_restores_options_from_existing_gmsh_session(
    tmp_path,
    fdtd_pdk_module,
) -> None:
    import gmsh

    original_options = {
        "Mesh.Algorithm": 6.0,
        "Mesh.Algorithm3D": 4.0,
        "Mesh.Binary": 1.0,
        "Mesh.ElementOrder": 2.0,
        "Mesh.MeshSizeExtendFromBoundary": 1.0,
        "Mesh.MeshSizeFromCurvature": 1.0,
        "Mesh.MeshSizeFromPoints": 1.0,
        "Mesh.MeshSizeMax": 34.5,
        "Mesh.MeshSizeMin": 12.5,
        "Mesh.MshFileVersion": 4.1,
        "Mesh.SaveAll": 1.0,
    }
    gmsh.initialize()
    try:
        for option_name, value in original_options.items():
            gmsh.option.setNumber(option_name, value)

        simulation = fdtd.Simulation(
            pdk=fdtd_pdk_module,
            default_port="o1",
            mesh_size_nm=750,
            background_padding_um=0.25,
        )
        simulation.geometry("straight", settings={"length": 2.0})
        simulation.write(tmp_path)

        assert gmsh.isInitialized()
        for option_name, expected in original_options.items():
            assert gmsh.option.getNumber(option_name) == pytest.approx(expected)
    finally:
        gmsh.clear()
        gmsh.finalize()
