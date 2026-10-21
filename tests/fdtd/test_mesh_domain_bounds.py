"""Disconnected sidewalls must stay inside the declared FDTD domain."""

import json
from dataclasses import replace
from math import radians, tan

import gdsfactory as gf
import meshio
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from gsim import fdtd
from gsim.fdtd import mesh_loft
from gsim.fdtd.mesh import generate_mesh
from gsim.fdtd.mesh_geometry import sidewall_slice_count


@pytest.mark.parametrize("angle", [0, 11, -11])
@pytest.mark.parametrize("stepped", [False, True])
def test_disconnected_sidewalls_are_bounded_before_port_tagging(
    tmp_path, fdtd_pdk_module, monkeypatch, angle, stepped
):
    if stepped:

        def incompatible(*_args):
            raise mesh_loft.LoftIncompatibleError("Exercise the stepped fallback.")

        monkeypatch.setattr(mesh_loft, "loft_section_polygons", incompatible)
    fdtd_pdk_module.PDK.activate()
    level = fdtd_pdk_module.PDK.layer_stack.layers["core"]
    level.sidewall_angle = angle
    level.thickness = 0.161
    component = gf.Component()
    # Public synthetic right-angle guide, plus disconnected slab pieces at
    # the -x input and +y output faces. Only the guide owns port endpoints.
    component.add_polygon(
        [(0, -0.25), (2.25, -0.25), (2.25, 2), (1.75, 2), (1.75, 0.25), (0, 0.25)],
        layer=(1, 0),
    )
    component.add_polygon([(0, 0.6), (1.2, 0.6), (1.2, 1), (0, 1)], layer=(1, 0))
    component.add_polygon([(0.7, 1.2), (1.1, 1.2), (1.1, 2), (0.7, 2)], layer=(1, 0))
    component.add_port("o1", center=(0, 0), width=0.5, orientation=180, layer=(1, 0))
    component.add_port("o2", center=(2, 2), width=0.5, orientation=90, layer=(1, 0))
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, mesh_size_nm=500)
    simulation.geometry(component)
    simulation.source(port="o1")
    artifacts = simulation.write(tmp_path)
    mesh = meshio.read(artifacts.mesh_path)
    config = json.loads(artifacts.config_path.read_text())
    bounds = config["grid"]["non_pml_bounds"]
    assert bounds["min"][0] == 0
    assert bounds["max"][1] == 2000
    assert np.all(mesh.points >= np.array(bounds["min"]) - 1e-7)
    assert np.all(mesh.points <= np.array(bounds["max"]) + 1e-7)
    tetrahedra = mesh.points[mesh.cells_dict["tetra"]]
    determinants = np.linalg.det(tetrahedra[:, 1:] - tetrahedra[:, :1])
    assert np.all(np.abs(determinants) > 1e-6)
    tags = mesh.cell_data_dict["gmsh:physical"]["tetra"]
    core = mesh.field_data["core"][0]
    points = mesh.points[np.unique(mesh.cells_dict["tetra"][tags == core])]
    # Transverse sidewalls away from the clipped end retain the exact taper.
    slab = points[(points[:, 0] < 1500) & (points[:, 1] > 400) & (points[:, 1] < 1100)]
    offset = 0.161 * 500 * tan(radians(angle))
    if stepped:
        slices = sidewall_slice_count(
            simulation.resolved.layers["core"],
            simulation.geometry.geometry_tolerance_nm,
        )
        offset *= 1 - 1 / slices
    bottom = slab[np.isclose(slab[:, 2], 0)]
    top = slab[np.isclose(slab[:, 2], 161)]
    assert bottom[:, 1].min() == pytest.approx(600 - offset, abs=1e-5)
    assert top[:, 1].min() == pytest.approx(600 + offset, abs=1e-5)
    for name, axis, coordinate in [("o1", 0, 0), ("o2", 1, 2000)]:
        tag = mesh.field_data[f"port_{name}"][0]
        triangles = mesh.cells_dict["triangle"][
            mesh.cell_data_dict["gmsh:physical"]["triangle"] == tag
        ]
        assert len(triangles) > 0
        assert mesh.points[np.unique(triangles), axis] == pytest.approx(coordinate)


def test_same_plane_ports_and_interior_holed_solid_are_preserved(
    tmp_path, fdtd_pdk_module
):
    fdtd_pdk_module.PDK.activate()
    component = gf.Component()
    for index, y in enumerate([-1, 1]):
        component.add_polygon(
            [(0, y - 0.25), (2, y - 0.25), (2, y + 0.25), (0, y + 0.25)],
            layer=(1, 0),
        )
        for side, x, orientation in [("in", 0, 180), ("out", 2, 0)]:
            component.add_port(
                f"{side}{index}",
                center=(x, y),
                width=0.5,
                orientation=orientation,
                layer=(1, 0),
            )
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    resolved = simulation.geometry(component)
    ring = Polygon(
        [(0.7, -0.3), (1.3, -0.3), (1.3, 0.3), (0.7, 0.3)],
        holes=[[(0.85, -0.15), (1.15, -0.15), (1.15, 0.15), (0.85, 0.15)]],
    )
    layer = resolved.layers["core"]
    layer = replace(layer, geometry=MultiPolygon([*layer.geometry.geoms, ring]))
    resolved = replace(resolved, layers={"core": layer})
    path = tmp_path / "mesh.msh"
    manifest = generate_mesh(
        resolved,
        path,
        background_material="SiO2",
        padding=0.25,
        mesh_size_nm=500,
        geometry_tolerance_nm=10,
    )
    assert set(manifest.ports) == {"in0", "in1", "out0", "out1"}
    mesh = meshio.read(path)
    tags = mesh.cell_data_dict["gmsh:physical"]["tetra"]
    tetrahedra = mesh.points[
        mesh.cells_dict["tetra"][tags == mesh.field_data["core"][0]]
    ]
    volumes = np.abs(np.linalg.det(tetrahedra[:, 1:] - tetrahedra[:, :1])) / 6
    assert np.all(volumes > 1e-6)
    slices = sidewall_slice_count(layer, 10)
    ring_volume = sum(
        ring.buffer(
            (0.5 - (index + 0.5) / slices) * 0.22 * tan(radians(10)),
            join_style=2,
        ).area
        * 0.22
        / slices
        for index in range(slices)
    )
    # Both guides retain their full tapered cross-section, and the holed
    # interior fallback retains its analytical volume without domain cuts.
    assert volumes.sum() == pytest.approx((2 * 2 * 0.5 * 0.22 + ring_volume) * 1e9)
