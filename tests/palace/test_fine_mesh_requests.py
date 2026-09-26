"""Native 2D refinement requests and recovery from incomplete gmsh metadata."""

from __future__ import annotations

from types import SimpleNamespace

import gdsfactory as gf
import gmsh
import meshio
import pytest

from gsim.common import Layer, LayerStack
from gsim.palace import BoundaryModeSim
from gsim.palace.mesh import generator


def test_fine_requests_without_active_gmsh(monkeypatch):
    monkeypatch.setattr(gmsh, "isInitialized", lambda: False)
    assert generator._collect_fine_size_requests({}, LayerStack(), 0.1) == []


def test_fine_requests_skip_failed_surfaces_and_domain_walls(monkeypatch):
    monkeypatch.setattr(gmsh, "isInitialized", lambda: True)
    stack = SimpleNamespace(
        layers={
            "junction": SimpleNamespace(mesh_resolution=0.01),
            "isolated": SimpleNamespace(mesh_resolution=0.02),
        }
    )
    groups = {
        "volumes": {"junction": {"tags": [10, 11]}, "isolated": {"tags": [12]}},
        "interface_surfaces": {"pn": {"tags": [2, 3]}},
    }

    def boundary(entities, **_kwargs):
        surface = entities[0][1]
        if surface == 10:
            raise RuntimeError("surface was removed")
        if surface == 12:
            return [(1, 8), (1, 9)]
        return [(1, 9), (1, 3), (1, 2), (1, 3), (0, 2)]

    monkeypatch.setattr(gmsh.model, "getBoundary", boundary)
    assert generator._collect_fine_size_requests(groups, stack, 0.1) == [([2, 3], 0.01)]


def _simulation(tmp_path):
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon([(0, 0), (2, 0), (2, 1), (0, 1)], layer=(1, 0))
    stack = LayerStack()
    stack.layers["core"] = Layer(
        name="core",
        gds_layer=(1, 0),
        zmin=0,
        zmax=0.2,
        thickness=0.2,
        material="silicon",
        layer_type="dielectric",
        mesh_resolution=0.05,
    )
    sim = BoundaryModeSim()
    sim.set_geometry(component)
    sim.set_stack(stack)
    sim.set_output_dir(tmp_path)
    sim.set_cross_section("x=1")
    sim.set_airbox(margin_x=0.5, margin_y=0.5, z_above=0.5, z_below=0.5)
    return sim


def test_fine_mesh_survives_unavailable_curve_bounds(monkeypatch, tmp_path):
    sim = _simulation(tmp_path)
    original_collect = generator._collect_fine_size_requests
    original_bounds = gmsh.model.getBoundingBox
    original_refinement = generator.gmsh_utils.setup_mesh_refinement
    requests = []

    def missing_curve_bounds(dim, tag):
        if dim == 1:
            raise RuntimeError("curve bounds unavailable")
        return original_bounds(dim, tag)

    def collect(*args):
        result = original_collect(*args)
        assert result
        monkeypatch.setattr(gmsh.model, "getBoundingBox", missing_curve_bounds)
        return result

    def refine(curves, minimum, maximum, **kwargs):
        requests.append((curves, minimum, kwargs["sampling"]))
        return original_refinement(curves, minimum, maximum, **kwargs)

    monkeypatch.setattr(generator, "_collect_fine_size_requests", collect)
    monkeypatch.setattr(generator.gmsh_utils, "setup_mesh_refinement", refine)
    sim.mesh(refined_mesh_size=0.2, max_mesh_size=0.5, verbose=False)
    assert any(size == 0.05 and sampling == 2000 for _, size, sampling in requests)
    mesh = meshio.read(tmp_path / "palace.msh")
    assert mesh.cells_dict["triangle"].size > 0
    assert "core" in mesh.field_data


def test_homogeneous_section_uses_global_mesh_sizes(monkeypatch, tmp_path):
    sim = _simulation(tmp_path)
    sim.stack.layers["core"].mesh_resolution = "medium"
    sim.set_airbox(margin_x=0, margin_y=0, z_above=0, z_below=0)
    configured_sizes = {}
    original_option = gmsh.option.setNumber

    def set_option(name, value):
        configured_sizes[name] = value
        return original_option(name, value)

    monkeypatch.setattr(gmsh.option, "setNumber", set_option)
    sim.mesh(refined_mesh_size=0.2, max_mesh_size=0.5, verbose=False)
    assert configured_sizes["Mesh.MeshSizeMin"] == pytest.approx(0.2)
    assert configured_sizes["Mesh.MeshSizeMax"] == pytest.approx(0.5)
    mesh = meshio.read(tmp_path / "palace.msh")
    assert mesh.cells_dict["triangle"].size > 0
    assert "core" in mesh.field_data
