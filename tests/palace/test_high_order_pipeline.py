"""Verify high-order mesh/config outputs through the actual Gmsh pipeline."""

import json
from unittest.mock import Mock

import gdsfactory as gf
import gmsh
import meshio
import numpy as np
import pytest

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh.generator import generate_mesh
from gsim.palace.models import BoundaryModeConfig, CrossSectionPlaneConfig, DrivenConfig


@pytest.fixture
def dielectric_device():
    gf.gpdk.PDK.activate()
    component = gf.Component()
    for index in [1, 2]:
        component.add_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], layer=(index, 0))
    stack = LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=(index, 0),
                zmin=index - 1,
                zmax=index,
                thickness=1,
                material=name,
                layer_type="dielectric",
            )
            for index, name in enumerate(["lower", "upper"], 1)
        },
        materials={
            "lower": {"permittivity": [4, 5, 6]},
            "upper": {"permittivity": [2, 2, 2]},
            "air": {"permittivity": 1},
        },
        dielectrics=[],
    )
    return component, stack


@pytest.mark.parametrize("kind", ["volume", "x", "y"])
@pytest.mark.parametrize("optimize", [False, True])
def test_high_order_pipeline_writes_quadratic_mesh_and_matching_config(
    monkeypatch, dielectric_device, tmp_path, kind, optimize
):
    component, stack = dielectric_device
    gui = Mock()
    monkeypatch.setattr(gmsh.fltk, "run", gui)
    result = generate_mesh(
        component,
        stack,
        [],
        tmp_path,
        simulation_type="driven" if kind == "volume" else "boundarymode",
        cross_section=None
        if kind == "volume"
        else CrossSectionPlaneConfig(axis=kind, value=1),
        driven_config=DrivenConfig(fmin=1e9, fmax=2e9, num_points=3),
        boundary_mode_config=BoundaryModeConfig(freq=1e9, num_modes=1),
        high_order_elements=True,
        high_order_order=2,
        high_order_optimize=optimize,
        max_mesh_size=2,
        refined_mesh_size=1,
        margin_x=1,
        margin_y=1,
        air_margin=1,
        airbox_material="test-background" if kind == "y" else "air",
        show_gui=True,
        verbosity=0,
    )
    gui.assert_called_once()
    assert result.config_path is not None
    config = json.loads(result.config_path.read_text())
    assert config["Problem"]["Type"].lower() == (
        "driven" if kind == "volume" else "boundarymode"
    )
    mesh = meshio.read(result.mesh_path)
    assert any(
        block.type == ("tetra10" if kind == "volume" else "triangle6")
        for block in mesh.cells
    )
    assert np.all(np.isfinite(mesh.points))
    assert result.mesh_stats["nodes"] == len(mesh.points)
    if kind != "volume":
        assert any(
            name.startswith("interface_")
            for name in result.groups["interface_surfaces"]
        )
    if kind == "y":
        assert stack.materials["test-background"]["permittivity"] == 1


@pytest.mark.parametrize("kind", ["missing-section", "ports", "refinement-box"])
def test_mesh_pipeline_rejects_incompatible_native_requests(
    dielectric_device, tmp_path, kind
):
    component, stack = dielectric_device
    options = {"simulation_type": "boundarymode", "cross_section": None}
    ports = []
    if kind == "ports":
        ports = [Mock()]
    elif kind == "refinement-box":
        options.update(simulation_type="driven", refinement_boxes=[(-1, -1, 1, 1, 0.5)])
    with pytest.raises(ValueError):
        generate_mesh(component, stack, ports, tmp_path, verbosity=0, **options)
