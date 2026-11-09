"""Explicit bounds trim material without a separate clipping switch."""

import json
import warnings
from dataclasses import replace

import gdsfactory as gf
import meshio
import numpy as np
import pytest
from gdsfactory.technology import LayerLevel, LogicalLayer

from gsim import fdtd
from gsim.fdtd.domain import background_bounds_nm
from gsim.fdtd.mesh import generate_mesh


@pytest.mark.parametrize("legacy", [True, False])
def test_legacy_calls_and_saved_domains_migrate(legacy):
    with pytest.warns(DeprecationWarning, match="ignored"):
        domain = fdtd.Domain.model_validate(
            {"clip_geometry": legacy, "z_bounds": [-0.1, 0.2]}
        )
    with pytest.warns(DeprecationWarning, match="ignored"):
        domain(clip_geometry=legacy)
    with pytest.warns(DeprecationWarning, match="ignored"):
        domain.clip_geometry = legacy
    with pytest.warns(DeprecationWarning, match="ignored"):
        assert domain.clip_geometry is True
    exported = domain.model_dump()
    assert "clip_geometry" not in exported
    assert "clip_geometry" not in domain.model_json_schema()["properties"]
    assert fdtd.Domain.model_validate_json(domain.model_dump_json()) == domain
    with pytest.warns(DeprecationWarning, match="ignored"):
        simulation = fdtd.Simulation(domain={**exported, "clip_geometry": legacy})
    assert simulation.domain == domain


def _one_port_component(module):
    module.PDK.activate()
    module.PDK.layer_stack.layers["core"].sidewall_angle = 0
    component = gf.Component()
    component.add_polygon([(0, -0.25), (2, -0.25), (2, 0.25), (0, 0.25)], layer=(1, 0))
    component.add_port("o1", center=(0, 0), width=0.5, orientation=180, layer=(1, 0))
    return component


def test_bounded_mesh_and_preview_share_clipped_geometry(tmp_path, fdtd_pdk_module):
    from matplotlib import pyplot as plt

    component = _one_port_component(fdtd_pdk_module)
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.domain(x_bounds=(0, 1), y_bounds=(-0.4, 0.4), z_bounds=(0.05, 0.15))
    simulation.source(port="o1")
    with pytest.warns(fdtd.GeometryClippingWarning, match=r"core \(x/z\)"):
        simulation.geometry(component, mesh_size_nm=600)
    assert simulation.resolved.clipped_layers == {"core": ("x", "z")}
    assert simulation.resolved.ports["o1"].z_bounds == pytest.approx((0.05, 0.15))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", fdtd.GeometryClippingWarning)
        artifacts = simulation.write(tmp_path)
        axes = simulation.plot_2d_static(axis="z")
        plt.close(axes.figure)
    assert not [
        w for w in caught if issubclass(w.category, fdtd.GeometryClippingWarning)
    ]
    config = json.loads(artifacts.config_path.read_text())
    assert config["grid"]["non_pml_bounds"] == {
        "min": [0, -400, 50],
        "max": [1000, 400, 150],
    }
    mesh = meshio.read(artifacts.mesh_path)
    assert mesh.points.min(axis=0) == pytest.approx([0, -400, 50])
    assert mesh.points.max(axis=0) == pytest.approx([1000, 400, 150])
    tetrahedra = mesh.points[mesh.cells_dict["tetra"]]
    assert np.all(np.abs(np.linalg.det(tetrahedra[:, 1:] - tetrahedra[:, :1])) > 1e-6)
    assert {"core", "background", "port_o1"} <= set(mesh.field_data)
    port_tag = mesh.field_data["port_o1"][0]
    faces = mesh.cells_dict["triangle"][
        mesh.cell_data_dict["gmsh:physical"]["triangle"] == port_tag
    ]
    vertices = mesh.points[faces]
    area = (
        np.linalg.norm(
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=1,
        ).sum()
        / 2
    )
    assert area == pytest.approx(500 * 100)
    assert component.dxsize == 2  # The PDK mask was not modified.


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_wholly_omitted_material_warns_before_lookup(fdtd_pdk_module, axis):
    component = _one_port_component(fdtd_pdk_module)
    layer = LayerLevel(layer=(9, 0), zmin=0, thickness=0.2, material="unknown")
    if axis == "z":
        layer.zmin = 3
    x, y = (3, 0) if axis == "x" else (0.5, 3) if axis == "y" else (0.5, 0)
    component.add_polygon(
        [(x, y), (x + 0.2, y), (x + 0.2, y + 0.2), (x, y + 0.2)], layer=(9, 0)
    )
    fdtd_pdk_module.PDK.layer_stack.layers["remote"] = layer
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.domain(**{f"{axis}_bounds": (-0.5, 2.5)})
    with pytest.warns(fdtd.GeometryClippingWarning, match=rf"remote \({axis}\)"):
        simulation.geometry(component)
    assert "remote" not in simulation.resolved.layers
    assert "unknown" not in simulation.resolved.materials


@pytest.mark.parametrize("extra_space", [0, 0.5])
def test_containing_or_touching_bounds_do_not_warn(fdtd_pdk_module, extra_space):
    component = _one_port_component(fdtd_pdk_module)
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.domain(
        x_bounds=(-extra_space, 2 + extra_space),
        y_bounds=(-0.25 - extra_space, 0.25 + extra_space),
        z_bounds=(-extra_space, 0.22 + extra_space),
    )
    # A marker outside the window is not material geometry.
    component.add_polygon([(4, 0), (5, 0), (5, 1), (4, 1)], layer=(8, 0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", fdtd.GeometryClippingWarning)
        simulation.geometry(component)
    assert not [
        w for w in caught if issubclass(w.category, fdtd.GeometryClippingWarning)
    ]
    assert not simulation.resolved.clipped_layers


@pytest.mark.parametrize("angle", [-10, 10])
def test_sidewall_overhang_warns_even_when_mask_fits(fdtd_pdk_module, angle):
    component = _one_port_component(fdtd_pdk_module)
    fdtd_pdk_module.PDK.layer_stack.layers["core"].sidewall_angle = angle
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.domain(y_bounds=(-0.25, 0.25))
    with pytest.warns(fdtd.GeometryClippingWarning, match=r"core \(y\)"):
        simulation.geometry(component)


def test_blanket_and_negative_films_warn_only_for_vertical_trimming(fdtd_pdk_module):
    component = _one_port_component(fdtd_pdk_module)
    fdtd_pdk_module.PDK.layer_stack.layers["film"] = LayerLevel(
        layer=(9, 0), zmin=0, thickness=-1, material="SiO2", background=True
    )
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.domain(x_bounds=(0, 2), y_bounds=(-0.4, 0.4), z_bounds=(-0.2, 0.5))
    with pytest.warns(fdtd.GeometryClippingWarning, match=r"film \(z\)"):
        simulation.geometry(component)
    assert simulation.resolved.clipped_layers == {"film": ("z",)}


def test_changed_window_warns_again_and_reset_does_not_warn(fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.domain(z_bounds=(0.05, 0.2))
    with pytest.warns(fdtd.GeometryClippingWarning):
        simulation.geometry("straight")
    simulation.domain(z_bounds=(0.06, 0.2))
    with pytest.warns(fdtd.GeometryClippingWarning):
        simulation.geometry()
    simulation.domain(z_bounds=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", fdtd.GeometryClippingWarning)
        simulation.geometry()
    assert not [
        w for w in caught if issubclass(w.category, fdtd.GeometryClippingWarning)
    ]


def test_factory_resolution_does_not_repeat_clipping_warning(tmp_path, fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.domain(z_bounds=(0.05, 0.2))
    simulation.source(port="o1")
    with pytest.warns(fdtd.GeometryClippingWarning):
        simulation.geometry("straight")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", fdtd.GeometryClippingWarning)
        simulation.geometry()
        simulation.write(tmp_path)
    assert not [
        w for w in caught if issubclass(w.category, fdtd.GeometryClippingWarning)
    ]


@pytest.mark.parametrize("blanket", [False, True])
def test_fully_removed_masks_do_not_report_material_clipping(fdtd_pdk_module, blanket):
    component = _one_port_component(fdtd_pdk_module)
    drawing = LogicalLayer(layer=(1, 0))
    level = LayerLevel(
        layer=drawing - drawing,
        derived_layer=LogicalLayer(layer=(9, 0)),
        zmin=3,
        thickness=1,
        material="unknown",
    )
    if blanket:
        object.__setattr__(level, "background", True)
        object.__setattr__(level, "background_exclude_layers", [(1, 0)])
    fdtd_pdk_module.PDK.layer_stack.layers["empty"] = level
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.domain(z_bounds=(-0.1, 0.3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", fdtd.GeometryClippingWarning)
        simulation.geometry(component)
    assert not [
        w for w in caught if issubclass(w.category, fdtd.GeometryClippingWarning)
    ]
    assert "unknown" not in simulation.resolved.materials


def test_clipping_does_not_relax_guided_port_faces(tmp_path, fdtd_pdk_module):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.domain(x_bounds=(0, 1))
    simulation.source(port="o1")
    with pytest.warns(fdtd.GeometryClippingWarning):
        simulation.geometry("straight")
    with pytest.raises(fdtd.FDTDGeometryError, match=r"o2.*face"):
        simulation.write(tmp_path)


def test_low_level_bounds_clip_and_legacy_mesh_argument_warns(
    tmp_path, fdtd_pdk_module
):
    component = _one_port_component(fdtd_pdk_module)
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    resolved = replace(simulation.geometry(component), ports={})
    assert background_bounds_nm(resolved, "air", 1, x_bounds=(0, 1))[3] == 1000
    with pytest.warns(DeprecationWarning, match="ignored"):
        manifest = generate_mesh(
            resolved,
            tmp_path / "mesh.msh",
            background_material="air",
            padding=1,
            mesh_size_nm=700,
            geometry_tolerance_nm=10,
            x_bounds=(0, 1),
            clip_geometry=False,
        )
    assert manifest.domain_max_nm[0] == 1000
