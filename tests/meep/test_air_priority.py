"""Overlapping voids obey PDK priority regardless of declaration order."""

import ast
import json
from typing import Any

import gdsfactory as gf
import numpy as np
import pytest

from gsim.common.stack import LayerStack
from gsim.common.stack.extractor import Layer
from gsim.meep import Simulation, mode_solver
from gsim.meep.layer_order import order_by_mesh_order
from gsim.meep.models.api import ModeSolver
from gsim.meep.models.config import MaterialData
from gsim.meep.script import _MEEP_MODE_SOLVER_TEMPLATE
from gsim.meep.simulation import _precompute_cross_section_geometry


@pytest.mark.parametrize("air", ["Air", "air"])
@pytest.mark.parametrize("void_first", [False, True])
@pytest.mark.parametrize("void_wins", [False, True])
def test_mesh_order_controls_overlapping_voids(
    tmp_path, backend, air, void_first, void_wins
):
    component = gf.Component()
    component.add_polygon([(0, -1), (2, -1), (2, 1), (0, 1)], layer=(1, 0))
    component.add_polygon([(0, -0.2), (2, -0.2), (2, 0.2), (0, 0.2)], layer=(2, 0))
    component.add_port(name="o1", center=(0, 0), width=2, layer=(1, 0), orientation=180)
    core = Layer(
        name="core",
        material="Si",
        gds_layer=(1, 0),
        zmin=0,
        zmax=0.2,
        thickness=0.2,
        layer_type="dielectric",
        mesh_order=2 if void_wins else 1,
    )
    void = Layer(
        name="void",
        material=air,
        gds_layer=(2, 0),
        zmin=0,
        zmax=0.2,
        thickness=0.2,
        layer_type="dielectric",
        mesh_order=1 if void_wins else 2,
    )
    layers = [void, core] if void_first else [core, void]
    stack = LayerStack(layers={layer.name: layer for layer in layers})
    geometry = _precompute_cross_section_geometry(
        component,
        stack,
        port="o1",
        y_span=2,
        background_material="Air",
        z_margin=0.5,
        pml_thickness=0.2,
    )
    winner = air if void_wins else "Si"
    assert geometry.blocks[-1].material == winner
    assert geometry.blocks[-1].mesh_order == 1
    profile = mode_solver.refractive_index_profile(
        stack,
        1.55,
        z_grid=np.array([-0.3, 0.1]),
        y_grid=np.array([0, 0.8]),
        component=component,
        port="o1",
        background_material="Air",
    )
    assert profile[1, 0] == pytest.approx(1) if void_wins else profile[1, 0] > 3
    assert profile[1, 1] > 3
    # The void is shallow; silicon's existing lower extension remains below it.
    assert np.all(profile[0] > 3)

    setup = Simulation(mode_solver=ModeSolver(where="slab", wavelengths=[1.55]))
    setup.geometry.stack = stack
    directory = setup.write_mode_solver_config(tmp_path)
    payload = json.loads((directory / "mode_solver_config.json").read_text())
    assert payload["layer_mesh_orders"] == {
        layer.name: layer.mesh_order for layer in layers
    }

    # The runner must also honor priority if serialized blocks arrive reordered.
    tree = ast.parse(_MEEP_MODE_SOLVER_TEMPLATE)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_order_mode_entries",
            "_is_ideal_air",
            "_mode_stack_z_extent",
            "build_cross_section_cell",
            "build_slab_cell",
        }
    ]
    namespace: dict[str, Any] = {
        "mp": backend,
        "_make_medium": lambda material: backend.Medium(
            epsilon=material["epsilon_diag"][0]
        ),
    }
    exec(  # noqa: S102 - execute only trusted runner-template helpers
        compile(ast.Module(body=functions, type_ignores=[]), "mode-priority", "exec"),
        namespace,
    )
    shuffled = list(reversed(geometry.model_dump()["blocks"]))
    ordered = namespace["_order_mode_entries"](shuffled)
    assert ordered[-1]["material"] == winner
    slabs = namespace["_order_mode_entries"](
        payload["layer_stack"], payload["layer_mesh_orders"]
    )
    assert slabs[-1]["material"] == winner
    payload["cross_section_geometry"] = geometry.model_dump()
    payload["pml_thickness"] = 0.2
    payload["cross_section_geometry"]["blocks"] = shuffled
    cross_section, _, _ = namespace["build_cross_section_cell"](payload)
    del payload["cross_section_geometry"]
    slab, _ = namespace["build_slab_cell"](payload)
    local_slab, _ = mode_solver._build_slab_xz_cell(
        stack,
        {
            name: MaterialData.model_validate(data)
            for name, data in payload["materials"].items()
        },
        resolution=32,
        background_material="Air",
    )
    natives = [cross_section, slab, local_slab]
    materials = {
        name: MaterialData.model_validate(data)
        for name, data in payload["materials"].items()
    }
    for plane in ("xz", "yz"):
        builder = getattr(mode_solver, f"_build_component_{plane}_cell")
        kwargs = {
            "x_span" if plane == "xz" else "y_span": 2,
            "y_cut" if plane == "xz" else "x_cut": 0 if plane == "xz" else 1,
        }
        native, _ = builder(
            component,
            stack,
            materials=materials,
            resolution=32,
            background_material="Air",
            **kwargs,
        )
        natives.append(native)
    for native in natives:
        epsilon = native.geometry[-1].material.epsilon
        assert epsilon == pytest.approx(1) if void_wins else epsilon > 9


def test_unranked_and_equal_priority_keep_legacy_input_order():
    entries = [
        {"name": "legacy1"},
        {"name": "rank1", "mesh_order": 2},
        {"name": "legacy2"},
        {"name": "rank2", "mesh_order": 2},
    ]
    assert [entry["name"] for entry in order_by_mesh_order(entries)] == [
        "legacy1",
        "legacy2",
        "rank1",
        "rank2",
    ]


@pytest.mark.parametrize("air", ["Air", "air"])
@pytest.mark.parametrize("void_last", [False, True])
def test_unranked_slab_air_cutouts_follow_declaration_order(
    tmp_path, backend, monkeypatch, air, void_last
):
    core = Layer(
        name="core",
        material="Si",
        gds_layer=(1, 0),
        zmin=0,
        zmax=1,
        thickness=1,
        layer_type="dielectric",
    )
    void = Layer(
        name="void",
        material=air,
        gds_layer=(2, 0),
        zmin=0.3,
        zmax=0.7,
        thickness=0.4,
        layer_type="dielectric",
    )
    layers = [core, void] if void_last else [void, core]
    stack = LayerStack(layers={layer.name: layer for layer in layers})
    setup = Simulation(mode_solver=ModeSolver(where="slab", wavelengths=[1.55]))
    setup.geometry.stack = stack
    directory = setup.write_mode_solver_config(tmp_path)
    payload = json.loads((directory / "mode_solver_config.json").read_text())
    materials = {
        name: MaterialData.model_validate(data)
        for name, data in payload["materials"].items()
    }
    local, _ = mode_solver._build_slab_xz_cell(
        stack, materials, resolution=32, background_material="Air"
    )
    silicon_epsilon = materials["Si"].epsilon_diag
    assert isinstance(silicon_epsilon, list)
    expected_epsilon = 1 if void_last else silicon_epsilon[0]
    assert local.geometry[-1].material.epsilon == pytest.approx(expected_epsilon)
    profile = mode_solver.refractive_index_profile(
        stack, 1.55, z_grid=np.array([0.1, 0.5, 0.9]), background_material="Air"
    )
    assert profile[1] ** 2 == pytest.approx(expected_epsilon)
    assert np.all(profile[[0, 2]] > 3)

    tree = ast.parse(_MEEP_MODE_SOLVER_TEMPLATE)
    names = {
        "_order_mode_entries",
        "_is_ideal_air",
        "_mode_stack_z_extent",
        "build_slab_cell",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, Any] = {
        "mp": backend,
        "_make_medium": lambda material: backend.Medium(
            epsilon=material["epsilon_diag"][0]
        ),
    }
    exec(  # noqa: S102 - execute trusted runner-template functions
        compile(ast.Module(body=functions, type_ignores=[]), "slab-cutout", "exec"),
        namespace,
    )
    cloud, _ = namespace["build_slab_cell"](payload)
    assert cloud.geometry[-1].material.epsilon == pytest.approx(expected_epsilon)

    def capture_result(simulation, _cell_size, _wavelength, **_kwargs):
        assert simulation.geometry[-1].material.epsilon == pytest.approx(
            expected_epsilon
        )
        return object()

    monkeypatch.setattr(mode_solver, "_compute_eigenmode", capture_result)
    monkeypatch.setattr(mode_solver, "_validate_mode", lambda *args, **kwargs: None)
    result = mode_solver.solve_slab_wavelength_sweep(
        stack, [1.55], background_material="Air"
    )
    assert len(result) == 1


@pytest.mark.parametrize("air", ["Air", "air"])
def test_uniform_air_cutout_is_retained_in_cross_section_and_xy_background(
    backend, air
):
    component = gf.c.straight(length=2, width=2)
    core = Layer(
        name="core",
        material="Si",
        gds_layer=(1, 0),
        zmin=0,
        zmax=1,
        thickness=1,
        layer_type="dielectric",
    )
    stack = LayerStack(
        layers={"core": core},
        dielectrics=[{"name": "air_cutout", "material": air, "zmin": 0.3, "zmax": 0.7}],
    )
    geometry = _precompute_cross_section_geometry(
        component, stack, port="o1", y_span=2, z_margin=0.5, background_material="Air"
    )
    cutout = geometry.blocks[-1]
    assert cutout.material == air
    assert cutout.z_center - cutout.z_size / 2 == pytest.approx(-0.2)
    assert cutout.z_center + cutout.z_size / 2 == pytest.approx(0.2)
    setup = Simulation()
    setup.solver(mode="2d", z_cut=0.5)
    assert setup._resolve_xy_background_material(component, stack) == air

    tree = ast.parse(_MEEP_MODE_SOLVER_TEMPLATE)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_order_mode_entries", "build_cross_section_cell"}
    ]
    namespace: dict[str, Any] = {
        "mp": backend,
        "_make_medium": lambda material: backend.Medium(
            epsilon=material["epsilon_diag"][0]
        ),
    }
    exec(  # noqa: S102 - execute trusted runner-template functions
        compile(ast.Module(body=functions, type_ignores=[]), "uniform-cutout", "exec"),
        namespace,
    )
    native, _, _ = namespace["build_cross_section_cell"](
        {
            "cross_section_geometry": geometry.model_dump(),
            "materials": {
                "Si": {"epsilon_diag": [12, 12, 12]},
                air: {"epsilon_diag": [1, 1, 1]},
            },
        }
    )
    assert native.geometry[-1].material.epsilon == 1
