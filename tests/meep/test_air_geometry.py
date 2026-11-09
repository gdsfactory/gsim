"""Patterned air regions must survive mode setup in a dielectric background."""

from types import SimpleNamespace

import gdsfactory as gf
import numpy as np
import pytest

from gsim.common.stack.extractor import Layer
from gsim.meep.mode_solver import refractive_index_profile
from gsim.meep.simulation import _precompute_cross_section_geometry


@pytest.fixture
def fake_mode_solver(monkeypatch):
    from gsim.meep import mode_solver

    fake_meep = SimpleNamespace(
        Vector3=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
        Block=lambda **kwargs: SimpleNamespace(**kwargs),
        Medium=lambda **kwargs: SimpleNamespace(**kwargs),
        Simulation=lambda **kwargs: SimpleNamespace(**kwargs),
        PML=lambda **kwargs: SimpleNamespace(**kwargs),
        X="x",
        Y="y",
        Z="z",
        inf=np.inf,
    )
    monkeypatch.setattr(mode_solver, "_import_meep", lambda: fake_meep)
    monkeypatch.setattr(mode_solver, "_meep_medium", lambda material: material)
    return mode_solver


@pytest.mark.parametrize("air", ["Air", "air"])
def test_patterned_air_is_exported_and_plotted_inside_oxide(air):
    component = gf.Component()
    component.add_polygon([(0, -0.2), (2, -0.2), (2, 0.2), (0, 0.2)], layer=(2, 0))
    component.add_port(
        name="o1", center=(0, 0), width=0.4, layer=(2, 0), orientation=180
    )
    layer = Layer(
        name="void",
        material=air,
        gds_layer=(2, 0),
        zmin=0,
        zmax=0.2,
        thickness=0.2,
        layer_type="dielectric",
    )
    stack = SimpleNamespace(layers={"void": layer}, dielectrics=[])
    geometry = _precompute_cross_section_geometry(
        component,
        stack,
        port="o1",
        y_span=2,
        background_material="SiO2",
        z_margin=0.5,
        pml_thickness=0.2,
    )
    assert any(block.material == air for block in geometry.blocks)
    assert len(geometry.blocks) == 1
    block = geometry.blocks[0]
    assert block.z_center - block.z_size / 2 == pytest.approx(-0.1)
    assert block.z_center + block.z_size / 2 == pytest.approx(0.1)
    profile = refractive_index_profile(
        stack,
        wavelength=1.55,
        z_grid=np.array([-0.3, 0.1, 0.4]),
        y_grid=np.array([0, 0.8]),
        component=component,
        port="o1",
        background_material="SiO2",
    )
    assert profile[1, 0] == pytest.approx(1)
    assert profile[1, 1] > 1.4
    assert np.all(profile[[0, 2], :] > 1.4)


@pytest.mark.parametrize("air", ["Air", "air"])
def test_undrawn_air_declaration_does_not_create_a_blanket_void(air):
    component = gf.c.straight(length=2)
    layer = Layer(
        name="undercut",
        material=air,
        gds_layer=(2, 0),
        zmin=0,
        zmax=0.2,
        thickness=0.2,
        layer_type="dielectric",
    )
    stack = SimpleNamespace(layers={"undercut": layer}, dielectrics=[])
    geometry = _precompute_cross_section_geometry(
        component, stack, port="o1", y_span=2, background_material="SiO2"
    )
    assert not geometry.blocks
    profile = refractive_index_profile(
        stack,
        wavelength=1.55,
        z_grid=np.array([0.1]),
        y_grid=np.array([0, 0.8]),
        component=component,
        port="o1",
        background_material="SiO2",
    )
    assert np.all(profile > 1.4)


@pytest.mark.parametrize("air", ["Air", "air"])
@pytest.mark.parametrize("plane", ["slab", "xz", "yz"])
def test_local_mode_air_geometry_keeps_declared_z_bounds(fake_mode_solver, air, plane):
    mode_solver = fake_mode_solver
    component = gf.Component()
    component.add_polygon([(0, -0.2), (2, -0.2), (2, 0.2), (0, 0.2)], layer=(2, 0))
    layer = Layer(
        name="void",
        material=air,
        gds_layer=(2, 0),
        zmin=0,
        zmax=0.2,
        thickness=0.2,
        layer_type="dielectric",
    )
    stack = SimpleNamespace(layers={"void": layer}, dielectrics=[])
    kwargs = dict(
        stack=stack,
        materials={air: object(), "SiO2": object()},
        background_material="SiO2",
        z_margin=0.5,
        pml_thickness=0.2,
        resolution=32,
    )
    if plane == "slab":
        simulation, _ = mode_solver._build_slab_xz_cell(**kwargs)
    else:
        builder = getattr(mode_solver, f"_build_component_{plane}_cell")
        kwargs["x_span" if plane == "xz" else "y_span"] = 2
        kwargs["y_cut" if plane == "xz" else "x_cut"] = 0 if plane == "xz" else 1
        simulation, _ = builder(component=component, **kwargs)
    assert len(simulation.geometry) == 1
    block = simulation.geometry[0]
    assert block.center.z - block.size.z / 2 == pytest.approx(-0.1)
    assert block.center.z + block.size.z / 2 == pytest.approx(0.1)


@pytest.mark.parametrize("air", ["Air", "air"])
def test_patterned_void_is_preserved_inside_silicon_with_air_background(air):
    component = gf.Component()
    component.add_polygon([(0, -1), (2, -1), (2, 1), (0, 1)], layer=(1, 0))
    component.add_polygon([(0, -0.2), (2, -0.2), (2, 0.2), (0, 0.2)], layer=(2, 0))
    component.add_port(name="o1", center=(0, 0), width=2, layer=(1, 0), orientation=180)
    stack = SimpleNamespace(
        layers={
            "core": Layer(
                name="core",
                material="Si",
                gds_layer=(1, 0),
                zmin=0,
                zmax=0.2,
                thickness=0.2,
                layer_type="dielectric",
            ),
            "void": Layer(
                name="void",
                material=air,
                gds_layer=(2, 0),
                zmin=0,
                zmax=0.2,
                thickness=0.2,
                layer_type="dielectric",
            ),
        },
        dielectrics=[],
    )
    geometry = _precompute_cross_section_geometry(
        component, stack, port="o1", y_span=2, background_material="Air"
    )
    assert any(block.material == air for block in geometry.blocks)
    profile = refractive_index_profile(
        stack,
        wavelength=1.55,
        z_grid=np.array([0.1]),
        y_grid=np.array([0, 0.8]),
        component=component,
        port="o1",
        background_material="Air",
    )
    assert profile[0, 0] == pytest.approx(1)
    assert profile[0, 1] > 3


@pytest.mark.parametrize("air", ["Air", "air"])
@pytest.mark.parametrize("bounds", ["auto", (-1.5, 1.5)])
def test_actual_cross_section_cells_and_grids_honor_reported_domain(
    fake_mode_solver, air, bounds
):
    from gsim.common.stack import LayerStack
    from gsim.meep import Simulation
    from gsim.meep.models.api import Domain

    component = gf.c.straight(length=2)
    stack = LayerStack(
        layers={
            "core": Layer(
                name="core",
                material="Si",
                gds_layer=(1, 0),
                zmin=0,
                zmax=0.2,
                thickness=0.2,
                layer_type="dielectric",
            ),
            "unused_air": Layer(
                name="unused_air",
                material=air,
                gds_layer=(2, 0),
                zmin=-100,
                zmax=100,
                thickness=200,
                layer_type="dielectric",
            ),
        },
        dielectrics=[{"name": "box", "material": "SiO2", "zmin": -0.5, "zmax": 0}],
    )
    setup = Simulation(domain=Domain(z_bounds=bounds))
    setup.geometry.stack = stack
    setup.geometry.component = component
    setup.mode_solver.port = "o1"
    window, margins = setup._resolve_mode_solver_z_window()
    expected = (-1, 0.7) if bounds == "auto" else bounds
    assert window == pytest.approx(expected)
    expected_span = expected[1] - expected[0] + 0.4
    precomputed = _precompute_cross_section_geometry(
        component,
        stack,
        port="o1",
        y_span=2,
        z_margin=margins,
        pml_thickness=0.2,
        resolution=100,
    )
    assert precomputed.cell_z_span == pytest.approx(expected_span)
    common = dict(
        stack=stack,
        materials={"Si": object(), air: object()},
        resolution=100,
        z_margin=margins,
        pml_thickness=0.2,
    )
    for plane in ("xz", "yz"):
        builder = getattr(
            fake_mode_solver,
            f"_build_component_{plane}_cell",
        )
        kwargs = {
            "component": component,
            "x_span" if plane == "xz" else "y_span": 2,
            "y_cut" if plane == "xz" else "x_cut": 0 if plane == "xz" else 1,
        }
        _, size = builder(**common, **kwargs)
        assert size.z == pytest.approx(expected_span)
    grid = fake_mode_solver.mode_z_grid(
        stack, 20, z_margin=margins, pml_thickness=0.2, component=component
    )
    half_step = expected_span / 40
    assert grid[0] == pytest.approx(expected[0] - 0.2 + half_step)
    assert grid[-1] == pytest.approx(expected[1] + 0.2 - half_step)


@pytest.mark.parametrize("air", ["Air", "air"])
@pytest.mark.parametrize("air_only", [False, True])
@pytest.mark.parametrize("where", ["cross_section", "slab"])
@pytest.mark.parametrize("uniform", [False, True])
def test_mode_window_contains_modeled_air(
    fake_mode_solver, air, air_only, where, uniform
):
    import ast

    from gsim.common.stack import LayerStack
    from gsim.meep import Simulation
    from gsim.meep.script import _MEEP_MODE_SOLVER_TEMPLATE

    component = gf.c.straight(length=2).copy()
    component.add_polygon([(0, -0.1), (2, -0.1), (2, 0.1), (0, 0.1)], layer=(2, 0))
    void = Layer(
        name="void",
        material=air,
        gds_layer=(2, 0),
        zmin=0,
        zmax=2,
        thickness=2,
        layer_type="dielectric",
    )
    layers = {} if uniform else {"void": void}
    if not air_only:
        layers["core"] = void.model_copy(
            update={
                "name": "core",
                "material": "Si",
                "gds_layer": (1, 0),
                "zmax": 0.2,
                "thickness": 0.2,
            }
        )
    stack = LayerStack(
        layers=layers,
        dielectrics=[{"name": "void", "material": air, "zmin": 0, "zmax": 2}]
        if uniform
        else [],
    )
    setup = Simulation()
    setup.geometry(component=component, stack=stack)
    setup.mode_solver(
        where=where,
        port="o1" if where == "cross_section" else None,
        background_material="SiO2",
    )
    window, margins = setup._resolve_mode_solver_z_window()
    assert window == pytest.approx((-0.5, 2.5))
    setup.domain.z_bounds = (-0.5, 1.5)
    with pytest.raises(ValueError, match="must contain"):
        setup._resolve_mode_solver_z_window()

    common = dict(
        stack=stack,
        materials={"Si": object(), air: object()},
        resolution=100,
        z_margin=margins,
        pml_thickness=0.2,
    )
    if where == "cross_section":
        geometry = _precompute_cross_section_geometry(
            component,
            stack,
            port="o1",
            y_span=2,
            z_margin=margins,
            pml_thickness=0.2,
            resolution=100,
        )
        assert geometry.cell_z_span == pytest.approx(3.4)
        for block in geometry.blocks:
            if block.material == air:
                assert (
                    abs(block.z_center) + block.z_size / 2 <= geometry.cell_z_span / 2
                )
        for plane in ("xz", "yz"):
            builder = getattr(fake_mode_solver, f"_build_component_{plane}_cell")
            _, size = builder(
                **common,
                component=component,
                **{
                    "x_span" if plane == "xz" else "y_span": 2,
                    "y_cut" if plane == "xz" else "x_cut": 0 if plane == "xz" else 1,
                },
            )
            assert size.z == pytest.approx(3.4)
        grid_component = component
    else:
        _, size = fake_mode_solver._build_slab_xz_cell(**common)
        assert size.z == pytest.approx(3.4)
        grid_component = None
    grid = fake_mode_solver.mode_z_grid(
        stack, 20, z_margin=margins, pml_thickness=0.2, component=grid_component
    )
    assert grid[[0, -1]] == pytest.approx([-0.7 + 3.4 / 40, 2.7 - 3.4 / 40])

    # Execute the generated slab builder against the same numeric reference.
    tree = ast.parse(_MEEP_MODE_SOLVER_TEMPLATE)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_is_ideal_air",
            "_mode_stack_z_extent",
            "_order_mode_entries",
            "build_slab_cell",
        }
    ]
    namespace = {
        "mp": fake_mode_solver._import_meep(),
        "_make_medium": lambda material: material,
    }
    exec(  # noqa: S102 - execute selected functions from the trusted runner template
        compile(ast.Module(body=functions, type_ignores=[]), "mode-builder", "exec"),
        namespace,
    )
    payload = {
        "layer_stack": [
            {"material": layer.material, "zmin": layer.zmin, "zmax": layer.zmax}
            for layer in stack.layers.values()
        ],
        "dielectrics": stack.dielectrics,
        "materials": common["materials"],
        "z_margin": list(margins),
        "pml_thickness": 0.2,
        "resolution": 100,
    }
    _, cloud_size = namespace["build_slab_cell"](payload)
    assert cloud_size.z == pytest.approx(3.4)
