"""Custom Air responses remain physical media throughout mode setup."""

import ast
import json

import gdsfactory as gf
import numpy as np
import pytest

from gsim.common.stack import LayerStack
from gsim.common.stack.extractor import Layer
from gsim.common.stack.materials import MaterialProperties
from gsim.meep import Simulation, mode_solver
from gsim.meep.mode_bounds import material_extent
from gsim.meep.models.api import Material, ModeSolver
from gsim.meep.models.config import MaterialData
from gsim.meep.script import _MEEP_MODE_SOLVER_TEMPLATE


@pytest.fixture(params=["override", "overlay"])
def custom_air_setup(request):
    def make(air, plane):
        component = gf.c.straight(length=2).copy()
        component.add_port(
            "cut",
            center=(1, 0),
            width=0.5,
            layer=(1, 0),
            orientation=0 if plane == "yz" else 90,
        )
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
                    mesh_order=1,
                ),
                "custom_cladding": Layer(
                    name="custom_cladding",
                    material=air,
                    gds_layer=(2, 0),
                    zmin=0,
                    zmax=2,
                    thickness=2,
                    layer_type="dielectric",
                    mesh_order=2,
                ),
            }
        )
        simulation = Simulation()
        simulation.geometry(component=component, stack=stack)
        simulation.mode_solver(
            where="cross_section",
            port="cut",
            wavelengths=[1.55],
            x_span=4,
            y_span=4,
            n_field_z=20,
        )
        simulation.solver.resolution = 100
        simulation.domain.pml = 0.2
        if request.param == "override":
            simulation.materials = {air: Material(permittivity=2.25)}
        else:
            simulation._pdk_overlay = {air: MaterialProperties(permittivity=2.25)}
        return simulation

    return make


@pytest.mark.parametrize("air", ["Air", "air"])
@pytest.mark.parametrize("plane", ["xz", "yz"])
def test_custom_air_matches_export_local_cell_and_profile(
    tmp_path,
    backend,
    custom_air_setup,
    air,
    plane,
):
    simulation = custom_air_setup(air, plane)
    component, stack = simulation.geometry.component, simulation.geometry.stack
    window, margins = simulation._resolve_mode_solver_z_window()
    assert window == pytest.approx((-0.5, 2.5))
    assert simulation._stack_material_extent() == (0, 2)
    directory = simulation.write_mode_solver_config(tmp_path)
    config = json.loads((directory / "mode_solver_config.json").read_text())
    cross_section = config["cross_section_geometry"]
    assert cross_section["cell_z_span"] == pytest.approx(3.4)
    air_block = next(
        block for block in cross_section["blocks"] if block["material"] == air
    )
    assert air_block["z_center"] + air_block["z_size"] / 2 == pytest.approx(1)
    assert config["materials"][air]["epsilon_diag"] == [2.25] * 3

    materials = {
        name: MaterialData.model_validate(data)
        for name, data in config["materials"].items()
    }
    builder = getattr(mode_solver, f"_build_component_{plane}_cell")
    local, size = builder(
        component,
        stack,
        materials=materials,
        resolution=100,
        z_margin=margins,
        pml_thickness=0.2,
        **{
            "x_span" if plane == "xz" else "y_span": 4,
            "y_cut" if plane == "xz" else "x_cut": 0 if plane == "xz" else 1,
        },
    )
    assert size.z == pytest.approx(3.4)
    film = next(shape for shape in local.geometry if shape.material.epsilon == 2.25)
    assert film.center.z + film.size.z / 2 == pytest.approx(1)
    grid = mode_solver.mode_z_grid(
        stack,
        20,
        z_margin=margins,
        pml_thickness=0.2,
        component=component,
        materials=materials,
    )
    assert grid[[0, -1]] == pytest.approx([-0.7 + 3.4 / 40, 2.7 - 3.4 / 40])
    profile = mode_solver.refractive_index_profile(
        stack,
        1.55,
        z_grid=np.array([0.8]),
        component=component,
        port="cut",
        **{"x_grid" if plane == "xz" else "y_grid": np.array([0.0, 1.0])},
        overrides=simulation._material_overrides(),
        overlay=simulation._pdk_overlay,
    )
    np.testing.assert_allclose(profile, 1.5)

    # Check material forwarding and field coordinates through the public local path.
    result = simulation.solve_modes_local()
    assert result is not None
    film = next(
        shape
        for shape in backend.instances[-1].geometry
        if shape.material.epsilon == 2.25
    )
    assert film.center.z + film.size.z / 2 == pytest.approx(1)
    simulation.domain.z_bounds = (-0.5, 1.5)
    with pytest.raises(ValueError, match="must contain"):
        simulation.write_mode_solver_config(tmp_path / "too_small")


@pytest.mark.parametrize("air", ["Air", "air"])
def test_custom_air_padding_extent_matches_generated_slab_runner(air):
    materials = {air: MaterialData(epsilon_diag=[2.25] * 3)}
    stack = LayerStack(
        layers={},
        dielectrics=[
            {"name": "air_box", "material": air, "zmin": -3, "zmax": 2},
        ],
    )
    assert material_extent(stack, materials=materials) == (-3, 2)
    tree = ast.parse(_MEEP_MODE_SOLVER_TEMPLATE)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_is_ideal_air", "_mode_stack_z_extent"}
    ]
    namespace = {}
    exec(  # noqa: S102 - selected functions from the trusted runner template
        compile(ast.Module(body=functions, type_ignores=[]), "custom-air", "exec"),
        namespace,
    )
    config = {
        "layer_stack": [],
        "dielectrics": stack.dielectrics,
        "materials": {name: data.model_dump() for name, data in materials.items()},
    }
    assert namespace["_mode_stack_z_extent"](config) == (-3, 2)


@pytest.mark.parametrize(
    "response",
    [
        {"epsilon_diag": [1, 2.25, 1]},
        {"D_conductivity": 0.1},
        {"epsilon_offdiag": [0.1, 0, 0]},
        {"mu_diag": [2, 2, 2]},
        {"epsilon_susceptibilities": [{"frequency": 1, "gamma": 0.1, "sigma": 1}]},
    ],
)
def test_nonideal_air_keeps_undrawn_layer_extent(response):
    stack = LayerStack(
        layers={
            "film": Layer(
                name="film",
                material="Air",
                gds_layer=(2, 0),
                zmin=-2,
                zmax=2,
                thickness=4,
                layer_type="dielectric",
            )
        }
    )
    material = MaterialData.model_validate(response)
    assert material_extent(stack, gf.c.straight(), {"Air": material}) == (-2, 2)


@pytest.mark.parametrize("air", ["Air", "air"])
@pytest.mark.parametrize(
    ("wavelengths", "bands"), [([1.55], 1), ([1.55], 2), ([1.5, 1.6], 1)]
)
def test_local_slab_variants_preserve_custom_air(
    backend,
    custom_air_setup,
    air,
    wavelengths,
    bands,
):
    simulation = custom_air_setup(air, "yz")
    simulation.mode_solver = ModeSolver(
        where="slab", wavelengths=wavelengths, num_bands=bands
    )
    simulation.solve_modes_local()
    for native in backend.instances:
        assert any(shape.material.epsilon == 2.25 for shape in native.geometry)
        assert native.cell_size.z == pytest.approx(3.4)


def test_empty_wavelength_sweep_still_returns_no_results(backend):
    assert mode_solver.solve_slab_wavelength_sweep(LayerStack(layers={}), []) == {}
    assert not backend.instances


@pytest.mark.parametrize("air", ["Air", "air"])
@pytest.mark.parametrize(
    ("where", "plane", "wavelengths", "bands"),
    [
        ("cross_section", "xz", [1.55], 1),
        ("cross_section", "yz", [1.55], 1),
        ("slab", "yz", [1.55], 1),
        ("slab", "yz", [1.55], 2),
        ("slab", "yz", [1.5, 1.6], 1),
    ],
)
def test_background_only_custom_air_is_resolved_everywhere(
    tmp_path,
    backend,
    custom_air_setup,
    air,
    where,
    plane,
    wavelengths,
    bands,
):
    simulation = custom_air_setup(air, plane)
    stack = simulation.geometry.stack
    del stack.layers["custom_cladding"]
    simulation.mode_solver = ModeSolver(
        where=where,
        port="cut" if where == "cross_section" else None,
        wavelengths=wavelengths,
        num_bands=bands,
        background_material=air,
        x_span=4 if where == "cross_section" else None,
        y_span=4 if where == "cross_section" else None,
    )
    directory = simulation.write_mode_solver_config(tmp_path)
    config = json.loads((directory / "mode_solver_config.json").read_text())
    assert config["materials"][air]["epsilon_diag"] == [2.25] * 3
    assert config["background_material"] == air
    assert all(entry["material"] != air for entry in config["layer_stack"])
    simulation.solve_modes_local()
    assert backend.instances
    for native in backend.instances:
        assert native.default_material.epsilon == 2.25
    for is_cross_section in (False, True):
        selection = {}
        if is_cross_section:
            selection = {
                "component": simulation.geometry.component,
                "port": "cut",
                "x_grid" if plane == "xz" else "y_grid": np.array([0.0, 1.0]),
            }
        profile = mode_solver.refractive_index_profile(
            stack,
            1.55,
            z_grid=np.array([0.8]),
            background_material=air,
            overrides=simulation._material_overrides(),
            overlay=simulation._pdk_overlay,
            **selection,
        )
        np.testing.assert_allclose(profile, 1.5)
