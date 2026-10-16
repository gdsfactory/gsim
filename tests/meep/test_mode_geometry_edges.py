"""Check cross-section serialization, explicit windows and invalid geometry."""

import pytest

from gsim.common.stack import LayerStack
from gsim.meep import Simulation
from gsim.meep.models.api import Domain, Material
from gsim.meep.simulation import _precompute_cross_section_geometry


@pytest.mark.parametrize("orientation", [0, 90])
@pytest.mark.parametrize("position", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_cross_section_blocks_ignore_air_and_zero_thickness_layers(
    index_context, orientation, position, filtered
):
    stack, component = index_context
    component.add_port(
        "cut", center=(0, 0), orientation=orientation, width=1, layer=(1, 0)
    )
    for name, material, top in [
        ("air", "air", 1),
        ("planar", "core", 0),
        ("unknown", "unlisted", 1),
    ]:
        stack.layers[name] = stack.layers["core"].model_copy(
            update={"name": name, "material": material, "zmax": top, "thickness": top}
        )
    stack.dielectrics.extend(
        [
            {"name": "air", "material": "air", "zmin": 0, "zmax": 1},
            {"name": "flat", "material": "clad", "zmin": 0, "zmax": 0},
        ]
    )
    selection = {"position": (0, 0)} if position else {"port": "cut"}
    result = _precompute_cross_section_geometry(
        component,
        stack,
        x_span=4,
        y_span=4,
        material_data={"core": {}, "clad": {}} if filtered else None,
        **selection,
    )
    assert result.plane == ("xz" if position or orientation == 90 else "yz")
    assert (result.cell_horizontal_span, result.cell_z_span) == (4, 2)
    core = next(block for block in result.blocks if block.material == "core")
    assert (core.horizontal_size, core.z_size, core.z_center) == (2, 1, 0.5)
    assert all(block.material != "air" and block.z_size > 0 for block in result.blocks)
    assert any(block.material == "unlisted" for block in result.blocks) == (
        not filtered
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "component",
        "stack",
        "missing-port",
        "yz-span",
        "xz-span",
        "position-span",
        "selector",
    ],
)
def test_serialized_cross_section_rejects_incomplete_inputs(index_context, invalid):
    stack, component = index_context
    component.add_port("east", center=(0, 0), orientation=0, width=1, layer=(1, 0))
    component.add_port("north", center=(0, 0), orientation=90, width=1, layer=(1, 0))
    options = {"port": "east", "x_span": 4, "y_span": 4}
    if invalid == "component":
        component = None
    elif invalid == "stack":
        stack = None
    elif invalid == "missing-port":
        options["port"] = "missing"
    elif invalid == "yz-span":
        options["y_span"] = None
    elif invalid == "xz-span":
        options.update(port="north", x_span=None)
    elif invalid == "position-span":
        options.update(port=None, position=(0, 0), x_span=None)
    else:
        options["port"] = None
    with pytest.raises(ValueError):
        _precompute_cross_section_geometry(component, stack, **options)


@pytest.mark.parametrize("bounds", [(-2, 3), (-0.5, 3), (-2, 0.5)])
def test_mode_domain_must_contain_the_material_stack(index_context, bounds):
    simulation = Simulation(domain=Domain(z_bounds=bounds))
    simulation.geometry.stack = index_context[0]
    if bounds == (-2, 3):
        actual, margins = simulation._resolve_mode_solver_z_window()
        assert actual == bounds
        assert margins == (1, 2)
    else:
        with pytest.raises(ValueError, match="must contain"):
            simulation._resolve_mode_solver_z_window()


@pytest.mark.parametrize("reference", ["core", "missing", "stack", None])
def test_crop_reference_and_legacy_mode_margins(index_context, reference):
    stack, component = index_context
    simulation = Simulation(domain=Domain(z_ref=reference, margin_z=(1, 2)))
    simulation.geometry(component=component, stack=stack)
    with pytest.warns(FutureWarning):
        assert simulation._resolve_mode_solver_z_window() == ((-2, 3), (1, 2))
    if reference == "missing":
        with pytest.raises(ValueError, match="not found"):
            simulation._resolve_z_ref_extent()
    else:
        result = simulation._resolve_z_ref_extent()
        assert result[:2] == ((0, 1) if reference == "core" else (-1, 1))


@pytest.mark.parametrize("reference", ["stack", None])
def test_empty_stack_cannot_define_a_mode_or_crop_reference(reference):
    simulation = Simulation(domain=Domain(z_ref=reference))
    simulation.geometry.stack = LayerStack()
    assert simulation._stack_material_extent() is None
    with pytest.raises(ValueError):
        simulation._resolve_mode_solver_z_window()
    with pytest.raises(ValueError):
        simulation._resolve_z_ref_extent()


def test_material_normalization_accepts_objects_and_rejects_unknown_values():
    material = Material(permittivity=4)
    assert Simulation(materials={"core": material}).materials["core"] is material
    with pytest.raises(TypeError, match="must be a Material"):
        Simulation(materials={"core": object()})
