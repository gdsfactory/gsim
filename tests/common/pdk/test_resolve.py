from __future__ import annotations

from types import SimpleNamespace

import gdsfactory as gf
import pytest
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer
from pdk_schema import MaterialCard

from gsim.common.materials import GSIM_MATERIAL_CARDS
from gsim.common.pdk import (
    ComponentResolutionError,
    LayerResolutionError,
    UnsupportedPortError,
    resolve_passive_pcell,
)


def demo_component(length: float = 2.0, port_orientation: float = 0) -> gf.Component:
    component = gf.Component()
    component.add_polygon(
        [(0, -0.25), (length, -0.25), (length, 0.25), (0, 0.25)],
        layer=(1, 0),
    )
    component.add_polygon(
        [(0.75, -0.25), (1.25, -0.25), (1.25, 0), (0.75, 0)],
        layer=(2, 0),
    )
    component.add_polygon(
        [(-0.5, -0.75), (length + 0.5, -0.75), (length + 0.5, 0.75), (-0.5, 0.75)],
        layer=(3, 0),
    )
    component.add_port(
        name="o1",
        center=(0, 0),
        width=0.5,
        orientation=180,
        layer=(1, 0),
    )
    component.add_port(
        name="o2",
        center=(length, 0),
        width=0.5,
        orientation=port_orientation,
        layer=(1, 0),
    )
    return component


def make_test_pdk(
    *,
    core_material: str | None = "Si",
    port_orientation: float = 0,
    attach_cards: bool = True,
) -> gf.Pdk:
    core_expression = LogicalLayer(layer=(1, 0)) - LogicalLayer(layer=(2, 0))
    layer_stack = LayerStack(
        layers={
            "core_key": LayerLevel(
                name="core_declared_name",
                layer=core_expression,
                derived_layer=LogicalLayer(layer=(10, 0)),
                thickness=0.22,
                zmin=0,
                sidewall_angle=10,
                width_to_z=0.5,
                bias=(0.01, 0.02),
                z_to_bias=([0, 0.22], [0, 0.01]),
                mesh_order=1,
                material=core_material,
            ),
            "oxide": LayerLevel(
                layer=LogicalLayer(layer=(3, 0)),
                thickness=2,
                zmin=-2,
                mesh_order=4,
                material="SiO2",
            ),
            "unused": LayerLevel(
                layer=LogicalLayer(layer=(99, 0)),
                thickness=1,
                zmin=0,
                material="MissingOnPurpose",
            ),
        }
    )

    def component_factory(length: float = 2.0) -> gf.Component:
        return demo_component(length=length, port_orientation=port_orientation)

    pdk = gf.Pdk(
        name=f"resolver_test_{core_material}_{port_orientation}",
        cells={"demo": component_factory},
        layer_stack=layer_stack,
    )
    if attach_cards:
        project_si = GSIM_MATERIAL_CARDS["Si-Li-293K"].model_copy(update={"name": "Si"})
        object.__setattr__(pdk, "material_cards", {"Si": project_si})
    return pdk


def test_resolves_named_component_stack_materials_and_ports() -> None:
    original_pdk = gf.get_active_pdk()
    result = resolve_passive_pcell(
        "demo",
        pdk=make_test_pdk(),
        settings={"length": 3.0},
        wavelength_um=1.55,
    )

    assert gf.get_active_pdk() is original_pdk
    assert result.component.name.startswith("Unnamed")
    assert set(result.layers) == {"core_key", "oxide"}
    assert set(result.materials) == {"Si", "SiO2"}
    assert result.materials["Si"].source == "project"
    assert result.materials["Si"].refractive_index == pytest.approx(3.475687046)
    assert result.materials["SiO2"].source == "gsim"
    assert result.bounds == ((-0.5, -0.75, -2.0), (3.5, 0.75, 0.22))
    assert result.ports["o1"].normal == (-1, 0, 0)
    assert result.ports["o2"].normal == (1, 0, 0)
    assert result.ports["o1"].layer_key == "core_key"


def test_preserves_derived_geometry_and_authoritative_layer_fields() -> None:
    result = resolve_passive_pcell("demo", pdk=make_test_pdk())
    core = result.layers["core_key"]

    assert core.key == "core_key"
    assert core.declared_name == "core_declared_name"
    assert core.geometry.area == pytest.approx(0.875)
    assert core.zmin == 0
    assert core.thickness == 0.22
    assert core.sidewall_angle == 10
    assert core.width_to_z == 0.5
    assert core.bias == (0.01, 0.02)
    assert core.z_to_bias == ([0, 0.22], [0, 0.01])
    assert core.mesh_order == 1
    assert (10, 0) in result.derived_component.layers


def test_preexisting_derived_target_does_not_replace_source_geometry() -> None:
    component = demo_component()
    component.add_polygon(
        [(-10, -10), (10, -10), (10, 10), (-10, 10)],
        layer=(10, 0),
    )

    result = resolve_passive_pcell(component, pdk=make_test_pdk())

    assert result.layers["core_key"].geometry.area == pytest.approx(0.875)


def test_accepts_callable_and_instantiated_components() -> None:
    pdk = make_test_pdk()

    from_callable = resolve_passive_pcell(
        demo_component,
        pdk=pdk,
        settings={"length": 2.5},
    )
    from_instance = resolve_passive_pcell(demo_component(), pdk=pdk)

    assert from_callable.bounds[1][0] == 3.0
    assert from_instance.bounds[1][0] == 2.5


def test_accepts_pdk_module_cards_without_mutating_pdk_model() -> None:
    pdk = make_test_pdk(attach_cards=False)
    project_si = GSIM_MATERIAL_CARDS["Si-Li-293K"].model_copy(update={"name": "Si"})
    pdk_module = SimpleNamespace(PDK=pdk, MATERIAL_CARDS={"Si": project_si})

    result = resolve_passive_pcell("demo", pdk=pdk_module)

    assert result.materials["Si"].source == "project"
    assert not hasattr(pdk, "material_cards")


def test_uses_active_pdk_when_none_is_passed() -> None:
    original_pdk = gf.get_active_pdk()
    active_pdk = make_test_pdk()
    active_pdk.activate()
    try:
        result = resolve_passive_pcell("demo")
        assert result.pdk is active_pdk
    finally:
        original_pdk.activate()


def test_resolves_component_from_gdsfactory_generic_pdk() -> None:
    generic_si = GSIM_MATERIAL_CARDS["Si"].model_copy(update={"name": "si"})
    generic_pdk_module = SimpleNamespace(
        PDK=gf.gpdk.PDK,
        MATERIAL_CARDS={"si": generic_si},
    )

    result = resolve_passive_pcell(
        "straight",
        pdk=generic_pdk_module,
        settings={"length": 1.0},
    )

    assert set(result.layers) == {"core"}
    assert result.materials["si"].source == "project"
    assert set(result.ports) == {"o1", "o2"}


@pytest.mark.parametrize(
    ("core_material", "message"),
    [(None, "no material"), ("silicon", "No MaterialCard")],
)
def test_fails_without_material_card_or_legacy_fallback(
    core_material: str | None,
    message: str,
) -> None:
    with pytest.raises(LayerResolutionError, match=message):
        resolve_passive_pcell("demo", pdk=make_test_pdk(core_material=core_material))


def test_invalid_project_card_is_not_replaced_by_fallback() -> None:
    pdk = make_test_pdk()
    invalid_si = MaterialCard(name="Si", optical=None, rf=None, info={})
    object.__setattr__(pdk, "material_cards", {"Si": invalid_si})

    with pytest.raises(LayerResolutionError, match="no optical permittivity"):
        resolve_passive_pcell("demo", pdk=pdk)


def test_rejects_non_axis_aligned_ports() -> None:
    with pytest.raises(UnsupportedPortError, match="axis-aligned"):
        resolve_passive_pcell(
            "demo",
            pdk=make_test_pdk(port_orientation=45),
        )


def test_preserves_vertical_port_type_and_in_plane_polarization_angle() -> None:
    component = demo_component()
    component.add_port(
        name="fiber",
        center=(1.0, 0.1),
        width=10,
        orientation=45,
        layer=(1, 0),
        port_type="vertical_te",
    )

    result = resolve_passive_pcell(component, pdk=make_test_pdk())

    fiber = result.ports["fiber"]
    assert fiber.is_vertical
    assert fiber.port_type == "vertical_te"
    assert fiber.orientation == pytest.approx(45)
    assert fiber.normal == (0, 0, 1)


def test_rejects_settings_for_component_instance() -> None:
    with pytest.raises(ComponentResolutionError, match="Settings cannot"):
        resolve_passive_pcell(
            demo_component(),
            pdk=make_test_pdk(),
            settings={"length": 4},
        )
