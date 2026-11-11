"""Check stack diagrams and scale toggles against physical layer dimensions."""

from types import SimpleNamespace

import numpy as np
import pytest
from gdsfactory.technology import LayerLevel, LayerStack

from gsim.common.stack._layer_utils import classify_layer_type
from gsim.common.stack.extractor import Layer as GsimLayer
from gsim.common.stack.extractor import LayerStack as GsimLayerStack
from gsim.common.stack.visualization import (
    parse_layer_stack,
    plot_stack,
    print_stack,
    print_stack_table,
)


@pytest.mark.parametrize("to_scale", [False, True])
def test_stack_diagrams_preserve_layer_order_dimensions_and_scale_toggle(to_scale):
    specifications = [
        ("substrate", -10, 10),
        ("active", 0, 0.2),
        ("poly", 0.1, 0.1),
        ("metal1", 1, 0.5),
        ("via1", 1.5, 0.005),
        ("topmetal2", 3, 1),
    ]
    stack = LayerStack(
        layers={
            name: LayerLevel(
                layer=(index + 1, 0), zmin=zmin, thickness=height, material="si"
            )
            for index, (name, zmin, height) in enumerate(specifications)
        }
    )
    layers = parse_layer_stack(stack)
    assert [layer.name for layer in layers] == [item[0] for item in specifications]
    assert [layer.layer_type for layer in layers] == [
        "substrate",
        "conductor",
        "conductor",
        "conductor",
        "via",
        "conductor",
    ]
    diagram = print_stack(SimpleNamespace(LAYER_STACK=stack))
    assert all(text in diagram for text in ["Substrate (Si)", "Metal1 (M1)", "5 nm"])
    table = print_stack_table(stack)
    assert table.index("topmetal2") < table.index("substrate")

    figure = plot_stack(stack, width=500, height=700, to_scale=to_scale)
    assert len(figure.data) == len(figure.layout.shapes) == len(layers)
    assert figure.layout.width == 500
    for index, (layer, shape) in enumerate(
        zip(layers, figure.layout.shapes, strict=True)
    ):
        np.testing.assert_allclose(
            [shape.y0, shape.y1],
            [layer.zmin, layer.zmax] if to_scale else [index, index + 1],
        )
        assert layer.name in figure.data[index].hovertext
    uniform, scaled = figure.layout.updatemenus[0].buttons
    assert [uniform.label, scaled.label] == ["Uniform", "To Scale"]
    np.testing.assert_allclose(
        scaled.args[0]["y"], [[layer.zmin + layer.thickness / 2] for layer in layers]
    )
    assert scaled.args[1]["yaxis"]["range"] == [-11, 5]


def test_empty_stack_diagrams_and_plain_layer_names():
    empty = LayerStack(layers={})
    assert print_stack(empty) == "No layers found in stack"
    assert plot_stack(empty).layout.annotations[0].text == "No layers found"
    stack = LayerStack(layers={"oxide": LayerLevel(layer=(1, 0), zmin=0, thickness=1)})
    assert "oxide" in print_stack(stack)
    assert "oxide" in print_stack_table(SimpleNamespace(LAYER_STACK=stack))


def _gsim_stack() -> GsimLayerStack:
    """A gsim LayerStack; ``m1_custom`` is named like a metal but typed via."""
    specifications = [
        # name, gds_layer, zmin, thickness, material, layer_type
        ("m1_custom", (5, 2), 0.5, 0.3, "aluminum", "via"),
        ("passivation", (7, 0), 1.5, 1.2, "sio2", "dielectric"),
        ("metal1", (8, 0), 0.0, 0.5, "aluminum", "conductor"),
        ("substrate", (1, 0), -2.0, 2.0, "si", "substrate"),
    ]
    return GsimLayerStack(
        layers={
            name: GsimLayer(
                name=name,
                gds_layer=gds_layer,
                zmin=zmin,
                zmax=zmin + thickness,
                thickness=thickness,
                material=material,
                layer_type=layer_type,
            )
            for name, gds_layer, zmin, thickness, material, layer_type in specifications
        }
    )


def test_gsim_layer_stack_keeps_its_own_layer_types_and_dimensions():
    stack = _gsim_stack()
    # The name alone would classify as a conductor; the stack says via.
    assert classify_layer_type("m1_custom") != "via"

    layers = parse_layer_stack(stack)
    assert [layer.name for layer in layers] == [
        "substrate",
        "metal1",
        "m1_custom",
        "passivation",
    ]
    assert [layer.layer_type for layer in layers] == [
        "substrate",
        "conductor",
        "via",
        "dielectric",
    ]
    assert [layer.gds_layer for layer in layers] == [1, 8, 5, 7]
    for layer in layers:
        source = stack.layers[layer.name]
        assert layer.zmin == source.zmin
        assert layer.thickness == source.thickness
        assert layer.zmax == pytest.approx(source.zmin + source.thickness)
        assert layer.material == source.material


def test_gsim_layer_stack_diagrams_and_plot():
    stack = _gsim_stack()
    layers = parse_layer_stack(stack)

    assert "m1_custom" in print_stack(stack)
    table = print_stack_table(stack)
    assert all(layer.name in table for layer in layers)
    assert table.index("passivation") < table.index("substrate")

    figure = plot_stack(stack)
    assert len(figure.data) == len(figure.layout.shapes) == len(layers)
    for layer, trace in zip(layers, figure.data, strict=True):
        assert layer.name in trace.hovertext
    custom = layers.index(next(layer for layer in layers if layer.name == "m1_custom"))
    assert "Type: via" in figure.data[custom].hovertext
    assert figure.layout.shapes[custom].fillcolor == "#87CEEB"
