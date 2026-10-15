"""Check stack diagrams and scale toggles against physical layer dimensions."""

from types import SimpleNamespace

import numpy as np
import pytest
from gdsfactory.technology import LayerLevel, LayerStack

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
