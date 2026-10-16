"""Check stack conversion and validation using physical layer definitions."""

from types import SimpleNamespace

import pytest
import yaml
from gdsfactory.technology import LayerLevel
from gdsfactory.technology import LayerStack as GfLayerStack

from gsim.common.stack.extractor import Layer, LayerStack, extract_from_pdk


@pytest.fixture
def layers():
    return [
        Layer(
            name=name,
            gds_layer=(index, 0),
            zmin=zmin,
            zmax=zmax,
            thickness=zmax - zmin,
            material=material,
            layer_type=kind,
        )
        for index, (name, zmin, zmax, material, kind) in enumerate(
            [
                ("substrate", -2, 0, "silicon", "dielectric"),
                ("oxide", 0, 2, "SiO2", "dielectric"),
                ("metal", 0.5, 1, "aluminum", "conductor"),
            ],
            start=1,
        )
    ]


def test_layer_list_round_trips_to_yaml_without_losing_physical_dimensions(
    layers, tmp_path
):
    stack = LayerStack.from_layer_list(layers)
    assert stack.validate_stack()
    assert stack.get_z_range() == (-2, 2)
    path = tmp_path / "nested" / "stack.yaml"
    document = yaml.full_load(stack.to_yaml(path))
    assert document == yaml.full_load(path.read_text()) == stack.to_dict()
    assert document["layers"]["metal"]["gds_layer"] == [3, 0]
    assert document["layers"]["metal"]["thickness"] == 0.5
    assert set(document["materials"]) == {"silicon", "SiO2", "aluminum"}
    assert set(stack.get_conductor_layers()) == {"metal"}


@pytest.mark.parametrize("duplicate", [False, True])
def test_layer_list_rejects_empty_or_duplicate_names(layers, duplicate):
    with pytest.raises(ValueError, match=r"Duplicate|empty"):
        LayerStack.from_layer_list([layers[0], layers[0]] if duplicate else [])


@pytest.mark.parametrize(
    "issue", ["negative", "unknown", "conductivity", "permittivity", "outside"]
)
def test_invalid_stacks_report_the_physical_problem(layers, issue):
    stack = LayerStack.from_layer_list(layers)
    expected = {
        "negative": "negative thickness",
        "unknown": "not defined",
        "conductivity": "missing conductivity",
        "permittivity": "missing permittivity",
        "outside": "below dielectric envelope",
    }[issue]
    if issue == "negative":
        stack.layers["metal"].thickness = -1
    elif issue == "unknown":
        del stack.materials["aluminum"]
    elif issue == "conductivity":
        stack.materials["aluminum"] = {"type": "conductor"}
    elif issue == "permittivity":
        stack.materials["SiO2"] = {"type": "dielectric"}
    else:
        stack.layers["metal"].zmin = -3
    report = stack.validate_stack()
    assert not report
    assert expected in str(report)


@pytest.mark.parametrize("wrapper", ["direct", "module", "getter", "nested"])
def test_pdk_wrapper_variants_keep_the_same_layer_coordinates(wrapper, tmp_path):
    stack = GfLayerStack(
        layers={"core": LayerLevel(layer=(1, 0), zmin=0, thickness=0.22, material="si")}
    )
    pdk = {
        "direct": SimpleNamespace(name="test", layer_stack=stack),
        "module": SimpleNamespace(__name__="test", LAYER_STACK=stack),
        "getter": SimpleNamespace(get_layer_stack=lambda: stack),
        "nested": SimpleNamespace(PDK=SimpleNamespace(name="test", layer_stack=stack)),
    }[wrapper]
    result = extract_from_pdk(pdk, output_path=tmp_path / "stack.yml")
    assert result.layers["core"].zmax == 0.22
    assert (
        yaml.full_load((tmp_path / "stack.yml").read_text())["layers"]["core"][
            "material"
        ]
        == "si"
    )
