"""Check stack conversion and validation using physical layer definitions."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from gdsfactory.technology import LayerLevel
from gdsfactory.technology import LayerStack as GfLayerStack

from gsim.common.stack import load_stack_yaml
from gsim.common.stack.extractor import (
    Layer,
    LayerStack,
    extract_from_pdk,
    extract_layer_stack,
)
from gsim.common.stack.materials import (
    MATERIALS_DB,
    DispersionModel,
    MaterialProperties,
)


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


@pytest.mark.parametrize("sidewall_angle", [-12.5, 0.0, 12.5])
def test_stack_yaml_round_trip_preserves_sidewall_angle(
    layers, tmp_path, sidewall_angle
):
    stack = LayerStack(pdk_name="test", layers={"metal": layers[2]})
    stack.layers["metal"].sidewall_angle = sidewall_angle
    path = tmp_path / "stack.yaml"
    stack.to_yaml(path)

    loaded_stack = load_stack_yaml(path)

    assert loaded_stack.layers["metal"].sidewall_angle == sidewall_angle


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


def _pdk_levels(specifications):
    return cast(
        Any,
        SimpleNamespace(
            layers={
                name: SimpleNamespace(
                    layer=layer,
                    zmin=zmin,
                    thickness=height,
                    material=material,
                    sidewall_angle=0,
                    info={"layer_type": kind},
                )
                for name, layer, zmin, height, material, kind in specifications
            }
        ),
    )


@pytest.mark.parametrize(
    ("photonic", "substrate", "box"),
    [
        (True, True, True),
        (True, False, True),
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_substrate_extraction_preserves_box_and_air_regions(photonic, substrate, box):
    specifications = [
        (
            "core" if photonic else "metal1",
            (1, 0),
            0,
            0.22,
            "silicon" if photonic else "aluminum",
            "dielectric" if photonic else "conductor",
        )
    ]
    if substrate:
        specifications.append(("substrate", (999, 0), -10, 8, "silicon", "substrate"))
    if box:
        specifications.append(("box", (999, 0), -2, 2, "SiO2", "dielectric"))
    stack = extract_layer_stack(
        _pdk_levels(specifications),
        include_substrate=True,
        substrate_thickness=5,
        air_below=2,
        air_above=1,
    )
    regions = {region["name"]: region for region in stack.dielectrics}
    assert (regions["substrate"]["zmin"], regions["substrate"]["zmax"]) == (
        -10 if photonic and substrate else -5,
        -2 if box else 0,
    )
    if box:
        assert (
            regions["box"]["zmin"],
            regions["box"]["zmax"],
            regions["box"]["material"],
        ) == (-2, 0, "SiO2")
    assert (regions["air_box_bottom"]["zmin"], regions["air_box_bottom"]["zmax"]) == (
        -7,
        -5,
    )
    assert regions["air_box"]["zmax"] == pytest.approx(1.22)


@pytest.mark.parametrize("kind", ["unknown", "dispersive", "conductive", "passivation"])
def test_bulk_background_uses_dielectric_properties_and_explicit_layer_types(
    monkeypatch, kind
):
    material = "unlisted-test-material"
    if kind != "unknown":
        properties = MaterialProperties(
            conductivity=[1e5, 1e5, 1e5] if kind == "conductive" else None,
            dispersion_models=[DispersionModel(type="constant", permittivity=3)]
            if kind == "dispersive"
            else [],
            permittivity=3 if kind == "passivation" else None,
        )
        monkeypatch.setitem(MATERIALS_DB, material, properties)
    name = "passivation" if kind == "passivation" else "region"
    stack = extract_layer_stack(
        _pdk_levels([(name, (1, 0), 0, 1, material, "dielectric")])
    )
    if kind == "conductive":
        assert stack.dielectrics[0]["material"] == "SiO2"
    else:
        assert any(region["material"] == material for region in stack.dielectrics)
    if kind == "unknown":
        assert stack.materials[material]["type"] == "unknown"


@pytest.mark.parametrize("gap", [-1, 0, 1])
def test_stack_validation_distinguishes_gaps_overlaps_and_planar_metals(layers, gap):
    stack = LayerStack.from_layer_list(layers)
    stack.layers["metal"].thickness = 0
    stack.dielectrics[1]["zmin"] = gap
    report = stack.validate_stack()
    assert bool(report) == (gap <= 0)
    assert "zero thickness" in str(report)
    assert ("Z-axis gap" in str(report)) == (gap > 0)
    assert ("Z-axis overlap" in str(report)) == (gap < 0)
    assert yaml.full_load(stack.to_yaml())["layers"]["metal"]["thickness"] == 0


def test_empty_stack_and_missing_pdk_are_reported():
    assert extract_layer_stack(_pdk_levels([])).layers == {}
    with pytest.raises(ValueError, match="Could not find layer stack"):
        extract_from_pdk(SimpleNamespace())
