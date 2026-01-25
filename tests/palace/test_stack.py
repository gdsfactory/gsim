"""Tests for layer stack extraction and validation."""

from __future__ import annotations

from gsim.common.stack import (
    MATERIALS_DB,
    Layer,
    LayerStack,
    ValidationResult,
    get_material_properties,
    material_is_conductor,
    material_is_dielectric,
)


class TestMaterials:
    """Test material property lookups."""

    def test_get_aluminum(self):
        """Test getting aluminum properties."""
        props = get_material_properties("aluminum")
        assert props is not None
        assert props.type == "conductor"
        assert props.conductivity == 3.77e7

    def test_get_copper(self):
        """Test getting copper properties."""
        props = get_material_properties("copper")
        assert props is not None
        assert props.type == "conductor"
        assert props.conductivity == 5.8e7

    def test_get_sio2(self):
        """Test getting SiO2 properties."""
        props = get_material_properties("SiO2")
        assert props is not None
        assert props.type == "dielectric"
        assert props.permittivity == 4.1

    def test_get_by_alias(self):
        """Test getting material by alias."""
        props = get_material_properties("al")
        assert props is not None
        assert props.type == "conductor"
        assert props.conductivity == 3.77e7

    def test_case_insensitive(self):
        """Test case-insensitive lookup."""
        props = get_material_properties("ALUMINUM")
        assert props is not None
        assert props.type == "conductor"

    def test_unknown_material(self):
        """Test unknown material returns None."""
        props = get_material_properties("unknown_material_xyz")
        assert props is None

    def test_material_is_conductor(self):
        """Test material_is_conductor function."""
        assert material_is_conductor("aluminum")
        assert material_is_conductor("copper")
        assert not material_is_conductor("SiO2")
        assert not material_is_conductor("air")

    def test_material_is_dielectric(self):
        """Test material_is_dielectric function."""
        assert material_is_dielectric("SiO2")
        assert material_is_dielectric("air")
        assert not material_is_dielectric("aluminum")
        assert not material_is_dielectric("copper")


class TestLayer:
    """Test Layer class."""

    def test_layer_creation(self):
        """Test creating a Layer."""
        layer = Layer(
            name="metal1",
            gds_layer=(8, 0),
            zmin=0.5,
            zmax=1.0,
            thickness=0.5,
            material="aluminum",
            layer_type="conductor",
        )
        assert layer.name == "metal1"
        assert layer.gds_layer == (8, 0)
        assert layer.thickness == 0.5
        assert layer.layer_type == "conductor"

    def test_layer_mesh_size_named(self):
        """Test mesh size calculation with named resolution."""
        layer = Layer(
            name="test",
            gds_layer=(1, 0),
            zmin=0,
            zmax=1,
            thickness=1,
            material="aluminum",
            layer_type="conductor",
            mesh_resolution="fine",
        )
        assert layer.get_mesh_size(base_size=2.0) == 1.0  # fine = 0.5 * base

    def test_layer_mesh_size_numeric(self):
        """Test mesh size calculation with numeric resolution."""
        layer = Layer(
            name="test",
            gds_layer=(1, 0),
            zmin=0,
            zmax=1,
            thickness=1,
            material="aluminum",
            layer_type="conductor",
            mesh_resolution=3.5,
        )
        assert layer.get_mesh_size(base_size=2.0) == 3.5

    def test_layer_to_dict(self):
        """Test layer serialization."""
        layer = Layer(
            name="metal1",
            gds_layer=(8, 0),
            zmin=0.5,
            zmax=1.0,
            thickness=0.5,
            material="aluminum",
            layer_type="conductor",
        )
        d = layer.to_dict()
        assert d["gds_layer"] == [8, 0]
        assert d["zmin"] == 0.5
        assert d["zmax"] == 1.0
        assert d["thickness"] == 0.5
        assert d["material"] == "aluminum"
        assert d["type"] == "conductor"


class TestLayerStack:
    """Test LayerStack class."""

    def test_empty_stack(self):
        """Test creating an empty stack."""
        stack = LayerStack(pdk_name="test")
        assert stack.pdk_name == "test"
        assert len(stack.layers) == 0
        assert len(stack.dielectrics) == 0

    def test_add_layers(self):
        """Test adding layers to stack."""
        stack = LayerStack(pdk_name="test")
        stack.layers["metal1"] = Layer(
            name="metal1",
            gds_layer=(8, 0),
            zmin=0.0,
            zmax=0.5,
            thickness=0.5,
            material="aluminum",
            layer_type="conductor",
        )
        assert len(stack.layers) == 1
        assert "metal1" in stack.layers

    def test_get_conductor_layers(self):
        """Test filtering conductor layers."""
        stack = LayerStack(pdk_name="test")
        stack.layers["metal1"] = Layer(
            name="metal1",
            gds_layer=(8, 0),
            zmin=0.0,
            zmax=0.5,
            thickness=0.5,
            material="aluminum",
            layer_type="conductor",
        )
        stack.layers["via1"] = Layer(
            name="via1",
            gds_layer=(9, 0),
            zmin=0.5,
            zmax=1.0,
            thickness=0.5,
            material="tungsten",
            layer_type="via",
        )
        conductors = stack.get_conductor_layers()
        assert len(conductors) == 1
        assert "metal1" in conductors

    def test_get_via_layers(self):
        """Test filtering via layers."""
        stack = LayerStack(pdk_name="test")
        stack.layers["metal1"] = Layer(
            name="metal1",
            gds_layer=(8, 0),
            zmin=0.0,
            zmax=0.5,
            thickness=0.5,
            material="aluminum",
            layer_type="conductor",
        )
        stack.layers["via1"] = Layer(
            name="via1",
            gds_layer=(9, 0),
            zmin=0.5,
            zmax=1.0,
            thickness=0.5,
            material="tungsten",
            layer_type="via",
        )
        vias = stack.get_via_layers()
        assert len(vias) == 1
        assert "via1" in vias


class TestValidation:
    """Test stack validation."""

    def test_validation_result_bool(self):
        """Test ValidationResult truthiness."""
        valid = ValidationResult(valid=True)
        invalid = ValidationResult(valid=False)
        assert bool(valid) is True
        assert bool(invalid) is False

    def test_validation_result_str(self):
        """Test ValidationResult string representation."""
        result = ValidationResult(
            valid=False,
            errors=["Test error"],
            warnings=["Test warning"],
        )
        s = str(result)
        assert "FAILED" in s
        assert "Test error" in s
        assert "Test warning" in s

    def test_valid_stack(self):
        """Test validation of a valid stack."""
        stack = LayerStack(pdk_name="test")

        # Add material
        stack.materials["aluminum"] = MATERIALS_DB["aluminum"].to_dict()
        stack.materials["SiO2"] = MATERIALS_DB["SiO2"].to_dict()
        stack.materials["air"] = MATERIALS_DB["air"].to_dict()

        # Add layer
        stack.layers["metal1"] = Layer(
            name="metal1",
            gds_layer=(8, 0),
            zmin=0.0,
            zmax=0.5,
            thickness=0.5,
            material="aluminum",
            layer_type="conductor",
        )

        # Add dielectrics
        stack.dielectrics = [
            {"name": "oxide", "zmin": -2.0, "zmax": 0.5, "material": "SiO2"},
            {"name": "air_box", "zmin": 0.5, "zmax": 100.0, "material": "air"},
        ]

        result = stack.validate_stack()
        assert result.valid

    def test_missing_material(self):
        """Test validation catches missing material."""
        stack = LayerStack(pdk_name="test")
        stack.layers["metal1"] = Layer(
            name="metal1",
            gds_layer=(8, 0),
            zmin=0.0,
            zmax=0.5,
            thickness=0.5,
            material="undefined_material",
            layer_type="conductor",
        )
        stack.dielectrics = [
            {
                "name": "oxide",
                "zmin": -2.0,
                "zmax": 100.0,
                "material": "undefined_dielectric",
            },
        ]

        result = stack.validate_stack()
        assert not result.valid
        assert any("undefined_material" in e for e in result.errors)

    def test_negative_thickness(self):
        """Test validation catches negative thickness."""
        stack = LayerStack(pdk_name="test")
        stack.materials["aluminum"] = MATERIALS_DB["aluminum"].to_dict()
        stack.layers["metal1"] = Layer(
            name="metal1",
            gds_layer=(8, 0),
            zmin=0.0,
            zmax=-0.5,
            thickness=-0.5,
            material="aluminum",
            layer_type="conductor",
        )
        stack.dielectrics = []

        result = stack.validate_stack()
        assert not result.valid
        assert any("negative thickness" in e for e in result.errors)

    def test_no_dielectrics(self):
        """Test validation catches missing dielectrics."""
        stack = LayerStack(pdk_name="test")
        stack.materials["aluminum"] = MATERIALS_DB["aluminum"].to_dict()
        stack.layers["metal1"] = Layer(
            name="metal1",
            gds_layer=(8, 0),
            zmin=0.0,
            zmax=0.5,
            thickness=0.5,
            material="aluminum",
            layer_type="conductor",
        )
        stack.dielectrics = []

        result = stack.validate_stack()
        assert not result.valid
        assert any("No dielectric" in e for e in result.errors)
