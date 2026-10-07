"""One rule for resolving a stack entry's material properties."""

from __future__ import annotations

import pytest

from gsim.common.stack import Layer, LayerStack
from gsim.common.stack.materials import (
    MaterialProperties,
    region_material_map,
    resolve_stack_material,
)

WL_UM = 1.55


class TestResolveStackMaterial:
    def test_the_stack_entry_wins_over_the_database(self):
        entry = {"permittivity": 7.5}
        resolved = resolve_stack_material("si", entry, WL_UM)
        assert resolved is not None
        assert resolved.permittivity_scalar == pytest.approx(7.5)

    def test_a_validated_entry_is_taken_as_it_stands(self):
        entry = MaterialProperties(permittivity=4.0)
        resolved = resolve_stack_material("sio2", entry, WL_UM)
        assert resolved is not None
        assert resolved.permittivity_scalar == pytest.approx(4.0)

    def test_the_database_answers_when_the_stack_carries_nothing(self):
        resolved = resolve_stack_material("si", None, WL_UM)
        assert resolved is not None
        assert resolved.permittivity_scalar is not None

    def test_an_entry_that_is_no_record_falls_through_to_the_database(self):
        resolved = resolve_stack_material("si", {"permittivity": "not a number"}, WL_UM)
        assert resolved is not None
        assert resolved.permittivity_scalar is not None

    def test_neither_resolving_is_reported_as_nothing(self):
        assert resolve_stack_material("unobtainium", None, WL_UM) is None

    def test_an_unusable_entry_for_an_unknown_material_is_nothing(self):
        assert (
            resolve_stack_material(
                "unobtainium", {"permittivity": "not a number"}, WL_UM
            )
            is None
        )


class TestRegionMaterialMap:
    def test_a_region_naming_a_layer_takes_that_layer_material(self):
        stack = LayerStack(pdk_name="test", units="um")
        stack.layers["core"] = Layer(
            name="core",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=0.22,
            thickness=0.22,
            material="si",
            layer_type="dielectric",
            mesh_resolution="fine",
        )
        assert region_material_map(stack, ["core", "sio2"]) == {
            "core": "si",
            "sio2": "sio2",
        }
