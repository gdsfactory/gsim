"""Thermal properties on the shared material registry."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError
from scipy.constants import c as C0  # noqa: N812

from gsim.common.stack import Layer, LayerStack
from gsim.common.stack.materials import (
    MATERIALS_DB,
    MaterialProperties,
    get_material_properties,
    thermal_properties,
)
from gsim.common.stack.overlays import load_overlay, merge_overlay
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models import BoundaryModeConfig


class TestThermalFields:
    def test_thermal_fields_round_trip_through_to_dict(self):
        props = MaterialProperties(
            permittivity=2.1,
            thermal_conductivity=1.38,
            specific_heat=709.0,
            mass_density=2203.0,
            dn_dT=9.5e-6,
        )
        d = props.to_dict()
        assert d["thermal_conductivity"] == 1.38
        assert d["specific_heat"] == 709.0
        assert d["mass_density"] == 2203.0
        assert d["dn_dT"] == 9.5e-6
        assert MaterialProperties.model_validate(d) == props

    def test_unset_thermal_fields_stay_out_of_to_dict(self):
        d = MaterialProperties(permittivity=2.1).to_dict()
        assert d == {"permittivity": 2.1}

    @pytest.mark.parametrize(
        "field", ["thermal_conductivity", "specific_heat", "mass_density"]
    )
    def test_negative_thermal_property_is_rejected(self, field):
        with pytest.raises(ValidationError, match=field):
            MaterialProperties(**{field: -1.0})

    def test_dn_dt_may_be_negative(self):
        assert MaterialProperties(dn_dT=-1e-5).dn_dT == -1e-5

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_dn_dt_must_be_finite(self, value):
        with pytest.raises(ValidationError, match="dn_dT"):
            MaterialProperties(dn_dT=value)


class TestDatabaseThermalValues:
    @pytest.mark.parametrize(
        ("name", "k", "c_p", "rho"),
        [
            ("silicon", 148.0, 711.0, 2330.0),
            ("SiO2", 1.38, 709.0, 2203.0),
            ("Si3N4", 2.5, 700.0, 3100.0),
            ("TiN", 28.0, 598.0, 5240.0),
            ("tungsten", 174.0, 132.0, 19300.0),
            ("aluminum", 237.0, 903.0, 2702.0),
            ("copper", 401.0, 385.0, 8933.0),
            ("air", 0.026, 1007.0, 1.177),
        ],
    )
    def test_heater_stack_materials_carry_thermal_constants(self, name, k, c_p, rho):
        props = get_material_properties(name)
        assert props is not None
        assert props.thermal_conductivity == pytest.approx(k)
        assert props.specific_heat == pytest.approx(c_p)
        assert props.mass_density == pytest.approx(rho)

    @pytest.mark.parametrize(
        ("name", "dn_dT"),
        [("si", 1.86e-4), ("oxide", 0.95e-5), ("nitride", 2.45e-5)],
    )
    def test_optical_materials_carry_dn_dt_at_1550(self, name, dn_dT):
        props = get_material_properties(name)
        assert props is not None
        assert props.dn_dT == pytest.approx(dn_dT)

    def test_tin_heater_keeps_its_electrical_conductivity(self):
        props = get_material_properties("TiN")
        assert props is not None
        assert props.conductivity is not None


def _stack_with(materials: dict) -> LayerStack:
    stack = LayerStack(pdk_name="test", units="um")
    stack.materials = materials
    return stack


class TestThermalPropertiesHelper:
    def test_database_supplies_a_material_the_stack_only_names(self):
        props = thermal_properties("si")
        assert props.thermal_conductivity == pytest.approx(148.0)
        assert props.dn_dT == pytest.approx(1.86e-4)

    def test_electrical_conductivity_comes_from_conductivity(self):
        assert thermal_properties("TiN").electrical_conductivity == pytest.approx(5e6)

    def test_stack_entry_wins_field_by_field_over_the_database(self):
        stack = _stack_with({"silicon": {"thermal_conductivity": 90.0}})
        props = thermal_properties("silicon", stack=stack)
        assert props.thermal_conductivity == pytest.approx(90.0)
        assert props.specific_heat == pytest.approx(711.0)

    def test_override_wins_over_the_stack(self):
        stack = _stack_with({"SiO2": {"thermal_conductivity": 1.3}})
        props = thermal_properties(
            "SiO2", stack=stack, overrides={"sio2": {"thermal_conductivity": 1.1}}
        )
        assert props.thermal_conductivity == pytest.approx(1.1)
        assert props.mass_density == pytest.approx(2203.0)

    def test_override_may_be_a_material_properties_record(self):
        props = thermal_properties(
            "TiN",
            overrides={"TiN": MaterialProperties(conductivity=1e6)},
        )
        assert props.electrical_conductivity == pytest.approx(1e6)
        assert props.thermal_conductivity == pytest.approx(28.0)

    def test_a_record_can_be_given_directly(self):
        props = thermal_properties(MaterialProperties(thermal_conductivity=3.0))
        assert props.thermal_conductivity == pytest.approx(3.0)
        assert props.specific_heat is None

    def test_missing_required_field_names_material_and_field(self):
        with pytest.raises(ValueError, match=r"poly_si.*thermal_conductivity"):
            thermal_properties("poly_si", required=("thermal_conductivity",))

    def test_custom_material_known_only_to_the_stack_resolves(self):
        stack = _stack_with({"heater_film": {"thermal_conductivity": 15.0}})
        props = thermal_properties("heater_film", stack=stack)
        assert props.thermal_conductivity == pytest.approx(15.0)

    def test_unknown_material_is_an_error_naming_it(self):
        with pytest.raises(ValueError, match="unobtainium"):
            thermal_properties("unobtainium")

    def test_unknown_required_field_is_an_error(self):
        with pytest.raises(ValueError, match="thermal_k"):
            thermal_properties("si", required=("thermal_k",))


class TestOverlaysCarryThermalFields:
    def test_overlay_yaml_sets_thermal_fields(self, tmp_path):
        path = tmp_path / "overlay.yaml"
        path.write_text(
            "materials:\n"
            "  SiO2:\n"
            "    permittivity: 4.0\n"
            "    thermal_conductivity: 1.2\n"
            "    dn_dT: 1.0e-5\n"
        )
        sio2 = load_overlay(path)["SiO2"]
        assert sio2.thermal_conductivity == pytest.approx(1.2)
        assert sio2.dn_dT == pytest.approx(1e-5)

    def test_merge_keeps_database_thermal_fields_the_overlay_leaves_unset(self):
        merged = merge_overlay({"SiO2": MaterialProperties(permittivity=4.0)})
        assert merged["SiO2"].thermal_conductivity == pytest.approx(1.38)
        assert merged["SiO2"].dn_dT == pytest.approx(0.95e-5)

    def test_merge_takes_the_overlay_thermal_value(self):
        merged = merge_overlay({"SiO2": MaterialProperties(thermal_conductivity=1.1)})
        assert merged["SiO2"].thermal_conductivity == pytest.approx(1.1)


class TestPalaceConfigIgnoresThermalFields:
    def test_no_thermal_key_reaches_the_palace_config(self, tmp_path):
        stack = _stack_with(
            {
                "silicon": MATERIALS_DB["silicon"].to_dict(),
                "SiO2": MATERIALS_DB["SiO2"].to_dict(),
            }
        )
        stack.layers["CORE"] = Layer(
            name="CORE",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=0.22,
            thickness=0.22,
            material="silicon",
            layer_type="dielectric",
        )
        groups = {
            "volumes": {
                "CORE": {"phys_group": 101, "is_shaped_dielectric": True},
                "SiO2": {"phys_group": 102},
            },
            "conductor_surfaces": {},
            "pec_surfaces": {},
            "port_surfaces": {},
            "boundary_surfaces": {},
        }
        config_path = generate_palace_config(
            groups=groups,
            ports=[],
            port_info=[],
            stack=stack,
            output_path=tmp_path,
            model_name="palace",
            fmax=100e9,
            simulation_type="boundarymode",
            boundary_mode_config=BoundaryModeConfig(
                freq=C0 / 1.55e-6, num_modes=2, save=1
            ),
            absorbing_boundary=False,
        )
        config = json.loads(config_path.read_text())
        entries = config["Domains"]["Materials"]
        assert {a for e in entries for a in e["Attributes"]} == {101, 102}
        text = config_path.read_text()
        for key in ("thermal_conductivity", "specific_heat", "mass_density", "dn_dT"):
            assert key not in text
