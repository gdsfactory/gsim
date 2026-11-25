"""Sheet impedance on the planar sheets of a conductor layer.

Zero-thickness conductors are written under ``Boundaries.PEC``. ``add_sheet_impedance``
moves their attributes to ``Boundaries.Impedance`` with per-square ``Ls``/``Rs``.
The synthetic tests drive ``generate_palace_config`` with hand-built groups; the
meshed tests check the real physical-group keys and ``validate_mesh``.
"""

from __future__ import annotations

import json
import logging

import gdsfactory as gf
import pytest
from pydantic import ValidationError

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import (
    BoundaryModeSim,
    DrivenSim,
    EigenmodeSim,
    ElectrostaticSim,
    MagnetostaticSim,
)
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models import DrivenConfig, SheetImpedanceConfig
from gsim.palace.ports.config import PalacePort, PortType

LS = 20e-12  # H/sq, a thin superconducting film
RS = 0.5  # ohm/sq


def _stack(thickness: float = 0.0) -> LayerStack:
    stack = LayerStack()
    stack.layers["sc"] = Layer(
        name="sc",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=thickness,
        thickness=thickness,
        material="niobium",
        layer_type="conductor",
    )
    stack.layers["oxide"] = Layer(
        name="oxide",
        gds_layer=(2, 0),
        zmin=0.0,
        zmax=1.0,
        thickness=1.0,
        material="sio2",
        layer_type="dielectric",
    )
    stack.layers["metal"] = Layer(
        name="metal",
        gds_layer=(3, 0),
        zmin=0.0,
        zmax=3.0,
        thickness=3.0,
        material="aluminum",
        layer_type="conductor",
    )
    stack.materials = {
        "niobium": {"conductivity": 0.0},
        "aluminum": {"conductivity": 3.77e7},
        "sio2": {"permittivity": 3.9},
    }
    return stack


def _groups(pec_surfaces: dict | None = None, conductor_surfaces: dict | None = None):
    return {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": conductor_surfaces
        if conductor_surfaces is not None
        else {"metal_xy": {"phys_group": 4}, "metal_z": {"phys_group": 5}},
        "pec_surfaces": pec_surfaces
        if pec_surfaces is not None
        else {"sc": {"phys_group": 9}, "pec_block_0": {"phys_group": 11}},
        "port_surfaces": {},
        "boundary_surfaces": {"absorbing": {"phys_group": 8}},
    }


def _generate(
    tmp_path,
    groups,
    *,
    sheets=None,
    simulation_type="driven",
    stack=None,
    ports=None,
    driven=None,
):
    config_path = generate_palace_config(
        groups=groups,
        ports=ports or [],
        port_info=[],
        stack=stack if stack is not None else _stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type=simulation_type,
        absorbing_boundary=True,
        sheet_impedances=sheets,
        driven_config=driven,
    )
    return json.loads(config_path.read_text())


def _sheets(**values) -> list[SheetImpedanceConfig]:
    return [SheetImpedanceConfig(layer="sc", **values)]


def _all_attributes(boundaries: dict) -> list[int]:
    found: list[int] = []
    for key, value in boundaries.items():
        if key in ("Terminal", "Postprocessing"):
            continue
        entries = value if isinstance(value, list) else [value]
        for entry in entries:
            found.extend(entry.get("Attributes", []))
    return found


class TestConfigWriting:
    def test_sheet_moves_from_pec_to_impedance(self, tmp_path):
        config = _generate(
            tmp_path,
            _groups(),
            sheets=_sheets(inductance=LS, resistance=RS),
        )
        boundaries = config["Boundaries"]
        assert boundaries["PEC"] == {"Attributes": [11]}
        assert boundaries["Impedance"] == [
            {"Attributes": [9], "Ls": LS, "Rs": RS},
        ]

    def test_values_are_per_square_and_unscaled(self, tmp_path):
        config = _generate(tmp_path, _groups(), sheets=_sheets(inductance=LS))
        entry = config["Boundaries"]["Impedance"][0]
        assert entry["Ls"] == LS
        assert "Rs" not in entry

    def test_sheets_are_not_cracked(self, tmp_path):
        config = _generate(
            tmp_path, _groups(), sheets=_sheets(inductance=LS, resistance=RS)
        )
        boundaries = config["Boundaries"]
        assert all("Crack" not in entry for entry in boundaries["Impedance"])
        assert "Crack" not in boundaries["PEC"]

    def test_no_attribute_under_two_boundary_types(self, tmp_path):
        config = _generate(
            tmp_path, _groups(), sheets=_sheets(inductance=LS, resistance=RS)
        )
        attributes = _all_attributes(config["Boundaries"])
        assert len(attributes) == len(set(attributes))

    def test_other_layers_stay_pec(self, tmp_path):
        pec_surfaces = {
            "sc": {"phys_group": 9},
            "other": {"phys_group": 12},
        }
        config = _generate(
            tmp_path,
            _groups(pec_surfaces=pec_surfaces),
            sheets=_sheets(inductance=LS),
        )
        assert config["Boundaries"]["PEC"] == {"Attributes": [12]}

    def test_resistance_only(self, tmp_path):
        config = _generate(tmp_path, _groups(), sheets=_sheets(resistance=RS))
        assert config["Boundaries"]["Impedance"] == [
            {"Attributes": [9], "Rs": RS},
        ]

    @pytest.mark.parametrize("simulation_type", ["driven", "eigenmode"])
    def test_driven_and_eigenmode(self, tmp_path, simulation_type):
        config = _generate(
            tmp_path,
            _groups(),
            sheets=_sheets(inductance=LS),
            simulation_type=simulation_type,
        )
        assert config["Boundaries"]["Impedance"] == [{"Attributes": [9], "Ls": LS}]

    def test_no_sheet_config_leaves_pec_unchanged(self, tmp_path):
        config = _generate(tmp_path, _groups(), sheets=None)
        assert config["Boundaries"]["PEC"] == {"Attributes": [9, 11]}
        assert "Impedance" not in config["Boundaries"]

    def test_sheet_in_boundarymode_with_zero_thickness_sheet(self, tmp_path):
        groups = _groups(conductor_surfaces={})
        config = _generate(
            tmp_path,
            groups,
            sheets=_sheets(inductance=LS),
            simulation_type="boundarymode",
            stack=_stack(thickness=0.0),
        )
        boundaries = config["Boundaries"]
        assert boundaries["Impedance"] == [{"Attributes": [9], "Ls": LS}]
        assert boundaries["PEC"] == {"Attributes": [11]}


def _wave_port_groups() -> dict:
    groups = _groups()
    groups["port_surfaces"] = {"P1": {"phys_group": 6}, "P2": {"phys_group": 7}}
    return groups


def _wave_ports() -> list[PalacePort]:
    return [
        PalacePort(name="o1", port_type=PortType.WAVEPORT, layer="sc"),
        PalacePort(name="o2", port_type=PortType.WAVEPORT, layer="sc", excited=False),
    ]


class TestWavePorts:
    def test_sheets_stay_out_of_waveport_pec(self, tmp_path):
        config = _generate(
            tmp_path,
            _wave_port_groups(),
            sheets=_sheets(inductance=LS),
            ports=_wave_ports(),
        )
        boundaries = config["Boundaries"]
        assert boundaries["Impedance"] == [{"Attributes": [9], "Ls": LS}]
        assert boundaries["WavePortPEC"] == {"Attributes": [4, 5, 8]}

    def test_other_impedance_boundaries_still_listed(self, tmp_path):
        from gsim.palace.models.ports import ImpedanceBoundaryConfig

        config_path = generate_palace_config(
            groups=_wave_port_groups(),
            ports=_wave_ports(),
            port_info=[],
            stack=_stack(),
            output_path=tmp_path,
            model_name="palace",
            fmax=100e9,
            simulation_type="driven",
            absorbing_boundary=True,
            sheet_impedances=_sheets(inductance=LS),
            impedance_boundaries=[
                ImpedanceBoundaryConfig(attributes=[20], resistance=1.0)
            ],
        )
        boundaries = json.loads(config_path.read_text())["Boundaries"]
        assert boundaries["WavePortPEC"] == {"Attributes": [4, 5, 8, 20]}

    def test_no_waveport_pec_when_only_sheets_remain(self, tmp_path):
        groups = _wave_port_groups()
        groups["conductor_surfaces"] = {}
        groups["boundary_surfaces"] = {}
        config = _generate(
            tmp_path,
            groups,
            sheets=_sheets(inductance=LS),
            ports=_wave_ports(),
        )
        assert "WavePortPEC" not in config["Boundaries"]

    def test_inherit_policy_also_keeps_sheets_out(self, tmp_path):
        config = _generate(
            tmp_path,
            _wave_port_groups(),
            sheets=_sheets(inductance=LS),
            ports=_wave_ports(),
            driven=DrivenConfig(waveport_boundary="inherit"),
        )
        assert "WavePortPEC" not in config["Boundaries"]
        assert config["Boundaries"]["Impedance"] == [{"Attributes": [9], "Ls": LS}]

    def test_warns_about_the_port_mode_target(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            _generate(
                tmp_path,
                _wave_port_groups(),
                sheets=_sheets(inductance=LS),
                ports=_wave_ports(),
            )
        assert "mode search target" in caplog.text
        assert "'sc'" in caplog.text

    def test_no_warning_without_wave_ports(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            _generate(tmp_path, _groups(), sheets=_sheets(inductance=LS))
        assert "mode search target" not in caplog.text


class TestErrors:
    def test_dielectric_layer_rejected(self, tmp_path):
        sheets = [SheetImpedanceConfig(layer="oxide", inductance=LS)]
        with pytest.raises(ValueError, match="must be a conductor layer"):
            _generate(tmp_path, _groups(), sheets=sheets)

    def test_unknown_layer_rejected(self, tmp_path):
        sheets = [SheetImpedanceConfig(layer="nope", inductance=LS)]
        with pytest.raises(ValueError, match="not in the stack"):
            _generate(tmp_path, _groups(), sheets=sheets)

    def test_thick_conductor_volume_rejected(self, tmp_path):
        groups = _groups(
            pec_surfaces={"pec_block_0": {"phys_group": 11}},
            conductor_surfaces={
                "sc_xy": {"phys_group": 4},
                "sc_z": {"phys_group": 5},
            },
        )
        with pytest.raises(ValueError, match="meshed as a volume"):
            _generate(tmp_path, groups, sheets=_sheets(inductance=LS))

    def test_layer_without_sheets_rejected(self, tmp_path):
        groups = _groups(pec_surfaces={"pec_block_0": {"phys_group": 11}})
        with pytest.raises(ValueError, match="has no planar sheets"):
            _generate(tmp_path, groups, sheets=_sheets(inductance=LS))

    def test_thick_layer_rejected_in_boundarymode(self, tmp_path):
        groups = _groups(conductor_surfaces={})
        with pytest.raises(ValueError, match="meshed as a volume"):
            _generate(
                tmp_path,
                groups,
                sheets=_sheets(inductance=LS),
                simulation_type="boundarymode",
                stack=_stack(thickness=3.0),
            )

    def test_electrostatic_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="electrostatic"):
            _generate(
                tmp_path,
                _groups(),
                sheets=_sheets(inductance=LS),
                simulation_type="electrostatic",
            )

    def test_magnetostatic_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="magnetostatic"):
            _generate(
                tmp_path,
                _groups(),
                sheets=_sheets(inductance=LS),
                simulation_type="magnetostatic",
            )


class TestSheetImpedanceConfig:
    def test_requires_a_value(self):
        with pytest.raises(ValueError, match="inductance"):
            SheetImpedanceConfig(layer="sc")

    @pytest.mark.parametrize("field", ["inductance", "resistance"])
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_rejects_non_positive(self, field, value):
        with pytest.raises(ValidationError):
            SheetImpedanceConfig(layer="sc", **{field: value})

    def test_zero_with_a_valid_value_is_rejected(self):
        with pytest.raises(ValidationError):
            SheetImpedanceConfig(layer="sc", inductance=LS, resistance=0.0)

    def test_round_trip(self):
        original = SheetImpedanceConfig(layer="sc", inductance=LS, resistance=RS)
        restored = SheetImpedanceConfig.model_validate_json(original.model_dump_json())
        assert restored == original


class TestSimApi:
    def test_stored_per_layer_and_replaced(self):
        sim = DrivenSim()
        sim.add_sheet_impedance("sc", inductance=LS)
        sim.add_sheet_impedance("sc", inductance=2 * LS, resistance=RS)
        assert list(sim._sheet_impedances) == ["sc"]
        stored = sim._sheet_impedances["sc"]
        assert stored.inductance == 2 * LS
        assert stored.resistance == RS

    def test_state_is_per_instance(self):
        first = DrivenSim()
        second = DrivenSim()
        first.add_sheet_impedance("sc", inductance=LS)
        assert second._sheet_impedances == {}

    def test_requires_a_value(self):
        with pytest.raises(ValueError):
            DrivenSim().add_sheet_impedance("sc")

    @pytest.mark.parametrize("field", ["inductance", "resistance"])
    def test_zero_rejected_at_call(self, field):
        with pytest.raises(ValueError):
            DrivenSim().add_sheet_impedance("sc", **{field: 0.0})

    def test_electrostatic_rejected_at_call(self):
        sim = ElectrostaticSim()
        with pytest.raises(ValueError, match="electrostatic"):
            sim.add_sheet_impedance("sc", inductance=LS)

    def test_magnetostatic_rejected_at_call(self):
        sim = MagnetostaticSim()
        with pytest.raises(ValueError, match="magnetostatic"):
            sim.add_sheet_impedance("sc", inductance=LS)

    @pytest.mark.parametrize("sim_cls", [DrivenSim, EigenmodeSim, BoundaryModeSim])
    def test_accepted_by_supported_sims(self, sim_cls):
        sim = sim_cls()
        sim.add_sheet_impedance("sc", inductance=LS)
        assert sim._sheet_impedances["sc"].inductance == LS


def _thin_cpw_component() -> gf.Component:
    gf.gpdk.PDK.activate()
    layer = gf.gpdk.LAYER.M1
    component = gf.Component()
    for y0, y1 in ((-60.0, -16.0), (-10.0, 10.0), (16.0, 60.0)):
        component.add_polygon(
            [(-50.0, y0), (50.0, y0), (50.0, y1), (-50.0, y1)], layer=layer
        )
    component.add_port(
        name="o1",
        center=(-50.0, 0),
        width=10.0,
        orientation=180,
        port_type="electrical",
        layer=layer,
    )
    component.add_port(
        name="o2",
        center=(50.0, 0),
        width=10.0,
        orientation=0,
        port_type="electrical",
        layer=layer,
    )
    return component


def _mesh_driven(tmp_path, *, planar_conductors: bool) -> DrivenSim:
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(_thin_cpw_component())
    sim.set_stack(substrate_thickness=2.0, air_above=300.0)
    sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
    sim.mesh(preset="coarse", planar_conductors=planar_conductors)
    return sim


@pytest.fixture(scope="module")
def planar_sim(tmp_path_factory):
    return _mesh_driven(tmp_path_factory.mktemp("planar"), planar_conductors=True)


@pytest.fixture(scope="module")
def volumetric_sim(tmp_path_factory):
    return _mesh_driven(tmp_path_factory.mktemp("volumetric"), planar_conductors=False)


class TestMeshedDriven:
    def test_planar_metal_becomes_impedance_sheet(self, planar_sim):
        planar_sim.add_sheet_impedance("metal1", inductance=LS, resistance=RS)
        config = json.loads(planar_sim.write_config().read_text())
        boundaries = config["Boundaries"]
        impedance = boundaries["Impedance"]
        assert len(impedance) == 1
        assert impedance[0]["Ls"] == LS
        assert impedance[0]["Rs"] == RS
        pec = set(boundaries.get("PEC", {}).get("Attributes", []))
        assert not pec & set(impedance[0]["Attributes"])
        assert "Crack" not in json.dumps(config)
        attributes = _all_attributes(boundaries)
        assert len(attributes) == len(set(attributes))

    def test_only_conductor_is_impedance_sheet_validates(self, planar_sim):
        planar_sim.add_sheet_impedance("metal1", inductance=LS)
        assert planar_sim.validate_mesh().valid

    def test_thick_metal_rejected(self, volumetric_sim):
        volumetric_sim.add_sheet_impedance("metal1", inductance=LS)
        with pytest.raises(ValueError, match="meshed as a volume"):
            volumetric_sim.write_config()

    def test_cache_input_changes_with_value(self, planar_sim):
        planar_sim.add_sheet_impedance("metal1", inductance=LS)
        first = planar_sim.write_config().read_bytes()
        planar_sim.add_sheet_impedance("metal1", inductance=2 * LS)
        second = planar_sim.write_config().read_bytes()
        assert first != second
