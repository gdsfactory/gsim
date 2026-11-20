"""Tests for WavePortPEC emission with numeric wave ports.

With ``waveport_boundary="pec"`` a model with numeric wave ports lists every
boundary attribute that would otherwise contribute a Robin term to the 2D
port eigenproblem (absorbing walls, finite-conductivity conductors, and
impedance boundaries) under ``Boundaries.WavePortPEC``. The default
"inherit" policy keeps those physical port boundaries, and so does a driven
run without a DrivenConfig. Eigenmode runs keep the PEC approximation.
"""

from __future__ import annotations

import json

import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import DrivenSim
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models import DrivenConfig
from gsim.palace.models.ports import ImpedanceBoundaryConfig
from gsim.palace.ports.config import PalacePort, PortType


def _stack_with_metal() -> LayerStack:
    stack = LayerStack()
    stack.layers["metal"] = Layer(
        name="metal",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=3.0,
        thickness=3.0,
        material="aluminum",
        layer_type="conductor",
    )
    stack.materials = {"aluminum": {"conductivity": 3.77e7}}
    return stack


def _groups(
    *,
    wave_ports: bool = True,
    conductors: bool = True,
    absorbing: bool = True,
) -> dict:
    groups: dict = {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }
    if conductors:
        groups["conductor_surfaces"] = {
            "metal_xy": {"phys_group": 4},
            "metal_z": {"phys_group": 5},
        }
    if wave_ports:
        groups["port_surfaces"] = {
            "P1": {"phys_group": 6},
            "P2": {"phys_group": 7},
        }
    if absorbing:
        groups["boundary_surfaces"] = {"absorbing": {"phys_group": [8, 12]}}
    return groups


def _wave_ports() -> list[PalacePort]:
    return [
        PalacePort(name="o1", port_type=PortType.WAVEPORT, layer="metal"),
        PalacePort(
            name="o2", port_type=PortType.WAVEPORT, layer="metal", excited=False
        ),
    ]


def _lumped_ports() -> list[PalacePort]:
    return [
        PalacePort(name="o1", port_type=PortType.LUMPED, layer="metal"),
        PalacePort(name="o2", port_type=PortType.LUMPED, layer="metal", excited=False),
    ]


def _generate(
    tmp_path,
    groups,
    ports,
    *,
    impedance_boundaries=None,
    absorbing_boundary=True,
    driven=None,
    simulation_type="driven",
):
    config_path = generate_palace_config(
        groups=groups,
        ports=ports,
        port_info=[],
        stack=_stack_with_metal(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type=simulation_type,
        absorbing_boundary=absorbing_boundary,
        driven_config=driven,
        impedance_boundaries=impedance_boundaries,
    )
    return json.loads(config_path.read_text())


class TestWavePortPEC:
    def test_inherit_keeps_all_physical_boundaries(self, tmp_path):
        impedance = [ImpedanceBoundaryConfig(attributes=[20], resistance=1.0)]
        sim = DrivenSim()
        sim.set_driven(waveport_boundary="pec")
        pec = _generate(
            tmp_path,
            _groups(),
            _wave_ports(),
            impedance_boundaries=impedance,
            driven=sim.driven,
        )
        sim.set_driven(waveport_boundary="inherit")
        inherited = _generate(
            tmp_path,
            _groups(),
            _wave_ports(),
            impedance_boundaries=impedance,
            driven=sim.driven,
        )
        assert pec["Boundaries"].pop("WavePortPEC") == {"Attributes": [4, 5, 8, 12, 20]}
        assert inherited == pec
        assert "waveport_boundary" not in inherited["Solver"]["Driven"]

    def test_default_is_inherit(self, tmp_path):
        """Since 0.11.0 a driven run keeps the physical boundaries in the
        port eigenproblem unless it asks for "pec"."""
        assert DrivenConfig().waveport_boundary == "inherit"
        sim = DrivenSim()
        sim.set_driven(fmin=1e9, fmax=10e9)
        assert sim.driven.waveport_boundary == "inherit"
        config = _generate(tmp_path, _groups(), _wave_ports(), driven=sim.driven)
        assert config["Boundaries"]["WavePort"]
        assert config["Boundaries"]["Conductivity"]
        assert "WavePortPEC" not in config["Boundaries"]

    def test_pec_restores_the_old_default(self, tmp_path):
        config = _generate(
            tmp_path,
            _groups(),
            _wave_ports(),
            driven=DrivenConfig(waveport_boundary="pec"),
        )
        assert config["Boundaries"]["WavePortPEC"] == {"Attributes": [4, 5, 8, 12]}

    def test_driven_without_driven_settings_is_inherit(self, tmp_path):
        """The low-level config writers use the DrivenConfig default when a
        driven run passes no DrivenConfig."""
        config = _generate(tmp_path, _groups(), _wave_ports(), driven=None)
        assert config["Boundaries"]["WavePort"]
        assert "WavePortPEC" not in config["Boundaries"]

    def test_eigenmode_port_solve_stays_pec(self, tmp_path):
        """Eigenmode runs have no driven settings and cannot choose a policy,
        so their wave ports keep the PEC approximation."""
        config = _generate(
            tmp_path, _groups(), _wave_ports(), simulation_type="eigenmode"
        )
        assert config["Boundaries"]["WavePortPEC"] == {"Attributes": [4, 5, 8, 12]}

    def test_a_saved_pec_setting_still_loads_as_pec(self):
        saved = '{"waveport_boundary": "pec"}'
        assert DrivenConfig.model_validate_json(saved).waveport_boundary == "pec"

    def test_saved_settings_without_the_key_load_as_inherit(self):
        """Settings saved before the option existed take the new default."""
        assert DrivenConfig.model_validate_json("{}").waveport_boundary == "inherit"

    @pytest.mark.parametrize("policy", ["pec", "inherit"])
    def test_policy_preserved_by_serialization(self, policy):
        original = DrivenConfig(waveport_boundary=policy)
        restored = DrivenConfig.model_validate_json(original.model_dump_json())
        assert restored.waveport_boundary == policy

    def test_rejects_unknown_policy(self):
        with pytest.raises(ValueError, match="waveport_boundary"):
            DrivenSim().set_driven(waveport_boundary="automatic")

    def test_union_of_absorbing_and_conductivity(self, tmp_path):
        config = _generate(
            tmp_path,
            _groups(),
            _wave_ports(),
            driven=DrivenConfig(waveport_boundary="pec"),
        )
        boundaries = config["Boundaries"]
        assert boundaries["WavePortPEC"] == {"Attributes": [4, 5, 8, 12]}

    def test_not_emitted_without_wave_ports(self, tmp_path):
        config = _generate(
            tmp_path,
            _groups(wave_ports=False),
            _lumped_ports(),
            driven=DrivenConfig(waveport_boundary="pec"),
        )
        assert "WavePortPEC" not in config["Boundaries"]

    def test_not_emitted_for_lumped_ports(self, tmp_path):
        groups = _groups()
        config = _generate(
            tmp_path,
            groups,
            _lumped_ports(),
            driven=DrivenConfig(waveport_boundary="pec"),
        )
        assert config["Boundaries"]["LumpedPort"]
        assert not config["Boundaries"]["WavePort"]
        assert "WavePortPEC" not in config["Boundaries"]

    def test_includes_impedance_attributes(self, tmp_path):
        impedance = [ImpedanceBoundaryConfig(attributes=[20], resistance=1.0)]
        config = _generate(
            tmp_path,
            _groups(),
            _wave_ports(),
            impedance_boundaries=impedance,
            driven=DrivenConfig(waveport_boundary="pec"),
        )
        assert config["Boundaries"]["WavePortPEC"] == {"Attributes": [4, 5, 8, 12, 20]}

    def test_not_emitted_when_no_robin_boundaries(self, tmp_path):
        groups = _groups(conductors=False, absorbing=False)
        config = _generate(
            tmp_path,
            groups,
            _wave_ports(),
            absorbing_boundary=False,
            driven=DrivenConfig(waveport_boundary="pec"),
        )
        boundaries = config["Boundaries"]
        assert boundaries["WavePort"]
        assert "WavePortPEC" not in boundaries

    def test_absorbing_scalar_phys_group(self, tmp_path):
        groups = _groups()
        groups["boundary_surfaces"] = {"absorbing": {"phys_group": 8}}
        config = _generate(
            tmp_path,
            groups,
            _wave_ports(),
            driven=DrivenConfig(waveport_boundary="pec"),
        )
        assert config["Boundaries"]["WavePortPEC"] == {"Attributes": [4, 5, 8]}

    @pytest.mark.parametrize("source", ["absorbing", "conductivity"])
    def test_inherit_with_one_robin_source(self, tmp_path, source):
        groups = _groups(
            conductors=source == "conductivity", absorbing=source == "absorbing"
        )
        pec = _generate(
            tmp_path,
            groups,
            _wave_ports(),
            driven=DrivenConfig(waveport_boundary="pec"),
        )
        inherited = _generate(
            tmp_path,
            groups,
            _wave_ports(),
            driven=DrivenConfig(waveport_boundary="inherit"),
        )
        expected = [8, 12] if source == "absorbing" else [4, 5]
        assert pec["Boundaries"].pop("WavePortPEC") == {"Attributes": expected}
        assert inherited == pec

    def test_inherit_does_not_change_lumped_ports(self, tmp_path):
        default = _generate(tmp_path, _groups(), _lumped_ports(), driven=DrivenConfig())
        inherited = _generate(
            tmp_path,
            _groups(),
            _lumped_ports(),
            driven=DrivenConfig(waveport_boundary="inherit"),
        )
        assert inherited == default
