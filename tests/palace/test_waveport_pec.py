"""Tests for WavePortPEC emission with numeric wave ports.

When a model has numeric wave ports, the generated config must list every
boundary attribute that would otherwise contribute a Robin term to the 2D
port eigenproblem (absorbing walls, finite-conductivity conductors, and
impedance boundaries) under ``Boundaries.WavePortPEC``.
"""

from __future__ import annotations

import json

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace.mesh.config_generator import generate_palace_config
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


def _generate(tmp_path, groups, ports, *, hints=None, absorbing_boundary=True):
    config_path = generate_palace_config(
        groups=groups,
        ports=ports,
        port_info=[],
        stack=_stack_with_metal(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type="driven",
        absorbing_boundary=absorbing_boundary,
        hints=hints,
    )
    return json.loads(config_path.read_text())


class TestWavePortPEC:
    def test_union_of_absorbing_and_conductivity(self, tmp_path):
        config = _generate(tmp_path, _groups(), _wave_ports())
        boundaries = config["Boundaries"]
        assert boundaries["WavePortPEC"] == {"Attributes": [4, 5, 8, 12]}

    def test_not_emitted_without_wave_ports(self, tmp_path):
        config = _generate(tmp_path, _groups(wave_ports=False), _lumped_ports())
        assert "WavePortPEC" not in config["Boundaries"]

    def test_not_emitted_for_lumped_ports(self, tmp_path):
        groups = _groups()
        config = _generate(tmp_path, groups, _lumped_ports())
        assert config["Boundaries"]["LumpedPort"]
        assert not config["Boundaries"]["WavePort"]
        assert "WavePortPEC" not in config["Boundaries"]

    def test_includes_impedance_attributes(self, tmp_path):
        hints = {
            "_impedance_boundaries": [
                ImpedanceBoundaryConfig(attributes=[20], resistance=1.0)
            ]
        }
        config = _generate(tmp_path, _groups(), _wave_ports(), hints=hints)
        assert config["Boundaries"]["WavePortPEC"] == {"Attributes": [4, 5, 8, 12, 20]}

    def test_not_emitted_when_no_robin_boundaries(self, tmp_path):
        groups = _groups(conductors=False, absorbing=False)
        config = _generate(tmp_path, groups, _wave_ports(), absorbing_boundary=False)
        boundaries = config["Boundaries"]
        assert boundaries["WavePort"]
        assert "WavePortPEC" not in boundaries

    def test_absorbing_scalar_phys_group(self, tmp_path):
        groups = _groups()
        groups["boundary_surfaces"] = {"absorbing": {"phys_group": 8}}
        config = _generate(tmp_path, groups, _wave_ports())
        assert config["Boundaries"]["WavePortPEC"] == {"Attributes": [4, 5, 8]}
