"""Eigenmode lumped-port config generation.

A Josephson junction is modelled as a lumped port carrying only reactive
elements (L and C). The default 50 Ohm port impedance must not be emitted in
parallel with those reactive elements, otherwise it would load the junction.
"""

from __future__ import annotations

import json

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models import EigenmodeConfig
from gsim.palace.ports import PalacePort


def _stack() -> LayerStack:
    stack = LayerStack(pdk_name="test")
    stack.layers["M1"] = Layer(
        name="M1",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=0.0,
        thickness=0.0,
        material="aluminum",
        layer_type="conductor",
    )
    stack.materials = {"aluminum": {"conductivity": 3.77e7, "type": "conductor"}}
    return stack


def _groups() -> dict:
    return {
        "volumes": {},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {
            "P1": {"phys_group": 10, "type": "lumped"},
            "P2": {"phys_group": 11, "type": "lumped"},
        },
        "boundary_surfaces": {},
    }


def _write(tmp_path, port: PalacePort) -> dict:
    config_path = generate_palace_config(
        groups=_groups(),
        ports=[port],
        port_info=[],
        stack=_stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=10e9,
        simulation_type="eigenmode",
        eigenmode_config=EigenmodeConfig(num_modes=2, target=4e9),
        absorbing_boundary=False,
    )
    return json.loads(config_path.read_text())


def test_junction_lc_port_has_no_parallel_resistance(tmp_path):
    """A reactive lumped port is emitted with L/C and no R."""
    port = PalacePort(
        name="junction",
        center=(0.0, 0.0),
        width=1.0,
        length=1.0,
        orientation=90.0,
        layer="M1",
        impedance=50.0,
        inductance=14.86e-9,
        capacitance=5.5e-15,
    )
    config = _write(tmp_path, port)
    junction = config["Boundaries"]["LumpedPort"][0]
    assert junction["L"] == 14.86e-9
    assert junction["C"] == 5.5e-15
    assert "R" not in junction


def test_resistive_lumped_port_keeps_impedance(tmp_path):
    """A purely resistive lumped port keeps its impedance as R."""
    port = PalacePort(
        name="feed",
        center=(0.0, 0.0),
        width=1.0,
        length=1.0,
        orientation=90.0,
        layer="M1",
        impedance=50.0,
    )
    config = _write(tmp_path, port)
    feed = config["Boundaries"]["LumpedPort"][0]
    assert feed["R"] == 50.0
    assert "L" not in feed
    assert "C" not in feed
