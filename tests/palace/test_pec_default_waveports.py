"""Wave-port driven runs default the outer wall to PEC (config level).

Palace solves the wave-port eigenproblem with the port's outer edges as PEC,
so an absorbing outer wall makes the 3D domain and the port mode disagree.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import DrivenSim
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.ports.config import PalacePort, PortType

OUTER = [8, 12]


def _stack() -> LayerStack:
    """Return a one-metal stack."""
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


def _sim(monkeypatch, tmp_path, port_type: PortType, **options) -> DrivenSim:
    """Return a DrivenSim holding a synthetic mesh result and two ports."""
    sim = DrivenSim(**options)
    if port_type == PortType.WAVEPORT:
        sim.add_wave_port("o1", layer="metal")
        sim.add_wave_port("o2", layer="metal", excited=False)
    monkeypatch.setattr(sim, "_resolve_stack", _stack)
    groups = {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {"metal_xy": {"phys_group": 4}},
        "pec_surfaces": {},
        "port_surfaces": {"P1": {"phys_group": 6}, "P2": {"phys_group": 7}},
        "boundary_surfaces": {"absorbing": {"phys_group": OUTER}},
    }
    sim._last_mesh_result = MeshResult(
        mesh_path=tmp_path / "palace.msh", groups=groups, output_dir=tmp_path
    )
    sim._last_ports = [
        PalacePort(name="o1", port_type=port_type, layer="metal"),
        PalacePort(name="o2", port_type=port_type, layer="metal", excited=False),
    ]
    return sim


def _boundaries(sim: DrivenSim, tmp_path) -> dict:
    """Write the config and return its Boundaries block."""
    sim.write_config(validate_mesh=False)
    return json.loads(Path(tmp_path, "config.json").read_text())["Boundaries"]


def test_wave_ports_default_to_pec_wall(monkeypatch, tmp_path):
    """Wave ports with nothing set: outer surfaces in PEC, no Absorbing."""
    sim = _sim(monkeypatch, tmp_path, PortType.WAVEPORT)
    boundaries = _boundaries(sim, tmp_path)
    assert "Absorbing" not in boundaries
    assert set(OUTER) <= set(boundaries["PEC"]["Attributes"])


def test_lumped_ports_keep_absorbing_wall(monkeypatch, tmp_path):
    """Lumped ports only: the absorbing default is unchanged."""
    boundaries = _boundaries(_sim(monkeypatch, tmp_path, PortType.LUMPED), tmp_path)
    assert boundaries["Absorbing"]["Attributes"] == OUTER
    assert "PEC" not in boundaries


def test_explicit_absorbing_wins_with_wave_ports(monkeypatch, tmp_path):
    """absorbing_boundary=True keeps the absorbing wall with wave ports."""
    sim = _sim(monkeypatch, tmp_path, PortType.WAVEPORT, absorbing_boundary=True)
    boundaries = _boundaries(sim, tmp_path)
    assert boundaries["Absorbing"]["Attributes"] == OUTER
    assert not set(OUTER) & set(boundaries.get("PEC", {}).get("Attributes", []))


def test_metallic_and_absorbing_are_still_refused(monkeypatch, tmp_path):
    """Both set explicitly: the outer wall is claimed twice."""
    sim = _sim(
        monkeypatch,
        tmp_path,
        PortType.WAVEPORT,
        absorbing_boundary=True,
        metallic_boundaries=True,
    )
    with pytest.raises(ValueError, match="both claim the outer wall"):
        sim.write_config(validate_mesh=False)


def test_pec_default_is_noted_once(monkeypatch, tmp_path, caplog):
    """The PEC default is announced on the first config write only."""
    sim = _sim(monkeypatch, tmp_path, PortType.WAVEPORT)
    with caplog.at_level(logging.INFO, logger="gsim.palace.base"):
        sim.write_config(validate_mesh=False)
        sim.write_config(validate_mesh=False)
    notes = [r for r in caplog.records if "outer wall is PEC" in r.getMessage()]
    assert len(notes) == 1
