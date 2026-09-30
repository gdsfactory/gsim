"""Public mesh handles on the Palace sim classes.

The Palace sim classes answer "where is your mesh, and what groups does it
have?" with the same public accessors the charge-transport sim class
exposes, so no supported path needs private mesh state.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gsim.palace import BoundaryModeSim, DrivenSim

SIM_CLASSES = [BoundaryModeSim, DrivenSim]


@pytest.mark.parametrize("sim_class", SIM_CLASSES)
class TestBeforeMeshing:
    def test_mesh_path_raises_actionable_error(self, sim_class):
        sim = sim_class()
        with pytest.raises(ValueError, match=r"mesh\(\)"):
            _ = sim.mesh_path

    def test_mesh_groups_raises_actionable_error(self, sim_class):
        sim = sim_class()
        with pytest.raises(ValueError, match=r"mesh\(\)"):
            _ = sim.mesh_groups

    def test_has_mesh_is_false(self, sim_class):
        assert sim_class().has_mesh is False


@pytest.mark.parametrize("sim_class", SIM_CLASSES)
class TestAfterMeshing:
    def test_mesh_path_is_the_generated_mesh(self, sim_class, tmp_path):
        sim = sim_class()
        sim._last_mesh_result = SimpleNamespace(
            mesh_path=tmp_path / "palace.msh", groups={}
        )
        assert sim.mesh_path == tmp_path / "palace.msh"
        assert isinstance(sim.mesh_path, Path)

    def test_mesh_groups_are_the_generated_groups(self, sim_class, tmp_path):
        groups = {"volumes": {"si": 1}, "contact_lines": {"anode": 2}}
        sim = sim_class()
        sim._last_mesh_result = SimpleNamespace(
            mesh_path=tmp_path / "palace.msh", groups=groups
        )
        assert sim.mesh_groups == groups

    def test_has_mesh_is_true(self, sim_class, tmp_path):
        sim = sim_class()
        sim._last_mesh_result = SimpleNamespace(
            mesh_path=tmp_path / "palace.msh", groups={}
        )
        assert sim.has_mesh is True

    def test_mesh_groups_default_to_empty_mapping(self, sim_class, tmp_path):
        sim = sim_class()
        sim._last_mesh_result = SimpleNamespace(
            mesh_path=tmp_path / "palace.msh", groups=None
        )
        assert sim.mesh_groups == {}


def test_charge_transport_sim_names_the_handles_identically():
    """All four sim classes answer the mesh question the same way."""
    from gsim.tcad import ChargeTransportSim

    for name in ("mesh_path", "mesh_groups", "has_mesh"):
        assert isinstance(getattr(ChargeTransportSim, name), property)
        for sim_class in SIM_CLASSES:
            assert isinstance(getattr(sim_class, name), property)


def test_charge_transport_sim_raises_before_meshing():
    from gsim.tcad import ChargeTransportSim

    sim = ChargeTransportSim()
    assert sim.has_mesh is False
    with pytest.raises(ValueError, match=r"mesh\(\)"):
        _ = sim.mesh_path
    with pytest.raises(ValueError, match=r"mesh\(\)"):
        _ = sim.mesh_groups
