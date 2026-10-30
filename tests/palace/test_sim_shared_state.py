"""Every Palace simulation starts from the same state and guards it alike.

A fresh simulation of any problem type has no mesh, no output directory and
no cloud job, and each call that needs one of them says which call is
missing instead of failing on an undeclared attribute.
"""

from __future__ import annotations

import pytest

from gsim.palace import (
    BoundaryModeSim,
    DrivenSim,
    EigenmodeSim,
    ElectrostaticSim,
    MagnetostaticSim,
)
from gsim.palace.models import MeshConfig

SIM_CLASSES = [
    DrivenSim,
    EigenmodeSim,
    ElectrostaticSim,
    MagnetostaticSim,
    BoundaryModeSim,
]


@pytest.fixture(params=SIM_CLASSES, ids=lambda cls: cls.__name__)
def sim(request):
    """A fresh simulation of each Palace problem type."""
    return request.param()


def test_fresh_sim_has_no_mesh(sim):
    assert sim.has_mesh is False
    assert sim.output_dir is None
    with pytest.raises(ValueError, match=r"Call mesh\(\) first"):
        _ = sim.mesh_path
    with pytest.raises(ValueError, match=r"Call mesh\(\) first"):
        _ = sim.mesh_groups


def test_write_config_needs_a_mesh(sim):
    with pytest.raises(ValueError, match=r"No mesh result\. Call mesh\(\) first\."):
        sim.write_config()


def test_validate_mesh_reports_missing_mesh(sim):
    with pytest.raises(RuntimeError, match=r"No mesh generated\. Call mesh\(\)"):
        sim.validate_mesh()


def test_print_mesh_stats_reports_missing_mesh(sim, capsys):
    sim.print_mesh_stats()
    assert "No mesh result. Call mesh() first." in capsys.readouterr().out


def test_upload_needs_an_output_dir_then_a_mesh(sim, tmp_path):
    with pytest.raises(ValueError, match=r"Call set_output_dir\(\) first"):
        sim.upload(verbose=False)
    sim.set_output_dir(tmp_path)
    with pytest.raises(ValueError, match=r"No mesh result\. Call mesh\(\) first\."):
        sim.upload(verbose=False)


def test_cloud_calls_need_a_job(sim):
    with pytest.raises(ValueError, match=r"Call upload\(\) first"):
        sim.start(verbose=False)
    with pytest.raises(ValueError, match="No job submitted yet"):
        sim.get_status()
    with pytest.raises(ValueError, match="No job submitted yet"):
        sim.wait_for_results()


def test_mesh_config_defaults_to_the_default_preset(sim):
    assert sim.mesh_config == MeshConfig.default()


def test_private_state_is_per_instance(sim):
    other = type(sim)()
    sim._hints["key"] = 1
    sim._pec_blocks.append(object())
    assert other._hints == {}
    assert other._pec_blocks == []
    assert other._last_ports == []
