"""Tests for exposing Palace's adaptive mesh refinement controls.

gsim wrote a fixed ``Model.Refinement`` block with ``MaxIts = 0``, so adaptive
mesh refinement could not be switched on. ``RefinementConfig`` maps onto that
block. The key names and bounds checked here are the ones in Palace's own JSON
schema (``scripts/schema/config-schema.json`` in awslabs/palace), which rejects
any key it does not know (``"additionalProperties": false``).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from gsim.common.stack.extractor import LayerStack
from gsim.palace import DrivenSim
from gsim.palace.mesh.config_generator import generate_palace_config, write_config
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.models import RefinementConfig

# The keys Palace's schema accepts in config["Model"]["Refinement"].
PALACE_REFINEMENT_KEYS = {
    "Tol",
    "MaxIts",
    "MaxSize",
    "UpdateFraction",
    "Nonconformal",
    "MaxNCLevels",
    "MaximumImbalance",
    "SaveAdaptIterations",
    "SaveAdaptMesh",
    "UniformLevels",
    "SerialUniformLevels",
    "Boxes",
    "Spheres",
}

# What gsim has always written, and Palace's own defaults for these three keys.
DEFAULT_BLOCK = {"UniformLevels": 0, "Tol": 0.01, "MaxIts": 0}


def _groups() -> dict:
    return {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {"absorbing": {"phys_group": [2]}},
    }


def _refinement_block(tmp_path, refinement_config=None) -> dict:
    config_path = generate_palace_config(
        groups=_groups(),
        ports=[],
        port_info=[],
        stack=LayerStack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type="driven",
        absorbing_boundary=True,
        refinement_config=refinement_config,
    )
    return json.loads(config_path.read_text())["Model"]["Refinement"]


def test_default_block_is_unchanged(tmp_path) -> None:
    assert _refinement_block(tmp_path) == DEFAULT_BLOCK
    assert _refinement_block(tmp_path, RefinementConfig()) == DEFAULT_BLOCK


def test_controls_reach_the_palace_config(tmp_path) -> None:
    block = _refinement_block(
        tmp_path,
        RefinementConfig(
            tol=1e-3,
            max_its=6,
            max_dofs=2_000_000,
            update_fraction=0.5,
            nonconformal=False,
            max_nc_levels=2,
            uniform_levels=1,
            save_adapt_iterations=True,
            save_adapt_mesh=True,
        ),
    )
    assert block == {
        "Tol": 1e-3,
        "MaxIts": 6,
        "MaxSize": 2_000_000,
        "UpdateFraction": 0.5,
        "Nonconformal": False,
        "MaxNCLevels": 2,
        "UniformLevels": 1,
        "SaveAdaptIterations": True,
        "SaveAdaptMesh": True,
    }
    assert set(block) <= PALACE_REFINEMENT_KEYS


def test_unset_controls_are_left_to_palace(tmp_path) -> None:
    block = _refinement_block(tmp_path, RefinementConfig(max_its=3))
    assert set(block) == set(DEFAULT_BLOCK)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tol": 0.0},
        {"max_its": -1},
        {"max_dofs": -1},
        {"update_fraction": 0.0},
        {"update_fraction": 1.0},
        {"max_nc_levels": -1},
        {"uniform_levels": -1},
    ],
)
def test_values_outside_palace_bounds_are_rejected(kwargs) -> None:
    with pytest.raises(ValidationError):
        RefinementConfig(**kwargs)


def test_set_refinement_replaces_the_config() -> None:
    sim = DrivenSim()
    assert sim.refinement == RefinementConfig()
    sim.set_refinement(max_its=5, tol=1e-3)
    assert sim.refinement == RefinementConfig(max_its=5, tol=1e-3)


def test_write_config_forwards_the_refinement(tmp_path) -> None:
    mesh_result = MeshResult(
        mesh_path=tmp_path / "palace.msh", groups=_groups(), output_dir=tmp_path
    )
    config_path = write_config(
        mesh_result=mesh_result,
        stack=LayerStack(),
        ports=[],
        refinement_config=RefinementConfig(max_its=4),
    )
    block = json.loads(config_path.read_text())["Model"]["Refinement"]
    assert block["MaxIts"] == 4


def test_sim_write_config_uses_set_refinement(monkeypatch, tmp_path) -> None:
    """The setting has to reach the config the sim writes, not just be stored."""
    captured = {}

    def fake_write_config(**kwargs):
        captured.update(kwargs)
        return tmp_path / "config.json"

    monkeypatch.setattr("gsim.palace.mesh.generator.write_config", fake_write_config)
    monkeypatch.setattr(DrivenSim, "_resolve_stack", lambda self: LayerStack())

    sim = DrivenSim()
    sim.set_refinement(max_its=4)
    sim._last_mesh_result = MeshResult(
        mesh_path=tmp_path / "palace.msh", groups=_groups(), output_dir=tmp_path
    )
    sim.write_config(validate_mesh=False)
    assert captured["refinement_config"] == RefinementConfig(max_its=4)
