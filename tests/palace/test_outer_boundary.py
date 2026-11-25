"""Outer-wall condition in the generated Palace config."""

from __future__ import annotations

import json

import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import (
    BoundaryModeSim,
    DrivenSim,
    EigenmodeSim,
    ElectrostaticSim,
    MagnetostaticSim,
)
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.ports.config import PortType
from tests.palace.test_magnetostatic_config import OUTER as MAG_OUTER
from tests.palace.test_magnetostatic_config import _config as _magnetostatic_config
from tests.palace.test_pec_default_waveports import _boundaries, _sim

OUTER = [8, 12]
PEC_SURFACE = 20


def _stack() -> LayerStack:
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


def _groups() -> dict:
    return {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {"metal_xy": {"phys_group": 4}},
        "pec_surfaces": {"pec_plate": {"phys_group": PEC_SURFACE}},
        "port_surfaces": {},
        "boundary_surfaces": {"absorbing": {"phys_group": OUTER}},
    }


def _eigenmode_config(tmp_path, groups=None, **kwargs):
    kwargs.setdefault("absorbing_boundary", False)
    config_path = generate_palace_config(
        groups=groups or _groups(),
        ports=[],
        port_info=[],
        stack=_stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type="eigenmode",
        **kwargs,
    )
    return json.loads(config_path.read_text())["Boundaries"]


def test_eigenmode_default_writes_outer_walls_as_pmc(tmp_path):
    boundaries = _eigenmode_config(tmp_path)
    assert boundaries["PMC"] == {"Attributes": OUTER}
    assert "Absorbing" not in boundaries
    assert PEC_SURFACE in boundaries["PEC"]["Attributes"]
    assert not set(OUTER) & set(boundaries["PEC"]["Attributes"])


def test_pec_outer_wall_merges_with_conductor_pec(tmp_path):
    boundaries = _eigenmode_config(tmp_path, outer_boundary="pec")
    assert boundaries["PEC"] == {"Attributes": sorted([*OUTER, PEC_SURFACE])}
    assert "PMC" not in boundaries
    assert "Absorbing" not in boundaries


def test_absorbing_outer_wall(tmp_path):
    boundaries = _eigenmode_config(tmp_path, outer_boundary="absorbing")
    assert boundaries["Absorbing"] == {"Attributes": OUTER, "Order": 2}
    assert "PMC" not in boundaries


def test_legacy_absorbing_flag_maps_to_absorbing(tmp_path):
    boundaries = _eigenmode_config(tmp_path, absorbing_boundary=True)
    assert boundaries["Absorbing"] == {"Attributes": OUTER, "Order": 2}


@pytest.mark.parametrize("outer", [None, "pec", "pmc", "absorbing"])
def test_outer_attributes_are_in_one_boundary_type(tmp_path, outer):
    boundaries = _eigenmode_config(tmp_path, outer_boundary=outer)
    written = [
        attr
        for key in ("PEC", "PMC", "Absorbing")
        for attr in boundaries.get(key, {}).get("Attributes", [])
    ]
    assert len(written) == len(set(written))


def test_pec_without_outer_group_raises(tmp_path):
    groups = _groups()
    groups["boundary_surfaces"] = {}
    with pytest.raises(ValueError, match="outer"):
        _eigenmode_config(tmp_path, groups=groups, outer_boundary="pec")


def test_pmc_skips_faces_claimed_by_periodic_donor_and_receiver(tmp_path):
    groups = _groups()
    groups["boundary_surfaces"]["absorbing"] = {"phys_group": [8, 12, 13, 14]}
    groups["boundary_surfaces"]["periodic_donor"] = {"phys_group": [13]}
    groups["boundary_surfaces"]["periodic_receiver"] = {"phys_group": [14]}
    boundaries = _eigenmode_config(tmp_path, groups=groups)
    assert boundaries["PMC"] == {"Attributes": OUTER}


def test_electrostatic_outer_wall_is_not_written(tmp_path):
    groups = _groups()
    config_path = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=_stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type="electrostatic",
        absorbing_boundary=False,
    )
    boundaries = json.loads(config_path.read_text()).get("Boundaries", {})
    assert "PMC" not in boundaries
    assert boundaries["PEC"] == {"Attributes": [PEC_SURFACE]}


def test_default_resolution_per_simulation_type():
    assert EigenmodeSim()._resolved_outer_boundary() == "pmc"
    assert DrivenSim()._resolved_outer_boundary() == "absorbing"
    assert BoundaryModeSim()._resolved_outer_boundary() == "pmc"
    assert EigenmodeSim(absorbing_boundary=True)._resolved_outer_boundary() == (
        "absorbing"
    )


def test_explicit_outer_boundary_overrides_class_default():
    sim = DrivenSim()
    sim.outer_boundary = "pec"
    assert sim._resolved_outer_boundary() == "pec"


def test_legacy_flag_and_outer_boundary_agree():
    sim = EigenmodeSim(absorbing_boundary=True, outer_boundary="absorbing")
    assert sim._resolved_outer_boundary() == "absorbing"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"absorbing_boundary": True, "outer_boundary": "pec"},
        {"absorbing_boundary": False, "outer_boundary": "absorbing"},
    ],
)
def test_conflicting_absorbing_and_outer_boundary_raise(kwargs):
    with pytest.raises(ValueError, match="contradicts"):
        EigenmodeSim(**kwargs)._resolved_outer_boundary()


def test_metallic_boundaries_is_pec_and_conflicts_with_pmc():
    sim = BoundaryModeSim(metallic_boundaries=True)
    assert sim._resolved_outer_boundary() == "pec"
    sim = BoundaryModeSim(metallic_boundaries=True, outer_boundary="pmc")
    with pytest.raises(ValueError, match="metallic_boundaries"):
        sim._resolved_outer_boundary()


def test_metallic_boundaries_and_absorbing_both_raise():
    sim = BoundaryModeSim(metallic_boundaries=True, absorbing_boundary=True)
    with pytest.raises(ValueError, match="both claim the outer wall"):
        sim._resolved_outer_boundary()


def test_outer_boundary_is_rejected_for_electrostatic():
    sim = ElectrostaticSim(outer_boundary="pec")
    with pytest.raises(ValueError, match="electrostatic"):
        sim._resolved_outer_boundary()


def test_wave_ports_keep_the_pec_default_wall():
    sim = DrivenSim()
    sim.add_wave_port("o1", layer="metal")
    assert sim._resolved_outer_boundary() == "pec"
    sim.absorbing_boundary = True
    assert sim._resolved_outer_boundary() == "absorbing"


@pytest.mark.parametrize(
    ("outer", "key"), [("pec", "PEC"), ("pmc", "PMC"), ("absorbing", "Absorbing")]
)
def test_explicit_outer_boundary_wins_over_the_wave_port_default(
    monkeypatch, tmp_path, outer, key
):
    sim = _sim(monkeypatch, tmp_path, PortType.WAVEPORT, outer_boundary=outer)
    assert sim._outer_wall() == (outer == "absorbing", outer == "pec", False)
    boundaries = _boundaries(sim, tmp_path)
    assert set(OUTER) <= set(boundaries[key]["Attributes"])
    others = {"PEC", "PMC", "Absorbing"} - {key}
    for other in others:
        assert not set(OUTER) & set(boundaries.get(other, {}).get("Attributes", []))


def test_wave_ports_without_outer_group_write_no_outer_wall(monkeypatch, tmp_path):
    sim = _sim(monkeypatch, tmp_path, PortType.WAVEPORT)
    del sim._last_mesh_result.groups["boundary_surfaces"]["absorbing"]
    boundaries = _boundaries(sim, tmp_path)
    assert not {"PEC", "PMC", "Absorbing"} & set(boundaries)


def test_explicit_pec_without_outer_group_is_strict_with_wave_ports(
    monkeypatch, tmp_path
):
    sim = _sim(monkeypatch, tmp_path, PortType.WAVEPORT, outer_boundary="pec")
    del sim._last_mesh_result.groups["boundary_surfaces"]["absorbing"]
    with pytest.raises(ValueError, match="outer-boundary physical group"):
        sim.write_config(validate_mesh=False)


def test_magnetostatic_default_wall_is_pec():
    assert MagnetostaticSim()._resolved_outer_boundary() == "pec"
    assert MagnetostaticSim(metallic_boundaries=False)._resolved_outer_boundary() == (
        "pmc"
    )


def test_magnetostatic_outer_boundary_pmc_overrides_the_metallic_default():
    sim = MagnetostaticSim(outer_boundary="pmc")
    assert sim._resolved_outer_boundary() == "pmc"


def test_magnetostatic_outer_boundary_contradicting_metallic_raises():
    sim = MagnetostaticSim(metallic_boundaries=True, outer_boundary="pmc")
    with pytest.raises(ValueError, match="metallic_boundaries"):
        sim._resolved_outer_boundary()


def test_magnetostatic_refuses_an_absorbing_outer_wall():
    sim = MagnetostaticSim(outer_boundary="absorbing")
    with pytest.raises(ValueError, match="magnetostatic"):
        sim._resolved_outer_boundary()


def test_magnetostatic_config_takes_pec_and_leaves_pmc_natural(tmp_path):
    pec = _magnetostatic_config(tmp_path, hints={}, outer_boundary="pec")
    assert set(MAG_OUTER) <= set(pec["Boundaries"]["PEC"]["Attributes"])
    pmc = _magnetostatic_config(tmp_path, outer_boundary="pmc", hints={})
    assert "PMC" not in pmc["Boundaries"]
    assert not set(MAG_OUTER) & set(pmc["Boundaries"]["PEC"]["Attributes"])


def test_magnetostatic_config_refuses_an_absorbing_outer_boundary(tmp_path):
    with pytest.raises(ValueError, match=r"Absorbing.*magnetostatic"):
        _magnetostatic_config(tmp_path, hints={}, outer_boundary="absorbing")
