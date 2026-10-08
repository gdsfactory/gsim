"""Tests for symmetry-plane boundaries in the generated Palace config."""

from __future__ import annotations

import json

import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models.ports import ImpedanceBoundaryConfig
from gsim.palace.ports.config import PalacePort, PortType

PLANE_PG = 30


def _stack_with_metal() -> LayerStack:
    """Stack with one conductor layer."""
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


def _groups(kind: str = "pmc", *, pec_surfaces: dict | None = None) -> dict:
    """Synthetic mesh groups with a symmetry plane."""
    return {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {
            "metal_xy": {"phys_group": 4},
            "metal_z": {"phys_group": 5},
        },
        "pec_surfaces": pec_surfaces or {},
        "port_surfaces": {"P1": {"phys_group": 6}, "P2": {"phys_group": 7}},
        "boundary_surfaces": {
            "absorbing": {"phys_group": [8, 12]},
            "symmetry": {
                "phys_group": [PLANE_PG],
                "tags": [1],
                "axis": "y",
                "position": 0.0,
                "kind": kind,
                "keep": "positive",
            },
        },
    }


def _wave_ports() -> list[PalacePort]:
    """Two wave ports."""
    return [
        PalacePort(name="o1", port_type=PortType.WAVEPORT, layer="metal"),
        PalacePort(
            name="o2", port_type=PortType.WAVEPORT, layer="metal", excited=False
        ),
    ]


def _lumped_ports() -> list[PalacePort]:
    """Two lumped ports."""
    return [
        PalacePort(name="o1", port_type=PortType.LUMPED, layer="metal"),
        PalacePort(name="o2", port_type=PortType.LUMPED, layer="metal", excited=False),
    ]


def _generate(
    tmp_path,
    groups,
    ports,
    *,
    simulation_type="driven",
    hints=None,
    absorbing_boundary=True,
    port_info=None,
):
    """Write a config and return it as a dict."""
    config_path = generate_palace_config(
        groups=groups,
        ports=ports,
        port_info=port_info or [],
        stack=_stack_with_metal(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type=simulation_type,
        absorbing_boundary=absorbing_boundary,
        hints=hints,
    )
    return json.loads(config_path.read_text())


def test_pmc_plane_writes_pmc(tmp_path):
    """A PMC plane becomes Boundaries.PMC and is not absorbing."""
    boundaries = _generate(tmp_path, _groups("pmc"), _wave_ports())["Boundaries"]
    assert boundaries["PMC"] == {"Attributes": [PLANE_PG]}
    assert PLANE_PG not in boundaries["Absorbing"]["Attributes"]
    assert PLANE_PG not in boundaries.get("PEC", {}).get("Attributes", [])


def test_pmc_plane_stays_out_of_waveport_pec(tmp_path):
    """The port solve keeps a PMC plane natural; Robin attributes stay listed."""
    boundaries = _generate(tmp_path, _groups("pmc"), _wave_ports())["Boundaries"]
    assert boundaries["WavePortPEC"] == {"Attributes": [4, 5, 8, 12]}


def test_pmc_plane_never_enters_waveport_pec_via_other_lists(tmp_path):
    """Even if another boundary list names the plane, WavePortPEC drops it.

    Guards against a change of the 'Robin boundaries become PEC' rule (#310)
    pulling the plane into the port eigenproblem as PEC.
    """
    hints = {
        "_impedance_boundaries": [
            ImpedanceBoundaryConfig(attributes=[PLANE_PG, 20], resistance=1.0)
        ]
    }
    boundaries = _generate(tmp_path, _groups("pmc"), _wave_ports(), hints=hints)[
        "Boundaries"
    ]
    assert boundaries["WavePortPEC"] == {"Attributes": [4, 5, 8, 12, 20]}


def test_pec_plane_merges_with_planar_pec_sorted_unique(tmp_path):
    """A PEC plane joins the PEC attributes, sorted without duplicates."""
    groups = _groups(
        "pec",
        pec_surfaces={"a": {"phys_group": 40}, "b": {"phys_group": 2}},
    )
    groups["pec_surfaces"]["c"] = {"phys_group": PLANE_PG}
    boundaries = _generate(tmp_path, groups, _wave_ports())["Boundaries"]
    assert boundaries["PEC"] == {"Attributes": [2, PLANE_PG, 40]}
    assert "PMC" not in boundaries
    assert PLANE_PG not in boundaries["Absorbing"]["Attributes"]


def test_pec_plane_creates_pec_entry(tmp_path):
    """Without other PEC surfaces the PEC entry holds just the plane."""
    boundaries = _generate(tmp_path, _groups("pec"), _lumped_ports())["Boundaries"]
    assert boundaries["PEC"] == {"Attributes": [PLANE_PG]}


def test_pec_plane_not_in_waveport_pec(tmp_path):
    """A PEC plane is not listed under WavePortPEC (it is already PEC).

    Documents the decision; see SYMMETRY.md for the unverified Palace behaviour.
    """
    boundaries = _generate(tmp_path, _groups("pec"), _wave_ports())["Boundaries"]
    assert PLANE_PG not in boundaries["WavePortPEC"]["Attributes"]


def test_pmc_written_without_absorbing_boundary(tmp_path):
    """The plane is emitted even when the absorbing boundary is off."""
    boundaries = _generate(
        tmp_path, _groups("pmc"), _lumped_ports(), absorbing_boundary=False
    )["Boundaries"]
    assert boundaries["PMC"] == {"Attributes": [PLANE_PG]}
    assert "Absorbing" not in boundaries


def test_plane_in_absorbing_group_raises(tmp_path):
    """A plane attribute that is also absorbing is a bug and raises."""
    groups = _groups("pmc")
    groups["boundary_surfaces"]["absorbing"]["phys_group"] = [8, PLANE_PG]
    with pytest.raises(ValueError, match="absorbing"):
        _generate(tmp_path, groups, _wave_ports())


def test_no_plane_has_no_pmc(tmp_path):
    """Without a plane there is no PMC key and no symmetry info."""
    groups = _groups()
    del groups["boundary_surfaces"]["symmetry"]
    config = _generate(tmp_path, groups, _wave_ports())
    assert "PMC" not in config["Boundaries"]
    info = json.loads((tmp_path / "port_information.json").read_text())
    assert "symmetry" not in info


@pytest.mark.parametrize("simulation_type", ["electrostatic", "boundarymode"])
def test_unsupported_solvers_raise(tmp_path, simulation_type):
    """Electrostatic and boundarymode configs reject a plane."""
    with pytest.raises(ValueError, match="Symmetry planes"):
        _generate(tmp_path, _groups(), [], simulation_type=simulation_type)


@pytest.mark.parametrize(("kind", "mode"), [("pmc", "even"), ("pec", "odd")])
def test_port_information_records_symmetry(tmp_path, kind, mode):
    """port_information.json records the plane and the cut flag per port."""
    port_info = [
        {"portnumber": 1, "type": "waveport", "cut_by_symmetry_plane": True},
        {"portnumber": 2, "type": "waveport"},
    ]
    _generate(tmp_path, _groups(kind), _wave_ports(), port_info=port_info)
    info = json.loads((tmp_path / "port_information.json").read_text())
    assert info["symmetry"] == {
        "axis": "y",
        "position": 0.0,
        "kind": kind,
        "keep": "positive",
        "mode": mode,
    }
    assert [p["cut_by_symmetry_plane"] for p in info["ports"]] == [True, False]
