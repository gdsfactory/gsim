"""3D conductor boundaries carry ``"Crack": true``; native 2D ones do not.

Palace 0.19 reads interior ``Boundaries.Conductivity`` entries as thin films
unless they are cracked. gsim meshes a 3D conductor as a shell with the
dielectric inside it, so every such boundary is interior and needs the key to
stay a thick conductor. A native 2D (boundary-mode) conductor is a hole in the
mesh, its boundary exterior, so it gets no key. Palace before 0.19 rejects the
key and always cracks, so a local run on it is handed a copy without the key.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import DrivenSim, EigenmodeSim
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models import DrivenConfig
from gsim.palace.ports.config import PalacePort, PortType
from gsim.palace.runtime import strip_crack
from tests.palace.test_boundarymode_conductor_pec import (
    config_of as boundarymode_config_of,
)
from tests.palace.test_boundarymode_conductor_pec import meshed_sim
from tests.palace.test_run_local import _mesh_result, _setup_local_palace, _setup_sim
from tests.palace.test_workflow import _make_cpw_component

TAG_ENV = "PALACETOOLKIT_PALACE_CPU_TAG"


# ---------------------------------------------------------------------------
# Real generated configurations (generic PDK, coarse mesh, no solver)
# ---------------------------------------------------------------------------


def _conductors(config: dict) -> list[dict]:
    return config["Boundaries"]["Conductivity"]


@pytest.fixture(scope="module")
def driven_config(tmp_path_factory) -> dict:
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path_factory.mktemp("crack-driven") / "sim"))
    sim.set_geometry(_make_cpw_component())
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=0, margin_y=50.0, z_above=100.0, z_below=100.0)
    sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=5)
    sim.mesh(preset="coarse")
    return json.loads(Path(sim.write_config()).read_text())


@pytest.fixture(scope="module")
def eigenmode_config(tmp_path_factory) -> dict:
    sim = EigenmodeSim()
    sim.set_output_dir(str(tmp_path_factory.mktemp("crack-eigenmode") / "sim"))
    sim.set_geometry(_make_cpw_component(with_ports=False))
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=50.0, margin_y=50.0, z_above=100.0, z_below=100.0)
    sim.set_eigenmode(num_modes=5, target=50e9)
    sim.mesh(preset="coarse")
    return json.loads(Path(sim.write_config()).read_text())


@pytest.fixture(scope="module")
def boundarymode_config(tmp_path_factory) -> dict:
    return boundarymode_config_of(meshed_sim(tmp_path_factory.mktemp("crack-2d")))


class TestGeneratedConfigs:
    def test_driven_conductors_are_cracked(self, driven_config):
        entries = _conductors(driven_config)
        assert entries, "the CPW has conductor boundaries"
        assert all(entry.get("Crack") is True for entry in entries)

    def test_eigenmode_conductors_are_cracked(self, eigenmode_config):
        entries = _conductors(eigenmode_config)
        assert entries, "the CPW has conductor boundaries"
        assert all(entry.get("Crack") is True for entry in entries)

    def test_boundarymode_conductors_are_not(self, boundarymode_config):
        entries = _conductors(boundarymode_config)
        assert entries, "the 2D cross-section has Robin conductor boundaries"
        assert not any("Crack" in entry for entry in entries)
        assert "Crack" not in json.dumps(boundarymode_config)

    @pytest.mark.parametrize("name", ["driven_config", "eigenmode_config"])
    def test_only_the_crack_key_is_new(self, request, name):
        for entry in _conductors(request.getfixturevalue(name)):
            assert set(entry) == {"Attributes", "Conductivity", "Thickness", "Crack"}
            assert entry["Conductivity"] > 0
            assert entry["Thickness"] > 0


# ---------------------------------------------------------------------------
# The Conductivity entries from a hand-made mesh description
# ---------------------------------------------------------------------------


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


def _groups(*, native_2d: bool = False, wave_ports: bool = False) -> dict:
    groups: dict = {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": (
            {"metal": {"phys_group": 4}}
            if native_2d
            else {"metal_xy": {"phys_group": 4}, "metal_z": {"phys_group": 5}}
        ),
        "pec_surfaces": {},
        "port_surfaces": {"P1": {"phys_group": 6}, "P2": {"phys_group": 7}}
        if wave_ports
        else {},
        "boundary_surfaces": {"absorbing": {"phys_group": [8, 12]}},
    }
    return groups


def _generate(tmp_path, simulation_type, groups, ports=(), **kwargs) -> dict:
    path = generate_palace_config(
        groups=groups,
        ports=list(ports),
        port_info=[],
        stack=_stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type=simulation_type,
        **kwargs,
    )
    return json.loads(path.read_text())


def _wave_ports() -> list[PalacePort]:
    return [
        PalacePort(name="o1", port_type=PortType.WAVEPORT, layer="metal"),
        PalacePort(
            name="o2", port_type=PortType.WAVEPORT, layer="metal", excited=False
        ),
    ]


class TestConductivityEntries:
    @pytest.mark.parametrize("simulation_type", ["driven", "eigenmode"])
    def test_three_dimensional_entries(self, tmp_path, simulation_type):
        config = _generate(tmp_path, simulation_type, _groups())
        expected = [
            {
                "Attributes": [attribute],
                "Conductivity": 3.77e7,
                "Thickness": 3.0,
                "Crack": True,
            }
            for attribute in (4, 5)
        ]
        assert _conductors(config) == expected

    def test_native_two_dimensional_entries(self, tmp_path):
        config = _generate(tmp_path, "boundarymode", _groups(native_2d=True))
        assert _conductors(config) == [
            {"Attributes": [4], "Conductivity": 3.77e7, "Thickness": 3.0}
        ]

    def test_conductors_as_pec_leave_no_entry_behind(self, tmp_path):
        config = _generate(
            tmp_path,
            "boundarymode",
            _groups(native_2d=True),
            hints={"_conductor_boundary": "pec"},
        )
        assert "Conductivity" not in config["Boundaries"]
        assert "Crack" not in json.dumps(config)

    @pytest.mark.parametrize("policy", ["pec", "inherit"])
    def test_wave_port_case_with_waveport_pec(self, tmp_path, policy):
        config = _generate(
            tmp_path,
            "driven",
            _groups(wave_ports=True),
            _wave_ports(),
            driven_config=DrivenConfig(waveport_boundary=policy),
        )
        boundaries = config["Boundaries"]
        assert [e["Crack"] for e in boundaries["Conductivity"]] == [True, True]
        assert boundaries["WavePort"]
        if policy == "inherit":
            assert "WavePortPEC" not in boundaries
        else:
            # conductors and absorbing walls are PEC in the port's 2D solve
            assert boundaries["WavePortPEC"] == {"Attributes": [4, 5, 8, 12]}
            assert "Crack" not in boundaries["WavePortPEC"]


# ---------------------------------------------------------------------------
# What a runtime older than 0.19 is handed
# ---------------------------------------------------------------------------


def _config_with_crack() -> dict:
    return {
        "Problem": {"Type": "Driven"},
        "Boundaries": {
            "Conductivity": [
                {
                    "Attributes": [4],
                    "Conductivity": 3.77e7,
                    "Thickness": 3.0,
                    "Crack": True,
                },
                {
                    "Attributes": [5],
                    "Conductivity": 3.77e7,
                    "Thickness": 3.0,
                    "Crack": True,
                },
            ],
            "PEC": {"Attributes": [9]},
        },
    }


def _without_crack(config: dict) -> dict:
    expected = copy.deepcopy(config)
    for entry in expected["Boundaries"]["Conductivity"]:
        del entry["Crack"]
    return expected


class TestStripCrack:
    @pytest.mark.parametrize("tag", ["0.18.0", "0.18.9", "0.17.2", "v0.18.0", "0.9.1"])
    def test_older_runtimes_lose_the_key(self, tag):
        config = _config_with_crack()
        assert strip_crack(config, tag) == _without_crack(config)

    @pytest.mark.parametrize("tag", ["0.19.0", "0.19.1", "0.20.0", "1.0.0", "v0.19.0"])
    def test_newer_runtimes_keep_it(self, tag):
        config = _config_with_crack()
        assert strip_crack(config, tag) == config

    @pytest.mark.parametrize("tag", ["nightly", "0dd8d015", "abc"])
    def test_an_unreadable_tag_keeps_it_for_palace_to_report(self, tag):
        config = _config_with_crack()
        assert strip_crack(config, tag) == config

    def test_the_input_is_left_untouched(self):
        config = _config_with_crack()
        snapshot = copy.deepcopy(config)
        stripped = strip_crack(config, "0.18.0")
        assert config == snapshot
        assert stripped is not config
        assert all("Crack" in e for e in config["Boundaries"]["Conductivity"])

    def test_everything_else_is_kept(self):
        config = _config_with_crack()
        stripped = strip_crack(config, "0.18.0")
        assert stripped["Problem"] == config["Problem"]
        assert stripped["Boundaries"]["PEC"] == config["Boundaries"]["PEC"]
        for kept, original in zip(
            stripped["Boundaries"]["Conductivity"],
            config["Boundaries"]["Conductivity"],
            strict=True,
        ):
            assert kept == {k: v for k, v in original.items() if k != "Crack"}

    @pytest.mark.parametrize(
        "config",
        [{}, {"Boundaries": {}}, {"Boundaries": {"Conductivity": []}}],
    )
    def test_a_config_without_conductors_passes_through(self, config):
        assert strip_crack(config, "0.18.0") == config

    def test_the_runtime_tag_variable_decides_when_no_tag_is_given(self, monkeypatch):
        config = _config_with_crack()
        monkeypatch.setenv(TAG_ENV, "0.18.0")
        assert strip_crack(config) == _without_crack(config)
        monkeypatch.setenv(TAG_ENV, "0.19.0")
        assert strip_crack(config) == config


# ---------------------------------------------------------------------------
# run_local hands an old runtime a config without the key
# ---------------------------------------------------------------------------


class TestRunLocal:
    @pytest.mark.parametrize(
        ("tag", "keeps_key"), [("0.18.0", False), ("0.19.0", True)]
    )
    def test_what_palace_reads(self, monkeypatch, tmp_path, tag, keeps_key):
        _setup_local_palace(monkeypatch, tmp_path)
        monkeypatch.delenv("PALACE_SIF", raising=False)
        monkeypatch.delenv("PALACE_EXECUTABLE", raising=False)
        monkeypatch.setenv(TAG_ENV, tag)
        output_dir = tmp_path / "sim"
        sim = DrivenSim()
        sim._last_mesh_result = _mesh_result(1_000, 1_000)
        _setup_sim(sim, output_dir)
        (output_dir / "config.json").write_text(json.dumps(_config_with_crack()))
        seen: dict = {}

        def fake_run(cmd, **kwargs):
            seen["config"] = json.loads((Path(kwargs["cwd"]) / cmd[-1]).read_text())
            return SimpleNamespace(stdout="", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        sim.run_local(num_processes=1, verbose=False)
        expected = _config_with_crack()
        assert seen["config"] == (expected if keeps_key else _without_crack(expected))
