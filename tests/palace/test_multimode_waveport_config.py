"""Multimode wave ports: several Palace ``WavePort`` entries on one face.

Palace takes one entry per (face, mode). Entries may share boundary
attributes, but only one may be ``"Active"`` (it terminates the face) and it
must have the lowest ``Index`` of the face. These tests pin what gsim writes:
the active entry keeps the gsim port's own index, the other modes follow after
all ports with ``"Active": false``, and ``port_modes.json`` maps each Palace
index back to (port, mode, active).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gdsfactory as gf
import pytest

from gsim.common.stack import LayerStack
from gsim.common.stack.extractor import Layer
from gsim.palace import DrivenSim, EigenmodeSim
from gsim.palace.boundarymode import BoundaryModeSim
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models.ports import WavePortConfig
from gsim.palace.ports import configure_wave_port, extract_ports
from gsim.palace.ports.config import PalacePort, PortType
from gsim.palace.ports.modes import (
    PORT_MODES_FILENAME,
    PortModeEntry,
    check_port_mode_entries,
    faces_by_index,
    read_port_modes,
    same_face_pairs,
    write_port_modes,
)

PILOT = [[0.0, 0.0, 1.0], [0.0, 6.0, 1.0]]


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


def _groups() -> dict:
    return {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {
            "metal_xy": {"phys_group": 4},
            "metal_z": {"phys_group": 5},
        },
        "pec_surfaces": {},
        "port_surfaces": {"P1": {"phys_group": 6}, "P2": {"phys_group": 7}},
        "boundary_surfaces": {"absorbing": {"phys_group": [8, 12]}},
    }


def _generate(tmp_path: Path, ports: list[PalacePort]) -> dict:
    config_path = generate_palace_config(
        groups=_groups(),
        ports=ports,
        port_info=[],
        stack=_stack_with_metal(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type="driven",
        absorbing_boundary=True,
        hints=None,
    )
    return json.loads(config_path.read_text())


def _wave(name: str, x: float, angle: float, **kwargs) -> PalacePort:
    return PalacePort(
        name=name,
        port_type=PortType.WAVEPORT,
        layer="metal",
        center=(x, 0.0),
        orientation=angle,
        **kwargs,
    )


def _two_ports(**kwargs) -> list[PalacePort]:
    return [
        _wave("o1", -100.0, 180.0, **kwargs),
        _wave("o2", 100.0, 0.0, **kwargs),
    ]


def _entries(config: dict) -> list[dict]:
    return config["Boundaries"]["WavePort"]


def _component(*ports: tuple[str, float, float]) -> gf.Component:
    """A component with a port ``(name, x, orientation)`` each, on the generic M1."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    for name, x, angle in ports:
        c.add_port(
            name=name,
            center=(x, 0),
            width=10,
            orientation=angle,
            port_type="electrical",
            layer=gf.gpdk.LAYER.M1,
        )
    return c


class TestWavePortConfigModes:
    def test_defaults_are_an_ordinary_wave_port(self) -> None:
        wp = WavePortConfig(name="o1", layer="metal")
        assert wp.modes == ()
        assert wp.active_mode is None
        assert wp.excite_modes == "active"
        assert wp.pilot_paths is None

    def test_active_mode_stands_in_for_mode(self) -> None:
        wp = WavePortConfig(name="o1", layer="metal", modes=(1, 2, 3), active_mode=2)
        assert wp.mode == 2

    def test_active_mode_must_be_listed(self) -> None:
        with pytest.raises(ValueError, match="not in modes"):
            WavePortConfig(name="o1", layer="metal", modes=(1, 2), active_mode=3)

    def test_modes_need_an_active_mode(self) -> None:
        with pytest.raises(ValueError, match="active_mode is required"):
            WavePortConfig(name="o1", layer="metal", modes=(1, 2))

    def test_mode_and_active_mode_must_agree(self) -> None:
        with pytest.raises(ValueError, match="must agree"):
            WavePortConfig(
                name="o1", layer="metal", mode=1, modes=(1, 2), active_mode=2
            )

    def test_a_mode_listed_twice_is_refused(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            WavePortConfig(name="o1", layer="metal", modes=(1, 1), active_mode=1)

    def test_modes_are_one_based(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            WavePortConfig(name="o1", layer="metal", modes=(0, 1), active_mode=1)

    def test_excite_modes_is_active_or_all(self) -> None:
        with pytest.raises(ValueError, match="excite_modes"):
            WavePortConfig(name="o1", layer="metal", excite_modes="some")

    @pytest.mark.parametrize(
        ("paths", "message"),
        [
            ({3: PILOT}, "not one of"),
            ({1: [[0.0, 0.0, 0.0]]}, "2\\+ points"),
            ({1: [[0.0, 0.0], [1.0, 1.0]]}, "x, y, z"),
            ({1: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}, "repeats the point"),
            ({1: [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]}, "finite"),
        ],
    )
    def test_pilot_paths_are_checked(self, paths, message) -> None:
        with pytest.raises(ValueError, match=message):
            WavePortConfig(
                name="o1", layer="metal", modes=(1, 2), active_mode=1, pilot_paths=paths
            )

    def test_changing_the_active_mode_after_the_fact_is_checked(self) -> None:
        wp = WavePortConfig(name="o1", layer="metal", modes=(1, 2), active_mode=1)
        with pytest.raises(ValueError, match="must agree"):
            wp.active_mode = 2


class TestMultimodeEntries:
    def test_an_ordinary_wave_port_is_unchanged(self, tmp_path) -> None:
        config = _generate(tmp_path, _two_ports())
        assert [e["Index"] for e in _entries(config)] == [1, 2]
        for entry in _entries(config):
            assert "Active" not in entry
            assert "VoltagePath" not in entry
        assert not (tmp_path / PORT_MODES_FILENAME).exists()

    def test_the_active_entry_keeps_the_port_index_and_the_rest_follow(
        self, tmp_path
    ) -> None:
        config = _generate(tmp_path, _two_ports(mode=1, modes=(1, 2, 3)))
        entries = _entries(config)
        assert [(e["Index"], e["Mode"]) for e in entries] == [
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 2),
            (6, 3),
        ]
        by_index = {e["Index"]: e for e in entries}
        for index in (1, 2):
            assert "Active" not in by_index[index]
        for index in (3, 4, 5, 6):
            assert by_index[index]["Active"] is False
        # One surface per face: every mode of a face shares its attribute.
        assert [by_index[i]["Attributes"] for i in (1, 3, 4)] == [[6]] * 3
        assert [by_index[i]["Attributes"] for i in (2, 5, 6)] == [[7]] * 3
        assert all(e["Offset"] == 0.0 for e in entries)

    def test_the_active_mode_need_not_be_the_first(self, tmp_path) -> None:
        config = _generate(tmp_path, _two_ports(mode=2, modes=(1, 2, 3)))
        by_index = {e["Index"]: e for e in _entries(config)}
        assert by_index[1]["Mode"] == 2
        assert "Active" not in by_index[1]
        assert (by_index[3]["Mode"], by_index[4]["Mode"]) == (1, 3)
        rows = read_port_modes(tmp_path)
        assert [(r.index, r.mode, r.active) for r in rows if r.port == "o1"] == [
            (1, 2, True),
            (3, 1, False),
            (4, 3, False),
        ]

    def test_only_the_active_entries_are_excited_by_default(self, tmp_path) -> None:
        ports = _two_ports(modes=(1, 2))
        ports[1].excited = False
        config = _generate(tmp_path, ports)
        excitation = {e["Index"]: e["Excitation"] for e in _entries(config)}
        assert excitation == {1: 1, 2: False, 3: False, 4: False}

    def test_excite_all_excites_every_entry_with_its_own_index(self, tmp_path) -> None:
        config = _generate(tmp_path, _two_ports(modes=(1, 2, 3), excite_modes="all"))
        assert {e["Index"]: e["Excitation"] for e in _entries(config)} == {
            i: i for i in range(1, 7)
        }

    def test_an_unexcited_port_stays_unexcited_with_excite_all(self, tmp_path) -> None:
        ports = _two_ports(modes=(1, 2), excite_modes="all")
        for port in ports:
            port.excited = False
        config = _generate(tmp_path, ports)
        assert all(e["Excitation"] is False for e in _entries(config))

    def test_voltage_paths_and_eigensolver_controls_reach_the_entries(
        self, tmp_path
    ) -> None:
        other = [[0.0, 0.0, 2.0], [0.0, 9.0, 2.0]]
        ports = _two_ports(
            modes=(1, 2),
            pilot_paths={1: PILOT, 2: other},
            eigensolver_type="SLEPc",
            eigensolver_max_size=40,
        )
        by_index = {e["Index"]: e for e in _entries(_generate(tmp_path, ports))}
        assert by_index[1]["VoltagePath"] == PILOT
        assert by_index[3]["VoltagePath"] == other
        for index in (1, 2, 3, 4):
            assert by_index[index]["SolverType"] == "SLEPc"
            assert by_index[index]["MaxSize"] == 40

    def test_a_mode_without_a_pilot_path_has_no_voltage_path(self, tmp_path) -> None:
        ports = _two_ports(modes=(1, 2), pilot_paths={2: PILOT})
        by_index = {e["Index"]: e for e in _entries(_generate(tmp_path, ports))}
        assert "VoltagePath" not in by_index[1]
        assert by_index[3]["VoltagePath"] == PILOT

    def test_the_sidecar_maps_every_palace_index(self, tmp_path) -> None:
        _generate(tmp_path, _two_ports(modes=(1, 2)))
        rows = read_port_modes(tmp_path / PORT_MODES_FILENAME)
        assert [
            (r.index, r.port, r.portnumber, r.mode, r.active, r.attributes)
            for r in rows
        ] == [
            (1, "o1", 1, 1, True, (6,)),
            (2, "o2", 2, 1, True, (7,)),
            (3, "o1", 1, 2, False, (6,)),
            (4, "o2", 2, 2, False, (7,)),
        ]
        assert faces_by_index(rows) == {1: "o1", 2: "o2", 3: "o1", 4: "o2"}
        assert [r.excited for r in rows] == [True, True, False, False]

    def test_the_written_entries_obey_palaces_rule_for_shared_faces(
        self, tmp_path
    ) -> None:
        """One active entry per attribute, and it comes first in Index order."""
        config = _generate(tmp_path, _two_ports(modes=(1, 2, 3)))
        taken: set[int] = set()
        for entry in sorted(_entries(config), key=lambda e: e["Index"]):
            (attribute,) = entry["Attributes"]
            if entry.get("Active", True):
                assert attribute not in taken
            taken.add(attribute)

    def test_a_stale_sidecar_does_not_outlive_a_plain_config(self, tmp_path) -> None:
        _generate(tmp_path, _two_ports(modes=(1, 2)))
        assert (tmp_path / PORT_MODES_FILENAME).exists()
        _generate(tmp_path, _two_ports())
        assert not (tmp_path / PORT_MODES_FILENAME).exists()

    def test_two_ports_on_one_face_are_refused(self, tmp_path) -> None:
        ports = [
            _wave("a", 100.0, 0.0, modes=(1, 2)),
            _wave("b", 100.0, 0.0),
        ]
        with pytest.raises(ValueError, match="same face"):
            _generate(tmp_path, ports)

    def test_a_single_mode_list_adds_no_entries(self, tmp_path) -> None:
        config = _generate(tmp_path, _two_ports(modes=(1,)))
        assert [e["Index"] for e in _entries(config)] == [1, 2]
        assert len(read_port_modes(tmp_path)) == 2


class TestSchema:
    """The entries obey Palace's own JSON schema, when the schema file is at hand."""

    def test_entries_validate_against_the_palace_schema(self, tmp_path) -> None:
        jsonschema = pytest.importorskip("jsonschema")
        location = os.environ.get("PALACE_CONFIG_SCHEMA")
        if not location or not Path(location).is_file():
            pytest.skip("set PALACE_CONFIG_SCHEMA to Palace's config-schema.json")
        schema = json.loads(Path(location).read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(
            {"$ref": "#/$defs/WavePort", "$defs": schema["$defs"]}
        )
        ports = _two_ports(
            modes=(1, 2, 3),
            excite_modes="all",
            pilot_paths={1: PILOT, 3: PILOT},
            eigensolver_type="SLEPc",
            eigensolver_tol=1e-8,
            eigensolver_max_size=40,
        )
        entries = _entries(_generate(tmp_path, ports))
        assert len(entries) == 6
        for entry in entries:
            validator.validate(entry)


class TestSidecarChecks:
    @staticmethod
    def _row(index, portnumber, mode, active, port="o1") -> PortModeEntry:
        return PortModeEntry(
            index=index,
            port=port,
            portnumber=portnumber,
            mode=mode,
            active=active,
            excited=active,
            attributes=(portnumber,),
        )

    def test_a_good_set_passes(self) -> None:
        check_port_mode_entries([self._row(1, 1, 1, True), self._row(2, 1, 2, False)])

    def test_the_active_entry_must_have_the_lowest_index(self) -> None:
        rows = [self._row(1, 1, 1, False), self._row(2, 1, 2, True)]
        with pytest.raises(ValueError, match="lowest Index"):
            check_port_mode_entries(rows)

    def test_a_face_needs_exactly_one_active_entry(self) -> None:
        with pytest.raises(ValueError, match="0 active"):
            check_port_mode_entries([self._row(1, 1, 1, False)])
        with pytest.raises(ValueError, match="2 active"):
            check_port_mode_entries(
                [self._row(1, 1, 1, True), self._row(2, 1, 2, True)]
            )

    def test_an_index_or_a_mode_cannot_repeat(self) -> None:
        with pytest.raises(ValueError, match="used twice"):
            check_port_mode_entries(
                [self._row(1, 1, 1, True), self._row(1, 2, 1, True, port="o2")]
            )
        with pytest.raises(ValueError, match="mode twice"):
            check_port_mode_entries(
                [self._row(1, 1, 1, True), self._row(3, 1, 1, False)]
            )

    def test_writing_refuses_a_bad_set_and_reading_refuses_a_bad_file(
        self, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="0 active"):
            write_port_modes(tmp_path, [self._row(1, 1, 1, False)])
        with pytest.raises(FileNotFoundError):
            read_port_modes(tmp_path)
        (tmp_path / PORT_MODES_FILENAME).write_text('{"entries": [{"index": 1}]}')
        with pytest.raises(ValueError, match="malformed"):
            read_port_modes(tmp_path)
        (tmp_path / PORT_MODES_FILENAME).write_text("[]")
        with pytest.raises(ValueError, match="no 'entries'"):
            read_port_modes(tmp_path)

    def test_same_face_pairs_needs_position_and_orientation(self) -> None:
        faces = [
            ("a", (0.0, 0.0), 0.0),
            ("b", (0.0, 0.0), 360.0),
            ("c", (0.0, 0.0), 180.0),  # faces the other way: the same face
            ("d", (5.0, 0.0), 0.0),  # somewhere else
            ("e", (0.0, 0.0), 90.0),  # across the other axis
        ]
        assert same_face_pairs(faces) == [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
        ]


class TestSimApi:
    def test_add_multimode_wave_port_declares_one_port_with_its_modes(self) -> None:
        sim = DrivenSim()
        sim.add_multimode_wave_port(
            "o1",
            layer="metal1",
            modes=(1, 2, 3),
            active_mode=2,
            excite="all",
            pilot_paths={2: PILOT},
            max_size=True,
        )
        (wp,) = sim.wave_ports
        assert (wp.name, wp.mode, wp.modes, wp.active_mode) == ("o1", 2, (1, 2, 3), 2)
        assert wp.excite_modes == "all"
        assert wp.pilot_paths == {2: PILOT}
        assert wp.max_size is True

    def test_declaring_a_port_again_replaces_it(self) -> None:
        sim = DrivenSim()
        sim.add_wave_port("o1", layer="metal1")
        sim.add_multimode_wave_port("o1", layer="metal1", modes=(1, 2))
        sim.add_multimode_wave_port("o1", layer="metal1", modes=(1, 2), active_mode=2)
        assert len(sim.wave_ports) == 1
        assert sim.wave_ports[0].mode == 2

    def test_bad_modes_are_refused_when_declared(self) -> None:
        sim = DrivenSim()
        with pytest.raises(ValueError, match="not in modes"):
            sim.add_multimode_wave_port(
                "o1", layer="metal1", modes=(2, 3), active_mode=1
            )
        assert sim.wave_ports == []

    def test_only_driven_simulations_take_one(self) -> None:
        with pytest.raises(ValueError, match="driven"):
            EigenmodeSim().add_multimode_wave_port("o1", layer="metal1", modes=(1, 2))

    def test_a_cross_section_takes_no_ports_at_all(self) -> None:
        sim = BoundaryModeSim()
        sim.set_cross_section("x=0")
        with pytest.raises(ValueError, match="cross_section-only native 2D"):
            sim.add_multimode_wave_port("o1", layer="metal1", modes=(1, 2))
        assert not hasattr(sim, "wave_ports")

    def test_the_modes_travel_through_the_component_port(self) -> None:
        c = _component(("o1", 0, 0))
        configure_wave_port(
            c.ports["o1"],
            layer="metal1",
            mode=2,
            modes=(1, 2, 3),
            excite_modes="all",
            pilot_paths={1: PILOT, 3: PILOT},
        )
        (port,) = extract_ports(c, LayerStack())
        assert port.mode == 2
        assert port.modes == (1, 2, 3)
        assert port.excite_modes == "all"
        assert port.pilot_paths == {1: PILOT, 3: PILOT}

    def test_a_plain_wave_port_carries_no_modes(self) -> None:
        c = _component(("o1", 0, 0))
        configure_wave_port(c.ports["o1"], layer="metal1")
        (port,) = extract_ports(c, LayerStack())
        assert port.modes == ()
        assert port.pilot_paths == {}
        assert port.excite_modes == "active"

    def test_validate_config_refuses_two_ports_on_one_face(self) -> None:
        c = _component(("o1", 10, 0), ("o2", 10, 0))
        sim = DrivenSim()
        sim.set_geometry(c)
        sim.add_multimode_wave_port("o1", layer="metal1", modes=(1, 2))
        sim.add_wave_port("o2", layer="metal1")
        result = sim.validate_config()
        assert not result.valid
        assert any("same face" in message for message in result.errors)

    def test_validate_config_accepts_separate_faces(self) -> None:
        c = _component(("o1", 0, 180), ("o2", 100, 0))
        sim = DrivenSim()
        sim.set_geometry(c)
        for name in ("o1", "o2"):
            sim.add_multimode_wave_port(name, layer="metal1", modes=(1, 2))
        result = sim.validate_config()
        assert not any("same face" in message for message in result.errors)


class TestMeshedLine:
    """A real (coarse) mesh: one surface per face, however many modes it follows."""

    @staticmethod
    def _line(length: float = 100.0) -> gf.Component:
        gf.gpdk.PDK.activate()
        layer = gf.gpdk.LAYER.M1
        c = gf.Component()
        for y0, y1 in [(-61.0, -11.0), (-5.0, 5.0), (11.0, 61.0)]:
            c.add_polygon(
                [
                    (-length / 2, y0),
                    (length / 2, y0),
                    (length / 2, y1),
                    (-length / 2, y1),
                ],
                layer=layer,
            )
        for name, x, angle in (("o1", -length / 2, 180), ("o2", length / 2, 0)):
            c.add_port(
                name=name,
                center=(x, 0),
                width=10,
                orientation=angle,
                port_type="electrical",
                layer=layer,
            )
        return c

    def test_each_face_is_one_physical_group_shared_by_its_entries(
        self, tmp_path
    ) -> None:
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(self._line())
        sim.set_stack(substrate_thickness=2.0)
        sim.set_airbox(margin_x=0.0, margin_y=40.0, z_above=40.0, z_below=10.0)
        for name in ("o1", "o2"):
            sim.add_multimode_wave_port(
                name, layer="metal1", modes=(1, 2), max_size=True, excite="all"
            )
        sim.set_driven(fmin=1e9, fmax=50e9, num_points=3)
        sim.mesh(preset="coarse")
        groups = sim._last_mesh_result.groups
        assert sorted(groups["port_surfaces"]) == ["P1", "P2"]
        config_path = sim.write_config(validate_mesh=False)
        entries = json.loads(Path(config_path).read_text())["Boundaries"]["WavePort"]
        surface = {k: [v["phys_group"]] for k, v in groups["port_surfaces"].items()}
        by_index = {e["Index"]: e for e in entries}
        assert sorted(by_index) == [1, 2, 3, 4]
        assert by_index[1]["Attributes"] == by_index[3]["Attributes"] == surface["P1"]
        assert by_index[2]["Attributes"] == by_index[4]["Attributes"] == surface["P2"]
        assert [r.index for r in read_port_modes(sim._output_dir)] == [1, 2, 3, 4]
