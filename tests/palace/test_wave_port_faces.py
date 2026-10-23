"""Wave ports whose generated faces overlap, and ports facing opposite ways.

The mesher builds one rectangle per wave port. A ``max_size`` port spans the
whole side of the domain whatever its centre, so two ``max_size`` ports on one
side get the same rectangle (review of the multimode wave port: two ports at
x=-50, y=-20 and y=+20). Overlapping rectangles merge into one surface, one
port keeps none and ``write_config()`` emits no ``WavePort`` entry for it.
Validation and the mesher now refuse such ports.

Orientation counts modulo 180 degrees for the same-face rule: 0 and 180 (or 90
and 270) at one centre build the same rectangle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import gdsfactory as gf
import gmsh
import pytest

from gsim.palace import DrivenSim
from gsim.palace.mesh.config_generator import _multimode_entries
from gsim.palace.mesh.geometry import (
    UNBOUNDED_DOMAIN,
    add_ports,
    overlapping_wave_ports,
    wave_port_face,
    wave_port_faces_overlap,
)
from gsim.palace.ports.config import PalacePort, PortType
from gsim.palace.ports.modes import same_face_pairs

# (xmin, ymin, zmin, xmax, ymax, zmax) of a domain around the ports below
DOMAIN = (-100.0, -80.0, -2.0, 100.0, 80.0, 6.0)


def _wave(
    name: str,
    x: float = -50.0,
    y: float = 0.0,
    angle: float = 180.0,
    layer: str = "m1",
    **kwargs,
) -> PalacePort:
    return PalacePort(
        name=name,
        port_type=PortType.WAVEPORT,
        layer=layer,
        center=(x, y),
        width=10.0,
        orientation=angle,
        **kwargs,
    )


def _overlaps(stack, *ports: PalacePort, domain=DOMAIN) -> list[tuple[str, str]]:
    return overlapping_wave_ports(ports, stack, domain_bounds=domain)


class TestFaceRectangles:
    def test_max_size_faces_ignore_the_centre(self, port_stack) -> None:
        """The review case: both faces are the full side of the domain."""
        faces = []
        for name, y in (("o1", -20.0), ("o2", 20.0)):
            port = _wave(name, y=y, max_size=True)
            face, is_y_axis, cut = wave_port_face(
                port, port_stack, port_stack.layers["m1"], domain_bounds=DOMAIN
            )
            assert not is_y_axis
            assert not cut
            faces.append(face)
        assert faces[0] == faces[1] == (-50.0, -80.0, -2.0, -50.0, 80.0, 6.0)

    def test_finite_faces_follow_width_and_margins(self, port_stack) -> None:
        port = _wave("o1", y=20.0, lateral_margin=3.0, z_margin=0.5)
        face, _, _ = wave_port_face(
            port, port_stack, port_stack.layers["m1"], domain_bounds=DOMAIN
        )
        assert face == (-50.0, 12.0, 0.5, -50.0, 28.0, 2.5)

    def test_the_unbounded_domain_makes_max_size_ports_span_everything(
        self, port_stack
    ) -> None:
        face, _, _ = wave_port_face(
            _wave("o1", max_size=True),
            port_stack,
            port_stack.layers["m1"],
            domain_bounds=UNBOUNDED_DOMAIN,
        )
        assert face[0] == face[3] == -50.0
        assert (face[1], face[4]) == (-math.inf, math.inf)
        assert (face[2], face[5]) == (-math.inf, math.inf)


class TestOverlap:
    def test_two_max_size_ports_on_one_side_overlap(self, port_stack) -> None:
        a = _wave("o1", y=-20.0, max_size=True)
        b = _wave("o2", y=20.0, max_size=True)
        assert _overlaps(port_stack, a, b) == [("o1", "o2")]
        # the domain does not matter: the same without one
        assert _overlaps(port_stack, a, b, domain=UNBOUNDED_DOMAIN) == [("o1", "o2")]

    def test_a_max_size_port_covers_a_finite_one_on_its_side(self, port_stack) -> None:
        a = _wave("o1", y=-20.0, max_size=True)
        b = _wave("o2", y=20.0)
        assert _overlaps(port_stack, a, b) == [("o1", "o2")]

    def test_separate_finite_ports_do_not_overlap(self, port_stack) -> None:
        a = _wave("o1", y=-20.0)
        b = _wave("o2", y=20.0)
        assert _overlaps(port_stack, a, b) == []

    def test_margins_can_make_faces_overlap(self, port_stack) -> None:
        # y = -20 +- (5 + 15) and y = 20 +- (5 + 15) touch at y = 0 only
        touching = [
            _wave(n, y=y, lateral_margin=15.0) for n, y in (("a", -20), ("b", 20))
        ]
        assert _overlaps(port_stack, *touching) == []
        wider = [_wave(n, y=y, lateral_margin=16.0) for n, y in (("a", -20), ("b", 20))]
        assert _overlaps(port_stack, *wider) == [("a", "b")]

    def test_opposite_sides_do_not_overlap(self, port_stack) -> None:
        a = _wave("o1", x=-50.0, angle=180.0, max_size=True)
        b = _wave("o2", x=50.0, angle=0.0, max_size=True)
        assert _overlaps(port_stack, a, b) == []

    def test_faces_in_perpendicular_planes_do_not_overlap(self, port_stack) -> None:
        a = _wave("o1", x=-50.0, angle=180.0, max_size=True)
        b = _wave("o2", y=-40.0, x=0.0, angle=270.0, max_size=True)
        assert _overlaps(port_stack, a, b) == []

    def test_ports_on_different_layers_overlap_only_with_a_z_margin(
        self, port_stack
    ) -> None:
        # m1 spans z 1..2 and m2 spans z 3..5: separate faces in one plane
        low = _wave("lo", layer="m1")
        high = _wave("hi", layer="m2")
        assert _overlaps(port_stack, low, high) == []
        # a margin of 1.5 takes m1 to z 0..3.5 and reaches into m2
        tall = _wave("lo", layer="m1", z_margin=1.5)
        assert _overlaps(port_stack, tall, high) == [("lo", "hi")]

    def test_ports_facing_opposite_ways_at_one_centre_overlap(self, port_stack) -> None:
        a = _wave("o1", x=10.0, angle=0.0)
        b = _wave("o2", x=10.0, angle=180.0)
        assert _overlaps(port_stack, a, b) == [("o1", "o2")]

    def test_only_single_element_wave_ports_on_the_stack_count(
        self, port_stack
    ) -> None:
        wave = _wave("o1", max_size=True)
        lumped = PalacePort(name="l", layer="m1", center=(-50.0, 0.0), width=10.0)
        elsewhere = _wave("o3", layer="absent", max_size=True)
        unlayered = PalacePort(
            name="o4", port_type=PortType.WAVEPORT, center=(-50.0, 0.0), width=10.0
        )
        assert _overlaps(port_stack, wave, lumped, elsewhere, unlayered) == []

    def test_several_pairs_come_in_port_order(self, port_stack) -> None:
        ports = [
            _wave(n, y=y, max_size=True) for n, y in (("a", -20), ("b", 0), ("c", 20))
        ]
        assert _overlaps(port_stack, *ports) == [("a", "b"), ("a", "c"), ("b", "c")]

    @pytest.mark.parametrize(
        ("first", "second", "expected"),
        [
            ((0, -5, 0, 0, 5, 4), (0, 5, 0, 0, 15, 4), False),  # edge contact
            ((0, -5, 0, 0, 5, 4), (0, 4, 0, 0, 15, 4), True),
            ((0, -5, 0, 0, 5, 4), (0, 0, 4, 0, 10, 8), False),  # z contact
            ((0, -5, 0, 0, 5, 4), (1, -5, 0, 1, 5, 4), False),  # parallel planes
            ((-5, 0, 0, 5, 0, 4), (-5, 0, 0, 5, 0, 4), True),  # constant y
            ((0, -5, 0, 0, 5, 4), (-5, 0, 0, 5, 0, 4), False),  # crossing planes
        ],
    )
    def test_the_face_overlap_rule(self, first, second, expected) -> None:
        assert wave_port_faces_overlap(first, second) is expected
        assert wave_port_faces_overlap(second, first) is expected


class TestMesherRefuses:
    def test_overlapping_faces_are_refused_before_anything_is_built(
        self, kernel, port_stack
    ) -> None:
        ports = [_wave(n, y=y, max_size=True) for n, y in (("o1", -20), ("o2", 20))]
        with pytest.raises(ValueError, match="'o1' and 'o2' generate overlapping"):
            add_ports(kernel, ports, port_stack, domain_bounds=DOMAIN)
        assert gmsh.model.getEntities(2) == []

    def test_separate_faces_get_a_surface_each(self, kernel, port_stack) -> None:
        ports = [_wave(n, y=y) for n, y in (("o1", -20), ("o2", 20))]
        tags, info = add_ports(kernel, ports, port_stack, domain_bounds=DOMAIN)
        assert sorted(tags) == ["P1", "P2"]
        assert [(p["ymin"], p["ymax"]) for p in info] == [(-25.0, -15.0), (15.0, 25.0)]


class TestSameFaceModulo180:
    @pytest.mark.parametrize(
        ("angle_a", "angle_b", "same"),
        [
            (0, 0, True),
            (0, 360, True),
            (0, 180, True),
            (180, 0, True),
            (90, 270, True),
            (-90, 90, True),
            (0, 180 + 1e-8, True),
            (0, 90, False),
            (0, 270, False),
            (0, 180.01, False),
            (45, 225, True),
        ],
    )
    def test_orientations_count_modulo_180(self, angle_a, angle_b, same) -> None:
        faces = [("a", (3.0, 4.0), angle_a), ("b", (3.0, 4.0), angle_b)]
        assert same_face_pairs(faces) == ([("a", "b")] if same else [])

    def test_the_centre_still_has_to_agree(self) -> None:
        faces = [("a", (0.0, 0.0), 0.0), ("b", (0.0, 5.0), 180.0)]
        assert same_face_pairs(faces) == []

    def test_the_multimode_entries_refuse_opposite_ports_at_one_centre(self) -> None:
        a = _wave("a", x=100.0, angle=0.0, modes=(1, 2))
        b = _wave("b", x=100.0, angle=180.0)
        with pytest.raises(ValueError, match="same face"):
            _multimode_entries([(1, a, 6)], 3, [{"Index": 1, "Mode": 1}], [a, b])


def _component(*ports: tuple[str, float, float, float]) -> gf.Component:
    """A strip on the generic M1 with a port ``(name, x, y, orientation)`` each."""
    gf.gpdk.PDK.activate()
    layer = gf.gpdk.LAYER.M1
    c = gf.Component()
    c.add_polygon([(-50, -61), (50, -61), (50, 61), (-50, 61)], layer=layer)
    for name, x, y, angle in ports:
        c.add_port(
            name=name,
            center=(x, y),
            width=10,
            orientation=angle,
            port_type="electrical",
            layer=layer,
        )
    return c


def _sim(
    *ports: tuple[str, float, float, float],
    multimode: bool = False,
    **options,
) -> DrivenSim:
    sim = DrivenSim()
    sim.set_geometry(_component(*ports))
    for name, *_ in ports:
        if multimode:
            sim.add_multimode_wave_port(name, layer="metal1", modes=(1, 2), **options)
        else:
            sim.add_wave_port(name, layer="metal1", **options)
    return sim


REVIEW_PORTS = (("o1", -50.0, -20.0, 180.0), ("o2", -50.0, 20.0, 180.0))


class TestValidateConfig:
    @pytest.mark.parametrize("multimode", [True, False], ids=["multimode", "plain"])
    def test_two_max_size_ports_on_one_side_are_refused(self, multimode) -> None:
        """The review case: x=-50, y=-20 and y=+20, ``max_size=True``."""
        result = _sim(
            *REVIEW_PORTS, multimode=multimode, max_size=True
        ).validate_config()
        assert not result.valid
        (message,) = [e for e in result.errors if "overlapping faces" in e]
        assert "'o1' and 'o2'" in message
        assert "max_size" in message

    def test_the_same_ports_without_max_size_are_accepted(self) -> None:
        result = _sim(*REVIEW_PORTS, multimode=True).validate_config()
        assert not any("overlapping" in e or "same face" in e for e in result.errors)
        assert result.valid

    def test_margins_that_overlap_are_refused(self) -> None:
        # y=-20 +- (5+16) and y=20 +- (5+16) overlap on y -1..1
        sim = _sim(*REVIEW_PORTS, lateral_margin=16.0)
        assert any("overlapping" in e for e in sim.validate_config().errors)
        sim = _sim(*REVIEW_PORTS, lateral_margin=10.0)
        assert not any("overlapping" in e for e in sim.validate_config().errors)

    def test_opposite_sides_with_max_size_are_accepted(self) -> None:
        sim = _sim(("o1", -50.0, 0.0, 180.0), ("o2", 50.0, 0.0, 0.0), max_size=True)
        result = sim.validate_config()
        assert result.valid, result.errors

    def test_one_port_is_never_a_pair(self) -> None:
        assert _sim(REVIEW_PORTS[0], max_size=True).validate_config().valid

    def test_ports_facing_opposite_ways_at_one_centre_are_one_face(self) -> None:
        ports = (("o1", 10.0, 0.0, 0.0), ("o2", 10.0, 0.0, 180.0))
        sim = _sim(*ports)
        sim.add_multimode_wave_port("o1", layer="metal1", modes=(1, 2))
        errors = sim.validate_config().errors
        assert any("same face" in e for e in errors)
        # reported once, by the face rule, not again as an overlap
        assert not any("overlapping" in e for e in errors)

    def test_plain_ports_at_one_centre_are_refused_as_overlapping(self) -> None:
        ports = (("o1", 10.0, 0.0, 0.0), ("o2", 10.0, 0.0, 180.0))
        errors = _sim(*ports).validate_config().errors
        assert len([e for e in errors if "overlapping faces" in e]) == 1

    def test_validating_leaves_the_stack_unresolved(self) -> None:
        sim = _sim(*REVIEW_PORTS, max_size=True)
        assert sim.stack is None
        sim.validate_config()
        assert sim.stack is None

    def test_meshing_refuses_before_any_work(self, tmp_path) -> None:
        sim = _sim(*REVIEW_PORTS, multimode=True, max_size=True)
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_driven(fmin=1e9, fmax=50e9, num_points=3)
        with pytest.raises(ValueError, match="overlapping faces"):
            sim.mesh(preset="coarse")
        assert not list((tmp_path / "palace-sim").glob("*.msh"))


class TestMeshedPorts:
    def test_ports_with_separate_faces_each_get_a_surface_and_an_entry(
        self, tmp_path
    ) -> None:
        """The fixed layout: o1, o2 at x=-50, y=-20/+20, as finite ports."""
        sim = _sim(
            *REVIEW_PORTS,
            ("o3", 50.0, 0.0, 0.0),
            multimode=True,
            z_margin=20.0,
            lateral_margin=10.0,
        )
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_stack(substrate_thickness=2.0)
        sim.set_airbox(margin_x=0.0, margin_y=20.0, z_above=40.0, z_below=10.0)
        sim.set_driven(fmin=1e9, fmax=50e9, num_points=3)
        sim.mesh(preset="coarse")
        groups = sim._last_mesh_result.groups
        assert sorted(groups["port_surfaces"]) == ["P1", "P2", "P3"]
        config_path = sim.write_config(validate_mesh=False)
        entries = json.loads(Path(config_path).read_text())["Boundaries"]["WavePort"]
        assert sorted(e["Index"] for e in entries) == [1, 2, 3, 4, 5, 6]
        attributes = {tuple(e["Attributes"]) for e in entries}
        assert len(attributes) == 3  # one surface per face
