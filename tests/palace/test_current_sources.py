"""Tests for resolving current sources to sheets (pure Python, no mesh)."""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common.stack import Layer, get_stack
from gsim.palace.current_sources import ResolvedCurrentSource, resolve_current_sources
from gsim.palace.mesh.geometry import extract_geometry
from gsim.palace.mesh.nets import extract_nets
from gsim.palace.models import CurrentSourceConfig
from gsim.palace.ports.config import PortGeometry, PortType


def _setup(component: gf.Component):
    """The generic-PDK stack and the nets of a component, from all its ports."""
    stack = get_stack()
    nets = extract_nets(extract_geometry(component, stack), stack, component.ports)
    return stack, nets


@pytest.fixture(scope="module")
def inductor():
    """``gf.components.inductor``: leads P1/P2 and loop-gap ports o1/o2 on M3."""
    gf.gpdk.PDK.activate()
    component = gf.components.inductor()
    return (component, *_setup(component))


def _resolve(setup, *sources: CurrentSourceConfig) -> list[ResolvedCurrentSource]:
    component, stack, nets = setup
    return resolve_current_sources(sources, component, stack, nets)


def _src(name: str, plus: str, minus: str, layer: str | None = None):
    return CurrentSourceConfig(name=name, plus_port=plus, minus_port=minus, layer=layer)


def test_inductor_leads_resolve_to_the_lead_gap(inductor) -> None:
    (source,) = _resolve(inductor, _src("L", "P1", "P2"))
    # leads 2.1 um wide, centres 4.2 um apart: the gap between them is `space`
    assert source.span == pytest.approx(2.1)
    assert source.direction == "-X"
    assert source.center == pytest.approx((0.0, 2.1))
    assert source.layer == "metal3"
    assert (source.plus_port, source.minus_port) == ("P1", "P2")
    assert source.net is not None


def test_inductor_loop_gap_ports_resolve_as_facing_ends(inductor) -> None:
    component, *_ = inductor
    o1, o2 = component.ports["o1"], component.ports["o2"]
    (source,) = _resolve(inductor, _src("L", "o1", "o2"))
    # o1 (x = +1.05) and o2 (x = -1.05) face each other across the loop gap
    assert source.span == pytest.approx(2.1)
    assert source.center == pytest.approx((0.0, o1.dcenter[1]))
    assert source.direction == "+X"  # from o2 (minus) to o1 (plus)
    assert o2.dcenter[0] < o1.dcenter[0]


def test_swapping_plus_and_minus_flips_only_the_direction(inductor) -> None:
    (forward,) = _resolve(inductor, _src("A", "P1", "P2"))
    (backward,) = _resolve(inductor, _src("A", "P2", "P1"))
    assert backward.direction == "+X"
    assert (backward.center, backward.span) == (forward.center, forward.span)


def test_a_source_becomes_a_surface_current_gap_port(inductor) -> None:
    _, stack, _ = inductor
    (source,) = _resolve(inductor, _src("L", "P1", "P2"))
    port = source.to_palace_port(stack)
    assert port.name == "L"
    assert port.port_type is PortType.SURFACE_CURRENT
    assert port.geometry is PortGeometry.GAP
    assert port.center == source.center
    assert port.width == pytest.approx(2.1)
    assert port.direction == "-x"  # the gap sheet runs along x
    assert (port.zmin, port.zmax) == (3.2, 5.2)  # metal3 of the generic stack
    assert port.layer == "metal3"


@pytest.mark.parametrize(
    ("plus", "minus", "direction"), [("o1", "o2", "x"), ("o2", "o1", "-x")]
)
def test_the_port_orientation_follows_the_current(
    inductor, plus, minus, direction
) -> None:
    _, stack, _ = inductor
    (source,) = _resolve(inductor, _src("L", plus, minus))
    assert source.to_palace_port(stack).direction == direction


def test_no_source_is_rejected(inductor) -> None:
    with pytest.raises(ValueError, match=r"no current source.*add_current_source"):
        _resolve(inductor)


def test_equal_plus_and_minus_ports_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"'L'.*two different ports"):
        _src("L", "P1", "P1")


def test_a_missing_port_is_rejected(inductor) -> None:
    with pytest.raises(ValueError, match=r"'L' \(missing port\).*'P9'.*P1"):
        _resolve(inductor, _src("L", "P1", "P9"))


def _inductor_with(*ports: tuple, layer: tuple[int, int] = (49, 0)):
    """The inductor with extra ports ``(name, x, y, orientation, width)``.

    The extra ports are on ``layer``, M3 (metal3) by default.
    """
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_ports((component << gf.components.inductor()).ports)
    for name, x, y, orientation, width in ports:
        component.add_port(
            name=name,
            center=(x, y),
            width=width,
            orientation=orientation,
            layer=layer,
            port_type="electrical",
        )
    return (component, *_setup(component))


Y_GAP = 30.685  # y of the loop-gap ports o1/o2


@pytest.mark.parametrize(
    ("ports", "plus", "detail"),
    [
        ([("Q", 2.1, 2.1, 0, 2.1)], "P1", "face each other"),  # L-shaped
        ([("Q", 2.1, 2.1, 90, 2.1)], "P1", "face each other"),  # opposite leads
        (  # back to back across the loop gap
            [("R", 1.05, Y_GAP, 0, 2.0), ("Q", -1.05, Y_GAP, 180, 2.0)],
            "R",
            "face each other",
        ),
        ([("Q", -1.05, Y_GAP, 180, 2.0)], "o1", "face each other"),  # in line
        ([("Q", 2.1, 2.1, 45, 2.1)], "P1", "cardinal orientation"),
        ([("Q", 2.1, 3.1, 270, 2.1)], "P1", "not on an x or y line"),
        ([("Q", -0.5, 2.1, 270, 2.1)], "P1", "must be positive"),  # leads overlap
    ],
    ids=[
        "l-shaped",
        "opposite",
        "back-to-back",
        "in-line",
        "non-cardinal",
        "off-axis",
        "span",
    ],
)
def test_ports_that_break_the_bridge_rule_are_rejected(ports, plus, detail) -> None:
    setup = _inductor_with(*ports)
    with pytest.raises(ValueError, match=rf"'L' \(bridge rule\).*{detail}"):
        _resolve(setup, _src("L", plus, "Q"))


def _with_layer(setup, name: str, gds_layer: tuple[int, int], zmin, zmax):
    """A copy of ``setup`` whose stack has one more conductor layer."""
    component, stack, nets = setup
    stack = stack.model_copy(deep=True)
    stack.layers[name] = Layer(
        name=name,
        gds_layer=gds_layer,
        zmin=zmin,
        zmax=zmax,
        thickness=zmax - zmin,
        material="aluminum",
        layer_type="conductor",
    )
    return component, stack, nets


def test_a_given_layer_is_used(inductor) -> None:
    (source,) = _resolve(inductor, _src("L", "P1", "P2", layer="metal3"))
    assert source.layer == "metal3"


def test_ports_on_different_layers_need_a_layer() -> None:
    setup = _inductor_with(("Q", 2.1, 2.1, 270, 2.1), layer=(45, 0))  # metal2
    with pytest.raises(ValueError, match=r"'L' \(layer rule\).*pass layer="):
        _resolve(setup, _src("L", "P1", "Q"))


def test_an_ambiguous_layer_needs_a_layer(inductor) -> None:
    setup = _with_layer(inductor, "metal3_copy", (49, 5), 6.0, 7.0)
    with pytest.raises(ValueError, match=r"'L' \(layer rule\).*pass layer="):
        _resolve(setup, _src("L", "P1", "P2"))
    (source,) = _resolve(setup, _src("L", "P1", "P2", layer="metal3"))
    assert source.layer == "metal3"


@pytest.mark.parametrize(
    ("layer", "detail"),
    [("via1", "not a conductor layer"), ("nope", "not a conductor layer")],
)
def test_a_layer_that_is_not_a_conductor_is_rejected(inductor, layer, detail):
    with pytest.raises(ValueError, match=rf"'L' \(layer rule\).*{detail}"):
        _resolve(inductor, _src("L", "P1", "P2", layer=layer))


def test_a_conductor_layer_without_thickness_is_rejected(inductor) -> None:
    setup = _with_layer(inductor, "flat", (99, 0), 6.0, 6.0)
    with pytest.raises(ValueError, match=r"'L' \(layer rule\).*no thickness"):
        _resolve(setup, _src("L", "P1", "P2", layer="flat"))


# Synthetic M3 layouts: boxes (x0, y0, x1, y1) and ports (name, x, y, angle, w)
LEAD_A = (-3, 0, -1, 10)
LEAD_B = (1, 0, 3, 10)
TOP = (-3, 10, 3, 12)
TIPS = [("P", -2, 0, 270, 2), ("M", 2, 0, 270, 2)]


def _layout(boxes, ports):
    """A component of M3 boxes with electrical ports, and its stack and nets."""
    gf.gpdk.PDK.activate()
    component = gf.Component()
    for x0, y0, x1, y1 in boxes:
        component.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=(49, 0))
    for name, x, y, orientation, width in ports:
        component.add_port(
            name=name,
            center=(x, y),
            width=width,
            orientation=orientation,
            layer=(49, 0),
            port_type="electrical",
        )
    return (component, *_setup(component))


def test_a_closed_u_loop_resolves_to_its_net() -> None:
    setup = _layout([LEAD_A, LEAD_B, TOP], TIPS)
    (source,) = _resolve(setup, _src("U", "P", "M"))
    assert source.net == "P+M"
    assert (source.center, source.span) == ((0.0, 0.0), 2.0)


def test_an_open_loop_is_rejected() -> None:
    setup = _layout([LEAD_A, LEAD_B], TIPS)  # no top bar
    with pytest.raises(ValueError, match=r"'U' \(open loop\).*'P'.*'M'"):
        _resolve(setup, _src("U", "P", "M"))


def test_a_port_on_no_metal_is_an_open_loop() -> None:
    setup = _inductor_with(("Q", 6.3, 2.1, 270, 2.1))
    with pytest.raises(ValueError, match=r"'L' \(open loop\).*'Q'.*no metal"):
        _resolve(setup, _src("L", "P2", "Q"))


def test_a_sheet_that_crosses_metal_is_rejected() -> None:
    island = (-0.5, -1, 0.5, 1)  # in the gap between the lead tips
    setup = _layout([LEAD_A, LEAD_B, TOP, island], TIPS)
    with pytest.raises(ValueError, match=r"'U' \(sheet crosses metal\)"):
        _resolve(setup, _src("U", "P", "M"))


E_LEADS = [(-5, 0, -3, 10), (-1, 0, 1, 10), (3, 0, 5, 10), (-5, 10, 5, 12)]
E_TIPS = [("L", -4, 0, 270, 2), ("C", 0, 0, 270, 2), ("R", 4, 0, 270, 2)]


def test_a_centre_tap_shares_one_port() -> None:
    setup = _layout(E_LEADS, E_TIPS)
    first, second = _resolve(setup, _src("A", "L", "C"), _src("B", "C", "R"))
    assert (first.center, second.center) == ((-2.0, 0.0), (2.0, 0.0))
    assert first.net == second.net


def test_two_sources_on_one_port_pair_are_rejected(inductor) -> None:
    with pytest.raises(ValueError, match=r"'B' \(duplicate port pair\).*'A'"):
        _resolve(inductor, _src("A", "P1", "P2"), _src("B", "P2", "P1"))


def test_two_sources_with_one_name_are_rejected() -> None:
    setup = _layout(E_LEADS, E_TIPS)
    with pytest.raises(ValueError, match=r"'A' \(duplicate name\)"):
        _resolve(setup, _src("A", "L", "C"), _src("A", "C", "R"))


def test_intersecting_sheets_are_rejected() -> None:
    # the inner faces of the leads face each other across the same gap
    faces = [("F1", -1, 0, 0, 1), ("F2", 1, 0, 180, 1)]
    setup = _layout([LEAD_A, LEAD_B, TOP], TIPS + faces)
    with pytest.raises(ValueError, match=r"'B' \(intersecting sheets\).*'A'"):
        _resolve(setup, _src("A", "P", "M"), _src("B", "F1", "F2"))


def test_sources_keep_the_order_they_are_given_in() -> None:
    setup = _layout(E_LEADS, E_TIPS)
    names = [s.name for s in _resolve(setup, _src("B", "C", "R"), _src("A", "L", "C"))]
    assert names == ["B", "A"]
