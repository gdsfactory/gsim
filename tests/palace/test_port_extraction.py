"""Check electrical port metadata against physical positions and field directions."""

from types import SimpleNamespace

import gdsfactory as gf
import pytest

from gsim.palace.ports.config import PortGeometry, PortType, extract_ports


@pytest.mark.parametrize("geometry", ["gap", "two_terminal"])
def test_simulation_port_configuration_roundtrip(port_stack, geometry):
    from gsim.palace import DrivenSim

    component = gf.Component()
    for name, x, orientation in [("plus", 0, 0), ("minus", 4, 180)]:
        component.add_port(
            name, center=(x, 0), orientation=orientation, width=2, layer=(1, 0)
        )
    sim = DrivenSim()
    sim.set_geometry(component)
    if geometry == "gap":
        sim.add_port("plus", geometry="gap", layer="m1", excited=True)
    else:
        sim.add_two_terminal_port("plus", "minus", layer="m1", excited=True)
        sim.add_two_terminal_port("plus", "minus", layer="m1", excited=True)
        assert len(sim.two_terminal_ports) == 1
    sim._configure_ports_on_component(port_stack)
    (port,) = extract_ports(component, port_stack)
    assert port.excited
    assert port.geometry == (
        PortGeometry.GAP if geometry == "gap" else PortGeometry.EDGE
    )


@pytest.mark.parametrize(
    ("orientation", "directions"),
    [
        (0, ["+X", "-X"]),
        (90, ["+Y", "-Y"]),
        (180, ["-X", "+X"]),
        (270, ["-Y", "+Y"]),
    ],
)
@pytest.mark.parametrize("layer", [None, "m1"])
def test_two_terminal_extraction_preserves_centers_polarity_and_stack_height(
    port_stack, orientation, directions, layer
):
    component = gf.Component()
    plus = component.add_port(
        "plus", center=(1, 2), width=2, orientation=orientation, layer=(1, 0)
    )
    component.add_port(
        "minus",
        center=(4, 6),
        width=2,
        orientation=(orientation + 180) % 360,
        layer=(1, 0),
    )
    plus.info.update(
        {"palace_type": "two_terminal", "minus_port_name": "minus", "layer": layer}
    )
    (port,) = extract_ports(component, port_stack)
    assert port.centers == [(1, 2), (4, 6)]
    assert port.directions == directions
    assert (port.zmin, port.zmax) == ((1, 2) if layer else (0, 0))
    assert port.multi_element


@pytest.mark.parametrize(
    ("kind", "metadata", "geometry", "height"),
    [
        ("lumped", {"layer": "m1"}, PortGeometry.INPLANE, (1, 2)),
        (
            "lumped",
            {"from_layer": "m1", "to_layer": "m2"},
            PortGeometry.INTERLAYER,
            (1, 5),
        ),
        ("lumped", {"layer": "missing"}, PortGeometry.INPLANE, (0, 0)),
        ("waveport", {"layer": "m2"}, PortGeometry.INPLANE, (3, 5)),
        ("waveport", {}, PortGeometry.INPLANE, (0, 0)),
        ("gap", {"layer": "m1"}, PortGeometry.GAP, (1, 2)),
    ],
)
def test_single_element_extraction_keeps_geometry_and_electrical_parameters(
    port_stack, kind, metadata, geometry, height
):
    port = SimpleNamespace(
        name="signal",
        center=(1, 2),
        width=2,
        orientation=None,
        info={"palace_type": kind, "resistance": 10, "capacitance": 1e-12, **metadata},
    )
    (result,) = extract_ports(SimpleNamespace(ports=[port]), port_stack)
    assert result.geometry == geometry
    assert result.port_type == (
        PortType.WAVEPORT if kind == "waveport" else PortType.LUMPED
    )
    assert (result.zmin, result.zmax) == height
    assert (result.resistance, result.capacitance, result.orientation) == (10, 1e-12, 0)


@pytest.mark.parametrize(
    "metadata",
    [
        {"palace_type": "two_terminal"},
        {"palace_type": "two_terminal", "minus_port_name": "missing"},
        {"palace_type": "lumped"},
        {"palace_type": "unrecognized"},
        {"palace_type": "gap", "layer": "missing"},
    ],
)
def test_extraction_rejects_incomplete_or_unknown_port_definitions(
    port_stack, metadata
):
    port = SimpleNamespace(
        name="signal", center=(0, 0), width=1, orientation=0, info=metadata
    )
    with pytest.raises(ValueError):
        extract_ports(SimpleNamespace(ports=[port]), port_stack)
