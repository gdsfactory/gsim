"""Verify port sheet area and orientation with real OCC surfaces."""

import pytest

from gsim.palace.mesh.geometry import add_ports
from gsim.palace.ports.config import PalacePort, PortGeometry, PortType


@pytest.mark.parametrize("orientation", [0, 90, 180, 270])
@pytest.mark.parametrize(
    "geometry",
    [
        PortGeometry.EDGE,
        PortGeometry.CPW,
        PortGeometry.INTERLAYER,
        PortGeometry.INPLANE,
        PortGeometry.GAP,
    ],
)
def test_port_surfaces_preserve_area_and_element_metadata(
    kernel, port_stack, orientation, geometry
):
    multi = geometry in {PortGeometry.EDGE, PortGeometry.CPW}
    axis = "X" if orientation % 180 == 0 else "Y"
    port = PalacePort(
        name="p",
        geometry=geometry,
        center=(0, 0),
        orientation=orientation,
        layer="m1",
        from_layer="m2",
        to_layer="m1",
        width=2,
        length=3,
        zmin=1,
        zmax=2,
        multi_element=multi,
        centers=[(0, 0), (4, 4)],
        directions=[f"+{axis}", f"-{axis}"],
    )
    tags, info = add_ports(kernel, [port], port_stack)
    kernel.synchronize()
    expected_area = (
        2
        if geometry in {PortGeometry.EDGE, PortGeometry.INTERLAYER, PortGeometry.GAP}
        else 6
    )
    assert len(tags["P1"]) == (2 if multi else 1)
    assert all(
        kernel.getMass(2, tag) == pytest.approx(expected_area) for tag in tags["P1"]
    )
    assert info[0]["portnumber"] == 1
    assert info[0]["name"] == "p"
    if multi:
        assert [
            element["direction"] for element in info[0]["elements"]
        ] == port.directions


@pytest.mark.parametrize(
    "options",
    [
        {"multi_element": True, "centers": None},
        {"multi_element": True, "directions": None},
        {"multi_element": True, "geometry": PortGeometry.EDGE, "layer": None},
        {"multi_element": True, "geometry": PortGeometry.EDGE, "layer": "absent"},
        {"multi_element": True, "layer": None},
        {"multi_element": True, "layer": "absent"},
        {"geometry": PortGeometry.INTERLAYER, "from_layer": None},
        {"geometry": PortGeometry.INTERLAYER, "from_layer": "absent"},
        {"layer": None},
        {"layer": "absent"},
    ],
)
def test_incomplete_port_metadata_is_not_meshed(kernel, port_stack, options):
    defaults = dict(
        name="p",
        layer="m1",
        from_layer="m1",
        to_layer="m2",
        centers=[(0, 0)],
        directions=["+X"],
        width=1,
    )
    tags, info = add_ports(kernel, [PalacePort(**(defaults | options))], port_stack)
    assert tags == {}
    assert info == []


@pytest.mark.parametrize("orientation", [0, 90])
@pytest.mark.parametrize("bounds_3d", [False, True])
def test_maximum_waveport_fills_requested_domain_face(
    kernel, port_stack, orientation, bounds_3d
):
    port = PalacePort(
        name="wave",
        port_type=PortType.WAVEPORT,
        layer="m1",
        width=1,
        orientation=orientation,
        max_size=True,
    )
    tags, info = add_ports(
        kernel,
        [port],
        port_stack,
        domain_bbox=(-3, -4, 3, 4),
        domain_bounds=(-3, -4, -2, 3, 4, 6) if bounds_3d else None,
    )
    kernel.synchronize()
    width = 8 if orientation == 0 else 6
    assert info[0]["width"] == width
    assert kernel.getMass(2, tags["P1"][0]) == pytest.approx(
        width * (8 if bounds_3d else 5)
    )


def test_maximum_waveport_requires_domain_bounds(kernel, port_stack):
    port = PalacePort(
        name="wave", port_type=PortType.WAVEPORT, layer="m1", width=1, max_size=True
    )
    with pytest.raises(ValueError, match="domain bounds"):
        add_ports(kernel, [port], port_stack)
