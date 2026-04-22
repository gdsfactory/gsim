"""Tests for gsim.meep.ports."""

from __future__ import annotations

from gsim.meep.models.config import PortData
from gsim.meep.ports import filter_ports_for_xz


def _port(name: str, x: float, y: float, orientation: float, width: float = 0.5):
    """Build a PortData with minimal fields for filter tests."""
    normal_axis = 0 if orientation in (0, 180) else 1
    direction = "-" if orientation in (0, 90) else "+"
    return PortData(
        name=name,
        center=[x, y, 0.0],
        orientation=orientation,
        width=width,
        normal_axis=normal_axis,
        direction=direction,
    )


class TestFilterPortsForXZ:
    """Tests for filter_ports_for_xz."""

    def test_keeps_port_intersecting_cut(self):
        ports = [_port("o1", x=0.0, y=0.0, orientation=180, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert [p.name for p in kept] == ["o1"]

    def test_drops_port_off_cut(self):
        ports = [_port("o1", x=0.0, y=3.0, orientation=180, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert kept == []

    def test_drops_y_facing_port(self):
        ports = [_port("o1", x=0.0, y=0.0, orientation=90, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert kept == []

    def test_partial_overlap_included(self):
        # Port at y=0.2, width 0.5 -> extends from y=-0.05 to y=0.45;
        # cut at y=0 falls inside.
        ports = [_port("o1", x=0.0, y=0.2, orientation=180, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert [p.name for p in kept] == ["o1"]

    def test_mixed_ports(self):
        ports = [
            _port("wg_in", x=-5.0, y=0.0, orientation=180, width=0.5),
            _port("wg_out", x=5.0, y=0.0, orientation=0, width=0.5),
            _port("y_oriented", x=0.0, y=0.0, orientation=90, width=0.5),
            _port("far_port", x=0.0, y=10.0, orientation=180, width=0.5),
        ]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert {p.name for p in kept} == {"wg_in", "wg_out"}
