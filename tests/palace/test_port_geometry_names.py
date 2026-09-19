"""Tests for canonical and legacy lumped-port geometry names."""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.palace import PortGeometry
from gsim.palace.ports import (
    configure_gap_port,
    configure_interlayer_port,
    configure_via_port,
)


def _electrical_port() -> gf.Port:
    gf.gpdk.PDK.activate()
    component = gf.Component()
    return component.add_port(
        name="feed",
        center=(0.0, 0.0),
        width=2.0,
        orientation=0.0,
        layer=(1, 0),
        port_type="electrical",
    )


def test_canonical_port_geometry_names() -> None:
    """The four public lumped geometries have stable, readable names."""
    assert PortGeometry.INPLANE.value == "inplane"
    assert PortGeometry.GAP.value == "gap"
    assert PortGeometry.INTERLAYER.value == "interlayer"
    assert PortGeometry.CPW.value == "cpw"


def test_via_enum_value_remains_backward_compatible() -> None:
    """The exported legacy Enum keeps its serialized value."""
    assert PortGeometry.VIA.value == "via"


def test_configure_interlayer_port_sets_layer_metadata() -> None:
    """The canonical helper configures a Z-directed layer-spanning port."""
    port = _electrical_port()
    configure_interlayer_port(port, from_layer="metal1", to_layer="topmetal2")
    assert port.info["palace_type"] == "lumped"
    assert port.info["from_layer"] == "metal1"
    assert port.info["to_layer"] == "topmetal2"


def test_configure_gap_port_sets_layer_metadata() -> None:
    """The gap helper marks one vertical, single-layer port surface."""
    port = _electrical_port()
    configure_gap_port(port, layer="topmetal2")
    assert port.info["palace_type"] == "gap"
    assert port.info["layer"] == "topmetal2"


def test_configure_via_port_warns_and_preserves_behavior() -> None:
    """The old helper warns but produces the same metadata."""
    port = _electrical_port()
    with pytest.warns(DeprecationWarning, match="configure_interlayer_port"):
        configure_via_port(port, from_layer="metal1", to_layer="topmetal2")
    assert port.info["palace_type"] == "lumped"
    assert port.info["from_layer"] == "metal1"
    assert port.info["to_layer"] == "topmetal2"
