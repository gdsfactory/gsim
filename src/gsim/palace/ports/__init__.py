"""Port definition for Palace EM simulation.

Usage:
    from gsim.palace.ports import configure_port, extract_ports

    # Configure ports on a component
    c = gf.get_component("straight_metal")
    configure_port(c.ports['o1'], type='lumped', layer='topmetal2')
    configure_port(c.ports['o2'], type='lumped', layer='topmetal2')

    # Extract ports for simulation
    ports = extract_ports(c, stack)
"""

from __future__ import annotations

from gsim.palace.ports.config import (
    PalacePort,
    PortGeometry,
    PortType,
    configure_cpw_port,
    configure_inplane_port,
    configure_via_port,
    extract_ports,
)

__all__ = [
    "PalacePort",
    "PortGeometry",
    "PortType",
    "configure_cpw_port",
    "configure_inplane_port",
    "configure_via_port",
    "extract_ports",
]
