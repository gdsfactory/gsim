"""Port definition for Palace EM simulation.

Usage:
    from gsim.palace.ports import configure_inplane_port, extract_ports

    # Configure ports on a component
    c = gf.get_component("straight_metal")
    configure_inplane_port(c.ports['o1'], layer='topmetal2', length=5.0)
    configure_inplane_port(c.ports['o2'], layer='topmetal2', length=5.0)

    # Extract ports for simulation
    ports = extract_ports(c, stack)
"""

from __future__ import annotations

from gsim.palace.ports.config import (
    PalacePort,
    PortGeometry,
    PortType,
    configure_cpw_port,
    configure_gap_port,
    configure_inplane_port,
    configure_interlayer_port,
    configure_two_terminal_port,
    configure_via_port,
    configure_wave_port,
    extract_ports,
)

__all__ = [
    "PalacePort",
    "PortGeometry",
    "PortType",
    "configure_cpw_port",
    "configure_gap_port",
    "configure_inplane_port",
    "configure_interlayer_port",
    "configure_two_terminal_port",
    "configure_via_port",
    "configure_wave_port",
    "extract_ports",
]
