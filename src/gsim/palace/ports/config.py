"""Port configuration for Palace EM simulation.

Ports define where excitation and measurement occur in the simulation.
This module provides helpers to configure gdsfactory ports with Palace metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gsim.palace.stack import LayerStack


class PortType(Enum):
    """Palace port types (maps to Palace config)."""

    LUMPED = "lumped"  # LumpedPort: internal boundary with circuit impedance
    WAVEPORT = "waveport"  # WavePort: domain boundary, modal port
    # SURFACE_CURRENT = "surface_current"  # Future: inductance matrix extraction
    # TERMINAL = "terminal"  # Future: capacitance matrix extraction (electrostatics)


class PortGeometry(Enum):
    """Internal geometry type for mesh generation."""

    INPLANE = "inplane"  # Horizontal surface on single metal layer (Direction: +X, +Y)
    VIA = "via"  # Vertical surface between two metal layers (Direction: +Z)


@dataclass
class PalacePort:
    """Port definition for Palace simulation."""

    name: str
    port_type: PortType = PortType.LUMPED  # Palace port type
    geometry: PortGeometry = PortGeometry.INPLANE  # Mesh geometry type
    center: tuple[float, float] = (0.0, 0.0)  # (x, y) in um
    width: float = 0.0  # um
    orientation: float = 0.0  # degrees (0=east, 90=north, 180=west, 270=south)

    # Z coordinates (filled from stack)
    zmin: float = 0.0
    zmax: float = 0.0

    # Layer info
    layer: str | None = None  # For inplane: target layer
    from_layer: str | None = None  # For via: bottom layer
    to_layer: str | None = None  # For via: top layer

    # Port geometry
    length: float | None = None  # Port extent along direction (um)

    # Multi-element support (for CPW)
    multi_element: bool = False
    centers: list[tuple[float, float]] | None = None  # Multiple centers for CPW
    directions: list[str] | None = (
        None  # Direction per element for CPW (e.g., ["+Y", "-Y"])
    )

    # Electrical properties
    impedance: float = 50.0  # Ohms
    excited: bool = True  # Whether this port is excited (vs just measured)

    @property
    def direction(self) -> str:
        """Get direction from orientation."""
        # Normalize orientation to 0-360
        angle = self.orientation % 360
        if angle < 45 or angle >= 315:
            return "x"  # East
        if 45 <= angle < 135:
            return "y"  # North
        if 135 <= angle < 225:
            return "-x"  # West
        return "-y"  # South


def configure_inplane_port(
    ports,
    layer: str,
    length: float,
    impedance: float = 50.0,
    excited: bool = True,
):
    """Configure gdsfactory port(s) as inplane (lumped) ports for Palace simulation.

    Inplane ports are horizontal ports on a single metal layer, used for CPW gaps
    or similar structures where excitation occurs in the XY plane.

    Args:
        ports: Single gdsfactory Port or iterable of Ports (e.g., c.ports)
        layer: Target conductor layer name (e.g., 'topmetal2')
        length: Port extent along direction in um (perpendicular to port width)
        impedance: Port impedance in Ohms (default: 50)
        excited: Whether port is excited vs just measured (default: True)

    Examples:
        ```python
        configure_inplane_port(c.ports["o1"], layer="topmetal2", length=5.0)
        configure_inplane_port(c.ports, layer="topmetal2", length=5.0)  # all ports
        ```
    """
    # Handle single port or iterable
    port_list = [ports] if hasattr(ports, "info") else ports

    for port in port_list:
        port.info["palace_type"] = "lumped"
        port.info["layer"] = layer
        port.info["length"] = length
        port.info["impedance"] = impedance
        port.info["excited"] = excited


def configure_via_port(
    ports,
    from_layer: str,
    to_layer: str,
    impedance: float = 50.0,
    excited: bool = True,
):
    """Configure gdsfactory port(s) as via (vertical) lumped ports.

    Via ports are vertical lumped ports between two metal layers, used for microstrip
    feed structures where excitation occurs in the Z direction.

    Args:
        ports: Single gdsfactory Port or iterable of Ports (e.g., c.ports)
        from_layer: Bottom conductor layer name (e.g., 'metal1')
        to_layer: Top conductor layer name (e.g., 'topmetal2')
        impedance: Port impedance in Ohms (default: 50)
        excited: Whether port is excited vs just measured (default: True)

    Examples:
        ```python
        configure_via_port(c.ports["o1"], from_layer="metal1", to_layer="topmetal2")
        configure_via_port(
            c.ports, from_layer="metal1", to_layer="topmetal2"
        )  # all ports
        ```
    """
    # Handle single port or iterable
    port_list = [ports] if hasattr(ports, "info") else ports

    for port in port_list:
        port.info["palace_type"] = "lumped"
        port.info["from_layer"] = from_layer
        port.info["to_layer"] = to_layer
        port.info["impedance"] = impedance
        port.info["excited"] = excited


def configure_cpw_port(
    port_upper,
    port_lower,
    layer: str,
    length: float,
    impedance: float = 50.0,
    excited: bool = True,
    cpw_name: str | None = None,
):
    """Configure two gdsfactory ports as a CPW (multi-element) lumped port.

    In CPW (Ground-Signal-Ground), E-fields are opposite in the two gaps.
    This function links two ports to form one multi-element lumped port
    that Palace will excite with proper CPW mode.

    Args:
        port_upper: gdsfactory Port for upper gap (signal-to-ground2)
        port_lower: gdsfactory Port for lower gap (ground1-to-signal)
        layer: Target conductor layer name (e.g., 'topmetal2')
        length: Port extent along direction (um)
        impedance: Port impedance in Ohms (default: 50)
        excited: Whether port is excited (default: True)
        cpw_name: Optional name for the CPW port (default: uses port_lower.name)

    Examples:
        ```python
        configure_cpw_port(
            port_upper=c.ports["gap_upper"],
            port_lower=c.ports["gap_lower"],
            layer="topmetal2",
            length=5.0,
        )
        ```
    """
    # Generate unique CPW group ID
    cpw_group_id = cpw_name or f"cpw_{port_lower.name}"

    # Auto-detect directions based on positions
    upper_y = float(port_upper.center[1])
    lower_y = float(port_lower.center[1])

    # The port farther from origin in Y gets negative direction (E toward signal)
    # The port closer to origin gets positive direction (E toward signal)
    if upper_y > lower_y:
        upper_direction = "-Y"
        lower_direction = "+Y"
    else:
        upper_direction = "+Y"
        lower_direction = "-Y"

    # Store metadata on BOTH ports, marking them as CPW elements
    for port, direction in [
        (port_upper, upper_direction),
        (port_lower, lower_direction),
    ]:
        port.info["palace_type"] = "cpw_element"
        port.info["cpw_group"] = cpw_group_id
        port.info["cpw_direction"] = direction
        port.info["layer"] = layer
        port.info["length"] = length
        port.info["impedance"] = impedance
        port.info["excited"] = excited


def extract_ports(component, stack: LayerStack) -> list[PalacePort]:
    """Extract Palace ports from a gdsfactory component.

    Handles all port types: inplane, via, and CPW (multi-element).
    CPW ports are automatically grouped by their cpw_group ID.

    Args:
        component: gdsfactory Component with configured ports
        stack: LayerStack from stack module

    Returns:
        List of PalacePort objects ready for simulation
    """
    palace_ports = []

    # First, collect CPW elements grouped by cpw_group
    cpw_groups: dict[str, list] = {}

    for port in component.ports:
        info = port.info
        palace_type = info.get("palace_type")

        if palace_type is None:
            continue

        if palace_type == "cpw_element":
            group_id = info.get("cpw_group")
            if group_id:
                if group_id not in cpw_groups:
                    cpw_groups[group_id] = []
                cpw_groups[group_id].append(port)
            continue

        # Handle single-element ports (lumped, waveport)
        center = (float(port.center[0]), float(port.center[1]))
        width = float(port.width)
        orientation = float(port.orientation) if port.orientation is not None else 0.0

        zmin, zmax = 0.0, 0.0
        from_layer = info.get("from_layer")
        to_layer = info.get("to_layer")
        layer_name = info.get("layer")

        if palace_type == "lumped":
            port_type = PortType.LUMPED
            if from_layer and to_layer:
                geometry = PortGeometry.VIA
                if from_layer in stack.layers:
                    zmin = stack.layers[from_layer].zmin
                if to_layer in stack.layers:
                    zmax = stack.layers[to_layer].zmax
            elif layer_name:
                geometry = PortGeometry.INPLANE
                if layer_name in stack.layers:
                    layer = stack.layers[layer_name]
                    zmin = layer.zmin
                    zmax = layer.zmax
            else:
                raise ValueError(f"Lumped port '{port.name}' missing layer info")

        elif palace_type == "waveport":
            port_type = PortType.WAVEPORT
            geometry = PortGeometry.INPLANE  # Waveport geometry TBD
            zmin, zmax = stack.get_z_range()

        else:
            raise ValueError(f"Unknown port type: {palace_type}")

        palace_port = PalacePort(
            name=port.name,
            port_type=port_type,
            geometry=geometry,
            center=center,
            width=width,
            orientation=orientation,
            zmin=zmin,
            zmax=zmax,
            layer=layer_name,
            from_layer=from_layer,
            to_layer=to_layer,
            length=info.get("length"),
            impedance=info.get("impedance", 50.0),
            excited=info.get("excited", True),
        )
        palace_ports.append(palace_port)

    # Now process CPW groups into multi-element PalacePort objects
    for group_id, ports in cpw_groups.items():
        if len(ports) != 2:
            raise ValueError(
                f"CPW group '{group_id}' must have exactly 2 ports, got {len(ports)}"
            )

        # Sort by Y position to get consistent ordering
        ports_sorted = sorted(ports, key=lambda p: p.center[1], reverse=True)
        port_upper, port_lower = ports_sorted[0], ports_sorted[1]

        info = port_lower.info
        layer_name = info.get("layer")

        # Get z coordinates from stack
        zmin, zmax = 0.0, 0.0
        if layer_name and layer_name in stack.layers:
            layer = stack.layers[layer_name]
            zmin = layer.zmin
            zmax = layer.zmax

        # Get centers and directions
        centers = [
            (float(port_upper.center[0]), float(port_upper.center[1])),
            (float(port_lower.center[0]), float(port_lower.center[1])),
        ]
        directions = [
            port_upper.info.get("cpw_direction", "-Y"),
            port_lower.info.get("cpw_direction", "+Y"),
        ]

        # Use average center for the main center field
        avg_center = (
            (centers[0][0] + centers[1][0]) / 2,
            (centers[0][1] + centers[1][1]) / 2,
        )

        cpw_port = PalacePort(
            name=group_id,
            port_type=PortType.LUMPED,
            geometry=PortGeometry.INPLANE,
            center=avg_center,
            width=float(port_lower.width),
            orientation=float(port_lower.orientation)
            if port_lower.orientation
            else 0.0,
            zmin=zmin,
            zmax=zmax,
            layer=layer_name,
            length=info.get("length"),
            multi_element=True,
            centers=centers,
            directions=directions,
            impedance=info.get("impedance", 50.0),
            excited=info.get("excited", True),
        )
        palace_ports.append(cpw_port)

    return palace_ports
