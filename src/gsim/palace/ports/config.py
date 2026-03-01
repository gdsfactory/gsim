"""Port configuration for Palace EM simulation.

Ports define where excitation and measurement occur in the simulation.
This module provides helpers to configure gdsfactory ports with Palace metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack


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
    port,
    layer: str,
    s_width: float,
    gap_width: float,
    length: float,
    impedance: float = 50.0,
    excited: bool = True,
):
    """Configure a gdsfactory port as a CPW (multi-element) lumped port.

    In CPW (Ground-Signal-Ground), E-fields are opposite in the two gaps.
    The port should be placed at the signal center. The upper and lower gap
    centers are computed from the signal width and gap width.

    Args:
        port: gdsfactory Port at the signal center
        layer: Target conductor layer name (e.g., 'topmetal2')
        s_width: Signal conductor width in um
        gap_width: Gap width between signal and ground in um
        length: Port extent along direction (um)
        impedance: Port impedance in Ohms (default: 50)
        excited: Whether port is excited (default: True)

    Examples:
        ```python
        configure_cpw_port(
            c.ports["o1"],
            layer="topmetal2",
            s_width=10.0,
            gap_width=6.0,
            length=5.0,
        )
        ```
    """
    import numpy as np

    center = np.array([float(port.center[0]), float(port.center[1])])
    orientation_rad = np.deg2rad(
        float(port.orientation) if port.orientation is not None else 0.0
    )

    # Transverse direction (perpendicular to port orientation, in-plane)
    # Port orientation points along the waveguide; transverse is 90° CCW
    transverse = np.array([-np.sin(orientation_rad), np.cos(orientation_rad)])

    # Gap center offset from signal center
    offset = (s_width + gap_width) / 2.0

    upper_center = center + transverse * offset
    lower_center = center - transverse * offset

    # Store computed CPW element info on the single port
    port.info["palace_type"] = "cpw"
    port.info["layer"] = layer
    port.info["length"] = length
    port.info["impedance"] = impedance
    port.info["excited"] = excited
    port.info["cpw_upper_center"] = (float(upper_center[0]), float(upper_center[1]))
    port.info["cpw_lower_center"] = (float(lower_center[0]), float(lower_center[1]))
    port.info["cpw_gap_width"] = gap_width


def extract_ports(component, stack: LayerStack) -> list[PalacePort]:
    """Extract Palace ports from a gdsfactory component.

    Handles all port types: inplane, via, and CPW (multi-element).

    Args:
        component: gdsfactory Component with configured ports
        stack: LayerStack from stack module

    Returns:
        List of PalacePort objects ready for simulation
    """
    palace_ports = []

    for port in component.ports:
        info = port.info
        palace_type = info.get("palace_type")

        if palace_type is None:
            continue

        if palace_type == "cpw":
            # Single-port CPW: gap centers were pre-computed by configure_cpw_port
            layer_name = info.get("layer")
            zmin, zmax = 0.0, 0.0
            if layer_name and layer_name in stack.layers:
                layer = stack.layers[layer_name]
                zmin = layer.zmin
                zmax = layer.zmax

            upper_center = info["cpw_upper_center"]
            lower_center = info["cpw_lower_center"]
            gap_width = info["cpw_gap_width"]

            centers = [
                (float(upper_center[0]), float(upper_center[1])),
                (float(lower_center[0]), float(lower_center[1])),
            ]
            # Upper element: E-field toward signal (negative transverse)
            # Lower element: E-field toward signal (positive transverse)
            directions = ["-Y", "+Y"]

            cpw_port = PalacePort(
                name=port.name,
                port_type=PortType.LUMPED,
                geometry=PortGeometry.INPLANE,
                center=(float(port.center[0]), float(port.center[1])),
                width=gap_width,
                orientation=float(port.orientation)
                if port.orientation is not None
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

    return palace_ports
