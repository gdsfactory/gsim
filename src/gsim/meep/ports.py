"""Port extraction for MEEP simulation.

Extracts port information from a gdsfactory component into a
serializable format for the MEEP config JSON.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Literal

from gsim.meep.models.config import PortData

if TYPE_CHECKING:
    from gdsfactory.component import Component

    from gsim.common import LayerStack

logger = logging.getLogger(__name__)


def get_port_normal(orientation: float) -> tuple[int, Literal["+", "-"]]:
    """Get port normal axis and direction from orientation angle.

    Args:
        orientation: Port orientation in degrees (0, 90, 180, 270)

    Returns:
        Tuple of (axis_index, direction) where axis_index is 0=x or 1=y

    Raises:
        ValueError: If orientation is not a standard angle
    """
    ort = round(orientation) % 360
    if ort == 0:
        return 0, "-"
    if ort == 90:
        return 1, "-"
    if ort == 180:
        return 0, "+"
    if ort == 270:
        return 1, "+"
    raise ValueError(f"Invalid port orientation: {orientation}")


def extract_port_info(
    component: Component,
    layer_stack: LayerStack,
    source_port: str | None = None,
    *,
    is_3d: bool = True,
) -> list[PortData]:
    """Extract port information from a gdsfactory component.

    Args:
        component: gdsfactory Component with ports
        layer_stack: LayerStack to determine z-coordinates
        source_port: Name of the source port. If None, first port is the source.
        is_3d: If False, all port z-centers are set to 0 (2D mode).

    Returns:
        List of PortData objects ready for JSON serialization
    """
    ports: list[PortData] = []

    z_center = _get_z_center(layer_stack) if is_3d else 0.0

    for i, gf_port in enumerate(component.ports):
        normal_axis, direction = get_port_normal(gf_port.orientation)

        is_source = gf_port.name == source_port if source_port is not None else i == 0

        ports.append(
            PortData(
                name=gf_port.name or f"port{i}",
                center=[
                    float(gf_port.center[0]),
                    float(gf_port.center[1]),
                    z_center,
                ],
                orientation=float(gf_port.orientation),
                width=float(gf_port.width),
                normal_axis=normal_axis,
                direction=direction,
                is_source=is_source,
            )
        )

    return ports


def _find_highest_n_layer(layer_stack: LayerStack) -> tuple[Any, float]:
    """Find the layer with the highest refractive index.

    Args:
        layer_stack: LayerStack from gsim.common

    Returns:
        (best_layer, best_n) -- (None, 0.0) if no optical data.
    """
    from gsim.common.stack.materials import get_material_properties

    best_layer = None
    best_eps = 0.0

    for layer in layer_stack.layers.values():
        props = get_material_properties(layer.material)
        if (
            props is not None
            and props.permittivity is not None
            and not isinstance(props.permittivity, list)
            and props.permittivity > best_eps
        ):
            best_eps = props.permittivity
            best_layer = layer

    return best_layer, math.sqrt(best_eps) if best_eps > 0 else 0.0


def filter_ports_for_xz(
    ports: list[PortData],
    y_cut: float,
) -> list[PortData]:
    """Return ports compatible with an XZ 2D cross-section at ``Y=y_cut``.

    Keeps an X-facing port (``normal_axis=0``) only when its Y span
    ``[center.y - width/2, center.y + width/2]`` contains ``y_cut``.

    Drops:
      - Y-facing ports (not meaningful in an XZ 2D cell).
      - X-facing ports whose mode slice misses the cut entirely.

    Emits a ``logger.warning`` for each dropped port.

    Args:
        ports: List of PortData objects.
        y_cut: Y coordinate of the XZ cross-section (um).

    Returns:
        List of PortData that intersect the cross-section.
    """
    kept: list[PortData] = []
    for p in ports:
        if p.normal_axis != 0:
            logger.warning(
                "Dropping port %r for XZ 2D sim (normal_axis=%d != 0)",
                p.name,
                p.normal_axis,
            )
            continue

        y_center = p.center[1]
        if abs(y_center - y_cut) > p.width / 2:
            logger.warning(
                "Dropping port %r for XZ 2D sim "
                "(center.y=%.4f, width=%.4f does not intersect y_cut=%.4f)",
                p.name,
                y_center,
                p.width,
                y_cut,
            )
            continue

        kept.append(p)

    return kept


def _get_z_center(layer_stack: LayerStack) -> float:
    """Get z-center for ports from the layer stack.

    For photonic simulation, uses the midpoint of the layer with the
    highest refractive index (waveguide core). Falls back to the
    midpoint of all layers if no optical data is available.

    Args:
        layer_stack: LayerStack from gsim.common

    Returns:
        z-center coordinate in um
    """
    best_layer, best_n = _find_highest_n_layer(layer_stack)

    if best_layer is not None and best_n > 1.5:
        return (best_layer.zmin + best_layer.zmax) / 2.0

    # Fall back to midpoint of all layers
    if not layer_stack.layers:
        return 0.0
    all_zmin = min(l.zmin for l in layer_stack.layers.values())
    all_zmax = max(l.zmax for l in layer_stack.layers.values())
    return (all_zmin + all_zmax) / 2.0
