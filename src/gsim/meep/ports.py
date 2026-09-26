"""Port extraction for MEEP simulation.

Extracts port information from a gdsfactory component into a
serializable format for the MEEP config JSON.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from gsim.meep.models.api import PortVerticalOverride
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
    port_margin: float = 0.0,
    port_overrides: Mapping[str, PortVerticalOverride] | None = None,
    y_cut: float | None = None,
) -> list[PortData]:
    """Extract port information from a gdsfactory component.

    Args:
        component: gdsfactory Component with ports
        layer_stack: LayerStack to determine z-coordinates
        source_port: Name of the source port. If None, first port is the source.
        is_3d: Whether Z is active. If False, all port z-centers are set to 0.
        port_margin: Extra mode-plane margin on each side of the physical port
            layer in Z. Ignored when Z is collapsed.
        port_overrides: Explicit per-port Z centers and spans. Each provided
            field takes precedence over automatic inference.
        y_cut: When provided for an XZ simulation, discard ports that do not
            intersect this Y coordinate before resolving their Z geometry.

    Returns:
        List of PortData objects ready for JSON serialization
    """
    ports: list[PortData] = []

    if not is_3d and port_overrides:
        raise ValueError("Port Z overrides require an active Z axis.")

    component_ports = list(component.ports)
    if y_cut is not None:
        component_ports = _filter_component_ports_for_xz(component_ports, y_cut)

    for i, gf_port in enumerate(component_ports):
        normal_axis, direction = get_port_normal(gf_port.orientation)

        is_source = gf_port.name == source_port if source_port is not None else i == 0
        port_name = gf_port.name or f"port{i}"
        if is_3d:
            z_center, z_span = _resolve_port_vertical_geometry(
                component,
                gf_port,
                layer_stack,
                port_margin=port_margin,
                override=(port_overrides or {}).get(port_name),
            )
        else:
            z_center = 0.0
            z_span = None

        ports.append(
            PortData(
                name=port_name,
                center=[
                    float(gf_port.center[0]),
                    float(gf_port.center[1]),
                    z_center,
                ],
                orientation=float(gf_port.orientation),
                width=float(gf_port.width),
                z_span=z_span,
                normal_axis=normal_axis,
                direction=direction,
                is_source=is_source,
            )
        )

    return ports


def _filter_component_ports_for_xz(
    ports: list[Any],
    y_cut: float,
) -> list[Any]:
    """Filter fabrication ports before XZ vertical-layer inference."""
    kept: list[Any] = []
    for port in ports:
        normal_axis, _ = get_port_normal(port.orientation)
        if normal_axis != 0:
            logger.warning(
                "Dropping port %r for XZ 2D sim (normal_axis=%d != 0)",
                port.name,
                normal_axis,
            )
            continue

        y_center = float(port.center[1])
        width = float(port.width)
        if abs(y_center - y_cut) > width / 2:
            logger.warning(
                "Dropping port %r for XZ 2D sim "
                "(center.y=%.4f, width=%.4f does not intersect y_cut=%.4f)",
                port.name,
                y_center,
                width,
                y_cut,
            )
            continue

        kept.append(port)

    return kept


def _port_gds_layer(gf_port: Any) -> tuple[int, int] | None:
    """Return the concrete fabrication tuple carried by a gdsfactory port."""
    layer_info = getattr(gf_port, "layer_info", None)
    if layer_info is None:
        try:
            layer_info = gf_port.kcl.get_info(gf_port.layer)
        except (AttributeError, TypeError, ValueError):
            return None
    try:
        return int(layer_info.layer), int(layer_info.datatype)
    except (AttributeError, TypeError, ValueError):
        return None


def _vertical_candidates(
    layers: list[Any],
    *,
    port_margin: float,
) -> tuple[float | None, float | None, set[tuple[float, float]]]:
    """Return independently unambiguous center/span values for layers."""
    extents = {(float(layer.zmin), float(layer.zmax)) for layer in layers}
    centers = {(zmin + zmax) / 2.0 for zmin, zmax in extents}
    spans = {(zmax - zmin) + 2 * port_margin for zmin, zmax in extents}
    z_center = next(iter(centers)) if len(centers) == 1 else None
    z_span = next(iter(spans)) if len(spans) == 1 else None
    if z_span is not None and z_span <= 0:
        z_span = None
    return z_center, z_span, extents


def _resolve_port_vertical_geometry(
    component: Component,
    gf_port: Any,
    layer_stack: LayerStack,
    *,
    port_margin: float,
    override: PortVerticalOverride | None,
) -> tuple[float, float]:
    """Resolve one port's source/monitor center and height.

    Port tuples refer to fabrication masks. Call this before physical-layer
    materialization remaps stack layers to simulation-only GDS tuples.
    """
    gds_layer = _port_gds_layer(gf_port)
    candidates = [
        layer
        for layer in layer_stack.layers.values()
        if gds_layer is not None and tuple(layer.gds_layer) == gds_layer
    ]
    resolution_source = f"port layer {gds_layer}"

    if not candidates:
        # If the component really draws the port mask but the active/cropped
        # stack does not contain it, an override cannot restore that missing
        # geometry. This commonly indicates that z_bounds excluded the layer.
        if gds_layer is not None and _component_draws_layer(component, gds_layer):
            raise ValueError(
                f"Port {gf_port.name!r} uses drawn layer {gds_layer}, but that "
                "layer is absent from the active simulation stack. Expand "
                "domain.z_bounds or include the layer in the stack."
            )

        candidates = [
            layer
            for layer in layer_stack.layers.values()
            if _component_draws_layer(component, tuple(layer.gds_layer))
        ]
        resolution_source = "drawn simulation layers"

    inferred_z, inferred_z_span, extents = _vertical_candidates(
        candidates,
        port_margin=port_margin,
    )
    if (
        not any(tuple(layer.gds_layer) == gds_layer for layer in candidates)
        and len(extents) == 1
    ):
        logger.warning(
            "Port %r layer %r is absent from the stack; using the unambiguous "
            "drawn-layer Z extent %s.",
            gf_port.name,
            gds_layer,
            next(iter(extents)),
        )

    resolved_z = (
        override.z if override is not None and override.z is not None else inferred_z
    )
    resolved_z_span = (
        override.z_span
        if override is not None and override.z_span is not None
        else inferred_z_span
    )
    missing_fields = [
        name
        for name, value in (("z", resolved_z), ("z_span", resolved_z_span))
        if value is None
    ]
    if missing_fields:
        candidate_text = (
            ", ".join(f"[{zmin:g}, {zmax:g}]" for zmin, zmax in sorted(extents))
            or "none"
        )
        raise ValueError(
            f"Could not infer {', '.join(missing_fields)} for port "
            f"{gf_port.name!r} from {resolution_source}; candidate Z extents: "
            f"{candidate_text}. Set sim.port_overrides[{gf_port.name!r}] with "
            f"the missing field(s)."
        )

    return cast(float, resolved_z), cast(float, resolved_z_span)


def _highest_n_among(layers: Any) -> tuple[Any, float]:
    """Highest-refractive-index layer among an iterable of Layer objects.

    Args:
        layers: Iterable of layer objects (each with ``.material``).

    Returns:
        (best_layer, best_n) -- (None, 0.0) if no optical data.
    """
    from gsim.common.stack.materials import get_material_properties

    best_layer = None
    best_eps = 0.0

    for layer in layers:
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


def _find_highest_n_layer(layer_stack: LayerStack) -> tuple[Any, float]:
    """Find the layer with the highest refractive index.

    Args:
        layer_stack: LayerStack from gsim.common

    Returns:
        (best_layer, best_n) -- (None, 0.0) if no optical data.
    """
    return _highest_n_among(layer_stack.layers.values())


def _component_draws_layer(component: Component, gds_layer: tuple[int, int]) -> bool:
    """Whether the component has any polygon on the given GDS layer tuple.

    Args:
        component: gdsfactory Component to inspect.
        gds_layer: GDS ``(layer, datatype)`` tuple.

    Returns:
        True if at least one polygon exists on that layer.
    """
    # merge=False so this works on locked @cell components (merge=True
    # mutates and raises LockedError on cached cells).
    raw = component.get_polygons(layers=(tuple(gds_layer),), merge=False)
    if not isinstance(raw, dict) or not raw:
        return False
    return any(
        (value if isinstance(value, list) else [value]) for value in raw.values()
    )


def _find_highest_n_layer_in_component(
    component: Component,
    layer_stack: LayerStack,
) -> tuple[Any, float]:
    """Highest-n stack layer among those the component actually draws.

    This is the auto vertical-crop reference: the photonic core the
    component defines (e.g. the Si ``core`` layer for a grating coupler),
    not undrawn high-index layers (e.g. Ge) or the full BEOL stack.

    Args:
        component: gdsfactory Component with drawn geometry.
        layer_stack: LayerStack from gsim.common.

    Returns:
        (best_layer, best_n). Falls back to the global highest-n layer with
        a warning if no drawn layer has optical data.
    """
    candidates = [
        layer
        for layer in layer_stack.layers.values()
        if _component_draws_layer(component, layer.gds_layer)
    ]
    best_layer, best_n = _highest_n_among(candidates)
    if best_layer is not None:
        return best_layer, best_n

    logger.warning(
        "No drawn layer with optical data found for auto z-crop reference; "
        "falling back to the global highest-index layer."
    )
    return _find_highest_n_layer(layer_stack)


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
