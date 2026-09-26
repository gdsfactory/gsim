"""Shared helpers for Meep X/Y domain sizing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from gdsfactory.component import Component

    from gsim.meep.models.config import DomainConfig

Axis = Literal["x", "y"]
RequiredRange = tuple[float, float, str]


def resolve_inner_axis_interval(
    domain: DomainConfig,
    axis: Axis,
    bbox_low: float,
    bbox_high: float,
) -> tuple[float, float]:
    """Resolve one X/Y PML-inner interval from bounds or bbox margins."""
    bounds = getattr(domain, f"{axis}_bounds")
    if bounds is not None:
        return bounds
    margin_low = getattr(domain, f"margin_{axis}_low")
    margin_high = getattr(domain, f"margin_{axis}_high")
    return bbox_low - margin_low, bbox_high + margin_high


def automatic_port_extension_length(
    component: Component,
    domain: DomainConfig,
) -> float:
    """Return enough extension for every optical port to cross its outer PML.

    Unlike the historical ``max(margins) + dpml`` rule, this accounts for an
    explicit inner interval and for ports that are not on the raw component
    bbox because a fabrication marker expands it.
    """
    bbox = component.dbbox()
    x_low, x_high = resolve_inner_axis_interval(
        domain, "x", float(bbox.left), float(bbox.right)
    )
    y_low, y_high = resolve_inner_axis_interval(
        domain, "y", float(bbox.bottom), float(bbox.top)
    )
    inner_edges = {
        0: x_high,
        90: y_high,
        180: x_low,
        270: y_low,
    }

    required_lengths: list[float] = []
    for port in component.ports:
        if str(getattr(port, "port_type", "optical")) != "optical":
            continue
        orientation = round(float(port.orientation)) % 360
        if orientation not in inner_edges:
            continue
        x = float(port.center[0])
        y = float(port.center[1])
        if orientation == 0:
            inner_distance = inner_edges[orientation] - x
        elif orientation == 90:
            inner_distance = inner_edges[orientation] - y
        elif orientation == 180:
            inner_distance = x - inner_edges[orientation]
        else:
            inner_distance = y - inner_edges[orientation]
        required_lengths.append(max(0.0, inner_distance) + domain.dpml)

    if required_lengths:
        return max(required_lengths)

    return (
        max(
            domain.margin_x_low,
            domain.margin_x_high,
            domain.margin_y_low,
            domain.margin_y_high,
        )
        + domain.dpml
    )


def materialized_component_xy_bounds(
    component: Any,
) -> tuple[float, float, float, float] | None:
    """Return a non-empty materialized component bbox as X/Y bounds."""
    integer_bbox = component.kdb_cell.bbox()
    if integer_bbox.empty():
        return None
    bbox = component.dbbox()
    return (
        float(bbox.left),
        float(bbox.bottom),
        float(bbox.right),
        float(bbox.top),
    )


def _geometry_required_ranges(
    component: Any,
    stack: Any,
    plane: str | None,
    y_cut: float | None,
) -> dict[Axis, list[RequiredRange]]:
    """Return simulated geometry ranges for each active X/Y axis."""
    required: dict[Axis, list[RequiredRange]] = {"x": [], "y": []}
    if plane == "xz":
        from gsim.common.cross_section import extract_xz_rectangles

        rectangles = extract_xz_rectangles(
            component,
            stack,
            y_cut=0.0 if y_cut is None else y_cut,
        )
        if rectangles:
            required["x"].append(
                (
                    min(rectangle.x0 for rectangle in rectangles),
                    max(rectangle.x1 for rectangle in rectangles),
                    "simulated XZ geometry",
                )
            )
        return required

    geometry_bounds = materialized_component_xy_bounds(component)
    if geometry_bounds is None:
        return required
    x_low, y_low, x_high, y_high = geometry_bounds
    required["x"].append((x_low, x_high, "simulated physical geometry"))
    required["y"].append((y_low, y_high, "simulated physical geometry"))
    return required


def _add_port_required_ranges(
    required: dict[Axis, list[RequiredRange]],
    ports: list[Any],
    domain: DomainConfig,
) -> None:
    """Add port planes and shifted source/monitor centers in place."""
    for port in ports:
        center = [float(coordinate) for coordinate in port.center]
        required["x"].append((center[0], center[0], f"port {port.name!r}"))
        required["y"].append((center[1], center[1], f"port {port.name!r}"))

        transverse_axis = 1 - port.normal_axis
        half_width = (float(port.width) + 2 * domain.port_margin) / 2
        transverse_name: Axis = "x" if transverse_axis == 0 else "y"
        transverse_center = center[transverse_axis]
        required[transverse_name].append(
            (
                transverse_center - half_width,
                transverse_center + half_width,
                f"mode plane for port {port.name!r}",
            )
        )

        direction_sign = 1.0 if port.direction == "+" else -1.0
        normal_name: Axis = "x" if port.normal_axis == 0 else "y"
        monitor_offset = domain.source_port_offset
        if port.is_source:
            monitor_offset += domain.distance_source_to_monitors
        for offset in (domain.source_port_offset, monitor_offset):
            shifted = center[port.normal_axis] + direction_sign * offset
            required[normal_name].append(
                (shifted, shifted, f"source/monitor for port {port.name!r}")
            )


def validate_explicit_xy_bounds(
    component: Any,
    stack: Any,
    ports: list[Any],
    domain: DomainConfig,
    plane: str | None,
    y_cut: float | None,
    fiber_source: Any | None,
) -> None:
    """Require explicit X/Y windows to contain simulated geometry and I/O."""
    explicit_bounds: dict[Axis, tuple[float, float] | None] = {
        "x": domain.x_bounds,
        "y": domain.y_bounds,
    }
    required = _geometry_required_ranges(component, stack, plane, y_cut)
    _add_port_required_ranges(required, ports, domain)

    if fiber_source is not None and explicit_bounds["x"] is not None:
        x_low, x_high = explicit_bounds["x"]
        safety = max(domain.dpml * 0.1, 0.1)
        max_half = min(fiber_source.x - x_low, x_high - fiber_source.x) - safety
        if max_half <= 0.0:
            raise ValueError(
                f"Explicit domain.x_bounds={explicit_bounds['x']} leaves no "
                "positive source half-span for fiber source placement at "
                f"x={fiber_source.x} after safety={safety}. Move the fiber "
                "farther inside x_bounds or use x_bounds='auto'."
            )

    tolerance = 1e-9
    for axis, bounds in explicit_bounds.items():
        if bounds is None:
            continue
        bound_low, bound_high = bounds
        for required_low, required_high, description in required[axis]:
            if (
                required_low >= bound_low - tolerance
                and required_high <= bound_high + tolerance
            ):
                continue
            raise ValueError(
                f"Explicit domain.{axis}_bounds={bounds} does not contain "
                f"{description} extent ({required_low}, {required_high}). "
                f"Expand {axis}_bounds or use {axis}_bounds='auto'."
            )


__all__ = [
    "automatic_port_extension_length",
    "materialized_component_xy_bounds",
    "resolve_inner_axis_interval",
    "validate_explicit_xy_bounds",
]
