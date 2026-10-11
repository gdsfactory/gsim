"""Physical domain bounds and directional padding for GDSFactory FDTD."""

from __future__ import annotations

from math import isfinite

from gsim.common.pdk import ResolvedPassivePcell
from gsim.fdtd.mesh_geometry import UM_TO_NM
from gsim.fdtd.models import FDTDGeometryError

Padding = float | tuple[float, float]


def padding_pair(value: Padding) -> tuple[float, float]:
    """Normalize finite, nonnegative padding to its negative/positive sides."""
    sides = value if isinstance(value, tuple) else (value, value)
    if any(not isfinite(side) or side < 0 for side in sides):
        raise ValueError("padding must be finite and nonnegative")
    return sides


def background_bounds_nm(
    resolved: ResolvedPassivePcell,
    background_material: str,
    padding: Padding,
    *,
    padding_x: Padding | None = None,
    padding_y: Padding | None = None,
    padding_z: Padding | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Size the physical domain, keeping automatic guided-port alignment.

    Explicit axis padding expands the resolved geometry bounds. Automatic Z
    sizing follows the PDK background stack when present. Padding is in um;
    the returned lower/upper coordinates are in nm.
    """
    lower, upper = resolved.bounds
    guided_ports = [port for port in resolved.ports.values() if not port.is_vertical]
    port_axes = {
        next(index for index, value in enumerate(port.normal) if value)
        for port in guided_ports
    }
    axis_padding = (padding_x, padding_y, padding_z)
    explicit_bounds = (x_bounds, y_bounds, z_bounds)
    try:
        fallback = padding_pair(padding)
        padding_sides = [
            padding_pair(value) if value is not None else fallback
            for value in axis_padding
        ]
    except ValueError as error:
        raise FDTDGeometryError(str(error)) from error

    for axis in (0, 1):
        if axis_padding[axis] is None and axis in port_axes:
            padding_sides[axis] = (0.0, 0.0)

    for port in guided_ports:
        axis = next(index for index, value in enumerate(port.normal) if value)
        side = 0 if port.normal[axis] < 0 else 1
        if axis_padding[axis] is not None and padding_sides[axis][side] > 0:
            axis_name = "xyz"[axis]
            direction = "-" if side == 0 else "+"
            raise FDTDGeometryError(
                f"domain.padding_{axis_name} cannot pad the {direction}{axis_name} "
                f"side containing guided port {port.name!r}; waveguide extensions "
                "are not supported. Set that side's padding to 0."
            )

    final_bounds = [
        *(lower[axis] - padding_sides[axis][0] for axis in range(3)),
        *(upper[axis] + padding_sides[axis][1] for axis in range(3)),
    ]
    if padding_z is None and z_bounds is None:
        background_z_bounds = []
        for level in resolved.layer_stack.layers.values():
            if level.material != background_material or level.thickness == 0:
                continue
            z_start = float(level.zmin)
            z_end = float(level.zmin + level.thickness)
            background_z_bounds.append((min(z_start, z_end), max(z_start, z_end)))
        if background_z_bounds:
            final_bounds[2] = min(lower[2], *(b[0] for b in background_z_bounds))
            final_bounds[5] = max(upper[2], *(b[1] for b in background_z_bounds))

    for axis, (axis_name, requested_bounds) in enumerate(
        zip(("x", "y", "z"), explicit_bounds, strict=True)
    ):
        if requested_bounds is None:
            continue
        if axis_padding[axis] is not None:
            raise FDTDGeometryError(
                f"Choose domain.{axis_name}_bounds or domain.padding_{axis_name}; "
                "they cannot be combined on the same axis."
            )
        requested_lower, requested_upper = requested_bounds
        if not isfinite(requested_lower) or not isfinite(requested_upper):
            raise FDTDGeometryError(f"domain.{axis_name}_bounds must be finite.")
        if requested_lower >= requested_upper:
            raise FDTDGeometryError(
                f"domain.{axis_name}_bounds lower bound must be smaller than "
                "its upper bound."
            )
        if requested_lower > lower[axis] or requested_upper < upper[axis]:
            raise FDTDGeometryError(
                f"domain.{axis_name}_bounds {requested_bounds} must contain "
                f"the geometry bounds ({lower[axis]}, {upper[axis]})."
            )
        final_bounds[axis] = requested_lower
        final_bounds[axis + 3] = requested_upper

    return (
        final_bounds[0] * UM_TO_NM,
        final_bounds[1] * UM_TO_NM,
        final_bounds[2] * UM_TO_NM,
        final_bounds[3] * UM_TO_NM,
        final_bounds[4] * UM_TO_NM,
        final_bounds[5] * UM_TO_NM,
    )
