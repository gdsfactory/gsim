"""Intersect requested plane-monitor windows with the physical FDTD domain."""

from __future__ import annotations

import warnings
from pathlib import Path

from gsim.fdtd.api import PlaneMonitor, Vector3
from gsim.fdtd.models import FDTDConfigError

_BOUNDS_TOLERANCE_NM = 1e-7
_GSIM_DIR = str(Path(__file__).parents[1])


def monitor_region_nm(
    monitor: PlaneMonitor,
    domain_bounds_nm: tuple[float, float, float, float, float, float],
    *,
    warn: bool = True,
) -> tuple[Vector3, Vector3]:
    """Crop in-plane extents, rejecting outside positions and disjoint windows.

    Requested settings remain unchanged. Tiny floating-point excursions are
    snapped to the domain boundary without warning; actual cropping emits one
    warning per monitor, with every changed axis expressed in micrometers.
    """
    normal_axis = "xyz".index(monitor.normal[-1])
    region_min = []
    region_max = []
    changes = []
    for axis, axis_name in enumerate("xyz"):
        lower_nm = monitor.center_um[axis] * 1000 - monitor.size_um[axis] * 500
        upper_nm = monitor.center_um[axis] * 1000 + monitor.size_um[axis] * 500
        domain_lower_nm = domain_bounds_nm[axis]
        domain_upper_nm = domain_bounds_nm[axis + 3]
        if axis == normal_axis and (
            lower_nm < domain_lower_nm - _BOUNDS_TOLERANCE_NM
            or upper_nm > domain_upper_nm + _BOUNDS_TOLERANCE_NM
        ):
            raise FDTDConfigError(
                f"Monitor {monitor.name!r} {axis_name}-position "
                f"{monitor.center_um[axis]:g} um lies outside the physical domain "
                f"[{domain_lower_nm / 1000:g}, {domain_upper_nm / 1000:g}] um. "
                "Move the plane or change the domain bounds."
            )
        cropped_lower_nm = min(max(lower_nm, domain_lower_nm), domain_upper_nm)
        cropped_upper_nm = max(min(upper_nm, domain_upper_nm), domain_lower_nm)
        disjoint = (
            upper_nm < domain_lower_nm - _BOUNDS_TOLERANCE_NM
            or lower_nm > domain_upper_nm + _BOUNDS_TOLERANCE_NM
        )
        collapsed = lower_nm < upper_nm and cropped_lower_nm >= cropped_upper_nm
        if axis != normal_axis and (disjoint or collapsed):
            raise FDTDConfigError(
                f"Monitor {monitor.name!r} has no area inside the physical domain: "
                f"{axis_name}-extent [{lower_nm / 1000:g}, {upper_nm / 1000:g}] um "
                f"does not overlap [{domain_lower_nm / 1000:g}, "
                f"{domain_upper_nm / 1000:g}] um without losing its extent."
            )
        region_min.append(cropped_lower_nm)
        region_max.append(cropped_upper_nm)
        if (
            abs(cropped_lower_nm - lower_nm) > _BOUNDS_TOLERANCE_NM
            or abs(cropped_upper_nm - upper_nm) > _BOUNDS_TOLERANCE_NM
        ):
            changes.append(
                f"{axis_name} extent [{lower_nm / 1000:g}, {upper_nm / 1000:g}] "
                f"-> [{cropped_lower_nm / 1000:g}, {cropped_upper_nm / 1000:g}] um"
            )
    if changes and warn:
        aperture_note = (
            " Cropping changes the measurement aperture for flux or fiber overlap."
            if monitor.flux or monitor.fiber_mode is not None
            else ""
        )
        warnings.warn(
            f"Monitor {monitor.name!r} cropped to the physical (non-PML) domain: "
            + "; ".join(changes)
            + "."
            + aperture_note,
            UserWarning,
            skip_file_prefixes=(_GSIM_DIR,),
        )
    return (
        (region_min[0], region_min[1], region_min[2]),
        (region_max[0], region_max[1], region_max[2]),
    )
