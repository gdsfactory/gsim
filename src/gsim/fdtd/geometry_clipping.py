"""Report material removed by explicit FDTD domain windows."""

from __future__ import annotations

import warnings
from typing import Any

from gsim.common.pdk import ResolvedPassivePcell
from gsim.fdtd.api import Domain, Geometry
from gsim.fdtd.models import GeometryClippingWarning


def warn_geometry_clipping(
    resolved: ResolvedPassivePcell,
    geometry: Geometry,
    domain: Domain,
    previous_key: tuple[Any, ...] | None,
) -> tuple[Any, ...] | None:
    """Warn once per component/window configuration and return its signature."""
    if not resolved.clipped_layers:
        return None
    key = (
        id(geometry.component),
        repr(geometry.settings),
        domain.x_bounds,
        domain.y_bounds,
        domain.z_bounds,
        tuple(resolved.clipped_layers.items()),
    )
    if key != previous_key:
        details = ", ".join(
            f"{name} ({'/'.join(axes)})"
            for name, axes in resolved.clipped_layers.items()
        )
        warnings.warn(
            "Explicit domain bounds trim or omit material geometry on "
            f"layers: {details}. Enlarge the bounds to retain this material.",
            GeometryClippingWarning,
            stacklevel=3,
        )
    return key
