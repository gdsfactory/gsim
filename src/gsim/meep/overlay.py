"""Simulation overlay metadata for 2D visualization.

Provides dataclasses that describe the simulation cell boundaries, PML regions,
and port locations for rendering on top of geometry cross-sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gsim.common.geometry_model import GeometryModel
    from gsim.meep.models.config import MarginConfig, PortData


@dataclass(frozen=True)
class PortOverlay:
    """Port location metadata for 2D overlay rendering.

    Attributes:
        name: Port name (e.g. "o1").
        center: (x, y, z) center of the port monitor.
        width: Transverse width of the port monitor in um.
        normal_axis: 0 for x-normal, 1 for y-normal.
        direction: "+" or "-" along the normal axis.
        is_source: Whether this port is the excitation source.
        z_span: Height of the port monitor in z.
    """

    name: str
    center: tuple[float, float, float]
    width: float
    normal_axis: int
    direction: str
    is_source: bool
    z_span: float


@dataclass(frozen=True)
class SimOverlay:
    """Simulation cell metadata for 2D visualization overlays.

    Attributes:
        cell_min: (xmin, ymin, zmin) of the full simulation cell.
        cell_max: (xmax, ymax, zmax) of the full simulation cell.
        pml_thickness: PML absorber thickness in um.
        ports: List of port overlays for rendering.
    """

    cell_min: tuple[float, float, float]
    cell_max: tuple[float, float, float]
    pml_thickness: float
    ports: list[PortOverlay] = field(default_factory=list)


def build_sim_overlay(
    geometry_model: GeometryModel,
    margin_config: MarginConfig,
    port_data: list[PortData],
    z_span: float | None = None,
) -> SimOverlay:
    """Build a SimOverlay from geometry model, margin config, and port data.

    Args:
        geometry_model: The geometry model providing the geometry bbox.
        margin_config: PML / margin configuration.
        port_data: List of PortData objects from port extraction.
        z_span: Port monitor z-span. If None, computed from geometry bbox.

    Returns:
        SimOverlay with computed cell boundaries and port overlays.
    """
    gmin, gmax = geometry_model.bbox
    pml = margin_config.pml_thickness
    mxy = margin_config.margin_xy
    mz = margin_config.margin_z
    pad_xy = pml + mxy
    pad_z = pml + mz

    cell_min = (gmin[0] - pad_xy, gmin[1] - pad_xy, gmin[2] - pad_z)
    cell_max = (gmax[0] + pad_xy, gmax[1] + pad_xy, gmax[2] + pad_z)

    if z_span is None:
        z_span = gmax[2] - gmin[2]

    ports: list[PortOverlay] = [
        PortOverlay(
            name=p.name,
            center=tuple(p.center),  # type: ignore[arg-type]
            width=p.width,
            normal_axis=p.normal_axis,
            direction=p.direction,
            is_source=p.is_source,
            z_span=z_span,
        )
        for p in port_data
    ]

    return SimOverlay(
        cell_min=cell_min,
        cell_max=cell_max,
        pml_thickness=pml,
        ports=ports,
    )
