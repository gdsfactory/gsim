"""Simulation overlay metadata for 2D visualization.

Provides dataclasses that describe the simulation cell boundaries, PML regions,
port locations, and dielectric background layers for rendering on top of
geometry cross-sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gsim.common.geometry_model import GeometryModel
    from gsim.meep.models.config import DomainConfig, PortData


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
class DielectricOverlay:
    """Background dielectric slab metadata for 2D overlay rendering.

    Attributes:
        name: Dielectric region name (e.g. "oxide", "substrate").
        material: Material name (e.g. "SiO2", "silicon").
        zmin: Bottom z-coordinate of the slab in um.
        zmax: Top z-coordinate of the slab in um.
    """

    name: str
    material: str
    zmin: float
    zmax: float


@dataclass(frozen=True)
class SimOverlay:
    """Simulation cell metadata for 2D visualization overlays.

    Attributes:
        cell_min: (xmin, ymin, zmin) of the full simulation cell.
        cell_max: (xmax, ymax, zmax) of the full simulation cell.
        dpml: PML absorber thickness in um.
        ports: List of port overlays for rendering.
        dielectrics: List of background dielectric slabs for rendering.
    """

    cell_min: tuple[float, float, float]
    cell_max: tuple[float, float, float]
    dpml: float
    ports: list[PortOverlay] = field(default_factory=list)
    dielectrics: list[DielectricOverlay] = field(default_factory=list)


def build_sim_overlay(
    geometry_model: GeometryModel,
    domain_config: DomainConfig,
    port_data: list[PortData],
    z_span: float | None = None,
    dielectrics: list[dict[str, Any]] | None = None,
) -> SimOverlay:
    """Build a SimOverlay from geometry model, domain config, and port data.

    Args:
        geometry_model: The geometry model providing the geometry bbox.
        domain_config: Domain / PML configuration.
        port_data: List of PortData objects from port extraction.
        z_span: Port monitor z-span. If None, computed from geometry bbox.
        dielectrics: List of dielectric dicts from stack.dielectrics.

    Returns:
        SimOverlay with computed cell boundaries and port overlays.
    """
    gmin, gmax = geometry_model.bbox
    dpml = domain_config.dpml
    margin_xy = domain_config.margin_xy

    # XY: margin_xy is gap between geometry bbox and PML
    # Z: margin_z_above/below is already baked into the geometry bbox via set_z_crop(),
    #    so only add dpml beyond the geometry z-extent
    cell_min = (
        gmin[0] - margin_xy - dpml,
        gmin[1] - margin_xy - dpml,
        gmin[2] - dpml,
    )
    cell_max = (
        gmax[0] + margin_xy + dpml,
        gmax[1] + margin_xy + dpml,
        gmax[2] + dpml,
    )

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

    diel_overlays: list[DielectricOverlay] = []
    if dielectrics:
        for d in dielectrics:
            diel_overlays.append(
                DielectricOverlay(
                    name=d["name"],
                    material=d["material"],
                    zmin=d["zmin"],
                    zmax=d["zmax"],
                )
            )

    return SimOverlay(
        cell_min=cell_min,
        cell_max=cell_max,
        dpml=dpml,
        ports=ports,
        dielectrics=diel_overlays,
    )
