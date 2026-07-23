"""Mesh generator for Palace EM simulation."""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import gmsh

from . import gmsh_utils
from .config_generator import (
    collect_mesh_stats,
    generate_palace_config,
    write_config,
)
from .geometry import (
    GeometryData,
    add_dielectrics,
    add_metals,
    add_patterned_dielectrics,
    add_pec_blocks,
    add_ports,
    build_entities,
    extract_geometry,
    resolve_dielectric_regions,
    resolve_mesh_domain_bounds,
)
from .groups import assign_physical_groups

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.models import (
        BoundaryModeConfig,
        CrossSectionPlaneConfig,
        DrivenConfig,
        EigenmodeConfig,
        NumericalConfig,
    )
    from gsim.palace.models.pec import PECBlockConfig
    from gsim.palace.ports.config import PalacePort

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for domain-boundary filtering
# ---------------------------------------------------------------------------


def _collect_pec_surface_lines(groups: dict) -> list[int]:
    """Collect boundary curves from PEC surfaces for mesh refinement.

    When planar conductors are embedded in dielectric volumes, the boolean
    pipeline may merge the conductor's boundary curves into the volume edges.
    This helper queries the live gmsh model for the boundary curves of each
    PEC surface and returns those that are valid dim-1 entities.

    Requires an active gmsh session.  Silently returns an empty list when
    gmsh has not been initialized (e.g. during unit tests with mocked geometry).
    """
    if not gmsh.isInitialized():
        return []
    lines: list[int] = []
    for surface_info in groups.get("pec_surfaces", {}).values():
        for stag in surface_info.get("tags", []):
            try:
                boundary = gmsh.model.getBoundary(
                    [(2, stag)], combined=False, oriented=False, recursive=False
                )
                for bdim, btag in boundary:
                    if bdim == 1:
                        try:
                            gmsh.model.getBoundingBox(1, btag)
                            lines.append(btag)
                        except Exception:
                            pass
            except Exception:
                pass
    return lines


def _get_domain_bbox(tol: float = 1e-3) -> tuple[float, float, float, float]:
    """Return the simulation domain XY bounding box from gmsh.

    Uses the overall bounding box of all entities; any volume that spans
    the full domain (airbox, vacuum, substrate) determines the extents.

    If gmsh has not been initialized (e.g. unit tests), falls back to a
    zero-sized box without emitting gmsh error noise.
    """
    if not gmsh.isInitialized():
        return (0.0, 0.0, 0.0, 0.0)
    try:
        xmin, ymin, _zmin, xmax, ymax, _zmax = gmsh.model.getBoundingBox(-1, -1)
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)
    return (xmin - tol, ymin - tol, xmax + tol, ymax + tol)


def _line_on_domain_boundary(
    line_tag: int,
    domain_bbox: tuple[float, float, float, float],
    tol: float = 1.0,
) -> bool:
    """Return True when a curve lies on the XY domain boundary.

    A curve is considered a domain-boundary edge when both of its
    endpoints sit on the same domain wall (x=xmin, x=xmax, y=ymin,
    or y=ymax).  This filters out large planar ground-plane sheets whose
    outer perimeter is just the simulation box outline.

    Silently returns False when gmsh has not been initialized (e.g.
    during unit tests with mocked geometry).
    """
    if not gmsh.isInitialized():
        return False
    try:
        tmin, tmax = gmsh.model.getParametrizationBounds(1, line_tag)
        p1 = gmsh.model.getValue(1, line_tag, tmin)
        p2 = gmsh.model.getValue(1, line_tag, tmax)
    except Exception:
        return False

    # Guard against empty coordinates (e.g. unit-test stubs)
    if len(p1) < 2 or len(p2) < 2:
        return False

    xmin, ymin, xmax, ymax = domain_bbox
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    on_xmin = abs(x1 - xmin) < tol and abs(x2 - xmin) < tol
    on_xmax = abs(x1 - xmax) < tol and abs(x2 - xmax) < tol
    on_ymin = abs(y1 - ymin) < tol and abs(y2 - ymin) < tol
    on_ymax = abs(y1 - ymax) < tol and abs(y2 - ymax) < tol

    return on_xmin or on_xmax or on_ymin or on_ymax


@dataclass
class MeshResult:
    """Result from mesh generation."""

    mesh_path: Path
    config_path: Path | None = None
    port_info: list = field(default_factory=list)
    mesh_stats: dict = field(default_factory=dict)
    # Data needed for deferred config generation
    groups: dict = field(default_factory=dict)
    output_dir: Path | None = None
    model_name: str = "palace"
    fmax: float = 100e9
    periodic_axis: str | None = None


def _extract_native_boundarymode_rectangles(
    component,
    stack: LayerStack,
    cross_section: CrossSectionPlaneConfig,
) -> list[dict[str, float | str]]:
    """Return axis-mapped 2D rectangles from the solver-agnostic section API."""
    from gsim.common import cross_section as section_utils

    section = section_utils.extract_plane_section(
        component,
        stack,
        axis=cross_section.axis,
        value=cross_section.value,
    )

    rects: list[dict[str, float | str]] = []
    if cross_section.axis == "x":
        for rect in section:
            y0 = getattr(rect, "y0", None)
            y1 = getattr(rect, "y1", None)
            z0 = getattr(rect, "zmin", None)
            z1 = getattr(rect, "zmax", None)
            if y0 is None or y1 is None or z0 is None or z1 is None:
                continue
            y0 = float(y0)
            y1 = float(y1)
            z0 = float(z0)
            z1 = float(z1)
            if y1 <= y0 or z1 <= z0:
                continue
            rects.append(
                {
                    "layer_name": str(getattr(rect, "layer_name", "layer")),
                    "h0": y0,
                    "h1": y1,
                    "v0": z0,
                    "v1": z1,
                }
            )
    elif cross_section.axis == "y":
        for rect in section:
            x0 = getattr(rect, "x0", None)
            x1 = getattr(rect, "x1", None)
            z0 = getattr(rect, "zmin", None)
            z1 = getattr(rect, "zmax", None)
            if x0 is None or x1 is None or z0 is None or z1 is None:
                continue
            x0 = float(x0)
            x1 = float(x1)
            z0 = float(z0)
            z1 = float(z1)
            if x1 <= x0 or z1 <= z0:
                continue
            rects.append(
                {
                    "layer_name": str(getattr(rect, "layer_name", "layer")),
                    "h0": x0,
                    "h1": x1,
                    "v0": z0,
                    "v1": z1,
                }
            )
    else:
        raise ValueError(
            "Boundary mode native 2D currently supports only x/y cross sections."
        )

    return rects


def _curve_on_native_2d_domain_wall(
    curve_tag: int,
    hmin: float,
    vmin: float,
    hmax: float,
    vmax: float,
    tol: float = 1e-6,
) -> bool:
    """Return True when a curve lies on the outer 2D domain boundary."""
    try:
        xmin, ymin, _zmin, xmax, ymax, _zmax = gmsh.model.getBoundingBox(1, curve_tag)
    except Exception:
        return False

    on_hmin = abs(xmin - hmin) <= tol and abs(xmax - hmin) <= tol
    on_hmax = abs(xmin - hmax) <= tol and abs(xmax - hmax) <= tol
    on_vmin = abs(ymin - vmin) <= tol and abs(ymax - vmin) <= tol
    on_vmax = abs(ymin - vmax) <= tol and abs(ymax - vmax) <= tol
    return on_hmin or on_hmax or on_vmin or on_vmax


def _generate_native_boundarymode_groups(
    *,
    kernel,
    component,
    geometry: GeometryData,
    stack: LayerStack,
    cross_section: CrossSectionPlaneConfig,
    margin_x: float,
    margin_y: float,
    air_margin: float,
    airbox_margin_x: float | None,
    airbox_margin_y: float | None,
    airbox_z_above: float | None,
    airbox_z_below: float | None,
) -> dict:
    """Build a native 2D gmsh model and groups for BoundaryMode."""
    if cross_section.axis not in {"x", "y"}:
        raise ValueError(
            "Boundary mode native 2D currently supports only x/y cross sections."
        )

    section_rects = _extract_native_boundarymode_rectangles(
        component=component,
        stack=stack,
        cross_section=cross_section,
    )
    if not section_rects:
        raise ValueError(
            f"Cross section '{cross_section.spec}' does not intersect any stack layer "
            "regions."
        )

    bounds = resolve_mesh_domain_bounds(
        geometry,
        stack,
        margin_x=margin_x,
        margin_y=margin_y,
        air_margin=air_margin,
        airbox_margin_x=airbox_margin_x,
        airbox_margin_y=airbox_margin_y,
        airbox_z_above=airbox_z_above,
        airbox_z_below=airbox_z_below,
    )

    if cross_section.axis == "x":
        hmin, hmax = bounds[1], bounds[4]
    else:
        hmin, hmax = bounds[0], bounds[3]
    vmin, vmax = bounds[2], bounds[5]

    if hmax <= hmin or vmax <= vmin:
        raise ValueError("Native BoundaryMode 2D domain has invalid bounds.")

    outer_surface = kernel.addRectangle(hmin, vmin, 0.0, hmax - hmin, vmax - vmin)
    layer_inputs: list[tuple[str, int]] = []

    for rect in section_rects:
        h0 = float(rect["h0"])
        h1 = float(rect["h1"])
        v0 = float(rect["v0"])
        v1 = float(rect["v1"])
        if h1 <= h0 or v1 <= v0:
            continue
        stag = kernel.addRectangle(h0, v0, 0.0, h1 - h0, v1 - v0)
        layer_inputs.append((str(rect["layer_name"]), stag))

    # Reuse the 3D dielectric-region resolver so native 2D picks the same
    # background stack materials as the volumetric pipeline.
    for region in resolve_dielectric_regions(
        geometry,
        stack,
        margin_x,
        margin_y,
        air_margin,
        airbox_margin_x=airbox_margin_x,
        airbox_margin_y=airbox_margin_y,
        airbox_z_above=airbox_z_above,
        airbox_z_below=airbox_z_below,
    ):
        if region.material == "airbox":
            continue

        if cross_section.axis == "x":
            rh0 = float(region.ymin)
            rh1 = float(region.ymax)
        else:
            rh0 = float(region.xmin)
            rh1 = float(region.xmax)

        h0 = max(rh0, hmin)
        h1 = min(rh1, hmax)
        z0 = max(region.zmin, vmin)
        z1 = min(region.zmax, vmax)
        if h1 <= h0 or z1 <= z0:
            continue

        stag = kernel.addRectangle(h0, z0, 0.0, h1 - h0, z1 - z0)
        layer_inputs.append((region.material, stag))

    if not layer_inputs:
        raise ValueError(
            f"Cross section '{cross_section.spec}' produced no meshing rectangles."
        )

    _, out_map = kernel.fragment(
        [(2, outer_surface)],
        [(2, tag) for _, tag in layer_inputs],
    )
    kernel.synchronize()

    groups: dict[str, dict] = {
        "volumes": {},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
        "refinement_lines": {},
    }

    outer_parts = {tag for dim, tag in out_map[0] if dim == 2}
    layer_surfaces: dict[str, set[int]] = {}
    for idx, (layer_name, _tag) in enumerate(layer_inputs, start=1):
        pieces = {tag for dim, tag in out_map[idx] if dim == 2}
        if pieces:
            layer_surfaces.setdefault(layer_name, set()).update(pieces)

    # A fragment piece can be mapped to multiple parents. For native 2D
    # conductor-as-boundary behavior, conductor/via pieces must not remain
    # inside dielectric domain groups.
    metal_piece_tags: set[int] = set()
    for layer_name, pieces in layer_surfaces.items():
        layer = stack.layers.get(layer_name)
        if layer is not None and layer.layer_type in {"conductor", "via"}:
            metal_piece_tags.update(pieces)

    if metal_piece_tags:
        for layer_name, pieces in list(layer_surfaces.items()):
            layer = stack.layers.get(layer_name)
            is_metal_like = layer is not None and layer.layer_type in {
                "conductor",
                "via",
            }
            if is_metal_like:
                continue
            filtered = pieces - metal_piece_tags
            if filtered:
                layer_surfaces[layer_name] = filtered
            else:
                layer_surfaces.pop(layer_name, None)

    def _material_conductivity(layer_name: str) -> float | list[float] | None:
        layer = stack.layers.get(layer_name)
        if layer is None:
            return None
        mat_props = stack.materials.get(layer.material, {})
        if isinstance(mat_props, dict):
            return mat_props.get("conductivity")
        return getattr(mat_props, "conductivity", None)

    def _is_conductive(value: float | list[float] | None) -> bool:
        if isinstance(value, list):
            try:
                return any(float(v) > 0.0 for v in value)
            except (TypeError, ValueError):
                return False
        if isinstance(value, int | float):
            return float(value) > 0.0
        return False

    assigned: set[int] = set()

    for layer_name, surface_tags in sorted(layer_surfaces.items()):
        sorted_tags = sorted(surface_tags)
        layer = stack.layers.get(layer_name)
        is_metal_like = layer is not None and layer.layer_type in {"conductor", "via"}

        # Always consume the conductor interior from air assignment, but do
        # not export it as a 2D domain material.
        assigned.update(sorted_tags)

        if not is_metal_like:
            pg = gmsh.model.addPhysicalGroup(2, sorted_tags)
            gmsh.model.setPhysicalName(2, pg, layer_name)

            entry: dict[str, object] = {
                "phys_group": pg,
                "tags": sorted_tags,
            }
            # Only layer-backed dielectric polygons are shaped dielectrics.
            # Background dielectric slabs (e.g. sio2/sin material regions)
            # should be treated as regular material domains.
            if layer is not None and layer.layer_type == "dielectric":
                entry["is_shaped_dielectric"] = True
            if layer is not None and layer.layer_type == "via":
                entry["is_via"] = True
            groups["volumes"][layer_name] = entry
            continue

        pec_curves: set[int] = set()
        for stag in sorted_tags:
            try:
                boundaries = gmsh.model.getBoundary(
                    [(2, stag)],
                    combined=False,
                    oriented=False,
                    recursive=False,
                )
            except Exception:
                continue
            for dim, ctag in boundaries:
                if dim == 1:
                    pec_curves.add(ctag)

        if pec_curves:
            curve_tags = sorted(pec_curves)
            sigma = _material_conductivity(layer_name)
            if _is_conductive(sigma):
                cond_pg = gmsh.model.addPhysicalGroup(1, curve_tags)
                gmsh.model.setPhysicalName(1, cond_pg, f"{layer_name}_conductivity")
                groups["conductor_surfaces"][layer_name] = {
                    "phys_group": cond_pg,
                    "tags": curve_tags,
                }
            else:
                pec_pg = gmsh.model.addPhysicalGroup(1, curve_tags)
                gmsh.model.setPhysicalName(1, pec_pg, f"{layer_name}_pec")
                groups["pec_surfaces"][layer_name] = {
                    "phys_group": pec_pg,
                    "tags": curve_tags,
                }

    air_tags = sorted(outer_parts - assigned)
    if air_tags:
        air_pg = gmsh.model.addPhysicalGroup(2, air_tags)
        gmsh.model.setPhysicalName(2, air_pg, "air")
        groups["volumes"]["air"] = {"phys_group": air_pg, "tags": air_tags}

    refinement_curves: set[int] = set()
    for info in groups["conductor_surfaces"].values():
        refinement_curves.update(int(t) for t in info.get("tags", []))
    for info in groups["pec_surfaces"].values():
        refinement_curves.update(int(t) for t in info.get("tags", []))
    for vol_info in groups["volumes"].values():
        for stag in vol_info.get("tags", []):
            try:
                boundaries = gmsh.model.getBoundary(
                    [(2, int(stag))],
                    combined=False,
                    oriented=False,
                    recursive=False,
                )
            except Exception:
                continue
            for dim, ctag in boundaries:
                if dim != 1:
                    continue
                if _curve_on_native_2d_domain_wall(ctag, hmin, vmin, hmax, vmax):
                    continue
                refinement_curves.add(int(ctag))

    if refinement_curves:
        groups["refinement_lines"]["native_internal_curves"] = {
            "tags": sorted(refinement_curves)
        }

    outer_curves: set[int] = set()
    for stag in outer_parts:
        try:
            boundaries = gmsh.model.getBoundary(
                [(2, stag)],
                combined=False,
                oriented=False,
                recursive=False,
            )
        except Exception:
            continue
        for dim, ctag in boundaries:
            if dim != 1:
                continue
            if _curve_on_native_2d_domain_wall(ctag, hmin, vmin, hmax, vmax):
                outer_curves.add(ctag)

    if outer_curves:
        curve_tags = sorted(outer_curves)
        outer_pg = gmsh.model.addPhysicalGroup(1, curve_tags)
        gmsh.model.setPhysicalName(1, outer_pg, "absorbing")
        groups["boundary_surfaces"]["absorbing"] = {
            "phys_group": [outer_pg],
            "tags": curve_tags,
        }

    return groups


def _setup_mesh_fields(
    kernel,
    groups: dict,
    geometry: GeometryData,
    stack: LayerStack,
    refined_cellsize: float,
    max_cellsize: float,
) -> None:
    """Set up mesh refinement fields.

    Args:
        kernel: gmsh OCC kernel
        groups: Physical group information
        geometry: Extracted geometry data
        stack: LayerStack with material properties
        refined_cellsize: Fine mesh size near conductors (um)
        max_cellsize: Coarse mesh size in air/dielectric (um)
    """
    boundary_lines: list[int] = []
    conductor_line_count = 0
    port_line_count = 0
    pec_line_count = 0
    shaped_dielectric_count = 0
    dielectric_line_count = 0

    # Get the overall simulation domain bbox.  Dielectric boxes span the
    # full domain; the largest volume's XY extent defines the boundary.
    # This is used to skip boundary lines that sit on the domain edge
    # (e.g. a planar conductor that covers the entire top surface).
    domain_bbox = _get_domain_bbox()

    # Conductor-surface edges are always refined — the refined_cellsize only
    # takes effect where boundary curves drive the Threshold field, and metal
    # edges are the dominant field-concentration sites.
    for surface_info in groups["conductor_surfaces"].values():
        for tag in surface_info["tags"]:
            lines = gmsh_utils.get_boundary_lines(tag, kernel)
            boundary_lines.extend(lines)
            conductor_line_count += len(lines)

    # User-defined PEC conductor surfaces are always refined — they mark
    # narrow vertical stitches between ground planes at port boundaries,
    # which are field-concentration sites by construction.
    #
    # Skip PEC edges that lie on the domain boundary (e.g. a full-domain
    # ground plane sheet).  Those edges do not represent a conductor
    # feature; they are just the simulation box outline.
    for surface_info in groups["pec_surfaces"].values():
        for tag in surface_info["tags"]:
            lines = gmsh_utils.get_boundary_lines(tag, kernel)
            for ltag in lines:
                if _line_on_domain_boundary(ltag, domain_bbox):
                    continue
                boundary_lines.append(ltag)
                pec_line_count += 1

    # Explicit refinement lines for planar conductors — embedded 2D PEC
    # surfaces lose their boundary curves during boolean fragmentation,
    # so independent wire loops are added to drive fine mesh at the
    # conductor perimeter.
    for line_info in groups.get("refinement_lines", {}).values():
        for ltag in line_info.get("tags", []):
            try:
                if _line_on_domain_boundary(ltag, domain_bbox):
                    continue
            except Exception:
                continue
            boundary_lines.append(ltag)
            pec_line_count += 1

    # For planar conductors, the PEC surface boundary curves may have been
    # merged into volume edges by the boolean pipeline.  The fallback in
    # assign_physical_groups already populates refinement_lines by querying
    # the live model, so _collect_pec_surface_lines is only needed when
    # that fallback was skipped (e.g. in unit tests with mocked groups).
    if not groups.get("refinement_lines"):
        pec_surface_lines = _collect_pec_surface_lines(groups)
        for ltag in pec_surface_lines:
            if ltag not in boundary_lines:
                try:
                    if _line_on_domain_boundary(ltag, domain_bbox):
                        continue
                except Exception:
                    continue
                boundary_lines.append(ltag)
                pec_line_count += 1

    # Shaped-dielectric volume boundaries are refined — the permittivity
    # discontinuity at the core-cladding interface concentrates fields.
    for vol_info in groups.get("volumes", {}).values():
        if vol_info.get("is_shaped_dielectric"):
            for tag in vol_info.get("tags", []):
                try:
                    boundary = gmsh.model.getBoundary(
                        [(3, tag)], combined=False, oriented=False, recursive=False
                    )
                    for bdim, btag in boundary:
                        if bdim == 2:
                            lines = gmsh_utils.get_boundary_lines(btag, kernel)
                            boundary_lines.extend(lines)
                            shaped_dielectric_count += len(lines)
                except Exception:
                    pass

    # Port boundaries are always refined — drives S-parameter / impedance
    # accuracy regardless of preset.
    for surface_info in groups["port_surfaces"].values():
        if surface_info.get("type") == "cpw":
            for elem in surface_info["elements"]:
                for tag in elem["tags"]:
                    lines = gmsh_utils.get_boundary_lines(tag, kernel)
                    boundary_lines.extend(lines)
                    port_line_count += len(lines)
        else:
            for tag in surface_info["tags"]:
                lines = gmsh_utils.get_boundary_lines(tag, kernel)
                boundary_lines.extend(lines)
                port_line_count += len(lines)

    # Dielectric interfaces are also field-concentration regions for
    # photonic structures; refine high-permittivity dielectric boundaries
    # (for example, core/air interfaces) when they exist as volume groups.
    for volume_name, volume_info in groups.get("volumes", {}).items():
        if volume_info.get("is_via", False):
            continue

        layer = stack.layers.get(volume_name)
        material_name = layer.material if layer is not None else volume_name
        material_props = stack.materials.get(material_name, {})
        permittivity = material_props.get("permittivity", 1.0)
        if isinstance(permittivity, list):
            try:
                permittivity = max(float(v) for v in permittivity)
            except (TypeError, ValueError):
                permittivity = 1.0
        if not isinstance(permittivity, int | float) or permittivity <= 1.0:
            continue

        for vtag in volume_info.get("tags", []):
            try:
                boundaries = gmsh.model.getBoundary(
                    [(3, vtag)],
                    combined=False,
                    oriented=False,
                    recursive=False,
                )
            except Exception:
                logger.debug(
                    "Skipping dielectric boundary refinement for stale volume tag %s",
                    vtag,
                )
                continue

            for dim, stag in boundaries:
                if dim != 2:
                    continue

                # Skip exterior/domain-boundary surfaces — these are surfaces
                # that only belong to one volume (labelled "...__None" in the
                # boolean pipeline). Refining their edges would force fine mesh
                # at the simulation domain boundary, wasting elements where no
                # internal field concentration exists.
                pg_tags = gmsh.model.getPhysicalGroupsForEntity(dim, stag)
                is_exterior = False
                for pg_tag in pg_tags:
                    name = gmsh.model.getPhysicalName(dim, pg_tag)
                    if name and "__None" in name:
                        is_exterior = True
                        break
                if is_exterior:
                    continue

                # For dielectric interfaces that span the full domain (e.g.
                # sapphire__vacuum at z=500), only refine the *internal*
                # edges — skip lines that sit on the XY domain boundary.
                # The outer perimeter of a large interface is just the
                # simulation box outline and does not need fine mesh.
                lines = gmsh_utils.get_boundary_lines(stag, kernel)
                for ltag in lines:
                    if _line_on_domain_boundary(ltag, domain_bbox):
                        continue
                    boundary_lines.append(ltag)
                    dielectric_line_count += 1

    boundary_lines = sorted(set(boundary_lines))

    logger.info(
        "Mesh refinement: %d boundary lines "
        "(conductor=%d, port=%d, pec=%d, shaped_dielectric=%d, dielectric=%d)",
        len(boundary_lines),
        conductor_line_count,
        port_line_count,
        pec_line_count,
        shaped_dielectric_count,
        dielectric_line_count,
    )

    # Setup main refinement field
    field_ids = []
    if boundary_lines:
        field_id = gmsh_utils.setup_mesh_refinement(
            boundary_lines, refined_cellsize, max_cellsize
        )
        field_ids.append(field_id)

    # Add box refinement for dielectrics based on permittivity
    xmin, ymin, xmax, ymax = geometry.bbox
    field_counter = 10

    for dielectric in stack.dielectrics:
        material_name = dielectric["material"]
        material_props = stack.materials.get(material_name, {})
        permittivity = material_props.get("permittivity", 1.0)
        if isinstance(permittivity, list):
            try:
                permittivity = max(float(v) for v in permittivity)
            except (TypeError, ValueError):
                permittivity = 1.0

        if permittivity > 1:
            local_max = max_cellsize / math.sqrt(permittivity)
            gmsh_utils.setup_box_refinement(
                field_counter,
                xmin,
                ymin,
                dielectric["zmin"],
                xmax,
                ymax,
                dielectric["zmax"],
                local_max,
                max_cellsize,
            )
            field_ids.append(field_counter)
            field_counter += 1

    if field_ids:
        gmsh_utils.finalize_mesh_fields(field_ids)


def generate_mesh(
    component,
    stack: LayerStack,
    ports: list[PalacePort],
    output_dir: str | Path,
    model_name: str = "palace",
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 300.0,
    margin_x: float = 50.0,
    margin_y: float = 50.0,
    air_margin: float = 50.0,
    airbox_margin_x: float | None = None,
    airbox_margin_y: float | None = None,
    airbox_z_above: float | None = None,
    airbox_z_below: float | None = None,
    fmax: float = 100e9,
    show_gui: bool = False,
    simulation_type: str = "driven",
    driven_config: DrivenConfig | None = None,
    eigenmode_config: EigenmodeConfig | None = None,
    numerical_config: NumericalConfig | None = None,
    boundary_mode_config: BoundaryModeConfig | None = None,
    cross_section: CrossSectionPlaneConfig | None = None,
    write_config: bool = True,
    planar_conductors: bool = False,
    pec_blocks: list[PECBlockConfig] | None = None,
    absorbing_boundary: bool = True,
    periodic_axis: str | None = None,
    merge_via_distance: float = 2.0,
    curve_fit_mode: Literal["line", "spline", "bspline"] = "line",
    curve_fit_layers: list[str] | None = None,
    curve_fit_tolerance_um: float = 0.0,
    curve_fit_min_points: int = 8,
    curve_fit_corner_angle_deg: float = 45.0,
    high_order_elements: bool = False,
    high_order_order: int = 2,
    high_order_optimize: bool = True,
    verbosity: int = 3,
    decimate_tolerance: float | None = None,
) -> MeshResult:
    """Generate mesh for Palace EM simulation.

    Args:
        component: gdsfactory Component
        stack: LayerStack from palace-api
        ports: List of PalacePort objects (single and multi-element)
        output_dir: Directory for output files
        model_name: Base name for output files
        refined_mesh_size: Mesh size near conductors (um)
        max_mesh_size: Max mesh size in air/dielectric (um)
        margin_x: X-axis margin around design (um)
        margin_y: Y-axis margin around design (um)
        air_margin: Legacy isotropic airbox margin (um)
        airbox_margin_x: Extra x-margin for explicit airbox (um)
        airbox_margin_y: Extra y-margin for explicit airbox (um)
        airbox_z_above: Extra +z margin for explicit airbox (um)
        airbox_z_below: Extra -z margin for explicit airbox (um)
        fmax: Max frequency for config (Hz)
        show_gui: Show gmsh GUI during meshing
        simulation_type: Type of simulation (driven, eigenmode or electrostatics)
        driven_config: Optional DrivenConfig for frequency sweep settings
        eigenmode_config: Optional EigenmodeConfig for eigenmode problems
        numerical_config: Optional NumericalConfig for solver settings
        boundary_mode_config: Optional BoundaryModeConfig for 2D mode problems
        cross_section: Explicit x/y cross-section plane for native BoundaryMode
        write_config: Whether to write config.json (default True)
        pec_blocks: PEC configuration
        planar_conductors: If True, treat conductors as 2D PEC surfaces
        absorbing_boundary: If True, use absorbing boundary conditions on outer surfaces
        periodic_axis: ("x" or "y") for meshing constraints on opposite domain sides
        merge_via_distance: Max gap between vias to merge (um)
        curve_fit_mode: Patterned dielectric boundary mode: line/spline/bspline
        curve_fit_layers: Layer names where curve fitting is applied
        curve_fit_tolerance_um: Point merge tolerance before curve fitting
        curve_fit_min_points: Min contour points required for curve fitting
        curve_fit_corner_angle_deg: Turn-angle threshold used to identify
            sharp corners during curve fitting segmentation
        high_order_elements: Enable high-order geometric mesh elements
        high_order_order: Polynomial order for high-order elements
        high_order_optimize: Run gmsh high-order optimization after meshing
        decimate_tolerance: Relative tolerance for polygon decimation
            (None = no decimation; typical 0.001-0.01)
        verbosity: Sets gmsh verbosity level

    Returns:
        MeshResult with paths and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msh_path = output_dir / f"{model_name}.msh"

    # Extract geometry
    logger.info("Extracting geometry...")
    geometry = extract_geometry(component, stack, decimate_tolerance=decimate_tolerance)
    logger.info("  Polygons: %s", len(geometry.polygons))
    logger.info("  Bbox: %s", geometry.bbox)

    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbosity)

    if "palace_mesh" in gmsh.model.list():
        gmsh.model.setCurrent("palace_mesh")
        gmsh.model.remove()
    gmsh.model.add("palace_mesh")

    kernel = gmsh.model.occ
    config_path: Path | None = None
    port_info: list = []

    try:
        if simulation_type == "boundarymode":
            if ports:
                raise ValueError(
                    "Boundary mode uses cross_section-only native 2D meshing and "
                    "does not support configured ports."
                )
            if cross_section is None:
                raise ValueError(
                    "Boundary mode requires an explicit cross section. "
                    "Call set_cross_section('x=<value>') or "
                    "set_cross_section('y=<value>')."
                )

            logger.info("Building native 2D BoundaryMode geometry...")
            groups = _generate_native_boundarymode_groups(
                kernel=kernel,
                component=component,
                geometry=geometry,
                stack=stack,
                cross_section=cross_section,
                margin_x=margin_x,
                margin_y=margin_y,
                air_margin=air_margin,
                airbox_margin_x=airbox_margin_x,
                airbox_margin_y=airbox_margin_y,
                airbox_z_above=airbox_z_above,
                airbox_z_below=airbox_z_below,
            )

            refinement_lines = sorted(
                {
                    int(tag)
                    for info in groups.get("refinement_lines", {}).values()
                    for tag in info.get("tags", [])
                }
            )
            if refinement_lines:
                aggressive_size = max(refined_mesh_size * 0.5, 1e-4)
                field_id = gmsh_utils.setup_mesh_refinement(
                    refinement_lines,
                    aggressive_size,
                    max_mesh_size,
                    sampling=400,
                    dist_max=max_mesh_size * 0.5,
                )
                gmsh_utils.finalize_mesh_fields([field_id])
            else:
                gmsh.option.setNumber("Mesh.MeshSizeMin", refined_mesh_size)
                gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh_size)

            if show_gui:
                gmsh.fltk.run()

            if high_order_elements:
                gmsh.option.setNumber("Mesh.ElementOrder", high_order_order)
                gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
                gmsh.option.setNumber(
                    "Mesh.HighOrderOptimize", 1 if high_order_optimize else 0
                )

            logger.info("Generating native 2D BoundaryMode mesh...")
            gmsh.model.mesh.generate(2)

            if high_order_elements:
                gmsh.model.mesh.setOrder(high_order_order)
                if high_order_optimize:
                    with contextlib.suppress(Exception):
                        gmsh.model.mesh.optimize("HighOrder")

            mesh_stats = collect_mesh_stats()

            gmsh.option.setNumber("Mesh.Binary", 0)
            gmsh.option.setNumber("Mesh.SaveAll", 0)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(str(msh_path))

            if write_config:
                config_path = generate_palace_config(
                    groups,
                    [],
                    [],
                    stack,
                    output_dir,
                    model_name,
                    fmax,
                    simulation_type,
                    driven_config,
                    eigenmode_config,
                    numerical_config,
                    boundary_mode_config,
                    absorbing_boundary,
                    periodic_axis,
                )

            return MeshResult(
                mesh_path=msh_path,
                config_path=config_path,
                port_info=[],
                mesh_stats=mesh_stats,
                groups=groups,
                output_dir=output_dir,
                model_name=model_name,
                fmax=fmax,
                periodic_axis=periodic_axis,
            )

        periodic_info: dict[str, object] | None = None

        # Add geometry
        logger.info("Adding metals...")
        metal_tags = add_metals(
            kernel, geometry, stack, planar_conductors, merge_via_distance
        )

        # Add PEC blocks if configured
        pec_block_tags: dict = {}
        if pec_blocks:
            logger.info("Adding PEC blocks...")
            pec_block_tags = add_pec_blocks(kernel, component, pec_blocks, stack)

        logger.info("Adding ports...")
        domain_bounds = resolve_mesh_domain_bounds(
            geometry,
            stack,
            margin_x=margin_x,
            margin_y=margin_y,
            air_margin=air_margin,
            airbox_margin_x=airbox_margin_x,
            airbox_margin_y=airbox_margin_y,
            airbox_z_above=airbox_z_above,
            airbox_z_below=airbox_z_below,
        )
        domain_bbox = (
            geometry.bbox[0] - margin_x,
            geometry.bbox[1] - margin_y,
            geometry.bbox[2] + margin_x,
            geometry.bbox[3] + margin_y,
        )
        port_tags, port_info = add_ports(
            kernel,
            ports,
            stack,
            domain_bbox=domain_bbox,
            domain_bounds=domain_bounds,
        )

        logger.info("Adding dielectrics...")
        dielectric_tags = add_dielectrics(
            kernel,
            geometry,
            stack,
            margin_x,
            margin_y,
            air_margin,
            airbox_margin_x=airbox_margin_x,
            airbox_margin_y=airbox_margin_y,
            airbox_z_above=airbox_z_above,
            airbox_z_below=airbox_z_below,
        )

        logger.info("Adding patterned dielectric layers...")
        patterned_dielectric_tags = add_patterned_dielectrics(
            kernel,
            geometry,
            stack,
            curve_fit_mode=curve_fit_mode,
            curve_fit_layers=curve_fit_layers,
            curve_fit_tolerance_um=curve_fit_tolerance_um,
            curve_fit_min_points=curve_fit_min_points,
            curve_fit_corner_angle_deg=curve_fit_corner_angle_deg,
        )

        all_dielectric_tags = {
            name: list(tags) for name, tags in dielectric_tags.items()
        }
        for layer_name, vol_tags in patterned_dielectric_tags.items():
            all_dielectric_tags.setdefault(layer_name, []).extend(vol_tags)

        # Build entities and run boolean pipeline
        logger.info("Running boolean pipeline...")
        entities = build_entities(
            metal_tags,
            dielectric_tags,
            patterned_dielectric_tags,
            port_tags,
            port_info,
            pec_block_tags=pec_block_tags or None,
            stack=stack,
        )
        pg_map = gmsh_utils.run_boolean_pipeline(entities)

        if periodic_axis in {"x", "y"}:
            periodic_info = gmsh_utils.set_periodic_mesh(pg_map, periodic_axis)

        # Assign physical groups
        logger.info("Assigning physical groups...")
        groups = assign_physical_groups(
            kernel,
            metal_tags,
            all_dielectric_tags,
            port_tags,
            port_info,
            entities,
            pg_map,
            stack,
            pec_block_tags=pec_block_tags or None,
        )

        # After assign_physical_groups, refinement_lines may reference
        # curves that were merged during boolean. Refresh them from the
        # live model so _setup_mesh_fields sees only valid tags.
        for line_info in groups.get("refinement_lines", {}).values():
            valid_tags = []
            for ltag in line_info.get("tags", []):
                try:
                    gmsh.model.getBoundingBox(1, ltag)
                    valid_tags.append(ltag)
                except Exception:
                    pass
            line_info["tags"] = valid_tags
        # Prune empty entries so the downstream loop stays quiet.
        groups["refinement_lines"] = {
            k: v for k, v in groups.get("refinement_lines", {}).items() if v.get("tags")
        }

        if periodic_info:
            donor_surfaces = periodic_info.get("master_surfaces")
            receiver_surfaces = periodic_info.get("slave_surfaces")
            donor_phys_groups = periodic_info.get("donor_phys_groups")
            receiver_phys_groups = periodic_info.get("receiver_phys_groups")

            if isinstance(donor_surfaces, list) and isinstance(receiver_surfaces, list):
                donor_tags = [int(t) for t in donor_surfaces if isinstance(t, Integral)]
                receiver_tags = [
                    int(t) for t in receiver_surfaces if isinstance(t, Integral)
                ]
                donor_pgs = (
                    [int(t) for t in donor_phys_groups if isinstance(t, Integral)]
                    if isinstance(donor_phys_groups, list)
                    else []
                )
                receiver_pgs = (
                    [int(t) for t in receiver_phys_groups if isinstance(t, Integral)]
                    if isinstance(receiver_phys_groups, list)
                    else []
                )

                if not donor_tags or not receiver_tags:
                    logger.warning(
                        "Periodic side surfaces were not discovered for axis %s",
                        periodic_axis,
                    )
                else:
                    if donor_pgs and receiver_pgs:
                        groups["boundary_surfaces"]["periodic_donor"] = {
                            "phys_group": donor_pgs,
                            "tags": donor_tags,
                        }
                        groups["boundary_surfaces"]["periodic_receiver"] = {
                            "phys_group": receiver_pgs,
                            "tags": receiver_tags,
                        }
                    else:
                        logger.warning(
                            "Periodic donor/receiver physical groups "
                            "were not created for axis %s",
                            periodic_axis,
                        )

        # Setup mesh fields
        logger.info("Setting up mesh refinement...")
        _setup_mesh_fields(
            kernel,
            groups,
            geometry,
            stack,
            refined_mesh_size,
            max_mesh_size,
        )

        # Show GUI if requested
        if show_gui:
            gmsh.fltk.run()

        if high_order_elements:
            logger.info(
                "Enabling high-order elements (order=%d, optimize=%s)",
                high_order_order,
                high_order_optimize,
            )
            gmsh.option.setNumber("Mesh.ElementOrder", high_order_order)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
            gmsh.option.setNumber(
                "Mesh.HighOrderOptimize", 1 if high_order_optimize else 0
            )

        logger.info("Generating mesh...")
        gmsh.model.mesh.generate(3)

        if high_order_elements:
            gmsh.model.mesh.setOrder(high_order_order)
            if high_order_optimize:
                try:
                    gmsh.model.mesh.optimize("HighOrder")
                except Exception as exc:
                    logger.warning(
                        "High-order optimization failed, using unoptimized high-order "
                        "elements: %s",
                        exc,
                    )

        # Collect mesh statistics
        mesh_stats = collect_mesh_stats()

        # Save mesh
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_path))

        logger.info("Mesh saved: %s", msh_path)

        # Generate config if requested
        config_path = None
        if write_config:
            logger.info("Generating Palace config...")
            config_path = generate_palace_config(
                groups,
                ports,
                port_info,
                stack,
                output_dir,
                model_name,
                fmax,
                simulation_type,
                driven_config,
                eigenmode_config,
                numerical_config,
                boundary_mode_config,
                absorbing_boundary,
                periodic_axis,
            )

    finally:
        gmsh.clear()
        gmsh.finalize()

    # Build result (store groups for deferred config generation)
    result = MeshResult(
        mesh_path=msh_path,
        config_path=config_path,
        port_info=port_info,
        mesh_stats=mesh_stats,
        groups=groups,
        output_dir=output_dir,
        model_name=model_name,
        fmax=fmax,
        periodic_axis=periodic_axis,
    )

    return result


# Re-export write_config from config_generator for backward compatibility
__all__ = ["GeometryData", "MeshResult", "generate_mesh", "write_config"]
