"""Visualization helpers for MEEP simulations.

Standalone functions that build geometry models and render 2D/3D plots.
Called directly by ``Simulation.plot_2d()`` / ``plot_3d()`` — no legacy
``MeepSim`` dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from gdsfactory.component import Component

    from gsim.common import LayerStack
    from gsim.common.geometry_model import GeometryModel
    from gsim.meep.models.config import DomainConfig


# ---------------------------------------------------------------------------
# Geometry model construction
# ---------------------------------------------------------------------------


def build_geometry_model(
    component: Component,
    stack: LayerStack | None,
    domain_config: DomainConfig,
    extend_ports_length: float | None = None,
) -> GeometryModel:
    """Build a GeometryModel from a component + stack for visualization.

    Uses the gdsfactory LayerStack from the active PDK (not the gsim
    LayerStack), because ``LayeredComponentBase`` needs gdsfactory's
    ``LayerStack`` for polygon extraction via ``DerivedLayer.get_shapes()``.

    When ``domain_config.extend_ports`` is configured, the waveguide
    ports are extended into the PML region so the visualization matches
    what the runner will simulate.

    Args:
        component: gdsfactory Component to visualize.
        stack: gsim LayerStack (used for z-crop clipping). May be None.
        domain_config: Domain configuration for port extension length.
        extend_ports_length: Override port extension length. Pass ``0``
            when the component has already been extended by
            :meth:`Simulation.build_config`. ``None`` (default) computes
            the length from *domain_config* as before.

    Returns:
        GeometryModel ready for visualization.

    Raises:
        ValueError: If no active PDK with a layer_stack.
    """
    import gdsfactory as gf

    from gsim.common.geometry_model import extract_geometry_model
    from gsim.common.layered_component import LayeredComponentBase

    pdk = gf.get_active_pdk()
    gf_layer_stack = pdk.layer_stack
    if gf_layer_stack is None:
        raise ValueError(
            "Active PDK has no layer_stack. Activate a PDK with a layer stack first."
        )

    # Compute port extension length
    if extend_ports_length is not None:
        extend_length = extend_ports_length
    else:
        # Default: same logic as build_config
        extend_length = domain_config.extend_ports
        if extend_length == 0.0:
            extend_length = domain_config.margin_xy + domain_config.dpml

    lc = LayeredComponentBase(
        component=component,
        layer_stack=gf_layer_stack,
        extend_ports=extend_length,
    )
    gm = extract_geometry_model(lc)

    # If the stack has been z-cropped, clip the geometry model to match
    if stack is not None:
        gm = crop_geometry_model(gm, stack)

    return gm


def crop_geometry_model(
    gm: GeometryModel,
    stack: LayerStack,
) -> GeometryModel:
    """Clip a GeometryModel's prisms and bbox to the stack z-range.

    Args:
        gm: Source geometry model.
        stack: LayerStack whose z-extent defines the crop window.

    Returns:
        New GeometryModel clipped to the stack z-range.
    """
    from gsim.common.geometry_model import GeometryModel, Prism

    # Use both layers and dielectrics to determine the full z-range.
    # PDKs like cspdk don't define box/clad as explicit layers, so relying
    # on layers alone would shrink the bbox to just the patterned geometry
    # (e.g. 0-0.22 um), causing PML to sit on the core.
    z_vals: list[float] = []
    for layer in stack.layers.values():
        z_vals.extend((layer.zmin, layer.zmax))
    for diel in stack.dielectrics:
        z_vals.extend((diel["zmin"], diel["zmax"]))
    z_lo = min(z_vals)
    z_hi = max(z_vals)

    cropped_prisms: dict[str, list[Prism]] = {}
    for layer_name, prism_list in gm.prisms.items():
        clipped = []
        for p in prism_list:
            if p.z_top <= z_lo or p.z_base >= z_hi:
                continue
            clipped.append(
                Prism(
                    vertices=p.vertices,
                    z_base=max(p.z_base, z_lo),
                    z_top=min(p.z_top, z_hi),
                    layer_name=p.layer_name,
                    material=p.material,
                    sidewall_angle=p.sidewall_angle,
                    original_polygon=p.original_polygon,
                )
            )
        if clipped:
            cropped_prisms[layer_name] = clipped

    # Recompute bbox
    old_min, old_max = gm.bbox
    new_bbox = (
        (old_min[0], old_min[1], z_lo),
        (old_max[0], old_max[1], z_hi),
    )

    return GeometryModel(
        prisms=cropped_prisms,
        bbox=new_bbox,
        layer_bboxes=gm.layer_bboxes,
        layer_mesh_orders=gm.layer_mesh_orders,
    )


# ---------------------------------------------------------------------------
# Overlay construction
# ---------------------------------------------------------------------------


def build_overlay(
    geometry_model: GeometryModel,
    component: Component,
    stack: LayerStack | None,
    domain_config: DomainConfig,
    source_port: str | None = None,
    port_data: list | None = None,
    component_bbox: list[float] | tuple[float, ...] | None = None,
    fiber_source: Any = None,
    monitor_z_span: float | None = None,
) -> Any:
    """Build a SimOverlay from config, if stack is available.

    Args:
        geometry_model: The geometry model (for bbox).
        component: Component (for bbox and port extraction fallback).
        stack: gsim LayerStack (needed for dielectrics). May be None.
        domain_config: Domain configuration.
        source_port: Source port name (or None for auto).
        port_data: Pre-computed port data from :meth:`Simulation.build_config`.
            When provided, skips port extraction (avoids duplicate work).
        component_bbox: Original component bbox ``[xmin, ymin, xmax, ymax]``
            from :meth:`Simulation.build_config`. When provided, cell
            boundaries are computed from this instead of ``component.dbbox()``.

    Returns:
        SimOverlay or None if stack isn't configured.
    """
    from gsim.meep.overlay import build_sim_overlay

    if stack is None:
        return None

    # Use pre-computed port data or extract from component
    if port_data is None:
        try:
            from gsim.meep.ports import extract_port_info

            comp_copy = component.copy()
            port_data = extract_port_info(comp_copy, stack, source_port=source_port)
        except Exception:
            port_data = []

    # Use pre-computed component bbox or extract from component
    if component_bbox is not None:
        orig_bbox = (
            component_bbox[0],
            component_bbox[1],
            component_bbox[2],
            component_bbox[3],
        )
    else:
        bbox = component.dbbox()
        orig_bbox = (bbox.left, bbox.bottom, bbox.right, bbox.top)

    dielectrics = stack.dielectrics if stack else []
    return build_sim_overlay(
        geometry_model,
        domain_config,
        port_data,
        z_span=monitor_z_span,
        dielectrics=dielectrics,
        component_bbox=orig_bbox,
        fiber_source=fiber_source,
    )


# ---------------------------------------------------------------------------
# Cross-section geometry from gsim LayerStack (no PDK dependency)
# ---------------------------------------------------------------------------


def build_cross_section_rectangles(
    component: Component,
    stack: LayerStack,
    slice_axis: str,
    slice_coord: float,
) -> list[dict[str, Any]]:
    """Build rectangle outlines for a cross-section of a gsim LayerStack.

    Extracts GDS polygons from *component* for each layer in *stack*,
    intersects them with a horizontal or vertical slice line, and returns
    rectangle data suitable for ``ax.add_patch(Rectangle(...))``.

    Unlike :func:`build_geometry_model`, this function does **not** rely on
    the active PDK's layer stack — it uses the *gsim* :class:`LayerStack`
    directly so cross-section geometry for custom material stacks
    (e.g. TFLN) is rendered correctly.

    Layers that have no GDS polygons at all (``_layer_has_any_polygon``
    returns ``False``) are skipped, since their boundaries are already
    implicit from adjacent layers.

    Args:
        component: gdsfactory Component with GDS polygons.
        stack: gsim :class:`LayerStack` defining z-extents and materials.
        slice_axis: ``"y"`` for an XZ cross-section (horizontal slice),
            ``"x"`` for a YZ cross-section (vertical slice).
        slice_coord: Coordinate of the slice line (µm).

    Returns:
        List of dicts with keys ``h_min``, ``h_max``, ``z_min``, ``z_max``,
        ``layer_name``, ``material``.
    """
    import numpy as np
    from shapely.geometry import LineString, MultiLineString, Polygon

    if not stack.layers:
        return []

    def _has_any_polygon(layer: object) -> bool:
        gds = getattr(layer, "gds_layer", None)
        if gds is None:
            return False
        try:
            polys = component.get_polygons_points(layers=(gds,), merge=True)
        except Exception:
            return False
        else:
            if not isinstance(polys, dict):
                return False
            for v in polys.values():
                items = v if isinstance(v, list) else [v]
                if len(items) > 0:
                    return True
            return False

    if slice_axis == "y":
        cut_line = LineString([(-1e6, slice_coord), (1e6, slice_coord)])
    elif slice_axis == "x":
        cut_line = LineString([(slice_coord, -1e6), (slice_coord, 1e6)])
    else:
        raise ValueError(f"slice_axis must be 'x' or 'y', got {slice_axis!r}")

    rectangles: list[dict[str, Any]] = []
    for layer in stack.layers.values():
        if layer.material == "air":
            continue
        gds_layer = layer.gds_layer
        if gds_layer is None:
            continue
        thickness = layer.zmax - layer.zmin
        if thickness <= 0:
            continue

        try:
            polys = component.get_polygons_points(layers=(gds_layer,), merge=True)
        except Exception:
            try:
                polys = component.dup().get_polygons_points(
                    layers=(gds_layer,), merge=True
                )
            except Exception:
                polys = {}

        poly_items: list = []
        if isinstance(polys, dict):
            for poly_list in polys.values():
                items = poly_list if isinstance(poly_list, list) else [poly_list]
                poly_items.extend(items)

        if not poly_items:
            if _has_any_polygon(layer):
                continue
            rectangles.append(
                {
                    "h_min": -float("inf"),
                    "h_max": float("inf"),
                    "z_min": layer.zmin,
                    "z_max": layer.zmax,
                    "layer_name": layer.name,
                    "material": layer.material,
                }
            )
            continue

        for poly in poly_items:
            if isinstance(poly, np.ndarray):
                coords = poly.reshape(-1, 2)
                if len(coords) >= 3:
                    spoly = Polygon(coords)
                else:
                    continue
            elif isinstance(poly, (tuple, list)):
                spoly = Polygon(poly)
            elif hasattr(poly, "to_simple_polygon"):
                sp = poly.to_simple_polygon()
                coords = [
                    (sp.point(i).x, sp.point(i).y) for i in range(sp.num_points())
                ]
                if len(coords) >= 3:
                    spoly = Polygon(coords)
                else:
                    continue
            else:
                continue

            if spoly.is_empty or not spoly.is_valid:
                continue

            intersection = spoly.intersection(cut_line)
            if intersection.is_empty:
                continue

            segments: list[LineString] = []
            if isinstance(intersection, LineString):
                segments.append(intersection)
            elif isinstance(intersection, MultiLineString):
                segments.extend(intersection.geoms)
            else:
                segments.extend(
                    g
                    for g in getattr(intersection, "geoms", [])
                    if isinstance(g, LineString)
                )

            for seg in segments:
                coords = list(seg.coords)
                if len(coords) < 2:
                    continue
                if slice_axis == "y":
                    h_vals = [c[0] for c in coords]
                else:
                    h_vals = [c[1] for c in coords]
                h_min, h_max = min(h_vals), max(h_vals)
                if h_max - h_min < 1e-9:
                    continue
                rectangles.append(
                    {
                        "h_min": h_min,
                        "h_max": h_max,
                        "z_min": layer.zmin,
                        "z_max": layer.zmax,
                        "layer_name": layer.name,
                        "material": layer.material,
                    }
                )

    return rectangles


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def plot_3d(
    component: Component,
    stack: LayerStack | None,
    domain_config: DomainConfig,
    backend: str = "open3d",
    extend_ports_length: float | None = None,
    **kwargs: Any,
) -> Any:
    """Create interactive 3D visualization of MEEP geometry.

    Args:
        component: gdsfactory Component to visualize.
        stack: gsim LayerStack (may be None).
        domain_config: Domain configuration.
        backend: "open3d" (Jupyter/VS Code) or "pyvista" (desktop).
        extend_ports_length: Override port extension length (pass 0 if
            the component is already extended).
        **kwargs: Extra args forwarded to the backend renderer.

    Returns:
        Renderer-specific widget or plotter object.
    """
    from gsim.common.viz import plot_prisms_3d, plot_prisms_3d_open3d

    gm = build_geometry_model(
        component, stack, domain_config, extend_ports_length=extend_ports_length
    )
    if backend == "pyvista":
        return plot_prisms_3d(gm, **kwargs)
    if backend == "open3d":
        return plot_prisms_3d_open3d(gm, **kwargs)
    raise ValueError(f"Unsupported backend: {backend}. Use 'open3d' or 'pyvista'")


def plot_2d_interactive(
    component: Component,
    stack: LayerStack | None,
    domain_config: DomainConfig,
    source_port: str | None = None,
    x: float | str | None = None,
    y: float | str | None = None,
    z: float | str | None = None,
    slices: str = "z",
    extend_ports_length: float | None = None,
    port_data: list | None = None,
    component_bbox: list[float] | tuple[float, ...] | None = None,
    fiber_source: Any = None,
    monitor_z_span: float | None = None,
) -> Any:
    """Plot an interactive 2D cross-section using Plotly.

    Each layer and overlay element is a separate trace, so users can zoom,
    pan, and toggle individual layers/materials on and off via the legend.

    Accepts the same geometry arguments as :func:`plot_2d`, except ``ax``
    and ``legend`` (Plotly manages its own legend).

    Only a single slice direction is supported per call.

    Args:
        component: gdsfactory Component to visualize.
        stack: gsim LayerStack (may be None).
        domain_config: Domain configuration.
        source_port: Source port name (for overlay).
        x: X-coordinate or layer name for slice plane.
        y: Y-coordinate or layer name for slice plane.
        z: Z-coordinate or layer name for slice plane.
        slices: Slice direction — "x", "y", or "z".
        extend_ports_length: Override port extension length (pass 0 if
            the component is already extended).
        port_data: Pre-computed port data (skips re-extraction).
        component_bbox: Original component bbox ``[xmin, ymin, xmax, ymax]``
            (for correct cell boundary computation with extended ports).
        fiber_source: Pre-computed fiber source config (for overlay).
        monitor_z_span: Port monitor z-span override.

    Returns:
        ``plotly.graph_objects.Figure``.
    """
    from gsim.common.viz import plot_prism_slices_interactive

    slices_to_plot = sorted(set(slices.lower()))
    if len(slices_to_plot) != 1:
        raise ValueError(
            f"plot_2d_interactive supports exactly one slice direction. Got: {slices!r}"
        )

    gm = build_geometry_model(
        component, stack, domain_config, extend_ports_length=extend_ports_length
    )
    overlay = build_overlay(
        gm,
        component,
        stack,
        domain_config,
        source_port,
        port_data=port_data,
        component_bbox=component_bbox,
        fiber_source=fiber_source,
        monitor_z_span=monitor_z_span,
    )

    slice_dir = slices_to_plot[0]
    kw: dict[str, Any] = {}
    if slice_dir == "x":
        kw["x"] = x if x is not None else "core"
    elif slice_dir == "y":
        kw["y"] = y if y is not None else "core"
    else:
        kw["z"] = z if z is not None else "core"

    return plot_prism_slices_interactive(gm, overlay=overlay, **kw)


def plot_2d(
    component: Component,
    stack: LayerStack | None,
    domain_config: DomainConfig,
    source_port: str | None = None,
    x: float | str | None = None,
    y: float | str | None = None,
    z: float | str = "core",
    ax: plt.Axes | None = None,
    legend: bool = True,
    slices: str = "z",
    extend_ports_length: float | None = None,
    port_data: list | None = None,
    component_bbox: list[float] | tuple[float, ...] | None = None,
    fiber_source: Any = None,
    monitor_z_span: float | None = None,
) -> plt.Axes | None:
    """Plot 2D cross-sections of the MEEP geometry.

    Args:
        component: gdsfactory Component to visualize.
        stack: gsim LayerStack (may be None).
        domain_config: Domain configuration.
        source_port: Source port name (for overlay).
        x: X-coordinate or layer name for slice plane.
        y: Y-coordinate or layer name for slice plane.
        z: Z-coordinate or layer name for slice plane.
        ax: Axes to draw on. If None, a new figure is created.
        legend: Whether to show the legend.
        slices: Slice direction(s) — "x", "y", "z", or combinations.
        extend_ports_length: Override port extension length (pass 0 if
            the component is already extended).
        port_data: Pre-computed port data (skips re-extraction).
        component_bbox: Original component bbox ``[xmin, ymin, xmax, ymax]``
            (for correct cell boundary computation with extended ports).

    Returns:
        ``plt.Axes`` when *ax* was provided, otherwise ``None``.
    """
    from gsim.common.viz import plot_prism_slices

    gm = build_geometry_model(
        component, stack, domain_config, extend_ports_length=extend_ports_length
    )
    overlay = build_overlay(
        gm,
        component,
        stack,
        domain_config,
        source_port,
        port_data=port_data,
        component_bbox=component_bbox,
        fiber_source=fiber_source,
        monitor_z_span=monitor_z_span,
    )
    return plot_prism_slices(gm, x, y, z, ax, legend, slices, overlay=overlay)
