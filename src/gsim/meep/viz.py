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
        dielectrics=dielectrics,
        component_bbox=orig_bbox,
    )


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
    )
    return plot_prism_slices(gm, x, y, z, ax, legend, slices, overlay=overlay)


def plot_2d_interactive(
    component: Component,
    stack: LayerStack | None,
    domain_config: DomainConfig,
    source_port: str | None = None,
    slice_axis: str = "y",
    slice_value: float | str | None = None,
    extend_ports_length: float | None = None,
    port_data: list | None = None,
    component_bbox: list[float] | tuple[float, ...] | None = None,
) -> Any:
    """Interactive 2D cross-section using Plotly.

    Produces a zoomable/pannable figure with toggleable layers
    via the Plotly legend. Click a layer name to hide/show it.

    Args:
        component: gdsfactory Component.
        stack: gsim LayerStack (may be None).
        domain_config: Domain configuration.
        source_port: Source port name (for overlay).
        slice_axis: Which axis to slice at — "x", "y", or "z".
        slice_value: Coordinate or layer name for the slice. None = "core".
        extend_ports_length: Override port extension length (pass 0 if
            already extended).
        port_data: Pre-computed port data list.
        component_bbox: Original component bbox.

    Returns:
        ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

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
    )

    # Resolve slice coordinate
    sv = slice_value if slice_value is not None else "core"
    axis_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
    if isinstance(sv, str):
        sv = gm.get_layer_center(sv)[axis_idx]

    # Determine plot axes
    if slice_axis == "z":
        h_idx, v_idx = 0, 1
        h_label, v_label = "x (um)", "y (um)"
    elif slice_axis == "y":
        h_idx, v_idx = 0, 2
        h_label, v_label = "x (um)", "z (um)"
    else:  # x
        h_idx, v_idx = 1, 2
        h_label, v_label = "y (um)", "z (um)"

    import numpy as np

    # Layer colours
    layer_names = gm.layer_names
    cmap = plt.colormaps.get_cmap("tab10")
    colors_arr = cmap(np.linspace(0, 1, max(len(layer_names), 1)))
    layer_colors = dict(zip(layer_names, colors_arr, strict=True))

    fig = go.Figure()

    # Draw layer polygons
    for name in layer_names:
        bbox = gm.get_layer_bbox(name)

        if slice_axis == "z":
            if not (bbox[0][2] <= sv <= bbox[1][2]):
                continue
        elif slice_axis == "y":
            if not (bbox[0][1] <= sv <= bbox[1][1]):
                continue
        elif slice_axis == "x" and not (bbox[0][0] <= sv <= bbox[1][0]):
            continue

        rgba = layer_colors.get(name, (0.8, 0.8, 0.8, 0.5))
        fill_color = (
            f"rgba({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)},0.5)"
        )
        line_color = (
            f"rgba({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)},1)"
        )

        first_trace = True
        if name in gm.prisms:
            for prism in gm.prisms[name]:
                if slice_axis == "z" and not (prism.z_base <= sv <= prism.z_top):
                    continue

                if slice_axis == "z":
                    # XY polygon
                    xs = [v[0] for v in prism.vertices] + [prism.vertices[0][0]]
                    ys = [v[1] for v in prism.vertices] + [prism.vertices[0][1]]
                elif slice_axis == "y":
                    # XZ rectangle from prism x-extent and z-extent
                    vx = prism.vertices[:, 0]
                    xs = [vx.min(), vx.max(), vx.max(), vx.min(), vx.min()]
                    ys = [
                        prism.z_base,
                        prism.z_base,
                        prism.z_top,
                        prism.z_top,
                        prism.z_base,
                    ]
                else:  # x
                    vy = prism.vertices[:, 1]
                    xs = [vy.min(), vy.max(), vy.max(), vy.min(), vy.min()]
                    ys = [
                        prism.z_base,
                        prism.z_base,
                        prism.z_top,
                        prism.z_top,
                        prism.z_base,
                    ]

                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        fill="toself",
                        fillcolor=fill_color,
                        line=dict(color=line_color, width=0.5),
                        name=name,
                        legendgroup=name,
                        showlegend=first_trace,
                        hoverinfo="name",
                    )
                )
                first_trace = False
        else:
            # Fill layer — draw bbox rectangle
            x0, x1 = bbox[0][h_idx], bbox[1][h_idx]
            y0, y1 = bbox[0][v_idx], bbox[1][v_idx]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, x1, x0, x0],
                    y=[y0, y1, y1, y0, y0],
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=0.5),
                    name=name,
                    legendgroup=name,
                    showlegend=first_trace,
                    hoverinfo="name",
                )
            )

    # Draw overlay: PML and ports
    if overlay is not None:
        dpml = overlay.dpml
        cmin, cmax = overlay.cell_min, overlay.cell_max
        h0, h1 = cmin[h_idx], cmax[h_idx]
        v0, v1 = cmin[v_idx], cmax[v_idx]

        # Sim cell boundary
        fig.add_trace(
            go.Scatter(
                x=[h0, h1, h1, h0, h0],
                y=[v0, v1, v1, v0, v0],
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                name="MEEP domain",
                legendgroup="MEEP domain",
                hoverinfo="name",
            )
        )

        # PML regions
        pml_color = "rgba(200,200,200,0.3)"
        pml_rects = [
            (h0, v0, h0 + dpml, v1),  # left
            (h1 - dpml, v0, h1, v1),  # right
            (h0 + dpml, v0, h1 - dpml, v0 + dpml),  # bottom
            (h0 + dpml, v1 - dpml, h1 - dpml, v1),  # top
        ]
        first_pml = True
        for rx0, ry0, rx1, ry1 in pml_rects:
            fig.add_trace(
                go.Scatter(
                    x=[rx0, rx1, rx1, rx0, rx0],
                    y=[ry0, ry0, ry1, ry1, ry0],
                    mode="lines",
                    fill="toself",
                    fillcolor=pml_color,
                    line=dict(color="gray", width=0.5),
                    name="PML",
                    legendgroup="PML",
                    showlegend=first_pml,
                    hoverinfo="name",
                )
            )
            first_pml = False

        # Ports
        for port in overlay.ports:
            cx, cy, cz = port.center
            ph, pv = (
                (cx, cy)
                if slice_axis == "z"
                else ((cx, cz) if slice_axis == "y" else (cy, cz))
            )
            color = "red" if port.is_source else "blue"
            hw = port.width / 2
            hz = port.z_span / 2

            if port.normal_axis == 2 and slice_axis == "y":
                # z-normal: horizontal line
                fig.add_trace(
                    go.Scatter(
                        x=[ph - hw, ph + hw],
                        y=[pv, pv],
                        mode="lines+text",
                        line=dict(color=color, width=2),
                        text=[None, port.name],
                        textposition="top center",
                        name=f"{'Source' if port.is_source else 'Monitor'} {port.name}",
                        legendgroup="Ports",
                        hoverinfo="name",
                    )
                )
            elif port.normal_axis == 0 and slice_axis == "y":
                # x-normal: vertical line
                fig.add_trace(
                    go.Scatter(
                        x=[ph, ph],
                        y=[pv - hz, pv + hz],
                        mode="lines+text",
                        line=dict(color=color, width=2),
                        text=[None, port.name],
                        textposition="top center",
                        name=f"{'Source' if port.is_source else 'Monitor'} {port.name}",
                        legendgroup="Ports",
                        hoverinfo="name",
                    )
                )
            elif port.normal_axis == 1 and slice_axis == "z":
                # y-normal on XY: horizontal line
                fig.add_trace(
                    go.Scatter(
                        x=[ph - hw, ph + hw],
                        y=[pv, pv],
                        mode="lines+text",
                        line=dict(color=color, width=2),
                        text=[None, port.name],
                        textposition="top center",
                        name=f"{'Source' if port.is_source else 'Monitor'} {port.name}",
                        legendgroup="Ports",
                        hoverinfo="name",
                    )
                )
            elif port.normal_axis == 0 and slice_axis == "z":
                # x-normal on XY: vertical line
                fig.add_trace(
                    go.Scatter(
                        x=[ph, ph],
                        y=[pv - hw, pv + hw],
                        mode="lines+text",
                        line=dict(color=color, width=2),
                        text=[None, port.name],
                        textposition="top right",
                        name=f"{'Source' if port.is_source else 'Monitor'} {port.name}",
                        legendgroup="Ports",
                        hoverinfo="name",
                    )
                )

    fig.update_layout(
        xaxis_title=h_label,
        yaxis_title=v_label,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        hovermode="closest",
        margin=dict(l=60, r=20, t=40, b=60),
    )
    return fig
