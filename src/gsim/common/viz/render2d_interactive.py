"""Interactive 2D cross-sectional plotting for GeometryModel using Plotly.

Provides a Plotly-based alternative to the matplotlib renderer in ``render2d.py``.
Each layer and overlay element is a separate trace, so users can zoom, pan,
and toggle individual layers/materials on and off via the legend.
"""

from __future__ import annotations

from typing import Any

from gsim.common.geometry_model import GeometryModel

# Plotly categorical colours (matches tab10 ordering)
_PLOTLY_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

_PML_FILL = "rgba(255, 153, 0, 0.20)"
_PML_LINE = "rgba(255, 153, 0, 0.50)"
_SRC_COLOR = "red"
_MON_COLOR = "royalblue"

_DIELECTRIC_COLORS: dict[str, str] = {
    "sio2": "rgba(173, 216, 230, 0.25)",
    "silicon": "rgba(153, 153, 166, 0.30)",
    "si": "rgba(153, 153, 166, 0.30)",
    "air": "rgba(255, 255, 255, 0.08)",
}
_DIELECTRIC_DEFAULT = "rgba(204, 204, 204, 0.20)"


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex colour to an rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def _diel_color(material: str) -> str:
    """Return the RGBA fill colour for a dielectric material name."""
    return _DIELECTRIC_COLORS.get(material.lower(), _DIELECTRIC_DEFAULT)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_prism_slices_interactive(
    geometry_model: GeometryModel,
    x: float | str | None = None,
    y: float | str | None = None,
    z: float | str | None = None,
    *,
    overlay: Any | None = None,
) -> Any:
    """Plot an interactive 2D cross-section using Plotly.

    Each layer is a separate trace that can be toggled via the legend.
    Supports zoom, pan, and hover inspection out of the box.

    Args:
        geometry_model: GeometryModel with prisms and bbox.
        x: X-coordinate (or layer name) for the slice plane.
        y: Y-coordinate (or layer name) for the slice plane.
        z: Z-coordinate (or layer name) for the slice plane.
        overlay: Optional SimOverlay with sim cell / PML / port metadata.

    Returns:
        ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    x_resolved, y_resolved, z_resolved = (
        geometry_model.get_layer_center(c)[i] if isinstance(c, str) else c
        for i, c in enumerate((x, y, z))
    )

    active_count = sum(
        [x_resolved is not None, y_resolved is not None, z_resolved is not None]
    )
    if active_count != 1:
        raise ValueError("Specify exactly one of x, y, or z for the slice plane")

    layer_names = geometry_model.layer_names
    color_map = {
        name: _PLOTLY_COLORS[i % len(_PLOTLY_COLORS)]
        for i, name in enumerate(layer_names)
    }

    def _layer_area(name: str) -> float:
        if name not in geometry_model.prisms:
            return float("inf")
        return sum(
            p.original_polygon.area
            for p in geometry_model.prisms[name]
            if p.original_polygon is not None
        )

    layer_areas = {nm: _layer_area(nm) for nm in layer_names}
    max_area = max(layer_areas.values()) if layer_areas else 1.0

    sorted_names = sorted(
        layer_names,
        key=lambda nm: (
            geometry_model.layer_mesh_orders.get(nm, 0),
            -layer_areas.get(nm, 0.0),
        ),
    )

    alpha_map = {}
    for nm in layer_names:
        ratio = layer_areas.get(nm, max_area) / max_area if max_area > 0 else 1.0
        alpha_map[nm] = 0.90 - 0.55 * ratio

    fig = go.Figure()

    # Dielectric backgrounds
    if overlay is not None:
        dielectrics = getattr(overlay, "dielectrics", [])
        if dielectrics:
            if z_resolved is not None:
                _add_dielectrics_xy(
                    fig, dielectrics, z_resolved, overlay.cell_min, overlay.cell_max
                )
            elif x_resolved is not None:
                _add_dielectrics_side(
                    fig, dielectrics, overlay.cell_min[1], overlay.cell_max[1]
                )
            elif y_resolved is not None:
                _add_dielectrics_side(
                    fig, dielectrics, overlay.cell_min[0], overlay.cell_max[0]
                )

    # Draw geometry layers
    seen_layers: set[str] = set()
    if z_resolved is not None:
        for name in sorted_names:
            bbox = geometry_model.get_layer_bbox(name)
            if not (bbox[0][2] <= z_resolved <= bbox[1][2]):
                continue
            if name in geometry_model.prisms:
                for prism in geometry_model.prisms[name]:
                    if not (prism.z_base <= z_resolved <= prism.z_top):
                        continue
                    show_legend = name not in seen_layers
                    seen_layers.add(name)
                    _add_polygon_trace(
                        fig,
                        prism.vertices.tolist(),
                        color_map.get(name, "#999"),
                        alpha_map.get(name, 0.7),
                        name,
                        show_legend,
                        prism.material,
                    )

    elif x_resolved is not None:
        for name in sorted_names:
            _draw_cross_sections(
                fig,
                geometry_model,
                name,
                "x",
                x_resolved,
                color_map.get(name, "#999"),
                alpha_map.get(name, 0.7),
                seen_layers,
            )

    elif y_resolved is not None:
        for name in sorted_names:
            _draw_cross_sections(
                fig,
                geometry_model,
                name,
                "y",
                y_resolved,
                color_map.get(name, "#999"),
                alpha_map.get(name, 0.7),
                seen_layers,
            )

    # Overlay: sim cell, PML, ports
    if overlay is not None:
        if z_resolved is not None:
            _add_overlay_xy(fig, overlay)
        elif x_resolved is not None:
            _add_overlay_yz(fig, overlay, x_resolved)
        elif y_resolved is not None:
            _add_overlay_xz(fig, overlay, y_resolved)

    # Axis labels and title
    if z_resolved is not None:
        xlabel, ylabel = "x (um)", "y (um)"
        title = f"XY cross section at z={z_resolved:.2f}"
    elif x_resolved is not None:
        xlabel, ylabel = "y (um)", "z (um)"
        title = f"YZ cross section at x={x_resolved:.2f}"
    else:
        xlabel, ylabel = "x (um)", "z (um)"
        title = f"XZ cross section at y={y_resolved:.2f}"

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain"),
        legend=dict(
            title="Layers / Materials",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Polygon trace helper
# ---------------------------------------------------------------------------


def _add_polygon_trace(
    fig: Any,
    vertices: list[list[float]],
    color: str,
    alpha: float,
    name: str,
    show_legend: bool,
    material: str = "",
) -> None:
    """Add a filled polygon as a Scatter trace."""
    import plotly.graph_objects as go

    xs = [v[0] for v in vertices] + [vertices[0][0]]
    ys = [v[1] for v in vertices] + [vertices[0][1]]

    hover = f"{name}"
    if material:
        hover += f" ({material})"

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            fill="toself",
            fillcolor=_rgba(color, alpha),
            line=dict(color="black", width=0.5),
            name=name,
            legendgroup=name,
            showlegend=show_legend,
            hoverinfo="text",
            text=hover,
        )
    )


def _add_rect_trace(
    fig: Any,
    x0: float,
    y0: float,
    w: float,
    h: float,
    fillcolor: str,
    line_color: str,
    name: str,
    show_legend: bool,
    legendgroup: str | None = None,
    hover: str = "",
) -> None:
    """Add a rectangle as a Scatter trace."""
    import plotly.graph_objects as go

    xs = [x0, x0 + w, x0 + w, x0, x0]
    ys = [y0, y0, y0 + h, y0 + h, y0]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            fill="toself",
            fillcolor=fillcolor,
            line=dict(color=line_color, width=0.5),
            name=name,
            legendgroup=legendgroup or name,
            showlegend=show_legend,
            hoverinfo="text",
            text=hover or name,
        )
    )


# ---------------------------------------------------------------------------
# Cross-section helpers (x/y slices)
# ---------------------------------------------------------------------------


def _draw_cross_sections(
    fig: Any,
    geometry_model: GeometryModel,
    layer_name: str,
    slice_axis: str,
    slice_coord: float,
    color: str,
    alpha: float,
    seen: set[str],
) -> None:
    """Intersect prism polygons with a slice plane, draw rectangles."""
    from shapely.geometry import LineString, MultiLineString
    from shapely.geometry import Polygon as ShapelyPolygon

    if layer_name not in geometry_model.prisms:
        return

    prisms = geometry_model.prisms[layer_name]
    if not prisms:
        return

    for prism in prisms:
        poly = ShapelyPolygon(prism.vertices)
        if poly.is_empty or not poly.is_valid:
            continue

        bounds = poly.bounds
        margin = 1.0
        if slice_axis == "y":
            line = LineString(
                [
                    (bounds[0] - margin, slice_coord),
                    (bounds[2] + margin, slice_coord),
                ]
            )
        else:
            line = LineString(
                [
                    (slice_coord, bounds[1] - margin),
                    (slice_coord, bounds[3] + margin),
                ]
            )

        intersection = poly.intersection(line)
        if intersection.is_empty:
            continue

        segments = []
        if isinstance(intersection, LineString):
            segments.append(intersection)
        elif isinstance(intersection, MultiLineString):
            segments.extend(intersection.geoms)

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

            show_legend = layer_name not in seen
            seen.add(layer_name)
            _add_rect_trace(
                fig,
                h_min,
                prism.z_base,
                h_max - h_min,
                prism.z_top - prism.z_base,
                fillcolor=_rgba(color, alpha),
                line_color="black",
                name=layer_name,
                show_legend=show_legend,
                legendgroup=layer_name,
                hover=f"{layer_name} ({prism.material})",
            )


# ---------------------------------------------------------------------------
# Dielectric helpers
# ---------------------------------------------------------------------------


def _add_dielectrics_xy(
    fig: Any,
    dielectrics: list,
    z_slice: float,
    cmin: tuple[float, float, float],
    cmax: tuple[float, float, float],
) -> None:
    """Draw dielectric background rectangles for an XY slice."""
    seen: set[str] = set()
    for diel in dielectrics:
        if not (diel.zmin <= z_slice <= diel.zmax):
            continue
        show = diel.name not in seen
        seen.add(diel.name)
        _add_rect_trace(
            fig,
            cmin[0],
            cmin[1],
            cmax[0] - cmin[0],
            cmax[1] - cmin[1],
            fillcolor=_diel_color(diel.material),
            line_color="rgba(0,0,0,0)",
            name=diel.name,
            show_legend=show,
            legendgroup=f"diel_{diel.name}",
            hover=f"{diel.name} ({diel.material})",
        )


def _add_dielectrics_side(
    fig: Any,
    dielectrics: list,
    h_min: float,
    h_max: float,
) -> None:
    """Draw dielectric background rectangles for a side (YZ/XZ) slice."""
    seen: set[str] = set()
    for diel in dielectrics:
        if diel.zmax <= diel.zmin:
            continue
        show = diel.name not in seen
        seen.add(diel.name)
        _add_rect_trace(
            fig,
            h_min,
            diel.zmin,
            h_max - h_min,
            diel.zmax - diel.zmin,
            fillcolor=_diel_color(diel.material),
            line_color="rgba(0,0,0,0)",
            name=diel.name,
            show_legend=show,
            legendgroup=f"diel_{diel.name}",
            hover=f"{diel.name} ({diel.material})",
        )


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------


def _add_pml_rect(
    fig: Any,
    x0: float,
    y0: float,
    w: float,
    h: float,
    show_legend: bool,
) -> None:
    """Add a PML region rectangle to the figure."""
    if w <= 0 or h <= 0:
        return
    _add_rect_trace(
        fig, x0, y0, w, h, _PML_FILL, _PML_LINE, "PML", show_legend, "PML", "PML"
    )


def _add_sim_cell_rect(
    fig: Any,
    x0: float,
    y0: float,
    w: float,
    h: float,
) -> None:
    """Add a dashed simulation-cell boundary rectangle."""
    import plotly.graph_objects as go

    xs = [x0, x0 + w, x0 + w, x0, x0]
    ys = [y0, y0, y0 + h, y0 + h, y0]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            name="Sim cell",
            legendgroup="sim_cell",
            showlegend=True,
            hoverinfo="skip",
        )
    )


def _add_overlay_xy(fig: Any, overlay: Any) -> None:
    """Draw sim cell, PML, and port overlays for an XY slice."""
    cmin, cmax = overlay.cell_min, overlay.cell_max
    pml = overlay.dpml
    x0, y0 = cmin[0], cmin[1]
    x1, y1 = cmax[0], cmax[1]
    w, h = x1 - x0, y1 - y0

    _add_sim_cell_rect(fig, x0, y0, w, h)
    _add_pml_rect(fig, x0, y0, pml, h, True)
    _add_pml_rect(fig, x1 - pml, y0, pml, h, False)
    _add_pml_rect(fig, x0 + pml, y0, w - 2 * pml, pml, False)
    _add_pml_rect(fig, x0 + pml, y1 - pml, w - 2 * pml, pml, False)

    _add_ports_xy(fig, overlay.ports)


def _add_overlay_yz(fig: Any, overlay: Any, x_slice: float) -> None:
    """Draw sim cell, PML, and port overlays for a YZ slice."""
    cmin, cmax = overlay.cell_min, overlay.cell_max
    pml = overlay.dpml
    y0, z0 = cmin[1], cmin[2]
    y1, z1 = cmax[1], cmax[2]
    w, h = y1 - y0, z1 - z0

    _add_sim_cell_rect(fig, y0, z0, w, h)
    _add_pml_rect(fig, y0, z0, pml, h, True)
    _add_pml_rect(fig, y1 - pml, z0, pml, h, False)
    _add_pml_rect(fig, y0 + pml, z0, w - 2 * pml, pml, False)
    _add_pml_rect(fig, y0 + pml, z1 - pml, w - 2 * pml, pml, False)

    _add_ports_yz(fig, overlay.ports, x_slice)


def _add_overlay_xz(fig: Any, overlay: Any, y_slice: float) -> None:
    """Draw sim cell, PML, port, and fiber overlays for an XZ slice."""
    cmin, cmax = overlay.cell_min, overlay.cell_max
    pml = overlay.dpml
    x0, z0 = cmin[0], cmin[2]
    x1, z1 = cmax[0], cmax[2]
    w, h = x1 - x0, z1 - z0

    _add_sim_cell_rect(fig, x0, z0, w, h)
    _add_pml_rect(fig, x0, z0, pml, h, True)
    _add_pml_rect(fig, x1 - pml, z0, pml, h, False)
    _add_pml_rect(fig, x0 + pml, z0, w - 2 * pml, pml, False)
    _add_pml_rect(fig, x0 + pml, z1 - pml, w - 2 * pml, pml, False)

    _add_ports_xz(fig, overlay.ports, y_slice)

    fiber = getattr(overlay, "fiber", None)
    if fiber is not None:
        _add_fiber_source(fig, fiber)


# ---------------------------------------------------------------------------
# Port drawing
# ---------------------------------------------------------------------------


def _add_ports_xy(fig: Any, ports: list) -> None:
    """Draw source/monitor port lines on an XY slice."""
    import plotly.graph_objects as go

    labeled: set[str] = set()
    for port in ports:
        cx, cy, _ = port.center
        color = _SRC_COLOR if port.is_source else _MON_COLOR
        legend_key = "Source" if port.is_source else "Monitor"
        show = legend_key not in labeled
        labeled.add(legend_key)
        hw = port.width / 2

        if port.normal_axis == 0:
            xs = [cx, cx]
            ys = [cy - hw, cy + hw]
        else:
            xs = [cx - hw, cx + hw]
            ys = [cy, cy]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+text",
                line=dict(color=color, width=2),
                name=legend_key,
                legendgroup=legend_key,
                showlegend=show,
                text=["", port.name],
                textposition="top center",
                textfont=dict(size=9, color=color),
                hoverinfo="text",
                hovertext=f"{port.name} ({legend_key})",
            )
        )


def _add_ports_yz(fig: Any, ports: list, x_slice: float) -> None:
    """Draw source/monitor port rectangles on a YZ slice."""
    import plotly.graph_objects as go

    labeled: set[str] = set()
    for port in ports:
        cx, cy, cz = port.center
        if port.normal_axis != 0 or abs(cx - x_slice) > 0.01:
            continue
        color = _SRC_COLOR if port.is_source else _MON_COLOR
        legend_key = "Source" if port.is_source else "Monitor"
        show = legend_key not in labeled
        labeled.add(legend_key)
        hw = port.width / 2
        hz = port.z_span / 2

        xs = [cy - hw, cy + hw, cy + hw, cy - hw, cy - hw]
        ys = [cz - hz, cz - hz, cz + hz, cz + hz, cz - hz]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+text",
                line=dict(color=color, width=1.5),
                fill="none",
                name=legend_key,
                legendgroup=legend_key,
                showlegend=show,
                text=["", "", port.name, "", ""],
                textposition="top center",
                textfont=dict(size=9, color=color),
                hoverinfo="text",
                hovertext=f"{port.name} ({legend_key})",
            )
        )


def _add_ports_xz(fig: Any, ports: list, y_slice: float) -> None:
    """Draw source/monitor port lines on an XZ slice."""
    import plotly.graph_objects as go

    labeled: set[str] = set()
    for port in ports:
        cx, cy, cz = port.center
        if port.normal_axis != 0:
            continue
        if abs(cy - y_slice) > port.width / 2 + 1e-6:
            continue
        color = _SRC_COLOR if port.is_source else _MON_COLOR
        legend_key = "Source" if port.is_source else "Monitor"
        show = legend_key not in labeled
        labeled.add(legend_key)
        hz = port.z_span / 2

        fig.add_trace(
            go.Scatter(
                x=[cx, cx],
                y=[cz - hz, cz + hz],
                mode="lines+text",
                line=dict(color=color, width=2),
                name=legend_key,
                legendgroup=legend_key,
                showlegend=show,
                text=["", port.name],
                textposition="top center",
                textfont=dict(size=9, color=color),
                hoverinfo="text",
                hovertext=f"{port.name} ({legend_key})",
            )
        )


def _add_fiber_source(fig: Any, fiber: Any) -> None:
    """Draw a fiber source arrow and waist indicator."""
    import math

    import plotly.graph_objects as go

    fx, fz = fiber.x, fiber.z
    theta = math.radians(fiber.angle_deg)
    arrow_len = max(fiber.waist * 0.6, 1.0)
    tail_x = fx - math.sin(theta) * arrow_len
    tail_z = fz + math.cos(theta) * arrow_len

    fig.add_trace(
        go.Scatter(
            x=[tail_x, fx],
            y=[tail_z, fz],
            mode="lines",
            line=dict(color=_SRC_COLOR, width=2),
            name="Source",
            legendgroup="Source",
            showlegend=False,
            hoverinfo="text",
            hovertext="fiber source",
        )
    )

    fig.add_annotation(
        x=fx,
        y=fz,
        ax=tail_x,
        ay=tail_z,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowcolor=_SRC_COLOR,
    )

    perp_x = math.cos(theta)
    perp_z = math.sin(theta)
    hw = fiber.waist

    fig.add_trace(
        go.Scatter(
            x=[fx - perp_x * hw, fx + perp_x * hw],
            y=[fz - perp_z * hw, fz + perp_z * hw],
            mode="lines+text",
            line=dict(color=_SRC_COLOR, width=2),
            name="Source",
            legendgroup="Source",
            showlegend=False,
            text=["", "fiber"],
            textposition="top center",
            textfont=dict(size=9, color=_SRC_COLOR),
            hoverinfo="text",
            hovertext="fiber source",
        )
    )
