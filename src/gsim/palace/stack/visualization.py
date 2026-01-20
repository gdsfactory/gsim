"""Stack visualization utility for PDK layer stacks.

Prints ASCII diagrams showing the layer stack structure with z-positions,
thicknesses, and layer numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go
from gdsfactory.technology import LayerLevel
from gdsfactory.technology import LayerStack as GfLayerStack


@dataclass
class StackLayer:
    """Parsed layer info for visualization."""

    name: str
    zmin: float
    zmax: float
    thickness: float
    material: str | None = None
    gds_layer: int | None = None
    layer_type: str = "conductor"  # conductor, via, dielectric, substrate


def _get_gds_layer_number(layer_level: LayerLevel) -> int | None:
    """Extract GDS layer number from LayerLevel."""
    layer: Any = layer_level.layer

    # Handle tuple
    if isinstance(layer, tuple):
        return int(layer[0])

    # Handle int
    if isinstance(layer, int):
        return int(layer)

    # Handle LogicalLayer or enum with nested layer
    if hasattr(layer, "layer"):
        inner = layer.layer
        if hasattr(inner, "layer"):
            return int(inner.layer)
        if isinstance(inner, int):
            return int(inner)

    # Handle enum with value
    if hasattr(layer, "value"):
        if isinstance(layer.value, tuple):
            return int(layer.value[0])
        return int(layer.value)

    return None


def _classify_layer(name: str) -> str:
    """Classify layer type based on name."""
    name_lower = name.lower()

    if "via" in name_lower or "cont" in name_lower:
        return "via"
    if "substrate" in name_lower or name_lower == "sub":
        return "substrate"
    if any(
        m in name_lower
        for m in ["metal", "topmetal", "m1", "m2", "m3", "m4", "m5", "poly", "active"]
    ):
        return "conductor"

    return "dielectric"


def parse_layer_stack(layer_stack: GfLayerStack) -> list[StackLayer]:
    """Parse a gdsfactory LayerStack into a list of StackLayer objects.

    Args:
        layer_stack: gdsfactory LayerStack object

    Returns:
        List of StackLayer objects sorted by zmin (ascending)
    """
    layers = []

    for name, level in layer_stack.layers.items():
        zmin = level.zmin if level.zmin is not None else 0.0
        thickness = level.thickness if level.thickness is not None else 0.0
        zmax = zmin + thickness
        material = level.material if level.material else None
        gds_layer = _get_gds_layer_number(level)
        layer_type = _classify_layer(name)

        layers.append(
            StackLayer(
                name=name,
                zmin=zmin,
                zmax=zmax,
                thickness=thickness,
                material=material,
                gds_layer=gds_layer,
                layer_type=layer_type,
            )
        )

    # Sort by zmin ascending
    layers.sort(key=lambda layer: layer.zmin)
    return layers


def _format_layer_name(name: str, _max_len: int = 20) -> str:
    """Format layer name with abbreviation in parentheses."""
    # Common abbreviations
    abbrevs = {
        "topmetal2": "TM2",
        "topmetal1": "TM1",
        "topvia2": "TV2",
        "topvia1": "TV1",
        "metal5": "M5",
        "metal4": "M4",
        "metal3": "M3",
        "metal2": "M2",
        "metal1": "M1",
        "via4": "V4",
        "via3": "V3",
        "via2": "V2",
        "via1": "V1",
        "poly": "Poly",
        "active": "Act",
        "substrate": "Sub",
    }

    name_lower = name.lower()
    if name_lower in abbrevs:
        abbrev = abbrevs[name_lower]
        display = name.capitalize() if name[0].islower() else name
        return f"{display} ({abbrev})"

    return name


def print_stack(pdk) -> str:
    """Print an ASCII diagram of the layer stack.

    Args:
        pdk: A PDK module with LAYER_STACK, or a LayerStack directly

    Returns:
        The formatted string (also prints to stdout)

    Examples:
        ```python
        import ihp

        print_stack(ihp)
        ```
    """
    # Extract LayerStack from PDK module if needed
    layer_stack = pdk.LAYER_STACK if hasattr(pdk, "LAYER_STACK") else pdk

    layers = parse_layer_stack(layer_stack)

    if not layers:
        return "No layers found in stack"

    # Separate layers by type
    substrate_layer = None
    active_layers = []  # active, poly, etc.
    metal_layers = []

    for layer in layers:
        if layer.layer_type == "substrate":
            substrate_layer = layer
        elif layer.name.lower() in ("active", "poly", "gatpoly"):
            active_layers.append(layer)
        else:
            metal_layers.append(layer)

    # Build the diagram
    lines = []
    width = 50
    box_width = width - 4

    # Title
    title = "Layer Stack"
    lines.append(f"  Z (um){title:^{width + 10}}")
    lines.append("  " + "─" * (width + 18))

    # Sort metal layers by zmax descending for top-down drawing
    metal_layers_sorted = sorted(
        metal_layers, key=lambda layer: layer.zmax, reverse=True
    )

    # Draw top border
    if metal_layers_sorted:
        first_layer = metal_layers_sorted[0]
        lines.append(f"{first_layer.zmax:7.2f} ┌{'─' * box_width}┐")

    # Draw each metal layer from top to bottom
    for _i, layer in enumerate(metal_layers_sorted):
        display_name = _format_layer_name(layer.name)
        thickness_str = (
            f"{layer.thickness:.2f} um"
            if layer.thickness >= 0.01
            else f"{layer.thickness * 1000:.0f} nm"
        )
        layer_str = f"Layer {layer.gds_layer}" if layer.gds_layer else ""

        name_part = f"{display_name:^{box_width - 24}}"
        info_part = f"{thickness_str:>10}  {layer_str:<10}"
        content = f"{name_part}{info_part}"

        lines.append(f"{'':>8}│{content:^{box_width}}│")
        lines.append(f"{layer.zmin:7.2f} ├{'─' * box_width}┤")

    # Dielectric/oxide region
    lines.append(f"{'':>8}│{'(dielectric / oxide)':^{box_width}}│")

    # Active layers (active, poly)
    if active_layers:
        active_sorted = sorted(
            active_layers, key=lambda layer: layer.zmax, reverse=True
        )
        z_top = max(layer.zmax for layer in active_layers)
        third = box_width // 3
        tail = "─" * (box_width - 2 * third - 2)
        lines.append(f"{z_top:7.2f} ├{'─' * third}┬{'─' * third}┬{tail}┤")

        names = "   ".join(layer.name.capitalize() for layer in active_sorted[:2])
        gds_layers = ", ".join(
            str(layer.gds_layer) for layer in active_sorted[:2] if layer.gds_layer
        )
        content = f"{names}  ~{active_sorted[0].thickness:.1f} um  Layer {gds_layers}"
        lines.append(f"{'':>8}│{content:^{box_width}}│")
        lines.append(f"{0.00:7.2f} ├{'─' * third}┴{'─' * third}┴{tail}┤")
    else:
        lines.append(f"{0.00:7.2f} ├{'─' * box_width}┤")

    # Substrate
    if substrate_layer:
        lines.append(f"{'':>8}│{'':^{box_width}}│")
        lines.append(f"{'':>8}│{'Substrate (Si)':^{box_width}}│")
        sub_thickness = f"{abs(substrate_layer.thickness):.0f} um"
        lines.append(f"{'':>8}│{sub_thickness:^{box_width}}│")
        lines.append(f"{'':>8}│{'':^{box_width}}│")
        lines.append(f"{substrate_layer.zmin:7.0f} └{'─' * box_width}┘")
    else:
        lines.append(f"{'':>8}└{'─' * box_width}┘")

    lines.append("  " + "─" * (width + 18))

    result = "\n".join(lines)
    return result


def _find_overlap_groups(layers: list[StackLayer]) -> list[list[StackLayer]]:
    """Group layers that overlap in z-range.

    Returns list of groups, where each group contains layers that overlap.
    Non-overlapping layers are in their own single-element groups.
    """
    if not layers:
        return []

    # Sort by zmin
    sorted_layers = sorted(layers, key=lambda layer: layer.zmin)

    groups = []
    current_group = [sorted_layers[0]]
    group_zmax = sorted_layers[0].zmax

    for layer in sorted_layers[1:]:
        # Check if this layer overlaps with current group
        if layer.zmin < group_zmax:
            # Overlaps - add to current group
            current_group.append(layer)
            group_zmax = max(group_zmax, layer.zmax)
        else:
            # No overlap - start new group
            groups.append(current_group)
            current_group = [layer]
            group_zmax = layer.zmax

    groups.append(current_group)
    return groups


def plot_stack(pdk, width: float = 600, height: float = 800, to_scale: bool = False):
    """Create an interactive plotly visualization of the layer stack.

    Args:
        pdk: A PDK module with LAYER_STACK, or a LayerStack directly
        width: Figure width in pixels
        height: Figure height in pixels
        to_scale: If True, show actual z dimensions. If False (default),
                  use fixed height for all layers for better visibility.

    Returns:
        plotly Figure object (displays automatically in notebooks)

    Examples:
        ```python
        import ihp

        plot_stack(ihp)
        ```
    """
    # Extract LayerStack from PDK module if needed
    layer_stack = pdk.LAYER_STACK if hasattr(pdk, "LAYER_STACK") else pdk

    layers = parse_layer_stack(layer_stack)

    if not layers:
        fig = go.Figure()
        fig.add_annotation(text="No layers found", x=0.5, y=0.5, showarrow=False)
        return fig

    # Color scheme by layer type
    colors = {
        "conductor": "#4CAF50",  # Green for metals/actives
        "via": "#87CEEB",  # Sky blue for vias
        "substrate": "#D0D0D0",  # Light gray for substrate
        "dielectric": "#D0D0D0",  # Light gray for dielectrics
    }

    # Find overlapping groups
    overlap_groups = _find_overlap_groups(layers)

    # Calculate column assignments and positions for each layer
    layer_columns = {}  # layer.name -> (column_index, total_columns_in_group)
    for group in overlap_groups:
        for i, layer in enumerate(group):
            layer_columns[layer.name] = (i, len(group))

    # Sort layers by zmin for consistent ordering
    sorted_layers = sorted(layers, key=lambda layer: layer.zmin)

    # Calculate uniform positions (not to scale)
    # In uniform mode, simply stack all layers vertically
    uniform_height = 1.0
    uniform_positions = {}  # layer.name -> (y0, y1)
    current_y = 0
    for layer in sorted_layers:
        uniform_positions[layer.name] = (current_y, current_y + uniform_height)
        current_y += uniform_height

    fig = go.Figure()

    # Base box width and x-coordinates
    total_width = 4
    base_x0 = 0

    # Helper to calculate x-coordinates for a layer based on its column
    # (for to-scale view).
    def get_x_coords_scaled(layer_name):
        col_idx, num_cols = layer_columns[layer_name]
        col_width = total_width / num_cols
        x0 = base_x0 + col_idx * col_width
        x1 = x0 + col_width
        return x0, x1

    # Add shapes and traces for both views
    # We'll use visibility toggling with buttons
    for _i, layer in enumerate(sorted_layers):
        color = colors.get(layer.layer_type, "#CCCCCC")

        # In uniform view, use full width; in to-scale view, use columns
        if to_scale:
            x0, x1 = get_x_coords_scaled(layer.name)
        else:
            x0, x1 = base_x0, base_x0 + total_width

        # Uniform (not to scale) positions
        u_y0, u_y1 = uniform_positions[layer.name]

        # To-scale positions
        s_y0, s_y1 = layer.zmin, layer.zmax

        # Use initial positions based on to_scale parameter
        y0 = s_y0 if to_scale else u_y0
        y1 = s_y1 if to_scale else u_y1

        # Add rectangle shape
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=color,
            line=dict(color="black", width=1),
            layer="below",
            name=f"shape_{layer.name}",
        )

        # Add invisible scatter for hover info
        fig.add_trace(
            go.Scatter(
                x=[(x0 + x1) / 2],
                y=[(y0 + y1) / 2],
                mode="markers",
                marker=dict(size=20, opacity=0),
                hoverinfo="text",
                hovertext=(
                    f"<b>{layer.name}</b><br>"
                    f"GDS Layer: {layer.gds_layer or 'N/A'}<br>"
                    f"Type: {layer.layer_type}<br>"
                    f"Z: {layer.zmin:.2f} - {layer.zmax:.2f} µm<br>"
                    f"Thickness: {layer.thickness:.3f} µm<br>"
                    f"Material: {layer.material or 'N/A'}"
                ),
                showlegend=False,
                name=f"hover_{layer.name}",
            )
        )

        # Add layer label
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=f"<b>{layer.name}</b>",
            showarrow=False,
            font=dict(size=10),
            name=f"label_{layer.name}",
        )

        # Add GDS layer number on the right side of the box
        if layer.gds_layer is not None:
            fig.add_annotation(
                x=x1 - 0.1,
                y=(y0 + y1) / 2,
                text=f"{layer.gds_layer}",
                showarrow=False,
                font=dict(size=9, color="gray"),
                xanchor="right",
            )

    # Build button data for toggling between views
    def build_layout_update(use_scale):
        shapes = []
        annotations = []

        for layer in sorted_layers:
            color = colors.get(layer.layer_type, "#CCCCCC")

            # Uniform view: full width; To-scale view: columns for overlaps
            if use_scale:
                lx0, lx1 = get_x_coords_scaled(layer.name)
                y0, y1 = layer.zmin, layer.zmax
            else:
                lx0, lx1 = base_x0, base_x0 + total_width
                y0, y1 = uniform_positions[layer.name]

            shapes.append(
                dict(
                    type="rect",
                    x0=lx0,
                    x1=lx1,
                    y0=y0,
                    y1=y1,
                    fillcolor=color,
                    line=dict(color="black", width=1),
                    layer="below",
                )
            )

            annotations.append(
                dict(
                    x=(lx0 + lx1) / 2,
                    y=(y0 + y1) / 2,
                    text=f"<b>{layer.name}</b>",
                    showarrow=False,
                    font=dict(size=10),
                )
            )

            # Add GDS layer number on the right side
            if layer.gds_layer is not None:
                annotations.append(
                    dict(
                        x=lx1 - 0.1,
                        y=(y0 + y1) / 2,
                        text=f"{layer.gds_layer}",
                        showarrow=False,
                        font=dict(size=9, color="gray"),
                        xanchor="right",
                    )
                )

        y_title = "Z (µm)" if use_scale else "Layer (not to scale)"
        if use_scale:
            y_range = [
                min(layer.zmin for layer in sorted_layers) - 1,
                max(layer.zmax for layer in sorted_layers) + 1,
            ]
        else:
            y_range = [-0.5, len(sorted_layers) + 0.5]

        return dict(
            shapes=shapes,
            annotations=annotations,
            yaxis=dict(
                title=y_title,
                range=y_range,
                showgrid=False,
                zeroline=False,
                showticklabels=use_scale,
            ),
        )

    # Build scatter y-positions for each view
    def build_scatter_update(use_scale):
        updates = []
        for layer in sorted_layers:
            if use_scale:
                y0, y1 = layer.zmin, layer.zmax
            else:
                y0, y1 = uniform_positions[layer.name]
            updates.append([(y0 + y1) / 2])
        return updates

    uniform_scatter_y = build_scatter_update(False)
    scale_scatter_y = build_scatter_update(True)

    # Initial y-range
    if to_scale:
        y_range = [
            min(layer.zmin for layer in sorted_layers) - 1,
            max(layer.zmax for layer in sorted_layers) + 1,
        ]
    else:
        y_range = [-0.5, len(sorted_layers) + 0.5]

    # Layout with buttons
    fig.update_layout(
        title="Layer Stack",
        width=width,
        height=height,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, 4.5],
        ),
        yaxis=dict(
            title="Layer (not to scale)" if not to_scale else "Z (µm)",
            showgrid=False,
            zeroline=False,
            showticklabels=to_scale,
            range=y_range,
        ),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Uniform",
                        method="update",
                        args=[
                            {"y": uniform_scatter_y},
                            build_layout_update(False),
                        ],
                    ),
                    dict(
                        label="To Scale",
                        method="update",
                        args=[
                            {"y": scale_scatter_y},
                            build_layout_update(True),
                        ],
                    ),
                ],
            ),
        ],
    )

    return fig


def print_stack_table(pdk) -> str:
    """Print a table of layer information.

    Args:
        pdk: A PDK module with LAYER_STACK, or a LayerStack directly

    Returns:
        The formatted string (also prints to stdout)

    Examples:
        ```python
        import ihp

        print_stack_table(ihp)
        ```
    """
    layer_stack = pdk.LAYER_STACK if hasattr(pdk, "LAYER_STACK") else pdk

    layers = parse_layer_stack(layer_stack)

    lines = []
    lines.append("\nLayer Stack Table")
    lines.append("=" * 80)
    lines.append(
        f"{'Layer':<15} {'GDS':<8} {'Type':<12} "
        f"{'Z-min':>10} {'Z-max':>10} {'Thick':>10} {'Material':<12}"
    )
    lines.append("-" * 80)

    # Sort by zmin descending (top to bottom)
    for layer in sorted(layers, key=lambda layer: layer.zmin, reverse=True):
        gds = str(layer.gds_layer) if layer.gds_layer else "-"
        material = layer.material or "-"
        lines.append(
            f"{layer.name:<15} {gds:<8} {layer.layer_type:<12} "
            f"{layer.zmin:>10.2f} {layer.zmax:>10.2f} "
            f"{layer.thickness:>10.2f} {material:<12}"
        )

    lines.append("=" * 80)

    result = "\n".join(lines)
    return result
