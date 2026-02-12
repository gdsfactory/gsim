"""Color and layer utilities shared across 3D rendering backends.

These helpers are consumed internally by the render3d_* modules.
"""

from __future__ import annotations


def generate_layer_colors(
    layer_names: list[str],
) -> dict[str, tuple[float, float, float]]:
    """Generate distinct RGB colours for each layer using a matplotlib colormap.

    Returns:
        Mapping from layer name to (r, g, b) with values in [0, 1].
    """
    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("tab10" if len(layer_names) <= 10 else "tab20")
    colors: dict[str, tuple[float, float, float]] = {}
    for i, name in enumerate(layer_names):
        rgb = cmap(i / max(len(layer_names) - 1, 1))[:3]
        colors[name] = rgb
    return colors


def generate_layer_colors_with_opacity(
    layer_names: list[str],
    layer_opacity: dict[str, float] | None = None,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Generate RGB colors and a separate opacity dict.

    Args:
        layer_names: Ordered list of layer names.
        layer_opacity: Optional per-layer opacity override.
            Defaults to core=1.0, everything else=0.2.

    Returns:
        (colors_dict, opacity_dict) where colors has [r, g, b] in [0, 1]
        and opacity has float in [0, 1].
    """
    import matplotlib.pyplot as plt

    if layer_opacity is None:
        layer_opacity = {name: 1.0 if name == "core" else 0.2 for name in layer_names}

    cmap = plt.cm.get_cmap("tab10" if len(layer_names) <= 10 else "tab20")
    colors: dict[str, list[float]] = {}
    for i, name in enumerate(layer_names):
        rgb = cmap(i / max(len(layer_names) - 1, 1))[:3]
        colors[name] = list(rgb)

    return colors, layer_opacity
