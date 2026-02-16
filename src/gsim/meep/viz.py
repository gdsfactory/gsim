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

    # Compute port extension length (same logic as write_config)
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

    z_lo = min(layer.zmin for layer in stack.layers.values())
    z_hi = max(layer.zmax for layer in stack.layers.values())

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
) -> Any:
    """Build a SimOverlay from config, if stack is available.

    Args:
        geometry_model: The geometry model (for bbox).
        component: Original component (for bbox and port extraction).
        stack: gsim LayerStack (needed for dielectrics). May be None.
        domain_config: Domain configuration.
        source_port: Source port name (or None for auto).

    Returns:
        SimOverlay or None if stack isn't configured.
    """
    from gsim.meep.overlay import build_sim_overlay

    if stack is None:
        return None

    try:
        from gsim.meep.ports import extract_port_info

        comp_copy = component.copy()
        port_data = extract_port_info(comp_copy, stack, source_port=source_port)
    except Exception:
        port_data = []

    # Pass original component bbox so cell boundaries are computed from
    # the original geometry, not the port-extended geometry.
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
    **kwargs: Any,
) -> Any:
    """Create interactive 3D visualization of MEEP geometry.

    Args:
        component: gdsfactory Component to visualize.
        stack: gsim LayerStack (may be None).
        domain_config: Domain configuration.
        backend: "open3d" (Jupyter/VS Code) or "pyvista" (desktop).
        **kwargs: Extra args forwarded to the backend renderer.

    Returns:
        Renderer-specific widget or plotter object.
    """
    from gsim.common.viz import plot_prisms_3d, plot_prisms_3d_open3d

    gm = build_geometry_model(component, stack, domain_config)
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

    Returns:
        ``plt.Axes`` when *ax* was provided, otherwise ``None``.
    """
    from gsim.common.viz import plot_prism_slices

    gm = build_geometry_model(component, stack, domain_config)
    overlay = build_overlay(gm, component, stack, domain_config, source_port)
    return plot_prism_slices(gm, x, y, z, ax, legend, slices, overlay=overlay)
