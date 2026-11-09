"""PDK display defaults for FDTD physical groups."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from gsim.common.pdk import ResolvedPassivePcell


class GroupStyle(TypedDict, total=False):
    """Initial visibility and surface opacity of a physical group."""

    opacity: float
    visible: bool


def _layer_tuple(
    layer: Any, pdk: Any = None, component: Any = None
) -> tuple[int, int] | None:
    """Return a concrete GDS layer, leaving boolean expressions unmatched."""
    if isinstance(layer, tuple) and len(layer) == 2:
        return int(layer[0]), int(layer[1])
    if isinstance(layer, str):
        named_layer = getattr(getattr(pdk, "layers", None), layer, None)
        return _layer_tuple(named_layer, pdk, component)
    nested_layer = getattr(layer, "layer", None)
    datatype = getattr(layer, "datatype", None)
    if isinstance(nested_layer, int) and isinstance(datatype, int):
        return nested_layer, datatype
    if nested_layer is not None:
        return _layer_tuple(nested_layer, pdk, component)
    if isinstance(layer, int) and component is not None:
        info = component.kcl.get_info(layer)
        return int(info.layer), int(info.datatype)
    return None


def _visibility_by_layer(
    views: Mapping[str, Any], parent_visible: bool = True, *, layers: Any = None
) -> dict[tuple[int, int], bool]:
    """Show a layer if any view and all of that view's parents are visible."""
    visibility = {}
    for name, view in views.items():
        visible = parent_visible and getattr(view, "visible", True)
        layer = _layer_tuple(getattr(view, "layer", None))
        if layer is None:
            layer = _layer_tuple(getattr(layers, name, None))
        if layer is not None:
            visibility[layer] = visibility.get(layer, False) or visible
        child_visibility = _visibility_by_layer(
            getattr(view, "group_members", {}), visible, layers=layers
        )
        for child_layer, child_visible in child_visibility.items():
            visibility[child_layer] = (
                visibility.get(child_layer, False) or child_visible
            )
    return visibility


def normalize_group_styles(styles: Mapping[str, GroupStyle]) -> dict[str, GroupStyle]:
    """Validate display metadata and make zero-opacity groups toggleable."""
    normalized = {}
    for name, style in styles.items():
        result: GroupStyle = {}
        if "visible" in style:
            visible = style["visible"]
            if not isinstance(visible, bool):
                raise ValueError(f"Group {name!r} visibility must be a bool")
            result["visible"] = visible
        if "opacity" in style:
            try:
                opacity = float(style["opacity"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Group {name!r} opacity must be between 0 and 1"
                ) from error
            if not isfinite(opacity) or not 0 <= opacity <= 1:
                raise ValueError(f"Group {name!r} opacity must be between 0 and 1")
            if opacity == 0:
                # A hidden group must become visible when switched on in the UI.
                result.update(opacity=1.0, visible=False)
            else:
                result["opacity"] = opacity
        if result:
            normalized[name] = result
    return normalized


def layer_group_styles(resolved: ResolvedPassivePcell | None) -> dict[str, GroupStyle]:
    """Map native stack and layer-view metadata to mesh physical-group names.

    Stack info.opacity supplies numeric opacity. Stack info.visible overrides
    the layer view's effective visibility. Derived layers use their derived
    GDS layer view where present. Only layers with resolved geometry are styled
    and validated. No display metadata changes simulation inputs.
    """
    if resolved is None:
        return {}
    stack = getattr(resolved, "layer_stack", None)
    if stack is None:
        return {}
    pdk = getattr(resolved, "pdk", None)
    component = getattr(resolved, "component", None)
    pdk_views = getattr(pdk, "layer_views", None)
    get_layer_views = getattr(pdk, "get_layer_views", None)
    if pdk_views is not None and callable(get_layer_views):
        pdk_views = get_layer_views()
    visibility = _visibility_by_layer(
        getattr(pdk_views, "layer_views", {}),
        layers=getattr(pdk_views, "layers", None),
    )
    styles: dict[str, GroupStyle] = {}
    for name, level in stack.layers.items():
        if name not in resolved.layers:
            continue
        style: GroupStyle = {}
        for declared_layer in (level.derived_layer, level.layer):
            layer = _layer_tuple(declared_layer, pdk, component)
            if layer is not None and layer in visibility:
                style["visible"] = visibility[layer]
                break
        info = level.info
        if info.get("visible") is not None:
            style["visible"] = info["visible"]
        if info.get("opacity") is not None:
            style["opacity"] = info["opacity"]
        if style:
            styles[name] = style
    return normalize_group_styles(styles)
