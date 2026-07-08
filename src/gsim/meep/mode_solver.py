"""Standalone MEEP eigenmode solver for 1D slab and 2D waveguide cross-sections.

Provides two public entry points:

- :func:`solve_slab_mode` — 1D slab mode from a layer stack
- :func:`solve_cross_section_mode` — 2D waveguide cross-section at a port or position

Both use ``sim.get_eigenmode()`` internally and return a :class:`ModeResult`.
MEEP is an optional dependency imported lazily; see :func:`_import_meep`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from gsim.meep.models.results import ModeResult

if TYPE_CHECKING:
    import numpy as np
    from gdsfactory import Component

    from gsim.common.stack.extractor import LayerStack
    from gsim.meep.models.config import MaterialData

# Lazy MEEP import guard.
_MEET_NOT_FOUND_MSG = (
    "MEEP is required for local mode solving. "
    "Install it via conda-forge:\n"
    "    conda install -c conda-forge pymeep"
)

_PARITY_MAP: dict[str, int] = {
    "NO_PARITY": 0,
    "EVEN_Y": 1,
    "ODD_Y": 2,
    "EVEN_Z": 3,
    "ODD_Z": 4,
}

# Per-process cache for material resolution — keyed by
# (frozenset(material_names), wavelength_um).  The same stack/lambda
# invoked many times (multi-band, parity, comparison sections in
# notebooks) will reuse the sellmeier evaluation.
_MATERIAL_CACHE: dict[tuple, dict] = {}

_MAX_MATERIAL_CACHE = 64


def _resolve_materials_cached(
    used_materials: set[str],
    overrides: dict | None = None,
    wavelength_um: float | None = None,
    overlay: dict | None = None,
) -> dict[str, MaterialData]:
    """:func:`resolve_materials` with a process-local LRU cache."""
    from gsim.meep.materials import resolve_materials

    key = (frozenset(used_materials), wavelength_um)
    cached = _MATERIAL_CACHE.get(key)
    if cached is not None:
        return cached
    result = resolve_materials(used_materials, overrides, wavelength_um, overlay)
    if len(_MATERIAL_CACHE) >= _MAX_MATERIAL_CACHE:
        _MATERIAL_CACHE.pop(next(iter(_MATERIAL_CACHE)))
    _MATERIAL_CACHE[key] = result
    return result


def _import_meep():
    """Lazily import meep — raises ImportError with install instructions if missing."""
    import logging

    try:
        import meep as mp
    except ImportError:
        raise ImportError(_MEET_NOT_FOUND_MSG) from None
    _level = logging.getLogger("meep.native").getEffectiveLevel()
    if _level == logging.NOTSET or _level >= logging.WARNING:
        mp.verbosity(0)
    elif _level >= logging.INFO:
        mp.verbosity(1)
    else:
        mp.verbosity(2)
    return mp


# ------------------------------------------------------------------
# Material helpers
# ------------------------------------------------------------------


def _meep_medium(material_data: MaterialData) -> Any:
    """Convert a resolved MaterialData to a MEEP Medium object."""
    mp = _import_meep()

    eps = material_data.epsilon_diag
    if isinstance(eps, list):
        eps = eps[0]

    return mp.Medium(epsilon=eps)


def _layer_has_any_polygon(component: Component, layer: object) -> bool:
    """Return ``True`` if *layer* has at least one GDS polygon in *component*.

    Used as a fallback gate: when ``_layer_y_intervals_at_x`` /
    ``_layer_x_intervals_at_y`` return no intervals AND this returns
    ``False``, the layer is treated as a full-width background layer
    (e.g. buried oxide, cladding) spanning the entire simulation domain.
    """
    gds_layer = getattr(layer, "gds_layer", None)
    if gds_layer is None:
        return False
    try:
        polys = component.get_polygons_points(layers=(tuple(gds_layer),), merge=True)
        if not isinstance(polys, dict):
            return False
        return any(len(v) > 0 for v in polys.values())
    except Exception:
        return False


# ------------------------------------------------------------------
# Slab (1D) geometry builder
# ------------------------------------------------------------------


def _build_slab_xz_cell(
    stack: LayerStack,
    materials: dict[str, MaterialData],
    resolution: float,
    z_margin: float | tuple[float, float] = 0.0,
    pml_thickness: float = 0.0,
    *,
    background_material: str = "air",
) -> tuple[Any, Any]:
    """Build a 2D XZ MEEP simulation cell for a 1D slab (uniform layers).

    Each non-air layer becomes a rectangular block spanning the full
    *x*-width. The cell is centred vertically on the stack midpoint
    adjusted for asymmetric margins.

    Args:
        stack: Resolved :class:`LayerStack`.
        materials: ``MaterialData`` keyed by material name.
        resolution: Pixels per µm.
        z_margin: Extra distance (µm) added below and above the stack.
            A single float adds the same margin on both sides; a
            ``(bottom, top)`` tuple adds asymmetric margins. Default 0.
        pml_thickness: PML absorber thickness in µm (default 0.0).
            Set to e.g. 1.0 to enable PML absorbing boundaries.
        background_material: Material name for the MEEP default medium
            (fills space not covered by any layer). Default ``"air"``
            (epsilon=1.0). Set to e.g. ``"sio2"`` to model a buried-oxide
            background without explicit full-span geometry layers.

    Returns:
        ``(sim, cell_size)`` — initialized MEEP :class:`Simulation` object
        and ``mp.Vector3`` cell dimensions.
    """
    mp = _import_meep()

    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    if isinstance(z_margin, (tuple, list)):
        z_margin_bottom, z_margin_top = z_margin
    else:
        z_margin_bottom = z_margin_top = z_margin
    z_center = (z_min + z_max) / 2.0 + (z_margin_top - z_margin_bottom) / 2.0

    # The slab problem is 1D but MEEP requires >=2 cells in the propagation
    # direction for MPB to distinguish the guided eigenmode from the trivial
    # vacuum plane-wave solution. 2 pixels is sufficient and minimizes memory.
    x_span = 2.0 / resolution
    z_span = (z_max - z_min) + z_margin_bottom + z_margin_top
    if pml_thickness > 0:
        x_span += 2 * pml_thickness
        z_span += 2 * pml_thickness
    # Snap to integer pixel counts to avoid meep grid-volume warnings.
    x_span = round(x_span * resolution) / resolution
    z_span = round(z_span * resolution) / resolution
    cell_size = mp.Vector3(x_span, 0.0, z_span)

    geometry: list[object] = []
    for layer in stack.layers.values():
        if layer.material == "air":
            continue
        mat_data = materials.get(layer.material)
        if mat_data is None:
            continue
        medium = _meep_medium(mat_data)
        z_size = layer.zmax - layer.zmin
        if z_size <= 0:
            continue
        z_lo = layer.zmin - z_center
        z_hi = z_lo + z_size
        if abs(layer.zmin - z_min) < 1e-12:
            z_lo = -z_span / 2.0
        z_size = z_hi - z_lo
        block_center_z = (z_lo + z_hi) / 2.0
        block = mp.Block(
            size=mp.Vector3(x_span, mp.inf, z_size),
            center=mp.Vector3(0.0, 0.0, block_center_z),
            material=medium,
        )
        geometry.append(block)

    bg = materials.get(background_material)
    default_medium = _meep_medium(bg) if bg is not None else mp.Medium(epsilon=1.0)

    sim_kwargs: dict = dict(
        cell_size=cell_size,
        geometry=geometry,
        resolution=resolution,
        default_material=default_medium,
    )
    if pml_thickness > 0:
        sim_kwargs["boundary_layers"] = [
            mp.PML(thickness=pml_thickness, direction=mp.X),
            mp.PML(thickness=pml_thickness, direction=mp.Z),
        ]
    sim = mp.Simulation(**sim_kwargs)
    return sim, cell_size


def _subtract_intervals(
    base: tuple[float, float],
    subtract: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the sub-intervals of *base* remaining after removing *subtract*.

    Used to carve "pillar" regions below a layer block that extend all
    the way down to the cell floor when no lower layer is present.
    """
    result = [base]
    for s0, s1 in subtract:
        new_result: list[tuple[float, float]] = []
        for r0, r1 in result:
            if s1 <= r0 or s0 >= r1:
                new_result.append((r0, r1))
            else:
                if r0 < s0:
                    new_result.append((r0, min(s0, r1)))
                if r1 > s1:
                    new_result.append((max(s1, r0), r1))
        result = new_result
        if not result:
            break
    return result


# ------------------------------------------------------------------
# Cross-section (2D) geometry builders
# ------------------------------------------------------------------


def _build_component_xz_cell(
    component: Component,
    stack: LayerStack,
    y_cut: float,
    x_span: float,
    materials: dict[str, MaterialData],
    resolution: float,
    z_margin: float | tuple[float, float] = 0.0,
    pml_thickness: float = 0.0,
    *,
    background_material: str = "air",
) -> tuple[Any, Any]:
    """Build a 2D XZ MEEP simulation cell at a *y*-slice of a component.

    For each non-air layer, the component's GDS polygons are intersected
    with the horizontal line *y = y_cut*.  Each contiguous interval on *x*
    becomes a rectangular block with the layer's material and *z* extent.

    Args:
        component: GDSFactory component (must have polygons).
        stack: Resolved :class:`LayerStack`.
        y_cut: *y* coordinate of the cross-section plane (µm).
        x_span: Total *x* extent of the simulation cell (µm).
        materials: ``MaterialData`` keyed by material name.
        resolution: Pixels per µm.
        z_margin: Extra distance (µm) added below and above the stack.
            A single float adds the same margin on both sides; a
            ``(bottom, top)`` tuple adds asymmetric margins. Default 0.
        pml_thickness: PML absorber thickness in µm (default 0.0).
            Set to e.g. 1.0 to enable PML absorbing boundaries.
        background_material: Material name for the MEEP default medium
            (fills space not covered by any layer). Default ``"air"``
            (epsilon=1.0).

    Returns:
        ``(sim, cell_size)`` — initialized MEEP :class:`Simulation` object
        and ``mp.Vector3`` cell dimensions.
    """
    mp = _import_meep()

    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    if isinstance(z_margin, (tuple, list)):
        z_margin_bottom, z_margin_top = z_margin
    else:
        z_margin_bottom = z_margin_top = z_margin
    z_center = (z_min + z_max) / 2.0 + (z_margin_top - z_margin_bottom) / 2.0
    xz_span = x_span
    if pml_thickness > 0:
        x_span += 2 * pml_thickness
    x_span = round(x_span * resolution) / resolution
    z_span = (z_max - z_min) + z_margin_bottom + z_margin_top
    if pml_thickness > 0:
        z_span += 2 * pml_thickness
    z_span = round(z_span * resolution) / resolution
    cell_size = mp.Vector3(x_span, 0.0, z_span)

    geometry: list[object] = []
    layer_data: list[dict] = []
    for layer in stack.layers.values():
        if layer.material == "air":
            continue
        mat_data = materials.get(layer.material)
        if mat_data is None:
            continue
        layer_thickness = layer.zmax - layer.zmin
        if layer_thickness <= 0:
            continue
        intervals = _layer_x_intervals_at_y(component, layer, y_cut)
        if not intervals and not _layer_has_any_polygon(component, layer):
            intervals = [(-xz_span / 2, xz_span / 2)]
        layer_data.append(
            {
                "layer": layer,
                "medium": _meep_medium(mat_data),
                "z_lo": layer.zmin - z_center,
                "z_hi": layer.zmax - z_center,
                "intervals": intervals,
                "sidewall_angle": getattr(layer, "sidewall_angle", 0.0),
            }
        )

    for ld in layer_data:
        below_intervals: list[tuple[float, float]] = []
        for od in layer_data:
            if od is ld:
                continue
            if od["z_hi"] <= ld["z_lo"] + 1e-12:
                below_intervals.extend(od["intervals"])

        for x0, x1 in ld["intervals"]:
            x_center = (x0 + x1) / 2.0
            x_size = x1 - x0
            if x_size <= 0:
                continue
            z_lo = ld["z_lo"]
            z_hi = ld["z_hi"]
            block_z_size = z_hi - z_lo
            sw_angle_deg = ld.get("sidewall_angle", 0.0)
            if block_z_size > 0:
                if sw_angle_deg:
                    sw_rad = math.radians(sw_angle_deg)
                    n_sub = max(2, math.ceil(block_z_size * resolution))
                    sub_h = block_z_size / n_sub
                    for k in range(n_sub):
                        frac = (k + 0.5) / n_sub
                        offset = frac * block_z_size * math.tan(sw_rad)
                        x0_k = x0 + offset
                        x1_k = x1 - offset
                        x_size_k = x1_k - x0_k
                        if x_size_k <= 0:
                            continue
                        sub_z_center = z_lo + (k + 0.5) * sub_h
                        block = mp.Block(
                            size=mp.Vector3(x_size_k, mp.inf, sub_h),
                            center=mp.Vector3((x0_k + x1_k) / 2.0, 0.0, sub_z_center),
                            material=ld["medium"],
                        )
                        geometry.append(block)
                else:
                    block_z_center = (z_lo + z_hi) / 2.0
                    block = mp.Block(
                        size=mp.Vector3(x_size, mp.inf, block_z_size),
                        center=mp.Vector3(x_center, 0.0, block_z_center),
                        material=ld["medium"],
                    )
                    geometry.append(block)

            bottom_free = _subtract_intervals((x0, x1), below_intervals)
            for bx0, bx1 in bottom_free:
                bx_center = (bx0 + bx1) / 2.0
                bx_size = bx1 - bx0
                if bx_size <= 0:
                    continue
                ext_z_lo = -z_span / 2.0
                ext_z_hi = ld["z_lo"]
                ext_z_size = ext_z_hi - ext_z_lo
                if ext_z_size <= 0:
                    continue
                block = mp.Block(
                    size=mp.Vector3(bx_size, mp.inf, ext_z_size),
                    center=mp.Vector3(bx_center, 0.0, (ext_z_lo + ext_z_hi) / 2.0),
                    material=ld["medium"],
                )
                geometry.append(block)

    bg = materials.get(background_material)
    default_medium = _meep_medium(bg) if bg is not None else mp.Medium(epsilon=1.0)

    sim_kwargs: dict = dict(
        cell_size=cell_size,
        geometry=geometry,
        resolution=resolution,
        default_material=default_medium,
    )
    if pml_thickness > 0:
        sim_kwargs["boundary_layers"] = [
            mp.PML(thickness=pml_thickness, direction=mp.X),
            mp.PML(thickness=pml_thickness, direction=mp.Z),
        ]
    sim = mp.Simulation(**sim_kwargs)
    return sim, cell_size


def _build_component_yz_cell(
    component: Component,
    stack: LayerStack,
    x_cut: float,
    y_span: float,
    materials: dict[str, MaterialData],
    resolution: float,
    z_margin: float | tuple[float, float] = 0.0,
    pml_thickness: float = 0.0,
    *,
    background_material: str = "air",
) -> tuple[Any, Any]:
    """Build a 2D YZ MEEP simulation cell at an *x*-slice of a component.

    For each non-air layer, the component's GDS polygons are intersected
    with the vertical line *x = x_cut*.  Each contiguous interval on *y*
    becomes a rectangular block with the layer's material and *z* extent.

    Args:
        component: GDSFactory component (must have polygons).
        stack: Resolved :class:`LayerStack`.
        x_cut: *x* coordinate of the cross-section plane (µm).
        y_span: Total *y* extent of the simulation cell (µm).
        materials: ``MaterialData`` keyed by material name.
        resolution: Pixels per µm.
        z_margin: Extra distance (µm) added below and above the stack.
            A single float adds the same margin on both sides; a
            ``(bottom, top)`` tuple adds asymmetric margins. Default 0.
        pml_thickness: PML absorber thickness in µm (default 0.0).
            Set to e.g. 1.0 to enable PML absorbing boundaries.
        background_material: Material name for the MEEP default medium
            (fills space not covered by any layer). Default ``"air"``
            (epsilon=1.0).

    Returns:
        ``(sim, cell_size)`` — initialized MEEP :class:`Simulation` object
        and ``mp.Vector3`` cell dimensions.
    """
    mp = _import_meep()

    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    if isinstance(z_margin, (tuple, list)):
        z_margin_bottom, z_margin_top = z_margin
    else:
        z_margin_bottom = z_margin_top = z_margin
    z_center = (z_min + z_max) / 2.0 + (z_margin_top - z_margin_bottom) / 2.0
    yz_span = y_span
    if pml_thickness > 0:
        y_span += 2 * pml_thickness
    y_span = round(y_span * resolution) / resolution
    z_span = (z_max - z_min) + z_margin_bottom + z_margin_top
    if pml_thickness > 0:
        z_span += 2 * pml_thickness
    z_span = round(z_span * resolution) / resolution
    cell_size = mp.Vector3(0.0, y_span, z_span)

    geometry: list[object] = []
    layer_data: list[dict] = []
    for layer in stack.layers.values():
        if layer.material == "air":
            continue
        mat_data = materials.get(layer.material)
        if mat_data is None:
            continue
        layer_thickness = layer.zmax - layer.zmin
        if layer_thickness <= 0:
            continue
        intervals = _layer_y_intervals_at_x(component, layer, x_cut)
        if not intervals and not _layer_has_any_polygon(component, layer):
            intervals = [(-yz_span / 2, yz_span / 2)]
        layer_data.append(
            {
                "layer": layer,
                "medium": _meep_medium(mat_data),
                "z_lo": layer.zmin - z_center,
                "z_hi": layer.zmax - z_center,
                "intervals": intervals,
                "sidewall_angle": getattr(layer, "sidewall_angle", 0.0),
            }
        )

    for ld in layer_data:
        below_intervals: list[tuple[float, float]] = []
        for od in layer_data:
            if od is ld:
                continue
            if od["z_hi"] <= ld["z_lo"] + 1e-12:
                below_intervals.extend(od["intervals"])

        for y0, y1 in ld["intervals"]:
            y_center = (y0 + y1) / 2.0
            y_size = y1 - y0
            if y_size <= 0:
                continue
            z_lo = ld["z_lo"]
            z_hi = ld["z_hi"]
            block_z_size = z_hi - z_lo
            sw_angle_deg = ld.get("sidewall_angle", 0.0)
            if block_z_size > 0:
                if sw_angle_deg:
                    sw_rad = math.radians(sw_angle_deg)
                    n_sub = max(2, math.ceil(block_z_size * resolution))
                    sub_h = block_z_size / n_sub
                    for k in range(n_sub):
                        frac = (k + 0.5) / n_sub
                        offset = frac * block_z_size * math.tan(sw_rad)
                        y0_k = y0 + offset
                        y1_k = y1 - offset
                        y_size_k = y1_k - y0_k
                        if y_size_k <= 0:
                            continue
                        sub_z_center = z_lo + (k + 0.5) * sub_h
                        block = mp.Block(
                            size=mp.Vector3(mp.inf, y_size_k, sub_h),
                            center=mp.Vector3(0.0, (y0_k + y1_k) / 2.0, sub_z_center),
                            material=ld["medium"],
                        )
                        geometry.append(block)
                else:
                    block_z_center = (z_lo + z_hi) / 2.0
                    block = mp.Block(
                        size=mp.Vector3(mp.inf, y_size, block_z_size),
                        center=mp.Vector3(0.0, y_center, block_z_center),
                        material=ld["medium"],
                    )
                    geometry.append(block)

            bottom_free = _subtract_intervals((y0, y1), below_intervals)
            for by0, by1 in bottom_free:
                by_center = (by0 + by1) / 2.0
                by_size = by1 - by0
                if by_size <= 0:
                    continue
                ext_z_lo = -z_span / 2.0
                ext_z_hi = ld["z_lo"]
                ext_z_size = ext_z_hi - ext_z_lo
                if ext_z_size <= 0:
                    continue
                block = mp.Block(
                    size=mp.Vector3(mp.inf, by_size, ext_z_size),
                    center=mp.Vector3(0.0, by_center, (ext_z_lo + ext_z_hi) / 2.0),
                    material=ld["medium"],
                )
                geometry.append(block)

    bg = materials.get(background_material)
    default_medium = _meep_medium(bg) if bg is not None else mp.Medium(epsilon=1.0)

    sim_kwargs: dict = dict(
        cell_size=cell_size,
        geometry=geometry,
        resolution=resolution,
        default_material=default_medium,
    )
    if pml_thickness > 0:
        sim_kwargs["boundary_layers"] = [
            mp.PML(thickness=pml_thickness, direction=mp.Y),
            mp.PML(thickness=pml_thickness, direction=mp.Z),
        ]
    sim = mp.Simulation(**sim_kwargs)
    return sim, cell_size


def _poly_to_shapely(poly):
    """Convert a polygon (numpy array, tuple of coords, or klayout) to shapely.

    Returns ``None`` if conversion fails.
    """
    import numpy as np
    from shapely.geometry import Polygon

    # numpy array from get_polygons_points (Nx2 or NxM)
    if isinstance(poly, np.ndarray):
        coords = poly.reshape(-1, 2)
        if len(coords) >= 3:
            return Polygon(coords)
        return None

    # list/tuple of coordinate pairs
    if isinstance(poly, (tuple, list)):
        return Polygon(poly)

    # klayout PolygonWithProperties
    if hasattr(poly, "to_simple_polygon"):
        sp = poly.to_simple_polygon()
        coords = [(sp.point(i).x, sp.point(i).y) for i in range(sp.num_points())]
        if len(coords) >= 3:
            return Polygon(coords)
        return None

    return None


def _layer_x_intervals_at_y(
    component: Component,
    layer: object,
    y_cut: float,
) -> list[tuple[float, float]]:
    """Return sorted *x*-intervals where *layer*'s polygons intersect *y = y_cut*.

    Args:
        component: GDSFactory component.
        layer: A :class:`Layer` with ``.gds_layer`` attribute.
        y_cut: *y* coordinate of the slicing line.

    Returns:
        Sorted list of ``(x_min, x_max)`` tuples.
    """
    from shapely.geometry import LineString, MultiLineString

    gds_layer = getattr(layer, "gds_layer", None)
    if gds_layer is None:
        return []

    try:
        polys = component.get_polygons_points(layers=(tuple(gds_layer),), merge=True)
    except Exception:
        try:
            polys = component.dup().get_polygons_points(
                layers=(tuple(gds_layer),), merge=True
            )
        except Exception:
            return []
    if not isinstance(polys, dict) or not polys:
        return []

    cut_line = LineString([(-1e6, y_cut), (1e6, y_cut)])

    intervals: list[tuple[float, float]] = []
    for polygons in polys.values():
        for poly in polygons if isinstance(polygons, list) else [polygons]:
            try:
                spoly = _poly_to_shapely(poly)
                if spoly is None:
                    continue
            except Exception:
                continue
            if spoly.is_empty:
                continue
            intersection = spoly.intersection(cut_line)
            if intersection.is_empty:
                continue
            lines: list = []
            if isinstance(intersection, LineString):
                lines = [intersection]
            elif isinstance(intersection, MultiLineString):
                lines = list(intersection.geoms)
            else:
                for g in getattr(intersection, "geoms", []):
                    if isinstance(g, LineString):
                        lines.append(g)
            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                xs = [c[0] for c in coords]
                intervals.append((min(xs), max(xs)))

    intervals.sort()
    return _merge_intervals(intervals)


def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _layer_y_intervals_at_x(
    component: Component,
    layer: object,
    x_cut: float,
) -> list[tuple[float, float]]:
    """Return sorted *y*-intervals where *layer*'s polygons intersect *x = x_cut*.

    Analogous to :func:`_layer_x_intervals_at_y` but slicing vertically
    at a fixed *x* coordinate.  Used for YZ waveguide cross-sections.

    Args:
        component: GDSFactory component.
        layer: A :class:`Layer` with ``.gds_layer`` attribute.
        x_cut: *x* coordinate of the slicing line.

    Returns:
        Sorted list of ``(y_min, y_max)`` tuples.
    """
    from shapely.geometry import LineString, MultiLineString

    gds_layer = getattr(layer, "gds_layer", None)
    if gds_layer is None:
        return []

    try:
        polys = component.get_polygons_points(layers=(tuple(gds_layer),), merge=True)
    except Exception:
        try:
            polys = component.dup().get_polygons_points(
                layers=(tuple(gds_layer),), merge=True
            )
        except Exception:
            return []
    if not isinstance(polys, dict) or not polys:
        return []

    cut_line = LineString([(x_cut, -1e6), (x_cut, 1e6)])

    intervals: list[tuple[float, float]] = []
    for polygons in polys.values():
        for poly in polygons if isinstance(polygons, list) else [polygons]:
            try:
                spoly = _poly_to_shapely(poly)
                if spoly is None:
                    continue
            except Exception:
                continue
            if spoly.is_empty:
                continue
            intersection = spoly.intersection(cut_line)
            if intersection.is_empty:
                continue
            lines: list = []
            if isinstance(intersection, LineString):
                lines = [intersection]
            elif isinstance(intersection, MultiLineString):
                lines = list(intersection.geoms)
            else:
                for g in getattr(intersection, "geoms", []):
                    if isinstance(g, LineString):
                        lines.append(g)
            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                ys = [c[1] for c in coords]
                intervals.append((min(ys), max(ys)))

    intervals.sort()
    return _merge_intervals(intervals)


# ------------------------------------------------------------------
# Eigenmode computation helpers
# ------------------------------------------------------------------


def _compute_eigenmode(
    sim: Any,
    cell_size: Any,
    wavelength: float,
    *,
    band_num: int = 1,
    parity: str = "NO_PARITY",
    kpoint: Any | None = None,
    eigensolver_tol: float = 1e-6,
    field_x_grid: np.ndarray | None = None,
    field_y_grid: np.ndarray | None = None,
    field_z_grid: np.ndarray | None = None,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    z_grid: np.ndarray | None = None,
    stack: Any | None = None,
    component: Any | None = None,
    port_or_position: str | tuple[float, float] | None = None,
    cross_section_plane: str | None = None,
) -> ModeResult:
    """Run ``sim.get_eigenmode()`` and pack the result into a :class:`ModeResult`.

    Field profiles are only extracted when both a horizontal grid
    (``field_x_grid`` or ``field_y_grid``) and ``field_z_grid`` are
    provided.  Otherwise ``result.fields`` is empty.

    For XZ cells, use ``field_x_grid`` + ``field_z_grid`` (samples at
    ``(x, 0, z)``).  For YZ cells, use ``field_y_grid`` +
    ``field_z_grid`` (samples at ``(0, y, z)``).

    Args:
        sim: Initialized MEEP :class:`Simulation` object.
        cell_size: ``mp.Vector3`` cell dimensions.
        wavelength: Free-space wavelength in µm.
        band_num: Mode band index (1 = fundamental).
        parity: Parity string (``"NO_PARITY"``, ``"EVEN_Y"``, etc.).
        kpoint: Optional ``mp.Vector3`` wavevector for the eigenmode.
            When set, MEEP can compute group velocity.
        eigensolver_tol: MPB convergence tolerance (default 1e-6).
        field_x_grid: 1D array of *x* coordinates for field sampling
            (µm, origin at cell centre).  For XZ cells.
        field_y_grid: 1D array of *y* coordinates for field sampling
            (µm, origin at cell centre).  For YZ cells.
        field_z_grid: 1D array of *z* coordinates for field sampling
            (µm, origin at cell centre).  ``None`` skips extraction.
        x_grid: User-facing X-axis grid (µm, absolute frame) stored on
            :class:`ModeResult` for downstream plotting.  ``None`` if
            unavailable.
        y_grid: User-facing Y-axis grid (µm, absolute frame) stored on
            :class:`ModeResult`.  ``None`` if unavailable.
        z_grid: User-facing Z-axis grid (µm, absolute frame) stored on
            :class:`ModeResult`.  ``None`` if unavailable.
        stack: :class:`LayerStack` stored on :class:`ModeResult` for
            index profile reconstruction.
        component: GDSFactory :class:`Component` for cross-section modes.
        port_or_position: Port name or ``(x, y)`` position for
            cross-section modes.
        cross_section_plane: ``"xz"`` or ``"yz"`` for cross-section
            modes.  ``None`` for slab modes.

    Returns:
        :class:`ModeResult` with effective index, fields (if grids
        provided), wavevectors, and group index.
    """
    mp = _import_meep()
    import numpy as np

    frequency = 1.0 / wavelength
    parity_int = _PARITY_MAP.get(parity, 0)
    if kpoint is None:
        kpoint = mp.Vector3()

    # The mode propagates along X — we place a flux plane at the centre
    # of the cell in the YZ plane.
    where = mp.Volume(
        center=mp.Vector3(0.0, 0.0, 0.0),
        size=mp.Vector3(0.0, cell_size.y, cell_size.z),
    )

    mode = sim.get_eigenmode(
        frequency,
        mp.X,  # propagation direction
        where,
        band_num=band_num,
        kpoint=kpoint,
        parity=parity_int,
        resolution=0,
        eigensolver_tol=eigensolver_tol,
    )

    if mode is None:
        raise RuntimeError(
            f"get_eigenmode returned None for band {band_num} "
            f"at wavelength {wavelength} µm"
        )

    # --- optional field profile sampling ---
    fields: dict[str, np.ndarray] = {}
    if field_z_grid is not None:
        _horizontal_grid = field_y_grid if field_y_grid is not None else field_x_grid
        if _horizontal_grid is not None:
            _use_yz = field_y_grid is not None
            for comp_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                try:
                    comp = getattr(mp, comp_name)
                    arr = np.zeros(
                        (len(field_z_grid), len(_horizontal_grid)),
                        dtype=np.complex128,
                    )
                    for iz, z in enumerate(field_z_grid):
                        for ih, h in enumerate(_horizontal_grid):
                            if _use_yz:
                                pt = mp.Vector3(0.0, float(h), float(z))
                            else:
                                pt = mp.Vector3(float(h), 0.0, float(z))
                            arr[iz, ih] = mode.amplitude(pt, comp)
                    if np.any(arr != 0):
                        fields[comp_name] = arr
                except Exception:
                    pass

    # Compute n_eff from dominant wavevector
    kdom_vec = mode.kdom
    kdom_list = [float(kdom_vec.x), float(kdom_vec.y), float(kdom_vec.z)]
    kdom_mag = float(np.linalg.norm(kdom_list))
    n_eff = kdom_mag / frequency if frequency > 0 else 0.0

    # Group index from MEEP's internal group_velocity (dw/dk at Gamma)
    n_group: float | None = None
    try:
        vg = mode.group_velocity
        if vg is not None and vg != 0:
            n_group = 1.0 / float(vg)  # c=1 in MEEP -> n_g = 1/vg
    except Exception:
        pass

    return ModeResult(
        n_eff=n_eff,
        wavelength=wavelength,
        frequency=frequency,
        fields=fields,
        kdom=kdom_list,
        n_group=n_group,
        band_num=band_num,
        parity=parity,
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        stack=stack,
        component=component,
        port_or_position=port_or_position,
        cross_section_plane=cross_section_plane,
    )


# ------------------------------------------------------------------
# Mode validation
# ------------------------------------------------------------------


def _validate_mode(
    result: ModeResult,
    material_data: dict[str, MaterialData],
    _stack: object,
    *,
    is_slab: bool = False,
) -> None:
    """Warn if the mode result appears unphysical (leaky / below cutoff).

    Checks performed:
    - **Minimum cladding index**: if ``n_eff`` is below the lowest
      non-air refractive index in the stack, the mode is likely a
      radiation / leaky mode rather than a guided mode.
    - **Slab kz cleanliness**: for a 1D slab mode the dominant
      wavevector should have negligible *z*-component (kz ~= 0).
      A large ``kz`` indicates a spurious mode picked up by MPB.
    - **Negative group index**: ``n_group < 0`` is unphysical for
      forward-propagating dielectric modes.
    """
    import logging
    import math

    logger = logging.getLogger(__name__)

    # --- minimum cladding index ---
    n_clad_min = None
    n_core_max = None
    for mat in material_data.values():
        eps = getattr(mat, "epsilon_diag", None)
        if eps is None:
            continue
        if isinstance(eps, list):
            eps = eps[0]
        if eps <= 0:
            continue
        n = math.sqrt(float(eps))
        if n_clad_min is None or n < n_clad_min:
            n_clad_min = n
        if n_core_max is None or n > n_core_max:
            n_core_max = n

    issues: list[str] = []

    if n_clad_min is not None and result.n_eff <= n_clad_min:
        issues.append(
            f"n_eff={result.n_eff:.6f} is <= minimum cladding index "
            f"(n_clad_min={n_clad_min:.6f}); mode is likely a radiation"
            f" / leaky mode, not a guided mode"
        )

    if n_core_max is not None and result.n_eff > n_core_max + 0.01:
        issues.append(
            f"n_eff={result.n_eff:.6f} exceeds maximum core index "
            f"(n_core_max={n_core_max:.6f}); result may be unphysical"
        )

    # --- slab kz purity ---
    if is_slab and len(result.kdom) >= 3:
        kz = abs(result.kdom[2])
        kx = abs(result.kdom[0])
        if kx > 0 and kz > kx * 0.05:
            issues.append(
                f"Slab mode has significant kz ({kz:.4f} vs kx={kx:.4f}); "
                f"result may be a spurious 2D parasitic mode picked up by MPB"
            )

    # --- negative group index ---
    if result.n_group is not None and result.n_group < 0:
        issues.append(
            f"n_group={result.n_group:.6f} is negative; "
            f"unphysical for forward-propagating dielectric modes"
        )

    if issues:
        summary = "; ".join(issues)
        logger.info(
            "Mode band %d physical validity concern: %s", result.band_num, summary
        )


# ------------------------------------------------------------------
# Public utilities
# ------------------------------------------------------------------


def mode_z_grid(
    stack: LayerStack,
    n_points: int,
    z_margin: float | tuple[float, float] = 0.0,
    pml_thickness: float = 0.0,
) -> np.ndarray:
    """Compute Z-axis coordinates in absolute frame matching layer ``zmin``/``zmax``.

    The returned grid has the same length as the Z (first) axis of
    ``ModeResult.fields`` arrays produced by :func:`solve_slab_mode`
    and :func:`solve_cross_section_mode`.

    Args:
        stack: :class:`LayerStack` defining the vertical material profile.
        n_points: Number of grid points (typically ``resolution * z_span``).
        z_margin: Extra distance (µm) added below and above the stack.  A
            single float adds the same margin on both sides; a ``(bottom,
            top)`` tuple adds asymmetric margins.  Must match the
            ``z_margin`` passed to the solver.  Default 0 — grid covers
            exactly the stack extent.
        pml_thickness: PML absorber thickness in µm (default 0.0).
            When set, extends the span by ``pml_thickness`` on each side
            to match the MEEP cell including PML boundaries.
            Must match the ``pml_thickness`` passed to the solver.

    Returns:
        ``np.ndarray`` of *z* coordinates in µm (absolute frame).
    """
    import numpy as np

    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    if isinstance(z_margin, (tuple, list)):
        z_margin_bottom, z_margin_top = z_margin
    else:
        z_margin_bottom = z_margin_top = z_margin

    bottom = z_min - z_margin_bottom - pml_thickness
    top = z_max + z_margin_top + pml_thickness
    span = top - bottom
    dz = span / n_points
    return np.linspace(bottom + dz / 2, top - dz / 2, n_points)


def mode_x_grid(n_points: int, x_span: float, pml_thickness: float = 0.0) -> np.ndarray:
    """Compute X-axis coordinates centred on the cell midpoint.

    The returned grid has the same length as the X (second) axis of
    ``ModeResult.fields`` arrays produced by
    :func:`solve_cross_section_mode`.

    Args:
        n_points: Number of grid points (typically ``resolution * x_span``).
        x_span: Total X-extent of the MEEP cell in µm (excluding PML).
        pml_thickness: PML absorber thickness in µm (default 0.0).
            When set, extends the span by ``2 * pml_thickness`` to match
            the MEEP cell including PML boundaries.

    Returns:
        ``np.ndarray`` of *x* coordinates in µm, origin at cell midpoint.
    """
    import numpy as np

    total_span = x_span + 2 * pml_thickness
    dx = total_span / n_points
    return np.linspace(-total_span / 2 + dx / 2, total_span / 2 - dx / 2, n_points)


def mode_y_grid(n_points: int, y_span: float, pml_thickness: float = 0.0) -> np.ndarray:
    """Compute Y-axis coordinates centred on the cell midpoint.

    The returned grid has the same length as the Y (second) axis of
    ``ModeResult.fields`` arrays produced when using a YZ cross-section
    in :func:`solve_cross_section_mode`.

    Args:
        n_points: Number of grid points (typically ``resolution * y_span``).
        y_span: Total Y-extent of the MEEP cell in µm (excluding PML).
        pml_thickness: PML absorber thickness in µm (default 0.0).
            When set, extends the span by ``2 * pml_thickness`` to match
            the MEEP cell including PML boundaries.

    Returns:
        ``np.ndarray`` of *y* coordinates in µm, origin at cell midpoint.
    """
    import numpy as np

    total_span = y_span + 2 * pml_thickness
    dy = total_span / n_points
    return np.linspace(-total_span / 2 + dy / 2, total_span / 2 - dy / 2, n_points)


def refractive_index_profile(
    stack: LayerStack,
    wavelength: float,
    *,
    z_grid: np.ndarray,
    y_grid: np.ndarray | None = None,
    x_grid: np.ndarray | None = None,
    component: Component | None = None,
    port: str | None = None,
    background_material: str = "air",
) -> np.ndarray:
    """Compute piecewise-constant refractive index at a wavelength.

    **1D** (default) — profile along *z* only, returns ``(nz,)``::

        n_prof = refractive_index_profile(stack, wavelength, z_grid=z_grid)

    **2D YZ cross-section** — returns ``(nz, ny)``::

        n_yz = refractive_index_profile(
            stack,
            wavelength,
            z_grid=z_grid,
            y_grid=y_grid,
            component=component,
            port="o1",
        )

    **2D XZ cross-section** — returns ``(nz, nx)``::

        n_xz = refractive_index_profile(
            stack,
            wavelength,
            z_grid=z_grid,
            x_grid=x_grid,
            component=component,
            port="o1",
        )

    The cross-section plane is auto-detected from the port orientation
    (same logic as :func:`solve_cross_section_mode`):
    - Port at 0° / 180°: YZ plane, ``x_cut`` from port centre *x*.
    - Port at 90° / 270°: XZ plane, ``y_cut`` from port centre *y*.

    Layers with no GDS polygons are treated as full-span background layers
    (same fallback as ``_build_component_yz_cell``).

    Args:
        stack: :class:`LayerStack` defining the vertical material profile.
        wavelength: Free-space wavelength in µm for material evaluation.
        z_grid: 1D array of *z* coordinates (absolute frame, matching
            ``mode_z_grid`` output).
        y_grid: 1D array of *y* coordinates (absolute frame).  Use with YZ
            cross-sections.  Mutually exclusive with ``x_grid``.
        x_grid: 1D array of *x* coordinates (absolute frame).  Use with XZ
            cross-sections.  Mutually exclusive with ``y_grid``.
        component: GDSFactory component (required for 2D mode together
            with ``port``).
        port: Port name in *component*.  Auto-derives the slice coordinate
            and cross-section plane.

    Returns:
        ``np.ndarray`` — 1D ``(nz,)`` or 2D ``(nz, ny)`` / ``(nz, nx)``.
    """
    import numpy as np

    used_materials: set[str] = {layer.material for layer in stack.layers.values()}
    used_materials.discard("air")
    if background_material != "air":
        used_materials.add(background_material)
    material_data = _resolve_materials_cached(
        used_materials,
        overrides={},
        wavelength_um=wavelength,
    )
    bg_mat = material_data.get(background_material)
    n_bg = 1.0
    if bg_mat is not None and bg_mat.epsilon_diag is not None:
        eps_bg = bg_mat.epsilon_diag
        if isinstance(eps_bg, list):
            eps_bg = eps_bg[0]
        if eps_bg > 0:
            n_bg = np.sqrt(eps_bg)

    if component is None or port is None:
        if y_grid is not None and x_grid is None and component is not None:
            raise ValueError(
                "port is required for 2D refractive_index_profile "
                "when component is given"
            )
        if x_grid is not None:
            raise ValueError(
                "port is required for 2D refractive_index_profile when x_grid is given"
            )
        n_profile = np.full_like(z_grid, n_bg)
        for layer in stack.layers.values():
            if layer.material == "air":
                continue
            mat = material_data.get(layer.material)
            if mat is None or mat.epsilon_diag is None:
                continue
            eps = mat.epsilon_diag
            if isinstance(eps, list):
                eps = eps[0]
            if eps <= 0:
                continue
            mask = (z_grid >= layer.zmin) & (z_grid < layer.zmax)
            n_profile[mask] = np.sqrt(eps)
        return n_profile

    # --- 2D path ---
    port_info = None
    for p in component.ports:
        if p.name == port:
            port_info = p
            break
    if port_info is None:
        available = [p.name for p in component.ports]
        raise ValueError(
            f"Port '{port}' not found in component. Available: {available}"
        )

    port_ori = float(getattr(port_info, "orientation", 0))
    use_yz = port_ori % 180 == 0

    if use_yz:
        if y_grid is None:
            raise ValueError(
                "y_grid is required for YZ cross-section (port orientation "
                f"{port_ori}°). Pass y_grid as returned by mode_y_grid()."
            )
        if x_grid is not None:
            raise ValueError(
                "x_grid is not valid for YZ cross-section (port orientation "
                f"{port_ori}°). Use y_grid instead."
            )
        x_cut = float(port_info.center[0])
        horizontal_grid = y_grid
        interval_func = _layer_y_intervals_at_x
        cut_coord = x_cut
    else:
        if x_grid is None:
            raise ValueError(
                "x_grid is required for XZ cross-section (port orientation "
                f"{port_ori}°). Pass x_grid as returned by mode_x_grid()."
            )
        if y_grid is not None:
            raise ValueError(
                "y_grid is not valid for XZ cross-section (port orientation "
                f"{port_ori}°). Use x_grid instead."
            )
        y_cut = float(port_info.center[1])
        horizontal_grid = x_grid
        interval_func = _layer_x_intervals_at_y
        cut_coord = y_cut

    nh = len(horizontal_grid)
    nz = len(z_grid)
    n_profile = np.full((nz, nh), n_bg)
    set_by_layer = np.zeros((nz, nh), dtype=bool)

    layer_h_masks: dict[tuple, np.ndarray] = {}
    for layer in stack.layers.values():
        if layer.material == "air":
            continue
        gds_layer = getattr(layer, "gds_layer", None)
        if gds_layer is None:
            continue
        intervals = interval_func(component, layer, cut_coord)
        if not intervals and not _layer_has_any_polygon(component, layer):
            intervals = [(-np.inf, np.inf)]
        mask = np.zeros(nh, dtype=bool)
        for lo, hi in intervals:
            mask |= (horizontal_grid >= lo) & (horizontal_grid <= hi)
        layer_h_masks[gds_layer] = mask

    for layer in stack.layers.values():
        if layer.material == "air":
            continue
        mat = material_data.get(layer.material)
        if mat is None or mat.epsilon_diag is None:
            continue
        eps = mat.epsilon_diag
        if isinstance(eps, list):
            eps = eps[0]
        if eps <= 0:
            continue
        gds_layer = getattr(layer, "gds_layer", None)
        h_mask = layer_h_masks.get(gds_layer)
        if h_mask is None:
            continue
        z_mask = (z_grid >= layer.zmin) & (z_grid < layer.zmax)
        n_profile[np.ix_(z_mask, h_mask)] = np.sqrt(eps)
        set_by_layer[np.ix_(z_mask, h_mask)] = True

    for ih in range(nh):
        col = n_profile[:, ih]
        col_set = set_by_layer[:, ih]
        if not col_set.any():
            continue
        first_set = int(np.where(col_set)[0][0])
        if first_set > 0:
            fill_mask = ~col_set[:first_set]
            col[:first_set][fill_mask] = col[first_set]

    return n_profile


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def solve_slab_mode(
    stack: LayerStack,
    wavelength: float,
    *,
    band_num: int = 1,
    parity: str = "NO_PARITY",
    resolution: float = 32,
    z_margin: float | tuple[float, float] = 0.0,
    pml_thickness: float = 0.0,
    eigensolver_tol: float = 1e-6,
    field_z_grid: np.ndarray | None = None,
    background_material: str = "air",
) -> ModeResult:
    """Solve for the 1D slab mode of a uniform layer stack.

    The slab is assumed infinite in *x* and *y*; the mode propagates along
    *x* with field variation only along *z*.  Uses a 2D XZ MEEP simulation
    cell.

    By default no field profiles are extracted.  Pass a ``field_z_grid``
    to sample the eigenmode fields along *z* at ``x=0`` (a single column).

    Args:
        stack: Layer stack defining the vertical material profile.
        wavelength: Free-space wavelength in µm.
        band_num: Mode band index (1 = fundamental TE/TM slab mode).
        parity: Parity of the mode (``"NO_PARITY"``, ``"EVEN_Y"``, etc.).
        resolution: Pixels per µm (default 32).
        z_margin: Extra distance (µm) added below and above the stack.  A
            single float adds the same margin on both sides; a ``(bottom,
            top)`` tuple adds asymmetric margins.  Default 0 — cell
            exactly spans the stack extent.
        pml_thickness: PML absorber thickness in µm (default 0.0).
            Set to e.g. 1.0 to enable PML absorbing boundaries.
        eigensolver_tol: MPB convergence tolerance (default 1e-6).
            Relax to 1e-5 for faster but less accurate solves.
        field_z_grid: 1D array of *z* coordinates (µm, absolute frame
            matching ``mode_z_grid``) at which to sample
            ``mode.amplitude()``.  The *x*-coordinate is fixed at 0.
            When ``None`` (default) no field extraction is performed
            and ``result.fields`` is empty.

    Returns:
        :class:`ModeResult` with effective index, field profiles (if
        ``field_z_grid`` provided), and group index.
    """
    _import_meep()
    import numpy as np

    if stack is None:
        raise ValueError("stack must be a LayerStack, got None")

    used_materials: set[str] = set()
    for layer in stack.layers.values():
        used_materials.add(layer.material)
    for diel in stack.dielectrics:
        used_materials.add(diel["material"])
    if background_material != "air":
        used_materials.add(background_material)

    material_data = _resolve_materials_cached(
        used_materials,
        overrides={},
        wavelength_um=wavelength,
    )

    _z_margin_cell = z_margin
    sim, cell_size = _build_slab_xz_cell(
        stack,
        material_data,
        resolution,
        z_margin=_z_margin_cell,
        pml_thickness=pml_thickness,
        background_material=background_material,
    )
    sim.init_sim()

    field_x_grid: np.ndarray | None = None
    field_z_grid_meep: np.ndarray | None = None
    if field_z_grid is not None:
        field_x_grid = np.array([0.0])
        # Compute z_cell_center same way as _build_slab_xz_cell
        z_min = min(layer.zmin for layer in stack.layers.values())
        z_max = max(layer.zmax for layer in stack.layers.values())
        if isinstance(z_margin, (tuple, list)):
            zm_bottom, zm_top = z_margin
        else:
            zm_bottom = zm_top = z_margin
        z_cell_center = (z_min + z_max) / 2.0 + (zm_top - zm_bottom) / 2.0
        field_z_grid_meep = field_z_grid - z_cell_center
    else:
        field_z_grid_meep = None

    try:
        result = _compute_eigenmode(
            sim,
            cell_size,
            wavelength,
            band_num=band_num,
            parity=parity,
            eigensolver_tol=eigensolver_tol,
            field_x_grid=field_x_grid,
            field_z_grid=field_z_grid_meep,
            z_grid=field_z_grid,
            stack=stack,
        )
    finally:
        sim.reset_meep()

    _validate_mode(result, material_data, stack, is_slab=True)
    return result


def solve_cross_section_mode(
    component: Component,
    stack: LayerStack,
    *,
    port: str | None = None,
    position: tuple[float, float] | None = None,
    x_span: float | None = None,
    y_span: float | None = None,
    cross_section_plane: str = "auto",
    z_margin: float | tuple[float, float] = 0.0,
    pml_thickness: float = 0.0,
    eigensolver_tol: float = 1e-6,
    wavelength: float,
    band_num: int = 1,
    parity: str = "NO_PARITY",
    resolution: float = 32,
    field_x_grid: np.ndarray | None = None,
    field_y_grid: np.ndarray | None = None,
    field_z_grid: np.ndarray | None = None,
    background_material: str = "air",
) -> ModeResult:
    """Solve for the eigenmode of a 2D waveguide cross-section.

    Takes a slice through the component at a specified port or position
    and builds a 2D MEEP cell.  The cross-section plane is determined by
    the port orientation (auto-detected by default):

    - **YZ plane**: port orientation 0° or 180° (waveguide along X).
      Slice at *x = port_center[0]*, build YZ cell.
    - **XZ plane**: port orientation 90° or 270° (waveguide along Y).
      Slice at *y = port_center[1]*, build XZ cell.

    By default no field profiles are extracted.  Pass horizontal and
    vertical grids to sample the eigenmode fields on a 2D grid.

    One of ``port`` or ``position`` must be given:
     - ``port="o1"`` — auto-extracts the cross-section at the port
       centre.  Requires ``y_span`` (YZ) or ``x_span`` (XZ) to be
       passed explicitly.
     - ``position=(x, y)`` — requires ``x_span`` for XZ or ``y_span`` for
       YZ, and ``cross_section_plane`` must be set explicitly.

    Args:
        component: GDSFactory component with ports and polygons.
        stack: Layer stack (already resolved by caller or via
            :meth:`Simulation._resolve_stack_and_materials`).
        port: Port name to auto-extract slice location and span.
        position: Arbitrary ``(x, y)`` point — only one coordinate is
            used depending on the cross-section plane.
        x_span: Total *x*-extent of the cell in µm. Required for XZ
            cross-sections when using ``position``.
        y_span: Total *y*-extent of the cell in µm. Used for YZ
            cross-sections; auto-derived from port width.
        cross_section_plane: ``"auto"`` (default), ``"xz"``, or ``"yz"``.
            ``"auto"`` detects from port orientation.
        z_margin: Extra distance (µm) added below and above the stack.  A
            single float adds the same margin on both sides; a ``(bottom,
            top)`` tuple adds asymmetric margins.  Default 0 — cell
            exactly spans the stack extent.
        pml_thickness: PML absorber thickness in µm (default 0.0).
            Set to e.g. 1.0 to enable PML absorbing boundaries.
        eigensolver_tol: MPB convergence tolerance (default 1e-6).
            Relax to 1e-5 for faster but less accurate solves.
        wavelength: Free-space wavelength in µm.
            band_num: Mode band index (1 = fundamental).
        parity: Parity of the mode.
        resolution: Pixels per µm (default 32).
        field_x_grid: 1D array of *x* coordinates for field sampling
            (µm, absolute frame).  For XZ cells.  ``None`` skips.
        field_y_grid: 1D array of *y* coordinates for field sampling
            (µm, absolute frame).  For YZ cells.  ``None`` skips.
        field_z_grid: 1D array of *z* coordinates for field sampling
            (µm, absolute frame matching ``mode_z_grid``).  ``None``
            skips extraction.
        background_material: Material name for the MEEP default medium
            (fills space not covered by any layer). Default ``"air"``
            (epsilon=1.0).

    Returns:
        :class:`ModeResult` with effective index, field profiles (if
        grids provided), and group index.
    """
    _import_meep()

    if stack is None:
        raise ValueError("stack must be a LayerStack, got None")
    if component is None:
        raise ValueError("component must be a Component, got None")

    _use_yz: bool
    _span: float

    if port is not None:
        port_info = None
        for p in component.ports:
            if p.name == port:
                port_info = p
                break
        if port_info is None:
            available = [p.name for p in component.ports]
            raise ValueError(
                f"Port '{port}' not found in component. Available: {available}"
            )
        port_ori = float(getattr(port_info, "orientation", 0))

        if cross_section_plane == "auto":
            _use_yz = port_ori % 180 == 0
        elif cross_section_plane == "yz":
            _use_yz = True
        elif cross_section_plane == "xz":
            _use_yz = False
        else:
            raise ValueError(
                f"cross_section_plane must be 'auto', 'xz', or 'yz', "
                f"got {cross_section_plane!r}"
            )

        if _use_yz:
            x_cut = float(port_info.center[0])
            if y_span is None:
                raise ValueError(
                    "y_span is required for YZ cross-section with port. "
                    "Pass y_span explicitly."
                )
            _span = y_span
        else:
            y_cut = float(port_info.center[1])
            if x_span is None:
                raise ValueError(
                    "x_span is required for XZ cross-section with port. "
                    "Pass x_span explicitly."
                )
            _span = x_span
    elif position is not None:
        if cross_section_plane == "auto":
            _use_yz = False  # default to XZ for position mode backward compat
        else:
            _use_yz = cross_section_plane == "yz"
        if _use_yz:
            if y_span is None:
                raise ValueError(
                    "y_span is required for YZ cross-section with position"
                )
            x_cut = float(position[0])
            _span = y_span
        else:
            if x_span is None:
                raise ValueError(
                    "x_span is required for XZ cross-section with position"
                )
            y_cut = float(position[1])
            _span = x_span
    else:
        raise ValueError("Either port or position must be specified")

    used_materials: set[str] = set()
    for layer in stack.layers.values():
        used_materials.add(layer.material)
    for diel in stack.dielectrics:
        used_materials.add(diel["material"])
    if background_material != "air":
        used_materials.add(background_material)

    material_data = _resolve_materials_cached(
        used_materials,
        overrides={},
        wavelength_um=wavelength,
    )

    if _use_yz:
        sim, cell_size = _build_component_yz_cell(
            component,
            stack,
            x_cut,
            _span,
            material_data,
            resolution,
            z_margin=z_margin,
            pml_thickness=pml_thickness,
            background_material=background_material,
        )
    else:
        sim, cell_size = _build_component_xz_cell(
            component,
            stack,
            y_cut,
            _span,
            material_data,
            resolution,
            z_margin=z_margin,
            pml_thickness=pml_thickness,
            background_material=background_material,
        )
    sim.init_sim()

    # Convert field_z_grid from absolute to MEEP frame
    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    if isinstance(z_margin, (tuple, list)):
        zm_bottom, zm_top = z_margin
    else:
        zm_bottom = zm_top = z_margin
    z_cell_center = (z_min + z_max) / 2.0 + (zm_top - zm_bottom) / 2.0
    field_z_grid_meep = (
        field_z_grid - z_cell_center if field_z_grid is not None else None
    )

    try:
        result = _compute_eigenmode(
            sim,
            cell_size,
            wavelength,
            band_num=band_num,
            parity=parity,
            eigensolver_tol=eigensolver_tol,
            field_x_grid=field_x_grid,
            field_y_grid=field_y_grid,
            field_z_grid=field_z_grid_meep,
            x_grid=field_x_grid,
            y_grid=field_y_grid,
            z_grid=field_z_grid,
            stack=stack,
            component=component,
            port_or_position=port if port is not None else position,
            cross_section_plane="yz" if _use_yz else "xz",
        )
    finally:
        sim.reset_meep()

    _validate_mode(result, material_data, stack, is_slab=False)
    return result


# ------------------------------------------------------------------
# Batch slab mode solver — reuses simulation across bands
# ------------------------------------------------------------------


def solve_slab_modes(
    stack: LayerStack,
    wavelength: float,
    *,
    band_nums: list[int] | None = None,
    parity: str = "NO_PARITY",
    resolution: float = 32,
    z_margin: float | tuple[float, float] = 0.0,
    pml_thickness: float = 0.0,
    background_material: str = "air",
    eigensolver_tol: float = 1e-6,
    field_z_grid: np.ndarray | None = None,
) -> dict[int, ModeResult]:
    """Solve multiple slab mode bands with a single simulation setup.

    Builds the MEEP cell once, then loops over *band_nums* reusing the
    same grid. Material resolution, geometry, and PML are computed once.

    Args:
        stack: Layer stack defining the vertical material profile.
        wavelength: Free-space wavelength in µm.
        band_nums: Mode band indices (default ``[1]`` — fundamental only).
        parity: Parity of the modes.
        resolution: Pixels per µm (default 32).
        z_margin: Extra distance (µm) added below and above the stack.
        pml_thickness: PML absorber thickness in µm (default 0.0).
        eigensolver_tol: MPB convergence tolerance (default 1e-6).
        field_z_grid: 1D array of *z* coordinates for field sampling.
            ``None`` skips extraction for all bands.

    Returns:
        Dict mapping ``band_num`` to :class:`ModeResult`. Bands that fail
        to converge are omitted from the dict.
    """
    _import_meep()
    import numpy as np

    if band_nums is None:
        band_nums = [1]

    if stack is None:
        raise ValueError("stack must be a LayerStack, got None")

    used_materials: set[str] = set()
    for layer in stack.layers.values():
        used_materials.add(layer.material)
    for diel in stack.dielectrics:
        used_materials.add(diel["material"])
    if background_material != "air":
        used_materials.add(background_material)

    material_data = _resolve_materials_cached(
        used_materials,
        overrides={},
        wavelength_um=wavelength,
    )

    _z_margin_cell = z_margin
    sim, cell_size = _build_slab_xz_cell(
        stack,
        material_data,
        resolution,
        z_margin=_z_margin_cell,
        pml_thickness=pml_thickness,
        background_material=background_material,
    )
    sim.init_sim()

    field_x_grid: np.ndarray | None = None
    field_z_grid_meep: np.ndarray | None = None
    if field_z_grid is not None:
        field_x_grid = np.array([0.0])
        z_min = min(layer.zmin for layer in stack.layers.values())
        z_max = max(layer.zmax for layer in stack.layers.values())
        if isinstance(z_margin, (tuple, list)):
            zm_bottom, zm_top = z_margin
        else:
            zm_bottom = zm_top = z_margin
        z_cell_center = (z_min + z_max) / 2.0 + (zm_top - zm_bottom) / 2.0
        field_z_grid_meep = field_z_grid - z_cell_center
    else:
        field_z_grid_meep = None

    results: dict[int, ModeResult] = {}
    try:
        for band_num in band_nums:
            try:
                result = _compute_eigenmode(
                    sim,
                    cell_size,
                    wavelength,
                    band_num=band_num,
                    parity=parity,
                    eigensolver_tol=eigensolver_tol,
                    field_x_grid=field_x_grid,
                    field_z_grid=field_z_grid_meep,
                    z_grid=field_z_grid,
                    stack=stack,
                )
                _validate_mode(result, material_data, stack, is_slab=True)
                results[band_num] = result
            except RuntimeError:
                pass
    finally:
        sim.reset_meep()

    return results


# ------------------------------------------------------------------
# Wavelength-sweep slab mode solver — reuses geometry across lambda
# ------------------------------------------------------------------


def solve_slab_wavelength_sweep(
    stack: LayerStack,
    wavelengths: list[float],
    *,
    band_num: int = 1,
    parity: str = "NO_PARITY",
    resolution: float = 32,
    z_margin: float | tuple[float, float] = 0.0,
    pml_thickness: float = 0.0,
    eigensolver_tol: float = 1e-6,
    field_z_grid: np.ndarray | None = None,
    background_material: str = "air",
) -> dict[float, ModeResult]:
    """Solve slab mode at multiple wavelengths reusing cell geometry.

    The cell size, grid, PML, and layer positions are computed once.
    Per-wavelength material epsilon values are resolved and a fresh
    simulation is built (required for correct dispersion), but the
    geometry coordinates and setup overhead are shared.

    This is 2-3x faster than calling :func:`solve_slab_mode` in a loop
    for moderate wavelength counts (5-30).

    Args:
        stack: Layer stack defining the vertical material profile.
        wavelengths: List of free-space wavelengths in µm.
        band_num: Mode band index (1 = fundamental).
        parity: Parity of the mode.
        resolution: Pixels per µm (default 32).
        z_margin: Extra distance (µm) below and above the stack.
        pml_thickness: PML absorber thickness in µm (default 0.0).
        eigensolver_tol: MPB convergence tolerance (default 1e-6).
        field_z_grid: 1D array of *z* coordinates for field sampling.
            ``None`` skips field extraction for all wavelengths.

    Returns:
        Dict mapping wavelength to :class:`ModeResult`.
    """
    import logging

    import numpy as np

    mp = _import_meep()
    logger = logging.getLogger(__name__)

    if stack is None:
        raise ValueError("stack must be a LayerStack, got None")

    # --- pre-compute geometry meta-data (wavelength-independent) ---
    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    if isinstance(z_margin, (tuple, list)):
        z_margin_bottom, z_margin_top = z_margin
    else:
        z_margin_bottom = z_margin_top = z_margin
    z_center = (z_min + z_max) / 2.0 + (z_margin_top - z_margin_bottom) / 2.0
    x_span = 2.0 / resolution
    z_span = (z_max - z_min) + z_margin_bottom + z_margin_top
    if pml_thickness > 0:
        x_span += 2 * pml_thickness
        z_span += 2 * pml_thickness
    x_span = round(x_span * resolution) / resolution
    z_span = round(z_span * resolution) / resolution
    cell_size = mp.Vector3(x_span, 0.0, z_span)

    # Layer geometric descriptors (wavelength-independent)
    _layer_specs: list[dict] = []
    for layer in stack.layers.values():
        if layer.material == "air":
            continue
        z_size = layer.zmax - layer.zmin
        if z_size <= 0:
            continue
        _layer_specs.append(
            {
                "name": layer.material,
                "center_z": layer.zmin + z_size / 2.0 - z_center,
                "z_size": z_size,
            }
        )

    used_materials: set[str] = {s["name"] for s in _layer_specs}
    if background_material != "air":
        used_materials.add(background_material)

    field_x_grid: np.ndarray | None = None
    field_z_grid_meep: np.ndarray | None = None
    if field_z_grid is not None:
        field_x_grid = np.array([0.0])
        field_z_grid_meep = field_z_grid - z_center
    else:
        field_z_grid_meep = None

    results: dict[float, ModeResult] = {}

    for wl in wavelengths:
        material_data = _resolve_materials_cached(
            used_materials,
            overrides={},
            wavelength_um=wl,
        )

        geometry: list[object] = []
        for spec in _layer_specs:
            mat_data = material_data.get(spec["name"])
            if mat_data is None:
                continue
            medium = _meep_medium(mat_data)
            geometry.append(
                mp.Block(
                    size=mp.Vector3(x_span, mp.inf, spec["z_size"]),
                    center=mp.Vector3(0.0, 0.0, spec["center_z"]),
                    material=medium,
                )
            )

        bg = material_data.get(background_material)
        default_medium = _meep_medium(bg) if bg is not None else mp.Medium(epsilon=1.0)

        sim_kwargs: dict = dict(
            cell_size=cell_size,
            geometry=geometry,
            resolution=resolution,
            default_material=default_medium,
        )
        if pml_thickness > 0:
            sim_kwargs["boundary_layers"] = [
                mp.PML(thickness=pml_thickness, direction=mp.X),
                mp.PML(thickness=pml_thickness, direction=mp.Z),
            ]

        sim = mp.Simulation(**sim_kwargs)
        sim.init_sim()

        try:
            result = _compute_eigenmode(
                sim,
                cell_size,
                wl,
                band_num=band_num,
                parity=parity,
                eigensolver_tol=eigensolver_tol,
                field_x_grid=field_x_grid,
                field_z_grid=field_z_grid_meep,
                z_grid=field_z_grid,
                stack=stack,
            )
            _validate_mode(result, material_data, stack, is_slab=True)
            results[wl] = result
        except RuntimeError:
            logger.warning(
                "solve_slab_wavelength_sweep: band %d failed at lambda=%.4f um",
                band_num,
                wl,
            )
        finally:
            sim.reset_meep()

    return results
