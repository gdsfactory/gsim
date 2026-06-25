"""Standalone MEEP eigenmode solver for 1D slab and 2D waveguide cross-sections.

Provides two public entry points:

- :func:`solve_slab_mode` — 1D slab mode from a layer stack
- :func:`solve_cross_section_mode` — 2D waveguide cross-section at a port or position

Both use ``sim.get_eigenmode()`` internally and return a :class:`ModeResult`.
MEEP is an optional dependency imported lazily; see :func:`_import_meep`.
"""

from __future__ import annotations

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


# ------------------------------------------------------------------
# Slab (1D) geometry builder
# ------------------------------------------------------------------


def _build_slab_xz_cell(
    stack: LayerStack,
    materials: dict[str, MaterialData],
    resolution: float,
) -> tuple[Any, Any]:
    """Build a 2D XZ MEEP simulation cell for a 1D slab (uniform layers).

    Each non-air layer becomes a rectangular block spanning the full
    *x*-width. The cell is centred vertically on the stack midpoint.

    Args:
        stack: Resolved :class:`LayerStack`.
        materials: ``MaterialData`` keyed by material name.
        resolution: Pixels per µm.

    Returns:
        ``(sim, cell_size)`` — initialized MEEP :class:`Simulation` object
        and ``mp.Vector3`` cell dimensions.
    """
    mp = _import_meep()

    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    z_center = (z_min + z_max) / 2.0

    # For slab mode solving, the x-extent is arbitrary — just needs to be
    # wide enough to avoid edge effects.
    x_span = 4.0
    z_span = z_max - z_min
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
        block = mp.Block(
            size=mp.Vector3(x_span, mp.inf, z_size),
            center=mp.Vector3(0.0, 0.0, layer.zmin + z_size / 2.0 - z_center),
            material=medium,
        )
        geometry.append(block)

    sim_kwargs: dict = dict(
        cell_size=cell_size,
        geometry=geometry,
        resolution=resolution,
        default_material=mp.Medium(epsilon=1.0),
    )
    sim = mp.Simulation(**sim_kwargs)
    return sim, cell_size


# ------------------------------------------------------------------
# Cross-section (2D) geometry builder
# ------------------------------------------------------------------


def _build_component_xz_cell(
    component: Component,
    stack: LayerStack,
    y_cut: float,
    x_span: float,
    materials: dict[str, MaterialData],
    resolution: float,
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

    Returns:
        ``(sim, cell_size)`` — initialized MEEP :class:`Simulation` object
        and ``mp.Vector3`` cell dimensions.
    """
    mp = _import_meep()

    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    z_center = (z_min + z_max) / 2.0
    # Snap to integer pixel counts to avoid meep grid-volume warnings.
    x_span = round(x_span * resolution) / resolution
    z_span = round((z_max - z_min) * resolution) / resolution
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

        intervals = _layer_x_intervals_at_y(component, layer, y_cut)
        for x0, x1 in intervals:
            x_center = (x0 + x1) / 2.0
            x_size = x1 - x0
            if x_size <= 0:
                continue
            block = mp.Block(
                size=mp.Vector3(x_size, mp.inf, z_size),
                center=mp.Vector3(x_center, 0.0, layer.zmin + z_size / 2.0 - z_center),
                material=medium,
            )
            geometry.append(block)

    sim_kwargs: dict = dict(
        cell_size=cell_size,
        geometry=geometry,
        resolution=resolution,
        default_material=mp.Medium(epsilon=1.0),
    )
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
    """Merge overlapping *x*-intervals."""
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
) -> ModeResult:
    """Run ``sim.get_eigenmode()`` and pack the result into a :class:`ModeResult`.

    Args:
        sim: Initialized MEEP :class:`Simulation` object.
        cell_size: ``mp.Vector3`` cell dimensions.
        wavelength: Free-space wavelength in µm.
        band_num: Mode band index (1 = fundamental).
        parity: Parity string (``"NO_PARITY"``, ``"EVEN_Y"``, etc.).
        kpoint: Optional ``mp.Vector3`` wavevector for the eigenmode.
            When set, MEEP can compute group velocity.

    Returns:
        :class:`ModeResult` with effective index, fields, wavevectors.
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
        eigensolver_tol=1e-12,
    )

    if mode is None:
        raise RuntimeError(
            f"get_eigenmode returned None for band {band_num} "
            f"at wavelength {wavelength} µm"
        )

    # Extract full 2D (X, Z) mode profile at y=0 by sampling
    # ``mode.amplitude()`` on a grid spanning the cell.
    fields: dict[str, np.ndarray] = {}
    nx = max(round(cell_size.x * sim.resolution), 1)
    nz = max(round(cell_size.z * sim.resolution), 1)
    dx = cell_size.x / nx
    dz = cell_size.z / nz
    x_vals = np.linspace(-cell_size.x / 2 + dx / 2, cell_size.x / 2 - dx / 2, nx)
    z_vals = np.linspace(-cell_size.z / 2 + dz / 2, cell_size.z / 2 - dz / 2, nz)
    for comp_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        try:
            comp = getattr(mp, comp_name)
            arr = np.zeros((nz, nx), dtype=np.complex128)
            for iz, z in enumerate(z_vals):
                for ix, x in enumerate(x_vals):
                    pt = mp.Vector3(float(x), 0.0, float(z))
                    arr[iz, ix] = mode.amplitude(pt, comp)
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


def mode_z_grid(stack: LayerStack, n_points: int) -> np.ndarray:
    """Compute Z-axis coordinates centred on the layer stack midpoint.

    The returned grid has the same length as the Z (first) axis of
    ``ModeResult.fields`` arrays produced by :func:`solve_slab_mode`
    and :func:`solve_cross_section_mode`.

    Args:
        stack: :class:`LayerStack` defining the vertical material profile.
        n_points: Number of grid points (typically ``resolution * z_span``).

    Returns:
        ``np.ndarray`` of *z* coordinates in µm, origin at stack midpoint.
    """
    import numpy as np

    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    span = z_max - z_min
    dz = span / n_points
    return np.linspace(-span / 2 + dz / 2, span / 2 - dz / 2, n_points)


def mode_x_grid(n_points: int, x_span: float) -> np.ndarray:
    """Compute X-axis coordinates centred on the cell midpoint.

    The returned grid has the same length as the X (second) axis of
    ``ModeResult.fields`` arrays produced by
    :func:`solve_cross_section_mode`.

    Args:
        n_points: Number of grid points (typically ``resolution * x_span``).
        x_span: Total X-extent of the MEEP cell in µm.

    Returns:
        ``np.ndarray`` of *x* coordinates in µm, origin at cell midpoint.
    """
    import numpy as np

    dx = x_span / n_points
    return np.linspace(-x_span / 2 + dx / 2, x_span / 2 - dx / 2, n_points)


def refractive_index_profile(
    stack: LayerStack,
    z_grid: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """Compute piecewise-constant refractive index along *z* at a wavelength.

    Uses :func:`resolve_materials` to evaluate dispersive materials at
    the target wavelength, then maps each *z*-coordinate to its enclosing
    layer and computes ``n = sqrt(epsilon_diag)``.

    Args:
        stack: :class:`LayerStack` defining the vertical material profile.
        z_grid: 1D array of *z* coordinates (MEEP frame, origin at stack
            midpoint).  Typically obtained via :func:`mode_z_grid`.
        wavelength: Free-space wavelength in µm for material evaluation.

    Returns:
        ``np.ndarray`` of refractive index values, same length as ``z_grid``.
    """
    import numpy as np

    from gsim.meep.materials import resolve_materials

    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    z_center = (z_min + z_max) / 2.0

    used_materials: set[str] = {layer.material for layer in stack.layers.values()}
    used_materials.discard("air")
    material_data = resolve_materials(
        used_materials,
        overrides={},
        wavelength_um=wavelength,
    )

    n_profile = np.ones_like(z_grid)
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
        z_lo = layer.zmin - z_center
        z_hi = layer.zmax - z_center
        mask = (z_grid >= z_lo) & (z_grid < z_hi)
        n_profile[mask] = np.sqrt(eps)
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
) -> ModeResult:
    """Solve for the 1D slab mode of a uniform layer stack.

    The slab is assumed infinite in *x* and *y*; the mode propagates along
    *x* with field variation only along *z*.  Uses a 2D XZ MEEP simulation
    cell.

    Args:
        stack: Layer stack defining the vertical material profile.
        wavelength: Free-space wavelength in µm.
        band_num: Mode band index (1 = fundamental TE/TM slab mode).
        parity: Parity of the mode (``"NO_PARITY"``, ``"EVEN_Y"``, etc.).
        resolution: Pixels per µm (default 32).

    Returns:
        :class:`ModeResult` with effective index, field profiles, and
        group index (from ``mode.group_velocity``).
    """
    _import_meep()

    from gsim.meep.materials import resolve_materials

    if stack is None:
        raise ValueError("stack must be a LayerStack, got None")

    used_materials: set[str] = set()
    for layer in stack.layers.values():
        used_materials.add(layer.material)
    for diel in stack.dielectrics:
        used_materials.add(diel["material"])

    material_data = resolve_materials(
        used_materials,
        overrides={},
        wavelength_um=wavelength,
    )

    sim, cell_size = _build_slab_xz_cell(stack, material_data, resolution)
    sim.init_sim()

    try:
        result = _compute_eigenmode(
            sim,
            cell_size,
            wavelength,
            band_num=band_num,
            parity=parity,
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
    wavelength: float,
    band_num: int = 1,
    parity: str = "NO_PARITY",
    resolution: float = 32,
) -> ModeResult:
    """Solve for the eigenmode of a 2D waveguide cross-section.

    Takes a *y*-slice through the component at a specified port or
    position, builds a 2D XZ MEEP cell with the layer stack, and computes
    the guided mode propagating along *x*.

    One of ``port`` or ``position`` must be given:
    - ``port="o1"`` — auto-extracts the cross-section at the port centre,
      with *x*-span = ``port_width + 2 µm``.
    - ``position=(x, y)`` — uses ``y`` as the cut plane; ``x_span`` must
      be given explicitly.

    Args:
        component: GDSFactory component with ports and polygons.
        stack: Layer stack (already resolved by caller or via
            :meth:`Simulation._resolve_stack_and_materials`).
        port: Port name to auto-extract slice location and *x*-span.
        position: Arbitrary ``(x, y)`` point — only ``y`` is used for
            the cut plane.
        x_span: Total *x*-extent of the cell in µm. Required when
            ``position`` is used; auto-derived when ``port`` is used.
        wavelength: Free-space wavelength in µm.
        band_num: Mode band index (1 = fundamental).
        parity: Parity of the mode.
        resolution: Pixels per µm (default 32).

    Returns:
        :class:`ModeResult` with effective index, field profiles, and
        group index (from ``mode.group_velocity``).
    """
    _import_meep()

    from gsim.meep.materials import resolve_materials

    if stack is None:
        raise ValueError("stack must be a LayerStack, got None")
    if component is None:
        raise ValueError("component must be a Component, got None")

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
        y_cut = float(port_info.center[1])
        port_width = float(port_info.width)
        x_span = x_span or (port_width + 2.0)
    elif position is not None:
        if x_span is None:
            raise ValueError("x_span is required when using position instead of port")
        y_cut = float(position[1])
    else:
        raise ValueError("Either port or position must be specified")

    used_materials: set[str] = set()
    for layer in stack.layers.values():
        used_materials.add(layer.material)
    for diel in stack.dielectrics:
        used_materials.add(diel["material"])

    material_data = resolve_materials(
        used_materials,
        overrides={},
        wavelength_um=wavelength,
    )

    sim, cell_size = _build_component_xz_cell(
        component, stack, y_cut, x_span, material_data, resolution
    )
    sim.init_sim()

    try:
        result = _compute_eigenmode(
            sim,
            cell_size,
            wavelength,
            band_num=band_num,
            parity=parity,
        )
    finally:
        sim.reset_meep()

    _validate_mode(result, material_data, stack, is_slab=False)
    return result
