"""Palace field-visualization routines.

This module provides direct-mesh rendering for boundary and volume fields
from Palace ParaView output, avoiding the NaN-producing probe-grid
resampling used in :mod:`gsim.viz`.

Key functions
-------------
- :func:`build_selector_context` — resolve entity names / boundary types
  to attribute tags from a Palace config + physical-group map.
- :func:`extract_boundary_cells` — exact cell extraction by attribute.
- :func:`activate_vector_component` — compute mag/x/y/z scalar in-place
  on mesh points.
- :func:`load_boundary_field_data` — load boundary mesh and select faces.
- :func:`load_volume_field_data` — load volume mesh for slice workflows.
- :func:`extract_axis_slice`, :func:`extract_plane_slice`,
  :func:`extract_slice_contours` — slicing helpers.
- :func:`plot_boundary_field` — **NaN-free** boundary-field renderer
  (plots the actual mesh cells directly via PyVista).
- :func:`plot_volume_slice` — slice + direct mesh render.
- :func:`plot_volume_contours` — contour lines from a volume slice.

All functions accept the same ``source`` types as
:func:`gsim.palace.results.load_fields` (results dict, directory path).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectorContext:
    """Context used to resolve user selectors to boundary attribute tags.

    Attributes:
    ----------
    pg_map : dict[str, int]
        Physical-group name -> attribute-tag mapping (from the mesh).
    boundaries_by_type : dict[str, tuple[int, ...]]
        Palace boundary-type -> attribute tags (from the config).
    """

    pg_map: dict[str, int]
    boundaries_by_type: dict[str, tuple[int, ...]]


@dataclass
class BoundaryFieldData:
    """Loaded boundary mesh + metadata for a selected step / selector.

    Attributes:
    ----------
    mesh : pyvista.DataSet
        The extracted boundary mesh (only cells matching the selected
        attributes).  Rendered directly — no resampling -> no NaN.
    dataset_name : str
    step_index : int
    timestep : float
    selected_attributes : tuple[int, ...]
    """

    mesh: Any
    dataset_name: str
    step_index: int
    timestep: float
    selected_attributes: tuple[int, ...]

    @property
    def point_arrays(self) -> tuple[str, ...]:
        """Names of point-data arrays available on the boundary mesh."""
        return tuple(self.mesh.point_data.keys())

    @property
    def cell_arrays(self) -> tuple[str, ...]:
        """Names of cell-data arrays available on the boundary mesh."""
        return tuple(self.mesh.cell_data.keys())


@dataclass
class VolumeFieldData:
    """Loaded volume mesh + metadata for a selected step.

    Attributes:
    ----------
    mesh : pyvista.DataSet
    dataset_name : str
    step_index : int
    timestep : float
    """

    mesh: Any
    dataset_name: str
    step_index: int
    timestep: float

    @property
    def point_arrays(self) -> tuple[str, ...]:
        """Names of point-data arrays available on the volume mesh."""
        return tuple(self.mesh.point_data.keys())

    @property
    def cell_arrays(self) -> tuple[str, ...]:
        """Names of cell-data arrays available on the volume mesh."""
        return tuple(self.mesh.cell_data.keys())


# ---------------------------------------------------------------------------
# Selector context construction
# ---------------------------------------------------------------------------


def _coerce_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Return a config dict from a path, file-like object, or existing dict."""
    if isinstance(config, dict):
        return config
    cfg_path = Path(config)
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _attrs_from_boundary_section(section: Any) -> list[int]:
    """Extract attribute-tag integers from a Palace config boundary section."""
    if isinstance(section, dict):
        return [int(a) for a in section.get("Attributes", [])]

    if isinstance(section, list):
        attrs: list[int] = []
        for item in section:
            if isinstance(item, dict):
                attrs.extend(int(a) for a in item.get("Attributes", []))
        return attrs
    return []


def build_selector_context(
    palace_config: str | Path | dict[str, Any],
    pg_map: dict[str, int],
) -> SelectorContext:
    """Create a :class:`SelectorContext` from a Palace config + ``pg_map``.

    Parameters
    ----------
    palace_config :
        Path to the Palace ``config.json`` (or ``.json`` dict).
    pg_map :
        Physical-group name -> attribute-tag mapping.  Usually obtained
        from the mesh::

            import meshio

            mio = meshio.read("palace.msh")
            pg_map = {name: tag for name, (tag, _dim) in mio.field_data.items()}

    Returns:
    -------
    SelectorContext
    """
    cfg = _coerce_config(palace_config)
    boundaries = cfg.get("Boundaries", {}) if isinstance(cfg, dict) else {}

    boundaries_by_type: dict[str, tuple[int, ...]] = {}
    for key, section in boundaries.items():
        attrs = tuple(sorted(set(_attrs_from_boundary_section(section))))
        if attrs:
            boundaries_by_type[key] = attrs

    return SelectorContext(pg_map=dict(pg_map), boundaries_by_type=boundaries_by_type)


def resolve_entity_attributes(
    entity_names: str | Iterable[str],
    context: SelectorContext,
) -> tuple[int, ...]:
    """Resolve entity name(s) to physical attribute tags via ``pg_map``."""
    names = [entity_names] if isinstance(entity_names, str) else list(entity_names)

    missing = [name for name in names if name not in context.pg_map]
    if missing:
        known = ", ".join(sorted(context.pg_map.keys()))
        raise KeyError(
            f"Unknown entity selector(s): {missing}. Known entity names: {known}"
        )

    attrs = sorted({int(context.pg_map[name]) for name in names})
    return tuple(attrs)


def resolve_boundary_type_attributes(
    boundary_type: str,
    context: SelectorContext,
) -> tuple[int, ...]:
    """Resolve a Palace boundary type (e.g. ``PEC``) to tags."""
    if boundary_type not in context.boundaries_by_type:
        known = ", ".join(sorted(context.boundaries_by_type.keys()))
        raise KeyError(
            f"Boundary type '{boundary_type}' not found in config. Known: {known}"
        )
    return context.boundaries_by_type[boundary_type]


# ---------------------------------------------------------------------------
# Boundary cell extraction
# ---------------------------------------------------------------------------


def extract_boundary_cells(
    mesh: Any,
    attributes: Iterable[int],
    attribute_array: str = "attribute",
) -> Any:
    """Extract cells whose ``attribute`` value is in ``attributes``.

    This is the key operation for NaN-free boundary rendering — it
    selects only the surface cells belonging to the requested attributes
    (e.g. ``PEC`` conductors), so that subsequent direct mesh rendering
    has no gaps or invalid probe points.

    Parameters
    ----------
    mesh : pyvista.DataSet
        The full boundary dataset from :func:`load_fields`.
    attributes :
        Attribute IDs to keep.
    attribute_array :
        Name of the cell-data array holding the attribute tag.

    Returns:
    -------
    pyvista.DataSet
        Subset containing only the matching cells.
    """
    attrs = {int(a) for a in attributes}
    if not attrs:
        raise ValueError("No attributes provided for boundary extraction")

    if attribute_array not in mesh.cell_data:
        available = ", ".join(mesh.cell_data.keys())
        raise KeyError(
            f"Cell array '{attribute_array}' not found. Available: {available}"
        )

    values = np.asarray(mesh.cell_data[attribute_array]).astype(int)
    ids = np.where(np.isin(values, list(attrs)))[0]
    if ids.size == 0:
        raise ValueError(
            f"No cells matched attributes {sorted(attrs)} "
            f"in cell array '{attribute_array}'"
        )

    return mesh.extract_cells(ids)


# ---------------------------------------------------------------------------
# Field loading
# ---------------------------------------------------------------------------


def _require_pyvista() -> Any:
    """Import pyvista or raise a clear error."""
    try:
        import pyvista as pv
    except Exception as exc:
        raise ImportError("pyvista is required for field visualization") from exc
    return pv


def load_boundary_field_data(
    source: str | Path | dict,
    selector_context: SelectorContext,
    *,
    entity_names: str | Iterable[str] | None = None,
    boundary_type: str | None = None,
    attributes: Iterable[int] | None = None,
    excitation: int = 1,
    step_index: int = 0,
) -> BoundaryFieldData:
    """Load boundary-only field data for selected conductor faces.

    Selection priority: ``attributes`` > ``entity_names`` > ``boundary_type``.

    Parameters
    ----------
    source :
        Same as :func:`gsim.palace.results.load_fields` — results dict
        or directory path.
    selector_context :
        Pre-built :class:`SelectorContext` from
        :func:`build_selector_context`.
    entity_names :
        Physical-group name(s) to select (e.g. ``"top_conductor"``).
    boundary_type :
        Palace boundary type (e.g. ``"PEC"``).
    attributes :
        Explicit attribute IDs.
    excitation :
        Excitation index (1-based).
    step_index :
        ParaView cycle index (0 = last available).

    Returns:
    -------
    BoundaryFieldData
        The extracted boundary mesh with only the matching cells.
        **No NaN** — rendered directly from the mesh cells.
    """
    from gsim.palace.results import load_fields

    full_mesh = load_fields(
        source,
        excitation=excitation,
        cycle=step_index or None,
        boundary=True,
    )

    if attributes is not None:
        selected = tuple(sorted({int(a) for a in attributes}))
    elif entity_names is not None:
        selected = resolve_entity_attributes(entity_names, selector_context)
    elif boundary_type is not None:
        selected = resolve_boundary_type_attributes(boundary_type, selector_context)
    else:
        raise ValueError(
            "Provide one selector: attributes, entity_names, or boundary_type"
        )

    boundary_mesh = extract_boundary_cells(
        full_mesh, selected, attribute_array="attribute"
    )

    return BoundaryFieldData(
        mesh=boundary_mesh,
        dataset_name="driven_boundary",
        step_index=step_index,
        timestep=float(step_index),
        selected_attributes=selected,
    )


def load_volume_field_data(
    source: str | Path | dict,
    *,
    excitation: int = 1,
    step_index: int = 0,
) -> VolumeFieldData:
    """Load volume field data for slice and volume rendering workflows.

    Parameters
    ----------
    source :
        Results dict or directory path.
    excitation :
        Excitation index (1-based).
    step_index :
        ParaView cycle index (0 = last available).

    Returns:
    -------
    VolumeFieldData
    """
    from gsim.palace.results import load_fields

    mesh = load_fields(
        source,
        excitation=excitation,
        cycle=step_index or None,
        boundary=False,
    )

    return VolumeFieldData(
        mesh=mesh,
        dataset_name="driven",
        step_index=step_index,
        timestep=float(step_index),
    )


def load_field_context(
    source: str | Path | dict,
    *,
    excitation: int = 1,
    step_index: int = 0,
    mesh_filename: str = "palace.msh",
    config_filename: str = "config.json",
) -> tuple[Any, Any, SelectorContext, dict[str, int]]:
    """Load volume + boundary meshes and build a :class:`SelectorContext`.

    This is a convenience helper that bundles the common field-visualization
    setup: it loads the ParaView volume and boundary datasets via
    :func:`gsim.palace.results.load_fields`, reads the mesh physical-group
    map (``pg_map``) from the Gmsh ``.msh`` file with ``meshio``, and builds
    a :class:`SelectorContext` from the Palace ``config.json``.

    The mesh and config files are expected to live in the simulation
    directory, which is resolved as the parent of the results directory
    (i.e. ``results_dir.parent.parent`` for the typical
    ``<sim_dir>/output/palace/`` layout).

    Parameters
    ----------
    source :
        Results dict or directory path (same as
        :func:`gsim.palace.results.load_fields`).
    excitation :
        Excitation index (1-based).
    step_index :
        ParaView cycle index (0 = last available).
    mesh_filename :
        Name of the Gmsh mesh file in the simulation directory.
    config_filename :
        Name of the Palace config file in the simulation directory.

    Returns:
    -------
    tuple
        ``(volume_mesh, boundary_mesh, selector_context, pg_map)`` where
        ``volume_mesh`` and ``boundary_mesh`` are ``pyvista.DataSet``
        objects, ``selector_context`` is a :class:`SelectorContext`, and
        ``pg_map`` is the physical-group name -> attribute-tag mapping.

    Raises:
    ------
    FileNotFoundError
        If the mesh or config file cannot be located in the resolved
        simulation directory.
    """
    import meshio

    from gsim.palace.results import _resolve_source, load_fields

    _, base_dir = _resolve_source(source, require_csv=False)
    sim_dir = Path(base_dir).parent.parent

    msh_path = sim_dir / mesh_filename
    config_path = sim_dir / config_filename
    if not msh_path.exists():
        msg = f"Mesh file not found at {msh_path}"
        raise FileNotFoundError(msg)
    if not config_path.exists():
        msg = f"Palace config not found at {config_path}"
        raise FileNotFoundError(msg)

    cycle = step_index or None
    vol = load_fields(source, excitation=excitation, cycle=cycle, boundary=False)
    bnd = load_fields(source, excitation=excitation, cycle=cycle, boundary=True)

    mio = meshio.read(str(msh_path))
    pg_map = {name: tag for name, (tag, _dim) in mio.field_data.items()}

    ctx = build_selector_context(config_path, pg_map)

    return vol, bnd, ctx, pg_map


# ---------------------------------------------------------------------------
# Slicing helpers
# ---------------------------------------------------------------------------


def extract_axis_slice(
    data: VolumeFieldData,
    axis: str,
    value: float,
) -> Any:
    """Extract an axis-aligned slice from volume data.

    Returns a PyVista mesh (the slice) suitable for direct rendering.
    """
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be one of: x, y, z")

    origin = [0.0, 0.0, 0.0]
    normal_map = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    origin[idx] = float(value)

    return data.mesh.slice(normal=normal_map[axis], origin=tuple(origin))


def extract_plane_slice(
    data: VolumeFieldData,
    origin: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> Any:
    """Extract an arbitrary plane slice from volume data."""
    return data.mesh.slice(normal=normal, origin=origin)


def extract_slice_contours(
    slice_mesh: Any,
    scalar_field: str,
    n_contours: int = 20,
) -> Any:
    """Extract contour lines from a sliced mesh for a scalar field."""
    if scalar_field not in slice_mesh.point_data:
        available = ", ".join(slice_mesh.point_data.keys())
        raise KeyError(
            f"Scalar field '{scalar_field}' not found on slice. Available: {available}"
        )

    return slice_mesh.contour(isosurfaces=int(n_contours), scalars=scalar_field)


# ---------------------------------------------------------------------------
# Vector component activation
# ---------------------------------------------------------------------------


def activate_vector_component(
    mesh: Any,
    field_name: str,
    component: str = "mag",
    output_name: str | None = None,
) -> str:
    """Create a scalar array from a vector field and return its array name.

    Parameters
    ----------
    component :
        One of ``mag``, ``x``, ``y``, ``z``.
    output_name :
        If ``None``, defaults to ``"{field_name}_{component}"``.

    Returns:
    -------
    str
        Name of the new scalar array (added in-place to ``mesh.point_data``).
    """
    if field_name not in mesh.point_data:
        available = ", ".join(mesh.point_data.keys())
        raise KeyError(f"Point field '{field_name}' not found. Available: {available}")

    vec = np.asarray(mesh.point_data[field_name])
    if vec.ndim != 2 or vec.shape[1] < 3:
        raise ValueError(
            f"Field '{field_name}' is not a vector field with 3 components"
        )

    if output_name is None:
        output_name = f"{field_name}_{component}"

    if component == "mag":
        scal = np.linalg.norm(vec[:, :3], axis=1)
    elif component == "x":
        scal = vec[:, 0]
    elif component == "y":
        scal = vec[:, 1]
    elif component == "z":
        scal = vec[:, 2]
    else:
        raise ValueError("component must be one of: mag, x, y, z")

    mesh.point_data[output_name] = scal
    return output_name


def resolve_scalar_field(
    mesh: Any,
    *,
    scalar_field: str | None = None,
    vector_field: str | None = None,
    component: str = "mag",
    output_name: str | None = None,
) -> str:
    """Resolve a scalar field name from scalar or vector input.

    If ``vector_field`` is provided, a scalar component array is created
    in-place on the mesh (via :func:`activate_vector_component`).
    """
    if scalar_field is not None and vector_field is not None:
        raise ValueError("Provide either scalar_field or vector_field, not both")

    if scalar_field is not None:
        if scalar_field not in mesh.point_data:
            available = ", ".join(mesh.point_data.keys())
            raise KeyError(
                f"Scalar field '{scalar_field}' not found. "
                f"Available point arrays: {available}"
            )
        return scalar_field

    if vector_field is not None:
        return activate_vector_component(
            mesh,
            field_name=vector_field,
            component=component,
            output_name=output_name,
        )

    raise ValueError("Provide one of: scalar_field or vector_field")


# ---------------------------------------------------------------------------
# Plotting — NaN-free direct mesh rendering
# ---------------------------------------------------------------------------


def _auto_clim(
    values: np.ndarray,
    *,
    log_scale: bool = False,
    signed: bool = False,
    lo_percentile: float = 2.0,
    hi_percentile: float = 98.0,
) -> tuple[float, float]:
    """Compute percentile-based color limits matching the legacy ``plot_topview``.

    The old :func:`gsim.viz.plot_topview` used ``vmin=0`` / ``vmax=98th
    percentile`` for linear scales and a :class:`~matplotlib.colors.LogNorm`
    spanning the 2nd-98th percentiles for log scales.  This compresses the
    dynamic range so localized peaks (e.g. current crowding at conductor
    edges) remain visible instead of being washed out by the full min-max
    range, which is what PyVista's default ``clim=None`` would do.

    Parameters
    ----------
    signed :
        If ``True``, the data is signed (e.g. a vector component ``E_y``
        rendered with a diverging colormap).  In that case symmetric limits
        ``[-vlim, +vlim]`` are returned, where ``vlim`` is the
        ``hi_percentile`` of ``|values|`` — matching the legacy ``symmetric``
        behavior so the colormap is centered at zero and both polarities are
        visible.  When ``False`` (default, e.g. for magnitudes), ``vmin`` is
        clamped to ``0.0``.
    """
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]

    if log_scale:
        pos = finite[finite > 0.0]
        if pos.size == 0:
            return (1e-10, 1e-9)
        vmin = float(np.percentile(pos, lo_percentile))
        vmax = float(np.percentile(pos, hi_percentile))
        if vmax <= vmin:
            vmax = vmin * 1.01
        return (max(vmin, 1e-12), max(vmax, vmin * 1.01))

    if finite.size == 0:
        return (0.0, 1.0)

    if signed:
        vlim = float(np.percentile(np.abs(finite), hi_percentile))
        if not np.isfinite(vlim) or vlim <= 0.0:
            vlim = float(np.nanmax(np.abs(finite)))
        if vlim <= 0.0:
            vlim = 1.0
        return (-vlim, vlim)

    vmax = float(np.percentile(finite, hi_percentile))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    return (0.0, vmax)


def plot_boundary_field(
    data: BoundaryFieldData,
    scalar_field: str | None = None,
    *,
    vector_field: str | None = None,
    component: str = "mag",
    output_name: str | None = None,
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    opacity: float = 1.0,
    show_edges: bool = False,
    log_scale: bool = False,
    scalar_bar_title: str | None = None,
    off_screen: bool = True,
    screenshot: str | Path | None = None,
) -> Any:
    """Render a selected boundary field with PyVista — **no NaN**.

    Unlike :func:`gsim.viz.plot_topview`, this function renders the
    actual boundary mesh cells directly, without resampling onto a
    regular grid.  This eliminates NaN values that occur when the
    probe grid does not intersect the surface mesh.

    Parameters
    ----------
    data :
        :class:`BoundaryFieldData` from :func:`load_boundary_field_data`.
    scalar_field :
        Name of an existing scalar array in ``data.mesh.point_data``.
    vector_field :
        Name of a 3-component vector field; its magnitude/component
        is computed in-place.
    component :
        Vector component to extract (``"mag"``, ``"x"``, ``"y"``, ``"z"``).
    clim :
        Explicit ``(vmin, vmax)`` color limits.  When ``None`` (default),
        percentile-based limits are computed automatically (``vmin=0``,
        ``vmax=98th percentile`` for linear; 2nd-98th percentile for
        ``log_scale``).  This matches the legacy ``plot_topview`` behavior
        so localized peaks such as current crowding at conductor edges
        remain clearly visible.
    log_scale :
        If ``True``, use logarithmic coloring.  Non-positive values are
        clamped to a small positive floor.
    off_screen :
        If ``True``, the plotter is not displayed (useful for screenshots).
    screenshot :
        If set, save a PNG to this path.

    Returns:
    -------
    pyvista.Plotter
        The plotter (call ``.show()`` to display in a notebook).

    Notes:
    -----
    When ``log_scale=True`` and the scalar contains non-positive values,
    they are clamped to ``min_positive * 1e-6`` so logarithmic coloring
    remains well-defined.
    """
    pv = _require_pyvista()

    scalar_name = resolve_scalar_field(
        data.mesh,
        scalar_field=scalar_field,
        vector_field=vector_field,
        component=component,
        output_name=output_name,
    )

    scalar_name_for_plot = scalar_name
    vals = np.asarray(data.mesh.point_data[scalar_name], dtype=float)
    if log_scale and np.any(vals <= 0.0):
        pos = vals[vals > 0.0]
        if pos.size == 0:
            raise ValueError(
                f"Cannot use log scale for '{scalar_name}': no positive values found"
            )
        floor = float(np.min(pos)) * 1e-6
        safe_name = f"{scalar_name}_logsafe"
        data.mesh.point_data[safe_name] = np.where(vals > 0.0, vals, floor)
        scalar_name_for_plot = safe_name
        vals = np.asarray(data.mesh.point_data[safe_name], dtype=float)

    if clim is None:
        # Signed data (vector components x/y/z, or scalar fields with
        # negative values) gets symmetric limits so diverging colormaps
        # are centered at zero and both polarities are visible.
        is_signed = (vector_field is not None and component in ("x", "y", "z")) or (
            scalar_field is not None and np.nanmin(vals) < 0.0
        )
        clim = _auto_clim(vals, log_scale=log_scale, signed=is_signed)

    pl = pv.Plotter(off_screen=off_screen)
    pl.set_background("white")
    pl.add_mesh(
        data.mesh,
        scalars=scalar_name_for_plot,
        cmap=cmap,
        clim=clim,
        opacity=opacity,
        show_edges=show_edges,
        log_scale=log_scale,
        scalar_bar_args={"title": scalar_bar_title or scalar_name},
    )
    pl.add_axes()
    pl.camera_position = "iso"

    if screenshot is not None:
        pl.screenshot(str(screenshot))

    return pl


def plot_volume_slice(
    data: VolumeFieldData,
    scalar_field: str | None = None,
    *,
    vector_field: str | None = None,
    component: str = "mag",
    output_name: str | None = None,
    axis: str = "z",
    value: float = 0.0,
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    opacity: float = 1.0,
    show_edges: bool = False,
    scalar_bar_title: str | None = None,
    off_screen: bool = True,
    screenshot: str | Path | None = None,
) -> Any:
    """Render an axis-aligned volume slice with a scalar field.

    The slice is extracted via :func:`extract_axis_slice` and rendered
    directly — no NaN from probe-grid resampling.

    When ``clim`` is ``None``, percentile-based color limits are computed
    automatically (see :func:`_auto_clim`) so localized peaks remain
    visible, matching the legacy ``plot_cross_section`` behavior.
    """
    pv = _require_pyvista()
    slice_mesh = extract_axis_slice(data, axis=axis, value=value)

    scalar_name = resolve_scalar_field(
        slice_mesh,
        scalar_field=scalar_field,
        vector_field=vector_field,
        component=component,
        output_name=output_name,
    )

    if clim is None:
        _slice_vals = np.asarray(slice_mesh.point_data[scalar_name], dtype=float)
        _is_signed = (vector_field is not None and component in ("x", "y", "z")) or (
            scalar_field is not None and np.nanmin(_slice_vals) < 0.0
        )
        clim = _auto_clim(_slice_vals, signed=_is_signed)

    pl = pv.Plotter(off_screen=off_screen)
    pl.set_background("white")
    pl.add_mesh(
        slice_mesh,
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        opacity=opacity,
        show_edges=show_edges,
        scalar_bar_args={"title": scalar_bar_title or scalar_name},
    )
    pl.add_axes()
    pl.camera_position = "iso"

    if screenshot is not None:
        pl.screenshot(str(screenshot))

    return pl


def plot_volume_contours(
    data: VolumeFieldData,
    scalar_field: str | None = None,
    *,
    vector_field: str | None = None,
    component: str = "mag",
    output_name: str | None = None,
    axis: str = "z",
    value: float = 0.0,
    n_contours: int = 20,
    line_width: float = 1.2,
    color: str = "black",
    off_screen: bool = True,
    screenshot: str | Path | None = None,
) -> Any:
    """Render contour lines extracted from an axis slice."""
    pv = _require_pyvista()
    slice_mesh = extract_axis_slice(data, axis=axis, value=value)
    scalar_name = resolve_scalar_field(
        slice_mesh,
        scalar_field=scalar_field,
        vector_field=vector_field,
        component=component,
        output_name=output_name,
    )
    contour_mesh = extract_slice_contours(
        slice_mesh,
        scalar_field=scalar_name,
        n_contours=n_contours,
    )

    pl = pv.Plotter(off_screen=off_screen)
    pl.set_background("white")
    pl.add_mesh(contour_mesh, color=color, line_width=float(line_width))
    pl.add_axes()
    pl.camera_position = "iso"

    if screenshot is not None:
        pl.screenshot(str(screenshot))

    return pl
