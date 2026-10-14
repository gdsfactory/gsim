"""Thermal cross-section mesher: every layer a Region, named Window edges.

Builds the 2D triangle mesh a thermal solve runs on, from a plane section
of a component through its layer stack:

- every stack layer the plane cuts becomes a Region named after the
  layer — heaters, metals and vias included (Conductor model = Region),
  so they can carry a heat source and their own conductivity;
- the full-width stack slabs the component does not draw (substrate,
  BOX, cladding) are laid across the Window and grouped by their
  material, as the Palace native-2D mesher does, and any area nothing
  claims becomes a background group (``"air"`` by default);
- the substrate is meshed to a **Modelled depth** below the rest of the
  stack, which is the solver's choice and not the wafer's thickness;
- the four Window edges are named 1D groups ``bottom``, ``top``, ``left``
  and ``right`` for the heat-sink and top/side boundary conditions;
- sizes are capped per Region (fine in small drawn features, coarse in
  the slabs) and graded away from the drawn features, so a Window
  hundreds of microns deep stays at thousands, not millions, of
  triangles.

The mesh is written as msh 2.2 (the format the Palace pipeline writes),
so the femwell adapter loads it too and the same mesher builds the
optical Window of a thermo-optic solve. :func:`build_cross_section_mesh`
returns it already loaded as a :class:`CrossSectionMesh` on
``skfem.MeshTri``; scikit-fem is the optional ``femwell`` extra.
"""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Literal

import meshio
import numpy as np
from numpy.typing import NDArray

from gsim.common.cross_section import (
    Rect2D,
    RectYZ2D,
    extract_plane_section,
    parse_plane_spec,
)
from gsim.common.mesh_regions import cell_blocks, group_names, group_tags
from gsim.femwell.runtime import require_skfem

if TYPE_CHECKING:
    import gdsfactory as gf
    from skfem import MeshTri

    from gsim.common.stack.extractor import LayerStack

logger = logging.getLogger(__name__)

__all__ = [
    "BOUNDARY_GROUPS",
    "CrossSectionMesh",
    "build_cross_section_mesh",
]

BOUNDARY_GROUPS: tuple[str, ...] = ("bottom", "top", "left", "right")
"""Names of the 1D groups on the four Window edges."""

_WINDOW_MARGIN_UM = 20.0
_DEFAULT_DEPTH_UM = 100.0
_SYNTHETIC_SUBSTRATE_MATERIAL = "silicon"
_FEATURE_DIVISIONS = 4.0
_COARSE_DIVISIONS = 10.0
_GRADING = 0.3
_PLACEHOLDER_GDS_LAYER = (999, 0)


@dataclass(frozen=True)
class CrossSectionMesh:
    """A meshed Cross-section, loaded and ready for a 2D solve.

    Coordinates are in um: ``mesh.p[0]`` is the in-plane axis (``y`` for an
    ``"x=..."`` plane, ``x`` for a ``"y=..."`` plane) and ``mesh.p[1]`` is
    the vertical ``z``.

    Attributes:
        mesh: The triangle mesh, in the msh file's triangle order.
        element_region: Region name of every triangle.
        region_material: Stack material of every Region.
        boundary_facets: Facet indices (into ``mesh.facets``) of each
            Window edge, keyed ``bottom``/``top``/``left``/``right``.
        msh_path: The msh 2.2 file the mesh was read from.
        plane: The plane spec the section was cut at, e.g. ``"x=0"``.
        window: In-plane extent ``(min, max)`` of the meshed Window (um).
        z_range: Vertical extent ``(zmin, zmax)`` of the meshed Window (um).
    """

    mesh: MeshTri
    element_region: NDArray[np.str_]
    region_material: dict[str, str]
    boundary_facets: dict[str, NDArray[np.int64]]
    msh_path: Path
    plane: str
    window: tuple[float, float]
    z_range: tuple[float, float]

    @property
    def regions(self) -> tuple[str, ...]:
        """Names of the Regions on the mesh, sorted."""
        return tuple(sorted(self.region_material))

    def region_mask(self, name: str) -> NDArray[np.bool_]:
        """Boolean mask over the triangles of one Region.

        Args:
            name: Region name.

        Returns:
            ``True`` for every triangle of the Region, in element order.

        Raises:
            ValueError: When the mesh has no Region of that name.
        """
        if name not in self.region_material:
            raise ValueError(
                f"Region '{name}' not found on the mesh. "
                f"Available Regions: {list(self.regions)}"
            )
        return np.asarray(self.element_region == name, dtype=np.bool_)

    def region_area(self, name: str) -> float:
        """Total area of one Region in um^2.

        Args:
            name: Region name.

        Returns:
            The summed area of the Region's triangles.

        Raises:
            ValueError: When the mesh has no Region of that name.
        """
        mask = self.region_mask(name)
        p = self.mesh.p
        t = self.mesh.t[:, mask]
        a, b, c = p[:, t[0]], p[:, t[1]], p[:, t[2]]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return float(0.5 * np.abs(cross).sum())


@dataclass(frozen=True)
class _Rect:
    """One rectangle to mesh, in (in-plane, vertical) coordinates."""

    group: str
    material: str
    kind: Literal["conductor", "dielectric", "slab"]
    h0: float
    h1: float
    v0: float
    v1: float

    @property
    def area(self) -> float:
        """Area of the rectangle (um^2)."""
        return (self.h1 - self.h0) * (self.v1 - self.v0)

    def clipped(
        self, window: tuple[float, float], z_range: tuple[float, float]
    ) -> _Rect | None:
        """The part of the rectangle inside the window, or None if outside."""
        h0, h1 = max(self.h0, window[0]), min(self.h1, window[1])
        v0, v1 = max(self.v0, z_range[0]), min(self.v1, z_range[1])
        if h1 <= h0 or v1 <= v0:
            return None
        return _Rect(self.group, self.material, self.kind, h0, h1, v0, v1)


# Lower ranks win where Regions overlap: a drawn metal carves out of
# whatever it sits in, a drawn dielectric out of the slab around it.
_KIND_RANK = {"conductor": 0, "dielectric": 1, "slab": 2}


def build_cross_section_mesh(
    component: gf.Component,
    stack: LayerStack,
    plane: str,
    *,
    window: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    depth: float | None = None,
    max_size: Mapping[str, float] | None = None,
    default_max_size: float | None = None,
    background: str = "air",
    output: Path | None = None,
) -> CrossSectionMesh:
    """Mesh a Cross-section of a component for a 2D thermal (or optical) solve.

    Args:
        component: The gdsfactory component to cut.
        stack: Layer stack giving each layer's z extent and material, and
            the full-width slabs (``stack.dielectrics``).
        plane: Plane spec, ``"x=<value>"`` or ``"y=<value>"``.
        window: In-plane ``(min, max)`` of the Window in um. Defaults to
            the drawn section's extent plus 20 um on each side.
        z_range: Vertical ``(zmin, zmax)`` of the Window in um. Defaults
            to the Modelled depth below the substrate top up to the top of
            the stack; give it to clip a small (e.g. optical) Window, whose
            edges then cut through the slabs.
        depth: Modelled depth of the substrate in um, measured down from
            the bottom of the rest of the stack. Defaults to the thickness
            of the stack's ``substrate`` slab, or 100 um when the stack has
            none (a silicon substrate is then synthesised, with a warning).
        max_size: Maximum element size in um per Region name, overriding
            the defaults.
        default_max_size: Coarse element size in um for the slabs and the
            background, and the size the grading grows towards. Defaults
            to a tenth of the Window's larger dimension. Drawn Regions
            default to a quarter of their smallest feature, slabs to half
            their thickness, both capped by this.
        background: Region name and material of any Window area no layer
            or slab claims.
        output: Path of the msh 2.2 file to write. Defaults to a file in a
            new temporary directory.

    Returns:
        The loaded mesh with its Regions, materials and Window edges.

    Raises:
        ValueError: When the plane is not an ``x``/``y`` plane, the plane
            cuts no drawn layer, or the Window is empty.
        ImportError: When scikit-fem (the ``femwell`` extra) is missing.
    """
    skfem = require_skfem()

    axis, value = parse_plane_spec(plane)
    if axis not in {"x", "y"}:
        raise ValueError(
            "Thermal cross-sections are vertical: use an 'x=<value>' or "
            f"'y=<value>' plane, not '{plane}'."
        )

    drawn = _drawn_rects(component, stack, axis, value)
    if not drawn:
        raise ValueError(f"Plane '{plane}' does not cut any drawn stack layer.")

    slabs, substrate = _stack_slabs(stack)
    sub_top, default_depth = _substrate_top_and_depth(drawn, slabs, substrate)
    depth = default_depth if depth is None else float(depth)
    if depth <= 0.0:
        raise ValueError(f"depth must be positive, got {depth}.")
    bottom = sub_top - depth
    if substrate is None:
        logger.warning(
            "The stack has no 'substrate' slab; meshing a synthetic %s "
            "substrate %.3g um deep under it.",
            _SYNTHETIC_SUBSTRATE_MATERIAL,
            depth,
        )
        substrate = _Rect(
            _SYNTHETIC_SUBSTRATE_MATERIAL,
            _SYNTHETIC_SUBSTRATE_MATERIAL,
            "slab",
            -math.inf,
            math.inf,
            bottom,
            sub_top,
        )
    else:
        substrate = _Rect(
            substrate.group,
            substrate.material,
            "slab",
            -math.inf,
            math.inf,
            bottom,
            substrate.v1,
        )
    slabs = [substrate, *slabs]

    if window is None:
        window = (
            min(r.h0 for r in drawn) - _WINDOW_MARGIN_UM,
            max(r.h1 for r in drawn) + _WINDOW_MARGIN_UM,
        )
    if z_range is None:
        z_range = (bottom, max(r.v1 for r in [*drawn, *slabs]))
    window = (float(window[0]), float(window[1]))
    z_range = (float(z_range[0]), float(z_range[1]))
    if window[1] <= window[0] or z_range[1] <= z_range[0]:
        raise ValueError(f"Empty Window: window={window}, z_range={z_range}.")

    rects = [
        clipped
        for rect in [*drawn, *slabs]
        if (clipped := rect.clipped(window, z_range)) is not None
    ]

    coarse = (
        float(default_max_size)
        if default_max_size is not None
        else max(window[1] - window[0], z_range[1] - z_range[0]) / _COARSE_DIVISIONS
    )
    sizes = _region_sizes(rects, coarse, background, max_size or {})

    if output is None:
        output = Path(tempfile.mkdtemp(prefix="gsim_thermal_")) / "cross_section.msh"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    _mesh_with_gmsh(rects, window, z_range, sizes, coarse, background, output)

    region_material = {r.group: r.material for r in rects}
    return _load(skfem, output, plane, window, z_range, region_material)


def _drawn_rects(
    component: gf.Component, stack: LayerStack, axis: str, value: float
) -> list[_Rect]:
    """The plane section, one rectangle per drawn layer interval."""
    section = extract_plane_section(
        component,
        stack,
        axis="x" if axis == "x" else "y",
        value=value,
    )
    rects: list[_Rect] = []
    for item in section:
        if isinstance(item, RectYZ2D):
            h0, h1 = item.y0, item.y1
        elif isinstance(item, Rect2D):
            h0, h1 = item.x0, item.x1
        else:  # pragma: no cover - x/y planes yield rectangles only
            continue
        v0, v1 = item.zmin, item.zmax
        if h1 <= h0 or v1 <= v0:
            continue
        layer = stack.layers[item.layer_name]
        kind: Literal["conductor", "dielectric"] = (
            "conductor" if layer.layer_type in {"conductor", "via"} else "dielectric"
        )
        rects.append(_Rect(item.layer_name, layer.material, kind, h0, h1, v0, v1))
    return rects


def _stack_slabs(stack: LayerStack) -> tuple[list[_Rect], _Rect | None]:
    """Full-width slabs of the stack, and its substrate slab if it has one.

    Slabs are grouped under their material name, as the Palace native-2D
    mesher groups them. A ``dielectrics`` entry that is itself a drawn
    layer of the stack is the drawn layer, not a slab, and is skipped.
    """
    slabs: list[_Rect] = []
    substrate: _Rect | None = None
    for entry in stack.dielectrics:
        name = str(entry.get("name", ""))
        zmin, zmax = float(entry["zmin"]), float(entry["zmax"])
        if zmax <= zmin:
            continue
        layer = stack.layers.get(name)
        if layer is not None and tuple(layer.gds_layer) != _PLACEHOLDER_GDS_LAYER:
            continue
        material = str(entry["material"])
        rect = _Rect(material, material, "slab", -math.inf, math.inf, zmin, zmax)
        if name.lower() == "substrate" and substrate is None:
            substrate = rect
        else:
            slabs.append(rect)
    return slabs, substrate


def _substrate_top_and_depth(
    drawn: list[_Rect], slabs: list[_Rect], substrate: _Rect | None
) -> tuple[float, float]:
    """Top of the substrate and its default Modelled depth."""
    if substrate is not None:
        return substrate.v1, substrate.v1 - substrate.v0
    return min(r.v0 for r in [*drawn, *slabs]), _DEFAULT_DEPTH_UM


def _region_sizes(
    rects: list[_Rect],
    coarse: float,
    background: str,
    overrides: Mapping[str, float],
) -> dict[str, float]:
    """Maximum element size of every Region."""
    sizes: dict[str, float] = {background: coarse}
    for rect in rects:
        if rect.kind == "slab":
            own = (rect.v1 - rect.v0) / 2.0
        else:
            own = min(rect.h1 - rect.h0, rect.v1 - rect.v0) / _FEATURE_DIVISIONS
        sizes[rect.group] = min(sizes.get(rect.group, coarse), own, coarse)
    for name, size in overrides.items():
        if size <= 0.0:
            raise ValueError(f"max_size['{name}'] must be positive, got {size}.")
        sizes[name] = float(size)
    return sizes


def _mesh_with_gmsh(
    rects: list[_Rect],
    window: tuple[float, float],
    z_range: tuple[float, float],
    sizes: Mapping[str, float],
    coarse: float,
    background: str,
    output: Path,
) -> None:
    """Fragment the rectangles, name groups, grade sizes, write msh 2.2."""
    import gmsh

    started = not gmsh.isInitialized()
    if started:
        # The mesh must depend only on gsim's settings, not a user's options file.
        gmsh.initialize(readConfigFiles=False)
    model_name = "gsim_thermal_cross_section"
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        if model_name in gmsh.model.list():
            gmsh.model.setCurrent(model_name)
            gmsh.model.remove()
        gmsh.model.add(model_name)
        occ = gmsh.model.occ

        hmin, hmax = window
        vmin, vmax = z_range
        outer = occ.addRectangle(hmin, vmin, 0.0, hmax - hmin, vmax - vmin)
        tags = [
            occ.addRectangle(r.h0, r.v0, 0.0, r.h1 - r.h0, r.v1 - r.v0) for r in rects
        ]
        _, out_map = occ.fragment([(2, outer)], [(2, t) for t in tags])
        occ.synchronize()

        surfaces = _assign_surfaces(rects, out_map, background)
        for name in sorted(surfaces):
            pg = gmsh.model.addPhysicalGroup(2, sorted(surfaces[name]))
            gmsh.model.setPhysicalName(2, pg, name)

        edges = _window_edge_curves(window, z_range)
        for name in BOUNDARY_GROUPS:
            pg = gmsh.model.addPhysicalGroup(1, edges[name])
            gmsh.model.setPhysicalName(1, pg, name)

        drawn_groups = {r.group for r in rects if r.kind != "slab"}
        _set_size_fields(surfaces, sizes, drawn_groups, coarse)

        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(output))
    finally:
        if started:
            gmsh.finalize()
        else:
            gmsh.model.remove()


def _assign_surfaces(
    rects: list[_Rect],
    out_map: list[list[tuple[int, int]]],
    background: str,
) -> dict[str, set[int]]:
    """Give each fragment piece to the highest-priority Region covering it.

    Drawn conductors win over drawn dielectrics, which win over slabs;
    within a kind the Region of smaller total area wins (the more
    detailed shape carves out of the larger one). Pieces no rectangle
    covers are the background.
    """
    area: dict[str, float] = {}
    for rect in rects:
        area[rect.group] = area.get(rect.group, 0.0) + rect.area
    rank = {r.group: (_KIND_RANK[r.kind], area[r.group], r.group) for r in rects}

    owner: dict[int, str] = {}
    for rect, pieces in zip(rects, out_map[1:], strict=True):
        for dim, tag in pieces:
            if dim != 2:
                continue
            current = owner.get(tag)
            if current is None or rank[rect.group] < rank[current]:
                owner[tag] = rect.group

    surfaces: dict[str, set[int]] = {}
    for dim, tag in out_map[0]:
        if dim == 2:
            surfaces.setdefault(owner.get(tag, background), set()).add(tag)
    return surfaces


def _window_edge_curves(
    window: tuple[float, float], z_range: tuple[float, float]
) -> dict[str, list[int]]:
    """Curves lying on each of the four Window edges."""
    import gmsh

    hmin, hmax = window
    vmin, vmax = z_range
    # OCC pads bounding boxes by its own tolerance (~1e-7).
    tol = 1e-6 * max(1.0, hmax - hmin, vmax - vmin)
    edges: dict[str, list[int]] = {name: [] for name in BOUNDARY_GROUPS}
    for _dim, curve in gmsh.model.getEntities(1):
        x0, y0, _z0, x1, y1, _z1 = gmsh.model.getBoundingBox(1, curve)
        if abs(y0 - vmin) <= tol and abs(y1 - vmin) <= tol:
            edges["bottom"].append(curve)
        elif abs(y0 - vmax) <= tol and abs(y1 - vmax) <= tol:
            edges["top"].append(curve)
        elif abs(x0 - hmin) <= tol and abs(x1 - hmin) <= tol:
            edges["left"].append(curve)
        elif abs(x0 - hmax) <= tol and abs(x1 - hmax) <= tol:
            edges["right"].append(curve)
    return edges


def _set_size_fields(
    surfaces: Mapping[str, set[int]],
    sizes: Mapping[str, float],
    drawn_groups: set[str],
    coarse: float,
) -> None:
    """Cap the size inside every Region and grade away from drawn ones."""
    import gmsh

    field = gmsh.model.mesh.field
    field_ids: list[int] = []
    next_id = 1
    for name, tags in sorted(surfaces.items()):
        size = sizes.get(name, coarse)
        surface_list = sorted(tags)

        const_id, restrict_id = next_id, next_id + 1
        next_id += 2
        field.add("MathEval", const_id)
        field.setString(const_id, "F", repr(float(size)))
        field.add("Restrict", restrict_id)
        field.setNumber(restrict_id, "InField", const_id)
        field.setNumbers(restrict_id, "SurfacesList", surface_list)
        field_ids.append(restrict_id)

        if name not in drawn_groups or size >= coarse:
            continue
        curves = sorted(
            {
                abs(tag)
                for dim, tag in gmsh.model.getBoundary(
                    [(2, t) for t in surface_list], combined=False, oriented=False
                )
                if dim == 1
            }
        )
        longest = max(_curve_length(c) for c in curves)
        distance_id, threshold_id = next_id, next_id + 1
        next_id += 2
        field.add("Distance", distance_id)
        field.setNumbers(distance_id, "CurvesList", curves)
        field.setNumber(
            distance_id, "Sampling", min(2000, max(20, math.ceil(2 * longest / size)))
        )
        field.add("Threshold", threshold_id)
        field.setNumber(threshold_id, "InField", distance_id)
        field.setNumber(threshold_id, "SizeMin", size)
        field.setNumber(threshold_id, "SizeMax", coarse)
        field.setNumber(threshold_id, "DistMin", 0.0)
        field.setNumber(threshold_id, "DistMax", (coarse - size) / _GRADING)
        field_ids.append(threshold_id)

    min_id = next_id
    field.add("Min", min_id)
    field.setNumbers(min_id, "FieldsList", field_ids)
    field.setAsBackgroundMesh(min_id)

    gmsh.option.setNumber("Mesh.MeshSizeMax", coarse)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay


def _curve_length(curve: int) -> float:
    """Length of a straight gmsh curve from its bounding box."""
    import gmsh

    x0, y0, _z0, x1, y1, _z1 = gmsh.model.getBoundingBox(1, curve)
    return math.hypot(x1 - x0, y1 - y0)


def _load(
    skfem: ModuleType,
    msh_path: Path,
    plane: str,
    window: tuple[float, float],
    z_range: tuple[float, float],
    region_material: dict[str, str],
) -> CrossSectionMesh:
    """Read the msh back into skfem, the way the femwell adapter does."""
    # Name the format: ".msh" is also ANSYS, whose failed try prints to stdout.
    mio = meshio.read(str(msh_path), file_format="gmsh")
    tris, tri_tags = cell_blocks(mio, "triangle")

    # Drop points no triangle references, keeping triangle order intact.
    points = np.asarray(mio.points)
    used = np.unique(tris)
    remap = np.full(points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    tris = remap[tris]
    points = points[used]

    mesh = skfem.MeshTri(
        np.ascontiguousarray(points[:, :2].T, dtype=np.float64),
        np.ascontiguousarray(tris.T, dtype=np.int64),
    )

    names = group_names(mio, dim=2)
    element_region = np.array([names.get(int(t), "") for t in tri_tags], dtype=np.str_)
    present = set(names.values())
    # The background names its own material.
    materials = {name: region_material.get(name, name) for name in present}

    lines, line_tags = cell_blocks(mio, "line")
    lines = remap[lines]
    n_nodes = points.shape[0]
    facets = np.sort(np.asarray(mesh.facets), axis=0)
    facet_keys = facets[0] * n_nodes + facets[1]
    order = np.argsort(facet_keys)
    sorted_keys = facet_keys[order]
    boundary_facets: dict[str, NDArray[np.int64]] = {}
    for name, tag in group_tags(mio, dim=1).items():
        pairs = np.sort(lines[line_tags == tag], axis=1)
        keys = pairs[:, 0] * n_nodes + pairs[:, 1]
        pos = np.minimum(np.searchsorted(sorted_keys, keys), sorted_keys.size - 1)
        unmatched = int(np.count_nonzero(sorted_keys[pos] != keys))
        if unmatched:
            raise RuntimeError(
                f"Boundary group '{name}' of {msh_path}: {unmatched} of its "
                f"{keys.size} line elements are not edges of the triangle "
                "mesh, so the mesh is inconsistent; please report it."
            )
        boundary_facets[name] = np.asarray(np.sort(order[pos]), dtype=np.int64)

    return CrossSectionMesh(
        mesh=mesh,
        element_region=element_region,
        region_material=materials,
        boundary_facets=boundary_facets,
        msh_path=msh_path,
        plane=plane,
        window=window,
        z_range=z_range,
    )
