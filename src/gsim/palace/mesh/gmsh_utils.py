"""Gmsh utility functions for Palace mesh generation."""

from __future__ import annotations

import logging
import math

import gmsh
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimalistic meshwell-style entity + boolean pipeline
# ---------------------------------------------------------------------------


class Entity:
    """A named geometric entity with priority (mesh_order) and tracked dimtags."""

    def __init__(self, name: str, dim: int, mesh_order: int, tags: list[int]):
        """Create an entity with name, dimension, priority and gmsh tags."""
        self.name = name
        self.dim = dim
        self.mesh_order = mesh_order
        self.dimtags = [(dim, t) for t in tags]

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"Entity({self.name!r}, dim={self.dim}, "
            f"order={self.mesh_order}, "
            f"tags={[t for _, t in self.dimtags]})"
        )


def run_boolean_pipeline(entities: list[Entity]) -> dict[str, int]:
    """Meshwell-style boolean pipeline (minimalistic).

    1. Group entities by dimension (descending: 3 → 0).
    2. Within each dimension, sort by mesh_order (ascending = higher priority).
    3. For dim=3: fragment all volumes together in one pass (avoids degenerate
       sliver faces from priority cuts on small volumes like vias).
       For dim=2,1,0: priority cuts — each entity is cut by all previously
       processed entities in the same dimension.
    4. Fragment current dimension against all higher-dimension entities
       already processed, then update tags via the returned mapping.
    5. Accumulate processed entities for the next (lower) dimension.
    6. Assign physical groups (volumes get their own name; surfaces are
       labelled by the pair of volume names they separate).

    Returns:
        pg_map: mapping from physical-group name → pg tag.
    """
    gmsh.model.occ.synchronize()

    # Group by dimension
    dim_groups: dict[int, list[Entity]] = {3: [], 2: [], 1: [], 0: []}
    for e in entities:
        dim_groups[e.dim].append(e)

    # Sort each group by mesh_order (ascending = higher priority first)
    for group in dim_groups.values():
        group.sort(key=lambda e: e.mesh_order)

    processed_higher_dims: list[Entity] = []

    for dim in (3, 2, 1, 0):
        current_group = dim_groups[dim]
        if not current_group:
            continue

        # --- A. Priority cuts within same dimension ---
        processed_in_dim: list[Entity] = []

        if dim == 3:
            # Fragment all 3D entities together in one pass.
            # This avoids degenerate sliver faces that occ.cut() creates
            # when small volumes (e.g. vias) are subtracted from large ones.
            all_3d_dimtags = [dt for e in current_group for dt in e.dimtags]
            if len(all_3d_dimtags) > 1:
                _, out_map = gmsh.model.occ.fragment(
                    all_3d_dimtags,
                    [],
                    removeObject=True,
                    removeTool=True,
                )
                gmsh.model.occ.synchronize()

                # Update each entity's dimtags from the output mapping.
                # Fragment maps shared pieces to ALL parent entities, so
                # de-duplicate: assign each fragment only to the highest-
                # priority entity (lowest mesh_order, processed first).
                claimed: set[tuple[int, int]] = set()
                idx = 0
                for entity in current_group:
                    new_dimtags: list[tuple[int, int]] = []
                    for _ in entity.dimtags:
                        for dt in out_map[idx]:
                            if dt not in claimed:
                                new_dimtags.append(dt)
                                claimed.add(dt)
                        idx += 1
                    entity.dimtags = list(set(new_dimtags))

            processed_in_dim = [e for e in current_group if e.dimtags]
        else:
            # Keep existing priority-cut logic for dim=2,1,0
            for entity in current_group:
                tool_dimtags = [dt for prev in processed_in_dim for dt in prev.dimtags]

                if tool_dimtags and entity.dimtags:
                    cut_result, _ = gmsh.model.occ.cut(
                        entity.dimtags,
                        tool_dimtags,
                        removeObject=True,
                        removeTool=False,
                    )
                    gmsh.model.occ.synchronize()
                    entity.dimtags = list(set(cut_result))

                if entity.dimtags:
                    processed_in_dim.append(entity)

        # --- B. Fragment against higher dimensions ---
        if processed_higher_dims and processed_in_dim:
            object_dimtags = [dt for e in processed_in_dim for dt in e.dimtags]
            tool_dimtags = [dt for e in processed_higher_dims for dt in e.dimtags]

            _, out_map = gmsh.model.occ.fragment(
                object_dimtags,
                tool_dimtags,
                removeObject=True,
                removeTool=True,
            )
            gmsh.model.occ.synchronize()

            # Update tags using the mapping
            idx = 0
            for entity in processed_in_dim:
                new_dimtags: list[tuple[int, int]] = []
                for _ in entity.dimtags:
                    new_dimtags.extend(out_map[idx])
                    idx += 1
                entity.dimtags = list(set(new_dimtags))

            for entity in processed_higher_dims:
                new_dimtags = []
                for _ in entity.dimtags:
                    new_dimtags.extend(out_map[idx])
                    idx += 1
                entity.dimtags = list(set(new_dimtags))

        processed_higher_dims.extend(processed_in_dim)

    # --- Assign physical groups ---

    # 1. Volume physical groups (dim=3)
    for entity in entities:
        if entity.dim == 3 and entity.dimtags:
            tags = [t for d, t in entity.dimtags if d == 3]
            if tags:
                gmsh.model.addPhysicalGroup(3, tags, name=entity.name)

    # 2. Surface → volume ownership map
    vol_entities = [e for e in entities if e.dim == 3 and e.dimtags]
    surf_to_names: dict[int, list[str]] = {}
    for entity in vol_entities:
        for dt in entity.dimtags:
            boundary = gmsh.model.getBoundary(
                [dt],
                combined=False,
                oriented=False,
                recursive=False,
            )
            for bdim, btag in boundary:
                if bdim == 2:
                    surf_to_names.setdefault(btag, [])
                    if entity.name not in surf_to_names[btag]:
                        surf_to_names[btag].append(entity.name)

    # 3. 2D entities keep their own name
    surf_entities = [e for e in entities if e.dim == 2 and e.dimtags]
    assigned_surfs: set[int] = set()
    for entity in surf_entities:
        tags = [t for d, t in entity.dimtags if d == 2]
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, name=entity.name)
            assigned_surfs.update(tags)

    # 4. Remaining surfaces labelled by sorted owner-name pair
    name_combo_to_surfs: dict[str, list[int]] = {}
    for stag, names in surf_to_names.items():
        if stag in assigned_surfs:
            continue
        label = "__".join(sorted(names)) if len(names) > 1 else f"{names[0]}__None"
        name_combo_to_surfs.setdefault(label, []).append(stag)

    for label, stags in name_combo_to_surfs.items():
        gmsh.model.addPhysicalGroup(2, stags, name=label)

    # Build pg_map from gmsh (authoritative)
    pg_map: dict[str, int] = {}
    for dim, pg_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        if name:
            pg_map[name] = pg_tag

    return pg_map


def create_box(
    kernel,
    xmin: float,
    ymin: float,
    zmin: float,
    xmax: float,
    ymax: float,
    zmax: float,
    meshseed: float = 0,
) -> int:
    """Create a 3D box volume in gmsh.

    Args:
        kernel: gmsh.model.occ kernel
        xmin, ymin, zmin: minimum coordinates
        xmax, ymax, zmax: maximum coordinates
        meshseed: mesh seed size at corners (0 = auto)

    Returns:
        Volume tag of created box
    """
    if meshseed == 0:
        # Use simple addBox
        return kernel.addBox(xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin)

    # Create box with explicit mesh seed at corners
    pt1 = kernel.addPoint(xmin, ymin, zmin, meshseed, -1)
    pt2 = kernel.addPoint(xmin, ymax, zmin, meshseed, -1)
    pt3 = kernel.addPoint(xmax, ymax, zmin, meshseed, -1)
    pt4 = kernel.addPoint(xmax, ymin, zmin, meshseed, -1)

    line1 = kernel.addLine(pt1, pt2, -1)
    line2 = kernel.addLine(pt2, pt3, -1)
    line3 = kernel.addLine(pt3, pt4, -1)
    line4 = kernel.addLine(pt4, pt1, -1)
    linetaglist = [line1, line2, line3, line4]

    curvetag = kernel.addCurveLoop(linetaglist, tag=-1)
    surfacetag = kernel.addPlaneSurface([curvetag], tag=-1)
    returnval = kernel.extrude([(2, surfacetag)], 0, 0, zmax - zmin)
    volumetag = returnval[1][1]

    return volumetag


def _create_wire_loop(
    kernel,
    pts_x: list[float],
    pts_y: list[float],
    z: float,
    meshseed: float = 0,
) -> int | None:
    """Create a closed curve loop from polygon vertices.

    Returns:
        Curve loop tag, or None if fewer than 3 valid edges.
    """
    verts = [
        kernel.addPoint(pts_x[v], pts_y[v], z, meshseed, -1) for v in range(len(pts_x))
    ]
    lines = []
    for v in range(len(verts)):
        try:
            ltag = kernel.addLine(verts[v], verts[(v + 1) % len(verts)], -1)
            lines.append(ltag)
        except Exception:
            pass  # Skip degenerate (zero-length) lines
    if len(lines) < 3:
        return None
    return kernel.addCurveLoop(lines, tag=-1)


def create_polygon_surface(
    kernel,
    pts_x: list[float],
    pts_y: list[float],
    z: float,
    meshseed: float = 0,
    holes: list[tuple[list[float], list[float]]] | None = None,
) -> int | None:
    """Create a planar surface from polygon vertices at z height.

    Holes are removed via ``occ.cut()`` for robustness — the OCC kernel
    handles boolean subtraction more reliably than passing multiple curve
    loops to ``addPlaneSurface``.

    Args:
        kernel: gmsh.model.occ kernel
        pts_x: list of x coordinates
        pts_y: list of y coordinates
        z: z coordinate of the surface
        meshseed: mesh seed size at vertices (0 = auto)
        holes: list of (hole_pts_x, hole_pts_y) tuples for interior holes

    Returns:
        Surface tag, or None if polygon is invalid
    """
    if len(pts_x) < 3:
        return None

    exterior_loop = _create_wire_loop(kernel, pts_x, pts_y, z, meshseed)
    if exterior_loop is None:
        return None

    exterior_surf = kernel.addPlaneSurface([exterior_loop], tag=-1)

    if not holes:
        return exterior_surf

    # Build hole surfaces and subtract them via boolean cut
    hole_dimtags: list[tuple[int, int]] = []
    for hx, hy in holes:
        hloop = _create_wire_loop(kernel, list(hx), list(hy), z, meshseed)
        if hloop is not None:
            hsurf = kernel.addPlaneSurface([hloop], tag=-1)
            hole_dimtags.append((2, hsurf))

    if not hole_dimtags:
        return exterior_surf

    result, _ = kernel.cut(
        [(2, exterior_surf)],
        hole_dimtags,
        removeObject=True,
        removeTool=True,
    )
    kernel.synchronize()

    if result:
        return result[0][1]

    logger.warning("Boolean cut for polygon holes returned empty result")
    return None


def extrude_polygon(
    kernel,
    pts_x: list[float],
    pts_y: list[float],
    zmin: float,
    thickness: float,
    meshseed: float = 0,
    holes: list[tuple[list[float], list[float]]] | None = None,
) -> int | None:
    """Create an extruded polygon volume (for vias, metals).

    Args:
        kernel: gmsh.model.occ kernel
        pts_x: list of x coordinates
        pts_y: list of y coordinates
        zmin: base z coordinate
        thickness: extrusion height
        meshseed: mesh seed size at vertices
        holes: list of (hole_pts_x, hole_pts_y) tuples for interior holes

    Returns:
        Volume tag if thickness > 0, surface tag if thickness == 0, or None if invalid
    """
    surfacetag = create_polygon_surface(
        kernel, pts_x, pts_y, zmin, meshseed, holes=holes
    )
    if surfacetag is None:
        return None

    if thickness > 0:
        result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
        # result[1] contains the volume (dim=3, tag)
        return result[1][1]

    return surfacetag


def create_port_rectangle(
    kernel,
    xmin: float,
    ymin: float,
    zmin: float,
    xmax: float,
    ymax: float,
    zmax: float,
    meshseed: float = 0,
) -> int:
    """Create a rectangular surface for a port.

    Handles both horizontal (z-plane) and vertical port surfaces.

    Args:
        kernel: gmsh.model.occ kernel
        xmin, ymin, zmin: minimum coordinates
        xmax, ymax, zmax: maximum coordinates
        meshseed: mesh seed size at corners

    Returns:
        Surface tag of created port rectangle
    """
    # Determine port orientation
    dx = xmax - xmin
    dz = zmax - zmin

    if dz < 1e-6:
        # Horizontal port (in xy plane)
        pt1 = kernel.addPoint(xmin, ymin, zmin, meshseed, -1)
        pt2 = kernel.addPoint(xmin, ymax, zmin, meshseed, -1)
        pt3 = kernel.addPoint(xmax, ymax, zmin, meshseed, -1)
        pt4 = kernel.addPoint(xmax, ymin, zmin, meshseed, -1)
    elif dx < 1e-6:
        # Vertical port in yz plane
        pt1 = kernel.addPoint(xmin, ymin, zmin, meshseed, -1)
        pt2 = kernel.addPoint(xmin, ymax, zmin, meshseed, -1)
        pt3 = kernel.addPoint(xmin, ymax, zmax, meshseed, -1)
        pt4 = kernel.addPoint(xmin, ymin, zmax, meshseed, -1)
    else:
        # Vertical port in xz plane
        pt1 = kernel.addPoint(xmin, ymin, zmin, meshseed, -1)
        pt2 = kernel.addPoint(xmin, ymin, zmax, meshseed, -1)
        pt3 = kernel.addPoint(xmax, ymin, zmax, meshseed, -1)
        pt4 = kernel.addPoint(xmax, ymin, zmin, meshseed, -1)

    line1 = kernel.addLine(pt1, pt2, -1)
    line2 = kernel.addLine(pt2, pt3, -1)
    line3 = kernel.addLine(pt3, pt4, -1)
    line4 = kernel.addLine(pt4, pt1, -1)
    linetaglist = [line1, line2, line3, line4]

    curvetag = kernel.addCurveLoop(linetaglist, tag=-1)
    surfacetag = kernel.addPlaneSurface([curvetag], tag=-1)

    return surfacetag


def fragment_all(kernel) -> tuple[list, list]:
    """Fragment all geometry to ensure conformal mesh at intersections.

    Args:
        kernel: gmsh.model.occ kernel

    Returns:
        (geom_dimtags, geom_map) - original dimtags and mapping to new tags
    """
    geom_dimtags = [x for x in kernel.getEntities() if x[0] in (2, 3)]
    _, geom_map = kernel.fragment(geom_dimtags, [])
    kernel.synchronize()
    return geom_dimtags, geom_map


def get_tags_after_fragment(
    original_tags: list[int],
    geom_dimtags: list,
    geom_map: list,
    dimension: int = 2,
) -> list[int]:
    """Get new tags after fragmenting, given original tags.

    Tags change after gmsh fragment operation. This function maps
    original tags to their new values using the fragment mapping.

    Args:
        original_tags: list of tags before fragmenting
        geom_dimtags: list of all original dimtags before fragmenting
        geom_map: mapping from fragment() function
        dimension: dimension for tags (2=surfaces, 3=volumes)

    Returns:
        List of new tags after fragmenting
    """
    if isinstance(original_tags, int):
        original_tags = [original_tags]

    indices = [
        i
        for i, x in enumerate(geom_dimtags)
        if x[0] == dimension and (x[1] in original_tags)
    ]
    raw = [geom_map[i] for i in indices]
    flat = [item for sublist in raw for item in sublist]
    newtags = [s[-1] for s in flat]

    return newtags


def assign_physical_group(
    dim: int,
    tags: list[int],
    name: str,
) -> int:
    """Assign tags to a physical group with a name.

    Args:
        dim: dimension (2=surfaces, 3=volumes)
        tags: list of entity tags
        name: physical group name

    Returns:
        Physical group tag
    """
    if not tags:
        return -1
    phys_group = gmsh.model.addPhysicalGroup(dim, tags, tag=-1)
    gmsh.model.setPhysicalName(dim, phys_group, name)
    return phys_group


def get_surface_normal(surface_tag: int) -> np.ndarray:
    """Get the normal vector of a surface.

    Args:
        surface_tag: surface entity tag

    Returns:
        Normal vector as numpy array [nx, ny, nz]
    """
    # Get the boundary of the surface
    boundary_lines = gmsh.model.getBoundary([(2, surface_tag)], oriented=True)

    # Get points from these lines
    points = []
    seen_points = set()

    for _dim, line_tag in boundary_lines:
        line_points = gmsh.model.getBoundary([(1, line_tag)], oriented=True)
        for _pdim, ptag in line_points:
            if ptag not in seen_points:
                coord = gmsh.model.getValue(0, ptag, [])
                points.append(np.array(coord))
                seen_points.add(ptag)
            if len(points) == 3:
                break
        if len(points) == 3:
            break

    if len(points) < 3:
        return np.array([0, 0, 1])  # Default to z-normal

    # Compute surface normal using cross product
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    return normal


def is_vertical_surface(surface_tag: int) -> bool:
    """Check if a surface is vertical (not in xy plane).

    Args:
        surface_tag: surface entity tag

    Returns:
        True if surface is vertical (z component of normal is ~0)
    """
    normal = get_surface_normal(surface_tag)
    n = normal[2]
    if not np.isnan(n):
        return int(abs(n)) == 0
    return False


def get_volumes_at_z_range(
    zmin: float,
    zmax: float,
    delta: float = 0.001,
) -> list[tuple[int, int]]:
    """Get all volumes within a z-coordinate range.

    Args:
        zmin: minimum z coordinate
        zmax: maximum z coordinate
        delta: tolerance for z comparison

    Returns:
        List of (dim, tag) tuples for volumes in the z range
    """
    volumes_in_bbox = gmsh.model.getEntitiesInBoundingBox(
        -math.inf,
        -math.inf,
        zmin - delta / 2,
        math.inf,
        math.inf,
        zmax + delta / 2,
        3,
    )

    volume_list = []
    for volume in volumes_in_bbox:
        volume_tag = volume[1]
        _, _, vzmin, _, _, vzmax = gmsh.model.getBoundingBox(3, volume_tag)
        if (
            abs(vzmin - (zmin - delta / 2)) < delta
            and abs(vzmax - (zmax + delta / 2)) < delta
        ):
            volume_list.append(volume)

    return volume_list


def get_surfaces_at_z(z: float, delta: float = 0.001) -> list[tuple[int, int]]:
    """Get all surfaces at a specific z coordinate.

    Args:
        z: z coordinate
        delta: tolerance for z comparison

    Returns:
        List of (dim, tag) tuples for surfaces at z
    """
    return gmsh.model.getEntitiesInBoundingBox(
        -math.inf,
        -math.inf,
        z - delta / 2,
        math.inf,
        math.inf,
        z + delta / 2,
        2,
    )


def get_boundary_lines(surface_tag: int, kernel) -> list[int]:
    """Get all boundary line tags of a surface.

    Args:
        surface_tag: surface entity tag
        kernel: gmsh.model.occ kernel

    Returns:
        List of curve/line tags forming the surface boundary
    """
    _clt, ct = kernel.getCurveLoops(surface_tag)
    lines = []
    for curvetag in ct:
        lines.extend(curvetag)
    return lines


def setup_mesh_refinement(
    boundary_line_tags: list[int],
    refined_cellsize: float,
    max_cellsize: float,
) -> int:
    """Set up mesh refinement near boundary lines.

    Args:
        boundary_line_tags: list of curve tags for refinement
        refined_cellsize: mesh size near boundaries
        max_cellsize: mesh size far from boundaries

    Returns:
        Field ID for the minimum field
    """
    # Distance field from boundary curves
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", boundary_line_tags)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)

    # Threshold field for gradual size transition
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", refined_cellsize)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", max_cellsize)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", max_cellsize)

    return 2


def setup_box_refinement(
    field_id: int,
    xmin: float,
    ymin: float,
    zmin: float,
    xmax: float,
    ymax: float,
    zmax: float,
    size_in: float,
    size_out: float,
) -> None:
    """Set up box-based mesh refinement.

    Args:
        field_id: field ID to use
        xmin, ymin, zmin: box minimum coordinates
        xmax, ymax, zmax: box maximum coordinates
        size_in: mesh size inside box
        size_out: mesh size outside box
    """
    gmsh.model.mesh.field.add("Box", field_id)
    gmsh.model.mesh.field.setNumber(field_id, "VIn", size_in)
    gmsh.model.mesh.field.setNumber(field_id, "VOut", size_out)
    gmsh.model.mesh.field.setNumber(field_id, "XMin", xmin)
    gmsh.model.mesh.field.setNumber(field_id, "XMax", xmax)
    gmsh.model.mesh.field.setNumber(field_id, "YMin", ymin)
    gmsh.model.mesh.field.setNumber(field_id, "YMax", ymax)
    gmsh.model.mesh.field.setNumber(field_id, "ZMin", zmin)
    gmsh.model.mesh.field.setNumber(field_id, "ZMax", zmax)


def finalize_mesh_fields(field_ids: list[int]) -> None:
    """Finalize mesh fields by setting up minimum field.

    Args:
        field_ids: list of field IDs to combine
    """
    min_field_id = max(field_ids) + 1
    gmsh.model.mesh.field.add("Min", min_field_id)
    gmsh.model.mesh.field.setNumbers(min_field_id, "FieldsList", field_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field_id)

    # Disable other mesh size sources
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay algorithm
