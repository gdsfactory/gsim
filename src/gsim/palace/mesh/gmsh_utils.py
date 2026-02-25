"""Gmsh utility functions for Palace mesh generation.

Extracted from gds2palace/util_simulation_setup.py and adapted
to work directly with palace-api data structures.
"""

from __future__ import annotations

import math

import gmsh
import numpy as np


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


def create_polygon_surface(
    kernel,
    pts_x: list[float],
    pts_y: list[float],
    z: float,
    meshseed: float = 0,
    holes: list[tuple[list[float], list[float]]] | None = None,
) -> int | None:
    """Create a planar surface from polygon vertices at z height.

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
    numvertices = len(pts_x)
    if numvertices < 3:
        return None

    linetaglist = []
    vertextaglist = []

    # Create vertices
    for v in range(numvertices):
        vertextag = kernel.addPoint(pts_x[v], pts_y[v], z, meshseed, -1)
        vertextaglist.append(vertextag)

    # Create lines connecting vertices
    for v in range(numvertices):
        pt_start = vertextaglist[v]
        pt_end = vertextaglist[(v + 1) % numvertices]
        try:
            linetag = kernel.addLine(pt_start, pt_end, -1)
            linetaglist.append(linetag)
        except Exception:
            pass  # Skip degenerate lines

    if len(linetaglist) < 3:
        return None

    # Create outer curve loop
    curvetag = kernel.addCurveLoop(linetaglist, tag=-1)

    # Create hole curve loops
    all_loops = [curvetag]
    for hx, hy in holes or []:
        hole_lines = []
        hole_verts = []
        for v in range(len(hx)):
            vtag = kernel.addPoint(hx[v], hy[v], z, meshseed, -1)
            hole_verts.append(vtag)
        for v in range(len(hx)):
            try:
                ltag = kernel.addLine(hole_verts[v], hole_verts[(v + 1) % len(hx)], -1)
                hole_lines.append(ltag)
            except Exception:
                pass
        if len(hole_lines) >= 3:
            hole_loop = kernel.addCurveLoop(hole_lines, tag=-1)
            all_loops.append(hole_loop)

    surfacetag = kernel.addPlaneSurface(all_loops, tag=-1)

    return surfacetag


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



# ---------------------------------------------------------------------------
# 2D cross-section helpers
# ---------------------------------------------------------------------------

def create_cutting_plane(margin: float, *, x: float | None = None, y: float | None = None) -> int:
    """Add a large rectangle orthogonal to either the x or y axis.

    Exactly one of *x* or *y* must be provided.

    Args:
        margin: Half-extent of the rectangle in the two free directions
                (should exceed the model bounds).
        x:      Fix x to this value → rectangle in the YZ-plane.
        y:      Fix y to this value → rectangle in the XZ-plane.

    Returns:
        The gmsh surface tag of the plane (not yet synchronised).
    """
    if (x is None) == (y is None):
        raise ValueError("Exactly one of 'x' or 'y' must be provided.")

    if x is not None:
        # YZ-plane at fixed x
        p1 = gmsh.model.occ.addPoint(x, -margin, -margin)
        p2 = gmsh.model.occ.addPoint(x,  margin, -margin)
        p3 = gmsh.model.occ.addPoint(x,  margin,  margin)
        p4 = gmsh.model.occ.addPoint(x, -margin,  margin)
    else:
        # XZ-plane at fixed y
        p1 = gmsh.model.occ.addPoint(-margin, y, -margin)
        p2 = gmsh.model.occ.addPoint( margin, y, -margin)
        p3 = gmsh.model.occ.addPoint( margin, y,  margin)
        p4 = gmsh.model.occ.addPoint(-margin, y,  margin)

    cl = gmsh.model.occ.addCurveLoop([
        gmsh.model.occ.addLine(p1, p2),
        gmsh.model.occ.addLine(p2, p3),
        gmsh.model.occ.addLine(p3, p4),
        gmsh.model.occ.addLine(p4, p1),
    ])
    return gmsh.model.occ.addPlaneSurface([cl])


def slice_with_plane(cut_plane: int) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Fragment a cutting plane against all named 3D volumes in the current model.

    Reads volume→name from the dim=3 physical groups, fragments the plane
    against all volumes in one OCC call, maps each resulting surface piece
    back to the name of the volume it borders, then removes the pre-fragment
    physical groups (which are now stale).

    Args:
        cut_plane: Surface tag of the cutting plane (must be synchronised).

    Returns:
        name_to_surfs: mapping from volume name → list of 2D surface tags that
                       form the cross-section inside that volume.
        pg_map:        mapping from physical group name → pg tag for all
                       newly created 2D and 1D physical groups.
    """
    # Snapshot physical groups and build vol_tag → name.
    pgs_before = list(gmsh.model.getPhysicalGroups())
    vol_tag_to_name: dict[int, str] = {}
    for pgd, pgt in pgs_before:
        if pgd != 3:
            continue
        pgname = gmsh.model.getPhysicalName(pgd, pgt)
        for vt in gmsh.model.getEntitiesForPhysicalGroup(pgd, pgt):
            vol_tag_to_name[vt] = pgname

    # Single fragment call: plane vs all volumes.
    # removeTool=False keeps volumes alive for boundary queries.
    all_vol_dimtags = [(3, t) for t in vol_tag_to_name]
    _, fmap = gmsh.model.occ.fragment(
        [(2, cut_plane)], all_vol_dimtags,
        removeObject=True, removeTool=False,
    )
    gmsh.model.occ.synchronize()

    # fmap[0]    → surface pieces carved from the cutting plane
    # fmap[i+1]  → updated dimtags for all_vol_dimtags[i]
    cs_surfs = [t for d, t in fmap[0] if d == 2]

    # Map each surface piece to the volume that contains it as a face.
    name_to_surfs: dict[str, list[int]] = {}
    for stag in cs_surfs:
        for i, (_, t_orig) in enumerate(all_vol_dimtags):
            for dv, tv in ((d, t) for d, t in fmap[i + 1] if d == 3):
                bnd = gmsh.model.getBoundary(
                    [(dv, tv)], combined=False, oriented=False, recursive=False
                )
                if any(bt == stag for bd, bt in bnd if bd == 2):
                    name_to_surfs.setdefault(vol_tag_to_name[t_orig], []).append(stag)
                    break
            else:
                continue
            break

    for name, tags in name_to_surfs.items():
        print(f"  '{name}' → surfaces {tags}")

    # Remove the now-stale pre-fragment physical groups.
    gmsh.model.removePhysicalGroups(pgs_before)

    # --- Surface physical groups ---
    pg_map: dict[str, int] = {}
    for name, stags in name_to_surfs.items():
        pg = gmsh.model.addPhysicalGroup(2, stags, name=name)
        pg_map[name] = pg
        print(f"  Physical '{name}' (dim=2): pg={pg}, tags={stags}")

    # --- Line physical groups (edges labelled by the pair of surfaces they separate) ---
    edge_to_names: dict[int, list[str]] = {}
    for name, stags in name_to_surfs.items():
        for stag in stags:
            for bd, bt in gmsh.model.getBoundary(
                [(2, stag)], combined=False, oriented=False, recursive=False
            ):
                if bd == 1:
                    edge_to_names.setdefault(bt, [])
                    if name not in edge_to_names[bt]:
                        edge_to_names[bt].append(name)
    label_to_edges: dict[str, list[int]] = {}
    for lt, ns in edge_to_names.items():
        lbl = "__".join(sorted(ns)) if len(ns) > 1 else f"{ns[0]}__None"
        label_to_edges.setdefault(lbl, []).append(lt)
    for lbl, lts in label_to_edges.items():
        pg = gmsh.model.addPhysicalGroup(1, lts, name=lbl)
        pg_map[lbl] = pg
        print(f"  Physical '{lbl}' (dim=1): pg={pg}, tags={lts}")

    return name_to_surfs, pg_map



