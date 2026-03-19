"""Generic GMSH mesh generator for any LayerStack.

Orchestrates geometry extraction, volume creation, dielectric boxes,
fragmentation, physical group assignment, refinement, and meshing.
Solver-agnostic — produces a .msh file usable by Palace, other solvers,
or standalone mesh inspection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh

from gsim.common.mesh import gmsh_utils
from gsim.common.mesh.geometry import (
    add_dielectrics,
    add_layer_volumes,
    extract_geometry,
)
from gsim.common.mesh.types import MeshResult

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack

logger = logging.getLogger(__name__)


def generate_mesh(
    component,
    stack: LayerStack,
    output_dir: str | Path,
    *,
    model_name: str = "mesh",
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 300.0,
    margin: float = 50.0,
    air_margin: float = 50.0,
    include_airbox: bool = True,
    include_ports: bool = True,
    mesh_scale: float | None = None,
    show_gui: bool = False,
) -> MeshResult:
    """Generate a generic GMSH mesh from a gdsfactory component and LayerStack.

    Steps:
        1. Extract polygon geometry from the component
        2. Extrude all layer polygons into 3D volumes
        3. Add dielectric background boxes and optionally airbox
        3b. (Optional) Add port surfaces from component ports
        4. Fragment all geometry for conformal meshing
        5. Assign physical groups (one per material, plus outer boundary)
        5b. Assign port physical groups
        6. Set up mesh refinement near layer volume boundaries
        7. Generate 3D mesh and write .msh file

    Args:
        component: gdsfactory Component
        stack: LayerStack with layer and material definitions
        output_dir: Directory for output files
        model_name: Base name for output files (default: "mesh")
        refined_mesh_size: Fine mesh size near layer boundaries (um)
        max_mesh_size: Coarse mesh size in air/dielectric (um)
        margin: XY margin around design (um)
        air_margin: Air box margin around dielectric envelope (um)
        include_airbox: Whether to add a surrounding airbox volume.
            Needed for RF (Palace); not needed for photonics (Meep).
        include_ports: Whether to create port surfaces from component
            ports as physical groups. Default ``True``.
        mesh_scale: Scale factor applied to mesh coordinates after generation.
            E.g. ``1000.0`` converts um → nm. Default ``None`` (no scaling).
        show_gui: Show gmsh GUI during meshing

    Returns:
        MeshResult with mesh path, statistics, and physical group info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    msh_path = output_dir / f"{model_name}.msh"

    # 1. Extract geometry
    logger.info("Extracting geometry...")
    geometry = extract_geometry(component, stack)
    logger.info("  Polygons: %s", len(geometry.polygons))
    logger.info("  Bbox: %s", geometry.bbox)

    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)

    model_tag = f"{model_name}_mesh"
    if model_tag in gmsh.model.list():
        gmsh.model.setCurrent(model_tag)
        gmsh.model.remove()
    gmsh.model.add(model_tag)

    kernel = gmsh.model.occ

    try:
        # 2. Extrude all layer polygons into volumes
        logger.info("Adding layer volumes...")
        layer_volume_tags = add_layer_volumes(kernel, geometry, stack)

        # 3. Add dielectric boxes (and optionally airbox)
        logger.info("Adding dielectrics...")
        dielectric_tags = add_dielectrics(
            kernel,
            geometry,
            stack,
            margin,
            air_margin,
            include_airbox=include_airbox,
        )

        # 4. Fragment all geometry
        logger.info("Fragmenting geometry...")
        geom_dimtags, geom_map = gmsh_utils.fragment_all(kernel)

        # 5. Assign physical groups
        logger.info("Assigning physical groups...")
        groups = _assign_generic_groups(
            kernel,
            layer_volume_tags,
            dielectric_tags,
            geom_dimtags,
            geom_map,
            stack,
        )

        # 5b. Find and tag port surfaces on existing volume boundaries
        if include_ports:
            logger.info("Identifying port surfaces...")
            _assign_port_groups(component, stack, groups)

        # 6. Mesh refinement
        logger.info("Setting up mesh refinement...")
        _setup_generic_refinement(kernel, groups, refined_mesh_size, max_mesh_size)

        # Show GUI if requested
        if show_gui:
            gmsh.fltk.run()

        # 7. Generate mesh and write
        logger.info("Generating 3D mesh...")
        gmsh.model.mesh.generate(3)

        if mesh_scale is not None:
            s = mesh_scale
            gmsh.model.mesh.affineTransform([s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0])

        mesh_stats = collect_mesh_stats()

        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_path))

        logger.info("Mesh saved: %s", msh_path)

    finally:
        gmsh.clear()
        gmsh.finalize()

    return MeshResult(
        mesh_path=msh_path,
        mesh_stats=mesh_stats,
        groups=groups,
    )


def _assign_generic_groups(
    kernel,
    layer_volume_tags: dict[str, list[int]],
    dielectric_tags: dict[str, list[int]],
    geom_dimtags: list,
    geom_map: list,
    stack: LayerStack,
) -> dict:
    """Assign physical groups for the generic mesh.

    Creates one volume physical group per material name (merging layers
    that share the same material) plus an outer boundary surface group
    from the airbox.

    Returns:
        {"volumes": {material: {"phys_group": int, "tags": [int]}},
         "layer_volumes": {layer_name: {"phys_group": int, "tags": [int]}},
         "outer_boundary": {"phys_group": int, "tags": [int]}}
    """
    groups: dict = {
        "volumes": {},
        "layer_volumes": {},
        "outer_boundary": {},
    }

    # --- Layer volume groups first (to know which tags to exclude from dielectrics) ---
    layer_volume_tag_set: set[int] = set()
    for layer_name, tags in layer_volume_tags.items():
        new_tags = gmsh_utils.get_tags_after_fragment(
            tags, geom_dimtags, geom_map, dimension=3
        )
        if new_tags:
            layer_volume_tag_set.update(new_tags)
            # Look up material for this layer to also merge into volume groups
            layer = stack.layers.get(layer_name)
            material_name = layer.material if layer else layer_name

            phys_group = gmsh_utils.assign_physical_group(3, new_tags, layer_name)
            groups["layer_volumes"][layer_name] = {
                "phys_group": phys_group,
                "tags": new_tags,
                "material": material_name,
            }

    # --- Dielectric / airbox volume groups (exclude layer volume fragments) ---
    for material_name, tags in dielectric_tags.items():
        new_tags = gmsh_utils.get_tags_after_fragment(
            tags, geom_dimtags, geom_map, dimension=3
        )
        if new_tags:
            # Exclude fragments that belong to layer volumes to avoid
            # double-assignment (e.g. waveguide core inside an oxide box)
            new_tags = [t for t in new_tags if t not in layer_volume_tag_set]
        if new_tags:
            phys_group = gmsh_utils.assign_physical_group(3, new_tags, material_name)
            groups["volumes"][material_name] = {
                "phys_group": phys_group,
                "tags": new_tags,
            }

    # --- Outer boundary from airbox ---
    if "airbox" in groups["volumes"]:
        airbox_tags = groups["volumes"]["airbox"]["tags"]
        if airbox_tags:
            try:
                _, simulation_boundary = kernel.getSurfaceLoops(airbox_tags[0])
                if simulation_boundary:
                    boundary_tags = list(next(iter(simulation_boundary)))
                    phys_group = gmsh_utils.assign_physical_group(
                        2, boundary_tags, "outer_boundary"
                    )
                    groups["outer_boundary"] = {
                        "phys_group": phys_group,
                        "tags": boundary_tags,
                    }
            except Exception:
                logger.warning("Could not extract outer boundary from airbox")

    kernel.synchronize()
    return groups


def _assign_port_groups(
    component,
    stack: LayerStack,
    groups: dict,
) -> None:
    """Find existing volume boundary faces at port locations.

    Assign them as physical groups.

    After fragmentation, the layer volumes have boundary faces at each
    component port.  This function identifies those faces by matching
    their bounding box to the expected port rectangle, then assigns
    them to named surface physical groups (``port_<name>``).

    Modifies *groups* in-place, adding a ``"port_surfaces"`` key.
    """
    from gsim.common.mesh.geometry import _resolve_port_layer

    # Collect all boundary surfaces of post-fragment layer volumes
    layer_boundary_surfs: set[int] = set()
    for layer_info in groups["layer_volumes"].values():
        for vtag in layer_info["tags"]:
            try:
                boundary = gmsh.model.getBoundary([(3, vtag)], oriented=False)
                for _, stag in boundary:
                    layer_boundary_surfs.add(stag)
            except Exception:
                pass

    if not layer_boundary_surfs:
        return

    # Cache bounding boxes for all boundary surfaces
    surf_bboxes: dict[int, tuple[float, ...]] = {}
    import contextlib

    for stag in layer_boundary_surfs:
        with contextlib.suppress(Exception):
            surf_bboxes[stag] = gmsh.model.getBoundingBox(2, stag)

    tol = 0.02  # bounding-box match tolerance (um)

    groups["port_surfaces"] = {}

    for port in component.ports:
        resolved = _resolve_port_layer(port, component, stack)
        if resolved is None:
            continue
        layer_name, zmin, zmax = resolved

        cx, cy = float(port.center[0]), float(port.center[1])
        hw = float(port.width) / 2
        angle = float(port.orientation) % 360

        # Expected port rectangle bounds
        if angle < 45 or angle >= 315 or (135 <= angle < 225):
            # East/West → YZ plane at x=cx
            exp = (cx, cy - hw, zmin, cx, cy + hw, zmax)
        else:
            # North/South → XZ plane at y=cy
            exp = (cx - hw, cy, zmin, cx + hw, cy, zmax)

        matching_tags: list[int] = []
        for stag, bbox in surf_bboxes.items():
            sxmin, symin, szmin, sxmax, symax, szmax = bbox
            if (
                abs(sxmin - exp[0]) < tol
                and abs(symin - exp[1]) < tol
                and abs(szmin - exp[2]) < tol
                and abs(sxmax - exp[3]) < tol
                and abs(symax - exp[4]) < tol
                and abs(szmax - exp[5]) < tol
            ):
                matching_tags.append(stag)

        if matching_tags:
            phys_group = gmsh_utils.assign_physical_group(
                2, matching_tags, f"port_{port.name}"
            )
            groups["port_surfaces"][port.name] = {
                "phys_group": phys_group,
                "tags": matching_tags,
                "center": [cx, cy],
                "width": float(port.width),
                "orientation": angle,
                "layer": layer_name,
                "z_range": [zmin, zmax],
            }
            logger.info(
                "Port '%s': phys_group=%d (%d face(s)) on layer '%s'",
                port.name,
                phys_group,
                len(matching_tags),
                layer_name,
            )
        else:
            logger.warning(
                "Port '%s': no matching boundary face found at center=(%.3f, %.3f)",
                port.name,
                cx,
                cy,
            )

    gmsh.model.occ.synchronize()


def _setup_generic_refinement(
    kernel,
    groups: dict,
    refined_cellsize: float,
    max_cellsize: float,
) -> None:
    """Set up mesh refinement near layer volume boundaries."""
    boundary_lines: list[int] = []

    # Collect boundary lines from all layer volumes
    for layer_info in groups["layer_volumes"].values():
        for vol_tag in layer_info["tags"]:
            # Get surfaces bounding this volume
            surfs = gmsh.model.getBoundary([(3, vol_tag)], oriented=False)
            for _dim, surf_tag in surfs:
                try:
                    lines = gmsh_utils.get_boundary_lines(surf_tag, kernel)
                    boundary_lines.extend(lines)
                except Exception:
                    pass

    field_ids: list[int] = []
    if boundary_lines:
        field_id = gmsh_utils.setup_mesh_refinement(
            boundary_lines, refined_cellsize, max_cellsize
        )
        field_ids.append(field_id)

    if field_ids:
        gmsh_utils.finalize_mesh_fields(field_ids)


def collect_mesh_stats() -> dict:
    """Collect mesh statistics from gmsh after mesh generation.

    Must be called while gmsh is initialized and the mesh is generated.

    Returns:
        Dict with mesh statistics including:
        - bbox: Bounding box coordinates
        - nodes: Number of nodes
        - elements: Total element count
        - tetrahedra: Tet count
        - quality: Shape quality metrics (gamma)
        - sicn: Signed Inverse Condition Number
        - edge_length: Min/max edge lengths
        - groups: Physical group info
    """
    stats: dict = {}

    # Get bounding box
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        stats["bbox"] = {
            "xmin": xmin,
            "ymin": ymin,
            "zmin": zmin,
            "xmax": xmax,
            "ymax": ymax,
            "zmax": zmax,
        }
    except Exception:
        pass

    # Get node count
    try:
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        stats["nodes"] = len(node_tags)
    except Exception:
        pass

    # Get element counts and collect tet tags for quality
    tet_tags = []
    try:
        element_types, element_tags, _ = gmsh.model.mesh.getElements()
        total_elements = sum(len(tags) for tags in element_tags)
        stats["elements"] = total_elements

        # Count tetrahedra (type 4) and save tags
        for etype, tags in zip(element_types, element_tags, strict=False):
            if etype == 4:  # 4-node tetrahedron
                stats["tetrahedra"] = len(tags)
                tet_tags = list(tags)
    except Exception:
        pass

    # Get mesh quality for tetrahedra
    if tet_tags:
        # Gamma: inscribed/circumscribed radius ratio (shape quality)
        try:
            qualities = gmsh.model.mesh.getElementQualities(tet_tags, "gamma")
            if len(qualities) > 0:
                stats["quality"] = {
                    "min": round(min(qualities), 3),
                    "max": round(max(qualities), 3),
                    "mean": round(sum(qualities) / len(qualities), 3),
                }
        except Exception:
            pass

        # SICN: Signed Inverse Condition Number (negative = invalid element)
        try:
            sicn = gmsh.model.mesh.getElementQualities(tet_tags, "minSICN")
            if len(sicn) > 0:
                sicn_min = min(sicn)
                invalid_count = sum(1 for s in sicn if s < 0)
                stats["sicn"] = {
                    "min": round(sicn_min, 3),
                    "mean": round(sum(sicn) / len(sicn), 3),
                    "invalid": invalid_count,
                }
        except Exception:
            pass

        # Edge lengths
        try:
            min_edges = gmsh.model.mesh.getElementQualities(tet_tags, "minEdge")
            max_edges = gmsh.model.mesh.getElementQualities(tet_tags, "maxEdge")
            if len(min_edges) > 0 and len(max_edges) > 0:
                stats["edge_length"] = {
                    "min": round(min(min_edges), 3),
                    "max": round(max(max_edges), 3),
                }
        except Exception:
            pass

    # Get physical groups with tags
    try:
        phys_groups: dict[str, list] = {"volumes": [], "surfaces": []}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            entry = {"name": name, "tag": tag}
            if dim == 3:
                phys_groups["volumes"].append(entry)
            elif dim == 2:
                phys_groups["surfaces"].append(entry)
        stats["groups"] = phys_groups
    except Exception:
        pass

    return stats


__all__ = ["collect_mesh_stats", "generate_mesh"]
