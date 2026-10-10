"""Mesh planar component electrodes between air and a dielectric with meshwell."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gdsfactory import Component
    from meshwell.geometry_entity import GeometryEntity
    from meshwell.model import ModelManager
    from shapely import Polygon

    from gsim.palace.mesh.generator import MeshResult


def mesh_sheets(
    component: Component,
    *,
    conductor_layer: tuple[int, int],
    terminal_ports: dict[str, str],
    domain_bounds: tuple[float, float, float, float],
    height: float,
    near_mesh: float,
    far_mesh: float,
    path: Path,
    cross_section: bool = False,
    minimum_feature_elements: float = 4,
) -> MeshResult:
    """Mesh zero-thickness conductors on silicon, selecting terminals by ports.

    All dimensions are in micrometres. Connected metal touching a selected port
    is its terminal; remaining metal is grounded. ``cross_section`` extracts
    the transverse plane at the domain's x midpoint for uniform lines, with y
    mapped to the mesh x coordinate. The domain extends ``height`` above and
    below the metal. Requires the optional ``meshwell`` extra.

    Args:
        component: Layout containing conductor polygons and named ports.
        conductor_layer: GDS layer and datatype of the metal sheets.
        terminal_ports: Terminal names mapped to component port names.
        domain_bounds: Explicit (xmin, ymin, xmax, ymax) domain envelope.
        height: Air and silicon thicknesses.
        near_mesh: Target conductor mesh size.
        far_mesh: Target far-field mesh size.
        path: Output MSH 2.2 file.
        cross_section: Generate a 2D rather than a 3D mesh.
        minimum_feature_elements: Minimum elements along isolated ground strips.

    Returns:
        Mesh and physical groups ready for Palace config generation.

    Raises:
        ValueError: If terminals are missing, shorted or outside the domain.
    """
    from shapely import Point, Polygon, box, union_all

    from gsim.palace.mesh.geometry import extract_pec_polygons

    if min(height, near_mesh, far_mesh, minimum_feature_elements) <= 0:
        raise ValueError("Mesh dimensions and feature resolution must be positive")
    if not terminal_ports or "ground" in terminal_ports:
        raise ValueError("Select at least one terminal; 'ground' is reserved")
    footprint = box(*domain_bounds)
    if footprint.is_empty or footprint.area <= 0:
        raise ValueError("The domain must have positive area")
    metal = union_all(
        [
            Polygon(
                list(zip(xs, ys, strict=True)),
                [list(zip(hx, hy, strict=True)) for hx, hy in holes],
            )
            for xs, ys, holes in extract_pec_polygons(component, conductor_layer)
        ]
    )
    polygons = list(metal.geoms) if hasattr(metal, "geoms") else [metal]
    selected = set()
    sheets = {}
    for name, port_name in terminal_ports.items():
        point = Point(component.ports[port_name].center)
        matches = [
            i
            for i, polygon in enumerate(polygons)
            if polygon.distance(point) <= component.kcl.dbu
        ]
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError(
                f"Terminal {name!r} must touch exactly one unselected conductor"
            )
        selected.add(matches[0])
        sheets[name] = polygons[matches[0]]
    remaining = [p for i, p in enumerate(polygons) if i not in selected]
    if remaining:
        sheets["ground"] = union_all(remaining)
    if not footprint.buffer(component.kcl.dbu).covers(metal):
        raise ValueError("The domain must enclose every conductor")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = (_mesh_cross_section if cross_section else _mesh_sheets)(
        sheets,
        footprint,
        height=height,
        near_mesh=near_mesh,
        far_mesh=far_mesh,
        path=path,
        minimum_feature_elements=minimum_feature_elements,
    )
    result.metadata.update(
        dimension=2 if cross_section else 3, terminal_names=tuple(terminal_ports)
    )
    return result


def _mesh_sheets(
    sheets: dict[str, Polygon],
    footprint: Polygon,
    *,
    height: float,
    near_mesh: float,
    far_mesh: float,
    path: Path,
    minimum_feature_elements: float,
) -> MeshResult:
    """Mesh zero-thickness conductors between equal-height air and substrate volumes."""
    import gmsh
    from meshwell.model import ModelManager
    from meshwell.polyprism import PolyPrism
    from meshwell.polysurface import (
        PolySurface,
    )

    if gmsh.isInitialized():
        raise RuntimeError(
            "Meshwell requires its own Gmsh session; finalize the existing session "
            "first"
        )
    model = ModelManager(n_threads=1, filename=str(path.with_suffix("")))
    entities: list[GeometryEntity] = [
        PolyPrism(footprint, {0: 0, height: 0}, physical_name="air"),
        PolyPrism(footprint, {-height: 0, 0: 0}, physical_name="silicon"),
        *(PolySurface(polygon, physical_name=name) for name, polygon in sheets.items()),
    ]
    return _mesh_entities(
        model,
        entities,
        sheets,
        dim=3,
        near_mesh=near_mesh,
        far_mesh=far_mesh,
        path=path,
        minimum_feature_elements=minimum_feature_elements,
    )


def _mesh_cross_section(
    sheets: dict[str, Polygon],
    footprint: Polygon,
    *,
    height: float,
    near_mesh: float,
    far_mesh: float,
    path: Path,
    minimum_feature_elements: float,
) -> MeshResult:
    """Mesh the transverse section of a uniform line with meshwell curves."""
    import gmsh
    from meshwell.model import ModelManager
    from meshwell.polyline import PolyLine
    from meshwell.polysurface import PolySurface
    from shapely import LineString, box

    if gmsh.isInitialized():
        raise RuntimeError(
            "Meshwell requires its own Gmsh session; finalize the existing session "
            "first"
        )
    lower, upper = footprint.bounds[1], footprint.bounds[3]
    model = ModelManager(n_threads=1, filename=str(path.with_suffix("")))
    # Different snapping grids leave conductor edges outside the dielectric mesh.
    entities: list[GeometryEntity] = [
        PolySurface(
            box(lower, 0, upper, height), physical_name="air", point_tolerance=1e-8
        ),
        PolySurface(
            box(lower, -height, upper, 0),
            physical_name="silicon",
            point_tolerance=1e-8,
        ),
    ]
    for name, polygon in sheets.items():
        polygons = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
        section = LineString(
            [(footprint.centroid.x, lower), (footprint.centroid.x, upper)]
        )
        lines = []
        for part in polygons:
            intersection = part.intersection(section)
            pieces = (
                list(intersection.geoms)
                if hasattr(intersection, "geoms")
                else [intersection]
            )
            lines.extend(
                LineString([(y, 0) for _, y in piece.coords])
                for piece in pieces
                if piece.geom_type == "LineString" and not piece.is_empty
            )
        if not lines:
            raise ValueError(f"Conductor {name!r} does not cross the selected plane")
        entities.append(PolyLine(lines, physical_name=name, point_tolerance=1e-8))
    return _mesh_entities(
        model,
        entities,
        sheets,
        dim=2,
        near_mesh=near_mesh,
        far_mesh=far_mesh,
        path=path,
        minimum_feature_elements=minimum_feature_elements,
    )


def _mesh_entities(
    model: ModelManager,
    entities: list[GeometryEntity],
    sheets: dict[str, Polygon],
    *,
    dim: Literal[2, 3],
    near_mesh: float,
    far_mesh: float,
    path: Path,
    minimum_feature_elements: float,
) -> MeshResult:
    """Keep meshing, physical groups and Palace export consistent across dimensions."""
    import gmsh
    from meshwell.resolution import ThresholdField

    from gsim.palace.mesh.generator import MeshResult

    try:
        model.cad.process_entities(entities, interface_delimiter="___")
        field = ThresholdField(
            apply_to="surfaces" if dim == 3 else "curves",
            sizemin=near_mesh,
            sizemax=far_mesh,
            distmin=1,
            distmax=25,
        )
        resolutions = {name: [field] for name in sheets if name != "ground"}
        ground_parts = (
            list(sheets["ground"].geoms)
            if "ground" in sheets and hasattr(sheets["ground"], "geoms")
            else [sheets["ground"]]
            if "ground" in sheets
            else []
        )
        strip = min(
            (part.bounds[3] - part.bounds[1] for part in ground_parts), default=far_mesh
        )
        if dim == 2 and len(ground_parts) > 2:
            # Refine the inner strip without refining the long outer ground rails.
            resolutions["ground"] = [
                ThresholdField(
                    apply_to="curves",
                    max_mass=strip * 1.001,
                    sizemin=min(near_mesh, strip / minimum_feature_elements),
                    sizemax=far_mesh,
                    distmin=min(1, strip / 4),
                    distmax=25,
                )
            ]
        model.mesh.process_geometry(
            dim=dim,
            default_characteristic_length=far_mesh,
            resolution_specs=resolutions,
            verbosity=0,
        )
        groups = {
            "volumes": {},
            "pec_surfaces": {},
            "conductor_surfaces": {},
            "port_surfaces": {},
            "boundary_surfaces": {},
        }
        # Palace rejects faces exported twice as conductor and dielectric interface.
        gmsh.model.removePhysicalGroups(
            [
                (group_dim, tag)
                for group_dim, tag in gmsh.model.getPhysicalGroups(dim - 1)
                if gmsh.model.getPhysicalName(group_dim, tag) not in sheets
            ]
        )
        for group_dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(group_dim, tag)
            if group_dim == dim:
                groups["volumes"][name] = {"phys_group": tag}
            elif name in sheets:
                groups["pec_surfaces"][name] = {"phys_group": tag}
        if set(groups["pec_surfaces"]) != set(sheets) or set(groups["volumes"]) != {
            "air",
            "silicon",
        }:
            raise ValueError(
                "Mesh physical groups do not match conductors and dielectric domains"
            )
        count = sum(len(elements) for elements in gmsh.model.mesh.getElements(dim)[1])
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        model.save_to_mesh(path)
        return MeshResult(
            mesh_path=path,
            output_dir=path.parent,
            groups=groups,
            mesh_stats={"tetrahedra" if dim == 3 else "triangles": count},
        )
    finally:
        model.finalize()
