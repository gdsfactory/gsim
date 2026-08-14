"""Coarse Gmsh artifact generation for ZapFDTD voxelization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import radians, tan
from pathlib import Path
from typing import Any

import gmsh
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient
from shapely.validation import explain_validity

from gsim.common.pdk import ResolvedLayer, ResolvedPassivePcell, ResolvedPort
from gsim.fdtd.mesh_validation import validate_mesh
from gsim.fdtd.models import (
    FDTDGeometryError,
    MeshGroup,
    MeshManifest,
    PortMeshGroup,
)

_UM_TO_NM = 1000.0
_GEOMETRY_TOLERANCE_NM = 1e-3


def _single_polygon(layer: ResolvedLayer) -> Polygon:
    """Return one valid connected polygon for an initial FDTD layer."""
    geometry = layer.geometry
    if isinstance(geometry, Polygon):
        polygon = geometry
    elif isinstance(geometry, MultiPolygon) and len(geometry.geoms) == 1:
        polygon = geometry.geoms[0]
    else:
        count = len(geometry.geoms) if hasattr(geometry, "geoms") else 0
        raise FDTDGeometryError(
            f"Layer {layer.key!r} has {count} disconnected polygons; the initial "
            "FDTD backend requires one connected solid per layer."
        )
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise FDTDGeometryError(
            f"Layer {layer.key!r} has invalid polygon geometry: "
            f"{explain_validity(polygon)}."
        )
    return polygon


def _scaled_ring(coordinates: Iterable[tuple[float, ...]]) -> list[tuple[float, float]]:
    """Convert one Shapely ring from micrometers to nanometers."""
    points = [
        (float(point[0]) * _UM_TO_NM, float(point[1]) * _UM_TO_NM)
        for point in coordinates
    ]
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    return points


def _add_wire(kernel: Any, polygon: Polygon, z_nm: float) -> int:
    """Create one closed OCC wire for lofting."""
    points = _scaled_ring(orient(polygon, sign=1.0).exterior.coords)
    if len(points) < 3:
        raise FDTDGeometryError("A loft profile has fewer than three vertices.")
    first_index = min(
        range(len(points)),
        key=lambda index: (points[index][0], points[index][1]),
    )
    points = points[first_index:] + points[:first_index]
    point_tags = [kernel.addPoint(x, y, z_nm) for x, y in points]
    line_tags = [
        kernel.addLine(point_tags[index], point_tags[(index + 1) % len(point_tags)])
        for index in range(len(point_tags))
    ]
    return kernel.addWire(line_tags, checkClosed=True)


def _profile_offset_um(layer: ResolvedLayer, normalized_z: float) -> float:
    """Return the PDK sidewall offset at one normalized z position."""
    return (
        (layer.width_to_z - normalized_z)
        * abs(layer.thickness)
        * tan(radians(layer.sidewall_angle))
    )


def _condition_profile_at_ports(
    profile: Polygon,
    ports: list[ResolvedPort],
    offset_um: float,
) -> Polygon:
    """Extend and clip a profile so every port is a full planar end face."""
    for port in ports:
        normal_axis = next(index for index, value in enumerate(port.normal) if value)
        if normal_axis not in {0, 1}:
            raise FDTDGeometryError(
                f"Port {port.name!r} is not in the component plane."
            )
        transverse_axis = 1 - normal_axis
        half_width = port.width / 2 + offset_um
        if half_width <= 0:
            raise FDTDGeometryError(
                f"Layer {port.layer_key!r} sidewall offset closes port {port.name!r}."
            )

        target = port.center[normal_axis]
        transverse_lower = port.center[transverse_axis] - half_width
        transverse_upper = port.center[transverse_axis] + half_width
        xmin, ymin, xmax, ymax = profile.bounds
        epsilon = _GEOMETRY_TOLERANCE_NM / _UM_TO_NM
        if normal_axis == 0:
            edge = xmin if port.normal[0] < 0 else xmax
            inner = max(target + epsilon, edge + epsilon)
            if port.normal[0] > 0:
                inner = min(target - epsilon, edge - epsilon)
            extension = box(
                min(target, inner),
                transverse_lower,
                max(target, inner),
                transverse_upper,
            )
        else:
            edge = ymin if port.normal[1] < 0 else ymax
            inner = max(target + epsilon, edge + epsilon)
            if port.normal[1] > 0:
                inner = min(target - epsilon, edge - epsilon)
            extension = box(
                transverse_lower,
                min(target, inner),
                transverse_upper,
                max(target, inner),
            )
        profile = profile.union(extension)

        xmin, ymin, xmax, ymax = profile.bounds
        margin = max(xmax - xmin, ymax - ymin, port.width, 1.0)
        if normal_axis == 0 and port.normal[0] < 0:
            clip = box(target, ymin - margin, xmax + margin, ymax + margin)
        elif normal_axis == 0:
            clip = box(xmin - margin, ymin - margin, target, ymax + margin)
        elif port.normal[1] < 0:
            clip = box(xmin - margin, target, xmax + margin, ymax + margin)
        else:
            clip = box(xmin - margin, ymin - margin, xmax + margin, target)
        profile = profile.intersection(clip)

    profile = profile.simplify(
        10 * _GEOMETRY_TOLERANCE_NM / _UM_TO_NM,
        preserve_topology=True,
    )
    if not isinstance(profile, Polygon) or profile.is_empty or not profile.is_valid:
        raise FDTDGeometryError("Port conditioning produced invalid layer geometry.")
    return profile


def _offset_profile(
    polygon: Polygon,
    layer: ResolvedLayer,
    ports: list[ResolvedPort],
    normalized_z: float,
) -> Polygon:
    """Apply the PDK sidewall offset and preserve full port end faces."""
    offset_um = _profile_offset_um(layer, normalized_z)
    profile = polygon.buffer(offset_um, join_style=2)
    if not isinstance(profile, Polygon) or profile.is_empty or not profile.is_valid:
        raise FDTDGeometryError(
            f"Layer {layer.key!r} sidewall profile becomes empty or disconnected "
            f"at normalized z={normalized_z:g}."
        )
    return _condition_profile_at_ports(profile, ports, offset_um)


def _add_layer_volume(
    kernel: Any,
    layer: ResolvedLayer,
    ports: list[ResolvedPort],
) -> list[int]:
    """Create one vertical or sidewall-tapered OCC layer volume."""
    if layer.bias not in (None, 0, 0.0) or layer.z_to_bias is not None:
        raise FDTDGeometryError(
            f"Layer {layer.key!r} uses bias or z_to_bias, which is not supported "
            "by the initial FDTD mesh writer."
        )
    if not 0 <= layer.width_to_z <= 1:
        raise FDTDGeometryError(
            f"Layer {layer.key!r} width_to_z must be between 0 and 1."
        )
    if abs(layer.sidewall_angle) >= 80:
        raise FDTDGeometryError(
            f"Layer {layer.key!r} sidewall angle is too steep to mesh safely."
        )

    polygon = _single_polygon(layer)
    z_lower_um, z_upper_um = layer.z_bounds
    z_lower_nm = z_lower_um * _UM_TO_NM
    z_upper_nm = z_upper_um * _UM_TO_NM
    if layer.sidewall_angle == 0:
        from gsim.palace.mesh.gmsh_utils import extrude_polygon

        polygon = _condition_profile_at_ports(polygon, ports, 0.0)
        exterior = _scaled_ring(polygon.exterior.coords)
        hole_coordinates = []
        for interior in polygon.interiors:
            points = _scaled_ring(interior.coords)
            hole_coordinates.append(
                ([point[0] for point in points], [point[1] for point in points])
            )
        volume_tag = extrude_polygon(
            kernel,
            [point[0] for point in exterior],
            [point[1] for point in exterior],
            z_lower_nm,
            z_upper_nm - z_lower_nm,
            holes=hole_coordinates,
        )
        if volume_tag is None:
            raise FDTDGeometryError(f"Could not extrude layer {layer.key!r}.")
        return [volume_tag]

    if polygon.interiors:
        raise FDTDGeometryError(
            f"Layer {layer.key!r} combines holes and sidewalls, which is not "
            "supported by the initial FDTD mesh writer."
        )
    lower_profile = _offset_profile(polygon, layer, ports, 0.0)
    upper_profile = _offset_profile(polygon, layer, ports, 1.0)
    lower_wire = _add_wire(kernel, lower_profile, z_lower_nm)
    upper_wire = _add_wire(kernel, upper_profile, z_upper_nm)
    dimtags = kernel.addThruSections(
        [lower_wire, upper_wire],
        makeSolid=True,
        makeRuled=True,
    )
    volume_tags = [tag for dimension, tag in dimtags if dimension == 3]
    if not volume_tags:
        raise FDTDGeometryError(f"Could not loft layer {layer.key!r}.")
    return volume_tags


def _priority_by_mesh_order(
    layers: Mapping[str, ResolvedLayer],
) -> dict[str, int]:
    """Invert lower-wins PDK mesh order into higher-wins Zap priority."""
    unique_orders = sorted({layer.mesh_order for layer in layers.values()})
    order_priority = {
        mesh_order: len(unique_orders) - index
        for index, mesh_order in enumerate(unique_orders)
    }
    return {name: order_priority[layer.mesh_order] for name, layer in layers.items()}


def _background_bounds_nm(
    resolved: ResolvedPassivePcell,
    background_material: str,
    padding_um: float,
) -> tuple[float, float, float, float, float, float]:
    """Build a port-aligned background box from PDK and component bounds."""
    lower, upper = resolved.bounds
    port_axes = {
        next(index for index, value in enumerate(port.normal) if value)
        for port in resolved.ports.values()
    }
    x_padding = 0.0 if 0 in port_axes else padding_um
    y_padding = 0.0 if 1 in port_axes else padding_um

    background_z_bounds = []
    for level in resolved.layer_stack.layers.values():
        if level.material != background_material or level.thickness == 0:
            continue
        level_zmax = float(level.zmin + level.thickness)
        background_z_bounds.append(
            (min(float(level.zmin), level_zmax), max(float(level.zmin), level_zmax))
        )
    if background_z_bounds:
        z_lower = min(lower[2], *(bounds[0] for bounds in background_z_bounds))
        z_upper = max(upper[2], *(bounds[1] for bounds in background_z_bounds))
    else:
        z_lower = lower[2] - padding_um
        z_upper = upper[2] + padding_um

    return (
        (lower[0] - x_padding) * _UM_TO_NM,
        (lower[1] - y_padding) * _UM_TO_NM,
        z_lower * _UM_TO_NM,
        (upper[0] + x_padding) * _UM_TO_NM,
        (upper[1] + y_padding) * _UM_TO_NM,
        z_upper * _UM_TO_NM,
    )


def _add_physical_group(dimension: int, tags: list[int], name: str) -> int:
    """Create a named Gmsh physical group and return its actual tag."""
    if not tags:
        raise FDTDGeometryError(f"Physical group {name!r} has no entities.")
    physical_tag = gmsh.model.addPhysicalGroup(dimension, tags)
    gmsh.model.setPhysicalName(dimension, physical_tag, name)
    return physical_tag


def _port_surface_tags(
    port: Any,
    volume_tags: list[int],
    claimed_surfaces: set[int],
) -> list[int]:
    """Find the owning layer boundary face at an axis-aligned port plane."""
    normal_axis = next(index for index, value in enumerate(port.normal) if value)
    target_nm = port.center[normal_axis] * _UM_TO_NM
    center_nm = tuple(coordinate * _UM_TO_NM for coordinate in port.center)
    candidates: list[int] = []
    boundary_bounds: list[tuple[int, tuple[float, ...]]] = []
    for volume_tag in volume_tags:
        for dimension, surface_tag in gmsh.model.getBoundary(
            [(3, volume_tag)],
            combined=False,
            oriented=False,
            recursive=False,
        ):
            if dimension != 2 or surface_tag in claimed_surfaces:
                continue
            bounds = gmsh.model.getBoundingBox(2, surface_tag)
            boundary_bounds.append((surface_tag, bounds))
            if (
                abs(bounds[normal_axis] - target_nm) > _GEOMETRY_TOLERANCE_NM
                or abs(bounds[normal_axis + 3] - target_nm) > _GEOMETRY_TOLERANCE_NM
            ):
                continue
            transverse_axes = [axis for axis in range(3) if axis != normal_axis]
            if all(
                bounds[axis] - _GEOMETRY_TOLERANCE_NM
                <= center_nm[axis]
                <= bounds[axis + 3] + _GEOMETRY_TOLERANCE_NM
                for axis in transverse_axes
            ):
                candidates.append(surface_tag)
    if not candidates:
        nearest_bounds = sorted(
            boundary_bounds,
            key=lambda item: min(
                abs(item[1][normal_axis] - target_nm),
                abs(item[1][normal_axis + 3] - target_nm),
            ),
        )[:3]
        raise FDTDGeometryError(
            f"Port {port.name!r} does not coincide with a boundary face of "
            f"layer {port.layer_key!r}; nearest boundary bounds are {nearest_bounds}."
        )
    claimed_surfaces.update(candidates)
    return candidates


def _validate_port_on_background_face(
    port: Any,
    background_bounds: tuple[float, float, float, float, float, float],
) -> None:
    """Require each port plane to lie on the material-union AABB face."""
    axis = next(index for index, value in enumerate(port.normal) if value)
    side = 0 if port.normal[axis] < 0 else 3
    background_face = background_bounds[axis + side]
    port_coordinate = port.center[axis] * _UM_TO_NM
    if abs(background_face - port_coordinate) > _GEOMETRY_TOLERANCE_NM:
        raise FDTDGeometryError(
            f"Port {port.name!r} is not on the background domain face required "
            "for unambiguous ZapFDTD port extrusion."
        )


def generate_mesh(
    resolved: ResolvedPassivePcell,
    mesh_path: Path,
    *,
    background_material: str,
    background_padding_um: float,
    mesh_size_nm: float,
) -> MeshManifest:
    """Generate and validate a coarse Zap-compatible tetrahedral mesh."""
    if background_padding_um <= 0:
        raise FDTDGeometryError("background_padding_um must be positive.")
    if mesh_size_nm <= 0:
        raise FDTDGeometryError("mesh_size_nm must be positive.")
    if "background" in resolved.layers:
        raise FDTDGeometryError("Layer name 'background' is reserved by FDTD.")

    initialized_here = not bool(gmsh.isInitialized())
    if initialized_here:
        gmsh.initialize()
    else:
        gmsh.clear()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("gsim_fdtd")
        kernel = gmsh.model.occ
        background_bounds = _background_bounds_nm(
            resolved,
            background_material,
            background_padding_um,
        )
        for port in resolved.ports.values():
            _validate_port_on_background_face(port, background_bounds)

        xmin, ymin, zmin, xmax, ymax, zmax = background_bounds
        background_tag = kernel.addBox(
            xmin,
            ymin,
            zmin,
            xmax - xmin,
            ymax - ymin,
            zmax - zmin,
        )
        layer_volume_tags = {
            name: _add_layer_volume(
                kernel,
                layer,
                [port for port in resolved.ports.values() if port.layer_key == name],
            )
            for name, layer in resolved.layers.items()
        }
        kernel.synchronize()

        background_physical_tag = _add_physical_group(3, [background_tag], "background")
        priorities = _priority_by_mesh_order(resolved.layers)
        layer_groups = {
            name: MeshGroup(
                name=name,
                physical_tag=_add_physical_group(3, tags, name),
                material=resolved.layers[name].material,
                priority=priorities[name],
            )
            for name, tags in layer_volume_tags.items()
        }
        claimed_surfaces: set[int] = set()
        port_groups = {}
        for name, port in resolved.ports.items():
            surface_tags = _port_surface_tags(
                port,
                layer_volume_tags[port.layer_key],
                claimed_surfaces,
            )
            physical_name = f"port_{name}"
            port_groups[name] = PortMeshGroup(
                name=name,
                physical_name=physical_name,
                physical_tag=_add_physical_group(2, surface_tags, physical_name),
                layer=port.layer_key,
                normal=port.normal,
            )
        manifest = MeshManifest(
            volumes={
                "background": MeshGroup(
                    name="background",
                    physical_tag=background_physical_tag,
                    material=background_material,
                    priority=0,
                )
            },
            layers=layer_groups,
            ports=port_groups,
        )

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_nm)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_nm)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(mesh_path))
    except FDTDGeometryError:
        raise
    except Exception as error:
        raise FDTDGeometryError(f"Gmsh mesh generation failed: {error}") from error
    finally:
        if initialized_here:
            gmsh.finalize()
        else:
            gmsh.clear()

    validate_mesh(mesh_path, manifest)
    return manifest


__all__ = ["generate_mesh", "validate_mesh"]
