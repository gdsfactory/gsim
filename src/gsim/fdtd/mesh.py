"""Coarse Gmsh artifact generation for GDSFactory FDTD voxelization."""

from __future__ import annotations

from collections.abc import Mapping
from math import dist
from pathlib import Path
from typing import Any

import gmsh

from gsim.common.pdk import ResolvedLayer, ResolvedPassivePcell, ResolvedPort
from gsim.fdtd.domain import Padding, background_bounds_nm
from gsim.fdtd.mesh_geometry import GEOMETRY_TOLERANCE_NM, UM_TO_NM
from gsim.fdtd.mesh_loft import add_layer_volumes
from gsim.fdtd.mesh_validation import validate_mesh
from gsim.fdtd.models import (
    FDTDGeometryError,
    MeshGroup,
    MeshManifest,
    PortMeshGroup,
)

_GMSH_OPTIONS_CHANGED = (
    "General.Terminal",
    "Mesh.Binary",
    "Mesh.Algorithm",
    "Mesh.Algorithm3D",
    "Mesh.ElementOrder",
    "Mesh.MeshSizeExtendFromBoundary",
    "Mesh.MeshSizeFromCurvature",
    "Mesh.MeshSizeFromPoints",
    "Mesh.MeshSizeMax",
    "Mesh.MeshSizeMin",
    "Mesh.MshFileVersion",
    "Mesh.SaveAll",
)


def _snapshot_gmsh_options() -> dict[str, float]:
    """Capture options that FDTD meshing temporarily changes."""
    return {
        option_name: gmsh.option.getNumber(option_name)
        for option_name in _GMSH_OPTIONS_CHANGED
    }


def _restore_gmsh_options(option_values: Mapping[str, float]) -> None:
    """Restore options owned by an existing caller Gmsh session."""
    for option_name, value in option_values.items():
        gmsh.option.setNumber(option_name, value)


def _force_lf_line_endings(path: Path) -> None:
    """Rewrite an ASCII file with LF line endings.

    Gmsh writes ASCII meshes in text mode, so they come out CRLF on Windows.
    The FDTD solver's MSH reader only accepts LF.

    ASCII only: this relies on ``generate_mesh`` setting ``Mesh.Binary = 0``.
    On a binary mesh it would corrupt any data bytes that happen to be CRLF.
    """
    content = path.read_bytes()
    if b"\r\n" in content:
        path.write_bytes(content.replace(b"\r\n", b"\n"))


def _priority_by_mesh_order(
    layers: Mapping[str, ResolvedLayer],
) -> dict[str, int]:
    """Invert lower-wins PDK mesh order into higher-wins GDSFactory FDTD priority."""
    unique_orders = sorted({layer.mesh_order for layer in layers.values()})
    order_priority = {
        mesh_order: len(unique_orders) - index
        for index, mesh_order in enumerate(unique_orders)
    }
    return {name: order_priority[layer.mesh_order] for name, layer in layers.items()}


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
    if normal_axis not in {0, 1}:
        raise FDTDGeometryError(
            f"Guided port {port.name!r} must have an in-plane normal."
        )
    target_nm = port.center[normal_axis] * UM_TO_NM
    transverse_axis = 1 - normal_axis
    transverse_center_nm = port.center[transverse_axis] * UM_TO_NM
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
                abs(bounds[normal_axis] - target_nm) > GEOMETRY_TOLERANCE_NM
                or abs(bounds[normal_axis + 3] - target_nm) > GEOMETRY_TOLERANCE_NM
            ):
                continue
            if (
                bounds[transverse_axis] - GEOMETRY_TOLERANCE_NM
                <= transverse_center_nm
                <= bounds[transverse_axis + 3] + GEOMETRY_TOLERANCE_NM
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
    candidates = sorted(set(candidates))
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
    port_coordinate = port.center[axis] * UM_TO_NM
    if abs(background_face - port_coordinate) > GEOMETRY_TOLERANCE_NM:
        raise FDTDGeometryError(
            f"Port {port.name!r} is not on the background domain face required "
            "for unambiguous GDSFactory FDTD port extrusion."
        )


def _guided_layer_key(port: ResolvedPort) -> str:
    """Return the physical owner required by a guided port."""
    if port.layer_key is None:
        raise FDTDGeometryError(
            f"Guided port {port.name!r} has no owning physical layer."
        )
    return port.layer_key


def _material_boundary_entities(
    layer_volume_tags: Mapping[str, list[int]],
) -> tuple[set[int], set[int]]:
    """Return every material surface and curve that must receive mesh elements."""
    surfaces: set[int] = set()
    for volume_tags in layer_volume_tags.values():
        for dimension, tag in gmsh.model.getBoundary(
            [(3, volume_tag) for volume_tag in volume_tags],
            combined=False,
            oriented=False,
            recursive=False,
        ):
            if dimension == 2:
                surfaces.add(tag)
    curves = {
        tag
        for dimension, tag in gmsh.model.getBoundary(
            [(2, surface_tag) for surface_tag in surfaces],
            combined=False,
            oriented=False,
            recursive=False,
        )
        if dimension == 1
    }
    return surfaces, curves


def _entity_element_count(dimension: int, tag: int) -> int:
    """Return the number of generated elements attached to one CAD entity."""
    return sum(
        len(element_tags)
        for element_tags in gmsh.model.mesh.getElements(dimension, tag)[1]
    )


def _validate_material_entity_mesh(
    layer_volume_tags: Mapping[str, list[int]],
    geometry_tolerance_nm: float,
) -> None:
    """Require complete CAD coverage and bound material-boundary projection error."""
    surfaces, curves = _material_boundary_entities(layer_volume_tags)
    unmeshed_curves = [
        tag for tag in sorted(curves) if _entity_element_count(1, tag) == 0
    ]
    unmeshed_surfaces = [
        tag for tag in sorted(surfaces) if _entity_element_count(2, tag) == 0
    ]
    unmeshed_volumes = [
        tag
        for volume_tags in layer_volume_tags.values()
        for tag in volume_tags
        if _entity_element_count(3, tag) == 0
    ]
    if unmeshed_curves or unmeshed_surfaces or unmeshed_volumes:
        raise FDTDGeometryError(
            "Gmsh did not cover every material CAD entity: "
            f"curves={unmeshed_curves[:5]}, surfaces={unmeshed_surfaces[:5]}, "
            f"volumes={unmeshed_volumes[:5]}."
        )

    maximum_deviation_nm = 0.0
    for surface_tag in surfaces:
        _, coordinates, _ = gmsh.model.mesh.getNodes(
            2,
            surface_tag,
            includeBoundary=True,
        )
        for coordinate_index in range(0, len(coordinates), 3):
            point = coordinates[coordinate_index : coordinate_index + 3]
            closest_point = gmsh.model.getClosestPoint(2, surface_tag, point)[0]
            maximum_deviation_nm = max(
                maximum_deviation_nm,
                dist(point, closest_point),
            )
    if maximum_deviation_nm > geometry_tolerance_nm:
        raise FDTDGeometryError(
            "Material-boundary mesh exceeds geometry_tolerance_nm: "
            f"{maximum_deviation_nm:.6g} nm > {geometry_tolerance_nm:.6g} nm."
        )


def generate_mesh(
    resolved: ResolvedPassivePcell,
    mesh_path: Path,
    *,
    background_material: str,
    padding: Padding,
    mesh_size_nm: float,
    geometry_tolerance_nm: float,
    padding_x: Padding | None = None,
    padding_y: Padding | None = None,
    padding_z: Padding | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
) -> MeshManifest:
    """Generate an exact-boundary transfer mesh with coarse volume tetrahedra."""
    if mesh_size_nm <= 0:
        raise FDTDGeometryError("mesh_size_nm must be positive.")
    if not 0 < geometry_tolerance_nm <= 30:
        raise FDTDGeometryError(
            "geometry_tolerance_nm must be greater than 0 and at most 30."
        )
    if "background" in resolved.layers:
        raise FDTDGeometryError(
            "Layer name 'background' is reserved by GDSFactory FDTD."
        )

    background_bounds = background_bounds_nm(
        resolved,
        background_material,
        padding,
        padding_x=padding_x,
        padding_y=padding_y,
        padding_z=padding_z,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
    )
    initialized_here = not bool(gmsh.isInitialized())
    caller_option_values = {} if initialized_here else _snapshot_gmsh_options()
    if initialized_here:
        gmsh.initialize(readConfigFiles=False)
    else:
        gmsh.clear()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("gsim_fdtd")
        kernel = gmsh.model.occ
        guided_ports = {
            name: port for name, port in resolved.ports.items() if not port.is_vertical
        }
        for port in guided_ports.values():
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
            name: add_layer_volumes(
                kernel,
                layer,
                [port for port in guided_ports.values() if port.layer_key == name],
                geometry_tolerance_nm=geometry_tolerance_nm,
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
        for name, port in guided_ports.items():
            layer_key = _guided_layer_key(port)
            surface_tags = _port_surface_tags(
                port,
                layer_volume_tags[layer_key],
                claimed_surfaces,
            )
            physical_name = f"port_{name}"
            port_groups[name] = PortMeshGroup(
                name=name,
                physical_name=physical_name,
                physical_tag=_add_physical_group(2, surface_tags, physical_name),
                layer=layer_key,
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
            domain_min_nm=(xmin, ymin, zmin),
            domain_max_nm=(xmax, ymax, zmax),
        )

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_nm)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)
        _validate_material_entity_mesh(layer_volume_tags, geometry_tolerance_nm)
        gmsh.write(str(mesh_path))
        _force_lf_line_endings(mesh_path)
    except FDTDGeometryError:
        raise
    except Exception as error:
        raise FDTDGeometryError(f"Gmsh mesh generation failed: {error}") from error
    finally:
        if initialized_here:
            gmsh.finalize()
        else:
            gmsh.clear()
            _restore_gmsh_options(caller_option_values)

    validate_mesh(mesh_path, manifest)
    return manifest


__all__ = ["background_bounds_nm", "generate_mesh", "validate_mesh"]
