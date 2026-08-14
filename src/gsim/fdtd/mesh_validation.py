"""Strict validation for ZapFDTD-compatible Gmsh artifacts."""

from __future__ import annotations

from pathlib import Path

import meshio

from gsim.fdtd.models import FDTDGeometryError, MeshManifest


def _manifest_names(manifest: MeshManifest) -> set[str]:
    """Return all physical names expected in a mesh."""
    return {
        *manifest.volumes,
        *manifest.layers,
        *(port.physical_name for port in manifest.ports.values()),
    }


def validate_mesh(mesh_path: Path, manifest: MeshManifest) -> None:
    """Preflight the strict MSH 2.2 groups and elements ZapFDTD reads."""
    lines = mesh_path.read_text(encoding="utf8").splitlines()
    try:
        mesh_format_index = lines.index("$MeshFormat")
        nodes_index = lines.index("$Nodes")
    except ValueError as error:
        raise FDTDGeometryError("Mesh is missing MSH 2.2 sections.") from error
    if lines[mesh_format_index + 1].strip() != "2.2 0 8":
        raise FDTDGeometryError("ZapFDTD requires ASCII Gmsh MSH 2.2 output.")
    node_count = int(lines[nodes_index + 1])
    node_ids = [
        int(lines[nodes_index + 2 + index].split()[0]) for index in range(node_count)
    ]
    if node_ids != list(range(1, node_count + 1)):
        raise FDTDGeometryError("ZapFDTD requires sequential one-based node IDs.")

    mesh = meshio.read(mesh_path)
    actual_names = set(mesh.field_data)
    expected_names = _manifest_names(manifest)
    if actual_names != expected_names:
        raise FDTDGeometryError(
            f"Mesh physical names {sorted(actual_names)} do not match manifest "
            f"{sorted(expected_names)}."
        )
    physical_data = mesh.cell_data_dict.get("gmsh:physical", {})
    for group in [*manifest.volumes.values(), *manifest.layers.values()]:
        field_tag, dimension = mesh.field_data[group.name]
        tetra_tags = physical_data.get("tetra", [])
        if (
            dimension != 3
            or field_tag != group.physical_tag
            or not any(tag == field_tag for tag in tetra_tags)
        ):
            raise FDTDGeometryError(
                f"Volume group {group.name!r} has no linear tetrahedra."
            )
    for port in manifest.ports.values():
        field_tag, dimension = mesh.field_data[port.physical_name]
        triangle_tags = physical_data.get("triangle", [])
        if (
            dimension != 2
            or field_tag != port.physical_tag
            or not any(tag == field_tag for tag in triangle_tags)
        ):
            raise FDTDGeometryError(
                f"Port group {port.physical_name!r} has no linear triangles."
            )


__all__ = ["validate_mesh"]
