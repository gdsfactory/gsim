"""Input-mesh sizing hints and extensible JSON metadata for Palace jobs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import gmsh
import numpy as np

METADATA_FILENAME = "metadata.json"
_EDGE_VERTICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_FACE_VERTICES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def tetrahedral_topology(blocks: list[tuple[int, np.ndarray, np.ndarray]]) -> dict:
    """Count unique edges/faces in tetrahedra, including shared interior faces.

    Only corner node tags define topology, even for curved geometry. Sorting
    node tags identifies shared entities across orientations and element blocks.
    No coordinates are merged and no Gmsh connectivity is changed. Process edges
    and faces separately so both expanded connectivity arrays are not retained.
    """
    corners = []
    for element_type, _tags, nodes in blocks:
        node_count = gmsh.model.mesh.getElementProperties(element_type)[3]
        corners.append(nodes.reshape(-1, node_count)[:, :4])
    tetrahedra = np.concatenate(corners)
    counts = {}
    for name, vertices in (
        ("edges", _EDGE_VERTICES),
        ("triangular_faces", _FACE_VERTICES),
    ):
        entities = tetrahedra[:, vertices].reshape(-1, len(vertices[0]))
        entities.sort(axis=1)
        counts[name] = len(np.unique(entities, axis=0))
        del entities
    return counts


def update_field_dofs(mesh_stats: dict, *, field_order: int, problem_type: str) -> None:
    """Set the input ND-space estimate for a purely tetrahedral 3D EM mesh.

    MFEM's order-p Nedelec space has p DOFs per edge, p(p-1) per triangular
    face and p(p-1)(p-2)/2 per tetrahedron interior. This is before Palace's
    boundary splitting, periodic constraints and mesh refinement. It is not a
    count of coarse-solver DOFs or a guaranteed bound on the final system size.
    """
    estimate: dict = {
        "estimated_field_dofs": None,
        "field_order": field_order,
        "problem_type": problem_type,
        "method": "tetrahedral_nd_topology_v1",
        "scope": "before_palace_preprocessing",
    }
    topology = mesh_stats.get("topology")
    if topology and problem_type.lower() in {"driven", "eigenmode"}:
        if (
            isinstance(field_order, bool)
            or not isinstance(field_order, int)
            or field_order < 1
        ):
            raise ValueError("Field order must be a positive integer")
        order = field_order
        estimate["estimated_field_dofs"] = (
            order * topology["edges"]
            + order * (order - 1) * topology["triangular_faces"]
            + order * (order - 1) * (order - 2) * mesh_stats["tetrahedra"] // 2
        )
    else:
        estimate["unavailable_reason"] = (
            "Requires a purely tetrahedral 3D driven or eigenmode mesh"
        )
    mesh_stats["field_dofs"] = estimate


def write_metadata(
    mesh_stats: dict, output_dir: Path, config_path: Path | None = None
) -> dict:
    """Write versioned sizing hints alongside the inputs, outside Palace config.

    The optional materialized config supplies the effective order and problem
    type, including config overrides. Consumers must tolerate additional JSON
    keys. DataLab owns allocation policy; this document contains measurements
    and estimates, never authoritative resource requests or pricing.
    """
    metadata: dict = {"schema_version": 1, "solver": "palace"}
    if config_path is not None:
        config = json.loads(config_path.read_text())
        problem_type = config["Problem"]["Type"]
        solver = config.get("Solver", {})
        field_order = solver.get("Order", 1)
        update_field_dofs(
            mesh_stats, field_order=field_order, problem_type=problem_type
        )
        metadata["simulation"] = {
            "problem_type": problem_type,
            "field_order": field_order,
            "device": solver.get("Device", "CPU"),
            "linear_solver": solver.get("Linear", {}),
            "refinement": config.get("Model", {}).get("Refinement", {}),
        }
    metadata["mesh"] = deepcopy(mesh_stats)
    path = output_dir / METADATA_FILENAME
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata
