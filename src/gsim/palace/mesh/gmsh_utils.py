"""Gmsh utility functions — backward-compatible re-export from common.

All generic GMSH helpers now live in ``gsim.common.mesh.gmsh_utils``.
This shim keeps existing ``from gsim.palace.mesh import gmsh_utils`` working.

Palace-specific helpers (``Entity``, ``run_boolean_pipeline``) are defined
here because they encode Palace mesh-order conventions and physical-group
labelling that don't belong in the solver-agnostic common module.
"""

from __future__ import annotations

import gmsh

# Re-export all generic helpers from common
from gsim.common.mesh.gmsh_utils import *  # noqa: F403
from gsim.common.mesh.gmsh_utils import is_vertical_surface


# ---------------------------------------------------------------------------
# Palace-specific: Entity + boolean pipeline
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
    3. Priority cuts: each entity is cut by all previously processed entities
       in the same dimension (removeObject=True, removeTool=False).
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
