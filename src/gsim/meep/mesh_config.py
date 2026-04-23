"""Mesh config writer for GMSH-based Meep simulations.

Serializes the full photonic problem definition (mesh reference, materials,
source, monitors, domain, solver settings) to a ``mesh_config.json`` file
alongside the ``.msh`` mesh file.

All spatial values are written in nanometers (length_scale = 1e-9).

Port surfaces carry a unit 3D ``normal`` vector in mesh coordinates
(outward-facing). Combined with the port surface triangles in the ``.msh``
file and the ``layer`` tag, this is sufficient for a mode solver at any port
orientation — in-plane rotation, out-of-plane tilt, or both. Width/height/
center are derivable from the mesh triangles; the local (u, v) basis in the
port plane can be chosen freely by the solver since the mode eigenproblem is
basis-invariant.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gsim.common.mesh.types import MeshResult
    from gsim.meep.models.api import FDTD, Domain, Material, ModeSource

logger = logging.getLogger(__name__)

# um → nm conversion factor (hardcoded for now)
_UM_TO_NM = 1000.0


def _serialize_materials(materials: dict[str, float | Material]) -> dict[str, Any]:
    """Convert materials dict to JSON-serializable form."""
    from gsim.meep.models.api import Material

    out: dict[str, Any] = {}
    for name, val in materials.items():
        if isinstance(val, (int, float)):
            out[name] = {"refractive_index": float(val), "extinction_coeff": 0.0}
        elif isinstance(val, Material):
            out[name] = {"refractive_index": val.n, "extinction_coeff": val.k}
        else:
            out[name] = val
    return out


def _get_refractive_index(mat: Any) -> float:
    """Extract refractive index from a material spec (float, Material, or dict)."""
    if isinstance(mat, (int, float)):
        return float(mat)
    n = getattr(mat, "n", None)
    if n is not None:
        return float(n)
    if isinstance(mat, dict):
        return float(mat.get("refractive_index", 0.0))
    return 0.0


def _compute_mesh_priorities(groups: dict, materials: dict[str, Any]) -> dict[str, int]:
    """Rank materials by refractive index → integer ``mesh_priority``.

    Convention matches gplugins / femwell / meshwell: **lower ``mesh_priority``
    wins when regions overlap**. Highest-n material gets ``mesh_priority=1``,
    next gets ``2``, etc. (e.g. silicon core=1, SiO2 cladding=2).
    Only materials actually used in ``volumes`` or ``layer_volumes`` are ranked.
    """
    used: set[str] = set()
    for name in groups.get("volumes", {}):
        used.add(name)
    for info in groups.get("layer_volumes", {}).values():
        mat = info.get("material")
        if mat:
            used.add(mat)
    # Sort by refractive index descending: highest n → mesh_priority 1
    ranked = sorted(
        used, key=lambda m: _get_refractive_index(materials.get(m)), reverse=True
    )
    return {name: i + 1 for i, name in enumerate(ranked)}


def _serialize_mesh_groups(
    groups: dict, materials: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Extract serializable mesh group info (drop internal gmsh tags).

    Adds a ``mesh_priority`` field to each volume / layer_volume (lower = wins
    on overlap; default ranks by refractive index, highest n = 1).
    """
    materials = materials or {}
    mesh_priorities = _compute_mesh_priorities(groups, materials)
    out: dict[str, Any] = {}

    if "volumes" in groups:
        out["volumes"] = {
            name: {
                "phys_group": info["phys_group"],
                "material": name,
                "mesh_priority": mesh_priorities.get(name, 0),
            }
            for name, info in groups["volumes"].items()
        }

    if "layer_volumes" in groups:
        out["layer_volumes"] = {}
        for name, info in groups["layer_volumes"].items():
            material_name = info.get("material", name)
            out["layer_volumes"][name] = {
                "phys_group": info["phys_group"],
                "material": material_name,
                "mesh_priority": mesh_priorities.get(material_name, 0),
            }

    if groups.get("outer_boundary"):
        out["outer_boundary"] = {"phys_group": groups["outer_boundary"]["phys_group"]}

    if "port_surfaces" in groups:
        out["port_surfaces"] = {}
        for name, info in groups["port_surfaces"].items():
            out["port_surfaces"][name] = {
                "phys_group": info["phys_group"],
                "normal": [float(n) for n in info["normal"]],
                "layer": info["layer"],
            }

    return out


def _serialize_source(source: ModeSource) -> dict[str, Any]:
    """Serialize source settings."""
    return {
        "port": source.port,
        "wavelength": source.wavelength * _UM_TO_NM,
        "wavelength_span": source.wavelength_span * _UM_TO_NM,
        "num_freqs": source.num_freqs,
    }


def _serialize_domain(domain: Domain, solver: FDTD) -> dict[str, Any]:
    """Serialize domain settings (pml expressed as integer number of cells)."""
    return {
        "pml_cells": round(domain.pml * solver.resolution),
    }


def _serialize_solver(solver: FDTD) -> dict[str, Any]:
    """Serialize solver settings."""
    return {
        "nanometers_per_cell": 1000.0 / solver.resolution,
        "stopping": {
            "mode": solver.stopping,
            "threshold": solver.stopping_threshold,
        },
        "wall_time_max": solver.wall_time_max,
    }


def _serialize_mesh_stats(mesh_stats: dict) -> dict[str, Any]:
    """Extract key mesh statistics."""
    out: dict[str, Any] = {}
    if "nodes" in mesh_stats:
        out["nodes"] = mesh_stats["nodes"]
    if "tetrahedra" in mesh_stats:
        out["tetrahedra"] = mesh_stats["tetrahedra"]
    return out


def write_mesh_config(
    mesh_result: MeshResult,
    materials: dict[str, float | Material],
    source: ModeSource,
    monitors: list[str],
    domain: Domain,
    solver: FDTD,
    output_dir: Path,
) -> Path:
    """Write mesh_config.json alongside the .msh file.

    Args:
        mesh_result: Result from ``generate_mesh()``.
        materials: Material name → refractive index or Material object.
        source: Mode source configuration.
        monitors: List of monitor port names.
        domain: Computational domain settings.
        solver: FDTD solver settings.
        output_dir: Directory to write the config file.

    Returns:
        Path to the written ``mesh_config.json``.
    """
    config = {
        "length_scale": 1e-9,  # mesh coordinates in nm; hardcoded for now
        "mesh_filename": mesh_result.mesh_path.name,
        "materials": _serialize_materials(materials),
        "mesh_groups": _serialize_mesh_groups(mesh_result.groups, materials),
        "source": _serialize_source(source),
        "monitors": list(monitors),
        "domain": _serialize_domain(domain, solver),
        "solver": _serialize_solver(solver),
        "mesh_stats": _serialize_mesh_stats(mesh_result.mesh_stats),
    }

    config_path = Path(output_dir) / "mesh_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Mesh config written: %s", config_path)
    return config_path
