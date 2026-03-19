"""Mesh config writer for GMSH-based Meep simulations.

Serializes the full photonic problem definition (mesh reference, materials,
source, monitors, domain, solver settings) to a ``mesh_config.json`` file
alongside the ``.msh`` mesh file.
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


def _serialize_mesh_groups(groups: dict) -> dict[str, Any]:
    """Extract serializable mesh group info (drop internal gmsh tags)."""
    out: dict[str, Any] = {}

    if "volumes" in groups:
        out["volumes"] = {
            name: {
                "phys_group": info["phys_group"],
                "material": name,
            }
            for name, info in groups["volumes"].items()
        }

    if "layer_volumes" in groups:
        out["layer_volumes"] = {
            name: {
                "phys_group": info["phys_group"],
                "material": info.get("material", name),
            }
            for name, info in groups["layer_volumes"].items()
        }

    if groups.get("outer_boundary"):
        out["outer_boundary"] = {"phys_group": groups["outer_boundary"]["phys_group"]}

    if "port_surfaces" in groups:
        out["port_surfaces"] = {}
        for name, info in groups["port_surfaces"].items():
            out["port_surfaces"][name] = {
                "phys_group": info["phys_group"],
                "center": info["center"],
                "width": info["width"],
                "orientation": info["orientation"],
                "layer": info["layer"],
                "z_range": info["z_range"],
            }

    return out


def _serialize_source(source: ModeSource) -> dict[str, Any]:
    """Serialize source settings."""
    return {
        "port": source.port,
        "wavelength": source.wavelength,
        "wavelength_span": source.wavelength_span,
        "num_freqs": source.num_freqs,
    }


def _serialize_domain(domain: Domain) -> dict[str, Any]:
    """Serialize domain settings."""
    return {
        "pml": domain.pml,
        "margin_xy": domain.margin,
        "margin_z_above": domain.margin_z_above,
        "margin_z_below": domain.margin_z_below,
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
        "mesh_groups": _serialize_mesh_groups(mesh_result.groups),
        "source": _serialize_source(source),
        "monitors": list(monitors),
        "domain": _serialize_domain(domain),
        "solver": _serialize_solver(solver),
        "mesh_stats": _serialize_mesh_stats(mesh_result.mesh_stats),
    }

    config_path = Path(output_dir) / "mesh_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Mesh config written: %s", config_path)
    return config_path
