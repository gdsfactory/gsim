"""Material resolution for MEEP simulation.

Resolves material names from the common materials database to optical
properties needed for photonic FDTD simulation. No MEEP imports.
"""

from __future__ import annotations

import warnings

from gsim.common.stack.materials import MaterialProperties, get_material_properties
from gsim.meep.models.config import MaterialData


def resolve_materials(
    used_material_names: set[str],
    overrides: dict[str, MaterialProperties] | None = None,
) -> dict[str, MaterialData]:
    """Resolve material names to optical properties for MEEP.

    Only resolves materials that actually have geometry (i.e., polygons
    extracted from the component). Materials present in the GDS but
    missing optical data emit a warning and are excluded from the config.

    Args:
        used_material_names: Material names that appear in extracted geometry
        overrides: User-supplied material property overrides (from set_material())

    Returns:
        Dict mapping material name to MaterialData
    """
    overrides = overrides or {}
    materials: dict[str, MaterialData] = {}

    for name in sorted(used_material_names):
        # Check user overrides first
        if name in overrides:
            props = overrides[name]
            if props.refractive_index is None:
                warnings.warn(
                    f"Material override '{name}' has no refractive_index — "
                    f"skipping. Use set_material('{name}', refractive_index=...)",
                    stacklevel=2,
                )
                continue
            materials[name] = MaterialData(
                refractive_index=props.refractive_index,
                extinction_coeff=props.extinction_coeff or 0.0,
            )
            continue

        # Look up in common DB
        db_props = get_material_properties(name)
        if db_props is not None and db_props.refractive_index is not None:
            materials[name] = MaterialData(
                refractive_index=db_props.refractive_index,
                extinction_coeff=db_props.extinction_coeff or 0.0,
            )
            continue

        warnings.warn(
            f"Material '{name}' has no optical properties (refractive_index) "
            f"— layer will be omitted from simulation. "
            f"Use sim.set_material('{name}', refractive_index=...) to include it.",
            stacklevel=2,
        )

    return materials
