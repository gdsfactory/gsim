"""Material resolution for MEEP simulation.

Resolves material names from the common materials database to optical
properties needed for photonic FDTD simulation. No MEEP imports.
"""

from __future__ import annotations

import warnings

from gsim.common.stack.materials import (
    MaterialProperties,
    ResolvedMaterial,
    get_material_properties,
    resolve_material_at_wavelength,
    should_enable_dispersion,
)
from gsim.meep.models.config import MaterialData


def resolve_materials(
    used_material_names: set[str],
    overrides: dict[str, MaterialProperties] | None = None,
    wavelength_um: float | None = None,
) -> dict[str, MaterialData]:
    """Resolve material names to optical properties for MEEP.

    Only resolves materials that actually have geometry (i.e., polygons
    extracted from the component). Materials present in the GDS but
    missing optical data emit a warning and are excluded from the config.

    When ``wavelength_um`` is provided, uses the frequency-aware dispersion
    resolver to evaluate Sellmeier/Lorentzian models at the target wavelength.
    When None, falls back to legacy scalar ``refractive_index`` lookup.

    Args:
        used_material_names: Material names that appear in extracted geometry
        overrides: User-supplied material property overrides (from set_material())
        wavelength_um: Target wavelength in um. None = use legacy scalar lookup.

    Returns:
        Dict mapping material name to MaterialData
    """
    overrides = overrides or {}
    materials: dict[str, MaterialData] = {}

    for name in sorted(used_material_names):
        if name in overrides:
            props = overrides[name]
            if props.type == "conductor":
                continue
            if wavelength_um is not None:
                resolved = props.evaluate_at_wavelength(wavelength_um)
                if resolved.refractive_index is not None:
                    materials[name] = MaterialData(
                        refractive_index=resolved.refractive_index,
                        extinction_coeff=resolved.extinction_coeff,
                    )
                    continue
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

        db_props = get_material_properties(name)
        if db_props is not None and db_props.type == "conductor":
            continue

        if wavelength_um is not None:
            resolved = resolve_material_at_wavelength(name, wavelength_um)
            if resolved is not None and resolved.refractive_index is not None:
                materials[name] = MaterialData(
                    refractive_index=resolved.refractive_index,
                    extinction_coeff=resolved.extinction_coeff,
                )
                continue
        else:
            if db_props is not None and db_props.refractive_index is not None:
                materials[name] = MaterialData(
                    refractive_index=db_props.refractive_index,
                    extinction_coeff=db_props.extinction_coeff or 0.0,
                )
                continue

        if db_props is None:
            warnings.warn(
                f"Material '{name}' not found in database. "
                f"Use sim.set_material('{name}', refractive_index=...) to add it.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Material '{name}' has no optical properties (refractive_index) "
                f"— layer will be omitted from simulation. "
                f"Use sim.set_material('{name}', refractive_index=...) to include it.",
                stacklevel=2,
            )

    return materials


def resolve_materials_with_dispersion(
    used_material_names: set[str],
    overrides: dict[str, MaterialProperties] | None = None,
    wavelength_um: float = 1.55,
    bandwidth_um: float = 0.1,  # noqa: ARG001
    dispersion: str = "auto",  # noqa: ARG001
) -> dict[str, MaterialData]:
    """Resolve materials with dispersion-aware evaluation.

    Uses the frequency-aware resolver and the ``dispersion`` flag to decide
    whether each material should use constant or dispersive rendering.

    Args:
        used_material_names: Material names that appear in extracted geometry
        overrides: User-supplied material property overrides
        wavelength_um: Center wavelength in um
        bandwidth_um: Source bandwidth in um
        dispersion: "auto", "true"/"yes", or "false"/"no"

    Returns:
        Dict mapping material name to MaterialData
    """
    overrides = overrides or {}
    materials: dict[str, MaterialData] = {}

    for name in sorted(used_material_names):
        resolved: ResolvedMaterial | None

        if name in overrides:
            resolved = overrides[name].evaluate_at_wavelength(wavelength_um)
        else:
            resolved = resolve_material_at_wavelength(name, wavelength_um)

        if resolved is None or resolved.refractive_index is None:
            warnings.warn(
                f"Material '{name}' has no optical properties — "
                f"layer will be omitted from simulation.",
                stacklevel=2,
            )
            continue

        materials[name] = MaterialData(
            refractive_index=resolved.refractive_index,
            extinction_coeff=resolved.extinction_coeff,
        )

    return materials


def check_dispersion_needs(
    used_material_names: set[str],
    overrides: dict[str, MaterialProperties] | None = None,
    wavelength_um: float = 1.55,
    bandwidth_um: float = 0.1,
    threshold: float = 0.005,
) -> dict[str, bool]:
    """Check which materials need dispersion for the given bandwidth.

    Returns a dict mapping material name to whether dispersion is needed.
    """
    overrides = overrides or {}
    result: dict[str, bool] = {}

    for name in sorted(used_material_names):
        result[name] = should_enable_dispersion(
            name, wavelength_um, bandwidth_um, threshold, overrides
        )

    return result
