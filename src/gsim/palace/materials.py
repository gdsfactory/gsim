"""Palace-specific material resolution with frequency-dependent dispersion.

Evaluates dispersion models from the materials database at a target frequency
and produces a materials dict suitable for Palace config generation.

This implements the RFC's external frequency loop strategy: since Palace
does not natively support epsilon(f), gsim evaluates each material's
dispersion model at the target frequency and writes scalar properties
to the Palace config JSON.
"""

from __future__ import annotations

from pdk_schema import MaterialCard
from scipy.constants import c as C0  # noqa: N812

from gsim.common.materials.snapshots import (
    WavelengthOutOfRangeError,
    resolve_material_snapshot,
)
from gsim.common.stack.materials import (
    get_material_properties,
    resolve_material_at_wavelength,
)


def _resolve_card_optical_band(props: dict, wavelength_um: float) -> dict:
    """Prefer a card's optical model within its explicitly declared band."""
    card = MaterialCard.model_validate(props["material_card"])
    optical = card.optical
    if optical is None or optical.permittivity is None:
        return dict(props)
    model = optical.permittivity
    validity = model.validity
    if validity is None or "wavelength" not in (validity.over or {}):
        return dict(props)
    try:
        snapshot = resolve_material_snapshot(
            card.name, wavelength_um, {card.name: card}
        )
    except WavelengthOutOfRangeError:
        return dict(props)
    if (
        optical.conductivity is not None
        or getattr(model, "conductivity", None) is not None
    ):
        raise ValueError(f"{card.name}: optical card conductivity is unsupported")
    if optical.permeability is not None or optical.perturbations:
        raise ValueError(
            f"{card.name}: optical magnetic models/perturbations are unsupported"
        )
    n, k = snapshot.refractive_index, snapshot.extinction_coefficient
    epsilon = n * n - k * k
    if epsilon <= 0:
        raise ValueError(f"{card.name}: Palace optical permittivity must be positive")
    return {
        **props,
        "permittivity": epsilon,
        "loss_tangent": 2 * n * k / epsilon,
        "conductivity": 0.0,
        "permeability": 1.0,
    }


def resolve_palace_materials_at_frequency(
    materials: dict[str, dict],
    frequency_hz: float,
) -> dict[str, dict]:
    """Evaluate material dispersion at a given frequency for Palace config.

    For each material in the dict, looks up the corresponding entry in
    MATERIALS_DB and evaluates its dispersion model at the target frequency.
    The resolved scalar values (permittivity, loss_tangent) replace the
    original values in the dict.

    Materials without dispersion models or without matching database entries
    are left unchanged.

    Args:
        materials: Dict of material name -> properties dict (from LayerStack)
        frequency_hz: Target frequency in Hz

    Returns:
        New materials dict with evaluated scalar properties
    """
    wavelength_um = C0 / frequency_hz * 1e6
    resolved: dict[str, dict] = {}

    for name, props in materials.items():
        if props.get("material_source") == "override":
            resolved[name] = dict(props)
            continue
        # Constant RF cards are already resolved. Dual-regime cards can select
        # their own optical model inside its declared wavelength band.
        if props.get("material_source") == "pdk_material_card":
            resolved[name] = _resolve_card_optical_band(props, wavelength_um)
            continue
        db_props = get_material_properties(name)
        if db_props is None:
            resolved[name] = dict(props)
            continue

        evaluated = resolve_material_at_wavelength(name, wavelength_um)
        if evaluated is not None and evaluated.behavior == "conductive":
            resolved[name] = dict(props)
            continue

        evaluated = resolve_material_at_wavelength(name, wavelength_um)
        if evaluated is None:
            resolved[name] = dict(props)
            continue

        new_props: dict[str, object] = dict(props)

        if evaluated.permittivity is not None:
            new_props["permittivity"] = evaluated.permittivity

        if evaluated.loss_tangent is not None:
            new_props["loss_tangent"] = evaluated.loss_tangent

        if evaluated.conductivity is not None:
            new_props["conductivity"] = evaluated.conductivity
        else:
            new_props.pop("conductivity", None)

        if evaluated.permeability is not None:
            new_props["permeability"] = evaluated.permeability

        resolved[name] = new_props

    return resolved
