"""Resolve MEEP materials from canonical MaterialCards and explicit overrides."""

from __future__ import annotations

import math
from collections.abc import Mapping

from pdk_schema import MaterialCard
from scipy.constants import c as C0  # noqa: N812

from gsim.common.materials import (
    MaterialNotFoundError,
    find_material_card,
    resolve_material_snapshot,
)
from gsim.common.stack.materials import (
    MaterialProperties,
    ResolvedMaterial,
    _as_list,
    _is_tensor,
    _normalize_material_keys,
)
from gsim.meep.material_cards import (
    material_data_from_card,
    warn_if_material_may_be_unstable,
)
from gsim.meep.models.config import MaterialData


def loss_tangent_to_conductivity(
    loss_tangent: float,
    permittivity: float,
    freq_hz: float,
) -> float:
    """Convert loss tangent to MEEP's dimensionless D-conductivity.

    ``permittivity`` is retained in the signature for API compatibility; it
    cancels when physical conductivity is normalized to MEEP units.
    """
    del permittivity
    meep_frequency = freq_hz * 1e-6 / C0
    return 2.0 * math.pi * meep_frequency * loss_tangent


def _is_identity_axes(material_axes: list[list[float]] | None) -> bool:
    """Check whether material axes describe the identity rotation."""
    if material_axes is None:
        return True
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return all(
        abs(value - reference) <= 1e-10
        for row, expected_row in zip(material_axes, identity, strict=False)
        for value, reference in zip(row, expected_row, strict=False)
    )


def _rotate_diagonal_tensor(
    diagonal: list[float], material_axes: list[list[float]]
) -> list[float]:
    """Rotate a diagonal tensor and return MEEP's three off-diagonal terms."""
    rotated = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(3):
            rotated[row][column] = sum(
                material_axes[row][axis] * diagonal[axis] * material_axes[column][axis]
                for axis in range(3)
            )
    return [rotated[0][1], rotated[0][2], rotated[1][2]]


def _resolved_to_material_data(
    resolved: ResolvedMaterial,
    wavelength_um: float,
) -> MaterialData:
    """Convert an explicit scalar/tensor override to MEEP material data."""
    if resolved.permittivity is None:
        raise ValueError("ResolvedMaterial has no permittivity")
    epsilon_diagonal = _as_list(resolved.permittivity, 3)
    data = MaterialData(epsilon_diag=epsilon_diagonal)

    if resolved.permeability is not None:
        data.mu_diag = _as_list(resolved.permeability, 3)

    frequency_hz = C0 / (wavelength_um * 1e-6)
    loss_tangent = resolved.loss_tangent_scalar
    if loss_tangent is not None and loss_tangent > 0:
        if _is_tensor(resolved.loss_tangent):
            data.D_conductivity_diag = [
                loss_tangent_to_conductivity(value, 1.0, frequency_hz)
                for value in _as_list(resolved.loss_tangent, 3) or []
            ]
        else:
            data.D_conductivity = loss_tangent_to_conductivity(
                loss_tangent,
                resolved.permittivity_scalar or 1.0,
                frequency_hz,
            )

    if (
        resolved.material_axes is not None
        and not _is_identity_axes(resolved.material_axes)
        and epsilon_diagonal is not None
    ):
        data.epsilon_offdiag = _rotate_diagonal_tensor(
            epsilon_diagonal, resolved.material_axes
        )
    return data


def _override_data(
    material_name: str,
    normalized_overrides: Mapping[str, MaterialProperties],
    wavelength_um: float,
) -> MaterialData | None:
    """Resolve a case-normalized explicit override, if present."""
    properties = normalized_overrides.get(material_name.lower())
    if properties is None:
        return None
    resolved = properties.evaluate_at_wavelength(wavelength_um)
    if resolved.behavior == "conductive":
        raise ValueError(
            f"Material override {material_name!r} is conductive and cannot be "
            "used as a passive optical medium."
        )
    return _resolved_to_material_data(resolved, wavelength_um)


def _center_wavelength_data(
    material_name: str,
    wavelength_um: float,
    project_material_cards: Mapping[str, MaterialCard] | None,
) -> MaterialData:
    """Evaluate a MaterialCard at one wavelength for mode solving and plots."""
    snapshot = resolve_material_snapshot(
        material_name,
        wavelength_um,
        project_material_cards,
    )
    epsilon_real = snapshot.refractive_index**2 - snapshot.extinction_coefficient**2
    return MaterialData(epsilon_diag=[epsilon_real] * 3)


def resolve_materials(
    used_material_names: set[str],
    overrides: dict[str, MaterialProperties] | None = None,
    wavelength_um: float = 1.55,
    project_material_cards: Mapping[str, MaterialCard] | None = None,
) -> dict[str, MaterialData]:
    """Resolve center-wavelength material data from project-first cards.

    This nondispersive snapshot path is used by mode solving and index plots.
    Production FDTD configuration uses :func:`resolve_fdtd_materials` so that
    authored Sellmeier, Lorentz, and Drude dispersion is preserved.
    """
    normalized_overrides = _normalize_material_keys(
        overrides or {}, used_names=used_material_names, label="overrides"
    )
    materials: dict[str, MaterialData] = {}
    for name in sorted(used_material_names):
        if name.lower() == "air":
            materials[name] = MaterialData(epsilon_diag=[1.0, 1.0, 1.0])
            continue
        override = _override_data(name, normalized_overrides, wavelength_um)
        materials[name] = override or _center_wavelength_data(
            name, wavelength_um, project_material_cards
        )
    return materials


def resolve_fdtd_materials(
    used_material_names: set[str],
    overrides: dict[str, MaterialProperties] | None = None,
    *,
    wavelength_um: float,
    wavelength_span_um: float,
    resolution: int,
    project_material_cards: Mapping[str, MaterialCard] | None = None,
) -> dict[str, MaterialData]:
    """Resolve full authored dispersion for a MEEP time-domain simulation."""
    lower_wavelength = wavelength_um - wavelength_span_um / 2
    upper_wavelength = wavelength_um + wavelength_span_um / 2
    if lower_wavelength <= 0:
        raise ValueError("The source wavelength span reaches a non-positive value.")
    wavelength_range = (lower_wavelength, upper_wavelength)
    normalized_overrides = _normalize_material_keys(
        overrides or {}, used_names=used_material_names, label="overrides"
    )
    materials: dict[str, MaterialData] = {}
    for name in sorted(used_material_names):
        if name.lower() == "air":
            materials[name] = MaterialData(epsilon_diag=[1.0, 1.0, 1.0])
            continue
        override = _override_data(name, normalized_overrides, wavelength_um)
        if override is not None:
            materials[name] = override
            continue
        try:
            card, _source = find_material_card(name, project_material_cards)
        except KeyError as error:
            raise MaterialNotFoundError(error.args[0]) from error
        material_data = material_data_from_card(
            card,
            wavelength_range,
            material_name=name,
        )
        warn_if_material_may_be_unstable(name, material_data, resolution)
        materials[name] = material_data
    return materials


__all__ = [
    "loss_tangent_to_conductivity",
    "resolve_fdtd_materials",
    "resolve_materials",
]
