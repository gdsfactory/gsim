"""Material resolution for MEEP simulation.

Resolves material names from the common materials database to optical
properties needed for photonic FDTD simulation. Supports three-tier
resolution: user override > PDK overlay > built-in database.

Anisotropic tensor mapping (RFC: unify the tensor model):
    - ``permittivity_diagonal`` -> ``epsilon_diag``
    - ``permeability`` -> ``mu_diag``
    - ``conductivity`` / ``conductivity_diagonal`` -> ``D_conductivity`` / ``D_conductivity_diag``
    - ``loss_tangent`` -> ``D_conductivity`` (converted at simulation frequency)
    - ``material_axes`` -> ``epsilon_offdiag`` (rotation of the tensor)
"""

from __future__ import annotations

import math
import warnings

from gsim.common.stack.materials import (
    MaterialProperties,
    ResolvedMaterial,
    get_material_properties,
    resolve_material_at_wavelength,
    should_enable_dispersion,
)
from gsim.meep.models.config import MaterialData

_EPS0_SI = 8.854187817e-12


def loss_tangent_to_conductivity(
    loss_tangent: float,
    permittivity: float,
    freq_hz: float,
) -> float:
    """Convert loss tangent to MEEP D_conductivity at a given frequency.

    sigma = 2*pi*f * eps0 * eps_r * tan(delta)

    This conversion is exact at the center frequency and approximate
    elsewhere, which is appropriate for narrowband simulations.

    Args:
        loss_tangent: Loss tangent tan(delta)
        permittivity: Relative permittivity eps_r
        freq_hz: Frequency in Hz

    Returns:
        Conductivity in S/m
    """
    return 2.0 * math.pi * freq_hz * _EPS0_SI * permittivity * loss_tangent


def _is_identity_axes(material_axes: list[list[float]] | None) -> bool:
    """Check if material_axes represents the identity rotation."""
    if material_axes is None:
        return True
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for row, ref in zip(material_axes, identity):
        for v, r in zip(row, ref):
            if abs(v - r) > 1e-10:
                return False
    return True


def _rotate_diagonal_tensor(
    diag: list[float],
    material_axes: list[list[float]],
) -> list[float]:
    """Rotate a diagonal tensor by material_axes, returning off-diagonal components.

    Given a diagonal tensor T = diag(d1, d2, d3) and rotation matrix R,
    the rotated tensor is T' = R T R^T. Returns the [u, v, w] off-diagonal
    components in MEEP convention: epsilon_offdiag = [T'01, T'02, T'12].

    Args:
        diag: Diagonal components [d1, d2, d3]
        material_axes: 3x3 rotation matrix (rows = new axis vectors)

    Returns:
        Off-diagonal components [T'01, T'02, T'12]
    """
    rotated = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                rotated[i][j] += material_axes[i][k] * diag[k] * material_axes[j][k]

    return [rotated[0][1], rotated[0][2], rotated[1][2]]


def _resolved_to_material_data(
    resolved: ResolvedMaterial,
    wavelength_um: float,
) -> MaterialData:
    """Convert a ResolvedMaterial to a MaterialData with full tensor support.

    Populates scalar, diagonal, and off-diagonal fields based on what the
    resolved material provides. Converts loss tangent to D_conductivity at
    the simulation frequency.

    Args:
        resolved: Evaluated material properties from the dispersion resolver
        wavelength_um: Simulation wavelength in um (for loss tangent conversion)

    Returns:
        MaterialData suitable for the MEEP config JSON
    """
    if resolved.refractive_index is None:
        raise ValueError("ResolvedMaterial has no refractive_index")

    data = MaterialData(
        refractive_index=resolved.refractive_index,
        extinction_coeff=resolved.extinction_coeff,
    )

    if resolved.permittivity_diagonal is not None:
        data.epsilon_diag = [v * 1.0 for v in resolved.permittivity_diagonal]
    elif resolved.permittivity is not None:
        data.epsilon_diag = [resolved.permittivity] * 3

    if resolved.permeability is not None:
        data.mu_diag = [v * 1.0 for v in resolved.permeability]

    freq_hz = 3e8 / (wavelength_um * 1e-6)

    if resolved.conductivity is not None and resolved.conductivity > 0:
        data.D_conductivity = resolved.conductivity
    elif resolved.loss_tangent is not None and resolved.loss_tangent > 0:
        eps_r = resolved.permittivity or (resolved.refractive_index**2)
        data.D_conductivity = loss_tangent_to_conductivity(
            resolved.loss_tangent, eps_r, freq_hz
        )

    if resolved.conductivity_diagonal is not None:
        data.D_conductivity_diag = [v * 1.0 for v in resolved.conductivity_diagonal]
    elif resolved.loss_tangent_diagonal is not None:
        eps_diag = (
            resolved.permittivity_diagonal
            or [resolved.permittivity or resolved.refractive_index**2] * 3
        )
        data.D_conductivity_diag = [
            loss_tangent_to_conductivity(lt, eps, freq_hz)
            for lt, eps in zip(resolved.loss_tangent_diagonal, eps_diag)
        ]

    if (
        resolved.material_axes is not None
        and not _is_identity_axes(resolved.material_axes)
        and data.epsilon_diag is not None
    ):
        data.epsilon_offdiag = _rotate_diagonal_tensor(
            data.epsilon_diag, resolved.material_axes
        )

    return data


def resolve_materials(
    used_material_names: set[str],
    overrides: dict[str, MaterialProperties] | None = None,
    wavelength_um: float | None = None,
    overlay: dict[str, MaterialProperties] | None = None,
) -> dict[str, MaterialData]:
    """Resolve material names to optical properties for MEEP.

    Only resolves materials that actually have geometry (i.e., polygons
    extracted from the component). Materials present in the GDS but
    missing optical data emit a warning and are excluded from the config.

    When ``wavelength_um`` is provided, uses the frequency-aware dispersion
    resolver to evaluate Sellmeier/Lorentzian models at the target wavelength
    and populates anisotropic tensor fields (epsilon_diag, mu_diag,
    D_conductivity, epsilon_offdiag) from the resolved material.

    Args:
        used_material_names: Material names that appear in extracted geometry
        overrides: User-supplied material property overrides (from set_material())
        wavelength_um: Target wavelength in um. None = use legacy scalar lookup.
        overlay: PDK overlay dict (foundry-specific values).

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
                    materials[name] = _resolved_to_material_data(
                        resolved, wavelength_um
                    )
                    continue
            if props.refractive_index is None:
                warnings.warn(
                    f"Material override '{name}' has no refractive_index -- "
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
            resolved = resolve_material_at_wavelength(
                name, wavelength_um, overlay=overlay
            )
            if resolved is not None and resolved.refractive_index is not None:
                materials[name] = _resolved_to_material_data(resolved, wavelength_um)
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
                f"-- layer will be omitted from simulation. "
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
    overlay: dict[str, MaterialProperties] | None = None,
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
        overlay: PDK overlay dict (foundry-specific values)

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
            resolved = resolve_material_at_wavelength(
                name, wavelength_um, overlay=overlay
            )

        if resolved is None or resolved.refractive_index is None:
            warnings.warn(
                f"Material '{name}' has no optical properties -- "
                f"layer will be omitted from simulation.",
                stacklevel=2,
            )
            continue

        materials[name] = _resolved_to_material_data(resolved, wavelength_um)

    return materials


def check_dispersion_needs(
    used_material_names: set[str],
    overrides: dict[str, MaterialProperties] | None = None,
    wavelength_um: float = 1.55,
    bandwidth_um: float = 0.1,
    threshold: float = 0.005,
    overlay: dict[str, MaterialProperties] | None = None,
) -> dict[str, bool]:
    """Check which materials need dispersion for the given bandwidth.

    Args:
        used_material_names: Material names that appear in extracted geometry
        overrides: User-supplied material property overrides
        wavelength_um: Center wavelength in um
        bandwidth_um: Source bandwidth in um
        threshold: Fractional index variation threshold (default 0.5%)
        overlay: PDK overlay dict (foundry-specific values)

    Returns:
        Dict mapping material name to whether dispersion is needed
    """
    overrides = overrides or {}
    result: dict[str, bool] = {}

    for name in sorted(used_material_names):
        result[name] = should_enable_dispersion(
            name, wavelength_um, bandwidth_um, threshold, overrides, overlay
        )

    return result
