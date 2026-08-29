"""Validate and translate optical MaterialCards for MEEP."""

from __future__ import annotations

import math
import warnings

from pdk_schema import (
    Drude,
    Index,
    Lorentz,
    MaterialCard,
    Permittivity,
    ScalarValue,
    Sellmeier,
    SellmeierPoleSquared,
)
from scipy.constants import c as C0  # noqa: N812

from gsim.meep.models.config import MaterialData, SusceptibilityConfig


class MeepMaterialCompatibilityError(ValueError):
    """Raised when a MaterialCard cannot be represented exactly by MEEP."""


def _compatibility_error(
    material_name: str, message: str
) -> MeepMaterialCompatibilityError:
    """Return a compatibility error scoped to one material."""
    return MeepMaterialCompatibilityError(f"Material {material_name!r} {message}")


def _dimensionless_scalar(value: object, material_name: str, field: str) -> float:
    """Extract a dimensionless scalar or raise a compatibility error."""
    if not isinstance(value, ScalarValue) or value.unit != "":
        raise _compatibility_error(
            material_name,
            f"{field} must be a dimensionless scalar for MEEP; "
            "tabulated, polynomial, external-reference, and tensor values "
            "require a causal Lorentz/Drude fit.",
        )
    return float(value.value)


def _wavelength_to_um(value: float, unit: str, material_name: str) -> float:
    """Convert one supported wavelength value to micrometers."""
    scale = {"m": 1e6, "um": 1.0, "nm": 1e-3}.get(unit)
    if scale is None:
        raise _compatibility_error(
            material_name, f"uses unsupported wavelength unit {unit!r}."
        )
    return float(value) * scale


def _frequency_to_meep(value: float, unit: str, material_name: str) -> float:
    """Convert a frequency value to MEEP units with one micrometer as length."""
    scale = {
        "Hz": 1.0,
        "MHz": 1e6,
        "GHz": 1e9,
        "THz": 1e12,
    }.get(unit)
    if scale is not None:
        return float(value) * scale * 1e-6 / C0
    if unit == "rad/s":
        return float(value) * 1e-6 / (2 * math.pi * C0)
    raise _compatibility_error(
        material_name, f"uses unsupported frequency unit {unit!r}."
    )


def _model_frequency_range(model: object, material_name: str) -> list[float] | None:
    """Return a model validity interval in MEEP frequency units."""
    validity = getattr(model, "validity", None)
    ranges = {} if validity is None else (validity.over or {})
    wavelength_band = ranges.get("wavelength")
    if wavelength_band is not None:
        lower_um = _wavelength_to_um(
            wavelength_band.min, wavelength_band.unit, material_name
        )
        upper_um = _wavelength_to_um(
            wavelength_band.max, wavelength_band.unit, material_name
        )
        if lower_um <= 0 or upper_um <= lower_um:
            raise _compatibility_error(
                material_name, "has an invalid wavelength validity range."
            )
        return [1.0 / upper_um, 1.0 / lower_um]

    frequency_band = ranges.get("frequency")
    if frequency_band is not None:
        lower = _frequency_to_meep(
            frequency_band.min, frequency_band.unit, material_name
        )
        upper = _frequency_to_meep(
            frequency_band.max, frequency_band.unit, material_name
        )
        if lower <= 0 or upper <= lower:
            raise _compatibility_error(
                material_name, "has an invalid frequency validity range."
            )
        return [lower, upper]
    return None


def _validate_source_range(
    model: object,
    wavelength_range_um: tuple[float, float] | None,
    material_name: str,
) -> None:
    """Require the simulation band to lie inside model validity."""
    if wavelength_range_um is None:
        return
    lower_um, upper_um = wavelength_range_um
    if lower_um <= 0 or upper_um < lower_um:
        raise _compatibility_error(
            material_name,
            "received an invalid simulation wavelength range; expected "
            "0 < lower <= upper.",
        )
    model_range = _model_frequency_range(model, material_name)
    if model_range is None:
        return
    source_range = [1.0 / upper_um, 1.0 / lower_um]
    if source_range[0] < model_range[0] or source_range[1] > model_range[1]:
        raise _compatibility_error(
            material_name,
            f"is not valid over the simulation band {lower_um:g} to {upper_um:g} um.",
        )


def _validate_card_envelope(card: MaterialCard, material_name: str) -> object:
    """Validate fields shared by every supported optical model."""
    if card.optical is None or card.optical.permittivity is None:
        raise _compatibility_error(material_name, "has no optical permittivity model.")
    if card.optical.conductivity is not None:
        raise _compatibility_error(
            material_name, "has regime conductivity, which is not yet supported."
        )
    if card.optical.permeability is not None:
        raise _compatibility_error(
            material_name, "has optical permeability, which is not yet supported."
        )
    if card.optical.perturbations:
        raise _compatibility_error(
            material_name,
            "has optical perturbations, but gsim.meep has no operating-condition "
            "inputs for applying them.",
        )
    model = card.optical.permittivity
    if getattr(model, "conductivity", None) is not None:
        raise _compatibility_error(
            material_name, "has model conductivity, which is not yet supported."
        )
    return model


def validate_meep_material_card(
    card: MaterialCard,
    wavelength_range_um: tuple[float, float] | None = None,
    *,
    material_name: str | None = None,
) -> None:
    """Validate that an optical MaterialCard has an exact MEEP representation.

    Constant, Sellmeier, Lorentz, and Drude models are accepted. Model-free
    wavelength-dependent data and Cauchy, Debye, or pole-residue models are
    rejected because MEEP requires a causal Lorentz/Drude fit for those forms.

    Args:
        card: Material card to validate.
        wavelength_range_um: Optional inclusive simulation band in micrometers.
        material_name: Registry name used in error messages.

    Raises:
        MeepMaterialCompatibilityError: If the card cannot be encoded exactly.
    """
    name = material_name or card.name
    model = _validate_card_envelope(card, name)
    _validate_source_range(model, wavelength_range_um, name)

    if isinstance(model, Index):
        refractive_index = _dimensionless_scalar(model.n, name, "n")
        if refractive_index <= 0:
            raise _compatibility_error(name, "has a non-positive refractive index.")
        if model.k is not None and _dimensionless_scalar(model.k, name, "k") != 0:
            raise _compatibility_error(
                name,
                "has a nonzero constant extinction coefficient; MEEP requires "
                "a causal Lorentz/Drude loss model.",
            )
        return

    if isinstance(model, Permittivity):
        epsilon = _dimensionless_scalar(model.eps_real, name, "eps_real")
        if epsilon <= 0:
            raise _compatibility_error(name, "has non-positive permittivity.")
        if (
            model.eps_imag is not None
            and _dimensionless_scalar(model.eps_imag, name, "eps_imag") != 0
        ):
            raise _compatibility_error(
                name,
                "has nonzero constant imaginary permittivity; MEEP requires "
                "a causal Lorentz/Drude loss model.",
            )
        return

    if isinstance(model, (Sellmeier, SellmeierPoleSquared)):
        if 1.0 + model.offset <= 0:
            raise _compatibility_error(name, "has non-positive epsilon infinity.")
        pole_positions = (
            [abs(term.c_um) for term in model.terms]
            if isinstance(model, Sellmeier)
            else [term.c_um2 for term in model.terms]
        )
        if any(position <= 0 for position in pole_positions):
            raise _compatibility_error(
                name, "has a non-positive Sellmeier pole position."
            )
        return

    if isinstance(model, (Lorentz, Drude)):
        if model.eps_inf <= 0:
            raise _compatibility_error(name, "has non-positive epsilon infinity.")
        return

    raise _compatibility_error(
        name,
        f"uses unsupported optical model {type(model).__name__}; provide a "
        "constant, Sellmeier, Lorentz, or Drude card.",
    )


def _angular_frequency_to_meep(angular_frequency: float) -> float:
    """Convert radians per second to MEEP frequency units (1/um)."""
    return angular_frequency * 1e-6 / (2 * math.pi * C0)


def material_data_from_card(
    card: MaterialCard,
    wavelength_range_um: tuple[float, float] | None = None,
    *,
    material_name: str | None = None,
) -> MaterialData:
    """Translate a compatible MaterialCard into serialized MEEP data."""
    name = material_name or card.name
    validate_meep_material_card(card, wavelength_range_um, material_name=name)
    optical = card.optical
    if optical is None or optical.permittivity is None:  # pragma: no cover
        raise AssertionError("MaterialCard validation did not reject an empty model.")
    model = optical.permittivity

    if isinstance(model, Index):
        refractive_index = _dimensionless_scalar(model.n, name, "n")
        return MaterialData(epsilon_diag=[refractive_index**2] * 3)
    if isinstance(model, Permittivity):
        epsilon = _dimensionless_scalar(model.eps_real, name, "eps_real")
        return MaterialData(epsilon_diag=[epsilon] * 3)

    susceptibilities: list[SusceptibilityConfig] = []
    if isinstance(model, Sellmeier):
        epsilon_inf = 1.0 + model.offset
        susceptibilities = [
            SusceptibilityConfig(
                kind="lorentzian",
                frequency=1.0 / abs(term.c_um),
                gamma=0.0,
                sigma=term.b,
            )
            for term in model.terms
            if term.b != 0
        ]
    elif isinstance(model, SellmeierPoleSquared):
        epsilon_inf = 1.0 + model.offset
        susceptibilities = [
            SusceptibilityConfig(
                kind="lorentzian",
                frequency=1.0 / math.sqrt(term.c_um2),
                gamma=0.0,
                sigma=term.b,
            )
            for term in model.terms
            if term.b != 0
        ]
    elif isinstance(model, Lorentz):
        epsilon_inf = model.eps_inf
        susceptibilities = [
            SusceptibilityConfig(
                kind="lorentzian",
                frequency=_angular_frequency_to_meep(term.omega_0),
                gamma=_angular_frequency_to_meep(term.gamma),
                sigma=term.delta_eps,
            )
            for term in model.terms
            if term.delta_eps != 0
        ]
    elif isinstance(model, Drude):
        epsilon_inf = model.eps_inf
        susceptibilities = [
            SusceptibilityConfig(
                kind="drude",
                frequency=_angular_frequency_to_meep(term.omega_p),
                gamma=_angular_frequency_to_meep(term.gamma),
                sigma=1.0,
            )
            for term in model.terms
        ]
    else:  # pragma: no cover - guarded by validation
        raise TypeError(f"Unhandled compatible model {type(model).__name__}")

    return MaterialData(
        epsilon_diag=[epsilon_inf] * 3,
        epsilon_susceptibilities=susceptibilities or None,
        valid_freq_range=_model_frequency_range(model, name),
    )


def warn_if_material_may_be_unstable(
    material_name: str,
    material_data: MaterialData,
    resolution: int,
    *,
    courant: float = 0.5,
) -> None:
    """Warn when a Lorentz resonance exceeds MEEP's rough stability limit."""
    maximum_frequency = resolution / (math.pi * courant)
    unstable = [
        term.frequency
        for term in material_data.epsilon_susceptibilities or []
        if term.kind == "lorentzian" and term.frequency >= maximum_frequency
    ]
    if not unstable:
        return
    minimum_resolution = math.floor(math.pi * courant * max(unstable)) + 1
    warnings.warn(
        f"Material {material_name!r} has a Lorentz pole at "
        f"{max(unstable):.4g} 1/um, above MEEP's rough stability limit for "
        f"resolution={resolution} and Courant={courant}. Use resolution >= "
        f"{minimum_resolution}.",
        RuntimeWarning,
        stacklevel=2,
    )


__all__ = [
    "MeepMaterialCompatibilityError",
    "material_data_from_card",
    "validate_meep_material_card",
    "warn_if_material_may_be_unstable",
]
