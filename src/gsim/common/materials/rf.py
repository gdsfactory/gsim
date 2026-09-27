"""Resolve constant, nonmagnetic RF material cards for stack consumers."""

from __future__ import annotations

import math
from collections.abc import Mapping

from pdk_schema import MaterialCard, Permittivity, ScalarValue


def _scalar(value: object, unit: str, name: str) -> float:
    """Read a finite, nonnegative scalar in the expected unit."""
    if not isinstance(value, ScalarValue) or value.unit != unit:
        raise ValueError(f"{name}: RF cards require constant values in {unit!r}")
    number = float(value.value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name}: RF material values must be finite and nonnegative")
    return number


def _constant(value: object, unit: str, name: str) -> float | list[float]:
    """Read a scalar or diagonal tensor without silently dropping units/loss."""
    if isinstance(value, list):
        if len(value) != 3:
            raise ValueError(f"{name}: RF cards require scalar or diagonal-3 values")
        return [_scalar(v, unit, name) for v in value]
    return _scalar(value, unit, name)


def _loss_tangent(
    real: float | list[float], imag: float | list[float], name: str
) -> float | list[float]:
    """Convert positive imaginary relative permittivity into dielectric loss."""
    if isinstance(real, list) or isinstance(imag, list):
        real_values = real if isinstance(real, list) else [real] * 3
        imag_values = imag if isinstance(imag, list) else [imag] * 3
        if any(r <= 0 for r in real_values):
            raise ValueError(f"{name}: RF permittivity must be positive")
        return [i / r for r, i in zip(real_values, imag_values, strict=True)]
    if real <= 0:
        raise ValueError(f"{name}: RF permittivity must be positive")
    return imag / real


def resolve_rf_material_card(card: MaterialCard) -> dict:
    """Resolve a constant RF card and retain its full provenance in the stack.

    Only scalar/diagonal constant permittivity and conductivity are supported.
    Reject dispersion, validity fences, variation, and magnetic models rather
    than substituting a generic material. Optical-only cards are handled by
    the optical material resolver, not this adapter.
    """
    regime = card.rf
    if regime is None:
        raise ValueError(f"{card.name}: material card has no RF regime")
    if regime.permeability is not None or regime.perturbations:
        raise ValueError(
            f"{card.name}: magnetic models and RF perturbations are unsupported"
        )
    model = regime.permittivity
    props: dict = {
        "permittivity": 1.0,
        "permeability": 1.0,
        "loss_tangent": 0.0,
        "material_source": "pdk_material_card",
        "material_card": card.model_dump(mode="json"),
    }
    if model is None:
        if regime.conductivity is None:
            raise ValueError(
                f"{card.name}: RF card has neither permittivity nor conductivity"
            )
        props["type"] = "conductor"
        props["conductivity"] = _constant(regime.conductivity, "S/m", card.name)
        return props
    if not isinstance(model, Permittivity):
        raise TypeError(
            f"{card.name}: unsupported RF model {model.kind!r}; "
            "expected constant permittivity"
        )
    if model.validity is not None or model.variation is not None:
        raise ValueError(
            f"{card.name}: RF validity constraints and variation are unsupported"
        )
    if regime.conductivity is not None:
        raise ValueError(
            f"{card.name}: conductivity must belong to the RF permittivity model"
        )
    props["type"] = "dielectric"
    props["permittivity"] = _constant(model.eps_real, "", card.name)
    imag = 0.0 if model.eps_imag is None else _constant(model.eps_imag, "", card.name)
    props["loss_tangent"] = _loss_tangent(props["permittivity"], imag, card.name)
    props["conductivity"] = (
        0.0
        if model.conductivity is None
        else _constant(model.conductivity, "S/m", card.name)
    )
    sigma = props["conductivity"]
    loss = props["loss_tangent"]
    has_sigma = any(v > 0 for v in sigma) if isinstance(sigma, list) else sigma > 0
    has_loss = any(v > 0 for v in loss) if isinstance(loss, list) else loss > 0
    if has_sigma and has_loss:
        raise ValueError(
            f"{card.name}: simultaneous RF conductivity and dielectric loss "
            "are not supported by the stack adapter"
        )
    return props


def apply_rf_material_cards(
    materials: dict[str, dict], cards: Mapping[str, MaterialCard]
) -> dict[str, dict]:
    """Use exact PDK material tokens; preserve legacy entries without RF cards."""
    return {
        name: resolve_rf_material_card(cards[name])
        if name in cards and cards[name].rf is not None
        else dict(props)
        for name, props in materials.items()
    }
