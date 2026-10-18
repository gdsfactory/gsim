"""Read constant scalar RF properties from a PDK's material cards."""

from pdk_schema import MaterialCard, Permittivity, ScalarValue

from gsim.common.stack.materials import MaterialProperties


def _scalar(value: object, unit: str, material_name: str) -> float:
    """Require a constant value in the units used by the stack."""
    if not isinstance(value, ScalarValue) or value.unit != unit:
        raise ValueError(
            f"RF material {material_name!r} requires a constant scalar in {unit!r}."
        )
    return value.value


def constant_rf_properties(card: MaterialCard) -> dict[str, object] | None:
    """Convert nominal RF constants; reject unsupported models explicitly."""
    regime = card.rf
    if regime is None:
        return None
    if regime.permeability is not None:
        raise ValueError(f"RF permeability models are unsupported for {card.name!r}.")

    model = regime.permittivity
    if model is not None and not isinstance(model, Permittivity):
        raise ValueError(f"RF material {card.name!r} requires constant permittivity.")

    conductivity = regime.conductivity if model is None else model.conductivity
    properties = MaterialProperties()
    if conductivity is not None:
        properties.conductivity = _scalar(conductivity, "S/m", card.name)
    if model is not None:
        permittivity = _scalar(model.eps_real, "", card.name)
        properties.permittivity = permittivity
        if model.eps_imag is not None:
            loss = _scalar(model.eps_imag, "", card.name)
            if permittivity <= 0:
                raise ValueError(
                    f"RF loss conversion requires positive permittivity: {card.name!r}."
                )
            properties.loss_tangent = loss / permittivity
    return properties.to_dict()
