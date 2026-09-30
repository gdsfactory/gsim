"""Regression tests for portable analytic PDK material cards."""

import pytest
from pdk_schema import AnalyticDispersion, MaterialCard

from gsim.common.materials import (
    GSIM_MATERIAL_CARDS,
    MaterialModelError,
    WavelengthOutOfRangeError,
    resolve_material_snapshot,
)


def _analytic_card(expression: str, **options) -> MaterialCard:
    """Round-trip an analytic project card through its portable JSON format."""
    model = AnalyticDispersion(
        expression=expression,
        **{
            "output": "n_squared",
            "inputs": {"wl": {"quantity": "wavelength", "unit": "um"}},
            "parameters": {},
            "validity": {
                "at": None,
                "over": {
                    "wavelength": {"min": 1.2, "max": 1.7, "unit": "um", "label": None}
                },
                "on_out_of_range": "raise",
            },
            "variation": None,
            "conductivity": None,
            **options,
        },
    )
    card = GSIM_MATERIAL_CARDS["SiN"].model_copy(deep=True)
    assert card.optical is not None
    card.optical.permittivity = model
    return MaterialCard.model_validate_json(card.model_dump_json())


@pytest.mark.parametrize(
    ("coefficients", "expected_index"),
    [
        ({"B1": 2.96415, "C1": 0.12946, "D": 0.02156}, 1.983223471),
        ({"B1": 1.10712, "C1": 0.09155, "D": 0.00590}, 1.448040431),
    ],
)
def test_an800_modified_sellmeier_cards(coefficients, expected_index) -> None:
    card = _analytic_card(
        "1 + B1*wl**2/(wl**2 - C1**2) - D*wl**2",
        parameters=coefficients,
    )

    snapshot = resolve_material_snapshot("SiN", 1.55, {"SiN": card})

    assert snapshot.refractive_index == pytest.approx(expected_index, abs=1e-9, rel=0)
    assert snapshot.extinction_coefficient == 0.0
    assert snapshot.source == "project"
    assert snapshot.card.optical is not None
    assert isinstance(snapshot.card.optical.permittivity, AnalyticDispersion)


@pytest.mark.parametrize(("unit", "scale_um"), [("um", 1.0), ("nm", 1e-3), ("m", 1e6)])
@pytest.mark.parametrize("output", ["n", "n_squared"])
def test_analytic_inputs_use_declared_wavelength_units(unit, scale_um, output) -> None:
    equation = "base + slope*wave"
    card = _analytic_card(
        equation if output == "n" else f"({equation})**2",
        output=output,
        inputs={"wave": {"quantity": "wavelength", "unit": unit}},
        parameters={"base": 1.5, "slope": 0.2 * scale_um},
    )

    snapshot = resolve_material_snapshot("SiN", 1.5525, {"SiN": card})

    assert snapshot.refractive_index == pytest.approx(1.8105, abs=1e-12, rel=0)


@pytest.mark.parametrize("wavelength_um", [1.2, 1.7])
def test_analytic_validity_accepts_endpoints(wavelength_um) -> None:
    card = _analytic_card("4")

    snapshot = resolve_material_snapshot("SiN", wavelength_um, {"SiN": card})

    assert snapshot.refractive_index == 2.0


@pytest.mark.parametrize("wavelength_um", [1.1999, 1.7001])
def test_analytic_validity_is_checked_before_evaluation(wavelength_um) -> None:
    card = _analytic_card("1/(wl-wl)")

    with pytest.raises(WavelengthOutOfRangeError, match="valid from"):
        resolve_material_snapshot("SiN", wavelength_um, {"SiN": card})


def test_analytic_validity_converts_band_units() -> None:
    card = _analytic_card(
        "4",
        validity={
            "at": None,
            "over": {
                "wavelength": {"min": 1200, "max": 1700, "unit": "nm", "label": None}
            },
            "on_out_of_range": "raise",
        },
    )

    assert resolve_material_snapshot("SiN", 1.55, {"SiN": card}).refractive_index == 2.0
    with pytest.raises(WavelengthOutOfRangeError, match="valid from"):
        resolve_material_snapshot("SiN", 1.8, {"SiN": card})


@pytest.mark.parametrize("output", ["n", "n_squared"])
@pytest.mark.parametrize(
    "expression", ["-1", "0", "(-1)**0.5", "sqrt(-1)", "1/(wl-wl)", "exp(1000)"]
)
def test_analytic_equations_reject_invalid_indices(expression, output) -> None:
    card = _analytic_card(expression, output=output)

    with pytest.raises(MaterialModelError):
        resolve_material_snapshot("SiN", 1.55, {"SiN": card})


def test_analytic_permittivity_is_not_implicitly_an_index() -> None:
    card = _analytic_card("4", output="eps_real")

    with pytest.raises(MaterialModelError, match="must output n or n_squared"):
        resolve_material_snapshot("SiN", 1.55, {"SiN": card})


def test_analytic_temperature_inputs_are_explicitly_unsupported() -> None:
    card = _analytic_card(
        "temperature",
        inputs={"temperature": {"quantity": "temperature", "unit": "K"}},
    )

    with pytest.raises(MaterialModelError, match="only supports wavelength inputs"):
        resolve_material_snapshot("SiN", 1.55, {"SiN": card})


def test_analytic_constants_do_not_require_inputs() -> None:
    card = _analytic_card("n_ref", output="n", inputs={}, parameters={"n_ref": 1.8})

    assert resolve_material_snapshot("SiN", 1.55, {"SiN": card}).refractive_index == 1.8


def test_changed_analytic_coefficients_do_not_reuse_old_results() -> None:
    first_card = _analytic_card("n_ref", output="n", parameters={"n_ref": 1.8})
    second_card = _analytic_card("n_ref", output="n", parameters={"n_ref": 1.9})

    first = resolve_material_snapshot("SiN", 1.55, {"SiN": first_card})
    second = resolve_material_snapshot("SiN", 1.55, {"SiN": second_card})

    assert first.refractive_index == 1.8
    assert second.refractive_index == 1.9
