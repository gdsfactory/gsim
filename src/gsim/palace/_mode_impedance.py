"""Read legacy scalar and complex Palace boundary-mode impedance columns."""

from __future__ import annotations

import math
import re

_SCALAR_COLUMN = re.compile(r"^(Z_PV|Z_VI|L_PV|C_PV|L_VI|C_VI)\[(\d+)\]")
_COMPLEX_COLUMN = re.compile(r"^(Re|Im)\{(Z_PV|Z_VI)\[(\d+)\]\}")

type ModeImpedances = dict[int, dict[int, dict[str, float]]]


def _finite_float(value: object) -> float | None:
    """Read a finite CSV number, or return None for unavailable values."""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_mode_impedances(rows: list[dict[str, str]]) -> ModeImpedances:
    """Group impedance components by path and mode, retaining scalar aliases."""
    result: ModeImpedances = {}
    modern_quantities: set[tuple[int, int, str]] = set()
    for row_index, raw_row in enumerate(rows, start=1):
        row = {
            str(key).strip(): value for key, value in raw_row.items() if key is not None
        }
        mode_number = _finite_float(row.get("m"))
        mode_id = int(mode_number) if mode_number is not None else row_index
        for column, raw_value in row.items():
            complex_match = _COMPLEX_COLUMN.match(column)
            scalar_match = _SCALAR_COLUMN.match(column)
            if complex_match is not None:
                part, quantity, index_text = complex_match.groups()
                modern_quantities.add((int(index_text), mode_id, quantity))
                quantity = f"{part}{{{quantity}}}"
            elif scalar_match is not None:
                quantity, index_text = scalar_match.groups()
            else:
                continue
            value = _finite_float(raw_value)
            if value is not None:
                result.setdefault(int(index_text), {}).setdefault(mode_id, {})[
                    quantity
                ] = value

    for index, mode_id, quantity in modern_quantities:
        metrics = result.get(index, {}).get(mode_id, {})
        metrics.pop(quantity, None)
        real_part = metrics.get(f"Re{{{quantity}}}")
        if real_part is not None:
            metrics[quantity] = real_part
    return result


def complex_mode_impedance(metrics: dict[str, float], quantity: str) -> complex | None:
    """Combine a finite component pair or read a legacy scalar as real-only."""
    real_part = metrics.get(f"Re{{{quantity}}}")
    imaginary_part = metrics.get(f"Im{{{quantity}}}")
    if real_part is None and imaginary_part is None:
        scalar = metrics.get(quantity)
        return complex(scalar) if scalar is not None else None
    if real_part is None or imaginary_part is None:
        return None
    return complex(real_part, imaginary_part)
