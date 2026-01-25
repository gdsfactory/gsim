"""Material database for FDTD simulations.

This module provides material definitions and utilities for working with
optical materials in Tidy3D simulations.
"""

from __future__ import annotations

from functools import partial
from typing import TypeAlias

import tidy3d as td
from tidy3d.components.medium import PoleResidue
from tidy3d.components.types import ComplexNumber

# Material name to Tidy3D medium mapping
material_name_to_tidy3d = {
    "si": td.material_library["cSi"]["Li1993_293K"],
    "sio2": td.material_library["SiO2"]["Horiba"],
    "sin": td.material_library["Si3N4"]["Luke2015PMLStable"],
}

# Simple material mapping with constant permittivity
material_name_to_medium = {
    "si": td.Medium(name="Si", permittivity=3.47**2),
    "sio2": td.Medium(name="SiO2", permittivity=1.47**2),
    "sin": td.Medium(name="SiN", permittivity=2.0**2),
}

MaterialSpecTidy3d: TypeAlias = (
    float
    | int
    | str
    | td.Medium
    | td.CustomMedium
    | td.PoleResidue
    | tuple[float, float]
    | tuple[str, str]
)


def get_epsilon(
    spec: MaterialSpecTidy3d,
    wavelength: float = 1.55,
) -> ComplexNumber:
    """Return permittivity from material database.

    Args:
        spec: material name or refractive index.
        wavelength: wavelength (um).

    Returns:
        Complex permittivity at the specified wavelength.
    """
    medium = get_medium(spec=spec)
    frequency = td.C_0 / wavelength
    return medium.eps_model(frequency)


def get_index(
    spec: MaterialSpecTidy3d,
    wavelength: float = 1.55,
) -> float:
    """Return refractive index from material database.

    Args:
        spec: material name or refractive index.
        wavelength: wavelength (um).

    Returns:
        Real part of refractive index.
    """
    eps_complex = get_epsilon(
        wavelength=wavelength,
        spec=spec,
    )
    n, _ = td.Medium.eps_complex_to_nk(eps_complex)
    return float(n)


def get_nk(
    spec: MaterialSpecTidy3d,
    wavelength: float = 1.55,
) -> tuple[float, float]:
    """Return refractive index and extinction coefficient from material database.

    Args:
        spec: material name or refractive index.
        wavelength: wavelength (um).

    Returns:
        Tuple of (n, k) - refractive index and extinction coefficient.
    """
    eps_complex = get_epsilon(
        wavelength=wavelength,
        spec=spec,
    )
    n, k = td.Medium.eps_complex_to_nk(eps_complex)
    return n, k


def get_medium(spec: MaterialSpecTidy3d) -> td.Medium:
    """Return Medium from materials database.

    Args:
        spec: material name or refractive index.

    Returns:
        Tidy3D Medium object.

    Raises:
        ValueError: If material specification is invalid.
    """
    if isinstance(spec, int | float):
        return td.Medium(permittivity=spec**2)
    elif isinstance(spec, td.Medium | td.Medium2D | td.CustomMedium):
        return spec
    elif isinstance(spec, str) and spec in material_name_to_tidy3d:
        return material_name_to_tidy3d[spec]
    elif isinstance(spec, PoleResidue):
        return spec
    elif isinstance(spec, str) and spec in td.material_library:
        variants = td.material_library[spec].variants
        if len(variants) == 1:
            return list(variants.values())[0].medium
        raise ValueError(
            f"You need to specify the variant of {td.material_library[spec].variants.keys()}"
        )
    elif isinstance(spec, tuple):
        if len(spec) == 2 and isinstance(spec[0], str) and isinstance(spec[1], str):
            return td.material_library[spec[0]][spec[1]]
        raise ValueError("Tuple must have length 2 and be made of strings")
    materials = set(td.material_library.keys())
    raise ValueError(f"Material {spec!r} not in {materials}")


# Convenience functions for common materials
si = partial(get_index, "si")
sio2 = partial(get_index, "sio2")
sin = partial(get_index, "sin")
