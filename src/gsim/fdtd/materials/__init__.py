"""Materials submodule for FDTD simulations.

This module provides material definitions and utilities for optical simulations.
"""

from gsim.fdtd.materials.database import (
    MaterialSpecTidy3d,
    get_epsilon,
    get_index,
    get_medium,
    get_nk,
    material_name_to_medium,
    material_name_to_tidy3d,
    si,
    sin,
    sio2,
)
from gsim.fdtd.materials.types import (
    Sparameters,
    Tidy3DElementMapping,
    Tidy3DMedium,
)

__all__ = [
    "MaterialSpecTidy3d",
    "Sparameters",
    "Tidy3DElementMapping",
    "Tidy3DMedium",
    "get_epsilon",
    "get_index",
    "get_medium",
    "get_nk",
    "material_name_to_medium",
    "material_name_to_tidy3d",
    "si",
    "sin",
    "sio2",
]
