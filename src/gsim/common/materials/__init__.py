"""Built-in optical material cards and strict snapshot evaluation."""

from gsim.common.materials.registry import (
    GSIM_MATERIAL_CARDS,
    find_material_card,
    get_material_card,
    get_project_material_cards,
)
from gsim.common.materials.si_li_293k import SI_LI_293K
from gsim.common.materials.si_salzberg import SI_SALZBERG
from gsim.common.materials.sin_luke import SIN_LUKE
from gsim.common.materials.sio2_malitson import SIO2_MALITSON
from gsim.common.materials.sio2_palik import (
    SIO2_PALIK,
    SIO2_PALIK_LOSSLESS,
)
from gsim.common.materials.snapshots import (
    MaterialModelError,
    MaterialNotFoundError,
    MaterialResolutionError,
    MaterialSnapshot,
    WavelengthOutOfRangeError,
    resolve_material_snapshot,
)

__all__ = [
    "GSIM_MATERIAL_CARDS",
    "SIN_LUKE",
    "SIO2_MALITSON",
    "SIO2_PALIK",
    "SIO2_PALIK_LOSSLESS",
    "SI_LI_293K",
    "SI_SALZBERG",
    "MaterialModelError",
    "MaterialNotFoundError",
    "MaterialResolutionError",
    "MaterialSnapshot",
    "WavelengthOutOfRangeError",
    "find_material_card",
    "get_material_card",
    "get_project_material_cards",
    "resolve_material_snapshot",
]
