"""Material properties database for EM simulation.

PDK LayerStack typically only has material names (e.g., "aluminum", "tungsten").
This database provides the EM properties needed for Palace simulation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MaterialProperties(BaseModel):
    """EM properties for a material."""

    model_config = ConfigDict(validate_assignment=True)

    type: Literal["conductor", "dielectric", "semiconductor"]
    conductivity: float | None = Field(default=None, ge=0)  # S/m (for conductors)
    permittivity: float | None = Field(default=None, ge=1.0)  # relative permittivity
    loss_tangent: float | None = Field(default=None, ge=0, le=1)

    # Optical properties (for photonic simulation, e.g. MEEP)
    refractive_index: float | None = Field(default=None, gt=0)
    extinction_coeff: float | None = Field(default=None, ge=0)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for YAML output."""
        d: dict[str, object] = {"type": self.type}
        if self.conductivity is not None:
            d["conductivity"] = self.conductivity
        if self.permittivity is not None:
            d["permittivity"] = self.permittivity
        if self.loss_tangent is not None:
            d["loss_tangent"] = self.loss_tangent
        if self.refractive_index is not None:
            d["refractive_index"] = self.refractive_index
        if self.extinction_coeff is not None:
            d["extinction_coeff"] = self.extinction_coeff
        return d

    @classmethod
    def conductor(cls, conductivity: float = 5.8e7) -> MaterialProperties:
        """Create a conductor material."""
        return cls(type="conductor", conductivity=conductivity)

    @classmethod
    def dielectric(
        cls, permittivity: float, loss_tangent: float = 0.0
    ) -> MaterialProperties:
        """Create a dielectric material."""
        return cls(
            type="dielectric", permittivity=permittivity, loss_tangent=loss_tangent
        )

    @classmethod
    def optical(
        cls,
        refractive_index: float,
        extinction_coeff: float = 0.0,
    ) -> MaterialProperties:
        """Create a material with optical properties for photonic simulation.

        Args:
            refractive_index: Refractive index (n)
            extinction_coeff: Extinction coefficient (k), default 0
        """
        return cls(
            type="dielectric",
            refractive_index=refractive_index,
            extinction_coeff=extinction_coeff,
        )


# Material properties database
# Sources:
# - IHP SG13G2 process documentation
# - Standard material properties from literature
MATERIALS_DB: dict[str, MaterialProperties] = {
    # Conductors (conductivity in S/m)
    "aluminum": MaterialProperties(
        type="conductor",
        conductivity=3.77e7,
    ),
    "copper": MaterialProperties(
        type="conductor",
        conductivity=5.8e7,
    ),
    "tungsten": MaterialProperties(
        type="conductor",
        conductivity=1.82e7,
    ),
    "gold": MaterialProperties(
        type="conductor",
        conductivity=4.1e7,
    ),
    "TiN": MaterialProperties(
        type="conductor",
        conductivity=5.0e6,
    ),
    "poly_si": MaterialProperties(
        type="conductor",
        conductivity=1.0e5,  # Heavily doped polysilicon
    ),
    # Dielectrics
    "SiO2": MaterialProperties(
        type="dielectric",
        permittivity=4.1,  # Matches gds2palace IHP SG13G2
        loss_tangent=0.0,
        refractive_index=1.44,
    ),
    "passive": MaterialProperties(
        type="dielectric",
        permittivity=6.6,  # IHP SG13G2 passivation layer
        loss_tangent=0.0,
    ),
    "Si3N4": MaterialProperties(
        type="dielectric",
        permittivity=7.5,
        loss_tangent=0.001,
        refractive_index=2.0,
    ),
    "polyimide": MaterialProperties(
        type="dielectric",
        permittivity=3.4,
        loss_tangent=0.002,
    ),
    "air": MaterialProperties(
        type="dielectric",
        permittivity=1.0,
        loss_tangent=0.0,
        refractive_index=1.0,
    ),
    "vacuum": MaterialProperties(
        type="dielectric",
        permittivity=1.0,
        loss_tangent=0.0,
        refractive_index=1.0,
    ),
    # Semiconductors (conductivity values from gds2palace IHP SG13G2)
    "silicon": MaterialProperties(
        type="semiconductor",
        permittivity=11.9,
        conductivity=2.0,  # ~50 Ω·cm substrate (matches gds2palace)
        refractive_index=3.47,
    ),
    "si": MaterialProperties(
        type="semiconductor",
        permittivity=11.9,
        conductivity=2.0,
        refractive_index=3.47,
    ),
    "quartz": MaterialProperties(
        type="dielectric",
        permittivity=4.5
    ),
    "tfln": MaterialProperties(
        type="dielectric",
        permittivity=44.0
    )
}

# Aliases for common variations in naming
MATERIAL_ALIASES: dict[str, str] = {
    "al": "aluminum",
    "cu": "copper",
    "w": "tungsten",
    "au": "gold",
    "tin": "TiN",
    "polysilicon": "poly_si",
    "poly": "poly_si",
    "oxide": "SiO2",
    "sio2": "SiO2",
    "nitride": "Si3N4",
    "sin": "Si3N4",
    "si3n4": "Si3N4",
}


def get_material_properties(material_name: str) -> MaterialProperties | None:
    """Look up material properties by name.

    Args:
        material_name: Material name from PDK (e.g., "aluminum", "tungsten")

    Returns:
        MaterialProperties if found, None otherwise
    """
    name_lower = material_name.lower().strip()

    # Check direct match
    if name_lower in MATERIALS_DB:
        return MATERIALS_DB[name_lower]

    # Check aliases
    if name_lower in MATERIAL_ALIASES:
        return MATERIALS_DB[MATERIAL_ALIASES[name_lower]]

    # Check case-insensitive match in DB
    for db_name, props in MATERIALS_DB.items():
        if db_name.lower() == name_lower:
            return props

    return None


def material_is_conductor(material_name: str) -> bool:
    """Check if a material is a conductor."""
    props = get_material_properties(material_name)
    return props is not None and props.type == "conductor"


def material_is_dielectric(material_name: str) -> bool:
    """Check if a material is a dielectric."""
    props = get_material_properties(material_name)
    return props is not None and props.type == "dielectric"
