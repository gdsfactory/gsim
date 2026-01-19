"""Material properties database for EM simulation.

PDK LayerStack typically only has material names (e.g., "aluminum", "tungsten").
This database provides the EM properties needed for Palace simulation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MaterialProperties:
    """EM properties for a material."""

    type: str  # "conductor", "dielectric", "semiconductor"
    conductivity: float | None = None  # S/m (for conductors)
    permittivity: float | None = None  # relative permittivity
    loss_tangent: float | None = None  # dielectric loss tangent

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for YAML output."""
        d: dict[str, object] = {"type": self.type}
        if self.conductivity is not None:
            d["conductivity"] = self.conductivity
        if self.permittivity is not None:
            d["permittivity"] = self.permittivity
        if self.loss_tangent is not None:
            d["loss_tangent"] = self.loss_tangent
        return d


# Material properties database
# Sources:
# - IHP SG13G2 process documentation
# - Standard material properties from literature
MATERIALS_DB: dict[str, MaterialProperties] = {
    # Conductors (conductivity in S/m)
    "aluminum": MaterialProperties(
        type="conductor",
        conductivity=3.77e7,  # S/m
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
        loss_tangent=0.0,  # Matches gds2palace (no loss)
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
    ),
    "vacuum": MaterialProperties(
        type="dielectric",
        permittivity=1.0,
        loss_tangent=0.0,
    ),
    # Semiconductors (conductivity values from gds2palace IHP SG13G2)
    "silicon": MaterialProperties(
        type="semiconductor",
        permittivity=11.9,
        conductivity=2.0,  # ~50 Ω·cm substrate (matches gds2palace)
    ),
    "si": MaterialProperties(
        type="semiconductor",
        permittivity=11.9,
        conductivity=2.0,  # ~50 Ω·cm substrate (matches gds2palace)
    ),
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
    # Normalize name
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
