"""Material properties database for EM simulation.

PDK LayerStack typically only has material names (e.g., "aluminum", "tungsten").
This database provides the EM properties needed for Palace and MEEP simulation.

Dispersion models:
    Each material can store one or more dispersion models (Sellmeier, Lorentzian,
    or constant-epsilon), each annotated with a domain of validity and citation.
    A material can have multiple models covering different frequency regimes
    (e.g. SiO2: Sellmeier for optical, constant epsilon for RF).
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ValidityRange(BaseModel):
    """Frequency or wavelength range over which a dispersion model is valid.

    Either ``valid_frequency`` or ``valid_wavelength`` should be provided
    (not both). If both are None the validity is unspecified (unknown).
    """

    model_config = ConfigDict(validate_assignment=True)

    valid_frequency: tuple[float, float] | None = Field(
        default=None,
        description="Frequency range in Hz: (f_min, f_max). None = unspecified.",
    )
    valid_wavelength: tuple[float, float] | None = Field(
        default=None,
        description="Wavelength range in um: (wl_min, wl_max). None = unspecified.",
    )

    @property
    def is_unspecified(self) -> bool:
        """Check if neither frequency nor wavelength range is specified."""
        return self.valid_frequency is None and self.valid_wavelength is None

    def covers_frequency(self, freq_hz: float) -> bool:
        """Check if a frequency in Hz falls within this validity range."""
        if self.valid_frequency is not None:
            return self.valid_frequency[0] <= freq_hz <= self.valid_frequency[1]
        if self.valid_wavelength is not None:
            wl_um = 3e8 / freq_hz * 1e6
            return self.valid_wavelength[0] <= wl_um <= self.valid_wavelength[1]
        return False

    def covers_wavelength(self, wl_um: float) -> bool:
        """Check if a wavelength in um falls within this validity range."""
        if self.valid_wavelength is not None:
            return self.valid_wavelength[0] <= wl_um <= self.valid_wavelength[1]
        if self.valid_frequency is not None:
            freq_hz = 3e8 / (wl_um * 1e-6)
            return self.valid_frequency[0] <= freq_hz <= self.valid_frequency[1]
        return False


class SellmeierTerm(BaseModel):
    """One term of the Sellmeier equation: n^2-1 = B*lam^2/(lam^2-C).

    C is stored as C (in um^2), i.e. already squared.
    """

    model_config = ConfigDict(validate_assignment=True)

    B: float
    C: float

    sigma_diagonal: list[float] | None = Field(
        default=None,
        description="Anisotropic oscillator strength [sx, sy, sz]. None = isotropic.",
    )

    def n_squared_contribution(self, wavelength_um: float) -> float:
        """Compute the Sellmeier term's contribution to n^2 at a given wavelength."""
        lam_sq = wavelength_um**2
        return self.B * lam_sq / (lam_sq - self.C)


class LorentzianTerm(BaseModel):
    """One Lorentzian susceptibility pole.

    eps(w) = eps_inf + sigma / (w0^2 - w^2 - i*gamma*w)

    All frequencies in MEEP normalized units (1/um).
    """

    model_config = ConfigDict(validate_assignment=True)

    frequency: float = Field(gt=0, description="Resonance frequency w0 (1/um)")
    gamma: float = Field(ge=0, description="Damping rate gamma (1/um)")
    sigma: float = Field(gt=0, description="Oscillator strength sigma")

    sigma_diagonal: list[float] | None = Field(
        default=None,
        description="Anisotropic strength [sx, sy, sz]. None = isotropic.",
    )


class DispersionModel(BaseModel):
    """A single dispersion model with validity range and citation.

    Supports three model types:
    - ``sellmeier``: Sellmeier equation with B, C terms (optical)
    - ``lorentzian``: Lorentzian susceptibility poles (general dispersive)
    - ``constant``: Constant permittivity or refractive index (single-frequency)
    """

    model_config = ConfigDict(validate_assignment=True)

    type: Literal["sellmeier", "lorentzian", "constant"]

    sellmeier_terms: list[SellmeierTerm] | None = Field(
        default=None,
        description="Sellmeier B, C terms. Required when type='sellmeier'.",
    )
    lorentzian_terms: list[LorentzianTerm] | None = Field(
        default=None,
        description="Lorentzian poles. Required when type='lorentzian'.",
    )
    permittivity: float | None = Field(
        default=None,
        ge=1.0,
        description="Constant relative permittivity. For type='constant'.",
    )
    refractive_index: float | None = Field(
        default=None,
        gt=0,
        description="Constant refractive index. For type='constant'.",
    )

    epsilon_inf: float = Field(
        default=1.0,
        ge=1.0,
        description="High-frequency permittivity eps_inf for Lorentzian/Sellmeier.",
    )

    validity: ValidityRange = Field(
        default_factory=ValidityRange,
        description="Validity range of this model.",
    )
    source: str = Field(
        default="",
        description="Citation string (paper, PDK, etc.).",
    )

    def evaluate_n(self, wavelength_um: float) -> float:
        """Evaluate the refractive index at a given wavelength in um."""
        if self.type == "sellmeier":
            if not self.sellmeier_terms:
                raise ValueError("Sellmeier model has no terms")
            n_sq = self.epsilon_inf
            for term in self.sellmeier_terms:
                n_sq += term.n_squared_contribution(wavelength_um)
            if n_sq < 0:
                raise ValueError(
                    f"Sellmeier model gives n^2<0 at wavelength={wavelength_um} um "
                    f"(likely outside validity range)"
                )
            return math.sqrt(n_sq)

        if self.type == "constant":
            if self.refractive_index is not None:
                return self.refractive_index
            if self.permittivity is not None:
                return math.sqrt(self.permittivity)
            raise ValueError(
                "Constant model has neither refractive_index nor permittivity"
            )

        if self.type == "lorentzian":
            if self.lorentzian_terms is None:
                raise ValueError("Lorentzian model has no terms")
            freq = 1.0 / wavelength_um
            eps_real = self.epsilon_inf
            for pole in self.lorentzian_terms:
                w0_sq = pole.frequency**2
                w_sq = freq**2
                denom_real = w0_sq - w_sq
                denom_mag_sq = denom_real**2 + (pole.gamma * freq) ** 2
                if denom_mag_sq == 0:
                    continue
                eps_real += pole.sigma * denom_real / denom_mag_sq
            if eps_real < 1.0:
                raise ValueError(
                    f"Lorentzian model gives eps<1 at lambda={wavelength_um} um"
                )
            return math.sqrt(eps_real)

        raise ValueError(f"Unknown dispersion model type: {self.type}")

    def evaluate_permittivity(self, wavelength_um: float) -> float:
        """Evaluate the relative permittivity at a given wavelength in um."""
        if self.type == "constant" and self.permittivity is not None:
            return self.permittivity
        n = self.evaluate_n(wavelength_um)
        return n**2


class ResolvedMaterial(BaseModel):
    """Result of evaluating a material's dispersion model at a specific frequency.

    Contains the scalar properties needed by solvers, plus metadata about
    which model was used and whether the evaluation frequency is within
    the model's validity range.
    """

    model_config = ConfigDict(validate_assignment=True)

    permittivity: float | None = Field(default=None, ge=1.0)
    refractive_index: float | None = Field(default=None, gt=0)
    extinction_coeff: float = Field(default=0.0, ge=0)
    conductivity: float | None = Field(default=None, ge=0)
    loss_tangent: float | None = Field(default=None, ge=0, le=1)

    permittivity_diagonal: list[float] | None = None
    conductivity_diagonal: list[float] | None = None
    permeability: list[float] | None = None
    loss_tangent_diagonal: list[float] | None = None
    material_axes: list[list[float]] | None = None

    model_type: str = Field(default="", description="Type of dispersion model used")
    model_source: str = Field(default="", description="Citation of the model used")
    within_validity: bool = Field(
        default=True,
        description=(
            "Whether the evaluation frequency is within the model's validity range"
        ),
    )
    validity_note: str = Field(
        default="",
        description="Human-readable note about validity status",
    )


class MaterialProperties(BaseModel):
    """EM properties for a material.

    Supports both legacy scalar properties (permittivity, refractive_index)
    and frequency-dependent dispersion models with validity ranges.
    """

    model_config = ConfigDict(validate_assignment=True)

    type: Literal["conductor", "dielectric", "semiconductor"]
    conductivity: float | None = Field(default=None, ge=0)
    permittivity: float | None = Field(default=None, ge=1.0)
    loss_tangent: float | None = Field(default=None, ge=0, le=1)

    permittivity_diagonal: list[float] | None = None
    permeability: list[float] | None = None
    loss_tangent_diagonal: list[float] | None = None
    material_axes: list[list[float]] | None = None
    conductivity_diagonal: list[float] | None = None

    refractive_index: float | None = Field(default=None, gt=0)
    extinction_coeff: float | None = Field(default=None, ge=0)

    dispersion_models: list[DispersionModel] = Field(
        default_factory=list,
        description="Frequency-dependent dispersion models with validity ranges.",
    )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for YAML/JSON output."""
        d: dict[str, object] = {"type": self.type}
        if self.conductivity is not None:
            d["conductivity"] = self.conductivity
        if self.permittivity is not None:
            d["permittivity"] = self.permittivity
        if self.loss_tangent is not None:
            d["loss_tangent"] = self.loss_tangent
        if self.permittivity_diagonal is not None:
            d["permittivity_diagonal"] = self.permittivity_diagonal
        if self.permeability is not None:
            d["permeability"] = self.permeability
        if self.loss_tangent_diagonal is not None:
            d["loss_tangent_diagonal"] = self.loss_tangent_diagonal
        if self.material_axes is not None:
            d["material_axes"] = self.material_axes
        if self.conductivity_diagonal is not None:
            d["conductivity_diagonal"] = self.conductivity_diagonal
        if self.refractive_index is not None:
            d["refractive_index"] = self.refractive_index
        if self.extinction_coeff is not None:
            d["extinction_coeff"] = self.extinction_coeff
        if self.dispersion_models:
            d["dispersion_models"] = [m.model_dump() for m in self.dispersion_models]
        return d

    def evaluate_at_wavelength(self, wavelength_um: float) -> ResolvedMaterial:
        """Evaluate material properties at a specific wavelength.

        Resolution strategy:
        1. Scan dispersion_models for one whose validity covers the wavelength.
        2. If no model covers it, use a model with unspecified validity (warn).
        3. If no dispersion models, fall back to legacy scalar fields.
        """
        covered_models = []
        unspecified_models = []

        for model in self.dispersion_models:
            if model.validity.is_unspecified:
                unspecified_models.append(model)
            elif model.validity.covers_wavelength(wavelength_um):
                covered_models.append(model)

        selected: DispersionModel | None = None
        within_validity = True
        validity_note = ""

        if covered_models:
            selected = covered_models[0]
            within_validity = True
        elif unspecified_models:
            selected = unspecified_models[0]
            within_validity = True
            validity_note = (
                f"validity range unspecified (source: {selected.source or 'unknown'})"
            )
            warnings.warn(
                f"Material model for evaluation at wavelength={wavelength_um} um "
                f"has unspecified validity range. {validity_note}",
                stacklevel=3,
            )

        if selected is not None:
            n = selected.evaluate_n(wavelength_um)
            return ResolvedMaterial(
                refractive_index=n,
                permittivity=n**2,
                extinction_coeff=self.extinction_coeff or 0.0,
                conductivity=self.conductivity,
                loss_tangent=self.loss_tangent,
                permittivity_diagonal=self.permittivity_diagonal,
                conductivity_diagonal=self.conductivity_diagonal,
                permeability=self.permeability,
                loss_tangent_diagonal=self.loss_tangent_diagonal,
                material_axes=self.material_axes,
                model_type=selected.type,
                model_source=selected.source,
                within_validity=within_validity,
                validity_note=validity_note,
            )

        if self.refractive_index is not None:
            return ResolvedMaterial(
                refractive_index=self.refractive_index,
                permittivity=self.refractive_index**2,
                extinction_coeff=self.extinction_coeff or 0.0,
                conductivity=self.conductivity,
                loss_tangent=self.loss_tangent,
                permittivity_diagonal=self.permittivity_diagonal,
                conductivity_diagonal=self.conductivity_diagonal,
                permeability=self.permeability,
                loss_tangent_diagonal=self.loss_tangent_diagonal,
                material_axes=self.material_axes,
                within_validity=True,
                validity_note="legacy scalar (no dispersion models)",
            )

        if self.permittivity is not None:
            return ResolvedMaterial(
                refractive_index=math.sqrt(self.permittivity),
                permittivity=self.permittivity,
                extinction_coeff=self.extinction_coeff or 0.0,
                conductivity=self.conductivity,
                loss_tangent=self.loss_tangent,
                permittivity_diagonal=self.permittivity_diagonal,
                conductivity_diagonal=self.conductivity_diagonal,
                permeability=self.permeability,
                loss_tangent_diagonal=self.loss_tangent_diagonal,
                material_axes=self.material_axes,
                within_validity=True,
                validity_note="legacy scalar (no dispersion models)",
            )

        return ResolvedMaterial(
            conductivity=self.conductivity,
            permittivity_diagonal=self.permittivity_diagonal,
            conductivity_diagonal=self.conductivity_diagonal,
            permeability=self.permeability,
            loss_tangent_diagonal=self.loss_tangent_diagonal,
            material_axes=self.material_axes,
            within_validity=False,
            validity_note="no optical or RF data available",
        )

    def evaluate_at_frequency(self, freq_hz: float) -> ResolvedMaterial:
        """Evaluate material properties at a specific frequency in Hz."""
        wavelength_um = 3e8 / freq_hz * 1e6
        return self.evaluate_at_wavelength(wavelength_um)

    def index_variation(self, wavelength_um: float, bandwidth_um: float) -> float:
        """Compute fractional index variation dn/n across a bandwidth.

        Returns 0.0 if the material has no dispersive model or scalar n.
        """
        if not self.dispersion_models:
            return 0.0

        wl_min = wavelength_um - bandwidth_um / 2
        wl_max = wavelength_um + bandwidth_um / 2

        resolved_center = self.evaluate_at_wavelength(wavelength_um)
        if resolved_center.refractive_index is None:
            return 0.0
        n_center = resolved_center.refractive_index
        if n_center == 0:
            return 0.0

        try:
            n_min = self.evaluate_at_wavelength(wl_min).refractive_index or n_center
            n_max = self.evaluate_at_wavelength(wl_max).refractive_index or n_center
        except (ValueError, ZeroDivisionError):
            return 0.0

        delta_n = max(abs(n_max - n_center), abs(n_min - n_center))
        return delta_n / n_center

    @classmethod
    def conductor(cls, conductivity: float = 5.8e7) -> MaterialProperties:
        """Create a conductor material with the given conductivity in S/m."""
        return cls(type="conductor", conductivity=conductivity)

    @classmethod
    def dielectric(
        cls, permittivity: float, loss_tangent: float = 0.0
    ) -> MaterialProperties:
        """Create a dielectric material with permittivity and loss tangent."""
        return cls(
            type="dielectric", permittivity=permittivity, loss_tangent=loss_tangent
        )

    @classmethod
    def optical(
        cls,
        refractive_index: float,
        extinction_coeff: float = 0.0,
    ) -> MaterialProperties:
        """Create a dielectric material with optical properties.

        For photonic simulation.
        """
        return cls(
            type="dielectric",
            refractive_index=refractive_index,
            extinction_coeff=extinction_coeff,
        )


MATERIALS_DB: dict[str, MaterialProperties] = {
    "aluminum": MaterialProperties(
        type="conductor",
        conductivity=3.77e7,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=1.0,
                validity=ValidityRange(valid_frequency=(0, 100e9)),
                source="conductivity: standard value",
            ),
        ],
    ),
    "copper": MaterialProperties(
        type="conductor",
        conductivity=5.8e7,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=1.0,
                validity=ValidityRange(valid_frequency=(0, 100e9)),
                source="conductivity: standard value",
            ),
        ],
    ),
    "tungsten": MaterialProperties(
        type="conductor",
        conductivity=1.82e7,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=1.0,
                validity=ValidityRange(valid_frequency=(0, 100e9)),
                source="conductivity: standard value",
            ),
        ],
    ),
    "gold": MaterialProperties(
        type="conductor",
        conductivity=4.1e7,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=1.0,
                validity=ValidityRange(valid_frequency=(0, 100e9)),
                source="conductivity: standard value",
            ),
        ],
    ),
    "TiN": MaterialProperties(
        type="conductor",
        conductivity=5.0e6,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=1.0,
                validity=ValidityRange(valid_frequency=(0, 100e9)),
                source="conductivity: IHP SG13G2",
            ),
        ],
    ),
    "poly_si": MaterialProperties(
        type="conductor",
        conductivity=1.0e5,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=1.0,
                validity=ValidityRange(valid_frequency=(0, 100e9)),
                source="conductivity: IHP SG13G2 (heavily doped polysilicon)",
            ),
        ],
    ),
    "SiO2": MaterialProperties(
        type="dielectric",
        permittivity=4.1,
        loss_tangent=0.0,
        refractive_index=1.44,
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=0.696, C=0.0684**2),
                    SellmeierTerm(B=0.408, C=0.1162**2),
                    SellmeierTerm(B=0.897, C=9.896**2),
                ],
                validity=ValidityRange(valid_wavelength=(0.21, 3.71)),
                source="Malitson 1965, Appl. Opt. 4(9)",
            ),
            DispersionModel(
                type="constant",
                permittivity=4.1,
                validity=ValidityRange(valid_frequency=(0, 10e9)),
                source="IHP SG13G2 PDK",
            ),
        ],
    ),
    "passive": MaterialProperties(
        type="dielectric",
        permittivity=6.6,
        loss_tangent=0.0,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=6.6,
                validity=ValidityRange(valid_frequency=(0, 10e9)),
                source="IHP SG13G2 passivation layer",
            ),
        ],
    ),
    "Si3N4": MaterialProperties(
        type="dielectric",
        permittivity=7.5,
        loss_tangent=0.001,
        refractive_index=2.0,
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=2.814, C=0.0136**2),
                    SellmeierTerm(B=0.388, C=0.2738**2),
                ],
                validity=ValidityRange(valid_wavelength=(0.31, 5.5)),
                source="Luke et al. 2015",
            ),
            DispersionModel(
                type="constant",
                permittivity=7.5,
                validity=ValidityRange(valid_frequency=(0, 10e9)),
                source="IHP SG13G2 PDK",
            ),
        ],
    ),
    "polyimide": MaterialProperties(
        type="dielectric",
        permittivity=3.4,
        loss_tangent=0.002,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=3.4,
                validity=ValidityRange(valid_frequency=(0, 10e9)),
                source="IHP SG13G2 PDK",
            ),
        ],
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
    "silicon": MaterialProperties(
        type="semiconductor",
        permittivity=11.9,
        conductivity=2.0,
        refractive_index=3.47,
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=10.357, C=0.8832**2),
                    SellmeierTerm(B=0.860, C=6.004**2),
                ],
                validity=ValidityRange(valid_wavelength=(1.36, 11)),
                source="Salzberg & Villa 1957",
            ),
            DispersionModel(
                type="constant",
                permittivity=11.9,
                validity=ValidityRange(valid_frequency=(0, 10e9)),
                source="IHP SG13G2 PDK",
            ),
        ],
    ),
    "si": MaterialProperties(
        type="semiconductor",
        permittivity=11.9,
        conductivity=2.0,
        refractive_index=3.47,
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=10.357, C=0.8832**2),
                    SellmeierTerm(B=0.860, C=6.004**2),
                ],
                validity=ValidityRange(valid_wavelength=(1.36, 11)),
                source="Salzberg & Villa 1957",
            ),
            DispersionModel(
                type="constant",
                permittivity=11.9,
                validity=ValidityRange(valid_frequency=(0, 10e9)),
                source="IHP SG13G2 PDK",
            ),
        ],
    ),
    "sapphire": MaterialProperties(
        type="dielectric",
        permittivity=9.3,
        loss_tangent=3e-5,
        refractive_index=1.77,
        permittivity_diagonal=[9.3, 9.3, 11.5],
        permeability=[0.99999975, 0.99999975, 0.99999979],
        loss_tangent_diagonal=[3e-5, 3e-5, 8.6e-5],
        material_axes=[[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=1.024, C=0.0532**2),
                    SellmeierTerm(B=1.076, C=0.1550**2),
                ],
                validity=ValidityRange(valid_wavelength=(0.2, 5.0)),
                source="Malitson & Dodge 1972",
            ),
            DispersionModel(
                type="constant",
                permittivity=9.3,
                validity=ValidityRange(valid_frequency=(0, 10e9)),
                source="RF handbook value",
            ),
        ],
    ),
    "quartz": MaterialProperties(
        type="dielectric",
        permittivity=4.5,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=4.5,
                source="unspecified",
            ),
        ],
    ),
    "germanium": MaterialProperties(
        type="semiconductor",
        permittivity=16.0,
        refractive_index=4.18,
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=8.288, C=0.4682**2),
                    SellmeierTerm(B=1.784, C=5.768**2),
                ],
                validity=ValidityRange(valid_wavelength=(2.5, 12)),
                source="Barnes & Piltch 1979",
            ),
        ],
    ),
    "ge": MaterialProperties(
        type="semiconductor",
        permittivity=16.0,
        refractive_index=4.18,
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=8.288, C=0.4682**2),
                    SellmeierTerm(B=1.784, C=5.768**2),
                ],
                validity=ValidityRange(valid_wavelength=(2.5, 12)),
                source="Barnes & Piltch 1979",
            ),
        ],
    ),
    "tfln": MaterialProperties(
        type="dielectric",
        permittivity=44.0,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=44.0,
                source="unspecified",
            ),
        ],
    ),
}

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
    "ge": "germanium",
}


def get_material_properties(material_name: str) -> MaterialProperties | None:
    """Look up material properties by name."""
    name_lower = material_name.lower().strip()

    if name_lower in MATERIALS_DB:
        return MATERIALS_DB[name_lower]

    if name_lower in MATERIAL_ALIASES:
        return MATERIALS_DB[MATERIAL_ALIASES[name_lower]]

    for db_name, props in MATERIALS_DB.items():
        if db_name.lower() == name_lower:
            return props

    return None


def resolve_material_at_wavelength(
    material_name: str,
    wavelength_um: float,
    overrides: dict[str, MaterialProperties] | None = None,
) -> ResolvedMaterial | None:
    """Resolve a material's properties at a specific wavelength.

    Priority:
    1. User override (if provided)
    2. Built-in database
    3. None + warning

    Args:
        material_name: Material name from PDK
        wavelength_um: Target wavelength in um
        overrides: User-supplied material property overrides

    Returns:
        ResolvedMaterial with evaluated properties, or None if not found
    """
    overrides = overrides or {}

    if material_name in overrides:
        return overrides[material_name].evaluate_at_wavelength(wavelength_um)

    props = get_material_properties(material_name)
    if props is not None:
        resolved = props.evaluate_at_wavelength(wavelength_um)
        if not resolved.within_validity and resolved.validity_note:
            warnings.warn(
                f"Material '{material_name}' at wavelength={wavelength_um} um: "
                f"{resolved.validity_note}",
                stacklevel=2,
            )
        return resolved

    warnings.warn(
        f"Material '{material_name}' not found in database. "
        f"Available: {list(MATERIALS_DB.keys())}",
        stacklevel=2,
    )
    return None


def should_enable_dispersion(
    material_name: str,
    wavelength_um: float,
    bandwidth_um: float,
    threshold: float = 0.005,
    overrides: dict[str, MaterialProperties] | None = None,
) -> bool:
    """Determine if dispersion should be enabled for a material.

    Compares dn/n across the bandwidth against a threshold.

    Returns True if dispersion is significant (dn/n > threshold).

    Args:
        material_name: Material name
        wavelength_um: Center wavelength in um
        bandwidth_um: Source bandwidth in um
        threshold: Fractional index variation threshold (default 0.5%)
        overrides: User-supplied material property overrides

    Returns:
        True if dispersion should be enabled
    """
    overrides = overrides or {}

    if material_name in overrides:
        return (
            overrides[material_name].index_variation(wavelength_um, bandwidth_um)
            > threshold
        )

    props = get_material_properties(material_name)
    if props is not None:
        return props.index_variation(wavelength_um, bandwidth_um) > threshold

    return False


def material_is_conductor(material_name: str) -> bool:
    """Check if a material is a conductor."""
    props = get_material_properties(material_name)
    return props is not None and props.type == "conductor"


def material_is_dielectric(material_name: str) -> bool:
    """Check if a material is a dielectric."""
    props = get_material_properties(material_name)
    return props is not None and props.type == "dielectric"
