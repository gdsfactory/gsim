"""Material properties database for EM simulation.

PDK LayerStack typically only has material names (e.g., "aluminum", "tungsten").
This database provides the EM properties needed for Palace and MEEP simulation.

Dispersion models:
    Each material can store one or more dispersion models (Sellmeier, Lorentzian,
    or constant-epsilon), each annotated with a domain of validity and citation.
    A material can have multiple models covering different frequency regimes
    (e.g. SiO2: Sellmeier for optical, constant epsilon for RF).

Unified tensor fields:
    ``permittivity``, ``conductivity``, ``loss_tangent``, and ``permeability``
    each accept either a scalar (isotropic) or a list of 3 floats (anisotropic).
    This replaces the old separate ``permittivity_diagonal`` etc. fields.

Resolution priority:
    1. dispersion_models evaluated at target frequency (-> permittivity)
    2. PDK overlay (constant-eps)
    3. ``permittivity`` scalar or tensor (directly specified)
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
            wl_um = 299_792_458 / freq_hz * 1e6
            return self.valid_wavelength[0] <= wl_um <= self.valid_wavelength[1]
        return False

    def covers_wavelength(self, wl_um: float) -> bool:
        """Check if a wavelength in um falls within this validity range."""
        if self.valid_wavelength is not None:
            return self.valid_wavelength[0] <= wl_um <= self.valid_wavelength[1]
        if self.valid_frequency is not None:
            freq_hz = 299_792_458 / (wl_um * 1e-6)
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
    - ``constant``: Constant permittivity (single-frequency)
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
            if self.permittivity is not None:
                return math.sqrt(self.permittivity)
            raise ValueError("Constant model has no permittivity")

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


def _as_list(val: float | list[float] | None, n: int = 3) -> list[float] | None:
    """Expand a scalar to a list of n identical values, or pass list through."""
    if val is None:
        return None
    if isinstance(val, list):
        return val
    return [val] * n


def _is_tensor(val: float | list[float] | None) -> bool:
    """Check if a value is an anisotropic tensor (list of 3)."""
    return isinstance(val, list)


class ResolvedMaterial(BaseModel):
    """Result of evaluating a material's dispersion model at a specific frequency.

    Contains the scalar/tensor properties needed by solvers, plus metadata about
    which model was used and whether the evaluation frequency is within
    the model's validity range.

    Tensor fields (permittivity, conductivity, loss_tangent, permeability)
    accept a scalar (isotropic) or list of 3 floats (anisotropic).

    The ``behavior`` property returns "conductive" or "dielectric" based on
    the resolved properties at the evaluation frequency — a material can be
    conductive at RF and dielectric at optical wavelengths.
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    permittivity: float | list[float] | None = Field(
        default=None,
        description="Relative permittivity. Scalar (isotropic) or [ex, ey, ez].",
    )
    conductivity: float | list[float] | None = Field(
        default=None, description="Conductivity S/m. Scalar or [sx, sy, sz]."
    )
    loss_tangent: float | list[float] | None = Field(
        default=None,
        description="Loss tangent. Scalar or [tx, ty, tz].",
    )
    permeability: float | list[float] | None = Field(
        default=None,
        description="Relative permeability. Scalar or [mx, my, mz].",
    )
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

    CONDUCTIVITY_THRESHOLD: float = 1e4

    @property
    def behavior(self) -> Literal["conductive", "dielectric"]:
        """Frequency-aware classification.

        Conductive if significant conductivity
        and no dispersive permittivity model active, else dielectric.
        """
        cond = self.conductivity_scalar
        if (
            cond is not None
            and cond >= self.CONDUCTIVITY_THRESHOLD
            and self.model_type not in ("sellmeier", "lorentzian")
        ):
            return "conductive"
        return "dielectric"

    @property
    def permittivity_scalar(self) -> float | None:
        """Return scalar permittivity, or first element of tensor."""
        if self.permittivity is None:
            return None
        if isinstance(self.permittivity, list):
            return self.permittivity[0]
        return self.permittivity

    @property
    def conductivity_scalar(self) -> float | None:
        """Return scalar conductivity, or first element of tensor."""
        if self.conductivity is None:
            return None
        if isinstance(self.conductivity, list):
            return self.conductivity[0]
        return self.conductivity

    @property
    def loss_tangent_scalar(self) -> float | None:
        """Return scalar loss tangent, or first element of tensor."""
        if self.loss_tangent is None:
            return None
        if isinstance(self.loss_tangent, list):
            return self.loss_tangent[0]
        return self.loss_tangent


class MaterialProperties(BaseModel):
    """EM properties for a material.

    Supports both legacy scalar properties and frequency-dependent dispersion
    models with validity ranges. Tensor fields (permittivity, conductivity,
    loss_tangent, permeability) accept a scalar (isotropic) or list of 3
    floats (anisotropic).

    The static ``type`` classification (conductor/dielectric/semiconductor)
    has been removed in favor of frequency-aware ``ResolvedMaterial.behavior``
    which correctly handles materials that are conductive at RF but dielectric
    at optical wavelengths.

    Resolution priority:
    1. dispersion_models evaluated at target frequency (-> permittivity)
        2. permittivity scalar or tensor (directly specified)
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    conductivity: float | list[float] | None = Field(
        default=None, description="Conductivity S/m. Scalar or [sx, sy, sz]."
    )
    permittivity: float | list[float] | None = Field(
        default=None,
        description="Relative permittivity. Scalar (isotropic) or [ex, ey, ez].",
    )
    loss_tangent: float | list[float] | None = Field(
        default=None,
        description="Loss tangent. Scalar or [tx, ty, tz].",
    )
    permeability: float | list[float] | None = Field(
        default=None,
        description="Relative permeability. Scalar or [mx, my, mz].",
    )
    material_axes: list[list[float]] | None = None

    dispersion_models: list[DispersionModel] = Field(
        default_factory=list,
        description="Frequency-dependent dispersion models with validity ranges.",
    )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for YAML/JSON output."""
        d: dict[str, object] = {}
        if self.conductivity is not None:
            d["conductivity"] = self.conductivity
        if self.permittivity is not None:
            d["permittivity"] = self.permittivity
        if self.loss_tangent is not None:
            d["loss_tangent"] = self.loss_tangent
        if self.permeability is not None:
            d["permeability"] = self.permeability
        if self.material_axes is not None:
            d["material_axes"] = self.material_axes
        if self.dispersion_models:
            d["dispersion_models"] = [m.model_dump() for m in self.dispersion_models]
        return d

    def evaluate_at_wavelength(self, wavelength_um: float) -> ResolvedMaterial:
        """Evaluate material properties at a specific wavelength.

        Resolution strategy:
        1. Scan dispersion_models for one whose validity covers the wavelength.
        2. If no model covers it, use a model with unspecified validity (warn).
        3. If no dispersion models, fall back to the permittivity field directly.
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

        base = ResolvedMaterial(
            permittivity=self.permittivity,
            conductivity=self.conductivity,
            loss_tangent=self.loss_tangent,
            permeability=self.permeability,
            material_axes=self.material_axes,
        )

        if selected is not None:
            eps = selected.evaluate_permittivity(wavelength_um)
            base.permittivity = (
                self.permittivity if _is_tensor(self.permittivity) else eps
            )
            # When a dispersive model (Sellmeier/Lorentzian) covers the
            # target wavelength, the material behaves as a dielectric at
            # this frequency — any conductivity from the RF constant model
            # is physically incorrect and must be dropped.  For example,
            # silicon has sigma=2 S/m at RF but is a pure dielectric
            # (n~3.47) at optical wavelengths.
            if selected.type in ("sellmeier", "lorentzian"):
                base.conductivity = None
            base.model_type = selected.type
            base.model_source = selected.source
            base.within_validity = within_validity
            base.validity_note = validity_note
            return base

        if self.permittivity is not None:
            base.within_validity = True
            base.validity_note = "constant permittivity (no dispersion models)"
            return base

        base.within_validity = False
        cond = self.conductivity
        if isinstance(cond, list):
            cond = cond[0]
        if cond is not None and cond >= 1e4:
            base.validity_note = "conductive material (no permittivity needed)"
        else:
            base.validity_note = "no permittivity data available"
        return base

    def evaluate_at_frequency(self, freq_hz: float) -> ResolvedMaterial:
        """Evaluate material properties at a specific frequency in Hz."""
        wavelength_um = 299_792_458 / freq_hz * 1e6
        return self.evaluate_at_wavelength(wavelength_um)

    def index_variation(self, wavelength_um: float, bandwidth_um: float) -> float:
        """Compute fractional permittivity variation deps/eps across a bandwidth.

        Returns 0.0 if the material has no dispersive model or constant eps.
        """
        if not self.dispersion_models:
            return 0.0

        wl_min = wavelength_um - bandwidth_um / 2
        wl_max = wavelength_um + bandwidth_um / 2

        resolved_center = self.evaluate_at_wavelength(wavelength_um)
        eps_center = resolved_center.permittivity_scalar
        if eps_center is None or eps_center == 0:
            return 0.0

        try:
            eps_min = (
                self.evaluate_at_wavelength(wl_min).permittivity_scalar or eps_center
            )
            eps_max = (
                self.evaluate_at_wavelength(wl_max).permittivity_scalar or eps_center
            )
        except (ValueError, ZeroDivisionError):
            return 0.0

        delta_eps = max(abs(eps_max - eps_center), abs(eps_min - eps_center))
        return delta_eps / eps_center

    @classmethod
    def conductor(cls, conductivity: float = 5.8e7) -> MaterialProperties:
        """Create a conductor material with the given conductivity in S/m."""
        return cls(conductivity=conductivity)

    @classmethod
    def dielectric(
        cls, permittivity: float, loss_tangent: float = 0.0
    ) -> MaterialProperties:
        """Create a dielectric material with permittivity and loss tangent."""
        return cls(permittivity=permittivity, loss_tangent=loss_tangent)


MATERIALS_DB: dict[str, MaterialProperties] = {
    "aluminum": MaterialProperties(
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
        permittivity=4.1,
        loss_tangent=0.0,
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=0.6961663, C=0.0684043**2),
                    SellmeierTerm(B=0.4079426, C=0.1162414**2),
                    SellmeierTerm(B=0.8974794, C=9.896161**2),
                ],
                validity=ValidityRange(valid_wavelength=(0.21, 6.7)),
                source="Malitson 1965 (meep: fused_quartz)",
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
        permittivity=7.5,
        loss_tangent=0.001,
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=3.0249, C=0.1353406**2),
                    SellmeierTerm(B=40314, C=1239.842**2),
                ],
                validity=ValidityRange(valid_wavelength=(0.31, 5.504)),
                source="Luke et al. 2015 (meep: Si3N4_NIR)",
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
        permittivity=1.0,
        loss_tangent=0.0,
    ),
    "vacuum": MaterialProperties(
        permittivity=1.0,
        loss_tangent=0.0,
    ),
    "silicon": MaterialProperties(
        permittivity=11.9,
        conductivity=2.0,
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=10.6684293, C=0.301516485**2),
                    SellmeierTerm(B=0.0030434748, C=1.13475115**2),
                    SellmeierTerm(B=1.54133408, C=1104**2),
                ],
                validity=ValidityRange(valid_wavelength=(1.36, 11)),
                source="Salzberg & Villa 1957 (meep: Si)",
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
        permittivity=[9.3, 9.3, 11.5],
        loss_tangent=[3e-5, 3e-5, 8.6e-5],
        permeability=[0.99999975, 0.99999975, 0.99999979],
        material_axes=[[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=1.4313493, C=0.0726631**2),
                    SellmeierTerm(B=0.65054713, C=0.1193242**2),
                    SellmeierTerm(B=5.3414021, C=18.02825**2),
                ],
                validity=ValidityRange(valid_wavelength=(0.2, 5.0)),
                source="Malitson & Dodge 1972 (meep: Al2O3 ordinary)",
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
        permittivity=16.0,
        dispersion_models=[
            DispersionModel(
                type="sellmeier",
                sellmeier_terms=[
                    SellmeierTerm(B=6.7288, C=0.6641159**2),
                    SellmeierTerm(B=0.21307, C=62.210127**2),
                ],
                epsilon_inf=9.28156,
                validity=ValidityRange(valid_wavelength=(2.5, 12)),
                source="Icenogle 1979 (meep: Ge, Barnes & Piltch)",
            ),
        ],
    ),
    "tfln": MaterialProperties(
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
    "si": "silicon",
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


def _resolve_with_overlay(
    material_name: str,
    overlay: dict[str, MaterialProperties] | None = None,
) -> MaterialProperties | None:
    """Look up material properties with optional PDK overlay.

    Priority: overlay entry > built-in database.

    Args:
        material_name: Material name from PDK
        overlay: PDK overlay dict (merged into MATERIALS_DB for lookup)

    Returns:
        MaterialProperties if found, else None
    """
    if overlay and material_name in overlay:
        return overlay[material_name]

    if overlay:
        from gsim.common.stack.overlays import merge_overlay

        merged = merge_overlay(overlay)
        if material_name in merged:
            return merged[material_name]

    return get_material_properties(material_name)


def resolve_material_at_wavelength(
    material_name: str,
    wavelength_um: float,
    overrides: dict[str, MaterialProperties] | None = None,
    overlay: dict[str, MaterialProperties] | None = None,
) -> ResolvedMaterial | None:
    """Resolve a material's properties at a specific wavelength.

    Priority:
    1. User override (if provided)
    2. PDK overlay (if provided)
    3. Built-in database
    4. None + warning

    Args:
        material_name: Material name from PDK
        wavelength_um: Target wavelength in um
        overrides: User-supplied material property overrides
        overlay: PDK overlay dict (foundry-specific values)

    Returns:
        ResolvedMaterial with evaluated properties, or None if not found
    """
    overrides = overrides or {}

    if material_name in overrides:
        return overrides[material_name].evaluate_at_wavelength(wavelength_um)

    props = _resolve_with_overlay(material_name, overlay)
    if props is not None:
        resolved = props.evaluate_at_wavelength(wavelength_um)
        if (
            not resolved.within_validity
            and resolved.validity_note
            and resolved.behavior != "conductive"
        ):
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
    overlay: dict[str, MaterialProperties] | None = None,
) -> bool:
    """Determine if dispersion should be enabled for a material.

    Compares deps/eps across the bandwidth against a threshold.

    Args:
        material_name: Material name
        wavelength_um: Center wavelength in um
        bandwidth_um: Source bandwidth in um
        threshold: Fractional index variation threshold (default 0.5%)
        overrides: User-supplied material property overrides
        overlay: PDK overlay dict (foundry-specific values)

    Returns:
        True if dispersion should be enabled
    """
    overrides = overrides or {}

    if material_name in overrides:
        return (
            overrides[material_name].index_variation(wavelength_um, bandwidth_um)
            > threshold
        )

    props = _resolve_with_overlay(material_name, overlay)
    if props is not None:
        return props.index_variation(wavelength_um, bandwidth_um) > threshold

    return False
