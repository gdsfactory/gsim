"""User-facing configuration objects of a thermal heater solve.

A :class:`gsim.thermal.Simulation` is configured by physical concern, the
way :class:`gsim.tcad.Simulation` and :class:`gsim.fdtd.Simulation` are:
the geometry and its stack, the domain (the cut, its Window and the
Modelled depth), the heater, the waveguides of interest, per-run material
overrides, the boundary conditions, and the solver. Each concern is a
validated model; calling it applies several settings at once, and single
attributes can be assigned.

Lengths are in um and field names carry their unit. Material properties
and the heat transfer coefficient are SI, as the stack's material records
are.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from gsim.common.config import MutableModel, NamedModels
from gsim.common.cross_section import parse_plane_spec
from gsim.common.stack.materials import MaterialProperties
from gsim.common.validation import AscendingInterval

__all__ = [
    "Boundary",
    "Domain",
    "Geometry",
    "Heater",
    "MaterialOverride",
    "Materials",
    "Solver",
    "Waveguide",
    "Waveguides",
]

#: Marks an argument left out, where None is a value of its own.
_UNSET: Any = object()

# Per precision: mesh-size factor on the mesher's defaults, transient steps.
_PRECISION: dict[str, tuple[float, int]] = {
    "fast": (1.0, 200),
    "default": (0.6, 400),
    "accurate": (0.35, 1000),
}

#: The Solver fields the steady solve and the meshes depend on; the others
#: set only the transient's time stepping.
STEADY_FIELDS = ("precision", "mesh_size_um", "degree")


class Geometry(MutableModel):
    """The component and the layer stack it is extruded with.

    Attributes:
        component: The gdsfactory component: the heater and the waveguides.
        stack: A gsim ``LayerStack`` (layer heights, materials and slabs),
            e.g. from :func:`gsim.common.stack.get_stack`.
    """

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", arbitrary_types_allowed=True
    )

    component: Any | None = None
    stack: Any | None = None

    @field_validator("stack")
    @classmethod
    def validate_stack(cls, value: Any) -> Any:
        """Accept a LayerStack or None."""
        from gsim.common.stack import LayerStack

        if value is None or isinstance(value, LayerStack):
            return value
        raise ValueError(
            f"stack must be a gsim LayerStack; got {type(value).__name__}. "
            "Build one with gsim.common.stack.get_stack()."
        )

    def __call__(  # ty: ignore[invalid-method-override]
        self, component: Any | None = None, *, stack: Any = _UNSET
    ) -> Self:
        """Set the component and, when given, its stack."""
        updates: dict[str, Any] = {}
        if component is not None:
            updates["component"] = component
        if stack is not _UNSET:
            updates["stack"] = stack
        return super().__call__(**updates)


class Domain(MutableModel):
    """The cut and the extent of the thermal cross-section.

    Attributes:
        plane: The cut, ``"x=<um>"`` or ``"y=<um>"``, across the heater
            and the waveguides.
        window: In-plane ``(min, max)`` extent of the thermal Window in
            um. ``None`` takes the drawn cut plus 20 um each side. The
            Window's sides are adiabatic by default, so make it wide next
            to the oxide thickness (a half-width of 5x is ample).
        depth_um: Modelled depth of the substrate below the stack in um,
            the bottom of which is the heat sink. ``None`` takes the
            stack's substrate thickness (100 um without one).
    """

    plane: str | None = None
    window: AscendingInterval | None = None
    depth_um: PositiveFloat | None = None

    @field_validator("plane")
    @classmethod
    def validate_plane(cls, value: str | None) -> str | None:
        """Require a vertical ``x=`` or ``y=`` cut."""
        if value is None:
            return None
        axis, _ = parse_plane_spec(value)
        if axis not in {"x", "y"}:
            raise ValueError(
                "A heater cross-section is vertical: use an 'x=<um>' or "
                f"'y=<um>' plane, not {value!r}."
            )
        return value


class Heater(MutableModel):
    """The heater: the layers whose drawn conductors carry its current.

    The current runs along the waveguide, so every conductor of the heater
    in the cut sees the same field, and they share the power in proportion
    to their conductance.

    Attributes:
        layers: Stack layer name(s) of the heater conductors; a single
            name is taken as a one-layer heater.
        length_um: Heater length along the waveguide in um. It sets the
            resistance, V_pi and the phase at a given power; P_pi does not
            depend on it in 2D.
    """

    layers: tuple[str, ...] = ()
    length_um: PositiveFloat | None = None

    @field_validator("layers", mode="before")
    @classmethod
    def coerce_layers(cls, value: Any) -> Any:
        """Accept one layer name as well as several."""
        return (value,) if isinstance(value, str) else value

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse empty and repeated layer names."""
        if any(not name for name in value):
            raise ValueError("heater layer names must not be empty")
        if len(set(value)) != len(value):
            raise ValueError(f"heater layers are repeated: {list(value)}")
        return value


class Waveguide(BaseModel):
    """A waveguide of interest, found in the cut by its in-plane position.

    Attributes:
        name: Name in the results (``result.crosstalk[name]``).
        position_um: In-plane coordinate in um (``y`` for an ``"x=..."``
            cut) inside the waveguide's drawn rectangle.
        layer: Stack layer of the waveguide core. Needed only when several
            non-heater layers are drawn at ``position_um``.
        window: In-plane ``(min, max)`` extent of the optical Window in
            um. ``None`` takes the core's drawn rectangle plus 1.5 um each
            side (and 1 um above and below), clipped to the thermal Window.
        polarization: ``"TE"`` or ``"TM"`` fundamental Mode.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    position_um: float
    layer: str | None = None
    window: AscendingInterval | None = None
    polarization: Literal["TE", "TM"] = "TE"


class Waveguides(NamedModels[Waveguide]):
    """The waveguides of interest, by name.

    The first one added is the phase shifter; every other one is a
    crosstalk victim.
    """

    _kind = "waveguide"
    _model = Waveguide

    def add(
        self,
        *,
        name: str,
        position_um: float,
        layer: str | None = None,
        window: tuple[float, float] | None = None,
        polarization: Literal["TE", "TM"] = "TE",
    ) -> Waveguide:
        """Add a waveguide at an in-plane position of the cut."""
        return self._insert(
            Waveguide(
                name=name,
                position_um=position_um,
                layer=layer,
                window=window,
                polarization=polarization,
            )
        )

    @property
    def phase_shifter(self) -> Waveguide | None:
        """The first waveguide added, the one the heater is for."""
        return next(iter(self), None)


class MaterialOverride(BaseModel):
    """Properties of one material replaced for this simulation only.

    Each field left ``None`` keeps the stack's value, else the database's.

    Attributes:
        thermal_conductivity: k in W/m/K.
        specific_heat: c_p in J/kg/K.
        mass_density: rho in kg/m^3.
        dn_dT: Thermo-optic coefficient in 1/K.
        conductivity: Electrical conductivity in S/m (a heater conductor).
        permittivity: Relative permittivity at the solve wavelength; it
            replaces the material's dispersion models.
        loss_tangent: Optical loss tangent at the solve wavelength.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    thermal_conductivity: float | None = None
    specific_heat: float | None = None
    mass_density: float | None = None
    dn_dT: float | None = None  # noqa: N815
    conductivity: float | None = None
    permittivity: float | None = None
    loss_tangent: float | None = None

    @model_validator(mode="after")
    def validate_properties(self) -> Self:
        """Hold the values to the stack's own material constraints."""
        MaterialProperties.model_validate(self.properties())
        return self

    def properties(self) -> dict[str, float]:
        """The properties set, as a material record takes them."""
        return self.model_dump(exclude_none=True)


class Materials(MutableModel):
    """Per-run material overrides, keyed by material name.

    An override changes every layer made of that material: to give one
    layer its own values, give it its own material name in the stack.
    Names match the stack's case-insensitively.

    Attributes:
        overrides: Material name to :class:`MaterialOverride` (or a
            mapping of its fields).
    """

    overrides: dict[str, MaterialOverride] = Field(default_factory=dict)

    @field_validator("overrides", mode="before")
    @classmethod
    def coerce_overrides(cls, value: Any) -> Any:
        """Allow ``None`` for no overrides."""
        return {} if value is None else value

    @field_validator("overrides")
    @classmethod
    def validate_names(
        cls, value: dict[str, MaterialOverride]
    ) -> dict[str, MaterialOverride]:
        """Refuse two names that differ only by case."""
        folded = [name.strip().lower() for name in value]
        clashes = sorted({n for n in folded if folded.count(n) > 1})
        if clashes:
            raise ValueError(
                f"material overrides {clashes} are given under several spellings"
            )
        return value

    def _key(self, name: str) -> str:
        """The stored spelling of ``name``, or ``name`` itself."""
        folded = name.strip().lower()
        return next((k for k in self.overrides if k.strip().lower() == folded), name)

    def __getitem__(self, name: str) -> MaterialOverride:
        """Return the override of one material, matched case-insensitively."""
        return self.overrides[self._key(name)]

    def __setitem__(
        self, name: str, value: MaterialOverride | Mapping[str, Any]
    ) -> None:
        """Validate and store (or replace) the override of one material."""
        updated = dict(self.overrides)
        updated.pop(self._key(name), None)
        updated[name] = MaterialOverride.model_validate(value)
        self(overrides=updated)

    def __delitem__(self, name: str) -> None:
        """Drop the override of one material."""
        updated = dict(self.overrides)
        del updated[self._key(name)]
        self(overrides=updated)

    def get(self, name: str) -> MaterialOverride | None:
        """The override of one material, or None."""
        key = self._key(name)
        return self.overrides.get(key)

    def as_records(self) -> dict[str, dict[str, float]]:
        """Every override, as ``thermal_properties`` takes them."""
        return {name: o.properties() for name, o in self.overrides.items()}


class Boundary(MutableModel):
    """The top and side boundary conditions.

    The bottom of the Modelled depth is always the heat sink, at ambient.
    A convective edge loses ``h dT`` to ambient; free convection in air
    (h ~ 10 W/m^2/K) is negligible next to the substrate path, hence the
    adiabatic default.

    Attributes:
        top: Condition on the top of the stack.
        sides: Condition on the two sides of the thermal Window.
        heat_transfer_coefficient: h in W/m^2/K, shared by every
            convective edge.
    """

    top: Literal["adiabatic", "convective"] = "adiabatic"
    sides: Literal["adiabatic", "convective"] = "adiabatic"
    heat_transfer_coefficient: NonNegativeFloat = 10.0

    def convective_edges(self) -> dict[str, float]:
        """H on every convective edge of the Window, by edge name."""
        edges: dict[str, float] = {}
        h = float(self.heat_transfer_coefficient)
        if self.top == "convective":
            edges["top"] = h
        if self.sides == "convective":
            edges["left"] = h
            edges["right"] = h
        return edges


class Solver(MutableModel):
    """Numerical settings; the defaults converge for a strip heater.

    Attributes:
        precision: ``"fast"``, ``"default"`` or ``"accurate"``: scales the
            mesh size everywhere (x1, x0.6, x0.35 of the mesher's
            defaults) and sets the transient's time steps (200, 400,
            1000).
        mesh_size_um: Maximum element size in um per Region name (a stack
            layer, or a slab by its material), overriding the precision on
            both the thermal and the optical meshes.
        degree: Lagrange element degree of the heat solve: 1 (linear), 2
            (quadratic) or 3. A higher degree reaches a given accuracy on a
            coarser mesh; the optical Mode solve is unaffected.
        time_steps: Number of constant transient time steps, overriding
            the precision.
        theta: Time scheme of the transient: 1.0 backward Euler (robust
            for a step drive), 0.5 Crank-Nicolson.
    """

    precision: Literal["fast", "default", "accurate"] = "default"
    mesh_size_um: dict[str, PositiveFloat] = Field(default_factory=dict)
    degree: Literal[1, 2, 3] = 2
    time_steps: PositiveInt | None = None
    theta: float = Field(default=1.0, ge=0.5, le=1.0)

    @property
    def size_factor(self) -> float:
        """Factor on the mesher's default element sizes."""
        return _PRECISION[self.precision][0]

    @property
    def steps(self) -> int:
        """Number of transient time steps."""
        if self.time_steps is not None:
            return int(self.time_steps)
        return _PRECISION[self.precision][1]
