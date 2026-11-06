"""User-facing configuration objects for the DEVSIM charge-transport solve.

A :class:`gsim.tcad.Simulation` is configured by physical concern, the way
:class:`gsim.fdtd.Simulation` is: the geometry and its stack, the meshed
domain, the materials, the doping, the contacts and interfaces, and the
solver. Each concern is a validated model; calling it applies several
settings at once, and single attributes can be assigned.

Lengths are in um, concentrations in cm^-3, frequencies in Hz and
voltages in V; field names carry their unit.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    TypeAdapter,
    field_validator,
    model_validator,
)

from gsim.common.carriers import MobilityModel
from gsim.common.config import MutableModel
from gsim.common.cross_section import parse_plane_spec
from gsim.common.validation import AscendingInterval
from gsim.tcad.doping import DopingProfile

#: A refinement box ``(h_min, h_max, z_min, z_max, size)`` in um, held to
#: ``size`` over its whole area.
RefinementBox = tuple[float, float, float, float, float]

_DOPING_PROFILE: TypeAdapter[DopingProfile] = TypeAdapter(DopingProfile)

#: Marks an argument left out, where None is a value of its own.
_UNSET: Any = object()


class Geometry(MutableModel):
    """The component and the layer stack it is extruded with.

    Attributes:
        component: The gdsfactory component the cross-section is cut from.
        stack: A gsim ``LayerStack``, the path of a stack YAML file, or
            ``None`` to extract the stack of the active PDK.
    """

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", arbitrary_types_allowed=True
    )

    component: Any | None = None
    stack: Any | None = None

    @field_validator("stack")
    @classmethod
    def validate_stack(cls, value: Any) -> Any:
        """Accept a LayerStack, a YAML path, or None."""
        from gsim.common.stack import LayerStack

        if value is None or isinstance(value, (LayerStack, str, Path)):
            return value
        raise ValueError(
            "stack must be a gsim LayerStack, a stack YAML path, or None "
            f"for the active PDK; got {type(value).__name__}"
        )

    def __call__(  # ty: ignore[invalid-method-override]
        self, component: Any | None = None, *, stack: Any = _UNSET
    ) -> Self:
        """Set the component and, when given, its stack (None: active PDK)."""
        updates: dict[str, Any] = {}
        if component is not None:
            updates["component"] = component
        if stack is not _UNSET:
            updates["stack"] = stack
        return super().__call__(**updates)


class Domain(MutableModel):
    """The cross-section plane and the extent of the meshed 2D domain.

    Attributes:
        plane: Cross-section plane, ``"x=<um>"`` or ``"y=<um>"``.
        window: In-plane ``(min, max)`` clip of the domain in um (y for an
            x-plane, x for a y-plane); ``None`` keeps the full extent.
        window_z: Vertical ``(min, max)`` clip of the domain in um.
        margin_x: Airbox margin along x in um.
        margin_y: Airbox margin along y in um.
        z_above: Airbox extension above the stack top in um.
        z_below: Airbox extension below the stack bottom in um.

    The four airbox fields default to ``None``: the mesh preset decides.
    Setting any of them configures the airbox, the unset ones at 0 um.
    """

    plane: str | None = None
    window: AscendingInterval | None = None
    window_z: AscendingInterval | None = None
    margin_x: float | None = Field(default=None, ge=0)
    margin_y: float | None = Field(default=None, ge=0)
    z_above: float | None = Field(default=None, ge=0)
    z_below: float | None = Field(default=None, ge=0)

    @field_validator("plane")
    @classmethod
    def validate_plane(cls, value: str | None) -> str | None:
        """Require an ``x=`` or ``y=`` plane spec."""
        if value is None:
            return None
        axis, _ = parse_plane_spec(value)
        if axis == "z":
            raise ValueError(
                "A charge-transport cross-section is cut at 'x=<um>' or "
                "'y=<um>', not along z."
            )
        return value


class Insulator(BaseModel):
    """An insulating Region taking part in the electrostatic solve only.

    Attributes:
        region: Mesh region (volume group) name, e.g. ``"sio2"``.
        relative_permittivity: Static relative permittivity of the Region.
    """

    model_config = ConfigDict(frozen=True)

    region: str
    relative_permittivity: float = Field(gt=0.0)


class Materials(MutableModel):
    """Semiconductor parameters and the insulators joining the solve.

    Attributes:
        temperature_k: Lattice temperature (K) of the silicon parameter set.
        semiconductor: DEVSIM material name given to the doped Regions.
        mobility: Low-field mobility against the total doping, evaluated
            node by node. DEVSIM's own silicon defaults are two constants
            whatever the doping, which misstates the series resistance of
            a slab doped from 1e17 to 1e20 cm^-3.
        insulators: Insulating Regions solved for the potential alone,
            Region name to static relative permittivity. Empty by default,
            which keeps the electrostatics inside the doped semiconductor;
            a declared insulator lets the field fringe around the Junction
            through it. An :class:`Insulator` or a list of them is also
            accepted.
    """

    temperature_k: float = Field(default=300.0, gt=0.0)
    semiconductor: str = Field(default="Silicon", min_length=1)
    mobility: MobilityModel = Field(default_factory=MobilityModel.masetti_silicon)
    insulators: dict[str, PositiveFloat] = Field(default_factory=dict)

    @field_validator("insulators", mode="before")
    @classmethod
    def coerce_insulators(cls, value: Any) -> Any:
        """Allow ``{"sio2": 3.9}``, Insulator models, or a list of them."""
        if value is None:
            return {}
        if isinstance(value, Insulator):
            value = [value]
        if isinstance(value, Mapping):
            return {
                name: item.relative_permittivity
                if isinstance(item, Insulator)
                else item
                for name, item in value.items()
            }
        items = list(value)
        if not all(isinstance(item, Insulator) for item in items):
            raise ValueError(
                "insulators must be a mapping of Region to relative "
                "permittivity, or a list of Insulator"
            )
        return {item.region: item.relative_permittivity for item in items}


#: Solver fields that shape the mesh; a change makes the mesh stale.
MESH_FIELDS: tuple[str, ...] = (
    "mesh_preset",
    "refined_mesh_size_um",
    "max_mesh_size_um",
    "refinement_boxes",
)


class Solver(MutableModel):
    """Mesh resolution, Newton convergence and small-signal settings.

    Attributes:
        mesh_preset: Mesh quality preset of the shared native-2D pipeline;
            ``None`` uses the pipeline's default.
        refined_mesh_size_um: Mesh size near conductors and regions marked
            fine (um), overriding the preset.
        max_mesh_size_um: Largest element size (um), overriding the preset.
        refinement_boxes: Boxes ``(h_min, h_max, z_min, z_max, size)`` in
            um held to ``size`` over their whole area, e.g. where a
            depletion edge moves.
        absolute_error: Newton absolute error target of ``devsim.solve``.
        relative_error: Newton relative error target of ``devsim.solve``.
            With extended precision the 2D drift-diffusion Newton reaches
            1e-12 on the test diode.
        max_iterations: Newton iteration cap.
        bias_step_v: Step (V) the swept contact is ramped in between biases.
        extended_precision: Use DEVSIM's 128-bit extended-precision
            assembly. Keep it on: in double precision the Newton on the
            test diode does not converge even to the default 1e-6.
        small_signal_freq_hz: Frequency (Hz) of the quasi-static AC solve
            that extracts C(V).
        junction_freq_hz: Frequency (Hz) of the AC solve whose admittance
            the series-RC junction fit reads: high enough that the
            capacitive current dominates the DC leakage, far below the
            branch's own RC rolloff.
        claim_devsim_namespace: Claim DEVSIM's process-global namespace at
            device setup: drop the circuit and the devices gsim created
            earlier. Foreign DEVSIM devices are never touched. Turn it off
            only when an earlier charge-transport device must stay alive.
    """

    mesh_preset: Literal["coarse", "default", "fine"] | None = None
    refined_mesh_size_um: PositiveFloat | None = None
    max_mesh_size_um: PositiveFloat | None = None
    refinement_boxes: list[RefinementBox] | None = None

    absolute_error: PositiveFloat = 1e10
    relative_error: PositiveFloat = 1e-6
    max_iterations: int = Field(default=100, gt=0)
    bias_step_v: PositiveFloat = 0.1
    extended_precision: bool = True

    small_signal_freq_hz: PositiveFloat = 1.0
    junction_freq_hz: PositiveFloat = 1e9

    claim_devsim_namespace: bool = True


class Doping:
    """The doping profiles, each naming the mesh Region it applies to."""

    def __init__(self, profiles: Iterable[DopingProfile] | None = None) -> None:
        """Validate an optional initial list of doping profiles."""
        self._items: list[DopingProfile] = []
        for profile in profiles or []:
            self.add(profile)

    def add(self, profile: DopingProfile | Mapping[str, Any]) -> DopingProfile:
        """Add a profile (see :mod:`gsim.tcad.doping`) and return it."""
        validated = _DOPING_PROFILE.validate_python(profile)
        self._items.append(validated)
        return validated

    def remove(self, profile: DopingProfile) -> None:
        """Remove one profile."""
        self._items.remove(profile)

    def clear(self) -> None:
        """Remove every profile."""
        self._items.clear()

    def regions(self) -> list[str]:
        """The doped Regions, in the order the profiles first name them."""
        return list(dict.fromkeys(profile.region for profile in self._items))

    def __iter__(self) -> Iterator[DopingProfile]:
        """Iterate over the profiles in insertion order."""
        return iter(list(self._items))

    def __len__(self) -> int:
        """Return the number of profiles."""
        return len(self._items)

    def __repr__(self) -> str:
        """Show the profiles."""
        return f"Doping({self._items!r})"


class Contact(BaseModel):
    """Where an electrode meets a doped Region: an ohmic contact.

    The curve the two Regions share becomes a named line group on the
    mesh, and DEVSIM drives the contact's bias there.

    Attributes:
        name: Contact name, also its mesh group name (e.g. ``"anode"``).
        semiconductor: The doped Region the contact injects into.
        electrode: The Region on the other side: the metal, or any Region
            that is neither doped nor declared insulating.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    semiconductor: str = Field(min_length=1)
    electrode: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sides(self) -> Self:
        """Require two different Regions."""
        if self.semiconductor == self.electrode:
            raise ValueError(
                f"Contact '{self.name}' joins '{self.semiconductor}' to itself."
            )
        return self


class Interface(BaseModel):
    """Where two doped Regions meet: potential and carriers are continuous.

    An Interface carries no terminal voltage; that is what separates it
    from a :class:`Contact`.

    Attributes:
        name: Interface name, also its mesh group name (e.g. ``"junction"``).
        region_a: One doped Region.
        region_b: The other doped Region.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    region_a: str = Field(min_length=1)
    region_b: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sides(self) -> Self:
        """Require two different Regions."""
        if self.region_a == self.region_b:
            raise ValueError(
                f"Interface '{self.name}' joins '{self.region_a}' to itself."
            )
        return self


class _Named[T: BaseModel]:
    """Insertion-ordered collection of models with a unique ``name``."""

    _kind: str
    _model: type[T]

    def __init__(self, items: Iterable[T | Mapping[str, Any]] | None = None) -> None:
        """Validate an optional initial list."""
        self._items: dict[str, T] = {}
        for item in items or []:
            self._insert(self._model.model_validate(item))

    def _insert(self, item: T) -> T:
        """Store an item, requiring a unique name."""
        name = item.name  # ty: ignore[unresolved-attribute]
        if name in self._items:
            raise ValueError(f"{self._kind} {name!r} already exists")
        self._items[name] = item
        return item

    def remove(self, name: str) -> T:
        """Remove and return one item by name."""
        return self._items.pop(name)

    def clear(self) -> None:
        """Remove every item."""
        self._items.clear()

    def __getitem__(self, name: str) -> T:
        """Return one item by name."""
        return self._items[name]

    def __contains__(self, name: object) -> bool:
        """Whether an item of that name is declared."""
        return name in self._items

    def __iter__(self) -> Iterator[T]:
        """Iterate over the items in insertion order."""
        return iter(list(self._items.values()))

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._items)

    def __repr__(self) -> str:
        """Show the items."""
        return f"{type(self).__name__}({list(self._items.values())!r})"


class Contacts(_Named[Contact]):
    """The declared contacts, by name."""

    _kind = "contact"
    _model = Contact

    def add(self, *, name: str, semiconductor: str, electrode: str) -> Contact:
        """Declare a contact between a doped Region and its electrode."""
        return self._insert(
            Contact(name=name, semiconductor=semiconductor, electrode=electrode)
        )


class Interfaces(_Named[Interface]):
    """The declared interfaces between doped Regions, by name."""

    _kind = "interface"
    _model = Interface

    def add(self, *, name: str, region_a: str, region_b: str) -> Interface:
        """Declare an interface between two doped Regions."""
        return self._insert(Interface(name=name, region_a=region_a, region_b=region_b))


class TCADMesh(BaseModel):
    """The mesh a :class:`gsim.tcad.Simulation` solves on.

    Attributes:
        mesh_path: The shared native-2D mesh (um), the one Palace
            BoundaryMode sees.
        devsim_mesh_path: Its copy with coordinates in cm, loaded by DEVSIM.
        groups: Physical groups of the mesh (volumes, contact lines,
            interface lines, shared curves).
    """

    model_config = ConfigDict(frozen=True)

    mesh_path: Path
    devsim_mesh_path: Path
    groups: dict[str, Any]


__all__ = [
    "MESH_FIELDS",
    "Contact",
    "Contacts",
    "Domain",
    "Doping",
    "Geometry",
    "Insulator",
    "Interface",
    "Interfaces",
    "Materials",
    "RefinementBox",
    "Solver",
    "TCADMesh",
]
