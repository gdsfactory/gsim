"""Cross-section specification models for Palace 2D mode simulations."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gsim.common.cross_section import parse_plane_spec
from gsim.common.validation import AscendingInterval


class _LayerPairSpec(BaseModel):
    """A named dim-1 physical group declared as a pair of layers.

    What a contact and an interface have in common and nothing more: a
    name to tag the shared curves between two layers' meshed regions
    with, and the two distinct layers to find them between. The two
    concepts stay apart — an Interface carries no terminal voltage,
    which is the whole difference — so this base is private and neither
    subclass is a substitute for the other.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(min_length=1, description="Physical-group name")
    layer_a: str = Field(min_length=1, description="First layer of the interface")
    layer_b: str = Field(min_length=1, description="Second layer of the interface")

    @model_validator(mode="after")
    def validate_layers_differ(self) -> Self:
        """A layer pair needs two distinct layers."""
        if self.layer_a == self.layer_b:
            raise ValueError("layer_a and layer_b must differ")
        return self


class ContactSpec(_LayerPairSpec):
    """Named contact between two layers on the native-2D cross-section mesh.

    The shared interface curves between the two layers' meshed regions are
    tagged as a dim-1 physical group named ``name``, so charge-transport
    solvers (DEVSIM ``add_gmsh_contact``) can bind boundary conditions to it
    by name.

    Attributes:
        name: Physical-group name of the contact (e.g. ``"anode"``).
        layer_a: First layer of the interface (e.g. the electrode layer).
        layer_b: Second layer of the interface (e.g. the doped semiconductor).
    """

    name: str = Field(min_length=1, description="Contact physical-group name")


class InterfaceSpec(_LayerPairSpec):
    """Named interface between two semiconductor layers on the native-2D mesh.

    The shared curves between the two layers' meshed regions are tagged as
    a dim-1 physical group named ``name``, so a charge-transport solver can
    bind continuity across it (DEVSIM ``add_gmsh_interface``) by name. An
    Interface carries no terminal voltage — that is what separates it from
    a :class:`ContactSpec`, and why the two are declared apart.

    Attributes:
        name: Physical-group name of the interface (e.g. ``"junction"``).
        layer_a: First semiconductor layer.
        layer_b: Second semiconductor layer.
    """

    name: str = Field(min_length=1, description="Interface physical-group name")


class CrossSectionPlaneConfig(BaseModel):
    """Axis-aligned cross-section plane for 2D mode extraction.

    An optional window clips the meshed 2D domain to a sub-region of the
    component cross-section instead of the full bounding box plus margins.
    This lets one component feed differently sized per-solver domains (full
    extent for RF, a small box around the rib for optics, the doped slab for
    charge transport).

    Attributes:
        axis: Plane normal axis ("x", "y", or "z").
        value: Plane coordinate in microns.
        window: Optional in-plane interval ``(min, max)`` in um clipping the
            transverse extent (y for an x-plane, x for a y-plane).
        window_z: Optional z interval ``(min, max)`` in um clipping the
            vertical extent.
    """

    model_config = ConfigDict(validate_assignment=True)

    axis: Literal["x", "y", "z"]
    value: float = Field(description="Plane coordinate in um")
    window: AscendingInterval | None = Field(
        default=None, description="In-plane clip interval (min, max) in um"
    )
    window_z: AscendingInterval | None = Field(
        default=None, description="Vertical clip interval (min, max) in um"
    )

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        """Ensure the plane coordinate is finite."""
        if not math.isfinite(self.value):
            raise ValueError("cross-section value must be finite")
        return self

    @classmethod
    def from_spec(cls, spec: str) -> Self:
        """Parse a string specification like ``x=0`` or ``y=100``."""
        axis, value = parse_plane_spec(spec)
        return cls(axis=axis, value=value)

    @property
    def spec(self) -> str:
        """Return the normalized string representation (e.g., ``x=0.0``)."""
        return f"{self.axis}={self.value}"


__all__ = ["ContactSpec", "CrossSectionPlaneConfig", "InterfaceSpec"]
