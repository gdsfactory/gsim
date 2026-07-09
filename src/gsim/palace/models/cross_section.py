"""Cross-section specification models for Palace 2D mode simulations."""

from __future__ import annotations

import math
import re
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

_PLANE_SPEC_RE = re.compile(
    r"^\s*([xXyYzZ])\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


class CrossSectionPlaneConfig(BaseModel):
    """Axis-aligned cross-section plane for 2D mode extraction.

    Attributes:
        axis: Plane normal axis ("x", "y", or "z").
        value: Plane coordinate in microns.
    """

    model_config = ConfigDict(validate_assignment=True)

    axis: Literal["x", "y", "z"]
    value: float = Field(description="Plane coordinate in um")

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        """Ensure the plane coordinate is finite."""
        if not math.isfinite(self.value):
            raise ValueError("cross-section value must be finite")
        return self

    @classmethod
    def from_spec(cls, spec: str) -> Self:
        """Parse a string specification like ``x=0`` or ``y=100``."""
        match = _PLANE_SPEC_RE.match(spec)
        if match is None:
            raise ValueError(
                "Invalid plane spec. Use 'x=<value>', 'y=<value>', or 'z=<value>'"
            )
        axis = cast(Literal["x", "y", "z"], match.group(1).lower())
        value = float(match.group(2))
        return cls(axis=axis, value=value)

    @property
    def spec(self) -> str:
        """Return the normalized string representation (e.g., ``x=0.0``)."""
        return f"{self.axis}={self.value}"


__all__ = ["CrossSectionPlaneConfig"]
