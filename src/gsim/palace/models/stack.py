"""Material configuration model for Palace simulations.

This module contains the MaterialConfig Pydantic model for material property overrides
in simulations. Layer and stack configuration is handled by gsim.common.stack.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field


class MaterialConfig(BaseModel):
    """EM properties for a material.

    Used for material property overrides in simulation classes.

    Attributes:
        type: Material type (conductor, dielectric, or semiconductor)
        conductivity: Conductivity in S/m (for conductors)
        permittivity: Relative permittivity (for dielectrics)
        loss_tangent: Dielectric loss tangent
    """

    model_config = ConfigDict(validate_assignment=True)

    type: Literal["conductor", "dielectric", "semiconductor"]
    conductivity: float | None = Field(default=None, ge=0)
    permittivity: float | None = Field(default=None, ge=1.0)
    loss_tangent: float | None = Field(default=None, ge=0, le=1)

    @classmethod
    def conductor(cls, conductivity: float = 5.8e7) -> Self:
        """Create a conductor material."""
        return cls(type="conductor", conductivity=conductivity)

    @classmethod
    def dielectric(cls, permittivity: float, loss_tangent: float = 0.0) -> Self:
        """Create a dielectric material."""
        return cls(
            type="dielectric", permittivity=permittivity, loss_tangent=loss_tangent
        )

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


__all__ = [
    "MaterialConfig",
]
