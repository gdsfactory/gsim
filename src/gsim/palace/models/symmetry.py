"""Symmetry-plane configuration for Palace simulations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

_GRID_TOL = 1e-6  # in database units (nm)


class SymmetryPlaneConfig(BaseModel):
    """A PEC or PMC symmetry plane that cuts the model in half.

    Attributes:
        axis: Axis normal to the plane (``"x"`` or ``"y"``).
        position: Plane position along ``axis`` in um, on the 1 nm grid.
        kind: ``"pmc"`` (even/common mode) or ``"pec"`` (odd/differential mode).
        keep: Which side of the plane is simulated.
        verify_symmetry: Check that the layout is mirror-symmetric.
    """

    model_config = ConfigDict(validate_assignment=True)

    axis: Literal["x", "y"] = "y"
    position: float = 0.0
    kind: Literal["pec", "pmc"] = "pmc"
    keep: Literal["positive", "negative"] = "positive"
    verify_symmetry: bool = True

    @field_validator("position")
    @classmethod
    def _position_on_grid(cls, value: float) -> float:
        """Require the position to lie on the 1 nm database grid."""
        nm = value * 1000.0
        if abs(nm - round(nm)) > _GRID_TOL:
            raise ValueError(
                f"Symmetry plane position {value} um is not on the 1 nm grid"
            )
        return value
