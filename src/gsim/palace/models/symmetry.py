"""Symmetry-plane configuration for Palace simulations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SymmetryPlaneConfig(BaseModel):
    """A PEC or PMC symmetry plane that cuts the model in half.

    Attributes:
        axis: Axis normal to the plane (``"x"`` or ``"y"``).
        position: Plane position along ``axis`` in um. Alignment to the owning
            layout's database grid is checked during geometry extraction.
        kind: ``"pmc"`` (even/common mode) or ``"pec"`` (odd/differential mode).
        keep: Which side of the plane is simulated.
        verify_symmetry: Check that the layout is mirror-symmetric.
    """

    model_config = ConfigDict(validate_assignment=True)

    axis: Literal["x", "y"] = "y"
    position: float = Field(default=0.0, allow_inf_nan=False)
    kind: Literal["pec", "pmc"] = "pmc"
    keep: Literal["positive", "negative"] = "positive"
    verify_symmetry: bool = True
