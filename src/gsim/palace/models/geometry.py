"""Geometry configuration models for Palace simulations.

This module contains Pydantic models for geometry-related settings.
Note: The actual gdsfactory Component is stored directly on the simulation
classes since it's not serializable with Pydantic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GeometryConfig(BaseModel):
    """Configuration for geometry settings.

    This model stores metadata about the component being simulated.
    The actual Component object is stored separately on simulation classes.

    Attributes:
        component_name: Name of the component being simulated
        bounds: Bounding box as (xmin, ymin, xmax, ymax)
    """

    model_config = ConfigDict(validate_assignment=True)

    component_name: str | None = None
    bounds: tuple[float, float, float, float] | None = Field(
        default=None, description="Bounding box (xmin, ymin, xmax, ymax)"
    )


__all__ = [
    "GeometryConfig",
]
