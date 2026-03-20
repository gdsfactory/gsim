"""PEC block configuration for Palace simulations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PECBlockConfig(BaseModel):
    """Configuration for a PEC (Perfect Electric Conductor) block.

    A PEC block is a user-drawn polygon on a GDS layer that gets extruded
    between two stack layers and treated as a PEC boundary. This is the
    standard HFSS practice for connecting ground planes across metal layers
    at port boundaries in terminal-driven simulations.

    Attributes:
        gds_layer: GDS layer tuple where the PEC polygon is drawn.
        from_layer: Stack layer name — extrusion starts at this layer's zmin.
        to_layer: Stack layer name — extrusion ends at this layer's zmax.
    """

    model_config = ConfigDict(validate_assignment=True)

    gds_layer: tuple[int, int]
    from_layer: str
    to_layer: str
