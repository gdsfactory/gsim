"""What every meshing Stage asks of the mesh pipeline, in one place.

The charge, optical and RF Stages all mesh a Cross-section, and all three
want the same background region and much the same element sizes. Kept as
three literals they drifted apart silently; kept here, a Stage overrides
only what its own physics needs — the charge Stage its finer elements —
and a user still overrides anything, per Stage, through the Stage's own
``mesh=`` and ``airbox=`` settings.

Each Stage's field factory copies what it takes from here, so no Stage can
mutate another's defaults::

    mesh: dict[str, Any] = Field(default_factory=STAGE_MESH.copy)
"""

from __future__ import annotations

from typing import Any

__all__ = ["STAGE_AIRBOX", "STAGE_MESH"]

#: Mesh-pipeline arguments a Stage meshes with unless it says otherwise.
STAGE_MESH: dict[str, Any] = {
    "preset": "coarse",
    "refined_mesh_size": 0.05,
    "max_mesh_size": 40.0,
    "verbose": False,
}

#: Background region a Stage puts around its Window.
STAGE_AIRBOX: dict[str, Any] = {
    "margin_x": 2.0,
    "margin_y": 2.0,
    "z_above": 1.5,
    "z_below": 1.0,
    "material": "sio2",
}
