"""gsim.thermal — 2D thermal cross-section solves (heater phase shifters).

scikit-fem is optional (the ``femwell`` extra); importing this package
does not need it, only building or solving a mesh does.
"""

from gsim.thermal.mesh import (
    BOUNDARY_GROUPS,
    CrossSectionMesh,
    build_cross_section_mesh,
)

__all__ = [
    "BOUNDARY_GROUPS",
    "CrossSectionMesh",
    "build_cross_section_mesh",
]
