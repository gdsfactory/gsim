"""gsim.femwell — femwell mode-solving adapter on the shared 2D mesh.

Loads the same native-2D cross-section mesh the Palace BoundaryMode
pipeline generates into skfem/femwell, with piecewise-constant materials
resolved from the stack database (cross-validation against Palace) or a
continuous carrier-derived eps(x, y) projected per element (which Palace
cannot express).

femwell/scikit-fem are optional: install the ``femwell`` extra
(``pip install 'gsim[femwell]'``). The epsilon-mapping helpers work
without them; only ``solve_modes`` needs the runtime.
"""

from gsim.femwell.adapter import (
    boundary_field_ratio,
    electrode_current,
    elementwise_epsilon,
    epsilon_by_region,
    field_fraction_outside,
    line_reading,
    solve_modes,
    z0_power_current,
)
from gsim.femwell.runtime import require_femwell, require_skfem

__all__ = [
    "boundary_field_ratio",
    "electrode_current",
    "elementwise_epsilon",
    "epsilon_by_region",
    "field_fraction_outside",
    "line_reading",
    "require_femwell",
    "require_skfem",
    "solve_modes",
    "z0_power_current",
]
