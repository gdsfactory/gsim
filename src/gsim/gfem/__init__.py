"""gfem adapter: 2D port modes and line solves on the cross-section mesh.

Loads the native-2D mesh :class:`~gsim.palace.BoundaryModeSim` writes into
`gfem <https://github.com/benvial/gfem>`_ and solves the port mode of its
cross-section, with the stack handled as the femwell route handles it. The
result is a :class:`~gsim.common.modes.LineReading`, the reading femwell and
Palace return too, so the three solvers are interchangeable on one line.

gfem is optional (``pip install 'gsim[gfem]'``). ``import gsim.gfem`` needs
neither gfem nor JAX; the solve path raises an error naming the extra.
"""

from __future__ import annotations

from gsim.gfem.adapter import (
    EdgeCondition,
    Medium,
    build_model,
    edge_conditions,
    impedance_paths_m,
    line_impedances,
    line_reading,
    region_media,
    solve_line,
    wall_mode_from_impedances,
)
from gsim.gfem.runtime import require_gfem

__all__ = [
    "EdgeCondition",
    "Medium",
    "build_model",
    "edge_conditions",
    "impedance_paths_m",
    "line_impedances",
    "line_reading",
    "region_media",
    "require_gfem",
    "solve_line",
    "wall_mode_from_impedances",
]
