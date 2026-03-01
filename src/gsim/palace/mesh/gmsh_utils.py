"""Gmsh utility functions — backward-compatible re-export from common.

All generic GMSH helpers now live in ``gsim.common.mesh.gmsh_utils``.
This shim keeps existing ``from gsim.palace.mesh import gmsh_utils`` working.
"""

from gsim.common.mesh.gmsh_utils import *  # noqa: F403
