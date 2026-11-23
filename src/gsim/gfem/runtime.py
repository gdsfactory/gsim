"""gfem runtime resolution for the gsim.gfem adapter.

gfem is an optional dependency installed through the ``gfem`` packaging
extra. Importing :mod:`gsim.gfem` never requires it, and never imports JAX
or NGSolve, which gfem pulls in; only the solve path calls
:func:`require_gfem`.
"""

from __future__ import annotations

import warnings
from types import ModuleType

from gsim.common.optional import require_module

__all__ = ["SUPPORTED_SERIES", "require_gfem"]

_INSTALL_HINT = (
    "gfem is required for the gfem mode-solving route but is not installed. "
    "Install the optional extra: pip install 'gsim[gfem]' (or: pip install gfem)."
)

#: The gfem release series this adapter was written and checked against.
SUPPORTED_SERIES = "0.1."


def require_gfem() -> ModuleType:
    """Import and return the ``gfem`` module.

    A gfem outside the :data:`SUPPORTED_SERIES` still imports, with a
    warning: the adapter reads gfem's public names, and a new minor series
    may rename them.

    Returns:
        The imported ``gfem`` module.

    Raises:
        ImportError: When gfem is not installed, with a message naming the
            ``gfem`` packaging extra.
    """
    module = require_module("gfem", extra="gfem", hint=_INSTALL_HINT)
    version = str(getattr(module, "__version__", ""))
    if not version.startswith(SUPPORTED_SERIES):
        warnings.warn(
            f"gsim.gfem was written for gfem {SUPPORTED_SERIES}x; found "
            f"{version or 'an unknown version'}. Results are unchecked.",
            stacklevel=2,
        )
    return module
