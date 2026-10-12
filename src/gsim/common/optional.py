"""Import guards for the optional dependencies a Backend needs.

A Backend package imports without its solver: ``import gsim.tcad`` works
with no DEVSIM, ``import gsim.femwell`` with no femwell, so a user who
installed neither extra can still read a stack, build a mesh, or ask a
Study what it would run. The solver import happens on the one path that
needs it, and when it fails the error has to name the packaging extra
that provides it rather than the bare ``ModuleNotFoundError`` a user
cannot act on.

:func:`require_module` is that shape. Each Backend keeps its own named
guard on top of it — the names are public, and one of them has a banner
to swallow — so what is shared here is the import, the message and the
chained cause, not the guard.

This module is imported by module path; nothing joins
``gsim.common.__all__``.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["require_module"]


def require_module(name: str, *, extra: str, hint: str | None = None) -> ModuleType:
    """Import an optional module, or fail naming the extra that carries it.

    Args:
        name: Importable module name, dotted as ``import`` takes it.
        extra: The gsim packaging extra that installs it, named in the
            default message.
        hint: The whole message to raise instead of the default one,
            for a Backend that has more to say than the module name —
            several modules behind one extra, or a way back to another
            Route.

    Returns:
        The imported module.

    Raises:
        ImportError: When the module is not installed, with the
            original failure as its cause.
    """
    try:
        return importlib.import_module(name)
    except ImportError as err:
        raise ImportError(
            hint
            or (
                f"{name} is required here but is not installed. Install the "
                f"optional extra: pip install 'gsim[{extra}]'."
            )
        ) from err
