"""femwell/skfem runtime resolution for the gsim.femwell adapter.

femwell and scikit-fem are optional dependencies installed through the
``femwell`` packaging extra. Importing :mod:`gsim.femwell` never requires
them; only the solve path calls :func:`require_femwell`.
"""

from __future__ import annotations

from types import ModuleType

from gsim.common.optional import require_module

_INSTALL_HINT = (
    "femwell/scikit-fem are required for the femwell mode-solving route "
    "but are not installed. Install the optional extra: "
    "pip install 'gsim[femwell]' (or: pip install femwell)."
)


def require_femwell() -> ModuleType:
    """Import and return the ``femwell`` module.

    Raises:
        ImportError: When femwell is not installed, with a message naming
            the ``femwell`` packaging extra.
    """
    return require_module("femwell", extra="femwell", hint=_INSTALL_HINT)


def require_skfem() -> ModuleType:
    """Import and return the ``skfem`` module.

    Raises:
        ImportError: When scikit-fem is not installed, with a message
            naming the ``femwell`` packaging extra.
    """
    return require_module("skfem", extra="femwell", hint=_INSTALL_HINT)


__all__ = ["require_femwell", "require_skfem"]
