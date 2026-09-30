"""DEVSIM runtime resolution for the gsim.tcad backend.

DEVSIM is an optional dependency installed through the ``tcad`` packaging
extra. Importing :mod:`gsim.tcad` never requires it; only the methods that
actually talk to the solver call :func:`require_devsim`.

DEVSIM reports through Python's ``sys.stdout`` (its import banner, every
Newton iteration, mesh statistics); :func:`devsim_output` silences it
unless asked to stream.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator
from types import ModuleType

from gsim.common.optional import require_module

_INSTALL_HINT = (
    "DEVSIM is required for the charge-transport solve but is not "
    "installed. Install the optional extra: pip install 'gsim[tcad]' "
    "(or: pip install devsim)."
)


def require_devsim() -> ModuleType:
    """Import and return the ``devsim`` module.

    Raises:
        ImportError: When DEVSIM is not installed, with a message naming
            the ``tcad`` packaging extra.
    """
    # The first import prints the BLAS/UMFPACK discovery banner.
    with contextlib.redirect_stdout(io.StringIO()):
        return require_module("devsim", extra="tcad", hint=_INSTALL_HINT)


def import_simple_physics() -> ModuleType:
    """Import DEVSIM's prebuilt Scharfetter-Gummel physics package.

    Raises:
        ImportError: When DEVSIM is not installed, with a message naming
            the ``tcad`` packaging extra.
    """
    return require_module(
        "devsim.python_packages.simple_physics", extra="tcad", hint=_INSTALL_HINT
    )


@contextlib.contextmanager
def devsim_output(verbose: bool) -> Iterator[None]:
    """Stream DEVSIM's solver output when *verbose*, discard it otherwise.

    Args:
        verbose: Let DEVSIM's output through to ``sys.stdout``.
    """
    if verbose:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


__all__ = ["devsim_output", "import_simple_physics", "require_devsim"]
