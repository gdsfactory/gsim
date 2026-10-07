"""DEVSIM runtime resolution for the gsim.tcad backend.

DEVSIM is an optional dependency installed through the ``tcad`` packaging
extra. Importing :mod:`gsim.tcad` never requires it; only the methods that
actually talk to the solver call :func:`require_devsim`.

DEVSIM loads its BLAS/LAPACK library once, while its first import
initialises, from the libraries ``DEVSIM_MATH_LIBS`` names. Unless the user
has set that variable, the first import goes through ``devsim-openblas``
(part of the ``tcad`` extra), which points DEVSIM at the OpenBLAS that
``scipy-openblas32`` ships, so no system BLAS/LAPACK or Intel MKL is needed.

DEVSIM reports through Python's ``sys.stdout`` (its import banner, every
Newton iteration, mesh statistics); :func:`devsim_output` silences it
unless asked to stream.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from collections.abc import Iterator
from types import ModuleType

from gsim.common.optional import require_module

_INSTALL_HINT = (
    "DEVSIM is required for the charge-transport solve but is not "
    "installed. Install the optional extra: pip install 'gsim[tcad]' "
    "(or: pip install devsim devsim-openblas)."
)

_MATH_LIBS_HINT = (
    "DEVSIM is installed but could not load a BLAS/LAPACK library. Install "
    "devsim-openblas, which pip install 'gsim[tcad]' includes, or set "
    "DEVSIM_MATH_LIBS to a BLAS/LAPACK library. DEVSIM cannot initialise twice "
    "in one process, so restart Python afterwards."
)


def _select_math_libraries() -> None:
    """Point DEVSIM at devsim-openblas's OpenBLAS before its first import.

    Leaves DEVSIM's own discovery alone when DEVSIM is already imported,
    when the user chose a library through ``DEVSIM_MATH_LIBS``, or when
    devsim-openblas is not installed.
    """
    if "devsim" in sys.modules or os.environ.get("DEVSIM_MATH_LIBS"):
        return
    try:
        import devsim_openblas
    except ImportError:
        return
    devsim_openblas.configure()


def _import_devsim(name: str) -> ModuleType:
    """Import *name* from DEVSIM, failing with a remedy when it cannot start."""
    _select_math_libraries()
    # The first import prints the BLAS/UMFPACK discovery banner.
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return require_module(name, extra="tcad", hint=_INSTALL_HINT)
        except RuntimeError as err:
            # DEVSIM raises "Issues initializing DEVSIM." from its C
            # initialiser when it finds no BLAS/LAPACK.
            raise ImportError(_MATH_LIBS_HINT) from err


def require_devsim() -> ModuleType:
    """Import and return the ``devsim`` module.

    Raises:
        ImportError: When DEVSIM is not installed, with a message naming
            the ``tcad`` packaging extra, or when it cannot load a
            BLAS/LAPACK library, with a message naming the remedies.
    """
    return _import_devsim("devsim")


def import_simple_physics() -> ModuleType:
    """Import DEVSIM's prebuilt Scharfetter-Gummel physics package.

    Raises:
        ImportError: When DEVSIM is not installed, with a message naming
            the ``tcad`` packaging extra, or when it cannot load a
            BLAS/LAPACK library, with a message naming the remedies.
    """
    return _import_devsim("devsim.python_packages.simple_physics")


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
