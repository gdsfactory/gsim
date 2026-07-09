"""Palace runtime/binary resolution.

Provides a unified resolver for locating a Palace executable, with
optional delegation to ``palacetoolkit`` when installed.

Resolution order
-----------------
1. ``PALACE_BIN`` environment variable.
2. ``PALACE_EXECUTABLE`` environment variable, or ``"palace"`` in ``PATH``.
3. ``palacetoolkit.palace_runtime.resolve_palace_binary()`` (when the
   optional ``palace-toolkit`` package is installed).
4. ``None`` if nothing was found.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _palacetoolkit_available() -> bool:
    """Check whether the optional ``palacetoolkit`` package can be imported."""
    return importlib.util.find_spec("palacetoolkit") is not None


def resolve_palace_binary(
    *,
    prefer_bundled: bool = False,
) -> Path | None:
    """Return a path to a runnable Palace executable, or ``None``.

    Parameters
    ----------
    prefer_bundled:
        If ``True``, skip the ``PALACE_BIN`` / ``PALACE_EXECUTABLE`` /
        ``PATH`` checks and go straight to the palace-toolkit bundled
        binary (useful when the caller explicitly wants the bundled
        runtime).

    Returns:
    -------
    Path | None
        Absolute path to a Palace executable, or ``None`` if no suitable
        binary was found.
    """
    if not prefer_bundled:
        # 1. PALACE_BIN env var (highest priority)
        env_bin = os.environ.get("PALACE_BIN", "").strip()
        if env_bin:
            candidate = Path(env_bin).expanduser().resolve()
            if candidate.is_file() and _binary_is_runnable(candidate):
                logger.info("resolve_palace_binary: using PALACE_BIN=%s", candidate)
                return candidate
            logger.warning(
                "resolve_palace_binary: PALACE_BIN=%s is not a runnable executable",
                candidate,
            )

        # 2. PALACE_EXECUTABLE env var or "palace" in PATH
        exe = os.environ.get("PALACE_EXECUTABLE", "").strip() or "palace"
        resolved = shutil.which(exe)
        if resolved is not None:
            logger.info(
                "resolve_palace_binary: using PALACE_EXECUTABLE/PATH %s -> %s",
                exe,
                resolved,
            )
            return Path(resolved).resolve()

    # 3. Optional palace-toolkit bundled binary
    if _palacetoolkit_available():
        # Lazy import — only load when palace-toolkit is installed
        from palacetoolkit.palace_runtime import (  # type: ignore[import-untyped,import-not-found]
            resolve_palace_binary as ptk_resolve,
        )

        ptk_bin = ptk_resolve()
        if ptk_bin is not None:
            logger.info(
                "resolve_palace_binary: using palace-toolkit bundled binary %s",
                ptk_bin,
            )
            return Path(ptk_bin).resolve()
        logger.info(
            "resolve_palace_binary: palace-toolkit is installed "
            "but no bundled binary was found"
        )
    else:
        logger.debug("resolve_palace_binary: palace-toolkit not installed — skipping")

    return None


def resolve_palace_library_dir() -> Path | None:
    """Return the Palace library directory (for ``LD_LIBRARY_PATH``).

    Only available when ``palace-toolkit`` is installed and provides a
    bundled ``lib/`` directory alongside its binary.

    Returns:
    -------
    Path | None
    """
    if not _palacetoolkit_available():
        return None

    from palacetoolkit.palace_runtime import (  # type: ignore[import-untyped,import-not-found]
        resolve_palace_library_dir as ptk_lib_dir,
    )

    lib_dir = ptk_lib_dir()
    return Path(lib_dir).resolve() if lib_dir is not None else None


def _binary_is_runnable(binary: Path, timeout: float = 15.0) -> bool:
    """Quick smoke test — run ``<binary> --version``."""
    import subprocess

    bin_str = str(binary)
    try:
        result = subprocess.run(  # noqa: S603
            [bin_str, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0
