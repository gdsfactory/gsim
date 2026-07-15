"""Palace runtime/binary resolution.

Provides a unified resolver for locating a Palace executable, with
optional delegation to the ``palacetoolkit_palace_cpu`` package (the
``palace-toolkit-cpu`` distribution) when installed.

Resolution order
-----------------
1. ``PALACE_BIN`` environment variable.
2. ``PALACE_EXECUTABLE`` environment variable, or ``"palace"`` in ``PATH``.
3. ``palacetoolkit_palace_cpu`` packaged binary (when the optional
   ``palace-toolkit-cpu`` extra is installed).
4. ``None`` if nothing was found.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _palace_cpu_available() -> bool:
    """Check whether the optional ``palacetoolkit_palace_cpu`` package is installed."""
    return importlib.util.find_spec("palacetoolkit_palace_cpu") is not None


def resolve_palace_binary(
    *,
    prefer_bundled: bool = False,
) -> Path | None:
    """Return a path to a runnable Palace executable, or ``None``.

    Parameters
    ----------
    prefer_bundled:
        If ``True``, skip the ``PALACE_BIN`` / ``PALACE_EXECUTABLE`` /
        ``PATH`` checks and go straight to the palace-toolkit-cpu bundled
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

    # 3. Optional palace-toolkit-cpu bundled binary
    if _palace_cpu_available():
        from palacetoolkit_palace_cpu import palace_binary_path

        candidate = palace_binary_path()
        if candidate.is_file() and _binary_is_runnable(candidate):
            logger.info(
                "resolve_palace_binary: using palace-toolkit-cpu bundled binary %s",
                candidate,
            )
            return candidate.resolve()
        logger.info(
            "resolve_palace_binary: palace-toolkit-cpu is installed "
            "but no bundled binary was found"
        )
    else:
        logger.debug(
            "resolve_palace_binary: palace-toolkit-cpu not installed — skipping"
        )

    return None


def resolve_palace_library_dir() -> Path | None:
    """Return the Palace library directory (for ``LD_LIBRARY_PATH``).

    Only available when ``palace-toolkit-cpu`` is installed and provides a
    bundled ``lib/`` directory alongside its binary.

    Returns:
    -------
    Path | None
    """
    if not _palace_cpu_available():
        return None

    from palacetoolkit_palace_cpu import palace_library_path

    lib_dir = palace_library_path()
    return lib_dir.resolve() if lib_dir.is_dir() else None


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
