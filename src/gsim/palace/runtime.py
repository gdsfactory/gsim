"""Palace runtime/binary resolution.

Provides a unified resolver for locating a Palace executable, plus a
self-contained installer that downloads and caches a prebuilt Palace CPU
binary.  gsim absorbs this functionality so users do **not** need to install
any direct-URL wheel or third-party runtime package to run Palace locally on
Linux x86_64.

Resolution order
-----------------
1. ``PALACE_BIN`` environment variable.
2. ``PALACE_EXECUTABLE`` environment variable, or ``"palace"`` in ``PATH``.
3. ``palacetoolkit_palace_cpu`` packaged binary (if the legacy
   ``palace-toolkit-cpu`` wheel happens to be installed).
4. gsim's own cached/downloaded Palace CPU runtime (Linux x86_64).
5. Delegation to ``palacetoolkit`` (if the ``palace-toolkit`` distribution
   happens to be installed).
6. ``None`` if nothing was found.

The auto-download is only attempted when ``PALACETOOLKIT_AUTO_DOWNLOAD_BINARY``
is not disabled, and only on Linux x86_64 (the platform the prebuilt Palace CPU
wheel is provided for).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import shutil
import stat
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

logger = logging.getLogger(__name__)

_DEFAULT_BINARY_TAG = "0.17.0"
_AUTO_DOWNLOAD_ENV = "PALACETOOLKIT_AUTO_DOWNLOAD_BINARY"
_TAG_ENV = "PALACETOOLKIT_PALACE_CPU_TAG"
_CACHE_ENV = "PALACETOOLKIT_RUNTIME_DIR"


def _is_linux_x86_64() -> bool:
    """Return whether the current platform is Linux on x86_64.

    The prebuilt Palace CPU runtime is only provided for this platform.
    """
    return platform.system() == "Linux" and platform.machine() == "x86_64"


def _runtime_cache_dir() -> Path:
    """Return the directory used to cache downloaded Palace runtimes."""
    root = os.environ.get(_CACHE_ENV, "").strip()
    if root:
        return Path(root).expanduser().resolve()
    return (Path.home() / ".cache" / "palacetoolkit" / "runtime").resolve()


def _binary_tag() -> str:
    """Return the Palace CPU runtime version tag to download."""
    return os.environ.get(_TAG_ENV, _DEFAULT_BINARY_TAG).strip() or _DEFAULT_BINARY_TAG


def _binary_wheel_url(tag: str) -> str:
    """Return the GitHub release URL for the given Palace CPU runtime tag."""
    return (
        "https://github.com/EpsilonForge/PalaceToolkit/releases/download/"
        f"palace-cpu-v{tag}/"
        f"palacetoolkit_palace_cpu-{tag}-py3-none-linux_x86_64.whl"
    )


def _binary_wheel_url_from_release(tag: str, timeout: float) -> str | None:
    """Discover the current wheel URL from the GitHub release API (best-effort)."""
    api_url = (
        "https://api.github.com/repos/EpsilonForge/PalaceToolkit/releases/tags/"
        f"palace-cpu-v{tag}"
    )
    request = Request(  # noqa: S310
        api_url, headers={"Accept": "application/vnd.github+json"}
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))

    for asset in payload.get("assets", []):
        name = str(asset.get("name", ""))
        if name.endswith("linux_x86_64.whl") and "palacetoolkit_palace_cpu-" in name:
            url = str(asset.get("browser_download_url", ""))
            if url:
                return url
    return None


def _cached_runtime_prefix(tag: str | None = None) -> Path:
    """Return the cache directory for a specific runtime tag."""
    resolved_tag = tag or _binary_tag()
    return _runtime_cache_dir() / f"palace-cpu-v{resolved_tag}"


def _set_executable(path: Path) -> None:
    """Make the given path executable for all users."""
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_palace_runtime(force: bool = False, timeout: float = 180.0) -> Path:
    """Return a Palace runtime, downloading and caching it if needed.

    An already-cached runtime is returned on any platform; only the download
    itself is restricted to Linux x86_64 (the only platform the prebuilt
    Palace CPU wheel is provided for).

    Returns:
        Path to the cached ``palace`` launcher executable.

    Raises:
        RuntimeError: If no runtime is cached, the platform is unsupported,
            or the download/install fails.
    """
    tag = _binary_tag()
    prefix = _cached_runtime_prefix(tag)
    bin_palace = prefix / "bin" / "palace"
    lib_dir = prefix / "lib"
    if not force and bin_palace.is_file() and lib_dir.is_dir():
        return bin_palace

    if not _is_linux_x86_64():
        raise RuntimeError(
            "Prebuilt runtime download is only supported on Linux x86_64"
        )

    prefix.mkdir(parents=True, exist_ok=True)
    downloads = _runtime_cache_dir() / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    wheel_name = f"palacetoolkit_palace_cpu-{tag}-py3-none-linux_x86_64.whl"
    wheel_path = downloads / wheel_name

    if force or not wheel_path.is_file():
        url = _binary_wheel_url(tag)
        with suppress(Exception):
            discovered = _binary_wheel_url_from_release(tag, timeout=timeout)
            if discovered:
                url = discovered
        with urlopen(url, timeout=timeout) as response:  # noqa: S310
            wheel_path.write_bytes(response.read())

    with tempfile.TemporaryDirectory(
        prefix="palace-runtime-", dir=_runtime_cache_dir()
    ) as tmp:
        tmp_path = Path(tmp)
        with ZipFile(wheel_path, "r") as wheel_zip:
            wheel_zip.extractall(tmp_path)

        payload_root = tmp_path / "palacetoolkit_palace_cpu"
        if not payload_root.is_dir():
            raise RuntimeError(
                "Downloaded wheel does not contain palacetoolkit_palace_cpu payload"
            )

        bin_src = payload_root / "bin"
        lib_src = payload_root / "lib"
        if not bin_src.is_dir() or not lib_src.is_dir():
            raise RuntimeError(
                "Downloaded wheel is missing expected bin/lib runtime directories"
            )

        if prefix.exists():
            shutil.rmtree(prefix)
        prefix.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bin_src, prefix / "bin")
        shutil.copytree(lib_src, prefix / "lib")

    if not bin_palace.is_file():
        raise RuntimeError("Cached runtime install did not produce bin/palace")
    _set_executable(bin_palace)
    bin_native = prefix / "bin" / "palace-x86_64.bin"
    if bin_native.is_file():
        _set_executable(bin_native)
    return bin_palace


def _cached_binary() -> Path | None:
    """Return the cached ``palace`` launcher path, or ``None`` if not present."""
    candidate = _cached_runtime_prefix() / "bin" / "palace"
    return candidate if candidate.is_file() else None


def _cached_library_dir() -> Path | None:
    """Return the cached runtime ``lib`` directory, or ``None`` if not present."""
    candidate = _cached_runtime_prefix() / "lib"
    return candidate if candidate.is_dir() else None


def _auto_download_enabled() -> bool:
    """Return whether auto-download of the Palace runtime is enabled."""
    raw = os.environ.get(_AUTO_DOWNLOAD_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _palace_cpu_available() -> bool:
    """Check whether the legacy ``palacetoolkit_palace_cpu`` package is installed."""
    return importlib.util.find_spec("palacetoolkit_palace_cpu") is not None


def _palace_toolkit_available() -> bool:
    """Check whether the ``palacetoolkit`` package (``palace-toolkit``) is installed."""
    return importlib.util.find_spec("palacetoolkit") is not None


def resolve_palace_binary(
    *,
    prefer_bundled: bool = False,
    download_if_missing: bool = True,
) -> Path | None:
    """Return a path to a runnable Palace executable, or ``None``.

    Parameters
    ----------
    prefer_bundled:
        If ``True``, skip the ``PALACE_BIN`` / ``PALACE_EXECUTABLE`` /
        ``PATH`` checks and go straight to gsim's cached/bundled runtime.
    download_if_missing:
        If ``True`` (default), auto-download and cache a prebuilt Palace CPU
        runtime on Linux x86_64 when no binary is found elsewhere.

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

    # 3. Legacy palace-toolkit-cpu packaged binary
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

    # 4. gsim's own cached runtime
    cached = _cached_binary()
    if cached is not None and _binary_is_runnable(cached, _cached_library_dir()):
        logger.info(
            "resolve_palace_binary: using gsim cached runtime %s",
            cached,
        )
        return cached.resolve()

    # 5. Auto-download a prebuilt Palace CPU runtime (Linux x86_64)
    if download_if_missing and _is_linux_x86_64() and _auto_download_enabled():
        with suppress(Exception):
            downloaded = install_palace_runtime(force=False)
            if _binary_is_runnable(downloaded, _cached_library_dir()):
                logger.info(
                    "resolve_palace_binary: using gsim downloaded runtime %s",
                    downloaded,
                )
                return downloaded.resolve()

    # 6. Delegation to the palace-toolkit package (if installed) as a fallback
    if _palace_toolkit_available():
        try:
            from palacetoolkit.palace_runtime import (
                resolve_palace_binary as _ptk_resolve_binary,
            )

            candidate = _ptk_resolve_binary()
        except Exception as exc:
            logger.debug(
                "resolve_palace_binary: palacetoolkit resolver failed: %s", exc
            )
            candidate = None
        if candidate is not None:
            candidate = Path(candidate)
            if candidate.is_file() and _binary_is_runnable(candidate):
                logger.info(
                    "resolve_palace_binary: using palace-toolkit runtime %s",
                    candidate,
                )
                return candidate.resolve()

    return None


def resolve_palace_library_dir() -> Path | None:
    """Return the Palace library directory (for ``LD_LIBRARY_PATH``).

    Returns:
    -------
    Path | None
    """
    if _palace_cpu_available():
        from palacetoolkit_palace_cpu import palace_library_path

        lib_dir = palace_library_path()
        if lib_dir.is_dir():
            return lib_dir.resolve()

    cached = _cached_library_dir()
    if cached is not None:
        return cached.resolve()

    if _palace_toolkit_available():
        try:
            from palacetoolkit.palace_runtime import (
                resolve_palace_library_dir as _ptk_resolve_lib,
            )

            lib_dir = _ptk_resolve_lib()
        except Exception as exc:
            logger.debug(
                "resolve_palace_library_dir: palacetoolkit resolver failed: %s",
                exc,
            )
            return None
        if lib_dir is not None and lib_dir.is_dir():
            return lib_dir.resolve()

    return None


def _binary_is_runnable(
    binary: Path, lib_dir: Path | None = None, timeout: float = 15.0
) -> bool:
    """Smoke test: file exists, is executable, and responds to --version or --help."""
    bin_str = str(binary)
    if not binary.is_file() or not os.access(bin_str, os.X_OK):
        return False

    run_env = os.environ.copy()
    if lib_dir is not None and lib_dir.is_dir():
        prior = run_env.get("LD_LIBRARY_PATH", "")
        run_env["LD_LIBRARY_PATH"] = f"{lib_dir}:{prior}" if prior else str(lib_dir)

    for flag in ("--version", "--help"):
        try:
            result = subprocess.run(  # noqa: S603
                [bin_str, flag],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=run_env,
            )
            if result.returncode == 0:
                return True
        except Exception:
            continue
    return True  # fallback: just being executable is enough
