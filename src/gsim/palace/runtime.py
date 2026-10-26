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

A caller that cannot proceed without one asks through
:func:`require_palace_binary`, which turns "nothing found" into the error a
user can act on instead of a ``None`` to check.

A resolved binary is not a working one, so the other half of this module is
:func:`local_abort_report`: what a caller can say to a user when the binary it
found died. That diagnosis belongs beside the resolution because it is about
the same thing — the runtime rather than the model.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import shutil
import signal
import stat
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile

logger = logging.getLogger(__name__)

__all__ = [
    "PalaceLaunch",
    "PalaceRuntime",
    "binary_that_ran",
    "install_palace_runtime",
    "local_abort_report",
    "require_palace_binary",
    "resolve_launch",
    "resolve_palace_binary",
    "resolve_palace_library_dir",
    "resolve_palace_runtime",
    "run_palace",
]

_DEFAULT_BINARY_TAG = "0.18.0"
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
    executable_mode = mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    if mode != executable_mode:
        path.chmod(executable_mode)


def _restore_bin_permissions(prefix: Path) -> None:
    """Restore executable bits lost by wheel extraction, including MPI helpers."""
    for path in (prefix / "bin").rglob("*"):
        if not path.is_symlink() and path.is_file():
            _set_executable(path)


def install_palace_runtime(force: bool = False, timeout: float = 180.0) -> Path:
    """Return a Palace runtime, downloading and caching it if needed.

    An already-cached runtime is returned on any platform; only the download
    itself is restricted to Linux x86_64 (the only platform the prebuilt
    Palace CPU wheel is provided for).

    A cache written by an older gsim holds only ``bin`` and ``lib``. It is
    replaced with the full payload, including the ``share`` support files the
    MPI launcher reads. The replacement reuses a wheel already in
    ``downloads/``, so it can work offline, and downloads one otherwise. Until
    the cache is complete it does not resolve, including on other platforms,
    with auto-download disabled, or with ``download_if_missing=False``.

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
    if (
        not force
        and bin_palace.is_file()
        and lib_dir.is_dir()
        and (prefix / "share").is_dir()
    ):
        _restore_bin_permissions(prefix)
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

    def _download_wheel() -> None:
        url = _binary_wheel_url(tag)
        try:
            discovered = _binary_wheel_url_from_release(tag, timeout=timeout)
        except Exception as exc:
            logger.debug("Palace release lookup failed (%s); using %s", exc, url)
        else:
            if discovered:
                url = discovered
        with urlopen(url, timeout=timeout) as response:  # noqa: S310
            wheel_path.write_bytes(response.read())

    # A half-written or truncated wheel left by an interrupted download would
    # otherwise be reused forever, since only its existence was checked.
    if force or not wheel_path.is_file() or not is_zipfile(wheel_path):
        _download_wheel()
    if not is_zipfile(wheel_path):
        wheel_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file for Palace {tag} is not a valid wheel archive; "
            "the cached copy was discarded. Check the download URL or network."
        )

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
        share_src = payload_root / "share"
        if not bin_src.is_dir() or not lib_src.is_dir() or not share_src.is_dir():
            raise RuntimeError(
                "Downloaded wheel is missing expected bin/lib/share runtime directories"
            )

        if prefix.exists():
            shutil.rmtree(prefix)
        shutil.copytree(payload_root, prefix)

    if not bin_palace.is_file():
        raise RuntimeError("Cached runtime install did not produce bin/palace")
    _restore_bin_permissions(prefix)
    return bin_palace


def _cached_binary() -> Path | None:
    """Return the cached ``palace`` launcher path, or ``None`` if not present."""
    candidate = _cached_runtime_prefix() / "bin" / "palace"
    if not candidate.is_file() or not (candidate.parent.parent / "share").is_dir():
        return None
    _restore_bin_permissions(candidate.parent.parent)
    return candidate


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


@dataclass(frozen=True)
class PalaceRuntime:
    """A Palace executable, where it came from, and the libraries it needs."""

    binary: Path
    source: str
    """``PALACE_BIN``, ``PATH``, ``palace-toolkit-cpu``, ``gsim cached runtime``,
    ``gsim downloaded runtime`` or ``palace-toolkit``."""
    library_dir: Path | None = None
    """The shared libraries shipped with this binary, for ``LD_LIBRARY_PATH``;
    ``None`` for a binary the user installed, which finds its own."""


def resolve_palace_runtime(
    *,
    prefer_bundled: bool = False,
    download_if_missing: bool = True,
) -> PalaceRuntime | None:
    """Return a runnable Palace executable with its source, or ``None``.

    The library directory comes from the same source as the binary, so a
    user's own Palace is never handed the libraries of a bundled one.

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
    PalaceRuntime | None
        The resolved runtime, or ``None`` if no suitable binary was found.
    """
    if not prefer_bundled:
        # 1. PALACE_BIN env var (highest priority)
        env_bin = os.environ.get("PALACE_BIN", "").strip()
        if env_bin:
            candidate = Path(env_bin).expanduser().resolve()
            if candidate.is_file() and _binary_is_runnable(candidate):
                logger.info("resolve_palace_binary: using PALACE_BIN=%s", candidate)
                return PalaceRuntime(candidate, "PALACE_BIN")
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
            return PalaceRuntime(Path(resolved).resolve(), "PATH")

    # 3. Legacy palace-toolkit-cpu packaged binary
    if _palace_cpu_available():
        from palacetoolkit_palace_cpu import palace_binary_path

        candidate = palace_binary_path()
        if candidate.is_file() and _binary_is_runnable(candidate):
            logger.info(
                "resolve_palace_binary: using palace-toolkit-cpu bundled binary %s",
                candidate,
            )
            return PalaceRuntime(
                candidate.resolve(), "palace-toolkit-cpu", _cpu_package_library_dir()
            )
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
    cached_lib = _cached_library_dir()
    if cached is not None and _binary_is_runnable(cached, cached_lib):
        logger.info(
            "resolve_palace_binary: using gsim cached runtime %s",
            cached,
        )
        return PalaceRuntime(
            cached.resolve(), "gsim cached runtime", _resolved(cached_lib)
        )

    # 5. Auto-download a prebuilt Palace CPU runtime (Linux x86_64)
    if download_if_missing and _is_linux_x86_64() and _auto_download_enabled():
        with suppress(Exception):
            downloaded = install_palace_runtime(force=False)
            cached_lib = _cached_library_dir()
            if _binary_is_runnable(downloaded, cached_lib):
                logger.info(
                    "resolve_palace_binary: using gsim downloaded runtime %s",
                    downloaded,
                )
                return PalaceRuntime(
                    downloaded.resolve(),
                    "gsim downloaded runtime",
                    _resolved(cached_lib),
                )

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
                return PalaceRuntime(
                    candidate.resolve(), "palace-toolkit", _toolkit_library_dir()
                )

    return None


def resolve_palace_binary(
    *,
    prefer_bundled: bool = False,
    download_if_missing: bool = True,
) -> Path | None:
    """Return a path to a runnable Palace executable, or ``None``.

    The binary of :func:`resolve_palace_runtime`, which also says where it
    came from and which libraries it needs.

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
    found = resolve_palace_runtime(
        prefer_bundled=prefer_bundled, download_if_missing=download_if_missing
    )
    return None if found is None else found.binary


def require_palace_binary(*, hint: str | None = None) -> Path:
    """Locate a Palace binary, or fail saying how to provide one.

    :func:`resolve_palace_binary` returns ``None`` when it finds
    nothing, which is the right answer for a caller with a fallback.
    This is the answer for one without: the same resolution, raising
    with what a user can do about it.

    It raises ``RuntimeError`` rather than ``ImportError`` and takes no
    packaging extra, because what is missing is an executable and no
    ``pip install`` produces it — which is why it lives here and not
    beside :func:`gsim.common.optional.require_module`.

    Args:
        hint: The whole message to raise instead of the default one,
            for a caller that knows a way out of needing Palace at all.

    Returns:
        Path to a runnable Palace executable.

    Raises:
        RuntimeError: When no binary is available, naming the ways to
            provide one.
    """
    message = hint or (
        "No Palace binary was found. Point PALACE_BIN at one, or put 'palace' on PATH."
    )
    try:
        binary = resolve_palace_binary()
    except Exception as err:  # pragma: no cover - resolver is environment bound
        raise RuntimeError(message) from err
    if binary is None:
        raise RuntimeError(message)
    return binary


def _resolved(directory: Path | None) -> Path | None:
    """The directory, absolute, or ``None`` when absent."""
    return directory.resolve() if directory is not None else None


def _cpu_package_library_dir() -> Path | None:
    """The libraries shipped with the legacy ``palace-toolkit-cpu`` wheel.

    A wheel that names no library directory leaves its binary to find its
    own, so the lookup never stops the binary from being resolved.
    """
    if not _palace_cpu_available():
        return None
    try:
        from palacetoolkit_palace_cpu import palace_library_path

        lib_dir = palace_library_path()
    except Exception as exc:
        logger.debug("palace-toolkit-cpu names no library directory: %s", exc)
        return None
    return lib_dir.resolve() if lib_dir.is_dir() else None


def _toolkit_library_dir() -> Path | None:
    """The libraries of the ``palace-toolkit`` runtime, if it names any."""
    if not _palace_toolkit_available():
        return None
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


def resolve_palace_library_dir() -> Path | None:
    """Return the Palace library directory (for ``LD_LIBRARY_PATH``).

    The first bundled runtime that ships libraries, whichever binary is
    used; :attr:`PalaceRuntime.library_dir` is the one that matches a
    resolved binary.

    Returns:
    -------
    Path | None
    """
    return (
        _cpu_package_library_dir()
        or _resolved(_cached_library_dir())
        or _toolkit_library_dir()
    )


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


_SIF_CANDIDATES = (
    Path("Palace.sif"),
    Path("palace.sif"),
    Path("bin/Palace.sif"),
    Path("bin/palace.sif"),
)


@dataclass(frozen=True)
class PalaceLaunch:
    """How to start Palace locally: the command before its arguments, and its env."""

    command: tuple[str, ...]
    """``(binary,)``, or ``("apptainer", "run", sif)``."""
    env: dict[str, str]
    """The process environment, with a bundled runtime's libraries on
    ``LD_LIBRARY_PATH``."""
    apptainer: bool = False

    @property
    def binary(self) -> str:
        """The executable that runs, as an abort report names it."""
        return self.command[0]


def _first_existing(
    roots: Sequence[Path], candidates: Sequence[Path], *, executable: bool = False
) -> Path | None:
    """The first ``root / candidate`` that exists (and is executable, if asked)."""
    for root in roots:
        for rel in candidates:
            candidate = (root / rel).resolve()
            if candidate.exists() and (not executable or os.access(candidate, os.X_OK)):
                return candidate
    return None


def resolve_launch(
    *,
    executable: str | Path | None = None,
    sif: str | Path | None = None,
    use_apptainer: bool = False,
    search_roots: Sequence[Path] = (),
    verbose: bool = False,
) -> PalaceLaunch:
    """Decide which Palace runs a local simulation, and how.

    An executable wins over a SIF. When neither is given, the precedence is
    ``PALACE_EXECUTABLE``, then ``bin/palace`` under one of ``search_roots``,
    then — unless ``use_apptainer`` and a SIF (``PALACE_SIF`` or a
    ``Palace.sif`` under the roots) is available — the binary that
    :func:`resolve_palace_runtime` finds, falling back to ``palace`` on
    ``PATH``.

    Args:
        executable: Palace executable to run directly.
        sif: Palace Apptainer image.
        use_apptainer: Run the SIF through Apptainer when no executable is
            selected.
        search_roots: Directories searched for ``bin/palace`` and a SIF.
        verbose: Log which binary was chosen and its version.

    Returns:
        The :class:`PalaceLaunch`.

    Raises:
        FileNotFoundError: If the chosen SIF or executable does not exist or
            is not executable.
        RuntimeError: If Apptainer is asked for but not installed.
    """
    roots: list[Path] = []
    for root in search_roots:
        resolved = Path(root).expanduser().resolve()
        if resolved not in roots:
            roots.append(resolved)

    if sif is None:
        sif = os.environ.get("PALACE_SIF") or _first_existing(roots, _SIF_CANDIDATES)
        if sif is not None and verbose and "PALACE_SIF" not in os.environ:
            logger.info("Auto-discovered Palace SIF: %s", sif)
    if executable is None:
        executable = os.environ.get("PALACE_EXECUTABLE") or None
    if executable is None:
        # Prefer a local Palace executable when available, even when a SIF is
        # configured via PALACE_SIF. This keeps local developer workflows on
        # native binaries by default.
        executable = _first_existing(roots, [Path("bin/palace")], executable=True)
        if executable is not None and verbose:
            logger.info("Auto-discovered Palace executable: %s", executable)

    if use_apptainer and executable is None and sif is not None:
        sif_path = Path(sif).expanduser().resolve()
        if not sif_path.exists():
            raise FileNotFoundError(
                f"Palace SIF file not found: {sif_path}. "
                "Install Palace via Apptainer or provide correct path."
            )
        if shutil.which("apptainer") is None:
            raise RuntimeError(
                "Apptainer not found. Install Apptainer to run local simulations "
                "with use_apptainer=True."
            )
        return PalaceLaunch(
            ("apptainer", "run", str(sif_path)), os.environ.copy(), apptainer=True
        )

    if use_apptainer and verbose:
        logger.info(
            "Using Palace executable (%s); ignoring use_apptainer=True.",
            executable or "palace",
        )

    library_dir: Path | None = None
    if executable is not None:
        exe_path = Path(executable).expanduser().resolve()
    else:
        found = resolve_palace_runtime()
        if found is None:
            exe_path = Path("palace")
        else:
            exe_path, library_dir = found.binary, found.library_dir
            if verbose:
                _log_runtime(found)

    if not exe_path.exists():
        on_path = shutil.which(str(exe_path))
        if on_path is None:
            raise FileNotFoundError(
                f"Palace executable not found: {exe_path}. "
                "Install Palace directly or provide correct path via "
                "palace_executable, or let gsim auto-download the "
                "prebuilt CPU runtime (Linux x86_64)."
            )
        exe_path = Path(on_path)
    if not os.access(exe_path, os.X_OK):
        raise FileNotFoundError(
            f"Palace executable is not executable: {exe_path}. "
            "Update file permissions or provide a valid executable path."
        )

    env = os.environ.copy()
    if library_dir is not None:
        prior = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{library_dir}:{prior}" if prior else str(library_dir)
        logger.debug(
            "Injected LD_LIBRARY_PATH for bundled Palace: %s", env["LD_LIBRARY_PATH"]
        )
    return PalaceLaunch((str(exe_path),), env)


def _log_runtime(found: PalaceRuntime) -> None:
    """Log a resolved runtime and, when it answers, its version."""
    logger.info("Palace binary: %s  (source: %s)", found.binary, found.source)
    with suppress(Exception):
        version = subprocess.run(  # noqa: S603
            [str(found.binary), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if version.returncode == 0:
            logger.info(version.stdout.strip())


def run_palace(
    launch: PalaceLaunch,
    *,
    cwd: Path,
    num_processes: int,
    num_threads: int | None = None,
    verbose: bool = True,
    emit: Callable[[str], None] | None = None,
) -> None:
    """Run Palace on the ``config.json`` in ``cwd``.

    Args:
        launch: Which Palace, and how, from :func:`resolve_launch`.
        cwd: The simulation directory.
        num_processes: MPI ranks.
        num_threads: OpenMP threads, when set.
        verbose: Stream Palace's output line by line through ``emit``.
        emit: Where streamed lines go; defaults to this module's logger.

    Raises:
        subprocess.CalledProcessError: If Palace exits nonzero, carrying the
            tail of its output on ``stderr``.
        RuntimeError: If the launcher itself cannot be found.
    """
    emit = emit or logger.info
    cmd = [*launch.command, "-np", str(num_processes)]
    if num_threads is not None:
        cmd.extend(["-nt", str(num_threads)])
    cmd.append("config.json")
    if verbose:
        how = "via Apptainer" if launch.apptainer else "directly"
        emit(f"Running Palace simulation in {cwd} {how}")
        emit(f"Command: {' '.join(cmd)}")
        emit(f"Processes: {num_processes}")

    try:
        if verbose:
            streamed_lines: list[str] = []
            with subprocess.Popen(  # noqa: S603
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=launch.env,
            ) as process:
                if process.stdout is not None:
                    for line in process.stdout:
                        line = line.rstrip("\n")
                        streamed_lines.append(line)
                        if line:
                            emit(line)
                returncode = process.wait()
            if returncode != 0:
                # The output was streamed as it arrived, so a report quoting
                # the decisive line loses nothing a user has not already seen.
                raise subprocess.CalledProcessError(
                    returncode,
                    cmd,
                    output="",
                    stderr="\n".join(streamed_lines[-_FAILURE_OUTPUT_TAIL_LINES:]),
                )
        else:
            result = subprocess.run(  # noqa: S603
                cmd,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                env=launch.env,
            )
            if result.stdout:
                logger.debug(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
    except FileNotFoundError as e:
        if launch.apptainer:
            raise RuntimeError(
                "Apptainer not found. Install Apptainer to run local simulations "
                "with use_apptainer=True."
            ) from e
        raise RuntimeError(
            "Palace executable not found. Install Palace directly or provide "
            "correct path via palace_executable parameter, "
            "or set PALACE_EXECUTABLE environment variable."
        ) from e


# Return codes carry POSIX signal numbers, even on a Windows host whose own
# ``signal`` module numbers them differently (SIGABRT is 22 there). These are
# the ones a dying solver usually reports, numbered alike on Linux and macOS.
# A Windows process that dies on an exception exits with the NTSTATUS of the
# exception rather than with a signal; these are the ones a dying solver
# usually reports.
_WINDOWS_STATUS = {
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}

_POSIX_SIGNALS = {
    1: "SIGHUP",
    2: "SIGINT",
    3: "SIGQUIT",
    4: "SIGILL",
    6: "SIGABRT",
    8: "SIGFPE",
    9: "SIGKILL",
    11: "SIGSEGV",
    13: "SIGPIPE",
    14: "SIGALRM",
    15: "SIGTERM",
}


# Printed by Palace's ``palace`` launcher script when its MPI launcher (``mpirun``
# unless ``--launcher`` is given) is not on PATH.
_MPI_LAUNCHER_MISSING = "Could not locate MPI launcher"
_FAILURE_OUTPUT_TAIL_LINES = 200


def _captured_lines(*streams: object) -> list[str]:
    """Split captured process output into lines, skipping empty streams."""
    lines: list[str] = []
    for stream in streams:
        if isinstance(stream, str) and stream:
            lines.extend(stream.splitlines())
    return lines


def _death_signal(returncode: int) -> int | None:
    """The signal a process died on, if its return code says it did.

    Args:
        returncode: What the process exited with — negative when
            ``subprocess`` saw the signal itself, 128 + signum when a
            shell in between reported it.

    Returns:
        The signal number, or ``None`` for a plain exit code — which a
        code above the shell's ``128 + signum`` range is too: a Windows
        process that died on an exception exits with that exception's
        NTSTATUS (``0xC0000005`` for an access violation), not a signal.
    """
    if returncode < 0:
        return -returncode
    if 128 < returncode < 128 + signal.NSIG:
        return returncode - 128
    return None


def binary_that_ran(
    err: subprocess.CalledProcessError, requested: str | Path | None
) -> str | Path:
    """The Palace executable an aborted run used.

    The caller names it when it resolved one itself; otherwise the
    command the failed run carries does, since a run that resolved its
    own binary still has to say which one died.

    Args:
        err: What the local run raised.
        requested: The executable the caller asked for, if any.

    Returns:
        The executable, as a path or as the command's first word.
    """
    if requested is not None:
        return requested
    cmd = err.cmd
    if isinstance(cmd, (list, tuple)) and cmd:
        return str(cmd[0])
    return str(cmd)


def local_abort_report(
    err: subprocess.CalledProcessError,
    *,
    binary: Path | str,
    output_dir: Path | str | None,
    wrote_output: bool,
    during: str,
    remedy: str | None = None,
) -> str:
    """Turn a locally run Palace binary's exit status into something actionable.

    A broken Palace runtime — typically a bundled MPI that cannot start —
    kills the binary before the solver writes anything, and the bare
    ``CalledProcessError`` carries an exit status and nothing else. What
    a user can act on is which binary ran and whether the solver got as
    far as producing output: a signal death with none means the runtime,
    not the model, and the fix is a different binary rather than a
    different problem. A plain nonzero exit is Palace refusing the run on
    its own terms, so that one is left to say why through its stderr
    rather than blamed on the runtime. One refusal is named outright:
    the ``palace`` launcher script reporting that it found no ``mpirun``,
    which the prebuilt runtime gsim installs does not ship.

    The report ends with the last 200 lines of the captured output, when
    there are any, so a run that was not streamed still shows where it
    died.

    This reads a ``CalledProcessError`` and a local run directory, which
    is why it says ``local``: a run submitted to the cloud fails in its
    own way and is not covered here.

    Args:
        err: What the local run raised.
        binary: The Palace executable that ran.
        output_dir: Where the run's output would be, named in the report.
        wrote_output: Whether the run left any solver output, which is
            what separates a dead runtime from a dead solve.
        during: What was being run when it died, named in the first
            sentence — a frequency, a sweep point, a problem name.
        remedy: A second way out, beyond pointing ``PALACE_BIN`` at a
            working runtime, phrased as an imperative clause.

    Returns:
        The report text.
    """
    code = err.returncode
    signum = _death_signal(code)
    signal_note = ""
    if signum is not None:
        name = _POSIX_SIGNALS.get(signum)
        if name is None and os.name != "nt":
            with suppress(ValueError):
                name = signal.Signals(signum).name
        signal_note = f" ({name})" if name else f" (signal {signum})"
    elif (status := _WINDOWS_STATUS.get(code)) is not None:
        signal_note = f" ({status})"
    killed = signum is not None or signal_note != ""

    # MPI closes its error blocks with a line of dashes, so the last
    # *worded* line is the one that says anything.
    stderr_note = ""
    stderr = err.stderr if isinstance(err.stderr, str) else ""
    if lines := [
        line for line in stderr.splitlines() if any(c.isalnum() for c in line)
    ]:
        stderr_note = f" Its last stderr line: {lines[-1].strip()!r}."

    output_lines = _captured_lines(err.output, err.stderr)
    if any(_MPI_LAUNCHER_MISSING in line for line in output_lines):
        diagnosis = (
            "Its launcher script could not find `mpirun`. Install an MPI "
            "runtime that provides it (for example `sudo apt install "
            "openmpi-bin` on Ubuntu), or run with use_apptainer=True; the "
            "prebuilt Palace runtime that gsim installs includes the MPI "
            "libraries but not `mpirun`."
        )
    elif wrote_output:
        diagnosis = (
            f"It got far enough to write partial solver output (in "
            f"{output_dir}), so the runtime did start; the solve itself "
            "died before finishing."
        )
    elif killed:
        diagnosis = (
            "It was killed before writing any solver output, which means "
            "the Palace runtime — typically its bundled MPI — failed "
            "before the solver started, not that the model is wrong."
        )
    else:
        diagnosis = (
            "It exited before writing any solver output; its stderr "
            "should say why it refused the run."
        )
    way_out = "Point PALACE_BIN at a Palace whose runtime works here"
    way_out = f"{way_out}, or {remedy}." if remedy else f"{way_out}."
    report = (
        f"Palace aborted {during} with exit status {code}{signal_note}. "
        f"The binary that ran is {binary}. {diagnosis}{stderr_note} {way_out}"
    )
    tail = "\n".join(output_lines[-_FAILURE_OUTPUT_TAIL_LINES:])
    if tail.strip():
        report += f"\n\nOutput (tail):\n{tail}"
    return report
