"""Deterministic hashing of simulation inputs for cloud result caching.

The cache key for a cloud job is derived from the *materialized* simulation
inputs — the files written by ``write_config()`` — rather than from the
simulation object itself.

Hashing the pydantic model directly would be fragile: the simulation models
hold gdsfactory ``Component`` references, ``LayerStack`` and material objects,
callables, and private path attributes, none of which serialize stably across
processes. More importantly, a model dump would miss the generated solver
script, which does change results.

The written config directory, by contrast, is exactly the set of bytes the
solver consumes, and it is produced on every submit anyway, so hashing it is
close to free::

    digest = sha256 over sorted (relpath, sha256(file_bytes)) pairs
    key    = sha256(digest || job_type || gsim_version)

Usage:
    from gsim.hashing import compute_input_hash

    sim.write_config(config_dir)
    input_hash = compute_input_hash(config_dir, job_type="meep")
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["HASH_PREFIX", "compute_dir_digest", "compute_input_hash"]

HASH_PREFIX = "sha256:"
"""Prefix the cloud API expects on an ``input_hash`` value."""

_CHUNK_SIZE = 1024 * 1024

#: Files and directories that never affect simulation results.
_IGNORED_NAMES = frozenset({".DS_Store", "__pycache__", ".ipynb_checkpoints"})


def _iter_files(input_dir: Path):
    """Yield regular files under *input_dir*, sorted by POSIX relative path.

    Ignores editor/interpreter droppings that cannot affect the solver run.

    Args:
        input_dir: Directory to walk.

    Yields:
        ``(relative_posix_path, absolute_path)`` pairs in sorted order.
    """
    entries: list[tuple[str, Path]] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(input_dir).parts
        if any(part in _IGNORED_NAMES for part in rel_parts):
            continue
        entries.append(("/".join(rel_parts), path))
    entries.sort(key=lambda item: item[0])
    yield from entries


def _file_digest(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dir_digest(input_dir: str | Path, *, exclude: tuple[str, ...] = ()) -> str:
    """Compute a deterministic digest of a directory's contents.

    The digest covers both file names (as POSIX-style relative paths, so it
    is stable across operating systems) and file bytes. It does not cover
    file modes, timestamps, or directory entries that hold no files.

    Args:
        input_dir: Directory to digest.
        exclude: Exact POSIX relative paths to omit. By default, diagnostic
            metadata is included along with all other non-ignored files.

    Returns:
        Lowercase hex SHA-256 digest of the directory contents.

    Raises:
        FileNotFoundError: If *input_dir* does not exist.
        ValueError: If *input_dir* contains no files.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    hasher = hashlib.sha256()
    count = 0
    for rel_path, path in _iter_files(input_dir):
        if rel_path in exclude:
            continue
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_file_digest(path).encode("ascii"))
        hasher.update(b"\n")
        count += 1

    if count == 0:
        raise ValueError(f"Input directory is empty: {input_dir}")

    return hasher.hexdigest()


def compute_input_hash(input_dir: str | Path, job_type: str) -> str:
    """Compute the cloud cache key for a prepared simulation input directory.

    Folds the solver name and the gsim version into the directory digest.
    Including the gsim version is deliberately conservative: gsim generates
    the solver driver script, so an upgrade may change results even when the
    written inputs are byte-identical. It can be dropped once the server-side
    key includes the solver image version.

    Palace reserves root-level ``metadata.json`` for SDK diagnostics. Exclude
    it from this result-cache key so new measurements do not rerun identical
    physics inputs. ``compute_dir_digest()`` still includes it by default for
    callers that need a digest of the complete input bundle.

    Args:
        input_dir: Directory holding the files that will be uploaded.
        job_type: Solver name, e.g. ``"meep"`` or ``"palace"``.

    Returns:
        ``"sha256:<hex>"`` — the value to pass as ``input_hash``.

    Raises:
        FileNotFoundError: If *input_dir* does not exist.
        ValueError: If *input_dir* contains no files.

    Example:
        >>> compute_input_hash("./sim", "meep")  # doctest: +SKIP
        'sha256:6a1f...'
    """
    from gsim import __version__

    exclude = ("metadata.json",) if job_type.lower() == "palace" else ()
    digest = compute_dir_digest(input_dir, exclude=exclude)
    hasher = hashlib.sha256()
    hasher.update(digest.encode("ascii"))
    hasher.update(b"\0")
    hasher.update(job_type.lower().encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(__version__.encode("utf-8"))
    return f"{HASH_PREFIX}{hasher.hexdigest()}"
