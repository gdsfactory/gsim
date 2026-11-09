"""Several wave-port modes on one port face (the multimode wave port).

Palace takes one ``WavePort`` entry per (face, mode): every entry runs its own
2D mode solve on the face and picks eigenvector ``Mode``. Entries may share a
boundary attribute, but only one of them may be ``"Active"`` (the one that
terminates the face), and the active entry must have the lowest ``Index`` of
the face, because Palace walks the entries in ``Index`` order and refuses an
active port whose attribute is already taken. The other entries are
observation-only: they are measured, and they can be excited, but they do not
load the face.

The config generator writes the entries of a multimode port and, next to
``config.json``, a sidecar ``port_modes.json`` that records for every Palace
``Index`` which gsim port (face) and mode it stands for and whether it is the
active one. The wave-port log checks and the later assembly of modal
S-parameters read the sidecar, because the S-parameter columns carry only the
Palace index.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

PORT_MODES_FILENAME = "port_modes.json"
"""Name of the sidecar next to ``config.json``."""

FACE_POSITION_TOL_UM = 1e-6
"""Two ports whose centres agree to this distance and whose orientations agree
(modulo 180 degrees) are on the same face."""

__all__ = [
    "FACE_POSITION_TOL_UM",
    "PORT_MODES_FILENAME",
    "PortModeEntry",
    "check_port_mode_entries",
    "check_wave_port_modes",
    "faces_by_index",
    "read_port_modes",
    "same_face_pairs",
    "write_port_modes",
]


@dataclass(frozen=True)
class PortModeEntry:
    """One Palace ``WavePort`` entry of a multimode port.

    Attributes:
        index: The Palace ``Index`` of the entry (the column index in
            ``port-S.csv``).
        port: Name of the gsim port, i.e. of the face the entry sits on.
        portnumber: 1-based position of the gsim port, which is also the
            ``Index`` of its active entry.
        mode: The Palace ``Mode`` (eigenvector number) the entry follows.
        active: Whether the entry terminates the face.
        excited: Whether the entry has an ``Excitation`` of its own.
        attributes: Boundary attributes of the face.
    """

    index: int
    port: str
    portnumber: int
    mode: int
    active: bool
    excited: bool
    attributes: tuple[int, ...] = ()


def check_port_mode_entries(entries: Sequence[PortModeEntry]) -> None:
    """Refuse a set of entries Palace would reject or that gsim cannot decode.

    Raises:
        ValueError: For a repeated ``Index``; for a face (port) with no active
            entry or more than one; for an active entry that does not have the
            lowest ``Index`` of its face (Palace visits entries in ``Index``
            order and refuses an active port on an attribute already taken);
            or for a mode that appears twice on one face.
    """
    seen: set[int] = set()
    by_face: dict[int, list[PortModeEntry]] = defaultdict(list)
    for entry in entries:
        if entry.index in seen:
            raise ValueError(f"Palace WavePort Index {entry.index} is used twice.")
        seen.add(entry.index)
        by_face[entry.portnumber].append(entry)
    for group in by_face.values():
        name = group[0].port
        active = [entry for entry in group if entry.active]
        if len(active) != 1:
            raise ValueError(
                f"Wave port {name!r} has {len(active)} active entries; Palace "
                "accepts exactly one active entry per face."
            )
        if active[0].index != min(entry.index for entry in group):
            raise ValueError(
                f"The active entry of wave port {name!r} (Index "
                f"{active[0].index}) must have the lowest Index of its face; "
                "Palace refuses an active port whose boundary attribute is "
                "already assigned to an earlier entry."
            )
        modes = [entry.mode for entry in group]
        if len(set(modes)) != len(modes):
            raise ValueError(f"Wave port {name!r} lists a mode twice: {sorted(modes)}.")


def check_wave_port_modes(
    *,
    mode: int,
    modes: Sequence[int],
    active_mode: int | None,
    pilot_paths: Mapping[int, Sequence[Sequence[float]]] | None,
    active_name: str = "active_mode",
) -> None:
    """Refuse an inconsistent mode list, active mode or set of pilot paths.

    One rule set for both ways to declare a wave port: the ``WavePortConfig``
    model and the metadata ``configure_wave_port`` writes on a component port.

    Args:
        mode: The mode the port excites; with ``modes`` it is the active one.
        modes: The Palace modes followed on the face (empty: an ordinary port).
        active_mode: The active mode as declared, or None when not given.
        pilot_paths: ``{mode: points}`` of ``(x, y, z)`` um points, or None.
        active_name: How the messages call the active mode.

    Raises:
        ValueError: For a mode listed twice, a mode below 1, a missing or
            unlisted active mode, an active mode that differs from ``mode``,
            or a pilot path for a mode the port does not follow, with fewer
            than two points, with a point that is not a finite ``(x, y, z)``
            or with two equal points in a row.
    """
    if modes:
        if len(set(modes)) != len(modes):
            raise ValueError(f"modes lists a mode twice: {list(modes)}")
        if min(modes) < 1:
            raise ValueError("modes are 1-based Palace mode numbers")
        if active_mode is None:
            raise ValueError(f"{active_name} is required with modes")
        if active_mode not in modes:
            raise ValueError(
                f"{active_name} {active_mode} is not in modes {list(modes)}"
            )
    if active_mode is not None and mode != active_mode:
        raise ValueError(
            f"mode ({mode}) and {active_name} ({active_mode}) "
            "must agree: the active entry follows the active mode"
        )
    known = tuple(modes) or (mode,)
    for key, path in (pilot_paths or {}).items():
        if key not in known:
            raise ValueError(
                f"pilot_paths has mode {key}, which is not one of {list(known)}"
            )
        if len(path) < 2:
            raise ValueError(f"pilot path of mode {key} needs 2+ points")
        for point in path:
            if len(point) != 3 or not all(math.isfinite(c) for c in point):
                raise ValueError(
                    f"pilot path of mode {key}: points are finite "
                    f"(x, y, z) in um, got {point}"
                )
        for first, second in pairwise(path):
            if first == second:
                raise ValueError(
                    f"pilot path of mode {key} repeats the point {first}; "
                    "Palace aborts on a zero-length segment"
                )


def faces_by_index(entries: Iterable[PortModeEntry]) -> dict[int, str]:
    """Map each Palace ``Index`` to the name of the face (gsim port) it is on."""
    return {entry.index: entry.port for entry in entries}


def write_port_modes(directory: str | Path, entries: Sequence[PortModeEntry]) -> Path:
    """Write the sidecar ``port_modes.json`` into ``directory``.

    Args:
        directory: Folder holding ``config.json``.
        entries: The entries of every multimode port, in ``Index`` order.

    Returns:
        The path of the sidecar.

    Raises:
        ValueError: If the entries fail :func:`check_port_mode_entries`.
    """
    check_port_mode_entries(entries)
    path = Path(directory) / PORT_MODES_FILENAME
    payload = {
        "entries": [
            {**asdict(entry), "attributes": list(entry.attributes)}
            for entry in sorted(entries, key=lambda entry: entry.index)
        ]
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=4)
    return path


def read_port_modes(source: str | Path) -> tuple[PortModeEntry, ...]:
    """Read a sidecar written by :func:`write_port_modes`.

    Args:
        source: The sidecar file, or the folder that holds it.

    Returns:
        The entries in ``Index`` order.

    Raises:
        FileNotFoundError: If there is no sidecar.
        ValueError: If the file does not hold a list of entries or they fail
            :func:`check_port_mode_entries`.
    """
    path = Path(source)
    if path.is_dir():
        path = path / PORT_MODES_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No {PORT_MODES_FILENAME} at {path}")
    with path.open() as handle:
        payload = json.load(handle)
    rows = payload.get("entries") if isinstance(payload, dict) else None
    if rows is None or not isinstance(rows, list):
        raise ValueError(f"{path} has no 'entries' list.")
    try:
        entries = tuple(
            PortModeEntry(
                index=int(row["index"]),
                port=str(row["port"]),
                portnumber=int(row["portnumber"]),
                mode=int(row["mode"]),
                active=bool(row["active"]),
                excited=bool(row["excited"]),
                attributes=tuple(int(a) for a in row.get("attributes", ())),
            )
            for row in rows
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} has a malformed entry: {error!r}") from error
    check_port_mode_entries(entries)
    return tuple(sorted(entries, key=lambda entry: entry.index))


def same_face_pairs(
    faces: Iterable[tuple[str, tuple[float, float], float]],
    *,
    tol_um: float = FACE_POSITION_TOL_UM,
) -> list[tuple[str, str]]:
    """Return the pairs of ports that sit on one face.

    Args:
        faces: ``(name, (x, y) centre in um, orientation in degrees)`` per
            port.
        tol_um: Distance below which two centres are the same place.

    Returns:
        The ``(name, name)`` pairs whose centres agree and whose orientations
        agree modulo 180 degrees. A face has no front for this purpose: ports
        that face opposite ways at one place (0 and 180, or 90 and 270
        degrees) build the same rectangle. A multimode port puts all its
        modes on one face with ``modes=``, so two ports there would be two
        surfaces on one place.
    """
    items = list(faces)
    pairs: list[tuple[str, str]] = []
    for i, (name_a, centre_a, angle_a) in enumerate(items):
        for name_b, centre_b, angle_b in items[i + 1 :]:
            turn = (angle_a - angle_b) % 180.0
            aligned = min(turn, 180.0 - turn) < 1e-6
            apart = math.dist(centre_a, centre_b)
            if aligned and apart <= tol_um:
                pairs.append((name_a, name_b))
    return pairs
