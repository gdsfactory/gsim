"""Modal S-parameters of the driven runs.

:func:`load_modal_run` reads ``port-S.csv`` by Palace index, decodes the extra
entries of a multimode port with the sidecar ``port_modes.json`` and the wave
numbers from the log; :func:`modal_s_from_runs` assembles one modal S from a
single run or from the active-mode cycling runs. :class:`LumpedTerminal` marks a
lumped port that is passed through to the terminal S unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gsim.palace._validation_log import port_mode_samples
from gsim.palace._validation_port_modes import FACE_MODE_RTOL, SPEED_OF_LIGHT_M_S
from gsim.palace.ports.modes import (
    PORT_MODES_FILENAME,
    check_port_mode_entries,
    read_port_modes,
)
from gsim.palace.results import (
    SParam,
    _numeric_csv_column,
    _parse_sparam_col,
    _resolve_source,
)
from gsim.palace.terminal import degenerate_mode_groups

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from gsim.palace._validation_models import ModeSample
    from gsim.palace.ports.modes import PortModeEntry


#: Largest relative difference between the frequency axes of runs that are
#: combined into one modal S.
FREQ_RTOL: float = 1e-6


#: Largest relative distance between the frequency of a logged mode solve and
#: a frequency of ``port-S.csv`` for the two to be the same point (the log
#: rounds the frequency, as ``ModeExpectation.frequency_rtol`` assumes).
LOG_FREQ_RTOL: float = 5e-4


_PORT_S = "port-S.csv"


class ModalPortMode(NamedTuple):
    """One row (and column) of a modal S-parameter matrix.

    Attributes:
        port: Name of the gsim port, i.e. of the face (the name of a lumped
            terminal for a lumped port).
        portnumber: Palace number of the port: the ``Index`` of its active
            entry, which orders the physical ports.
        mode: The Palace ``Mode`` of the entry; ``0`` for a lumped port, which
            has no modes.
    """

    port: str
    portnumber: int
    mode: int


@dataclass(frozen=True)
class LumpedTerminal:
    """A lumped port that is passed through to the terminal S unchanged.

    A lumped port is one terminal with one mode whose voltage and current are
    defined by its own reference resistance, so its block of ``T_V`` and
    ``T_I`` is ``sqrt(R)`` and ``1 / sqrt(R)`` and the terminal reference
    impedance is ``R``. That makes ``A = 1`` and ``B = 0`` for the port: it
    keeps the S-parameters Palace reported, referred to ``R``.

    Attributes:
        name: Name of the terminal in the output network.
        index: The Palace port ``Index`` (the number in ``S[i][j]``).
        resistance: Reference resistance of the port in ohm, the value Palace
            used for its S-parameters.
    """

    name: str
    index: int
    resistance: float = 50.0

    def __post_init__(self) -> None:
        """Validate the name, index and resistance.

        Raises:
            ValueError: For an empty name, an index below 1 or a resistance
                that is not finite and positive.
        """
        if not self.name:
            raise ValueError("A lumped terminal needs a name.")
        if int(self.index) != self.index or self.index < 1:
            raise ValueError(
                f"Lumped terminal {self.name!r}: the Palace index must be an "
                f"integer >= 1, got {self.index!r}."
            )
        if not (np.isfinite(self.resistance) and self.resistance > 0.0):
            raise ValueError(
                f"Lumped terminal {self.name!r}: the resistance must be finite "
                f"and positive, got {self.resistance!r}."
            )


@dataclass(frozen=True, eq=False)
class ModalRun:
    """The S-parameters of one driven run, keyed by Palace index.

    Attributes:
        freq_ghz: Frequencies of ``port-S.csv`` in GHz.
        s: ``{(i, j): S[i][j] vs frequency}``: the response at Palace port
            index ``i`` to the excitation of index ``j``. Only the excited
            columns are present.
        entries: The ``port_modes.json`` entries of the run (which port, mode
            and active flag each Palace index stands for).
        samples: The port mode solves printed in the run's log, if it was
            read; they carry the wave number of every entry.
        label: A name for messages.
    """

    freq_ghz: NDArray[np.float64]
    s: Mapping[tuple[int, int], NDArray[np.complex128]]
    entries: tuple[PortModeEntry, ...]
    samples: tuple[ModeSample, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        """Coerce the arrays and check the entries.

        Raises:
            ValueError: When the frequencies are not a non-empty finite 1-D
                array, a column does not have one value per frequency, or the
                entries are not a valid set (see ``check_port_mode_entries``).
        """
        freq = np.asarray(self.freq_ghz, dtype=float)
        if freq.ndim != 1 or freq.size == 0 or not np.all(np.isfinite(freq)):
            raise ValueError(
                f"Run {self.label!r}: the frequencies must be a non-empty finite "
                "1-D array."
            )
        columns: dict[tuple[int, int], NDArray[np.complex128]] = {}
        for key, values in self.s.items():
            array = np.asarray(values, dtype=np.complex128)
            if array.shape != freq.shape:
                raise ValueError(
                    f"Run {self.label!r}: S{list(key)} has shape {array.shape}, "
                    f"expected {freq.shape} (one value per frequency)."
                )
            columns[(int(key[0]), int(key[1]))] = array
        entries = tuple(self.entries)
        check_port_mode_entries(entries)
        object.__setattr__(self, "freq_ghz", freq)
        object.__setattr__(self, "s", columns)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "samples", tuple(self.samples))


def _find_sidecar(base: Path, csv_path: Path) -> Path | None:
    """Search the places a run folder may keep ``port_modes.json``."""
    candidates = [
        csv_path.parent / PORT_MODES_FILENAME,
        base / PORT_MODES_FILENAME,
        base / "output" / PORT_MODES_FILENAME,
        csv_path.parent.parent / "input" / PORT_MODES_FILENAME,
        csv_path.parent.parent / PORT_MODES_FILENAME,
        csv_path.parent.parent.parent / PORT_MODES_FILENAME,
    ]
    return next((path for path in candidates if path.exists()), None)


def _read_port_s(
    csv_path: Path,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], NDArray[np.complex128]]]:
    """Read ``port-S.csv`` by Palace index, with no port names in between.

    ``load_sparams`` names the columns from ``port_information.json``, which
    knows the ports but not the extra entries of a multimode port. The
    headers carry the Palace index, so they are read directly.

    Raises:
        ValueError: If there is no frequency column or a magnitude column has
            no phase column (or the other way round).
    """
    import pandas as pd

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    frame.columns = frame.columns.str.strip()
    freq_col = next((c for c in frame.columns if c.startswith("f")), None)
    if freq_col is None:
        raise ValueError(f"{csv_path} has no frequency column.")
    freq = _numeric_csv_column(frame[freq_col], csv_path, freq_col)
    raw: dict[tuple[int, int], dict[str, NDArray[np.float64]]] = {}
    for col in frame.columns:
        parsed = _parse_sparam_col(col)
        if parsed is None:
            continue
        i, j, kind = parsed
        raw.setdefault((i, j), {})[kind] = _numeric_csv_column(
            frame[col], csv_path, col, allow_negative_infinity=kind == "db"
        )
    s: dict[tuple[int, int], NDArray[np.complex128]] = {}
    for (i, j), parts in sorted(raw.items()):
        if "db" not in parts or "deg" not in parts:
            raise ValueError(
                f"{csv_path}: S[{i}][{j}] needs both a magnitude and a phase column."
            )
        s[(i, j)] = np.asarray(
            SParam(db=parts["db"], deg=parts["deg"]).complex, dtype=np.complex128
        )
    return freq, s


def load_modal_run(
    source: str | Path | dict,
    *,
    sidecar: str | Path | None = None,
    log: str | Path | Sequence[str] | None = None,
) -> ModalRun:
    """Read one driven run: ``port-S.csv``, the sidecar and, optionally, the log.

    Args:
        source: The run folder (the simulation folder, ``output/palace`` or a
            downloaded result folder), a ``port-S.csv`` file, or the results
            dict of ``sim.run()``.
        sidecar: The ``port_modes.json`` of the run. By default it is looked
            for next to ``port-S.csv`` and in the usual input folders.
        log: The Palace log of the run (a file, or its lines). It carries the
            wave number of every port mode, which the gauge alignment pairs the
            2D modes with; without it :attr:`ModalS.wave_numbers` is ``None``.

    Returns:
        The run.

    Raises:
        FileNotFoundError: When there is no ``port-S.csv`` or no sidecar.
        ValueError: When a file is malformed.
    """
    if isinstance(source, Mapping):
        csv_path, base = _resolve_source(dict(source))
    else:
        path = Path(source)
        if path.is_file():
            csv_path, base = path, path.parent
        else:
            csv_path, base = _resolve_source(path)
    if csv_path is None:
        raise FileNotFoundError(f"{_PORT_S} not found in {source!r}")
    if sidecar is not None:
        sidecar_path: Path | None = Path(sidecar)
    else:
        sidecar_path = _find_sidecar(base, csv_path)
    if sidecar_path is None or not sidecar_path.exists():
        raise FileNotFoundError(
            f"No {PORT_MODES_FILENAME} found for {csv_path}. It is written next "
            "to config.json by a simulation with a multimode wave port; pass "
            "sidecar= if it was moved."
        )
    entries = read_port_modes(sidecar_path)
    freq, s = _read_port_s(csv_path)
    samples: tuple[ModeSample, ...] = ()
    if log is not None:
        if isinstance(log, str | Path):
            lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
        else:
            lines = list(log)
        samples = port_mode_samples(lines)
    return ModalRun(
        freq_ghz=freq,
        s=s,
        entries=entries,
        samples=samples,
        label=str(csv_path.parent),
    )


@dataclass(frozen=True)
class DegenerateFlag:
    """Modes of one face whose wave numbers coincide at some frequencies.

    Attributes:
        port: The face.
        modes: The Palace mode numbers that are within the pairing tolerance of
            each other.
        freq_ghz: The frequencies at which they are.
    """

    port: str
    modes: tuple[int, ...]
    freq_ghz: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class ModalS:
    """The modal S-parameters assembled from one or several runs.

    Attributes:
        freq_ghz: Frequencies in GHz.
        s_modal: ``(nf, M, M)`` modal S, ordered as ``order`` (port by port in
            Palace port order, modes in ascending order inside a port), in
            the gauge Palace used for each (face, mode).
        order: The row and column labels.
        source_run: Index of the run each column was taken from.
        exact: Whether each column comes from a run in which its entry was the
            active one. The other columns were measured with the face
            terminated for a different mode: the entry is excited and measured
            as in the active case, but the face reflects part of it, and so
            do the other modes arriving there.
        wave_numbers: ``(nf, M)`` complex wave numbers (1/m) of the port modes
            from the run logs, NaN where no solve was logged and for lumped
            ports; ``None`` if no log was read.
        degenerate: Modes of a face whose wave numbers coincide.
    """

    freq_ghz: NDArray[np.float64]
    s_modal: NDArray[np.complex128]
    order: tuple[ModalPortMode, ...]
    source_run: tuple[int, ...]
    exact: tuple[bool, ...]
    wave_numbers: NDArray[np.complex128] | None
    degenerate: tuple[DegenerateFlag, ...]

    def port_columns(self, port: str) -> list[int]:
        """The rows (and columns) of ``s_modal`` that belong to a port, in order."""
        return [i for i, item in enumerate(self.order) if item.port == port]


def _as_run(run: ModalRun | str | Path | dict) -> ModalRun:
    """Accept a loaded run or anything :func:`load_modal_run` takes."""
    return run if isinstance(run, ModalRun) else load_modal_run(run)


def _layout(runs: Sequence[ModalRun]) -> dict[int, tuple[str, tuple[int, ...]]]:
    """``{port number: (name, modes)}``, the same in every run.

    Raises:
        ValueError: When two runs do not have the same ports and modes.
    """
    layout: dict[int, tuple[str, tuple[int, ...]]] | None = None
    for run in runs:
        this: dict[int, tuple[str, list[int]]] = {}
        for entry in run.entries:
            name, modes = this.setdefault(entry.portnumber, (entry.port, []))
            if name != entry.port:
                raise ValueError(
                    f"Run {run.label!r}: port number {entry.portnumber} is named "
                    f"{name!r} and {entry.port!r}."
                )
            modes.append(entry.mode)
        found = {
            number: (name, tuple(sorted(modes)))
            for number, (name, modes) in this.items()
        }
        if layout is None:
            layout = found
        elif found != layout:
            raise ValueError(
                f"Run {run.label!r} has the ports and modes {found}, but the first "
                f"run has {layout}: the runs must be the same simulation."
            )
    return layout or {}


def _choose_source(
    runs: Sequence[ModalRun],
    column: ModalPortMode,
    lumped_index: int | None,
) -> tuple[int, int, bool]:
    """Pick the run and Palace excitation index that supply a modal column.

    A multimode entry is taken from the run in which it is the active entry
    (the face is terminated for exactly that mode); if there is none, from the
    first run in which it is excited. A lumped port is taken from the first run
    that has a column for it.

    Returns:
        ``(run position, Palace index of the excitation, exact)``.

    Raises:
        ValueError: When no run excites the column, or two runs both have the
            entry as the active one.
    """
    if lumped_index is not None:
        for k, run in enumerate(runs):
            if any(j == lumped_index for _, j in run.s):
                terminated = all(entry.active for entry in run.entries)
                return k, lumped_index, terminated
        raise ValueError(
            f"No run has an S column for the lumped port {column.port!r} "
            f"(Palace index {lumped_index})."
        )
    active: list[tuple[int, int]] = []
    excited: list[tuple[int, int]] = []
    for k, run in enumerate(runs):
        for entry in run.entries:
            if (
                entry.port == column.port
                and entry.mode == column.mode
                and entry.excited
            ):
                (active if entry.active else excited).append((k, entry.index))
    if len(active) > 1:
        labels = [runs[k].label for k, _ in active]
        raise ValueError(
            f"Mode {column.mode} of port {column.port!r} is the active mode in "
            f"{len(active)} runs ({labels}); give each mode one run."
        )
    if active:
        return active[0][0], active[0][1], True
    if excited:
        return excited[0][0], excited[0][1], False
    raise ValueError(
        f"No run excites mode {column.mode} of port {column.port!r}: its modal "
        "column is missing."
    )


def _k0(freq_ghz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Free-space wave number in 1/m at frequencies in GHz."""
    return np.asarray(2.0 * np.pi * freq_ghz * 1e9 / SPEED_OF_LIGHT_M_S, dtype=float)


def _wave_numbers(
    runs: Sequence[ModalRun],
    order: Sequence[ModalPortMode],
    freq_ghz: NDArray[np.float64],
) -> NDArray[np.complex128] | None:
    """The logged wave number of every (port, mode) at every frequency.

    The log's ``Port <index>`` is the Palace index of an entry, which differs
    from run to run for the observation entries, so each run decodes its own
    samples with its own sidecar. Repeated solves (one per excitation) are
    reduced to their median.
    """
    series: dict[tuple[str, int], list[tuple[float, complex]]] = {}
    for run in runs:
        keys = {entry.index: (entry.port, entry.mode) for entry in run.entries}
        for sample in run.samples:
            key = keys.get(sample.port)
            frequency = sample.frequency_ghz
            if key is None or frequency is None or not np.isfinite(frequency):
                continue
            series.setdefault(key, []).append((float(frequency), sample.wave_number))
    if not series:
        return None
    out = np.full((freq_ghz.size, len(order)), np.nan + 0j, dtype=np.complex128)
    for c, column in enumerate(order):
        points = series.get((column.port, column.mode))
        if not points:
            continue
        logged = np.array([f for f, _ in points])
        values = np.array([k for _, k in points])
        for f_index, frequency in enumerate(freq_ghz):
            near = np.abs(logged - frequency) <= LOG_FREQ_RTOL * abs(frequency)
            if near.any():
                out[f_index, c] = complex(
                    float(np.median(values[near].real)),
                    float(np.median(values[near].imag)),
                )
    return out


def _degenerate_flags(
    order: Sequence[ModalPortMode],
    wave_numbers: NDArray[np.complex128] | None,
    freq_ghz: NDArray[np.float64],
) -> tuple[DegenerateFlag, ...]:
    """Groups of modes of one face whose ``n_eff`` coincide, per frequency."""
    if wave_numbers is None:
        return ()
    found: dict[tuple[str, tuple[int, ...]], list[float]] = {}
    ports = dict.fromkeys(item.port for item in order if item.mode > 0)
    k0 = _k0(freq_ghz)
    for port in ports:
        cols = [c for c, item in enumerate(order) if item.port == port]
        modes = [order[c].mode for c in cols]
        for f_index, frequency in enumerate(freq_ghz):
            n_eff = wave_numbers[f_index, cols].real / k0[f_index]
            if len(cols) < 2 or not np.all(np.isfinite(n_eff)):
                continue
            for group in degenerate_mode_groups(n_eff, rtol=FACE_MODE_RTOL):
                if len(group) > 1:
                    key = (port, tuple(modes[g] for g in group))
                    found.setdefault(key, []).append(float(frequency))
    return tuple(
        DegenerateFlag(port=port, modes=modes, freq_ghz=tuple(freqs))
        for (port, modes), freqs in found.items()
    )


def modal_s_from_runs(
    runs: Sequence[ModalRun | str | Path | dict],
    *,
    lumped: Sequence[LumpedTerminal] = (),
) -> ModalS:
    """Assemble the modal S-parameters of all faces from one run or several.

    Palace measures every wave-port entry, but only the active entry of a face
    terminates it. The recommended way (active-mode cycling) therefore runs once
    per mode ``k``, with mode ``k`` active on every face and only the mode-``k``
    entries excited. Column ``(face p, mode k)`` of the modal S then comes from
    run ``k`` and has every entry of every face as its rows. A single run with
    mode 1 active and every entry excited gives the same matrix with the
    columns of modes 2 and up approximate.

    The rule per column: take it from the run in which its entry is active, else
    from the first run that excites it. :attr:`ModalS.exact` says which.

    Args:
        runs: Loaded runs (see :func:`load_modal_run`), or what it accepts.
        lumped: Lumped ports of the simulation that are part of the terminal
            network (their Palace index is the port number).

    Returns:
        The assembled matrix, ordered port by port in Palace port number,
        modes ascending inside a port (the order of
        :func:`gsim.palace.terminal.modal_to_terminal_s`).

    Raises:
        ValueError: When no run is given, the runs disagree on their ports,
            modes or frequencies, a lumped port clashes with a multimode port,
            a column is supplied by no run or by two active runs, or a needed
            S entry is absent from the file.
    """
    loaded = [_as_run(run) for run in runs]
    if not loaded:
        raise ValueError("modal_s_from_runs needs at least one run.")
    freq = loaded[0].freq_ghz
    for run in loaded[1:]:
        if run.freq_ghz.shape != freq.shape or not np.allclose(
            run.freq_ghz, freq, rtol=FREQ_RTOL, atol=0.0
        ):
            raise ValueError(
                f"Run {run.label!r} has a different frequency axis from run "
                f"{loaded[0].label!r}; the runs must share one sweep."
            )
    layout = _layout(loaded)
    taken_names = {name for name, _ in layout.values()}
    if len({item.name for item in lumped}) != len(lumped) or any(
        item.name in taken_names for item in lumped
    ):
        raise ValueError(
            "Lumped terminal names must be unique and different from the names "
            "of the multimode ports."
        )
    taken_indices = {entry.index for run in loaded for entry in run.entries}
    if len({item.index for item in lumped}) != len(lumped) or any(
        item.index in taken_indices for item in lumped
    ):
        raise ValueError(
            "Lumped terminal Palace indices must be unique and different from "
            "the indices of the multimode entries."
        )

    columns: list[ModalPortMode] = [
        ModalPortMode(name, number, mode)
        for number, (name, modes) in layout.items()
        for mode in modes
    ]
    lumped_index: dict[str, int] = {}
    for item in lumped:
        columns.append(ModalPortMode(item.name, item.index, 0))
        lumped_index[item.name] = item.index
    order = tuple(sorted(columns, key=lambda c: (c.portnumber, c.mode)))

    row_index: list[dict[tuple[str, int], int]] = []
    for run in loaded:
        rows = {(entry.port, entry.mode): entry.index for entry in run.entries}
        rows.update({(name, 0): index for name, index in lumped_index.items()})
        row_index.append(rows)

    size = len(order)
    s_modal = np.empty((freq.size, size, size), dtype=np.complex128)
    chosen = [
        _choose_source(
            loaded, column, lumped_index.get(column.port) if column.mode == 0 else None
        )
        for column in order
    ]
    source = [k for k, _, _ in chosen]
    exact = [is_exact for _, _, is_exact in chosen]
    for c, (k, j, _) in enumerate(chosen):
        column = order[c]
        run = loaded[k]
        for r, row in enumerate(order):
            i = row_index[k][(row.port, row.mode)]
            values = run.s.get((i, j))
            if values is None:
                raise ValueError(
                    f"Run {run.label!r} has no S[{i}][{j}] (response of "
                    f"{row.port!r} mode {row.mode} to the excitation of "
                    f"{column.port!r} mode {column.mode})."
                )
            s_modal[:, r, c] = values
    wave_numbers = _wave_numbers(loaded, order, freq)
    return ModalS(
        freq_ghz=freq,
        s_modal=s_modal,
        order=order,
        source_run=tuple(source),
        exact=tuple(exact),
        wave_numbers=wave_numbers,
        degenerate=_degenerate_flags(order, wave_numbers, freq),
    )
