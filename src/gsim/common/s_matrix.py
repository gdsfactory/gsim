"""Solver-independent S-matrix: export to a SAX model and to Touchstone.

Each solver builder (``gsim.fdtd.s_matrix``) turns its results into one
:class:`SMatrixRun` per run and hands them to :func:`assemble_s_matrix`, which
applies the shared refusals and assumptions and returns an :class:`SMatrix`.

Inside gsim an S-matrix is stored as ``s[f, receive, source]`` with time
convention exp(+jwt). Each boundary converts on its own side:

- a builder converts its solver's convention to exp(+jwt) on entry;
- :meth:`SMatrix.sax_model` conjugates (SAX phase is exp(+j beta L), that is
  exp(-iwt)) and keys entries ``(in, out)``;
- :meth:`SMatrix.write_touchstone` keeps exp(+jwt), as every other Touchstone file
  gsim writes.

``skrf`` is imported only when a file is written. ``sax`` and ``jax`` are never
imported: the model is a plain function following the SAX calling convention.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SPEED_OF_LIGHT_UM_HZ = 299792458.0e6
_BAND_EDGE_TOLERANCE_UM = 1e-9
_GRID_RTOL = 1e-9


@dataclass(frozen=True)
class SMatrixRun:
    """One solver run's measured S-parameters, ready for :func:`assemble_s_matrix`.

    Attributes:
        label: Names the run in error messages.
        wavelength_um: Sample wavelengths [um], in any order.
        ports: The run's port names.
        entries: Samples on ``wavelength_um`` by ``(receive, source)``, already in
            time convention exp(+jwt).
        valid: False for samples the solver flags as unusable.
    """

    label: str
    wavelength_um: np.ndarray
    ports: Sequence[str]
    entries: Mapping[tuple[str, str], np.ndarray]
    valid: np.ndarray


@dataclass(frozen=True, eq=False)
class SMatrix:
    """An n-port S-matrix over a wavelength grid, in time convention exp(+jwt).

    Attributes:
        wavelength_um: Strictly ascending sample wavelengths [um].
        ports: Port names, giving the order of the matrix indices.
        s: Complex array ``nf x n x n``; ``s[f, i, j]`` is S(receive=i, source=j).
        measured: Boolean array ``n x n``, False where an entry was assumed rather
            than measured.
    """

    wavelength_um: np.ndarray
    ports: tuple[str, ...]
    s: np.ndarray
    measured: np.ndarray

    def __post_init__(self) -> None:
        """Hold the stated invariants: every method relies on them."""
        wavelength_um = np.asarray(self.wavelength_um, dtype=float)
        ports = _unique_ports(self.ports)
        s = np.asarray(self.s, dtype=complex)
        measured = np.asarray(self.measured, dtype=bool)
        if (
            wavelength_um.ndim != 1
            or wavelength_um.size == 0
            or not (wavelength_um[0] > 0 and np.all(np.diff(wavelength_um) > 0))
        ):
            raise ValueError(
                "wavelength_um must be positive and strictly ascending, got "
                f"{wavelength_um}."
            )
        n_ports = len(ports)
        if s.shape != (wavelength_um.size, n_ports, n_ports):
            raise ValueError(
                f"s has shape {s.shape}, expected "
                f"{(wavelength_um.size, n_ports, n_ports)}."
            )
        if measured.shape != (n_ports, n_ports):
            raise ValueError(
                f"measured has shape {measured.shape}, expected {(n_ports, n_ports)}."
            )
        object.__setattr__(self, "wavelength_um", wavelength_um)
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "measured", measured)

    @property
    def frequency_hz(self) -> np.ndarray:
        """Return the sample frequencies c/wavelength [Hz], descending."""
        return _SPEED_OF_LIGHT_UM_HZ / self.wavelength_um

    def sax_model(self) -> Callable[..., dict[tuple[str, str], np.ndarray]]:
        """Return the band as a SAX model ``model(*, wl=<band midpoint>)``.

        The model keys every entry ``(in, out)``, so ``model()[("o1", "o2")]`` is
        S(o2,o1), and returns the conjugate of the stored exp(+jwt) values, which
        is SAX's exp(+j beta L) phase. This is the supported route into SAX:
        ``sax.parsers.parse_touchstone`` applies neither conversion.

        Magnitude and unwrapped phase are interpolated linearly in frequency, since
        the phase of a delay is linear in frequency while its real and imaginary
        parts collapse between samples. Output shape follows ``wl`` and a scalar
        stays 0-d.

        The model works in ``sax.get_settings`` and ``sax.circuit``. It computes
        with plain numpy, so it cannot be traced by ``jax.jit``.

        Returns:
            The model. It holds its own copy of the data, so later changes to this
            matrix do not reach it, and it raises ``ValueError`` for a wavelength
            more than 1e-9 um outside the band.
        """
        frequency_hz = self.frequency_hz[::-1].copy()
        stored = self.s[::-1]
        magnitude = np.abs(stored)
        phase = np.unwrap(np.angle(stored), axis=0)
        ports = self.ports
        wl_min = float(self.wavelength_um[0])
        wl_max = float(self.wavelength_um[-1])

        def model(
            *, wl: float | np.ndarray = (wl_min + wl_max) / 2
        ) -> dict[tuple[str, str], np.ndarray]:
            wavelength = np.asarray(wl, dtype=float)
            inside = (wavelength >= wl_min - _BAND_EDGE_TOLERANCE_UM) & (
                wavelength <= wl_max + _BAND_EDGE_TOLERANCE_UM
            )
            if not np.all(inside):
                raise ValueError(
                    f"wl={wl} is outside the simulated band "
                    f"[{wl_min}, {wl_max}] um; a simulated band is not a model "
                    "outside it."
                )
            frequency = _SPEED_OF_LIGHT_UM_HZ / wavelength
            return {
                (port_in, port_out): np.asarray(
                    np.interp(frequency, frequency_hz, magnitude[:, i_out, i_in])
                    * np.exp(
                        -1j * np.interp(frequency, frequency_hz, phase[:, i_out, i_in])
                    )
                )
                for i_in, port_in in enumerate(ports)
                for i_out, port_out in enumerate(ports)
            }

        return model

    def write_touchstone(self, path: str | Path, *, z0: float = 50.0) -> Path:
        """Write the matrix to a Touchstone file (``.sNp``), ascending in frequency.

        The file keeps gsim's time convention exp(+jwt), the RF-tool reading, and
        ``sax.parsers.parse_touchstone`` keeps the file's sign and reads ``s[i, j]``
        as ``(in, out)``, so use :meth:`sax_model` to reach SAX. Port names are
        embedded as ``! Port[i]`` comments and values are written as real and
        imaginary parts with 16-digit precision.

        Args:
            path: Destination path. ``.sNp`` is appended when the name has no such
                suffix, so ``run_1.5`` becomes ``run_1.5.s2p``.
            z0: Nominal reference impedance [Ohm]. S is power-wave modal, normalised
                per mode, so ``z0`` only fills the file header.

        Returns:
            The path written.

        Raises:
            ValueError: If the suffix of ``path`` names a different port count.
        """
        n_ports = len(self.ports)
        path = Path(path)
        if not re.fullmatch(r"\.s\d+p", path.suffix.lower()):
            # Append rather than with_suffix: "run_1.5" must keep its ".5".
            path = Path(f"{path}.s{n_ports}p")
        elif path.suffix.lower() != f".s{n_ports}p":
            raise ValueError(
                f"Touchstone suffix {path.suffix!r} does not match the {n_ports} ports "
                f"of this matrix; use '.s{n_ports}p'."
            )

        import skrf as rf

        network = rf.Network(
            f=self.frequency_hz[::-1], s=self.s[::-1], z0=z0, f_unit="Hz"
        )
        network.port_names = list(self.ports)
        fmt = "{:.16e}"
        network.write_touchstone(
            path,
            form="ri",
            format_spec_A=fmt,
            format_spec_B=fmt,
            format_spec_freq=fmt,
            write_z0=False,
        )
        logger.info("Touchstone written to %s", path)
        return path


def assemble_s_matrix(
    runs: Sequence[SMatrixRun],
    *,
    assume_reciprocal: bool = False,
    assume_unmeasured_zero: bool = False,
) -> SMatrix:
    """Combine solver runs, each one S-matrix column, into an :class:`SMatrix`.

    A sample is dropped, with a WARNING giving the count, when any run flags it
    invalid or any entry is non-finite there. Entries are filled from assumptions
    only when asked, and each one is logged at INFO and marked False in
    ``measured``. Filling is opt-in because SAX reads an absent key as zero, so a
    one-run model would otherwise claim zero reflection and zero backward
    transmission.

    Args:
        runs: The runs, on one port list and one wavelength grid.
        assume_reciprocal: Copy a missing S(a,b) from a measured S(b,a).
        assume_unmeasured_zero: Set entries that are still missing to zero.

    Returns:
        The matrix, on ascending wavelength.

    Raises:
        ValueError: If there are no runs; a run has no S-parameters; port lists or
            wavelength grids differ (``rtol=1e-9`` after sorting); an entry appears
            twice, names an unknown port or has the wrong length; every sample is
            invalid; or an entry is missing and no flag fills it.
    """
    wavelength_um, ports, entries, keep = _merge_runs(list(runs))
    keep = keep & np.all([np.isfinite(v) for v in entries.values()], axis=0)
    if not keep.any():
        raise ValueError(
            f"Every one of the {keep.size} samples is invalid (below the noise floor "
            "or non-finite), so there is no S-matrix to build."
        )
    if not keep.all():
        logger.warning(
            "Dropped %d of %d samples flagged invalid or non-finite.",
            keep.size - np.count_nonzero(keep),
            keep.size,
        )

    n_ports = len(ports)
    s = np.zeros((np.count_nonzero(keep), n_ports, n_ports), dtype=complex)
    measured = np.zeros((n_ports, n_ports), dtype=bool)
    for (receive, source), values in entries.items():
        i, j = ports.index(receive), ports.index(source)
        s[:, i, j] = values[keep]
        measured[i, j] = True

    missing = []
    for i, receive in enumerate(ports):
        for j, source in enumerate(ports):
            if measured[i, j]:
                continue
            name = _entry_name(receive, source)
            if assume_reciprocal and measured[j, i]:
                s[:, i, j] = s[:, j, i]
                mirror = _entry_name(source, receive)
                logger.info("Assumed %s = measured %s (reciprocity).", name, mirror)
            elif assume_unmeasured_zero:
                logger.info("Assumed %s = 0 (unmeasured).", name)
            else:
                missing.append(name)
    if missing:
        raise ValueError(
            f"Missing S-parameters: {', '.join(missing)}. Run a source at the source "
            "ports of those entries, or state the assumption with "
            "assume_reciprocal=True (copies a measured mirror entry) and "
            "assume_unmeasured_zero=True (sets what is left to zero)."
        )
    matrix = SMatrix(wavelength_um[keep], ports, s, measured)
    _warn_if_not_causal(matrix)
    return matrix


def _warn_if_not_causal(matrix: SMatrix) -> None:
    """Warn when the strongest measured transmission has a non-positive delay.

    In exp(+jwt) a passive device delays its output, so
    ``tau = -d(unwrapped phase)/d(omega)`` is positive. The check is the one guard
    on a solver's assumed phase convention, so it only warns.
    """
    off_diagonal = matrix.measured & ~np.eye(len(matrix.ports), dtype=bool)
    if matrix.wavelength_um.size < 3 or not off_diagonal.any():
        return
    strength = np.where(off_diagonal, np.abs(matrix.s).sum(axis=0), -1.0)
    i, j = np.unravel_index(np.argmax(strength), strength.shape)
    trace = matrix.s[::-1, i, j]
    magnitude = np.abs(trace)
    weights = (magnitude[1:] + magnitude[:-1]) / 2
    if not weights.any():
        return
    omega = 2 * np.pi * matrix.frequency_hz[::-1]
    delays = -np.diff(np.unwrap(np.angle(trace))) / np.diff(omega)
    tau = float(np.average(delays, weights=weights))
    if tau <= 0:
        logger.warning(
            "%s has a non-positive group delay (%.3g s): the solver's phase convention "
            "differs from the assumed one, or the sampling is too coarse to unwrap "
            "phase.",
            _entry_name(matrix.ports[i], matrix.ports[j]),
            tau,
        )


def _entry_name(receive: str, source: str) -> str:
    """Format one entry as gsim names it, ``S(receive,source)``."""
    return f"S({receive},{source})"


def _unique_ports(ports: Sequence[str]) -> tuple[str, ...]:
    """Return ports as a tuple, refusing a name that would alias two indices."""
    names = tuple(ports)
    if len(set(names)) != len(names):
        raise ValueError(f"Port list {list(names)} has duplicate names.")
    return names


def _merge_runs(
    runs: list[SMatrixRun],
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    dict[tuple[str, str], np.ndarray],
    np.ndarray,
]:
    """Refuse runs that do not form one matrix and align them on ascending grid."""
    if not runs:
        raise ValueError("Got no results to build an S-matrix from.")
    ports = _unique_ports(runs[0].ports)
    reference_um = np.sort(np.asarray(runs[0].wavelength_um, dtype=float))
    keep = np.ones(reference_um.size, dtype=bool)
    entries: dict[tuple[str, str], np.ndarray] = {}
    for run in runs:
        if not run.entries:
            raise ValueError(f"{run.label} has no S-parameters.")
        if tuple(run.ports) != ports:
            raise ValueError(
                f"{run.label} has port list {list(run.ports)}, expected {list(ports)}."
            )
        wavelength_um = np.asarray(run.wavelength_um, dtype=float)
        order = np.argsort(wavelength_um)
        if wavelength_um.shape != reference_um.shape or not np.allclose(
            wavelength_um[order], reference_um, rtol=_GRID_RTOL, atol=0.0
        ):
            raise ValueError(
                f"{run.label} has a different wavelength grid from {runs[0].label}."
            )
        valid = np.asarray(run.valid, dtype=bool)
        if valid.shape != reference_um.shape:
            raise ValueError(f"{run.label} has {valid.size} validity flags.")
        keep &= valid[order]
        for key, values in run.entries.items():
            name = _entry_name(*key)
            unknown = [port for port in key if port not in ports]
            if unknown:
                raise ValueError(
                    f"{run.label}: {name} names {unknown}, not in port list "
                    f"{list(ports)}."
                )
            if key in entries:
                raise ValueError(f"{name} appears in more than one result.")
            samples = np.asarray(values, dtype=complex)
            if samples.shape != reference_um.shape:
                raise ValueError(
                    f"{run.label}: {name} has {samples.size} samples, expected "
                    f"{reference_um.size}."
                )
            entries[key] = samples[order]
    return reference_um, ports, entries, keep
