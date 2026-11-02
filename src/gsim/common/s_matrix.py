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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_SPEED_OF_LIGHT_UM_HZ = 299792458.0e6
_BAND_EDGE_TOLERANCE_UM = 1e-9


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


def assemble_s_matrix(runs: Sequence[SMatrixRun]) -> SMatrix:
    """Combine solver runs, each one S-matrix column, into an :class:`SMatrix`."""
    ports = tuple(runs[0].ports)
    index = {port: i for i, port in enumerate(ports)}
    order = np.argsort(runs[0].wavelength_um)
    wavelength_um = np.asarray(runs[0].wavelength_um, dtype=float)[order]
    s = np.zeros((wavelength_um.size, len(ports), len(ports)), dtype=complex)
    measured = np.zeros((len(ports), len(ports)), dtype=bool)
    for run in runs:
        run_order = np.argsort(run.wavelength_um)
        for (receive, source), values in run.entries.items():
            s[:, index[receive], index[source]] = np.asarray(values)[run_order]
            measured[index[receive], index[source]] = True
    return SMatrix(wavelength_um, ports, s, measured)
