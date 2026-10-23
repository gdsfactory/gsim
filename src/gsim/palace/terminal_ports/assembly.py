"""Terminal S-parameters from the modal S of the runs and the 2D matrices.

:func:`build_terminal_s` joins the modal S (:mod:`gsim.palace.terminal_ports.modal`),
the 2D ``T_V`` and ``T_I`` (:mod:`gsim.palace.terminal_ports.modes2d`) and their
gauge alignment (:mod:`gsim.palace.terminal_ports.gauge`) and keeps all the pieces
in a :class:`TerminalSResult`; :func:`terminal_s` returns the network only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gsim.palace._validation_port_modes import FACE_MODE_RTOL
from gsim.palace.terminal import modal_to_terminal_s
from gsim.palace.terminal_ports.gauge import PAIRING_RTOL, AlignedPortModes, align_gauge
from gsim.palace.terminal_ports.modal import (
    LumpedTerminal,
    ModalRun,
    ModalS,
    modal_s_from_runs,
)
from gsim.palace.terminal_ports.modes2d import PortModes2D
from gsim.palace.terminal_ports.spec import TerminalPortSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray
    from skrf import Network


@dataclass(frozen=True, eq=False)
class TerminalSResult:
    """Terminal S-parameters of a simulation, with what they were built from.

    Attributes:
        freq_ghz: Frequencies in GHz.
        s: ``(nf, Nt, Nt)`` terminal S, ordered port by port (Palace port
            number), terminals in the order of each :class:`TerminalPortSpec`.
        port_names: ``"{port}:{terminal}"`` per terminal, the name of a lumped
            terminal for a lumped port.
        z_ref: ``(Nt,)`` real reference impedance of each terminal.
        modal: The assembled modal S.
        aligned: The aligned ``T_V``, ``T_I`` of every multimode port.
    """

    freq_ghz: NDArray[np.float64]
    s: NDArray[np.complex128]
    port_names: tuple[str, ...]
    z_ref: NDArray[np.float64]
    modal: ModalS
    aligned: dict[str, AlignedPortModes]

    def network(self, name: str | None = None) -> Network:
        """The terminal S as a scikit-rf network (one port per terminal).

        Args:
            name: Name of the network.

        Returns:
            The network, with the port names and per-port reference impedances.
        """
        import skrf as rf

        z0 = np.broadcast_to(self.z_ref, self.s.shape[:2]).copy()
        net = rf.Network(f=self.freq_ghz * 1e9, s=self.s, z0=z0, f_unit="Hz")
        net.name = name
        net.port_names = list(self.port_names)
        return net

    def write_touchstone(self, path: str | Path, *, name: str | None = None) -> Path:
        """Write the terminal S to a Touchstone ``.sNp`` file.

        The reference impedance of every port is written as a comment as well
        (``write_z0``), and the values with 16 digits so that a round trip
        keeps the S-parameters to far below 1e-9.

        Args:
            path: Destination; ``.sNp`` is appended when the suffix is missing.
            name: Name of the network.

        Returns:
            The file written.
        """
        target = Path(path)
        count = len(self.port_names)
        if target.suffix.lower() != f".s{count}p":
            target = Path(f"{target}.s{count}p")
        fmt = "{:.16e}"
        self.network(name).write_touchstone(
            target,
            form="ri",
            format_spec_A=fmt,
            format_spec_B=fmt,
            format_spec_freq=fmt,
            write_z0=True,
        )
        return target


def _port_z_ref(
    z_ref: float | Mapping[str, float | Sequence[float]], port: str, count: int
) -> NDArray[np.float64]:
    """The reference impedances of a port's terminals, validated."""
    if isinstance(z_ref, Mapping):
        if port not in z_ref:
            raise ValueError(f"z_ref has no entry for port {port!r}.")
        value = z_ref[port]
    else:
        value = z_ref
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        if np.any(arr.imag != 0):
            raise ValueError("Complex reference impedances are not supported.")
        arr = arr.real
    out = np.broadcast_to(np.asarray(arr, dtype=float), (count,)).copy()
    if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError(f"z_ref of port {port!r} must be finite and positive.")
    return out


def build_terminal_s(
    runs: Sequence[ModalRun | str | Path | dict],
    specs: Mapping[str, TerminalPortSpec] | Sequence[TerminalPortSpec],
    modes_2d: Mapping[str, Sequence[PortModes2D]],
    *,
    z_ref: float | Mapping[str, float | Sequence[float]] = 50.0,
    lumped: Sequence[LumpedTerminal] = (),
    pilot: Mapping[str, Sequence[int | str]] | None = None,
    reference_voltages: Mapping[str, Sequence[ArrayLike]] | None = None,
    rtol: float = PAIRING_RTOL,
    degenerate_rtol: float = FACE_MODE_RTOL,
    allow_degenerate: bool = False,
    extrapolate: bool = False,
    interpolation: str = "inverse",
) -> TerminalSResult:
    """Terminal S-parameters from driven runs and 2D mode solves.

    The pipeline: :func:`modal_s_from_runs` assembles the modal S;
    :func:`align_gauge` brings each port's 2D ``T_V`` and ``T_I`` into the gauge
    of its 3D modes on the S-parameter frequencies; lumped ports get the
    pass-through block; :func:`gsim.palace.terminal.modal_to_terminal_s` does the
    transformation.

    Args:
        runs: The driven runs (one for a single run, one per mode for cycling).
            Read with :func:`load_modal_run` including the log, for the wave
            numbers.
        specs: The terminal layout of each multimode port, by port name (or a
            list; the port names are the specs' names).
        modes_2d: Per port name, its 2D solves (see :func:`port_modes_2d`).
        z_ref: Reference impedance of the terminals in ohm: a number for all, or
            ``{port: number or one value per terminal}``. Lumped terminals
            use their own resistance.
        lumped: Lumped ports to include in the network.
        pilot: Per port name, the pilot terminal per mode (see
            :func:`align_gauge`).
        reference_voltages: Per port name, the reference voltages of the 3D
            modes per 2D solve (see :func:`align_gauge`).
        rtol: Pairing tolerance on ``n_eff``.
        degenerate_rtol: Tolerance under which 3D modes count as degenerate (see
            :func:`align_gauge`).
        allow_degenerate: See :func:`align_gauge`.
        extrapolate: See :func:`align_gauge`.
        interpolation: See :func:`align_gauge`.

    Returns:
        The terminal S with its modal S and aligned matrices.

    Raises:
        ValueError: When specs or 2D solves do not match the ports of the runs,
            a port has a different number of modes and terminals, the runs carry
            no wave numbers, or any step above refuses its input.
    """
    modal = modal_s_from_runs(runs, lumped=lumped)
    spec_by_name = (
        dict(specs)
        if isinstance(specs, Mapping)
        else {spec.name: spec for spec in specs}
    )
    lumped_by_number = {item.index: item for item in lumped}
    ports = {item.portnumber: item.port for item in modal.order if item.mode > 0}
    wanted = set(ports.values())
    if wanted != set(spec_by_name):
        raise ValueError(
            f"The terminal specs are for ports {sorted(spec_by_name)} but the runs "
            f"have the multimode ports {sorted(wanted)}."
        )
    if missing := sorted(wanted - set(modes_2d)):
        raise ValueError(f"No 2D solves were given for the ports {missing}.")

    t_vs: list[NDArray[np.complex128]] = []
    t_is: list[NDArray[np.complex128]] = []
    names: list[str] = []
    zr: list[NDArray[np.float64]] = []
    aligned: dict[str, AlignedPortModes] = {}
    nf = modal.freq_ghz.size
    for number in sorted({item.portnumber for item in modal.order}):
        if number in lumped_by_number and number not in ports:
            item = lumped_by_number[number]
            root = np.sqrt(item.resistance)
            t_vs.append(np.full((nf, 1, 1), root, dtype=np.complex128))
            t_is.append(np.full((nf, 1, 1), 1.0 / root, dtype=np.complex128))
            names.append(item.name)
            zr.append(np.array([item.resistance]))
            continue
        port = ports[number]
        spec = spec_by_name[port]
        cols = modal.port_columns(port)
        if len(cols) != len(spec.terminals):
            raise ValueError(
                f"Port {port!r} has {len(cols)} modes in the runs but "
                f"{len(spec.terminals)} terminals: they must match."
            )
        if modal.wave_numbers is None:
            raise ValueError(
                "The runs carry no wave numbers: load them with "
                "load_modal_run(..., log=<the Palace log>)."
            )
        result = align_gauge(
            modes_2d[port],
            freq_ghz=modal.freq_ghz,
            wave_numbers=modal.wave_numbers[:, cols],
            terminal_names=spec.terminal_names,
            pilot=None if pilot is None else pilot.get(port),
            reference_voltages=(
                None if reference_voltages is None else reference_voltages.get(port)
            ),
            rtol=rtol,
            degenerate_rtol=degenerate_rtol,
            allow_degenerate=allow_degenerate,
            extrapolate=extrapolate,
            interpolation=interpolation,
        )
        aligned[port] = result
        t_vs.append(result.t_v)
        t_is.append(result.t_i)
        names.extend(f"{port}:{terminal}" for terminal in spec.terminal_names)
        zr.append(_port_z_ref(z_ref, port, len(spec.terminals)))
    z_vec = np.concatenate(zr)
    s_terminal = modal_to_terminal_s(modal.s_modal, t_vs, t_is, z_ref=z_vec)
    return TerminalSResult(
        freq_ghz=modal.freq_ghz,
        s=s_terminal,
        port_names=tuple(names),
        z_ref=z_vec,
        modal=modal,
        aligned=aligned,
    )


def terminal_s(
    runs: Sequence[ModalRun | str | Path | dict],
    specs: Mapping[str, TerminalPortSpec] | Sequence[TerminalPortSpec],
    modes_2d: Mapping[str, Sequence[PortModes2D]],
    *,
    z_ref: float | Mapping[str, float | Sequence[float]] = 50.0,
    lumped: Sequence[LumpedTerminal] = (),
    pilot: Mapping[str, Sequence[int | str]] | None = None,
    reference_voltages: Mapping[str, Sequence[ArrayLike]] | None = None,
    rtol: float = PAIRING_RTOL,
    degenerate_rtol: float = FACE_MODE_RTOL,
    allow_degenerate: bool = False,
    extrapolate: bool = False,
    interpolation: str = "inverse",
    name: str | None = None,
) -> Network:
    """Terminal S-parameters as a scikit-rf network, one port per terminal.

    The public entry point: see :func:`build_terminal_s` for the arguments and
    what is done, and use that function when the modal S, the pairing and the
    aligned matrices are wanted as well. Ports are named
    ``"{port}:{terminal}"``; write a Touchstone file with
    ``network.write_touchstone`` or :meth:`TerminalSResult.write_touchstone`.

    Args:
        runs: The driven runs.
        specs: The terminal layout of each multimode port.
        modes_2d: Per port name, its 2D solves.
        z_ref: Reference impedance of the terminals in ohm.
        lumped: Lumped ports to include.
        pilot: Per port, the pilot terminal per mode.
        reference_voltages: Per port, the 3D reference voltages per 2D solve.
        rtol: Pairing tolerance on ``n_eff``.
        degenerate_rtol: Tolerance under which 3D modes count as degenerate.
        allow_degenerate: See :func:`align_gauge`.
        extrapolate: See :func:`align_gauge`.
        interpolation: See :func:`align_gauge`.
        name: Name of the network.

    Returns:
        The terminal network.
    """
    return build_terminal_s(
        runs,
        specs,
        modes_2d,
        z_ref=z_ref,
        lumped=lumped,
        pilot=pilot,
        reference_voltages=reference_voltages,
        rtol=rtol,
        degenerate_rtol=degenerate_rtol,
        allow_degenerate=allow_degenerate,
        extrapolate=extrapolate,
        interpolation=interpolation,
    ).network(name)
