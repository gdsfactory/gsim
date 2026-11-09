"""Terminal (multi-pin) ports: which conductors, which paths, which ``T_V`` and ``T_I``.

:mod:`gsim.palace.terminal` turns *modal* S-parameters into *terminal* S-parameters
once it is given, for every physical port, the matrix of terminal voltages
``T_V`` and terminal currents ``T_I`` of the port's modes. It says nothing about
where those matrices come from. This module is the first half of that answer
for a Palace port:

* :class:`TerminalPortSpec` names a port's terminals and its reference conductor
  as rectangles on the port's cross-section (the
  :class:`gsim.common.modes.Conductor` the line-impedance code already uses);
* :func:`terminal_paths` draws one voltage path per terminal, from the terminal
  to the reference, that never crosses a conductor;
* :func:`declare_terminal_paths` registers those paths on a
  :class:`~gsim.palace.boundarymode.BoundaryModeSim`, so that Palace reports the
  voltage of every mode on every path;
* :func:`port_terminal_matrices` reads the saved fields of the solved modes and
  returns ``T_V`` and ``T_I`` in SI units, one column per mode, normalised to
  unit power and with a fixed phase.

Everything is stated in the cross-section's own ``(h, v)`` coordinates (um).

Conventions
-----------

* Time convention ``exp(+j w t)``, as in the rest of gsim.
* Column ``k`` of ``T_V`` holds the voltages of terminal ``i`` against the
  reference conductor for mode ``k``, from the line integral of the transverse
  electric field along the terminal's path (the path is declared from the
  terminal to the reference, so a terminal above the reference has a positive
  voltage when the field points from the terminal to the reference).
* Column ``k`` of ``T_I`` holds the terminal currents *into the network* for a
  forward wave of that mode. The contour integral of ``H`` around a conductor
  gives the current along the propagation axis; whichever way the port face
  points, the column is oriented so that ``Re(T_V[:, k]^H T_I[:, k]) > 0``,
  which is the statement that the incident wave carries power into the network.
* Each column is scaled so that ``|2 P| = 1`` with ``P`` the complex power flux
  of the saved mode, matching Palace's unit-power convention for wave-port
  modes, and rotated so that the voltage on its *pilot* terminal (by default the
  terminal with the largest voltage) is real and positive. Scaling a column of
  both matrices by one complex constant leaves the terminal S unchanged, so
  neither choice matters to :func:`gsim.palace.terminal.modal_to_terminal_s` on
  its own; they exist so that these columns can be compared with the wave-port
  modes of a 3D run, which carry their own scale and phase.

The second half joins these 2D matrices to the modal S of the driven runs:

* :func:`load_modal_run` and :func:`modal_s_from_runs` read ``port-S.csv`` by
  Palace index, decode the extra entries of a multimode port with the sidecar
  ``port_modes.json`` and assemble one modal S from a single run or from the
  active-mode cycling runs;
* :func:`align_gauge` pairs the 2D modes with the 3D port modes by ``n_eff`` and
  puts ``T_V`` and ``T_I`` into the gauge of the 3D modes (the pilot-terminal
  rule, or a fit to the voltages of Palace's own saved port modes);
  :func:`port_mode_reference_voltages` reads those saved port-mode fields off a
  driven run and returns the voltages to fit to;
* :func:`terminal_s` (and :func:`build_terminal_s`, which keeps the pieces) turns
  that into terminal S-parameters, with lumped ports passed through;
* :func:`short_terminals` and :func:`mixed_mode` act on the resulting network.

Modules
-------

The code is split by what it computes; everything public is re-exported here,
so ``gsim.palace.terminal_ports.<name>`` and ``gsim.palace.<name>`` keep
working:

* :mod:`~gsim.palace.terminal_ports.spec`: :class:`TerminalPortSpec` and the
  voltage paths;
* :mod:`~gsim.palace.terminal_ports.modes2d`: ``T_V`` and ``T_I`` from the
  saved 2D fields;
* :mod:`~gsim.palace.terminal_ports.modal`: modal S from the driven runs;
* :mod:`~gsim.palace.terminal_ports.gauge`: :func:`align_gauge`;
* :mod:`~gsim.palace.terminal_ports.assembly`: :func:`build_terminal_s` and
  :func:`terminal_s`;
* :mod:`~gsim.palace.terminal_ports.network`: :func:`short_terminals` and
  :func:`mixed_mode`.
"""

from __future__ import annotations

# The quadrature default of the voltage integrals stays readable here.
from gsim.palace.mode_fields import VOLTAGE_NSAMPLES  # noqa: F401
from gsim.palace.terminal_ports.assembly import (
    TerminalSResult,
    build_terminal_s,
    terminal_s,
)
from gsim.palace.terminal_ports.gauge import (
    INTERPOLATIONS,
    PAIRING_RTOL,
    PORT_FIELD_TOL_UM,
    AlignedPortModes,
    align_gauge,
    port_mode_reference_voltages,
)
from gsim.palace.terminal_ports.modal import (
    FREQ_RTOL,
    LOG_FREQ_RTOL,
    DegenerateFlag,
    LumpedTerminal,
    ModalPortMode,
    ModalRun,
    ModalS,
    load_modal_run,
    modal_s_from_runs,
)
from gsim.palace.terminal_ports.modes2d import (
    MAX_COND,
    PortModes2D,
    PortTerminalMatrices,
    port_modes_2d,
    port_terminal_matrices,
)
from gsim.palace.terminal_ports.network import mixed_mode, short_terminals
from gsim.palace.terminal_ports.spec import (
    TerminalPortSpec,
    declare_terminal_paths,
    terminal_paths,
)

__all__ = [
    "FREQ_RTOL",
    "INTERPOLATIONS",
    "LOG_FREQ_RTOL",
    "MAX_COND",
    "PAIRING_RTOL",
    "PORT_FIELD_TOL_UM",
    "AlignedPortModes",
    "DegenerateFlag",
    "LumpedTerminal",
    "ModalPortMode",
    "ModalRun",
    "ModalS",
    "PortModes2D",
    "PortTerminalMatrices",
    "TerminalPortSpec",
    "TerminalSResult",
    "align_gauge",
    "build_terminal_s",
    "declare_terminal_paths",
    "load_modal_run",
    "mixed_mode",
    "modal_s_from_runs",
    "port_mode_reference_voltages",
    "port_modes_2d",
    "port_terminal_matrices",
    "short_terminals",
    "terminal_paths",
    "terminal_s",
]
