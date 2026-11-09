# Wave-port mode excerpts

Trimmed excerpts of three real driven Palace logs of a generic CPW line on the IHP stack (two wave ports, 31 frequencies
from 1 to 10 GHz in 0.3 GHz steps, two excitations, one mode per port). They are parser fixtures for
`test_port_mode_continuity.py`, not complete runs. Only the lines that `gsim.palace._validation_log.port_mode_samples`
reads are kept, verbatim and in order (188 lines each): `Sweeping excitation index N`,
`Calculating boundary modes at wave ports for ...`, and `Port N, mode M: k_n = ...`. Unicode symbols are retained, so
read them as UTF-8.

- `port_modes_old_palace_spurious.txt`: Palace `cfa430a`. Excitation 2 has three spurious port solves (port 1 and port 2
  at 1 GHz, port 2 at 1.3 GHz) with n_eff between 2.0 and 3.4 and a large imaginary wave number (gsim#375).
- `port_modes_old_palace_clean.txt`: Palace `cfa430a`, another run of this kind of line (archived as the "coarse"
  variant); no spurious solve.
- `port_modes_new_palace_clean.txt`: newer Palace `e1bdf9b7`, same input as the spurious run; no spurious solve.
