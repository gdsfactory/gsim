This is unmodified numerical output from a converged GDSFactory FDTD run with complex-plane output enabled. The generic
geometry is a straight 3 µm strip, 0.5 µm wide and 0.22 µm thick, with index 3.47 in a uniform index-1.44 cladding. It
uses a 50 nm Yee grid. The paired arrays and result JSON are raw solver output.

An eigenmode source launches from the negative X face. Monitors at X = 1.2 and 1.4 µm record flux, intensity, and
Ey/Ez/Hy/Hz at 1.55 µm. The source samples five wavelengths from 1.45 to 1.65 µm. The run converged after 4,864 steps.

`result.json` reports plane dimensions `[32, 25]` in `(u, v)` order. Each NPY sidecar is `(25, 32)` in `(v, u)` order.
The actual exported coordinates include the snapped Z range −0.5 to 0.7 µm, rather than the requested −0.49 to 0.71 µm.

The DFT kernel is `exp(-i*omega*t)`, corresponding to positive-time phasors `exp(+i*omega*t)`. The Ey phase decreases by
about 1.957 radians between the monitors, consistent with forward propagation and the exported S-parameters.

The independent colocated-field integral `0.5 * Re(Ey * conj(Hz) - Ez * conj(Hy))` gives 125,121,021 and 125,026,882 in
solver power units. The respective native fluxes are 126,484,095 and 126,425,110, differing by 1.08% and 1.11%. The
native flux retains transverse Yee staggering; colocation averages the exported fields to common primal points, so exact
equality is not expected. Convert coordinate differences to solver length units using `solver_time_unit_seconds * c`
before integrating.

The regression tests read these files locally and never submit a simulation.
