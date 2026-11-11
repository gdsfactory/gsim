This sample records a converged GDSFactory FDTD simulation of the public generic-PDK grating-coupler example at 1.554
µm, on a 50 nm Yee grid. It launches the fundamental guided mode through a 3 µm straight input lead. The grating has 16
periods with 0.324 µm gaps, 0.343 µm teeth, and a 14 µm taper. Silicon and cladding indices are 3.47 and 1.443.

A +Z radiation plane at z = 1.05 µm captures Ex, Ey, Hx, and Hy. An independent native Gaussian monitor on the same
plane supplies the solver reference. `fiber_mode.json` describes its 5 µm waist and 15° tilt. A +X input flux plane, 1.5
µm from the source, spans 3 µm in Y and −1 to 1.55 µm in Z. Its net flux and source-port S11 determine incident guided
power.

The retained JSON selects the center wavelength from the 101-sample spectrum. The radiation arrays are reduced from
`(400, 401)` to `(101, 101)`: every fourth sample is retained, together with both endpoints, and values are converted
from float64 to float32. Coordinates retain the actual sample positions; the final V interval is 150 nm instead of 200
nm. No fields are source-normalized or rephased. Native fluxes and normalized overlap are unchanged solver outputs.

The native power CE is 0.317255071956. The full-resolution offline result is 0.317326796573, a 0.0226% relative
difference. The reduced sample gives 0.317331870, a 0.0242% relative difference. The full-resolution field integral and
native radiation flux differ by 0.061%; colocation and Yee staggering mean they need not agree exactly. Moving the input
plane to 1 or 2 µm from the source, or widening it to 5 µm, changes recovered incident power by less than 0.1% in this
run.

The solver ran for 122.855 seconds and converged after 19,712 steps. These files allow numerical regression tests
without cloud access. They test the small helper in `nbs/fdtd_focusing_grating_coupler_gpdk.ipynb` against the native
Gaussian projection as an independent validation reference.
