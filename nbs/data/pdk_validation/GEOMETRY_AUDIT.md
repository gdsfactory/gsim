# Geometry and stack audit of the recorded Meep comparisons

The results in `runs/summary.json` are comparisons, not acceptance tests. A shared nominal parameter does not prove that
the simulated GDS is the mask used for a measured spectrum. The UBC `.mat` files in `_fetch.py` contain spectra; this
validation set does not include the fabricated ring, Bragg, or MZI mask layouts. The generic-PDK SiEPIC case crosses
PDKs by construction.

<!-- markdownlint-disable MD013 -->

| Case                              | Reference and simulated geometry                                                                                                  | Recorded result                                                                     | Geometry status                                                                                                                                                                                                                                                                                                                                                                                  |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| UBC straight                      | SAX straight model and `ubcpdk.cells.straight`, both 20 µm                                                                        | S21 RMS 0.006 dB                                                                    | Nominal parameters match; useful pipeline check                                                                                                                                                                                                                                                                                                                                                  |
| UBC directional coupler           | SAX LUT and `ubcpdk.cells.coupler`, gap 0.2 µm, length 10 µm                                                                      | S41 4.43 dB, S31 4.83 dB RMS                                                        | Nominal parameters match; LUT process/cross-section equivalence unverified                                                                                                                                                                                                                                                                                                                       |
| UBC ring coupling area            | SAX racetrack coupling LUT and `ubcpdk.cells.coupler_ring`, gap 0.1 µm, radius 10 µm, length_x 4 µm                               | S41 9.47 dB, S31 3.68 dB RMS                                                        | Nominal parameters match; the earlier notebook displayed `ring_single` instead of the simulated cell                                                                                                                                                                                                                                                                                             |
| UBC double-bus ring               | Measured `LukasC_RingDoubleTER10g100_1495.mat`; simulation uses `ubcpdk.cells.ring_double`                                        | S21 8.00 dB RMS against detector 1; 25.66 dB against the previously used detector 0 | Detector 0 has periodic drop-port peaks; detector 1 has the higher, smoother through response and is the plausible match to o1→o2. Detector labels are absent from the `.mat`, so this mapping is an inference. Radius and gap are inferred from the filename; fabricated mask and remaining geometry are unverified. The earlier notebook displayed `ring_single` instead of the simulated cell |
| UBC sinusoidal Bragg              | Measured `ELEC_413_lukasc_BraggSet1Num10_1272.mat`; simulation uses a custom 200-period, 318 nm-period, 120 nm-modulation polygon | S11 23.37 dB RMS                                                                    | Reference mask and exact dimensions are unverified; the `.mat` filename identifies a design, not its polygon geometry                                                                                                                                                                                                                                                                            |
| Cornerstone coupler and racetrack | Cornerstone LUTs and derived `core = WG - GRA` geometry                                                                           | S41 7.25 and 1.91 dB RMS, respectively                                              | Nominal LUT parameters match; PDK overlay and derived-layer interpretation need process confirmation                                                                                                                                                                                                                                                                                             |
| SiEPIC half-ring                  | Measured SiEPIC half-ring S-parameters; generic photonic PDK geometry                                                             | S31 1.20, S41 6.18 dB RMS                                                           | Cross-PDK comparison; exact geometry and stack matching are unverified                                                                                                                                                                                                                                                                                                                           |
| UBC MZI                           | Measured spectrum; `ubcpdk.cells.mzi` with nominal length difference                                                              | No accepted run                                                                     | Runtime-excluded; no validation conclusion                                                                                                                                                                                                                                                                                                                                                       |

<!-- markdownlint-enable MD013 -->

The straight-guide 0.006 dB result checks the S21 extraction pipeline. Both models have nearly flat unit transmission in
this 2-D setting, so it cannot establish sensitivity to the fabricated cross section or process loss.

The
[pinned UBC Bragg design description](https://github.com/gdsfactory/ubc/blob/23a845acbcb03fccad77f96fdb85a0aa27306601/ubcpdk/data/dbr.md)
states `10 | 120 | sinusoidal` under `Design # | Δw | Type` for Set 1, and describes 200 periods per device. This
supports the nominal 120 nm sinusoidal modulation and period count. The measured structure uses a reflection fixture
with a 2×2 splitter, two grating couplers and a terminator. The `.mat` loader returns detector power in dBm, and the
validation harness only normalizes its level. It does not remove the fixture's wavelength response or prove the chosen
detector column maps to bare-grating S11. The 23.37 dB RMS value is a fixture-mismatched diagnostic, not a
device-validation error.

The previous documentation GDS for Bragg was a 9.38 × 0.60 µm rectangular device, while the recorded Meep layout is
66.60 × 0.56 µm and sinusoidal. The previous ring documentation GDS also differed in extent and polygon count from the
recorded run. The eight cases with recorded spectra now publish byte-identical copies of their archived `layout.gds`,
and the page builder prefers those run layouts. The MZI has no accepted spectrum and is marked runtime-excluded.

All recorded spectra in this audit use an XY 2-D effective-index approximation. The committed `sim_config.json` files
record a reviewed PDK overlay, layer mapping, effective core permittivity, cladding and stack hash. For example, the UBC
Bragg run uses `epsilon_core_2d = 8.1081` and SiO2 background. This is an explicit model assumption, not proof that the
PDK's fabricated cross section or dispersion matches the measurement. The correction that pins the 2-D cut to the core
is already present in these runs; the large ring and Bragg errors remain after that correction.

The 100 nm coupling gap has only three grid cells at 30 pixels/µm (four at 40 pixels/µm); the Bragg width modulation has
a 60 nm half-amplitude, less than two cells at 30 pixels/µm. Mesh refinement and a 3-D comparison are needed before
attributing those errors solely to the PDK. The measured ring through dips near 1.55 µm are about 0.17–0.25 nm wide,
whereas 101 simulated samples across the 1.50–1.60 µm window are roughly 1 nm apart. The recorded ring run stopped at
its 300 s wall cap and the Bragg run reached its 1000 Meep-time cap; their spectral features should not be called
converged.

## UBC coupler mesh sensitivity

The archived 30 pixels/µm run and `mesh_sweeps/coupler_r50/` use identical GDS, 101 wavelengths, Si core slab
permittivity 8.1081139, SiO2 background and port definitions. Both stopped at about 250 Meep time units. The 50
pixels/µm run used a tighter field-energy threshold (1e-6 rather than 1e-4); it used one CPU core, 701,250 grid cells
and 456 s wall time. Its S41 and S31 RMS against the SAX LUT changed from 4.43/4.83 dB to 4.06/4.68 dB, respectively.

Across the full sampled band, the 30-to-50 raw monitor-plane complex RMS changes are 0.388 for S41 and 0.443 for S31.
After de-embedding to the device port planes, they are 0.379 and 0.429. Their magnitude changes are 0.37 and 0.15 dB
RMS, but their wrapped phase changes are 42.2° and 30.6° RMS. This is **mesh sensitivity, not complex-S convergence**.
The S11 magnitude change is 6.01 dB RMS because reflection is small; its absolute complex RMS change is 0.0112.

An initial 50 pixels/µm run used a full-stack slab eigenmode as its 2-D core index. It selected the thick silicon
substrate mode (core permittivity 12.0744 instead of 8.1081), giving an apparently excellent 0.55 dB S41 match. That
result changes the physical model as well as the mesh and is excluded from the convergence comparison. The harness now
computes the local asymmetric core slab index deterministically, independent of whether Meep is installed in the driver
environment.

The 70 pixels/µm run in `mesh_sweeps/coupler_r70/` retains the same GDS, 101 wavelengths, materials and ports and uses
the original 30 pixels/µm stopping settings. It used one CPU core, 1.37 million cells and 834 s wall time. S41/S31 RMS
against the SAX LUT is 3.97/4.64 dB. Relative to 50 pixels/µm, the S41/S31 magnitude changes fall to 0.091/0.038 dB RMS
and the raw monitor-plane complex changes to 0.0698/0.1286 RMS; de-embedded device-port changes are 0.065/0.122. Wrapped
phase changes are 7.1°/8.9° RMS. The corresponding 30-to-50 raw complex changes were 0.388/0.443, so the trend is
encouraging but **does not meet a 0.01 complex-S mesh gate**. S11 still changes by 0.0070 complex RMS. The 70 pixels/µm
run stopped at 200 Meep time units instead of about 250, a stopping-time difference that also needs control before any
strict mesh-convergence claim. Meep's o1 mode index at 1.55 µm changes from 2.6000 to 2.6111 to 2.6162 across the three
meshes, consistent with substantial phase sensitivity.

Next comparison should use a pinned reference GDS or a documented layout parameter table for each measured device,
verify the derived core polygons and port orientation, then run a resolution and stopping-time sweep. A better fit
obtained by changing nominal geometry without reference-mask evidence would be a fit to measurement, not an independent
validation.
