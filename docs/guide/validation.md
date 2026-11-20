# Validation and benchmarks

This page lists the evidence in this repository about how far gsim results can be trusted, grouped by what kind of evidence
it is. Each entry says what was compared with what, what the stated result is, and what the comparison does not show.

A stated result is a number that a test, a notebook or a docs page in this repository gives. Unless the entry says otherwise,
it was not re-run for this page. The page does not summarise accuracy as one figure, because the evidence does not support
one.

## 1. One solver against another

| Evidence | Compared | Stated result | Does not show |
| --- | --- | --- | --- |
| [FDTD benchmark](../fdtd-benchmark.md) | GDSFactory FDTD, a commercial CPU solver, a commercial GPU solver and Meep on a 20 µm silicon directional coupler with a 236 nm gap and SiO<sub>2</sub> cladding (geometry from a published study) | At 1550 nm, 39.1 % cross and 60.9 % through transmission, each within 0.7 percentage points of the commercial CPU result | Agreement with a measurement. The solver outputs and scripts behind the charts are not in this repository, only the charts and the page text. One device, one wavelength is quoted |
| `tests/femwell/test_cross_validation.py` | femwell against Palace `BoundaryModeSim` on the same mesh and the same materials: n<sub>eff</sub> of a silicon strip at 1.55 µm, and the complex n<sub>eff</sub> with a conductive core | Test bound 1e-3 relative. The test's comments record 4.1e-5 (optical) and 2.9e-4 (conductive core) as measured on that mesh, and 9e-3 for femwell at order 1 | Agreement with the drawn device: both solvers get the same mesh and materials, so a shared mesh error cancels. It needs a Palace binary and femwell, carries the `palace_local` marker, is excluded from the default test run, and no workflow in `.github/workflows` selects it |

## 2. A solver against an analytic reference

| Evidence | Compared | Stated result | Does not show |
| --- | --- | --- | --- |
| [Magnetostatic inductance notebook](../nbs/palace_magnetostatic.md) | Palace `MagnetostaticSim` against closed-form self and mutual inductance of two concentric rings, and against a ribbon-bundle reference computed without finite elements | The notebook asserts agreement with the closed forms within 2 % and states agreement with the ribbon bundle within 1 %. It says the 1 % bound was set after the runs, so it acts as a regression gate. Run with Palace v0.18.1 at order 2 | Thin rings only. A 2 % bound is the test tolerance, not the observed error. The notebook states the errors shrink with mesh refinement, and come from below |
| `tests/tcad/test_solve_runtime.py` (marker `tcad_local`; run by the TCAD workflow) | TCAD small-signal C(V) against the abrupt-junction depletion approximation (Sze), on a 1e18 / 1e18 cm<sup>-3</sup> junction at 0, 0.5 and 1 V | The test accepts a largest relative deviation of 0.35 | A tight match. 0.35 is a loose bound that allows for oxide fringing. A separate test checks that C scales as the square root of the permittivity |
| [PN junction notebook](../nbs/tcad_pn_junction.md) | The same comparison on its own device, five biases from 0 to 2 V, no oxide in the Poisson solve | Not stored. Running its setup and sweep cells while writing this guide gave a largest deviation of 2.8 % (at 0 V) and 0.5 % to 1.0 % at the other biases, with gsim 0.10.0 on one Windows machine | A converged result. A coarser mesh without the notebook's refinement box moved the capacitance by up to 6.9 % in the same check. Rerun to confirm |
| `tests/femwell/test_z0_extraction.py` | femwell power-current Z<sub>0</sub> against the analytic impedance of a coaxial line, copper inner conductor and perfect inner conductor | Test bounds 8 % (copper, measured 3.2 % high per the test comment) and 1 % (perfect conductor). The comments say order 1 reads 8 % to 29 % high on the perfect-conductor mesh, against within 0.4 % at order 2 | A planar line such as a CPW. The check is on a coax |
| Time convention of the FDTD export ([FDTD guide](../api/fdtd.md#export-to-sax-and-touchstone)) | Phase of a 10 µm silicon straight waveguide against a separate mode solver | A delay of 139 fs against 137 fs from the group index | One device. The guide says so, and `s_matrix` warns when a negative group delay appears |

## 3. Checks of post-processing, not of the solver

- `tests/common/test_s_matrix.py` compares the S-matrix assembly and the SAX and Touchstone exports with closed forms and with
  independent libraries (`sax.models.straight`, `skrf.Network`). It tests the arithmetic of the export. It does not test an
  FDTD result.
- The [signal-integrity notebook](../nbs/palace_signal_integrity.md) runs analytic checks of mixed-mode algebra, port
  reordering, uniform-line impedance and a known reflection time. Its own conclusions say they validate the post-processing on
  those examples and not the IHP electromagnetic model. Figures from its recorded runs are in `docs/assets/plots/palace-si-*.png`;
  the raw runs are not bundled.
- The paper-reproduction notebooks (for example the [Lu broadband coupler](../nbs/fdtd_lu_broadband_coupler.md)) are
  reconstructions of published devices. The Lu notebook says that its saved spectrum is an example result and not a claim of
  mesh convergence, and that the paper's output-normalised splitting does not establish absolute throughput. The other three
  paper notebooks were not read for this page.

## 4. Run evidence

[Checking a Palace run](../palace-validation.md) describes what `validate_solution` reports: solver convergence, estimator
status, adaptive sampling, process exit, port-mode continuity and input provenance. Its first paragraph says it does not compare
results with analytic, measured or published references. A converged run can still be wrong, for example with a wrong mode or a
mesh that does not resolve the geometry.

Cloud regression tests, such as `tests/palace/test_cloud_eigenmode_cavity.py`, compare a run with the stored output of an earlier
run. They detect a change, not an error.

## 5. Measured devices

There is no measured-device comparison in this repository that is presented as validation.

The IHP L2n0 inductor material under `nbs/data/ihp_validation/` is exploratory. Its own README calls the runs "recorded 1 GHz
screening runs, not a mesh-converged validation". By the extraction it uses, the two measured dies give 1.674 and 1.672 nH, and
the best recorded two-port value is 0.802 nH, about a factor of two low. It is not accuracy evidence for Palace or for gsim,
and the notebook that reads it (`nbs/ihp_l2n0_openems_validation.ipynb`) is not part of the published documentation.

## What the evidence above does not cover

- Photonic S-parameters of any device against a measurement.
- Ring resonators and any circuit built from FDTD S-matrices: only the export arithmetic is tested.
- Waveguide group index and loss from femwell against a measurement or an analytic waveguide.
- CPW and microstrip impedance and loss from Palace against a closed-form line or a measurement.
- Resonant frequency and Q of superconducting resonators against a measurement or an analytic resonator.
- Free-carrier effects (n<sub>eff</sub> shift, V<sub>π</sub>L) against measured modulators. The TCAD checks above cover capacitance
  only.
- Thermal effects. gsim has no thermal solver; see [thermal tuning](thermal-tuning.md).

Treat results in these areas as unvalidated. Use the convergence checks on each task page, and compare with your own
measurements or an independent reference before relying on a number.

See also [choosing a solver](choosing-a-solver.md).
