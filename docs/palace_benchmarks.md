# Palace, femwell and TCAD: what is validated against what

This page lists the checks that exist in the gsim repository, what each one compares, and how closely the two sides
agree. It is a map of evidence, not a claim that every solver is validated for every device. Every number below is copied
from a test assertion, a comment next to one, or a result stored in the repository; the **Where** column gives the file
and lines. To check the evidence of a run you made yourself, see
[Checking a Run](palace-validation.md) instead.

How to read the numbers:

- "limit" is the tolerance the test asserts, not the error the solver makes. Where the test comment records a measured
  value, the table gives that value and the limit.
- A comparison of two solvers on the same mesh and materials shows that the solvers agree. It does not show that either
  agrees with a device.
- "No number stored" means the notebook prints a result when it runs, but no output is kept in this repository, so the
  page quotes none.

## What is compared with what

| Case | Solvers | Reference | Agreement | Where |
| --- | --- | --- | --- | --- |
| Coax, copper core, 100 GHz | femwell | analytic Z0 | 3.2 % high (limit 8 %) | `tests/femwell/test_z0_extraction.py:100-104` |
| Coax, perfect core, order 2 | femwell | analytic Z0 | limit 1 % | `tests/femwell/test_z0_extraction.py:192-204` |
| Same mesh and materials | Palace BoundaryMode, femwell | each other | 4.1e-5, 2.9e-4 (limit 1e-3) | `tests/femwell/test_cross_validation.py:40-45` |
| Terminal S, synthetic coupled lines | gsim post-processing | chain matrix | at most 3.4e-14 | `docs/terminal_wave_ports.md:62-70` |
| Two coupled GSSG lines | Palace (cloud) | 2D port solve, uniform line | n_eff 0.05 %; S21 phase 0.29 deg | `docs/terminal_wave_ports.md:400-411` |
| CPW de-embedding fit | scikit-rf, gsim fit | synthetic line | L, C, Rs limit 1e-3 | `tests/palace/test_cpw_notebook_validation.py:249-251` |
| PN junction C(V) | DEVSIM | depletion approximation | limit 35 % | `tests/tcad/test_solve_runtime.py:92-100` |
| Capacitor on an obtuse mesh | DEVSIM | scikit-fem P1 solve | limit 1e-9 | `tests/tcad/test_edge_couples.py:223-225` |
| MIM capacitor | Palace electrostatic | IHP compact model | no number stored | `nbs/palace_electrostatic.ipynb:402-413` |
| IHP SG13G2 L2n0 inductor | Palace | measured S2P, openEMS | **not validated**, see below | `nbs/data/ihp_validation/CONVERGENCE_AUDIT.md:3-6` |

## How each case is set up

| Case | Ports | Conductor model | Mesh convergence shown |
| --- | --- | --- | --- |
| Coax, copper core | 2D mode solve | copper volume, finite conductivity | no, one mesh |
| Coax, perfect core | 2D mode solve | perfect conductor (a hole) | three meshes, in a test comment |
| Palace against femwell | 2D mode solve | none (dielectric, doped silicon) | no |
| Terminal S, synthetic | none | none | not applicable |
| Two coupled GSSG lines | multimode wave ports | perfect conductor (notebook), vias with conductivity | no |
| CPW de-embedding fit | none (the notebook uses lumped and wave ports) | not applicable | not applicable |
| PN junction C(V) | none | not applicable | no sweep |
| Capacitor on an obtuse mesh | none | not applicable | not applicable |
| MIM capacitor | electrostatic terminals | ideal equipotential | no, plate-size sweep only |
| IHP L2n0 inductor | lumped vertical ports | PDK stack with uniform aluminum conductivity | no |

## Notes on the rows

**Coax with a copper core.** The inner conductor is a disk of copper (5.8e7 S/m) in a dielectric with relative
permittivity 2.25, inner radius 1 um, outer radius 3 um, at 100 GHz. The test comment records "3.2% high with a small
negative reactance", the expected finite-conductivity correction, and asserts 8 %
(`tests/femwell/test_z0_extraction.py:100-104`). The 3.2 % is a comment, not an assertion.

**Coax with a perfect core, order 2.** The current comes from Ampere's law around the hole in the mesh. The test comment
says the same three meshes come out 29 %, 17 % and 8 % high at order 1 and within 0.4 % at order 2
(`tests/femwell/test_z0_extraction.py:141-145`); the test itself runs one mesh and asserts 1 %. femwell's default element
order is 2 in this checkout (`src/gsim/femwell/adapter.py:220`).

**Palace against femwell.** Both solvers get the same mesh and the same materials, so this checks the agreement of the
two solvers and not of either with a drawn device (`tests/femwell/test_cross_validation.py:18-21`). The test runs only
with `pytest -m palace_local` and needs a Palace binary. The comment records 4.1e-5 relative on Re(n_eff) for the optical
case and 2.9e-4 on the complex n_eff with RF materials, and 9e-3 for femwell at order 1.

**Terminal S on synthetic lines.** Largest difference of the terminal S against an independent chain-matrix solution,
over the four cases of the table in `docs/terminal_wave_ports.md:65-70`. The page documents these figures; no test
asserts that bound.

**Two coupled GSSG lines.** The values are the cloud measurements printed on the terminal wave-port page, not re-run for
this page. They compare the 3D run with the 2D port solve. The notebook runs both with perfect conductors, so that the
comparison tests the terminal-S machinery against the same physics, and leaves out the loss of real metal
(`nbs/palace_terminal_waveport_gssg.ipynb:341-349`). The uniform-line rows of the page also mention a lossy 2D solve, so
read the page to see which run each figure comes from. See the [GSSG notebook](nbs/palace_terminal_waveport_gssg.md) and
[Terminal (multi-pin) Wave Ports](terminal_wave_ports.md).

**CPW de-embedding fit.** The test feeds the notebook's own extraction and fit cells with a synthetic line of known
inductance, capacitance and skin-effect resistance, and recovers all three to 1e-3. It checks the post-processing, not
Palace. The notebook itself runs 100, 400 and 800 um lines with lumped and with wave ports; its text says a physical
comparison between the two port types also needs airbox and mesh convergence
([CPW Length Sweep and De-embedding](nbs/palace_cpw_deembedding.md)).

**PN junction C(V).** `tcad.compare_capacitance` evaluates the depletion approximation (Sze) at the biases of the sweep.
The test asserts that the simulated C(V) is within 35 % of it on a fully depleted abrupt junction. No measured deviation
is stored. The notebook [TCAD: PN Junction C(V)](nbs/tcad_pn_junction.md) prints it when it runs.

**Capacitor on an obtuse mesh.** A carriers-off capacitor with obtuse triangles in both dielectrics: DEVSIM with gsim's
signed couples against an independent P1 finite-element solve. The assertion is 1e-9; the TCAD guide reports 1e-14 and
35 % high for DEVSIM's own couples (`docs/api/tcad.md:233-234`). The test needs DEVSIM.

**MIM capacitor.** The notebook compares the Palace mutual capacitance with the IHP `cap_cmim` compact model, but the
dielectric permittivity is set so that the area term matches the model. The comparison therefore tests the fringe field
only, and no number is stored in the repository (`nbs/palace_electrostatic.ipynb:402-413`). The notebook is not in the
navigation.

## The IHP L2n0 inductor is not validated

The recorded best Palace sweep (11 points, 1 to 50 GHz) disagrees with the two pinned measured two-port files:

- S21 magnitude RMS error 8.86 dB and 8.83 dB, phase RMS error 44.0 and 45.4 degrees. S11, S12 and S22 disagree too.
- Differential inductance about 0.80 nH simulated against 1.67 nH measured, and Q at 1 GHz about 0.087 against about 6.6.
- Palace error indicator about 0.18 against an adaptive tolerance of 0.02, so discretization error is unresolved.

All of this is from the audit in
[`nbs/data/ihp_validation/CONVERGENCE_AUDIT.md`](https://github.com/gdsfactory/gsim/blob/main/nbs/data/ihp_validation/CONVERGENCE_AUDIT.md)
(lines 3-6, 12-16). The audit also lists the setup differences from the openEMS reference (ports, ground, boundary, stack,
mesh) and a controlled sequence of next steps. The audit states that the model "is not validated". Do not use the L2n0
setup as a reference result.

The comparison notebook, which runs from bundled data and no solver, is
[`nbs/ihp_l2n0_openems_validation.ipynb`](https://github.com/gdsfactory/gsim/blob/main/nbs/ihp_l2n0_openems_validation.ipynb).
It is **work in progress** and is not in the navigation.

## Other benchmarks

For the FDTD solver compared with other FDTD solvers on one directional coupler (response, throughput and cost), see
[FDTD Benchmark](fdtd-benchmark.md).

## Not in the table yet

The Baehr-Jones 2012 electrode comparison (femwell, Palace and the paper) is in gsim pull request
[#522](https://github.com/gdsfactory/gsim/pull/522) and is not in this table until that merges.
