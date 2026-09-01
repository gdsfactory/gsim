# Palace EIC benchmark handoff

## Goal

Reproduce two finite-conductivity, on-chip two-port structures with `gsim.palace` and compare the full complex S-matrix
against public ANSYS references:

1. IHP SG13G2 differential spiral inductor — direct HFSS 3D Layout reference.
1. NIST Parylene-C platinum CPW — ANSYS Q2D cascade plus measured reference.

The final benchmark must use finite-conductivity signal metal; PEC is allowed only as an explicitly labelled diagnostic
control. Run at most two cloud jobs concurrently and target jobs below 30 minutes. Every new job has a hard limit of
110,000 tetrahedra.

## Current execution status (2026-09-01)

The user approved ten additional NIST debug jobs while retaining the 110,000-tetrahedron gate. All ten slots have now
been used. No further NIST cloud job is authorized; never run more than two jobs concurrently.

- NIST initial job `01a05b6c-d591-7bf0-8f35-ee5477d7e4a7` failed because the airbox/port topology was invalid.
- NIST repaired numeric-wave-port job `01a05b71-e7ff-70c0-a69a-ccda87b54375` had 273,551 tetrahedra and failed at the
  maximum runtime. Its final log was still in the first 0.1 GHz boundary mode; no 3D frequency solve began.
- NIST capped 3D lumped-port job `01a05c94-afbd-7c42-a72b-4a54d30004b0` (input hash `71ecf571...`) had 71,370 tetrahedra
  and completed six first-order solves in 12.9 s. It is scientifically rejected: the end absorbing planes were flush
  with the lumped ports and S21 was -29.7/-65.3/-79.7 dB at 0.1/10.05/20 GHz.
- Corrected lumped-port job `01a05ccd-6ed7-7672-8369-6749ab21b960` (`palace-2e433b2b`) had 72,768 tetrahedra and gave
  S21 -5.78/-55.78/-67.71 dB. It is reciprocal/passive but scientifically wrong.
- Backplane plus two-gap-span job `01a05cd2-a4cb-75e0-b05f-e3922d69781e` (`palace-55df317c`) had 80,146 tetrahedra and
  order 1; S21 was -5.81/-49.51/-66.65 dB.
- The same 80,146-tet mesh at order 2, job `01a05cd5-301d-77e2-ad24-8aeb1bf7988e` (`palace-bec4b807`), improved S21 to
  -5.25/-39.37/-45.31 dB but remained far from the NIST Q2D target.
- Localized numeric-wave-port jobs `01a05ce0-c0e5-78c2-bc4d-b16b5e248cc1` (`palace-4ee3b95b`, 78,266 tets) and
  `01a05ce3-d4e4-7cb3-a2fc-4a443e98e7b9` (`palace-dabdc9c2`, 78,266 tets) remained in boundary-mode solves beyond one
  hour and were manually cancelled. Their port faces omit most of the substrate and backplane; reject them regardless of
  their failed terminal status.
- The official-guidance candidate uses HXT, four 0.625 um gap spans, 100 um lateral air, order 2, a `1e-8` linear
  tolerance, and first-order absorption on outer air only. It has 50,217 tetrahedra, 9,277 nodes, zero invalid SICN
  elements, and identical finite-Pt/PEC-control mesh hashes. Both configs passed the Palace v0.17.0 schema.
- Paired finite-Pt job `01a05d31-61b7-7bc3-bf51-a0de6300bbcc` (`palace-5ea50ea3`) and top-Pt PEC control
  `01a05d31-553a-7c93-a943-4c3b97ba94f3` (`palace-52aabcf5`) completed in 10m08s and 10m13s. At 0.1/10.05/20 GHz their
  S21 values were respectively -5.24/-29.59/-43.82 dB and -0.03/-1.08/-2.48 dB, versus the NIST Q2D -5.15/-9.66/-12.96
  dB. The same-mesh result localizes the excess attenuation to the finite-metal treatment; PEC is only an under-lossy
  control.
- Zero-geometric-thickness 200 nm Pt sheet job `01a05d3d-38ce-7cf0-965a-cee09e94bcf0` (`palace-e1132a86`) completed in
  9m34s on 47,844 tetrahedra. Its S21 was -5.26/-29.23/-39.38 dB, so removing the 0.2 um extrusion did not fix the
  discrepancy.
- The same coarse sheet represented independently as a 0.5531 ohm/square `Impedance` boundary, job
  `01a05d4e-f22c-75c0-bda5-2a724c77d912` (`palace-1a559c2d`), completed in 9m06s. S21 changed by no more than 0.067 dB,
  ruling out a defect specific to the finite-thickness conductivity formula.
- HXT 150/150 sheet convergence job `01a05d49-319f-7123-bc21-15fc2ccb54fa` (`palace-0e2f581d`) completed in 17m48s on
  87,475 tetrahedra. Its S21 was -5.25/-28.65/-46.16 dB, a +0.012/+0.581/-6.779 dB change from the 47,844-tet sheet
  result. The capped full-chip model is not converged at 20 GHz and remains about 19 dB below NIST at 10.05 GHz. Do not
  launch a production sweep or claim parity.

The completed diagnostics are reciprocal and passive to their solver accuracy, but several linear and error-indicator
solves terminate above the requested tolerance. The next investigation should isolate one uniform NIST cross-section as
a genuine 3D Palace line, compare its extracted propagation to the published section RLCG, and only then return to the
complete nine-section chip. That work requires a new cloud-job authorization.

The notebook enforces the tetrahedron budget after meshing and immediately before cloud submission. Independently
recount the final staged `.msh` as well.

## Repository layout to add

```text
nbs/palace_eic_ihp_hfss.ipynb
nbs/palace_eic_nist_cpw.ipynb
tests/palace/test_eic_reference_data.py
tests/palace/test_eic_geometry.py
tests/data/eic/<small, redistributable reference files>
```

Keep geometry/reference parsing in small Python modules if notebook code becomes reusable. Do not commit large vendor
archives or IHP artifacts until their redistribution license is confirmed; pin downloads and verify SHA-256 instead.

## 1. IHP SG13G2 HFSS inductor

Sources, pinned at commit `a98509eb57989f60e7965bee88214d081d53c27e`:

- [Artifact directory](https://github.com/SkillSurf/frac-n-pll-vco-smacd_2026/tree/a98509eb57989f60e7965bee88214d081d53c27e/HFSS%20Inductor%20Files)
- Native model: `HFSS_Parametric_Model/Feb_01st.aedt`, SHA-256
  `3a3c0980c55631250063b922c2681fb451add3c199275c2759193e0175331454`
- Layout: `Results/Results for the 2port inductor/4nH_2port_test1.gds`, SHA-256
  `6e0a9573bcba530103fea9bd4da0df753848d10a6268a2c05a6494b0cfe8a97b`
- Reference: `Results/Results for the 2port inductor/realimaginary.s2p`, SHA-256
  `1db51374a3980bf6f24c99b84a4f6f7937b077483bac6575d18234d2ff2b50d3`
- Stack: `IHP PDK xml file.xml`, SHA-256 `c610fd87231debde837e8ab3c66842b407e4c274387d5fb594b32171bd695cd0`

Reference case:

- Three-turn octagonal differential inductor in IHP SG13G2.
- Frozen variables: `F_S=40 um`, `R_out=213 um`, `Theta=22.5 deg`, `W=30 um`, `S=11 um`; GDS bounding box is about
  `456 x 496.216 um`.
- Stack highlights: 280 um silicon (`er=11.9`, `sigma=2 S/m`), 3.75 um epi (`er=11.9`, `sigma=5 S/m`), 15.7303 um SiO2
  (`er=4.1`) and air. Use the native `EMDesign2` stack and z extents wherever other artifacts differ.
- Metal conductivities include Metal1 `2.164e7 S/m`, TopMetal1 `2.78e7 S/m`, TopMetal2 `3.03e7 S/m`, TopVia1
  `2.191e6 S/m`, and TopVia2 `3.143e6 S/m`.
- HFSS ports are two 50-ohm vertical single-strip gap ports referenced to `Metal1:rect_20`. Preserve their exact
  rectangles and reference conductor.
- Reference sweep: 41 points, 1.5–3.5 GHz, 0.05 GHz spacing, 50 ohms.
- At 2.45 GHz: `S11=0.2732186941+j0.3886981271`, `S21=S12=0.6971307009-j0.4564828412`, `S22=0.2727959136+j0.3903455789`;
  `Ldiff=4.005845 nH`, `Qdiff=16.2104`.

Acceptance gate: the Touchstone header omits its HFSS design name. Before calling this authoritative, reopen
`Feb_01st.aedt`, select `EMDesign2`, apply the variables above, re-export the sweep, and verify it matches
`realimaginary.s2p`. Do not substitute the paper's later `S=14 um` OpenEMS design or `Final_Inductor_Layout.gds`.

Implementation notes:

- Import the frozen GDS into a component and construct a custom `LayerStack`.
- Use finite metal conductivity with `planar_conductors=False`; verify generated Palace configuration contains
  `Boundaries.Conductivity` with thicknesses.
- Model the HFSS gap ports as vertical lumped ports only after checking their surfaces against `EMDesign2`.
- First compare all four complex S-parameters; derive Ldiff and Qdiff second.

## 2. NIST Parylene-C platinum CPW

Official sources:

- [Simulation dataset, DOI 10.18434/mds2-2817](https://doi.org/10.18434/mds2-2817)
- [Measurement dataset, DOI 10.18434/mds2-2808](https://doi.org/10.18434/mds2-2808)
- [NIST paper](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=935729)

Use the air-filled, 2.5 um-gap, 6.5 mm H2O-chip variant first:

- CPW: 50 um signal, 2.5 um gaps and 200 um ground widths.
- Pt: `sigma=9.04e6 S/m`; use the ANSYS model's 0.2 um thickness for solver parity. Use the measured `405 +/- 5 nm`
  thickness only in a separate measurement comparison.
- Substrate: 0.5 mm fused silica, `er=3.82`.
- Parylene C: 6.67 um, `er=3`, `tan_delta=0.002`.
- PDMS: `er=2.77`, `tan_delta=0.054`; central channel is air.
- Symmetric section lengths in um from one end to the centre: `60 + 690 + 732.4 + 420.1 + 2695.5/2`, then mirror.
- `air_data_vs_sim.mat` contains 638 complex simulated points from 40 kHz to 110 GHz; the raw measurement archive
  contains complex `.s2p` data.

Implementation notes:

- Reproduce `constructSfromSim.m` locally and test the imported complex arrays before building the Palace model.
  Preserve column order `S11,S12,S21,S22`.
- Use two two-element CPW lumped ports at the physical ends, normalized to 50 ohms. A numeric wave port would need to
  span the full substrate/backplane cross-section; the localized wave-port diagnostics are invalid.
- Use the complete 6.5 mm structure, not a periodic or shortened surrogate.
- Start the Palace comparison at 0.1–20 GHz. Expand only after the mesh study.
- Compare the 0.2 um model to ANSYS and the 405 nm model to measurement as two separately labelled experiments.

## Execution order and budget

1. Add pinned reference download/loaders, hashes and unit tests; no cloud jobs.
1. Build both geometries and inspect GDS, stack, ports, mesh groups and generated Palace JSON locally; no cloud jobs.
1. Run one coarse three-frequency smoke test per structure in parallel: 2 jobs.
1. Run one production sweep per structure in parallel: 2 jobs.
1. Use at most one targeted mesh refinement per structure if needed: up to 2 jobs.

The original target was 4 jobs with a planning ceiling of 6. The later user-approved ten-job NIST debug budget is
exhausted. Abort or coarsen any future setup projected to exceed 30 minutes, and obtain a new authorization before
submitting it.

## Validation and completion criteria

- Record source URLs, commit/DOIs, SHA-256 hashes, geometry, stack, ports, solver settings, gsim version and cloud job
  IDs.
- Compare complex `S11`, `S12`, `S21`, and `S22` on identical frequencies and 50-ohm normalization. Report magnitude and
  phase plus max/median `abs(delta S)`.
- Check reciprocity, passivity and energy balance before interpreting material loss.
- Demonstrate mesh convergence separately from agreement with the reference.
- Do not tune dimensions or conductivity to fit results until port placement, normalization, reference planes and mesh
  convergence are verified.
- Completion means both notebooks run from clean inputs, reproduce their stored reference checks, submit Palace runs and
  emit concise parity plots/tables.
