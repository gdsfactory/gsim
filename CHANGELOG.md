# Changelog

## Unreleased

- Silicon in the O-band: the stack material `silicon` gets a Li (1980) model at 293 K, valid 1.2-14 um, computed from
  the existing `si_li_293k` coefficients through a new `cauchy` dispersion type (eps = eps_inf + sum A_k / lambda^(2k)).
  Below 1.36 um silicon previously fell back silently to eps = 11.9 and the RF conductivity of 2 S/m. Optical models now
  drop the base conductivity, a `DispersionCoverageWarning` is emitted when no model covers an optical or infrared
  wavelength (up to 100 um), and MEEP renders a Cauchy-only wavelength non-dispersive with a warning.
- EMX process-file import: `load_emx_proc` builds a `LayerStack` from an EMX `.proc` file (dielectrics, conductors,
  vias, GDS layer map, sheet resistance converted to conductivity); unsupported features (bias, fill, via merge,
  temperature dependence) are skipped with an `EmxImportWarning`.
- CPW de-embedding notebook ([#341](https://github.com/gdsfactory/gsim/issues/341)): physical RLGC from the de-embedded
  propagation constant plus impedance anchors, in one joint passive-line fit of both sections (wave ports:
  `BoundaryModeSim` voltage-power impedance at 10, 25, 50, 75 and 100 GHz; lumped ports: low-frequency de-embedded
  |Zc|), instead of reading the IEEE P370-biased Zc section by section, which gave negative G at high frequency. The
  notebook now explains the P370 split bias and the assigned wave-port reference, and regression tests reproduce the
  bias with synthetic lines.
- Palace AC circuit synthesis ([#272](https://github.com/gdsfactory/gsim/issues/272)):
  `set_driven(..., circuit_synthesis=True)` emits `AdaptiveCircuitSynthesis` for adaptive driven sweeps (requires
  `adaptive_tol > 0`), and the new `gsim.palace.circuit` module parses the exported `rom-*.csv` matrices into a
  `CircuitSynthesis` object with `Y(ω)` assembly, port-admittance condensation via Schur complement, port-load
  subtraction, and S/Z/Y access. The same module provides the reusable EM-to-circuit fit: `fit_rlc` with `model="rlc1p"`
  (one-pole R, L, C, f0, Q; JAX/Adam in log space or a scipy fallback) and `model="vector_fit"` (scikit-rf VectorFitting
  multi-pole rational model with stability/passivity test and enforcement, spurious-pole detection and SPICE subcircuit
  export via `VectorFit`). `CircuitSynthesis.fit_rlc()` fits either model to the exported circuit in one call; one-port
  circuits use the driving-point impedance.
- Notebook refactor to the reusable workflow: `palace_inductor.ipynb` fits via `fit_rlc` (the hand-rolled JAX/Adam
  section is gone) with `SParams.to_skrf()` + `differential_impedance` for S/Z access;
  `palace_inductor_port_comparison.ipynb` drops its manual `s_to_z` in favour of the package helpers;
  `palace_transformer.ipynb` uses `to_skrf()` for S/Z/Y access, adds a broadband vector-fit section with passivity
  checks and SPICE export, and its circulax fit netlist is migrated to the SAX port-reference style required by circulax
  0.2.3.
- **Behavior changes called out for review** (beyond circuit synthesis; happy to split into a separate PR if preferred):
  (a) the prebuilt local Palace runtime default moved from v0.17.0 to v0.18.0 (set `PALACETOOLKIT_PALACE_CPU_TAG` to
  stay on 0.17.0); (b) `SParams.to_skrf()` now defaults to the reference impedance recorded in `port_information.json`
  instead of a hardcoded 50 Ohm, and S-parameter plots label it (#74). All new functionality lives in the single module
  `gsim.palace.circuit`.
- Notebook refactor to the reusable workflow: `palace_inductor.ipynb` fits via `fit_rlc` (the hand-rolled JAX/Adam
  section is gone) with `SParams.to_skrf()` + `differential_impedance` for S/Z access;
  `palace_inductor_port_comparison.ipynb` drops its manual `s_to_z` in favour of the package helpers;
  `palace_transformer.ipynb` uses `to_skrf()` for S/Z/Y access, adds a broadband vector-fit section with passivity
  checks and SPICE export, and its circulax fit netlist is migrated to the SAX port-reference style required by circulax
  0.2.3.
- First-class S\<->Z\<->Y conversion utilities (`gsim.palace.parameters`): batched
  `s_to_z`/`z_to_s`/`s_to_y`/`y_to_s`/`z_to_y`/`y_to_z` with explicit scalar or per-port reference impedance, preserved
  frequency units and port order; incomplete matrices are rejected. `palace_inductor_port_comparison.ipynb` is merged
  into `palace_inductor.ipynb` as a controlled two-interlayer-vs-gap-port comparison section (identical guard ring)
  using these conversions.
- Touchstone export/import with full-fidelity round trips (`SParams.write_touchstone` / `SParams.from_touchstone`):
  frequency in Hz, port order and names preserved via `! Port[i]` comments, reference impedance restored from the file
  header; complex-S round-trip error at machine precision (below the 1e-9 acceptance for 2- and 4-port networks,
  tested).
- Reference impedance now flows from `port_information.json` into `SParams.z0` (and the npz cache), and S-parameter
  plots label it (#74).
- `palace_transformer.ipynb`: the extracted transformer is demonstrated inside a matching network — a series input
  capacitor swept around the analytic estimate drives a 50 Ohm load through the fitted circulax model, closing the EM ->
  extracted-parameters -> circuit-design loop.
- Gap-port comparison leg: committed local Palace v0.18.0 outputs (rom matrices, port-S and provenance) under
  `nbs/data/inductor/circuit_synthesis_gap/`; the committed caches are intentionally tracked so the notebooks'
  circuit-synthesis sections replay in CI without re-running Palace..
- PN-junction depletion model from Sze *Physics of Semiconductor Devices* (`PNJunctionConfig`,
  `make_pn_junction_profile`): computes built-in voltage, depletion width `W` (abrupt or linearly graded), asymmetric
  P/N split `x_p`/`x_n`, and capacitance `C_j = eps_s A / W`. The depletion region is represented automatically — meshed
  as a dielectric strip in high-res mode when `W >= ~1/5` of the flanking doped sections, otherwise applied as a lumped
  Impedance boundary via `sim.set_pn_junction()`. The 2D TWMZM demo now illustrates both modes.
- Fix: `build_doped_cross_section()` now registers doping/rib materials on `stack.materials`; previously doped domains
  silently resolved to eps=1.0 without conductivity in generated Palace configs.
- Consolidation: `common/stack/junction.py` + `common/stack/doping.py` merged into `common/stack/pn_junction.py`;
  `test_junction_physics.py`, `test_junction_profile.py` and `test_pn_junction_modes.py` merged into
  `tests/common/test_pn_junction.py`. Import from `gsim.common.stack.pn_junction` (re-exported at `gsim.common.stack`).
- 1D Sze-based complex permittivity (`carrier_profile_1d`, `epsilon_eff_relative`, `optical_params`,
  `junction_epsilon_profile`): depletion-approximation carrier profile plus full Drude plasma dispersion at optical
  wavelengths, with `Re(eps) -> Permittivity` / `Im(eps) -> Conductivity` mapping for Palace. At `1e18 cm^-3` the
  quasi-neutral rib carries `Δn ≈ -1e-3` (`σ ≈ 0.5 S/m`) while the depletion slice stays at the Sellmeier background.
- Segmented optical junction (`make_segmented_junction_profile`): bins each rib half into uniform strips
  (`p_1..p_N`/`n_1..n_N`, junction-outward), each sampling the 1D permittivity at its centre. The 2D TWMZM demo's
  optical run now uses 8+8 strips instead of a homogeneous body; `build_optical_cross_section()` accepts per-region
  `device_materials`/`extra_materials` to support it.

## 0.1.0

- Electrostatic simulation end-to-end for Palace ([#146](https://github.com/gdsfactory/gsim/pull/146))
- 2D Palace BoundaryMode solver support ([#150](https://github.com/gdsfactory/gsim/pull/150))
- Frequency-dependent material dispersion and API refactor ([#143](https://github.com/gdsfactory/gsim/pull/143))
- Explicit Material with refractive_index support in notebooks and improved mesh refinement
  ([#155](https://github.com/gdsfactory/gsim/pull/155))
- Interactive Plotly 2D plots with layer toggle for Meep ([#157](https://github.com/gdsfactory/gsim/pull/157))
- Simulation.run_local() for MEEP ([#144](https://github.com/gdsfactory/gsim/pull/144))
- Curved-element meshing and Palace 3D photonics example ([#131](https://github.com/gdsfactory/gsim/pull/131))
- Decimate tolerance, verbosity, and stale tag fix for meshing ([#145](https://github.com/gdsfactory/gsim/pull/145))
- Dark theme toggle and gdsfactory header link in docs ([#177](https://github.com/gdsfactory/gsim/pull/177))
- Add jupytext sync for notebook diffs ([#135](https://github.com/gdsfactory/gsim/pull/135))

### Bug Fixes

- Notebook rendering: LaTeX math delimiters and widget outputs ([#185](https://github.com/gdsfactory/gsim/pull/185))
- Case-insensitive material override and overlay matching ([#163](https://github.com/gdsfactory/gsim/pull/163),
  [#172](https://github.com/gdsfactory/gsim/pull/172))
- Restore air domain in CPW test fixtures via set_airbox ([#165](https://github.com/gdsfactory/gsim/pull/165))
- Slice animation/diagnostics at core layer, not stack midpoint ([#164](https://github.com/gdsfactory/gsim/pull/164))
- Correct metal_tags typing to satisfy ty ([#152](https://github.com/gdsfactory/gsim/pull/152))
- Align PDK stack defaults ([#151](https://github.com/gdsfactory/gsim/pull/151))
- Replace run_local with run in notebooks, add inductor to docs ([#140](https://github.com/gdsfactory/gsim/pull/140))
- Reorder pyproject.toml sections to satisfy tombi-format ([#139](https://github.com/gdsfactory/gsim/pull/139))
- Force utf-8 on write_text so Windows cp1252 doesn't break output ([#127](https://github.com/gdsfactory/gsim/pull/127))
- Enforce cp1252 compatibility in Python sources ([#125](https://github.com/gdsfactory/gsim/pull/125))
- Run waveport on cloud and build width sweep notebook ([#170](https://github.com/gdsfactory/gsim/pull/170))
- Tighten cloud sim S-param tolerances to absolute 0.01 ([#178](https://github.com/gdsfactory/gsim/pull/178))

### Documentation

- Migrate docs from mkdocs to zensical ([#176](https://github.com/gdsfactory/gsim/pull/176))
- Transmon qubit example with inductance port ([#110](https://github.com/gdsfactory/gsim/pull/110))
- Palace driven simulation for spiral inductor with guard ring ([#132](https://github.com/gdsfactory/gsim/pull/132))
- T-Bar CPW electrode MZM example ([#129](https://github.com/gdsfactory/gsim/pull/129))
- RLC model fitting to inductor notebook ([#154](https://github.com/gdsfactory/gsim/pull/154))
- Use explicit get_stack() from PDK in Palace notebooks ([#134](https://github.com/gdsfactory/gsim/pull/134))

### Maintenance

- Per-PR sim_smoke_test cloud check ([#160](https://github.com/gdsfactory/gsim/pull/160))
- Claude Code PR review workflows ([#159](https://github.com/gdsfactory/gsim/pull/159))
- Add cdaunt, flaport, and das-dias to CODEOWNERS ([#141](https://github.com/gdsfactory/gsim/pull/141))
- Clean up PEC block test marks ([#142](https://github.com/gdsfactory/gsim/pull/142))

## 0.0.16

- Replace Unicode arrow with ASCII for Windows compatibility ([#123](https://github.com/gdsfactory/gsim/pull/123))

## 0.0.15

- XZ 2D FDTD with fiber source and grating-coupler notebook ([#120](https://github.com/gdsfactory/gsim/pull/120))

## 0.0.14

### New Features

- Auto-size mesh, always refine ports, improve CPW defaults ([#111](https://github.com/gdsfactory/gsim/pull/111))
- 2D effective-index MEEP simulation mode ([#108](https://github.com/gdsfactory/gsim/pull/108))
- Code coverage with Codecov ([#103](https://github.com/gdsfactory/gsim/pull/103))
- Test workflow ([#101](https://github.com/gdsfactory/gsim/pull/101))
- Wave port support ([#53](https://github.com/gdsfactory/gsim/pull/53))
- Surface booleans via occ.cut() ([#79](https://github.com/gdsfactory/gsim/pull/79))
- Offset parameter for inplane and via ports ([#106](https://github.com/gdsfactory/gsim/pull/106))

### Bug Fixes

- Silence ty warnings in viz.py and rename 2D page to 2D FDTD ([#118](https://github.com/gdsfactory/gsim/pull/118))
- Log z_crop application instead of silently rewriting layers ([#116](https://github.com/gdsfactory/gsim/pull/116))
- Auto-size mesh for all presets, detect CPW gap widths ([#114](https://github.com/gdsfactory/gsim/pull/114))
- Respect PDK layer_type metadata in extract_layer_stack ([#105](https://github.com/gdsfactory/gsim/pull/105))
- Correctly re-identify conductor volumes after dedup ([#104](https://github.com/gdsfactory/gsim/pull/104))
- Handle 403 Forbidden for accounts without cloud sim ([#99](https://github.com/gdsfactory/gsim/pull/99))
- Handle transient HTTP errors in job polling loop ([#98](https://github.com/gdsfactory/gsim/pull/98))
- Add nbformat dependency for plotly notebook rendering ([#97](https://github.com/gdsfactory/gsim/pull/97))
- Add nest_asyncio2 for PyVista trame in VS Code ([#95](https://github.com/gdsfactory/gsim/pull/95))

### Refactoring

- Consolidate MeshConfig/MeshResult, drop pipeline.py ([#115](https://github.com/gdsfactory/gsim/pull/115))
- Drop graded preset and PEC refinement flag ([#113](https://github.com/gdsfactory/gsim/pull/113))
- Remove redundant tests, add workflow integration tests ([#102](https://github.com/gdsfactory/gsim/pull/102))

## 0.0.13

### New Features

- Energy decay stopping + fix OR condition bug ([#93](https://github.com/gdsfactory/gsim/pull/93))
- Interactive Plotly S-parameter plotting for Palace ([#89](https://github.com/gdsfactory/gsim/pull/89),
  [#75](https://github.com/gdsfactory/gsim/pull/75))
- Add sparams_path ([#81](https://github.com/gdsfactory/gsim/pull/81))

### Bug Fixes

- Use PLOTLY_RENDERER env var for interactive charts in CI ([#77](https://github.com/gdsfactory/gsim/pull/77))
- Render interactive Plotly charts on GitHub Pages ([#76](https://github.com/gdsfactory/gsim/pull/76))

### Refactoring

- Remove MPI process count from client API ([#92](https://github.com/gdsfactory/gsim/pull/92))

## 0.0.12

- Port name mapping for Palace S-parameter results ([#73](https://github.com/gdsfactory/gsim/pull/73))

## 0.0.11

### New Features

- Via volume meshing with fragment-based boolean pipeline ([#69](https://github.com/gdsfactory/gsim/pull/69))

### Bug Fixes

- Revert Netgen meshing for via ports ([#71](https://github.com/gdsfactory/gsim/pull/71))
- Resolve type errors for ty 0.0.25 ([#70](https://github.com/gdsfactory/gsim/pull/70))

## 0.0.10

- Auto-label PRs for categorized release notes
- Live log streaming for cloud simulation jobs
- PEC block support for Palace simulations
- Field saving parameters for Palace simulations
- Fix CPW bug phase shift

## 0.0.9

- Default CPW port length to 0.1 um
- Add missing trame core package to dependencies
- Fuse overlapping same-layer surfaces before extrusion in Palace mesh

## 0.0.8

Initial packaged release with core Palace and Meep simulation support.

## 0.0.6

Initial development release.
