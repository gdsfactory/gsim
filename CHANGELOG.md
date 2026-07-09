# Changelog

## 0.1.0

### New Features

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
