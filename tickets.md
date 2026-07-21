# Tickets: Meep mode solver cloud execution

Default `sim.solve_modes()` to cloud execution with `.npy` field reconstruction, local fallback via
`sim.solve_modes_local()`.

Work the **frontier**: any ticket whose blockers are all done.

## Cloud slab mode solver with full fields

**What to build:** Run slab eigenmode solves on the cloud and get back `ModeSweepResult` with full field arrays usable
for `result.plot_mode()`. Slab mode is the simplest path — no GDS, no component, no cross-section geometry.

**Blocked by:** None — can start immediately.

- [x] `ModeSolverConfig` pydantic model serializes slab parameters: wavelengths, bands, parity, resolution, PML
  thickness, z_margin, background_material, eigensolver_tol, n_field_z, layer_stack, materials
- [x] `generate_meep_mode_solver_script()` produces a self-contained `run_meep.py` that reads `mode_solver_config.json`,
  builds a 1D slab MEEP cell, calls `get_eigenmode()` per wavelength/band, saves `mode_results.json` (metadata) and
  `mode_f_band{N}_wl{W}.npy` (field arrays)
- [x] `_parse_meep_result` dispatches: `mode_results.json` present → `ModeSweepResult` with reconstructed `ModeResult`
  entries; otherwise falls through to `SParameterResult.from_csv()`
- [x] Cloud-only end-to-end test: build config, upload, wait, parse → `ModeSweepResult` with non-empty `.results` and
  `fields` dicts

## Cloud cross-section mode solver

**What to build:** Extend the cloud mode solver to handle YZ and XZ 2D cross-section eigenmode solves. Geometry is
pre-computed client-side into `CrossSectionGeometry` and serialized in the config — no GDS upload needed by the runner.

**Blocked by:** Cloud slab mode solver with full fields

- [x] `ModeSolverConfig` extended with cross-section fields: `CrossSectionGeometry` with pre-computed
  `CrossSectionBlock` list, `n_field_x`, `n_field_y`
- [x] Runner script extended to build YZ/XZ 2D cells from serialized `cross_section_geometry` blocks (no gdsfactory
  needed on cloud)
- [x] End-to-end test with a component, port, and cross-section solve → `ModeResult` with 2D field arrays and correct
  `cross_section_plane`

## `solve_modes()` cloud default + `solve_modes_local()` fallback

**What to build:** Wire the cloud pipeline into `Simulation.solve_modes()` so the user gets cloud execution by default.
Add `solve_modes_local()` as the opt-in local path. The existing low-level functions (`solve_slab_mode`,
`solve_cross_section_mode`, etc.) remain unchanged.

**Blocked by:** Cloud slab mode solver with full fields, Cloud cross-section mode solver

- [x] `Simulation.solve_modes()`: build mode solver config, write to tmp dir, upload, start, poll, download, parse →
  `ModeSweepResult`
- [x] `Simulation.solve_modes_local()`: existing local meep behavior, moved from the current `solve_modes()` body
- [x] `Simulation.write_mode_solver_config()`: writes `mode_solver_config.json` with optional `cross_section_geometry`
  and `run_meep.py` (mode solver variant); no GDS needed
- [x] Solver type dispatch: `_parse_meep_result` detects `mode_results.json` → `ModeSweepResult`, `s_parameters.csv` →
  `SParameterResult`
- [x] `solve_modes()` uses `gcloud` pipeline directly (tempdir → write_mode_solver_config → upload → start →
  wait_for_results)
