# Changelog

## Unreleased

- `BoundaryModeSim` owns its run and its postprocessing paths. `add_impedance_path(name, voltage=..., current=...)`
  declares the line integrals Palace evaluates on a solved Mode and returns the index the impedance is reported under;
  `run_local()` clears the previous run's tables, runs, and returns this run's `PalaceTextResults`; `read_results()`,
  `last_run_files` and `read_mode_field()` read that run back. **Behaviour change:** the boundary-mode simulation takes
  no 3D ports any more: `add_port`, `add_cpw_port`, `add_wave_port` and `add_terminal` refuse, and the `ports`,
  `cpw_ports`, `wave_ports` and `terminals` lists are gone from it. `PortConfig` and `CPWPortConfig` go back to
  describing a 3D excitation port only — the `voltage_path`, `current_path`, `nsamples`, `center`, `orientation`,
  `width` and `order` fields, the path derivation from a port's geometry, and the matching `add_port` / `add_cpw_port`
  keyword arguments are removed. Mode paths are `(h, v)` cross-section coordinates; nothing is projected from layout
  coordinates. `nbs/palace_cpw_deembedding.ipynb` and `nbs/palace_2d_twmzm.ipynb` declare their gap and junction paths
  this way, under the same indices they read back.

- A native-2D cross-section can be windowed.
  `BoundaryModeSim.set_cross_section(plane, window=(h_min, h_max), window_z=(z_min, z_max))` clips the meshed domain to
  a sub-region of the section instead of the whole component bounding box plus margins: the per-layer section rectangles
  are clipped to the window and the background box is sized to it, so one component can feed differently sized domains
  to different solvers.

- A native-2D mesh can hold an element size over an area.
  `BoundaryModeSim.mesh(refinement_boxes=[(h_min, h_max, z_min, z_max, size), ...])` keeps the elements inside each box
  (um) to `size`: refinement lines size the elements on them only, and the size grows at once with the distance, so a
  region that has to stay resolved — a depletion edge, a gap — needs a box rather than a line. Passing boxes to a 3D
  mesh is refused.

- A native-2D mesh names its contacts and interfaces. `BoundaryModeSim.add_contact(name=, layer_a=, layer_b=)` and
  `add_interface(...)` declare a layer pair, and mesh generation tags the curves the two layers' regions share as a
  dim-1 physical group carrying that name, listed under `mesh_groups`, so a charge-transport solver can bind to it. A
  pair that does not touch, or names an unknown layer, raises naming the pair. `contact_specs` and `interface_specs`
  list what was declared. `ContactSpec` and `InterfaceSpec` share a private `_LayerPairSpec` base: an Interface carries
  no terminal voltage, so neither is a substitute for the other and the base stays private.

- Fix: the native-2D mesher dropped every perfect-conductor and conductivity curve it found. It keeps a conductor's
  outline curve only when an adjacent surface carries a volume physical group, and read that adjacency out of the wrong
  half of gmsh's `getAdjacencies` result — the curve's end points rather than its adjacent surfaces — so the test never
  matched and the group came out empty. A perfect-conductor electrode was therefore meshed as an unconditioned slot
  rather than as a conductor.

- `gsim.palace.mode_fields` reads a saved Palace boundary Mode's fields back off disk (`load_boundary_mode_field`,
  `BoundaryModeField`, both exported from `gsim.palace`) and runs the Marks-Williams power-current integrals on them. It
  reads the two transverse components in the order Palace writes them, and its power-current impedance flips a Mode the
  solver chose to propagate against the plane normal, whose Poynting flux — and so whose reported `Z0` — would otherwise
  come out negative. `check_field_is_the_mode` is public, beside `field_index_ratio` and `FIELD_INDEX_RTOL`: asking a
  saved field whether it is the Mode it was fetched for is a statement about the field alone.

- `gsim.palace.line_impedance` holds the whole line-impedance story as one unit: `ImpedancePaths`,
  `line_impedance_paths` and `declare_impedance_paths` place a voltage path across the gap and a current loop around the
  signal conductor from the two conductors' extents, `native_line_impedance` reads Palace's own `mode-Z.csv` tables back
  with the wall-Mode diagnostic the gap voltage gives (`MIN_VOLTAGE_POWER_RATIO`), `field_line_impedance` integrates the
  saved fields, and `palace_line_impedance` is the tables-first, fields-as-fallback policy. It is not re-exported from
  `gsim.palace`: the functions are one policy and are meant to be read together. They take a Mode as its `mode_id` and
  `n_eff`, so nothing in the module knows who is asking. `BoundaryModeSim.mesh_extent` is a property returning the
  meshed `(h, v)` extent, read off the simulation's own mesh.

- `gsim.common.modes` is the line-mode vocabulary the Backends share: `Conductor` and `Extent` describe a conductor pair
  on a cross-section, `LineReading` is what a solve reports for one Mode, `propagating_modes`, `common_mode_fraction`,
  `wall_mode_from_currents` and `select_line_mode` pick the line Mode out of a solve — `select_line_mode` takes a
  `max_loss_ratio` bound on `|Im(n_eff)| / Re(n_eff)`, since a transmission line advances several radians of phase per
  radian of loss and the Modes just inside a bound of one are the discretization's — and
  `z0_power_current(power, current)` owns `Z_0 = 2 P / |I|^2`, the zero-current refusal and the sign flip a Mode saved
  travelling against the plane normal needs. `gsim.palace.mode_fields.z0_power_current` keeps its name and signature and
  now only reduces its own integrals and delegates; the quadrature stays with the Backend, the definition is what is
  shared.

- **Behaviour change:** every Palace simulation now diagnoses a binary that died on a signal, on both local paths — the
  streaming `verbose=True` default as well as the quiet one, which used to report a bare exit status.
  `PalaceSimMixin.run_local` wires `gsim.palace.runtime.local_abort_report` and takes a `remedy`, so `DrivenSim`,
  `EigenSim` and `ElectrostaticSim` raise `RuntimeError` where they used to surface a raw
  `subprocess.CalledProcessError` — a caller catching `CalledProcessError` around a Palace run stops catching it, and
  the original error is chained as the cause. The report names the binary that ran and what was being run (the frequency
  for a boundary-mode solve, the simulation type for the others), says whether the solver got as far as writing output —
  a signal death with none means the runtime, typically its bundled MPI, not the model — quotes the last worded stderr
  line, ends with the last 200 lines of the captured output, and names a `palace` launcher script that found no `mpirun`
  as such, with the install hint. `binary_that_ran`, `local_abort_report` and `require_palace_binary(hint=...)` live in
  `gsim.palace.runtime`.

- `BoundaryModeSim.run_local(salvage=True)` uses what a crashed run left on disk when it is everything the run was asked
  for: every Mode in `mode-kn.csv`, a `mode-V.csv` and `mode-Z.csv` row for each of them under every declared path's
  index, and a saved ParaView cycle for each Mode `save` asked for. Palace (0.17) intermittently corrupts its heap while
  shutting down a `BoundaryMode` solve (`free(): corrupted unsorted chunks`), after the solve has finished and written
  its results; the run directory is cleared before each run so what is read back is that run's, the salvage warns, and
  `salvage=False` keeps the raise reachable. A run that died short of any of it is the failure it looks like.

- `palace_line_impedance` reads a lossy conductor's impedance off the saved fields. Palace's `mode-Z.csv` holds
  magnitudes, and with finite conductivity `Z_VI² / Z_PV` is not the power-current impedance: 85.3 Ω off the tables
  against 74.6 − 8.3j Ω off the fields on a 10/6 µm aluminium CPW at 10 GHz, 14.5 % off, where PEC electrodes put all
  three readings within 4 %. A `"pec"` signal conductor still reads the tables; a `"volume"` one integrates the saved
  fields and keeps the tables for the wall-Mode diagnostic alone, and comes back NaN with a warning rather than the
  table value when the fields are not there.

- A Palace process that died on a Windows exception is reported by its NTSTATUS (`0xC0000005` as
  `STATUS_ACCESS_VIOLATION`) and blamed on the runtime like a signal death, instead of as "signal 3221225349";
  `_death_signal` reads `128 + signum` only within the shell's signal range. `load_boundary_mode_field` reads cell
  offsets off VTK's own array rather than pyvista's deprecated `offset`.

- `ModePath` refuses a path that repeats a point and opens a current loop handed in closed. Palace joins the last corner
  of a `CurrentPath` back to the first itself and aborts the whole run on a zero-length segment, so a loop written
  `[a, b, c, d, a]` crashed it at start-up with no log; it is now written as four corners, and `[a, b, b, c]` or
  `[a, b, a]` raise `ValueError`. The `current` docstring says why the loop should hug the conductor: it encloses the
  gap's displacement current along with the metal's own, so a loop standing off reads less current and a higher `Z_VI`;
  the generated `line_impedance_paths` loop agrees with the field-based power-current impedance.

- `PalaceTextResults.mode_voltage` reads `mode-V.csv` again. It was dropped as an unread parse — nothing in `src/` or
  `tests/` called it — but `nbs/palace_cpw_deembedding.ipynb` does, to pick the CPW mode out of the mode table by
  comparing the two signal-to-ground voltages and to reject the slotline-like mode whose gap voltages disagree in sign.
  `BoundaryModeSim.mode_postprocessing()` never stopped emitting the `Voltage` block, so Palace went on writing the file
  with nothing able to read it.

- One material-resolution rule. `gsim.common.stack.materials.resolve_stack_material(name, entry, wavelength_um)` owns
  the contested part every Backend had written out: validate the stack entry as an override so a user's `set_material`
  scalars and any dispersion model survive, fall through to the database when the entry is no valid `MaterialProperties`
  record, and answer `None` when neither resolves. Each Backend keeps its own policy on top — Palace skips a conductive
  material and leaves an unresolvable entry as the stack wrote it. `region_material_map`, which names a mesh region's
  material, sits directly above it.

- A simulation that delegates its meshing no longer has to hand-mirror the surface. `gsim.palace.base.MeshSourceMixin`
  carries the thirteen configuration and mesh accessors and the "call mesh() first" guard over an abstract
  `_mesh_source()` hook, for a simulation that owns a `BoundaryModeSim` rather than inheriting from one. The readers
  answer for an unconfigured simulation instead of building one to ask, and the guard names the concrete class.

- Two spec-validation rules are written once. `gsim.common.validation.AscendingInterval` is the annotated pydantic type
  behind every `(min, max)` window, so the rule travels with the field declaration instead of with a call a new field
  can forget; the message bodies are unchanged, and the field name moves into pydantic's error `loc`.
  `gsim.common.cross_section.parse_plane_spec` is the one reading of a `"x=<value>"` plane spec, behind
  `CrossSectionPlaneConfig.from_spec`; a bare `"x="` names no plane and is rejected.

- Optional dependencies are guarded in one shape. `gsim.common.optional.require_module(name, extra=..., hint=...)` holds
  the import, the message naming the packaging extra and the chained cause.
  `gsim.palace.runtime.require_palace_binary(hint=...)` is the same idea for an executable — it raises `RuntimeError`,
  not `ImportError`, since no `pip install` produces what it is missing.

- Tests that run the Palace binary locally carry the `palace_local` mark and are deselected by default, like
  `meep_local`; run them with `pytest -m palace_local`.

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
