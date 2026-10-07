# Changelog

## Unreleased

- `pip install 'gsim[tcad]'` and `'gsim[modulator]'` now give a DEVSIM that runs on Linux, macOS arm64 and Windows with
  no system BLAS/LAPACK and no Intel MKL. Both extras depend on `devsim-openblas`, which hands DEVSIM the OpenBLAS from
  `scipy-openblas32`, whose `scipy_`-prefixed symbols DEVSIM cannot load itself (proposed upstream as
  devsim/devsim#167). `require_devsim()` and `import_simple_physics()` configure it before DEVSIM's first import — the
  only moment DEVSIM reads `DEVSIM_MATH_LIBS` — unless `DEVSIM_MATH_LIBS` is already set, so a user who chose MKL or
  their own OpenBLAS keeps it. **Behaviour change:** a DEVSIM that still finds no BLAS/LAPACK now raises `ImportError`
  naming both remedies, chained to DEVSIM's `RuntimeError: Issues initializing DEVSIM.`, where the raw `RuntimeError`
  used to escape. The test guard `skip_without_devsim()` goes through `require_devsim()`, and the modulator end-to-end
  tests use it instead of `pytest.importorskip("devsim")`, which bypassed both.

- `ChargeTransportSim.reset_device()` releases its DEVSIM device through the module already in `sys.modules` instead of
  importing `devsim` again. A DEVSIM installed without its BLAS/LAPACK libraries raises
  `RuntimeError: Issues initializing DEVSIM.` partway through a one-shot C initialiser that has already declared its
  default derivatives, and leaves no `sys.modules` entry behind, so the next import redeclares them and raises the same
  error — out of a best-effort release path that only means to clean up, and that `except ImportError` never caught.
  With no DEVSIM module in the process there is nothing to delete, so the release is a no-op and the sim's solver state
  is forgotten as before.

- `PalaceTextResults.mode_voltage` reads `mode-V.csv` again. It was dropped as an unread parse — nothing in `src/` or
  `tests/` called it — but `nbs/palace_cpw_deembedding.ipynb` does, to pick the CPW mode out of the mode table by
  comparing the two signal-to-ground voltages and to reject the slotline-like mode whose gap voltages disagree in sign.
  `BoundaryModeSim.mode_postprocessing()` never stopped emitting the `Voltage` block, so Palace went on writing the file
  with nothing able to read it.

- **Behaviour change:** every Palace simulation now diagnoses a binary that died on a signal, not only
  `BoundaryModeSim`, and on both local paths — the streaming `verbose=True` default as well as the quiet one, which used
  to report a bare exit status. `PalaceSimMixin.run_local` wires `gsim.palace.runtime.local_abort_report` and takes a
  `remedy`, so `DrivenSim`, `EigenSim` and `ElectrostaticSim` raise `RuntimeError` where they used to surface a raw
  `subprocess.CalledProcessError` — a caller catching `CalledProcessError` around a Palace run stops catching it, and
  the original error is chained as the cause. Each sim names what was being run through `_run_context()`: the frequency
  for a boundary-mode solve, the simulation type for the others. `binary_that_ran` moves to `gsim.palace.runtime` beside
  the report it feeds.

- `docs/api/modulator.md` describes the modulator surface: the Study and its namespaced Stage sections, the device
  description, the result types and the presets, plus the three modules imported by path — `gsim.modulator.twmzm` for
  the MZM physics, `gsim.modulator.report` for what `Study.report()` returns, and `gsim.modulator.staircase`. It says
  what the narrowed `__all__` means for a caller: a Route is selected by name (`route="palace"`), because the Route
  classes are not exported.

- One material-resolution rule. `gsim.common.stack.materials.resolve_stack_material(name, entry, wavelength_um)` owns
  the contested part both Backends had written out: validate the stack entry as an override so a user's `set_material`
  scalars and any dispersion model survive, fall through to the database when the entry is no valid `MaterialProperties`
  record, and answer `None` when neither resolves. Each Backend keeps its own policy on top — femwell raises naming the
  region, Palace skips a conductive material and leaves an unresolvable entry as the stack wrote it.
  `region_material_map` moves to the same file, directly above it, and leaves `gsim.femwell.__all__` with no shim: it
  reads the stack and names mesh regions, with nothing femwell in it.

- One scatter interpolator behind every mesh-to-mesh sample.
  `gsim.common.interpolate.sample_at(points, values, targets, fill=...)` is the linear interpolant over a cloud's
  triangulation with nearest-neighbour or a scalar outside its hull, and it always returns the missing mask, so a caller
  with its own notion of missing — a region restriction, a per-column fill — composes with it instead of recomputing
  NaNs. `values` may be real or complex and `(n,)` or `(n, k)`, so several columns share one hull and one triangulation.
  `common.carrier_transfer`, `femwell.elementwise_epsilon` and the Staircase's band average now go through it, with
  behaviour preserved and two gaps closed: complex values were only handled in the femwell copy, and the
  degenerate-cloud guard only in the Staircase's. A cloud that spans no area now raises `DegenerateSampleCloudError`
  instead of being swallowed into a nearest-neighbour answer; the Staircase catches it and keeps returning `None`.

- A simulation that delegates its meshing no longer hand-mirrors the surface. `gsim.palace.base.MeshSourceMixin` carries
  the thirteen accessors and the "call mesh() first" guard over an abstract `_mesh_source()` hook, so
  `ChargeTransportSim` — which delegates to a lazily-created `BoundaryModeSim` rather than inheriting from one — keeps
  its delegate private and loses about 90 lines of mirroring. The readers answer for an unconfigured simulation instead
  of building one to ask, which is the behaviour the hand-written `has_mesh` had, and the guard still names the concrete
  class. tcad keeps `devsim_mesh_path`, which has no counterpart.

- `ContactSpec` and `InterfaceSpec` gain a private `_LayerPairSpec` base in the file they already share, keeping their
  own names and docstrings. They were field-identical and validator-identical; they are still different concepts — an
  Interface carries no terminal voltage — so the base is private and neither is a substitute for the other.
  `gsim.palace.mesh.generator`, which already treats them alike, types against the base.

- A sweep of points over one scalar key has one base. `gsim.common.sweep.ScalarSweep[PointT]` holds the key array and
  the tolerant `point_at` lookup that `CarrierResponseSweep` and `BiasSweepResult` each wrote out, with a `_key(point)`
  hook naming the swept axis; everything else about the two stays where it was, and `BiasSweepResult` keeps `voltages`
  as the named alias of `keys` so its surface is unchanged. `BIAS_TOL_V` moves there with it, which closes a layering
  edge: `gsim.modulator` imported a tolerance constant upward out of the tcad Backend. The bare `* 1e2` in
  `gsim.tcad.results` is now `PER_CM_TO_PER_M` — a DEVSIM 2D device is one cm deep, and per-cm to per-m is the one thing
  that factor was ever doing.

- Two spec-validation rules are written once. `gsim.common.validation.AscendingInterval` is the annotated pydantic type
  behind every `(min, max)` window: `tcad.doping`'s five hand-written `_validate_interval` calls and the identical loop
  inside Palace's `CrossSectionPlaneConfig` validator are gone, and the rule now travels with the field declaration
  instead of with a call a new field can forget. The message bodies are unchanged; the field name moves into pydantic's
  error `loc`, so assert on the body rather than on the whole rendered error.
  `gsim.common.cross_section.parse_plane_spec` is the one reading of a `"x=<value>"` plane spec, shared by
  `CrossSectionPlaneConfig.from_spec` and `gsim.modulator.Study.plane`. The Study's own `partition("=")` variant
  accepted a bare `x=`, naming no coordinate; it is now rejected where every other spec is.

- Optional dependencies are guarded in one shape. `gsim.common.optional.require_module(name, extra=..., hint=...)` holds
  the import, the message naming the packaging extra and the chained cause that `require_devsim`,
  `import_simple_physics`, `require_femwell` and `require_skfem` each wrote out. The per-backend guards keep their names
  and their messages: `require_devsim` still swallows DEVSIM's BLAS/UMFPACK banner, and the femwell pair is in
  `gsim.femwell.__all__`. `gsim.palace.runtime.require_palace_binary(hint=...)` is the same idea for an executable — it
  raises `RuntimeError`, not `ImportError`, and no `pip install` produces what it is missing, so it stays in the Palace
  package rather than joining `require_module`. It replaces the Palace Route's own wrapper; the Route keeps only the
  hint text naming the Stage that asked and the way back to `route='femwell'`, and
  `gsim.modulator.palace_route.require_palace_binary` is gone.

- One power-current impedance definition serves both Routes. `gsim.common.modes.z0_power_current(power, current)` owns
  `Z_0 = 2 P / |I|^2`, the zero-current refusal and the sign flip a Mode saved travelling against the plane normal
  needs; `gsim.palace.mode_fields.z0_power_current` and `gsim.femwell.adapter.z0_power_current` keep their names and
  signatures and now only reduce their own integrals and delegate. Each Backend keeps its own quadrature — Palace
  integrates nodal arrays on second-order Lagrange triangles in numpy, femwell assembles skfem forms over the Basis the
  solver still holds — so what is shared is the definition, not a field-sampler protocol that would put an skfem-shaped
  interface into `gsim.common`. Both arguments arrive in the Backend's own coordinate scale and the ratio cancels it, so
  the combiner does no unit work. `Conductor.model` now records that Palace's contour integral reads the same enclosed
  current for either model and so never branches on the field, while femwell's `electrode_current` does.

- Palace runtime knowledge lives in `gsim.palace`, not in the Palace Route. About 400 lines leave
  `gsim.modulator.palace_route`, and nothing under `gsim.palace` imports `gsim.modulator` to take them: every type the
  moved code needs — `Conductor`, `Extent`, `LineReading` — was already in `gsim.common.modes`.

  - New `gsim.palace.line_impedance` holds the whole impedance story as one unit: `ImpedancePaths`,
    `line_impedance_paths`, `PATH_CLEARANCE_FRACTION`, `IMPEDANCE_PORT`, `declare_impedance_paths`,
    `native_line_impedance`, `MIN_VOLTAGE_POWER_RATIO`, `field_line_impedance` and the tables-then-fields policy
    `palace_line_impedance`. It is not re-exported from `gsim.palace`, for the reason `mode_fields` gives about its
    integrals: the four functions are one policy and are meant to be read together. Splitting the table reader into
    `results.py` and the field reader into `mode_fields.py` would have pushed line-mode vocabulary — signal against
    return conductor, the gap voltage, the wall Mode — into two modules that know only tables and only fields, and left
    the policy with no home. The functions take a Mode as its `mode_id` and `n_eff` rather than as a `PalaceMode`, which
    is what keeps the new module free of the Route.
  - `BoundaryModeSim.mesh_extent` is a property returning `gsim.common.modes.Extent`: it is a meshio read of the
    simulation's own mesh, so it belongs on the simulation rather than in a module.
  - `gsim.palace.mode_fields.check_field_is_the_mode` is public, beside `field_index_ratio` and carrying
    `FIELD_INDEX_RTOL`, with the old `stage_name` argument generalised to `context`. Asking a saved field whether it is
    the Mode it was fetched for is a statement about the field alone. `field_line_impedance` still calls it outside its
    own `try`, so a caller running with warnings as errors sees the wrong-Mode diagnostic rather than an unreadable-file
    report.
  - `gsim.palace.runtime.local_abort_report` turns a dead Palace binary's exit status into something actionable — a
    signal death with no solver output means the runtime, typically its bundled MPI, not the model. It is named
    `local_abort_report` because it reads a `subprocess.CalledProcessError` and a local run directory, which makes it
    visible that a future cloud path is not covered. The one Route-shaped token, "re-solve on the default route with
    `route='femwell'`", is a caller-supplied `remedy`, optional because a Backend caller with no second way out should
    not have to invent one. What was being run is named by `during`, not `context`, which in `line_impedance` and
    `mode_fields` means who is asking.
  - `BoundaryModeSim.run_local` owns what a bad local run means: it salvages a complete mode table left by an abnormal
    exit (Palace 0.17 corrupts its heap on shutdown of a boundary-mode solve, *after* answering) behind `salvage=True`,
    and reports an aborted binary through `local_abort_report`. The salvage default sits on the Backend because the bug
    is a Palace fact, and `salvage=False` keeps the raise reachable. The report is wired into `BoundaryModeSim` only;
    wiring `PalaceSimMixin` would change the exception type on `DrivenSim`, `EigenSim` and `ElectrostaticSim` and is a
    behaviour change rather than a lift.

  `gsim.modulator.palace_route` keeps what names a Stage, a Window or a Staircase: `PalaceMode`, `PalaceSolve`,
  `solve_palace_modes`, `PalaceRoute`, `containment_unmeasurable`, `conductor_clearance` and `require_palace_binary`,
  and its `__all__` narrows to them. `solve_palace_modes` no longer wraps the run in crash handling; it hands its
  femwell remedy down instead.

- gmsh physical groups are read in one place: `gsim.common.mesh_regions` holds `group_tags` / `group_names` (the two
  directions of the `field_data` lookup, over an explicit `dim`), `cell_blocks` (the block-concatenation loop) and the
  triangle conveniences `element_regions`, `node_regions` and `region_elements` on top. The join between a group's name
  and its cells' tags had been written four times — twice inside `gsim.common.carrier_transfer`, twice in
  `gsim.femwell.adapter`, and once more at dim 1 in `gsim.tcad.mesh.line_group_points`, which now keeps its coordinate
  extraction and drops its copy of the mechanics. `region_elements` moves with them: it is meshio and gmsh only, with
  nothing femwell in it, so it is `gsim.common.mesh_regions.region_elements` and leaves `gsim.femwell.__all__` with no
  shim. `epsilon_by_region` and `elementwise_epsilon` stay in femwell, which owns the `exp(+i omega t)` convention they
  carry. Nothing joins `gsim.common.__all__`; the module is imported by path. A cell block the mesh gives no physical
  tag reads tag `0` — gmsh's "no physical group" — so a mesh written without `gmsh:physical` still yields its cells,
  naming no region and matching no group; `line_group_points` now says a mesh holds no line cells rather than returning
  no points.

- `gsim.common.modes` gives up its modulator vocabulary: `wall_mode_hint` is gone from the module and from `__all__`,
  and the sentence it returned is now `RFStage._wall_mode_hint`, beside the `window_hint` it reads like. It named a
  Stage and its settings, which `gsim.common` must not know about, and the RF Stage was its only caller. `Extent` — the
  `((h_min, h_max), (v_min, v_max))` rectangle the module already returns — joins `gsim.common.modes.__all__`, where it
  had been public in all but name.

- The Staircase is `gsim.modulator.staircase`, not `gsim.common.stack.staircase`, and `gsim.common.stack` no longer
  re-exports `staircase_profile` / `strip_averages_from_nodes`, so `import gsim.common` stops loading the Staircase at
  all. No Backend imported it: every consumer already sat under `gsim.modulator`, and what femwell, Palace and tcad mesh
  is the `LayerStack` that `StaircaseCrossSection.stack()` resolves to. The whole module moves in one piece, with no
  compatibility shims; with it modulator-side, its `build_doped_cross_section` import is no longer deferred to dodge a
  cycle. `surroundings_from_section` raises the new `CrossSectionOrientationError` naming the x-normal contract where a
  y-normal Cross-section used to reach an `AttributeError`.

- `gsim.modulator.__all__` drops `Stage`, `Route`, `FemwellRoute`, `PalaceRoute` and `DEFAULT_PALACE_STRIPS`, each of
  which stays importable from its own module (`gsim.modulator.stage`, `.route`, `.femwell_route`, `.palace_route`).
  Nothing outside the package subclasses or instantiates them: a Route is selected by name string (`route="palace"`).
  `StageNotRunError` stays public because callers catch it, and `EMRoute` stays because it is the literal type of a
  setting users pass.

- The line theory and the modulator physics are separate modules, and `gsim.common.twmzm` / `gsim.common.twmzm_report`
  are gone with no shims. `gsim.common.transmission_line` keeps what describes a transmission line with no modulator in
  sight — `RFLineParams`, `line_params_from_neff` / `line_params_from_gamma`, `rlgc_from_line_params`, `JunctionBranch`,
  `series_rc_from_admittance`, `loaded_line_params`, the new public `section_abcd`, and the segmented-electrode set
  (`segmented_period_abcd`, `segmented_line_params`, `segmented_line`, `bragg_fraction`). `gsim.modulator.twmzm` takes
  the MZM physics (`eo_response`, `eo_bandwidth`, the two walk-off limits, `vpi_length_vcm`, `mzm_transfer`,
  `mzm_transfer_figures`, `mzm_chirp`, `mzm_drive_range`, `segmented_eo_response`) and `gsim.modulator.report` what
  `Study.report()` returns (`TWMZMReport`, `OpticalPhaseSweep`, `LoadedLineComparison`, `twmzm_figures_of_merit`).
  `gsim.common.circuit` now imports the line theory at module level instead of lazily, its driven-line response and its
  S-matrix share one broadcast guard, and the driven response builds its telegrapher ABCD with `section_abcd` rather
  than rebuilding it.

- Strip averages weight area, the charge mesh resolves the depletion region, and the oxide is in the charge solve by
  default. Three changes that only hold together, and what followed from them:

  - `strip_averages_from_nodes` averages a band of the Carrier map over its area (the mean over the band's height of the
    cloud's linear interpolant, the field the continuous Route's transfer reads off the same nodes) instead of counting
    nodes. A charge mesh puts a third of its nodes on the silicon's top and bottom lines, which have no area: once the
    carriers vary in depth the node count over-read the carriers a Strip loses under bias by 33 to 80 %, and by more on
    a finer mesh. `staircase_profile` sums its trapezoids Strip by Strip, so a depleted Strip decades below its
    neighbours is no longer lost to cancellation. A band's `v_range` now takes the nodes within 1e-9 um of it
    (`BAND_TOL_UM`): a charge mesh hands the silicon's top surface over at 0.22000000000000003, and that row, where the
    depletion peaks, was dropped. A band is a rectangle of silicon: give a rib and the slab beside it a call each, as a
    Staircase's segments do.
  - The charge Stage holds its mesh to the refinement lines' size across the two Regions the Junction separates
    (`ChargeStage.junction_boxes()`, through the new native-2D mesh option
    `refinement_boxes=[(h_min, h_max, z_min, z_max, size)]` of `mesh()` / `MeshConfig`). The mesh pipeline sizes
    elements on its lines and lets them grow at once with the distance, so the depletion edge, 50 nm from the Junction
    under bias, sat in elements several times the refined size however fine the Junction line: the rib device's C_j at 0
    V read 328 pF/m at the default size and 315 at 5 nm, against a converged 289 (10 and 5 nm boxes agree to 0.5 %, and
    the silicon-only C(V) is now within 3 % of the depletion formula, from 7 %). No interpolation makes up for an
    unresolved edge: a linear one smears it (a depleted width of 25 nm for 90), a log-space one over-reads the carriers
    removed by 9 to 14 %. The charge solve costs about 1.5 times what it did. `refinement_boxes` in
    `study.charge(mesh=...)` replaces the box. The option is a `MeshConfig` setting like the others: validated there,
    kept across `mesh()` calls, meshed by `preview()` too, and refused on a 3D mesh.
  - The continuous optical solve meshes the same boxes: a Carrier map with a resolved edge, transferred onto elements
    several times coarser, read the index shift 15 % low (so V_pi L 15 % high) at the optical Stage's own 0.05 um.
  - `ChargeStage.oxide` defaults to `True` (it was opt-in until Strips averaged by area). C_j and the series-RC Junction
    branch count the field fringing around the Junction: 269 against 159 pF/m at 2 V on the rib device, a nearly
    bias-independent 100 to 120 pF/m. **Every capacitance, Junction branch and loaded-line figure of a Study moves with
    it**; `study.charge(oxide=False)` is the silicon-only solve. One limit comes with it: the Carrier map now varies in
    depth, a Strip is uniform in depth, and an optical Staircase spreads the surface depletion over the whole height,
    where the Mode is strongest. Against the continuous Route it reads the index shift 6 to 7 % high from 32 Strips up,
    and no Strip count removes that (silicon only it converges: -2.8 % at 32 Strips, -1.6 % at 256). The continuous
    Route, the optical Stage's default, is not affected.
  - The loaded-line cross-check is settled on it. With the oxide the two routes agree on the rib device to 2-5 % on n_RF
    and 10 % on |Z0| (shunt-C gap 10-40 pF/m, from 115-150 silicon only), and to 15 % and 23 % on the abrupt demo
    device; the loss gap stays at 33-41 % and is not explained. `LoadedLineComparison.check()` tightens its defaults to
    just outside that gap (`rtol_n_rf` 0.3 to 0.2, `rtol_alpha` 0.7 to 0.5, `rtol_z0` 0.45 to 0.3); a silicon-only
    charge solve needs the new `SILICON_ONLY_RTOL`.
  - An insulator Interface node of the mesh that matches no node of its two Regions is reported with a warning instead
    of being dropped; C_j with the oxide on moves by 0.4 % between 10 and 5 nm, which bounds the error of the nodes the
    binding leaves out on purpose.

- The charge Window can include the oxide around the Junction.
  `ChargeTransportSim.add_insulator(region=, relative_permittivity=)` solves Poisson in an insulating Region as well —
  the potential alone, continuous across the Interfaces it shares with the doped Regions, no carriers — and the charge
  Stage's `oxide` setting turns it on for the stack's oxide. The small-signal capacitance and the series-RC Junction
  branch then count the field fringing around the Junction: +79 to +96 pF/m on the abrupt demo device, 169.5 to 283.7
  pF/m at 2 V on the rib device (on the charge mesh of the time; the entry above has the converged figures). The Carrier
  map still holds the doped Regions only. It shipped opt-in and is on by default since Strips average by area (the entry
  above, which also settles the loaded-line cross-check this was a test of).

- A graded Junction on the demo Phase shifter. `rib_phase_shifter(lateral_straggle_um=...)` smears every doping step of
  the drawn device into an error function of that standard deviation, as implant straggle and diffusion do, and hands
  the Study the result through `Device.doping` as `TableDoping` profiles — one per dopant present in each Region, so
  donors and acceptors overlap across the Junction and compensate there. Zero, the default, keeps today's `StepDoping`
  exactly. The net doping changes sign at the drawn Junction when the two cores are doped alike, and a fraction of a
  straggle into the lighter one when they are not. The charge solve's mobility reads the total doping in the compensated
  zone, not the net; the RF conductivity keeps its stand-in for the impurities, the carriers themselves (`n + p`), which
  reads the net doping there, on silicon the reverse bias depletes. The TCAD TW-MZM notebook solves the graded device
  next to the abrupt one: a lower Junction capacitance and a higher Modulation efficiency figure, with the analytic
  depletion check left to the abrupt device it describes.

- A segmented Traveling-wave electrode, by fill factor. `study.line(fill_factor=, period_um=)` loads the electrode part
  of the way — loaded sections alternating with unloaded ones, which is how a modulator reaches 50 ohm — and the report
  becomes the periodic line's: Bloch impedance, RF index and loss (`TWMZMReport.z0_ohm`, and the new `n_rf` /
  `alpha_rf_np_m`), an EO response in which only the loaded sections modulate, and a Modulation efficiency divided by
  the fill factor; the Mach-Zehnder figures (`v_pi_v`, insertion loss, extinction ratio) are taken over the loaded
  length, `fill_factor * length`. A segmented electrode is the whole number of periods nearest `length_um`, in the
  report (`TWMZMReport.length_m`) and in the exports alike. The second line is the RF Stage's unloaded solve, run
  through the RF Stage when it holds no result; the Stage warns when the period approaches the Bragg condition inside
  the reported band. The pure functions are `segmented_period_abcd`, `segmented_line_params` (passive branch; within
  1e-3 of the length-weighted average of the two lines' series impedance and shunt admittance while `bragg_fraction`
  stays under 0.05), `bragg_fraction` and `segmented_eo_response`, and `twmzm_figures_of_merit` takes `unloaded=`,
  `fill_factor=`, `period_m=`. A fill factor of one — the default — is the report as it was, bit for bit. An
  approximation of the 3D structure: both sections share one electrode Cross-section, with no loading fins drawn.

- The report goes past the Phase shifter to the modulator. The line Stage puts the Phase shifter in both arms of a
  Mach-Zehnder and reports the static intensity transfer against the drive voltage (`TWMZMReport.drive_v`, `transfer`)
  and the datasheet numbers read off it — `v_pi_v` at this length, `insertion_loss_db`, `extinction_ratio_db` — plus the
  small-signal `chirp` per Bias point.
  `study.line(drive="push-pull" | "single-drive", arm_bias_v=, arm_imbalance_db=, phase_offset_rad=)` sets how the arms
  are driven, where they rest, how evenly the light is split and where the interferometer is parked (quadrature by
  default); the drive voltage is the voltage between the arms in both configurations. The extinction ratio is limited by
  the splitter imbalance and by the bias-dependent loss unbalancing the arms. Chirp follows the Koyama-Iga definition
  under `exp(+j omega t)`, positive for a frequency that rises with the intensity: `+1` for a lossless single drive at
  the default quadrature point, exactly 0 for a balanced push-pull drive whose absorption does not move with bias. A
  Bias sweep too short to take the transfer from a peak to a null leaves the three figures `None` and says why in
  `transfer_message`, rather than extrapolating. The pure functions are `gsim.modulator.twmzm.mzm_transfer`,
  `mzm_transfer_figures`, `mzm_chirp` and `mzm_drive_range`.

- The optical group index comes from the optical Stage. `study.optical.group_index()` solves the Mode at the reference
  Bias point `group_index_step_um` (10 nm) either side of the Stage's wavelength and forms
  `n_g = n_eff - lambda * d(n_eff)/d(lambda)`, every material resolved again at each wavelength, so material dispersion
  is in the answer as well as the guide's own; a core that resolves to one index at both wavelengths is warned about.
  The two extra solves are kept on the sweep (`OpticalSweep.group_index`, a `GroupIndex` record) and dropped with it.
  The line Stage takes the computed value whenever `n_group` is unset — it stood the phase index in, and warned, about
  2.5 against 3.9 on a silicon guide — so a Study reports its Velocity mismatch and Walk-off bandwidth with no
  hand-typed index; a configured `n_group` is used as given and costs no solve. On a silicon slab in oxide the femwell
  Route lands within 1e-5 of the closed-form 3.7067 with second-order elements, 0.13 of which is material dispersion.
  `LineStage.group_index()` no longer takes the sweep.

- Carrier mobility follows the doping. `gsim.common.carriers.MobilityModel` is the Masetti low-field fit for silicon
  (`masetti_silicon()`, or `constant(mu_n_cm2=, mu_p_cm2=)` for doping-independent values), and one model now serves
  both places a mobility enters: `ChargeTransportSim.mobility` sets DEVSIM's electron and hole mobilities node by node
  from the total doping (they were `simple_physics`' constants, 400 / 200 cm^2/Vs), and the carriers Stage evaluates the
  RF conductivity with the charge Stage's model (it used the lattice values, 1417 / 470.5, whatever the doping — some
  twenty times too conductive in a 1e20 contact Region). Configure it with `study.charge(mobility=...)`;
  `study.carriers(mobility=...)` overrides the RF conductivity alone. `CarriersStage.mu_n_cm2` / `mu_p_cm2` are gone,
  and `carrier_conductivity` takes `mobility=`.

- The report's Walk-off bandwidth is no longer read off the band's mean RF index, which vanishes — and sends the limit
  to hundreds of GHz — when a loaded line's index crosses the optical group index. `walkoff_bandwidth_dispersive` solves
  the walk-off condition with the Velocity mismatch the line has at that frequency, holding the last solved index past
  the solved range; a flat index recovers `walkoff_bandwidth`.

- `rib_phase_shifter` draws a depletion Phase shifter as foundries build one — a 500 x 220 nm rib on a 90 nm slab, a
  lightly doped core, 1e19 plus and 1e20 contact Regions under 10 x 1 um electrodes — and the TCAD TW-MZM notebook now
  runs on it. The RF Staircase follows the drawn device: no Strip straddles two Regions and each stands at its own
  Region's height, so a slab is not drawn as tall as the rib (`StripSegment`,
  `build_staircase_cross_section(segments=...)`, `EMStage.region_segments`, `Strips.zmin_um` / `zmax_um`).
  `RFStage.n_strips` now counts the Strips across the Junction extent and `strips_per_region` those across every other
  doped Region; the rib-resolution warning is gone, since the rib keeps its count however wide the slab.
  `pn_phase_shifter` takes the Traveling-wave `electrodes=`. The optical Staircase still draws every Strip at the rib's
  height, and warns on a device whose slab is thinner.

- The RF Stage lands on the line Mode at its defaults. It builds its Staircase with 21 Strips (the preset's
  `DEFAULT_N_STRIPS` and `RFStage.n_strips`, both 5 before; the preset's Palace optical Staircase takes the same count),
  and its `max_loss_ratio` defaults to one (0.5 before): the loaded line of a depletion Phase shifter loses most of a
  radian per radian at 10 GHz even at a depleted Bias, and the tighter bound dropped it for the wall Mode. The Stage
  warns when no Strip of the loaded Staircase is depleted — none a dielectric at the highest frequency — because the
  slab then shunts the electrodes (ADR 0005).

- The modulator's EM Stages reach their Backend through one Route interface (`gsim.modulator.route.Route`) with a
  femwell and a Palace implementation (`FemwellRoute`, `PalaceRoute`); a Stage no longer tests which Route it is on.
  Each Route reads the selected RF Mode in one call — index, characteristic impedance and whether it is the wall Mode
  (`gsim.common.modes.LineReading`) — given the electrodes as `gsim.common.modes.Conductor` descriptors, which is also
  how a conductor is now named to the femwell current integrals: `z0_power_current` and `electrode_current` take
  `conductor=` and `mesh=` in place of `sigma_s_per_m`, `current_elements` and `current_facets`, and
  `boundary_facets_on_rect` becomes `boundary_facets_within`. The settings both EM Stages took separately — `num_modes`,
  `min_index`, `boundary_field_tol`, `metallic_boundaries`, `order`, `n_guess` — are declared once on the shared EM
  Stage with the same per-Stage defaults.

- A Staircase (`gsim.common.stack.staircase`) is built from a Carrier map, the carriers Stage's coupling (`response=`),
  one typed strip input per Stage (`OpticalStripMaterial` or `RFStripMaterial`) and a `StaircaseDrawing` record; the
  plasma-dispersion, mobility, wavelength, index, permittivity and frequency keyword arguments are gone. Its Strips are
  a typed `Strips` record, its drawn layers are public, `stack()` takes no target, `unloaded()` switches the carriers
  off, and the one-dimensional Strip averaging `staircase_profile` lives in the Staircase module rather than in
  `gsim.common.carriers`. `RFLineParams` records the Bias and signal Contact it was solved at and resamples itself
  (`resampled`); the RF Stage's `solved_bias_v` is gone. `Stage.seed` hands a Stage a result without a solve;
  `Stage.reset` is removed. A Carrier map's `potential_v` and `net_doping_cm3` are optional.

- The charge-transport simulation declares semiconductor Interfaces apart from Contacts: `BoundaryModeSim.add_interface`
  tags them as their own line groups (`mesh_groups["interface_lines"]`), and `ChargeTransportSim.contact_specs` lists
  Contacts only, with `interface_specs` beside it.

- `BoundaryModeSim` owns its run and its postprocessing paths. `add_impedance_path(name, voltage=..., current=...)`
  declares the line integrals Palace evaluates on a solved Mode and returns the index the impedance is reported under;
  `run_local()` clears the previous run's tables, runs, and returns this run's `PalaceTextResults`; `read_results()`,
  `last_run_files` and `read_mode_field()` read that run back. The boundary-mode simulation takes no 3D ports any more:
  `add_port`, `add_cpw_port`, `add_wave_port` and `add_terminal` refuse, and the `ports`, `cpw_ports`, `wave_ports` and
  `terminals` lists are gone from it. `PortConfig` and `CPWPortConfig` go back to describing a 3D excitation port only —
  the `voltage_path`, `current_path`, `nsamples`, `center`, `orientation`, `width` and `order` fields, the path
  derivation from a port's geometry, and the matching `add_port` / `add_cpw_port` keyword arguments are removed. Mode
  paths are `(h, v)` cross-section coordinates; nothing is projected from layout coordinates.

- Removed: the pre-TCAD analytic Junction cluster and the Palace 2D TW-MZM notebook that was its only user (ADR 0006).
  `gsim.common.stack` no longer exports the doping-geometry builders (`make_doping_profile`, `make_pn_junction_profile`,
  `make_segmented_junction_profile`, `select_junction_mode`, `PNJunctionConfig.select_mode`), the one-dimensional Drude
  optics (`junction_epsilon_profile`, `carrier_profile_1d`, `epsilon_eff_relative`, `optical_params`,
  `refractive_index`, `drude_relaxation_times`, `default_eps_bg_rel`) or their constants (`MU_N_CM2_VS`, `MU_P_CM2_VS`,
  `M_CE_STAR`, `M_CH_STAR`, `SIGMA_NEGLIGIBLE_SM`), and `gsim.common.cross_section.build_optical_cross_section` is gone.
  The modulator workflow carries one plasma dispersion — the carriers Stage's, with substitutable coefficients — and one
  Staircase seam for every Stage. The Sze depletion model stays: `PNJunctionConfig`, `built_in_voltage`,
  `depletion_width`, `depletion_extents`, `junction_capacitance_per_area`, `sim.set_pn_junction()` and
  `gsim.tcad.compare_capacitance`.

- The modulator RF Stage's Palace Route solves an electrode-loaded Staircase and reports its characteristic impedance. A
  Staircase's electrodes now carry a `conductor_model` (ADR 0003): `"volume"` meshes each as a Region of lossy metal,
  `"pec"` leaves its interior out of the meshed domain and makes its outline a perfect conductor. Palace's
  shift-and-invert search returns a metal Region's own modes rather than the line's, so the Palace Route takes `"pec"`
  and the femwell Route, which carries the metal's loss, keeps `"volume"`; setting both to the same value is what makes
  the two Routes comparable. No existing femwell answer moves.

- `gsim.palace.mode_fields` reads a saved Palace boundary Mode's fields back off disk and runs the Marks-Williams
  power-current integrals on them, so `study.rf(route="palace")` reports `z0_ohm` instead of NaN.

- `gsim.femwell.adapter.z0_power_current` accepts `current_facets`: the signal current of a perfect conductor, which
  carries no volume current, as Ampere's contour integral around it. Validated against the analytic PEC-coax impedance.

- Both Routes put the same condition on the outer wall of an RF Window, so the cross-Route gate now covers the RF Stage
  as well as the optical one. `metallic_boundaries` puts a perfect conductor there; femwell has always honoured it and
  nothing in the Palace pipeline expressed it, leaving Palace to apply its own default of PMC — the opposite wall — so
  the two Routes solved different boundary-value problems on the identical mesh. `BoundaryModeSim.metallic_boundaries`
  now emits the domain's outer-boundary group under `Boundaries.PEC`, and the RF Stage threads its own setting onto the
  simulation it builds. On the shipped demo Cross-section the two Routes land on one line Mode's effective index to 1%
  and on its characteristic impedance to 5%, held by `tests/modulator/test_palace_route_runtime.py`.

- Fix: the native-2D mesher dropped every perfect-conductor and conductivity curve it found. It keeps a conductor's
  outline curve only when an adjacent surface carries a volume physical group, and read that adjacency out of the wrong
  half of gmsh's `getAdjacencies` result — the curve's end points rather than its adjacent surfaces — so the test never
  matched and the group came out empty. A `"pec"` Staircase's electrodes were therefore meshed as unconditioned slots
  rather than as conductors.

- Fix: `gsim.palace.mode_fields` reads a saved boundary Mode's two transverse components in the order Palace writes
  them. The reader swapped them, on the diagnosis that a mode's `E` came out tangential to a perfect conductor and its
  `H` normal to it; those fields came from meshes whose electrode outlines had lost their perfect-conductor groups to
  the bug above, so the swap was correcting a wrongly conditioned solve rather than a wrongly ordered file. Its
  power-current impedance also flips a Mode the solver chose to propagate against the plane normal, whose Poynting flux
  — and so whose reported `Z0` — came out negative.

- The Palace Route uses a crashed run's mode table when the table is complete. Palace 0.17 intermittently corrupts its
  heap while shutting down a `BoundaryMode` solve (`free(): corrupted unsorted chunks`), after the solve has finished
  and written its results; the Route clears the output directory before each run so what it reads back is that run's,
  warns, and re-raises when the table is short or missing.

- Fix: `make_pn_junction_profile` measured its default 0.22 um layer thickness from zero rather than from `zmin`, so a
  junction placed above the substrate got a thickness of `0.22 - zmin` instead.

- `gsim.common.modes.select_line_mode` takes a `max_loss_ratio` bound on `|Im(n_eff)| / Re(n_eff)`, and the RF Stage
  tightens it to 0.5: a transmission line advances several radians of phase per radian of loss, and the Modes sitting
  just inside the previous bound of one are the discretization's rather than the line's.

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
