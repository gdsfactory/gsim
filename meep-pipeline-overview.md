# gsim.meep Pipeline Overview

## Architecture: 5-Block Pipeline

```
Block 1              Block 2              Block 3
GDS + PDK  ──────►  3D Geometry  ──────►  Client-Side
Layer Parsing        Representation       Visualization
                     (no meep)            (no meep)
                          │
                          ▼
                     Block 4              Block 5
                     Sim Config  ──────►  Cloud Upload
                     (no meep)            + Runner (meep)
```

**Key constraint:** gsim is the client SDK and must NOT depend on meep. Meep only runs on the cloud.

---

## Implementation Status

| Block                | Status | Notes                                                                         |
| -------------------- | ------ | ----------------------------------------------------------------------------- |
| **1. Layer parsing** | Done   | DerivedLayer support via `LayeredComponentBase` in `common/`                  |
| **2. 3D geometry**   | Done   | GDS-file approach — `GeometryModel` + `Prism` dataclass, no meep deps         |
| **3. Visualization** | Done   | `common/viz/` — PyVista, Open3D+Plotly, Three.js, Matplotlib. Port arrows on all ports (source + monitor). |
| **4. Sim config**    | Done   | `SimConfig` JSON with layer_stack, ports, materials, fdtd, source, stopping, resolution, domain, accuracy, verbose_interval, diagnostics, symmetries, component_bbox |
| **5. Cloud runner**  | Done   | `run_meep.py` reads GDS via gdsfactory, Delaunay triangulation for holes, symmetry + decay stopping, explicit eigenmode kpoint, port_margin for monitors, polygon simplification, configurable subpixel averaging, verbose progress stepping, eigenmode debug logging (`meep_debug.json`), geometry/field diagnostics PNG output, preview-only mode, `component_bbox`-aware cell sizing |
| **6. Docker**        | Done   | `simulation-engines/meep/` — MPI-enabled pymeep, follows palace conventions   |

Tests: 118 meep-specific passing.

**Note:** The `gsim.fdtd` module (Tidy3D wrapper) was removed. MEEP is now the sole FDTD solver. The fdtd module had zero tests and zero usage outside itself. Its mode solver functionality (Waveguide, WaveguideCoupler, sweep functions) was Tidy3D-specific and not portable.

---

## Module Structure

### `gsim.common` — Shared Infrastructure

Solver-agnostic modules reusable by meep and palace.

| Module                           | Purpose                                                                                                                                     |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `common/layered_component.py`    | `LayeredComponentBase` — ported from gplugins. GDS component + layer stack → polygons, ports, bbox, z-coords.                               |
| `common/geometry_model.py`       | `GeometryModel` + `Prism` dataclass — generic 3D prism representation. `extract_geometry_model(lc)` builds from any `LayeredComponentBase`. |
| `common/polygon.py`              | `fuse_polygons()` — DerivedLayer-aware polygon extraction with merge + simplify.                                                            |
| `common/types.py`                | Shared type aliases.                                                                                                                        |
| `common/stack/extractor.py`      | `Layer`, `LayerStack` — extracts layer metadata from any PDK.                                                                               |
| `common/stack/materials.py`      | `MaterialProperties` with `refractive_index`, `extinction_coeff`, `optical()` classmethod.                                                  |
| `common/viz/__init__.py`         | Public API for all visualization backends.                                                                                                  |
| `common/viz/render2d.py`         | Matplotlib 2D cross-sections (XY, XZ, YZ slices). Supports `SimOverlay` for PML shading, sim cell boundary, source/monitor port markers.    |
| `common/viz/render3d_pyvista.py` | PyVista 3D rendering with Delaunay hole support.                                                                                            |
| `common/viz/render3d_open3d.py`  | Open3D + Plotly 3D rendering (Jupyter-friendly).                                                                                            |
| `common/viz/render3d_threejs.py` | Three.js + FastAPI live server visualization.                                                                                               |
| `common/viz/_colors.py`          | Layer color generation utilities.                                                                                                           |
| `common/viz/_mesh_helpers.py`    | Shared mesh building helpers (prism vertices, simulation box).                                                                              |

### `gsim.meep` — MEEP Photonic FDTD Simulation

| Module                   | Purpose                                                                                                                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `meep/__init__.py`       | Public API: `MeepSim`, `AccuracyConfig`, `DiagnosticsConfig`, `FDTDConfig`, `DomainConfig`, `ResolutionConfig`, `SimConfig`, `SourceConfig`, `SParameterResult`, `SymmetryEntry`                                                      |
| `meep/base.py`           | `MeepSimMixin` — fluent API (`set_geometry`, `set_stack`, `set_z_crop`, `set_wavelength`, `set_resolution`, `set_domain`, `set_material`). Builds `GeometryModel` + `SimOverlay` for viz.        |
| `meep/sim.py`            | `MeepSim` — main class. `set_source()`, `set_stopping()`, `set_symmetry()`, `set_accuracy()`, `set_diagnostics()`, `write_config()` exports `layout.gds` + `sim_config.json` + `run_meep.py`. Serializes layer stack, dielectrics, source, stopping, accuracy, diagnostics, verbose_interval, symmetries. `write_config()` auto-extends waveguide ports into PML via `gf.components.extend_ports()` and stores original bbox in `SimConfig.component_bbox`. `simulate()` calls gcloud. Deprecated: `set_source_port()` → `set_source(port=...)`, stopping kwargs on `set_wavelength()` → `set_stopping()`. |
| `meep/models/config.py`  | Pydantic models: `AccuracyConfig`, `DiagnosticsConfig`, `FDTDConfig`, `SourceConfig`, `StoppingConfig`, `SymmetryEntry`, `ResolutionConfig`, `DomainConfig` (with `extend_ports`), `SimConfig` (with `component_bbox`, `source`, `stopping`), `PortData`, `LayerStackEntry`, `MaterialData` |
| `meep/models/results.py` | `SParameterResult` — parses CSV output, complex S-params from mag+phase. Auto-loads `meep_debug.json` into `debug_info` field and diagnostic PNGs into `diagnostic_images`. `from_directory()` for preview-only (no CSV). `show_diagnostics()` for Jupyter display. |
| `meep/overlay.py`        | `SimOverlay` + `PortOverlay` + `DielectricOverlay` dataclasses, `build_sim_overlay()` — visualization metadata for sim cell, PML regions, port markers, dielectric backgrounds.                                    |
| `meep/ports.py`          | `extract_port_info()` — port center/direction/normal from gdsfactory ports. `_get_z_center()` uses highest refractive index layer.                                                                                |
| `meep/materials.py`      | `resolve_materials()` — resolves layer material names to (n, k) via common DB.                                                                                                                                    |
| `meep/script.py`         | `generate_meep_script()` — cloud runner template. Reads GDS + config, builds `mp.Block` slabs + `mp.Prism` layers, decay-based or fixed stopping, `split_chunks_evenly`, polygon simplification, configurable `eps_averaging`/`subpixel_maxeval`/`subpixel_tol`, verbose progress stepping via `mp.at_every()`, saves S-params CSV + eigenmode debug log (`meep_debug.json`) via `save_debug_log()`. Uses `component_bbox` from config for cell sizing (falls back to GDS bbox for backward compat). Server-side diagnostics: geometry cross-section PNGs (`save_geometry_diagnostics`), field snapshot (`save_field_snapshot`), raw epsilon (`save_epsilon_raw`). Preview-only mode (`preview_only` in config) skips FDTD run. **Symmetries are ignored during S-parameter extraction** (causes incorrect eigenmode normalization); only applied in preview-only mode. |

### `gsim.palace` — RF EM Simulation

Unchanged. Uses `common/stack/` for layer extraction and `common/geometry.py` for GDS geometry.

### `gsim.gcloud` — Cloud Execution

`run_simulation()` handles upload → start → wait → download. Supports `job_type="meep"` and `job_type="palace"`.

---

## Key Design Decisions

### GDS-file approach (not polygon-coords-in-JSON)

**Decision:** Send the raw GDS file to the cloud instead of extracting polygon coordinates client-side.

**Rationale:**

- Complex geometries have 1000s of nodes — JSON payloads too large
- DerivedLayers (e.g., `core = WG - DEEP_ETCH`) require gdsfactory to resolve correctly
- Exact GDS fidelity preserved
- Cloud already needs gdsfactory for other processing

**Upload package:**

```
output_dir/
  sim_config.json    # layer_stack, dielectrics, ports, materials, fdtd, source, stopping, resolution, domain, accuracy, diagnostics, verbose_interval, symmetries, component_bbox
  layout.gds         # raw GDS file
  run_meep.py        # self-contained cloud runner script
```

### LayeredComponentBase ported to gsim.common

**Decision:** Port `LayeredComponentBase` from gplugins into `gsim.common` instead of keeping the dependency.

**Rationale:**

- Removes gplugins as a required dependency
- Allows meep and palace to share the same polygon extraction logic
- The class is stable and self-contained (~300 lines)

### Visualization in gsim.common.viz

**Decision:** Solver-agnostic rendering via a generic `GeometryModel` interface.

**Rationale:**

- The rendering code only reads `.vertices` and `.height` — no solver types needed
- Generic `Prism` dataclass replaces solver-specific prism objects
- Split into separate backend files (~350 lines each) for maintainability

### Domain configuration (margins + PML + dielectric backgrounds)

**Decision:** Add a `DomainConfig` model with `dpml`, `margin_xy`, `margin_z_above`, `margin_z_below`, `port_margin`, `extend_ports` fields, exposed via `sim.set_domain()`.

**Rationale:**

- PML thickness was hardcoded to 1.0 um in the runner script — not configurable
- Margins control how much material (cladding) is kept around the waveguide core
- `margin_z_above`/`margin_z_below` are used by `set_z_crop()` to determine the crop window
- `margin_xy` is the gap between geometry bbox and PML inner edge
- Background dielectric slabs (`stack.dielectrics`) are serialized to JSON and created as `mp.Block` objects in the runner, ensuring the entire simulation domain is filled with the correct material at each z-position (no vacuum gaps)
- `mp.Block` slabs are placed first in the geometry list; MEEP's "later objects override" rule ensures patterned waveguide prisms take precedence
- `extend_ports` (default 0.0 = auto) extends waveguide ports into PML via `gf.components.extend_ports()` at `write_config()` time. Auto-calculates extension length as `margin_xy + dpml`. Original component bbox is stored in `SimConfig.component_bbox` so the runner computes cell size from the original (non-extended) geometry. Port centers are extracted from the original component and remain unchanged.

### Simulation overlay visualization

**Decision:** Add `SimOverlay` / `PortOverlay` / `DielectricOverlay` dataclasses and draw them on `plot_2d()` cross-sections.

**Rationale:**

- `plot_2d()` previously only showed geometry prisms and a dashed bbox matching geometry bounds
- Users need to see the actual simulation cell (geometry + margin + PML), PML regions, and port locations
- Overlay rendering is solver-agnostic — `render2d.py` draws from generic overlay metadata
- Source ports shown in red with arrow, monitor ports in blue with arrow, PML as semi-transparent orange
- Dielectric background slabs shown as coloured bands in XZ/YZ views (SiO2 light blue, silicon grey, air transparent)

### Waveguide port extension into PML

**Decision:** Automatically extend waveguide ports through the margin+PML region at `write_config()` time using `gf.components.extend_ports()`.

**Rationale:**

- MEEP's PML requires waveguides to extend straight into the absorber for proper mode absorption
- Without extension, waveguides terminate abruptly at the component boundary inside the margin/PML, creating spurious reflections that corrupt S-parameters
- This is the same approach used by gplugins: extend GDS geometry while keeping sources/monitors at original port positions and computing cell size from the original bbox

**Implementation:**

- `DomainConfig.extend_ports` (default 0.0 = auto) controls the extension length
- Auto mode computes `margin_xy + dpml` (the minimum to reach through PML)
- `write_config()` calls `gf.components.extend_ports(original_component, length=extend_length)`
- Original component bbox is stored in `SimConfig.component_bbox` so the runner computes cell size from the original geometry, not the extended one
- Port centers are extracted from the original component (unchanged positions)
- The runner falls back to `component.dbbox()` when `component_bbox` is absent (backward compat with old configs)

```
|--- PML ---|--- margin_xy ---|--- original component ---|--- margin_xy ---|--- PML ---|
|=== extended waveguide extends through margin+PML ===========================...========|
                               ^                         ^
                        source/monitor             source/monitor
                        (at original port)         (at original port)
```

### Port z-center uses highest refractive index

**Decision:** For photonic simulations, port z-center is placed at the midpoint of the layer with the highest refractive index (waveguide core), not the conductor layer.

**Rationale:** RF (Palace) ports span between metal layers, but photonic ports need to be centered on the waveguide core for proper mode overlap.

### Z-domain cropping via `set_z_crop()`

**Decision:** Add `set_z_crop()` to crop the layer stack along z around the photonic core, removing non-essential layers (metals, heaters, vias) and clipping oversized cladding/substrate.

**Rationale:**

- Full PDK stacks include metal layers up to z=5.2um and substrate down to z=-8um, but the waveguide core is only 0.22um thick
- The MEEP simulation cell z-extent is derived from `min/max` of all layer z-values — unnecessary layers inflate the domain
- For a 220nm SOI waveguide, MEEP tutorials use ~4-6um total z-extent (2um cladding above/below core)
- Full stack: 15.2um cell → cropped: 6.2um cell (**60% reduction** in z, massive compute savings in 3D)

**Implementation:**

- Auto-detects core layer via highest refractive index (same logic as port z-center)
- Crops tightly to `[ref.zmin, ref.zmax]` — actual gap before PML is controlled by `set_padding()`
- Removes layers entirely outside the crop window
- Clips layers that partially overlap (adjusts zmin/zmax/thickness)
- Called after `set_stack()`, before `write_config()`
- No runner changes needed — the runner derives z-extent from the serialized layer_stack

### Run time: `until_after_sources` (not `run_time_factor`)

**Decision:** Replace `run_time_factor` (gsim invention: `run_time = factor / df`) with `run_after_sources` using MEEP's standard `sim.run(until_after_sources=N)`.

**Rationale:**

- MEEP tutorials universally use `until_after_sources=100` for S-parameter extraction
- The old `run_time_factor / df` formula was fragile (division by zero if bandwidth=0) and non-standard
- `until_after_sources` lets the Gaussian source decay naturally, then runs N more time units for fields to ring down
- Default: 100 time units (matches MEEP GDS import tutorial)

### Performance optimizations (symmetry, decay stopping, reduced defaults)

**Decision:** Add five performance features to reduce simulation time without sacrificing accuracy.

**1. ~~Symmetry exploitation~~ (DISABLED for S-parameter runs)**

- `SymmetryEntry` model with `direction` (X/Y/Z) and `phase` (+1/-1)
- `sim.set_symmetry(y=-1)` is accepted but **ignored** during S-parameter extraction
- MEEP's `get_eigenmode_coefficients` with `add_mode_monitor` (which uses `use_symmetry=false` internally) produces incorrect normalization when the source monitor straddles a symmetry plane — the incident coefficient is underestimated by ~2x, inflating all S-parameters
- This matches gplugins, which also never uses `mp.Mirror` for S-parameter extraction
- Symmetries are only applied in **preview-only** mode (geometry validation, no FDTD)
- The `set_symmetry()` API and `SymmetryEntry` model are retained for future non-S-parameter use cases

**2. Stopping modes (1.5-3x speedup)**

- `StoppingConfig` is now a top-level field on `MeepSim` (via `sim.set_stopping()`), serialized to `config["stopping"]` in JSON. Decoupled from `FDTDConfig` which now only holds wavelength/frequency settings.
- `mode="fixed"|"decay"|"dft_decay"`, `max_time` (→ `run_after_sources`), `threshold` (→ `decay_by`)
- Fixed mode: `sim.run(until_after_sources=N)` (original behavior)
- Decay mode: `[mp.stop_when_fields_decayed(...), run_after]` list — MEEP's native OR logic stops on whichever fires first (decay or time cap)
- DFT-decay mode: `mp.stop_when_dft_decayed(tol, minimum_run_time, maximum_run_time)` — monitors convergence of all DFT monitors (best for S-parameter extraction), has built-in min/max time bounds
- **`dft_min_run_time` default: 100** time units — prevents false convergence on near-zero fields before the pulse has traversed the device. Must exceed `device_length * n_group` (e.g. 15um Y-branch with n_g≈4 needs ≥60 time units). Previous default of 0 caused premature stopping.
- `resolve_decay_monitor_point()` auto-selects the first non-source port center (decay mode only)
- `_COMPONENT_MAP` maps string names ("Ez", "Ey", ...) to MEEP field components
- Runner reads stopping from `config["stopping"]` with fallback to `config["fdtd"]["stopping"]` for old JSON configs

**2b. Source bandwidth decoupled from monitor span**

- `SourceConfig` controls the Gaussian source `fwidth` and source port, serialized to `config["source"]` in JSON
- `sim.set_source(bandwidth=None, port=None)` — `bandwidth` in wavelength um, `None` = auto
- Auto fwidth: `max(3 * monitor_df, 0.2 * fcen)` — ensures edge frequencies receive adequate spectral power, matching gplugins' `dfcen=0.2` convention
- Previously source `fwidth = monitor df`, which starved edge frequencies of spectral power when the monitor bandwidth was narrow, causing |S|>1 blowup at band edges
- Runner reads `fwidth` from `config["source"]["fwidth"]` with fallback to `config["fdtd"]["df"]` for old configs

**3. Reduced default margins (1.3-2x speedup)**

- `DomainConfig` margins changed from 1.0 to **0.5** um (xy, z_above, z_below)
- PML (`dpml`) stays at 1.0 um — important for 1.55um wavelength
- 0.5um margin is sufficient for evanescent field decay in typical photonic waveguides
- Users can still override via `sim.set_domain(margin_xy=1.0)` for leaky structures

**4. Reduced default `num_freqs` (~2x speedup)**

- Default changed from 21 to **11** frequency points
- 11 points across a 100nm bandwidth gives ~10nm resolution — sufficient for most S-parameter characterization
- Users can still request more points via `sim.set_wavelength(num_freqs=51)`

**5. `split_chunks_evenly=False` (MPI load balancing)**

- Default `False` in `SimConfig`, passed through to `mp.Simulation(split_chunks_evenly=False)`
- MEEP's default (`True`) splits the domain into equal-sized chunks regardless of geometry
- `False` lets MEEP assign more processors to chunks with more geometry, improving MPI efficiency

**6. Polygon simplification (10-100x fewer vertices)**

- `AccuracyConfig.simplify_tol` (default 0.0 = off) applies Shapely's `simplify(tol, preserve_topology=True)` to merged polygons before extrusion
- The ebeam_y_1550 Y-branch exports a prism with **1292 vertices** due to dense curve sampling in the GDS — MEEP's subpixel averaging is O(n) to O(n²) in vertex count
- `simplify_tol=0.01` (10nm) reduces this to ~50-100 vertices with negligible geometry error
- Vertex count is logged: `Total vertices across all prisms: N`

**7. Configurable subpixel averaging**

- `AccuracyConfig.eps_averaging` (default `True`) toggles MEEP's subpixel smoothing — set to `False` for fast initial runs
- `AccuracyConfig.subpixel_maxeval` (default 0 = unlimited) caps the number of integration evaluations per pixel — reduces accuracy but speeds up initialization
- `AccuracyConfig.subpixel_tol` (default 1e-4) controls convergence tolerance for the subpixel integration
- All three are passed directly to `mp.Simulation()` constructor

**8. Verbose progress stepping (observability)**

- `verbose_interval` (default 0 = off) in MEEP time units between progress prints during FDTD time-stepping
- When > 0, builds `mp.at_every(interval, func)` that prints `t=... | wall=...s` with wall-clock time
- Passed as a step function to `sim.run()` — empty list when off, preserving existing behavior
- Solves the "3-hour simulation with no output" problem

**9. Eigenmode debug logging (`meep_debug.json`)**

- `extract_s_params()` now collects per-port eigenmode diagnostics alongside S-parameters
- `save_debug_log()` writes `meep_debug.json` with: metadata (resolution, cell size, wall time, timesteps, stopping mode), per-port eigenmode info (n_eff, kdom, group velocity), raw forward/backward coefficients, incident coefficient magnitudes, and power conservation per frequency
- `SParameterResult.from_csv()` auto-loads `meep_debug.json` into the `debug_info` field when present
- Primary use: verify n_eff values match expected guided mode (e.g. ~2.44 for Si at 1550nm) and power conservation ~1.0
- Catches port_margin or geometry issues immediately without re-running the simulation

**10. Server-side diagnostics and preview mode**

- `DiagnosticsConfig` model controls what the runner outputs beyond S-parameters: `save_geometry` (pre-run epsilon cross-sections), `save_fields` (post-run field snapshot), `save_epsilon_raw` (numpy array), `preview_only` (skip FDTD entirely)
- `sim.set_diagnostics(save_geometry=True, save_fields=True, preview_only=False)` — fluent API
- Runner calls `sim.init_sim()` before diagnostics (needed for epsilon grid), then `sim.plot2D()` with matplotlib for cross-section PNGs
- For 3D sims: XY (at z=core), XZ (at y=center), YZ (at x=center) geometry cross-sections
- `preview_only=True` initializes the sim and saves geometry diagnostics, then exits without running FDTD — fast geometry validation (seconds instead of minutes)
- `HAS_MATPLOTLIB` flag provides graceful degradation on Docker images without matplotlib (though gdsfactory already depends on it)
- MPI-safe: all ranks call `plot2D()` (collective via `get_array()`), only rank 0 saves files
- `SParameterResult.from_directory()` loads preview-only results (no CSV, just debug JSON + diagnostic PNGs)
- `result.show_diagnostics()` displays diagnostic PNGs inline in Jupyter

**11. Waveguide port extension into PML (accuracy)**

- `DomainConfig.extend_ports` (default 0.0 = auto) extends waveguide ports through `margin_xy + dpml` into PML at `write_config()` time
- Uses `gf.components.extend_ports()` on the GDS component before writing
- Original bbox stored in `SimConfig.component_bbox` for correct cell sizing in the runner
- Prevents spurious reflections from abrupt waveguide termination inside PML
- On by default — no user action needed for standard photonic simulations

**Implementation constraint:** All configuration happens client-side via Pydantic models; the generated `run_meep.py` reads the JSON config at runtime. gsim has no meep dependency.

### Removed gsim.fdtd (Tidy3D wrapper)

**Decision:** Remove the `gsim.fdtd` module entirely. MEEP is the sole FDTD solver.

**Rationale:**

- fdtd had zero tests and zero usage outside itself
- fdtd was tightly coupled to Tidy3D (commercial solver) — every file imported `tidy3d`
- All shared code (LayeredComponentBase, viz, geometry model) was already extracted to `common/`
- Mode solver functionality (Waveguide, WaveguideCoupler) was Tidy3D-specific and not portable to MEEP

---

## Approaches Considered and Ruled Out

| Approach                               | Why ruled out                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------ |
| GMSH as intermediate representation    | Round-trip overhead, no Jupyter support, `mp.Prism` handles sidewall angles natively |
| Polygon coordinates in JSON            | Payloads too large for complex geometries (1000s of nodes)                           |
| MEEP's native `get_GDSII_prisms()`     | No `(layer, datatype)` tuple support, no holes, no sidewall angles                   |
| MEEP's `epsilon_input_file` (HDF5)     | Only frequency-independent real permittivities, loses subpixel smoothing             |
| MEEP's `epsilon_func` / `MaterialGrid` | Very slow startup, designed for topology optimization                                |
| Keeping gsim.fdtd alongside gsim.meep  | Redundant FDTD modules, fdtd was unmaintained with no tests                          |
| `run_time_factor / df` for run time    | Non-standard, fragile (div-by-zero), MEEP uses `until_after_sources` everywhere      |
| `mp.Mirror` symmetry for S-param runs | `add_mode_monitor` uses `use_symmetry=false` internally; `get_eigenmode_coefficients` doesn't apply `S.multiplicity()` correction, so source monitors on the symmetry plane get ~2x underestimated coefficients. gplugins never uses `mp.Mirror` either. See "Symmetry and eigenmode coefficient normalization" below. |

---

## Remaining Work / Known Issues

- **Typed `SimConfig` + JSON schema:** `SimConfig` currently uses `dict[str, Any]` for sub-sections (fdtd, source, stopping, domain, etc.) because `write_config()` calls `.to_dict()` on each typed model and passes plain dicts. Refactor to use typed Pydantic fields directly (e.g. `fdtd: FDTDConfig` instead of `fdtd: dict`), making computed properties (`fcen`, `df`) into real fields. This would let `SimConfig.model_json_schema()` produce a fully typed JSON schema as the single source of truth, and allow storing `sim_config_schema.json` in the repo for runner-side validation.
- **Cloud testing:** The full cloud round-trip (upload → run meep → download results) hasn't been tested end-to-end with a real cloud instance.
- **`_build_geometry_model()` note:** Uses `gf.get_active_pdk().layer_stack` (gdsfactory's LayerStack) instead of `self.stack` (gsim's LayerStack) because `LayeredComponentBase` needs gdsfactory's type for DerivedLayer resolution.
- **Symmetry for S-parameter speedup:** Currently disabled (see bug fix below). Two possible future approaches:
  1. **`add_flux` + `eig_parity` path** — MEEP's `add_flux` uses `use_symmetry=true`, so `get_eigenmode_coefficients` applies `S.multiplicity()` correctly. Requires passing the correct `eig_parity` (e.g. `mp.ODD_Z+mp.EVEN_Y` for TE) which depends on polarization and symmetry phase. The MEEP mode decomposition tutorials use this approach for gratings with symmetry. Would need auto-detection of the correct parity from the symmetry config and waveguide polarization.
  2. **gplugins-style port symmetries** — Instead of MEEP's `mp.Mirror` (which halves the compute domain), exploit device symmetry at a higher level: run fewer source ports and copy S-parameters between symmetric port pairs (e.g. S31=S21 for a Y-branch). This doesn't speed up individual simulations but reduces the number of runs needed for multi-port devices. gplugins implements this via `port_symmetries` dicts.

## Bug Fixes

### S-parameter direction index fix (extract_s_params)

**Problem:** `extract_s_params()` used `alpha[0, :, 0]` (forward/+normal coefficient) for ALL ports, then normalized by dividing everything by S11. Since S11 was the forward (incident) coefficient at the source port, `S11 / S11 = 1.0` always. The reflected wave was never extracted.

**Symptom:** S11 = 1.000000 (0 dB) at all wavelengths — 100% apparent reflection. S21/S31 ≈ 0.183 (-14.7 dB) instead of expected -4 dB. Power conservation violated (|S11|² + |S21|² + |S31|² = 1.067).

**Root cause:** MEEP's `get_eigenmode_coefficients` returns `alpha[band, freq, direction]` where direction index 0 = forward (+normal) and 1 = backward (-normal). The old code always used index 0 for all ports, capturing the incident wave at the source port instead of the reflected wave.

**Fix:** Use the port `direction` field from the config to select the correct alpha index:

- Port `direction` = the direction of the **incoming** mode along normal_axis (from `ports.py:get_port_normal()`)
- At each port, the **outgoing** (reflected or transmitted) wave goes in the **opposite** direction
- `direction="+"` → incoming index 0, outgoing index 1
- `direction="-"` → incoming index 1, outgoing index 0
- Normalization uses the incident coefficient at the source port (incoming direction), computed separately from S11

For the ebeam_y_1550 Y-branch:
- Source port o1 (x=-7.4, direction="+"): S11 = `alpha[0,:,1]` (backward/reflected) / `alpha[0,:,0]` (forward/incident)
- Output port o2 (x=+7.4, direction="-"): S21 = `alpha[0,:,0]` (forward/transmitted) / incident
- Output port o3 (x=+7.4, direction="-"): S31 = `alpha[0,:,0]` (forward/transmitted) / incident

**Files changed:** `gsim/meep/script.py` (template), `simulation-engines/meep/src/run_meep.py` (deployed copy). 97 tests pass.

### Port monitor/source width fix (CRITICAL — 70% power loss)

**Problem:** Port width from gdsfactory (`gf_port.width` = physical waveguide width, e.g. 0.5 um) was used directly as the transverse size of both `EigenModeSource` and mode monitors. The fundamental waveguide mode's evanescent tails extend ~0.5-1.0 um beyond the waveguide edges into the cladding. A 0.5 um monitor truncates these tails, so `get_eigenmode_coefficients` captures only ~30% of the mode power.

**Symptom:** S11 ≈ 0.49 (-6 dB, should be -20 dB), S21 = S31 ≈ 0.183 (-14.7 dB, should be -4 dB). Power conservation: |S11|² + |S21|² + |S31|² ≈ 0.307 (only 31% accounted for, ~70% lost).

**Root cause:** The mode overlap integral in `get_eigenmode_coefficients` integrates over the monitor cross-section. With a 0.5 um monitor matching the waveguide width, the significant evanescent field outside the waveguide is excluded from the overlap, artificially reducing all mode coefficients.

**Fix:** Added `port_margin` field to `DomainConfig` (default: 0.5 um). The runner script now uses `width + 2 * port_margin` for source/monitor transverse size (e.g. 0.5 + 1.0 = 1.5 um). This matches gplugins' default `port_margin=0.5`.

**Files changed:**
- `meep/models/config.py` — `DomainConfig.port_margin: float = 0.5`
- `meep/sim.py` — `set_domain(..., port_margin=0.5)`
- `meep/script.py` — `build_sources()` and `build_monitors()` use `width + 2 * port_margin`
- `meep/overlay.py` — `build_sim_overlay()` applies `port_margin` to overlay port width for accurate visualization

### Eigenmode kpoint and direction fix

**Problem:** `EigenModeSource` was created without explicit `direction`, `eig_kpoint`, or `eig_match_freq`. The `get_eigenmode_coefficients` calls didn't pass `kpoint_func`. Without these, MEEP's auto-detection of "forward" vs "backward" may not match our assumed alpha index convention, and the eigenmode solver has no directional preference for the initial k-vector guess.

**Fix (following gplugins pattern):**

1. **`EigenModeSource`** now sets:
   - `direction=mp.X` or `mp.Y` based on `normal_axis`
   - `eig_kpoint` pointing INTO the device (e.g. `Vector3(x=1)` for a west-facing source)
   - `eig_match_freq=True` for correct mode at each frequency

2. **`get_eigenmode_coefficients`** now passes `kpoint_func=lambda f, n, kp=kp: kp` with a positive-axis kpoint (`Vector3(x=1)` for x-normal ports). This anchors the convention: `alpha[0]=forward (+axis)`, `alpha[1]=backward (-axis)`, which is exactly what `_incoming_idx` / `_outgoing_idx` assumes.

3. Added `_port_kpoint(port)` helper function that returns the positive-axis kpoint vector for a given port.

**Files changed:** `meep/script.py` — `build_sources()`, `extract_s_params()`, new `_port_kpoint()` helper.

### Port margin default reduced from 2.0 to 0.5 (cladding mode fix)

**Problem:** With `port_margin=2.0`, the mode monitor cross-section for a 0.5 um waveguide was 0.5 + 2*2.0 = 4.5 um. MPB's band 1 eigenmode in this oversized cell converged to a cladding mode (n_eff ~ 1.43) rather than the guided TE0 mode (n_eff ~ 2.44). All S-parameters were computed against the wrong mode.

**Symptom:** S11 ~ 0.33, S21 ~ 0.35 for the ebeam_y_1550 Y-branch (expected: S11 ~ 0, S21 ~ 0.7). Power conservation ~ 0.35 (severe violation).

**Root cause:** The 4.5 um monitor cross-section is ~9x the waveguide width. In a cell this large, the lowest-order eigenmode (band 1) is a plane wave in the cladding material rather than the guided waveguide mode. The guided mode becomes band 2 or higher, but `eig_band=1` always requests band 1.

**Fix:** Reduced `port_margin` default from 2.0 to 0.5 (matching gplugins' default). Monitor cross-section: 0.5 + 2*0.5 = 1.5 um. In this tighter cell, band 1 is correctly the guided TE0 mode (n_eff ~ 2.44).

**Diagnostic:** Added `meep_debug.json` output with per-port `n_eff` values so this issue can be caught immediately on future runs — n_eff should be ~2.44 for silicon waveguides at 1550nm, not ~1.44 (cladding index).

**Files changed:** `meep/models/config.py`, `meep/sim.py`, `meep/script.py`, `tests/meep/test_meep_sim.py`.

### Decay component default changed to "Ey"

**Problem:** Default `decay_component` was `"Ez"`, but for TE-polarized modes in SOI (the dominant mode in 220nm silicon strip waveguides), the dominant electric field component is **Ey** (transverse to propagation, in-plane with the slab). Ez is the weakest TE component and can decay 10-100x faster than Ey, causing premature simulation termination.

**Fix:** Changed default from `"Ez"` to `"Ey"` in `StoppingConfig`, `set_wavelength()`, and the runner script fallback.

**Files changed:** `meep/models/config.py`, `meep/sim.py`, `meep/script.py`.

### Monitor port direction arrows in visualization

**Problem:** Only source ports showed direction arrows in the XY cross-section view. Monitor ports were drawn as blue lines without any directional indicator, making it hard to verify port orientation.

**Fix:** Removed the `if port.is_source:` guards in `_draw_overlay_xy()`. Now ALL ports (source and monitor) show direction arrows. Source arrows remain red, monitor arrows are blue — the color is already set per-port based on `port.is_source`.

**Files changed:** `common/viz/render2d.py`.

### Symmetry and eigenmode coefficient normalization (CRITICAL — ~2x S-param inflation)

**Problem:** When `mp.Mirror(mp.Y, phase=-1)` was used for a Y-branch simulation, S21/S31 magnitudes were ~1.35 instead of expected ~0.7. Power conservation was ~3.6 instead of ~1.0. The source port (o1, at y=0 on the symmetry plane) had its eigenmode coefficient underestimated by ~2x, inflating all S-parameters that normalize against it. S11 was unaffected since both numerator and denominator are at the same port.

**Symptom:** S11 ≈ 0.10 (correct), S21 = S31 ≈ 1.35 (should be ~0.7), power conservation ≈ 3.6 (should be ~1.0). Edge frequencies blow up to S21 > 2.0.

**Root cause (from MEEP source code and documentation):**

1. `add_mode_monitor` internally calls `add_dft_flux` with `use_symmetry=false` (in MEEP's C++ `src/dft.cpp`). This means the DFT monitor stores fields over the full unsymmetrized domain.

2. In `get_eigenmode_coefficients` (MEEP's C++ `src/mpb.cpp`), the eigenmode coefficient scaling factor `csc` uses `S.multiplicity()` only when `flux.use_symmetry` is `true`:
   ```cpp
   double csc = sqrt((flux.use_symmetry ? S.multiplicity() : 1.0) / abs(normfac));
   ```
   Since `add_mode_monitor` sets `use_symmetry=false`, the multiplicity correction is **never applied**, regardless of whether `mp.Mirror` symmetries are active in the simulation.

3. When the simulation domain is halved by `mp.Mirror(mp.Y)`, the source port monitor at y=0 straddles the symmetry plane. The overlap integral between the simulated fields (reconstructed from the half-domain) and the eigenmode profile is computed correctly for the full domain, but the power normalization doesn't account for the symmetry-reduced source power. The result is that the incident coefficient at the source port is ~2x too small.

4. Output port monitors (o2 at y=+2.75, o3 at y=-2.75) are entirely within one half of the domain and are not affected. Their coefficients are correct.

5. Since S11 = reflected_at_o1 / incident_at_o1, and both terms are equally affected (same monitor), S11 is correct. But S21 = transmitted_at_o2 / incident_at_o1, and only the denominator is underestimated, so S21 is inflated by ~2x.

**Verification:** Correcting by factor of 2: S21_true = 1.346/2 = 0.673 ≈ 1/√2 (expected for 3dB splitter). True power conservation = 3.63/4 = 0.91 (reasonable with some insertion loss).

**gplugins approach:** gplugins **never uses `mp.Mirror` symmetries** for S-parameter extraction. It avoids the issue entirely. Instead, gplugins implements "port symmetries" at a higher level — copying S-parameter values between symmetric port pairs (e.g. S31=S21 for a symmetric Y-branch) to reduce the number of source ports that need separate simulation runs.

**MEEP documentation notes:**
- `add_mode_monitor` "works properly with arbitrary symmetries, but may be suboptimal because the Fourier-transformed region does not exploit the symmetry" ([Python User Interface](https://meep.readthedocs.io/en/latest/Python_User_Interface/))
- The alternative `add_flux` path uses `use_symmetry=true` and requires explicit `eig_parity` — the MEEP mode decomposition tutorials use this for gratings with symmetry, but no tutorial combines it with multi-port S-parameter extraction
- [Issue #957](https://github.com/NanoComp/meep/issues/957) (open) tracks better symmetry support in `get_eigenmode_coefficients`
- The [GDSII Import tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/) (directional coupler S-params) does not use symmetry

**Fix:** The runner now ignores symmetries during S-parameter extraction (only applies them in preview-only mode). `set_symmetry()` logs a warning. Symmetry entries are still serialized to JSON for future use.

**Files changed:** `meep/script.py` (runner template), `meep/sim.py` (`set_symmetry` warning), `nbs/generate_meep_config.py` (removed `set_symmetry` call), `nbs/test-meep.ipynb` (removed symmetry, updated docs). 106 tests pass.

---

## Docker Integration

### Directory Structure

In `simulation-engines/meep/` (separate repo):

```
meep/
├── Dockerfile           # Production (linux/amd64, non-root user, MPI)
├── Dockerfile.local     # Local testing (no platform pin, no non-root user)
├── entrypoint.sh        # Handles run_meep.py discovery + output collection
├── utils.sh             # S3 presigned URL download/upload helpers
├── src/                 # Bundled gsim test files (fallback when no presigned URL)
│   ├── layout.gds
│   ├── sim_config.json
│   └── run_meep.py
└── legacy/              # Old files (preserved for reference)
    ├── Dockerfile
    ├── Dockerfile.local
    ├── entrypoint.sh
    └── src/main.py
```

### Key Docker Details

**Base image:** `continuumio/miniconda3:24.11.1-0`

**Conda env `mp` (python 3.12):**
- `gdsfactory`, `numpy<2`, `scipy`, `shapely` (pip)
- `pymeep=*=mpi_mpich_*`, `nlopt` (conda-forge) — MPI-enabled MEEP

**Entrypoint conventions** (matches palace):
- `MEEP_NP` env var controls MPI processes (default 8)
- `OMPI_MCA` oversubscribe flags for containerized environments
- Downloads input from `INPUT_DOWNLOAD_PRESIGNED_URL` (clears `/app/src/*` first)
- `cd /app/src` for relative path resolution
- Finds `run_meep.py` first, falls back to `main.py`
- Runs via `mpirun -np $NP python run_meep.py`
- Collects outputs (csv/h5/pkl/png) to `/app/data/`
- Uploads via `OUTPUT_UPLOAD_PRESIGNED_URL`

**Runner script requires PDK activation:** `gf.gpdk.PDK.activate()` is called in `load_gds_component()` because `component.get_polygons()` internally calls `get_active_pdk()`. The generic PDK is sufficient since all layer info comes from `sim_config.json`.

### Contract Summary

```
gsim.meep.MeepSim.write_config()
  └─ output_dir/
      ├── layout.gds          # GDS geometry
      ├── sim_config.json     # All simulation parameters
      └── run_meep.py         # Self-contained cloud runner

    ↓ (zipped and uploaded)

Docker container /app/src/
  ├── layout.gds
  ├── sim_config.json
  └── run_meep.py
  entrypoint: cd /app/src && mpirun -np $NP python run_meep.py

    ↓ (runner writes to CWD)

/app/src/s_parameters.csv      # S-parameter results
/app/src/meep_debug.json       # Eigenmode diagnostics (n_eff, kdom, power conservation)
/app/src/meep_geometry_xy.png  # Pre-run geometry cross-section (XY at z=core)
/app/src/meep_geometry_xz.png  # Pre-run geometry cross-section (XZ, 3D only)
/app/src/meep_geometry_yz.png  # Pre-run geometry cross-section (YZ, 3D only)
/app/src/meep_fields_xy.png    # Post-run field snapshot (Ey on epsilon)
  entrypoint: cp *.csv *.json *.png /app/data/

    ↓ (tarballed and uploaded)

gsim receives results → SParameterResult.from_csv() or .from_directory()
  (debug_info auto-loaded from meep_debug.json, diagnostic_images from PNGs)
```

### Local Testing

**Automated (recommended):** Use `nbs/test_meep_local.sh` which runs the full loop:

```bash
# From gsim project root:
./nbs/test_meep_local.sh          # default 4 MPI procs
MEEP_NP=8 ./nbs/test_meep_local.sh  # override proc count
```

The script: regenerates config → copies to Docker context → rebuilds image → runs container → prints CSV results with dB summary and power conservation check.

**Manual:**

```bash
cd simulation-engines/meep
docker build -f Dockerfile.local -t meep-local .
docker run --rm -e MEEP_NP=4 -v /tmp/meep-out:/app/data meep-local
cat /tmp/meep-out/s_parameters.csv
```

**Config regeneration only** (no Docker):

```bash
uv run python nbs/generate_meep_config.py
# Output: nbs/meep-sim-test/{layout.gds, sim_config.json, run_meep.py}
```

---

## Typical Workflow

```python
import gdsfactory as gf
from gdsfactory.gpdk import get_generic_pdk

pdk = get_generic_pdk()
pdk.activate()

component = gf.components.mmi1x2()

from gsim.meep import MeepSim

sim = MeepSim()
sim.set_geometry(component)
sim.set_stack()
sim.set_domain(0.5)  # 0.5um margins (default), 1um PML, auto-extends ports into PML
sim.set_z_crop()  # crop to margin_z_above/below around core
sim.set_material("si", refractive_index=3.47)
sim.set_wavelength(wavelength=1.55, bandwidth=0.1)
sim.set_source()                         # auto fwidth (~3x monitor bw), auto port
sim.set_stopping(mode="dft_decay", max_time=200, threshold=1e-3)  # best for S-params
sim.set_resolution(pixels_per_um=40)
sim.set_accuracy(
    simplify_tol=0.01,       # simplify dense GDS polygons (10nm tolerance)
    eps_averaging=True,       # subpixel averaging (disable for quick tests)
    verbose_interval=5.0,     # progress prints every 5 MEEP time units
)
sim.set_diagnostics(save_geometry=True, save_fields=True)  # server-side PNGs
sim.set_output_dir("./meep-sim-test")

# Visualize before running (client-side, no MEEP)
sim.plot_2d(slices="xyz")

# Write config files
sim.write_config()

# Run on cloud
result = sim.simulate()
result.plot()

# View server-side diagnostics (geometry + field PNGs from MEEP)
result.show_diagnostics()

# Preview-only: fast geometry validation (seconds, no FDTD)
sim.set_diagnostics(preview_only=True)
sim.write_config()
result = sim.simulate()
result.show_diagnostics()  # geometry PNGs only, no fields or S-params
```
