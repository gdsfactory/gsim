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
| **4. Sim config**    | Done   | `SimConfig` JSON with layer_stack, ports, materials, fdtd, resolution, domain, accuracy, verbose_interval, symmetries |
| **5. Cloud runner**  | Done   | `run_meep.py` reads GDS via gdsfactory, Delaunay triangulation for holes, symmetry + decay stopping, explicit eigenmode kpoint, port_margin for monitors, polygon simplification, configurable subpixel averaging, verbose progress stepping |
| **6. Docker**        | Done   | `simulation-engines/meep/` — MPI-enabled pymeep, follows palace conventions   |

Tests: 97 meep-specific, 155 total passing.

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
| `meep/__init__.py`       | Public API: `MeepSim`, `AccuracyConfig`, `FDTDConfig`, `DomainConfig`, `ResolutionConfig`, `SimConfig`, `SParameterResult`, `SymmetryEntry`                                                                         |
| `meep/base.py`           | `MeepSimMixin` — fluent API (`set_geometry`, `set_stack`, `set_z_crop`, `set_wavelength`, `set_resolution`, `set_domain`, `set_material`, `set_source_port`). Builds `GeometryModel` + `SimOverlay` for viz.        |
| `meep/sim.py`            | `MeepSim` — main class. `set_symmetry()`, `set_accuracy()`, `write_config()` exports `layout.gds` + `sim_config.json` + `run_meep.py`. Serializes layer stack, dielectrics, accuracy, verbose_interval, symmetries. `simulate()` calls gcloud. |
| `meep/models/config.py`  | Pydantic models: `AccuracyConfig`, `FDTDConfig`, `StoppingConfig`, `SymmetryEntry`, `ResolutionConfig`, `DomainConfig`, `SimConfig`, `PortData`, `LayerStackEntry`, `MaterialData`                                   |
| `meep/models/results.py` | `SParameterResult` — parses CSV output, complex S-params from mag+phase.                                                                                                                                          |
| `meep/overlay.py`        | `SimOverlay` + `PortOverlay` + `DielectricOverlay` dataclasses, `build_sim_overlay()` — visualization metadata for sim cell, PML regions, port markers, dielectric backgrounds.                                    |
| `meep/ports.py`          | `extract_port_info()` — port center/direction/normal from gdsfactory ports. `_get_z_center()` uses highest refractive index layer.                                                                                |
| `meep/materials.py`      | `resolve_materials()` — resolves layer material names to (n, k) via common DB.                                                                                                                                    |
| `meep/script.py`         | `generate_meep_script()` — cloud runner template. Reads GDS + config, builds `mp.Block` slabs + `mp.Prism` layers, `mp.Mirror` symmetries, decay-based or fixed stopping, `split_chunks_evenly`, polygon simplification, configurable `eps_averaging`/`subpixel_maxeval`/`subpixel_tol`, verbose progress stepping via `mp.at_every()`, saves S-params CSV. |

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
  sim_config.json    # layer_stack, dielectrics, ports, materials, fdtd, resolution, domain, accuracy, verbose_interval, symmetries
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

**Decision:** Add a `DomainConfig` model with `dpml`, `margin_xy`, `margin_z_above`, `margin_z_below`, `port_margin` fields, exposed via `sim.set_domain()`.

**Rationale:**

- PML thickness was hardcoded to 1.0 um in the runner script — not configurable
- Margins control how much material (cladding) is kept around the waveguide core
- `margin_z_above`/`margin_z_below` are used by `set_z_crop()` to determine the crop window
- `margin_xy` is the gap between geometry bbox and PML inner edge
- Background dielectric slabs (`stack.dielectrics`) are serialized to JSON and created as `mp.Block` objects in the runner, ensuring the entire simulation domain is filled with the correct material at each z-position (no vacuum gaps)
- `mp.Block` slabs are placed first in the geometry list; MEEP's "later objects override" rule ensures patterned waveguide prisms take precedence

### Simulation overlay visualization

**Decision:** Add `SimOverlay` / `PortOverlay` / `DielectricOverlay` dataclasses and draw them on `plot_2d()` cross-sections.

**Rationale:**

- `plot_2d()` previously only showed geometry prisms and a dashed bbox matching geometry bounds
- Users need to see the actual simulation cell (geometry + margin + PML), PML regions, and port locations
- Overlay rendering is solver-agnostic — `render2d.py` draws from generic overlay metadata
- Source ports shown in red with arrow, monitor ports in blue with arrow, PML as semi-transparent orange
- Dielectric background slabs shown as coloured bands in XZ/YZ views (SiO2 light blue, silicon grey, air transparent)

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

**1. Symmetry exploitation (2-4x speedup)**

- `SymmetryEntry` model with `direction` (X/Y/Z) and `phase` (+1/-1)
- `sim.set_symmetry(y=-1)` adds `mp.Mirror` symmetries to the MEEP runner
- MEEP only simulates half (or quarter) of the domain, mirroring the fields
- Most photonic components (MMIs, directional couplers) have at least one mirror plane

**2. Decay-based stopping (1.5-3x speedup)**

- `StoppingConfig` model embedded in `FDTDConfig` with `mode="fixed"|"decay"`
- Fixed mode: `sim.run(until_after_sources=N)` (original behavior)
- Decay mode: `mp.stop_when_fields_decayed(dt, component, point, decay_by)` — stops when fields at a monitor point decay below threshold, with `run_after_sources` as safety cap
- `resolve_decay_monitor_point()` auto-selects the first non-source port center
- `_COMPONENT_MAP` maps string names ("Ez", "Ey", ...) to MEEP field components

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

---

## Remaining Work / Known Issues

- **Cloud testing:** The full cloud round-trip (upload → run meep → download results) hasn't been tested end-to-end with a real cloud instance.
- **`_build_geometry_model()` note:** Uses `gf.get_active_pdk().layer_stack` (gdsfactory's LayerStack) instead of `self.stack` (gsim's LayerStack) because `LayeredComponentBase` needs gdsfactory's type for DerivedLayer resolution.

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

**Fix:** Added `port_margin` field to `DomainConfig` (default: 2.0 um). The runner script now uses `width + 2 * port_margin` for source/monitor transverse size (e.g. 0.5 + 4.0 = 4.5 um). This follows the gplugins convention where `port_margin=3.0` by default.

**Files changed:**
- `meep/models/config.py` — `DomainConfig.port_margin: float = 2.0`
- `meep/sim.py` — `set_domain(..., port_margin=2.0)`
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

### Decay component default changed to "Ey"

**Problem:** Default `decay_component` was `"Ez"`, but for TE-polarized modes in SOI (the dominant mode in 220nm silicon strip waveguides), the dominant electric field component is **Ey** (transverse to propagation, in-plane with the slab). Ez is the weakest TE component and can decay 10-100x faster than Ey, causing premature simulation termination.

**Fix:** Changed default from `"Ez"` to `"Ey"` in `StoppingConfig`, `set_wavelength()`, and the runner script fallback.

**Files changed:** `meep/models/config.py`, `meep/sim.py`, `meep/script.py`.

### Monitor port direction arrows in visualization

**Problem:** Only source ports showed direction arrows in the XY cross-section view. Monitor ports were drawn as blue lines without any directional indicator, making it hard to verify port orientation.

**Fix:** Removed the `if port.is_source:` guards in `_draw_overlay_xy()`. Now ALL ports (source and monitor) show direction arrows. Source arrows remain red, monitor arrows are blue — the color is already set per-port based on `port.is_source`.

**Files changed:** `common/viz/render2d.py`.

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

/app/src/s_parameters.csv
  entrypoint: cp *.csv /app/data/

    ↓ (tarballed and uploaded)

gsim receives s_parameters.csv → SParameterResult.from_csv()
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
sim.set_domain(0.5)  # 0.5um margins (default), 1um PML
sim.set_z_crop()  # crop to margin_z_above/below around core
sim.set_material("si", refractive_index=3.47)
sim.set_wavelength(
    wavelength=1.55, bandwidth=0.1,
    stop_when_decayed=True,  # decay-based stopping (1.5-3x faster)
)
sim.set_resolution(pixels_per_um=40)
sim.set_symmetry(y=-1)  # mirror symmetry (2-4x faster for symmetric components)
sim.set_accuracy(
    simplify_tol=0.01,       # simplify dense GDS polygons (10nm tolerance)
    eps_averaging=True,       # subpixel averaging (disable for quick tests)
    verbose_interval=5.0,     # progress prints every 5 MEEP time units
)
sim.set_output_dir("./meep-sim-test")

# Visualize before running
sim.plot_2d(slices="xyz")

# Write config files
sim.write_config()

# Run on cloud
result = sim.simulate()
result.plot()
```
