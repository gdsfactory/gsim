# gsim.meep — Overview & Roadmap

## Architecture

```
GDS + PDK  ──►  3D Geometry  ──►  Client Viz  ──►  SimConfig JSON  ──►  Cloud Runner
(common/)       (GeometryModel)   (no meep)        (no meep)            (meep in Docker)
```

**Key constraint:** gsim is the client SDK — no meep dependency. MEEP runs only in Docker/cloud.

**Upload package:** `layout.gds` + `sim_config.json` + `run_meep.py` -> Docker -> `s_parameters.csv` + `meep_debug.json` + diagnostic PNGs.

---

## API: Declarative `Simulation`

```python
from gsim import meep

sim = meep.Simulation()
sim.geometry(component=ybranch, z_crop="auto")
sim.materials = {"si": 3.47, "SiO2": 1.44}       # float shorthand
sim.source(port="o1", wavelength=1.55, bandwidth=0.1, num_freqs=11)
sim.monitors = ["o1", "o2"]
sim.domain(pml=1.0, margin=0.5)
sim.solver(resolution=32, simplify_tol=0.01)
sim.plot_2d(slices="xyz")
result = sim.run()
```

### Design principles

- **6 typed physics objects** -- `Geometry`, `Material`, `ModeSource`, `Domain`, `FDTD` + `monitors: list[str]` -- assigned to a `Simulation` container. No ordering dependencies.
- **Callable, attribute, or constructor style** -- `sim.source(port="o1")` (callable), `sim.source.port = "o1"` (attribute), or `sim.source = ModeSource(port="o1")` (constructor). All work via Pydantic `validate_assignment=True`.
- **Float shorthand for materials** -- `{"si": 3.47}` auto-normalizes to `Material(n=3.47)` via validator.
- **5 stopping methods** -- Flat `stopping` mode string on `FDTD` (`field_decay`/`energy_decay`/`dft_decay`/`fixed`) plus convenience methods: `stop_when_fields_decayed()`, `stop_when_energy_decayed()`, `stop_when_dft_decayed()`, `stop_after_sources()`. A 5th method `stop_after_walltime(seconds)` sets an orthogonal wall-clock safety net (field `wall_time_max`) that can be combined with any mode.
- **Source defines spectral window** -- `WavelengthConfig` derived from `ModeSource.wavelength/bandwidth/num_freqs`. Monitors are just port name strings.
- **JSON contract** -- `write_config()` translates new API -> existing `SimConfig` -> JSON. `StoppingConfig.mode` extended with `"energy_decay"` (runner updated to handle it).

---

## Module Structure

### `gsim.meep` -- Public API

| Module              | Purpose                                                                                                                                                               |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`       | Exports: `Simulation`, all model classes                                                                                                                              |
| `models/api.py`     | Declarative models: `Geometry`, `Material`, `ModeSource`, `Domain`, `FDTD`, `Symmetry`. All except `Material`/`Symmetry` have `__call__(**kwargs)` for fluent updates |
| `simulation.py`     | `Simulation` container -- `write_config()`, `run()`, `validate_config()`, `plot_2d()`/`plot_3d()`, `estimate_meep_np()`                                               |
| `viz.py`            | Standalone viz helpers -- `build_geometry_model()`, `plot_2d()`, `plot_3d()` (calls `common/viz`)                                                                     |
| `models/config.py`  | `SimConfig` + all sub-configs (JSON serialization layer, strict — no defaults on fields that `build_config` always populates)                                         |
| `models/results.py` | `SParameterResult` -- CSV + debug JSON + diagnostic PNGs                                                                                                              |
| `ports.py`          | `extract_port_info()` -- port center/direction/normal from gdsfactory                                                                                                 |
| `materials.py`      | `resolve_materials()` -- material names -> (n, k) via common DB                                                                                                       |
| `script.py`         | `generate_meep_script()` -- cloud runner template (string in Python)                                                                                                  |
| `overlay.py`        | `SimOverlay` + `PortOverlay` + `DielectricOverlay` -- viz metadata                                                                                                    |

### `gsim.common` -- Shared infrastructure

Solver-agnostic: `LayeredComponentBase`, `GeometryModel`/`Prism`, `LayerStack`, `ValidationResult`, `viz/` (Matplotlib 2D, PyVista/Open3D 3D).

Key sub-modules:

- `stack/_layer_utils.py` — shared `get_gds_layer_tuple()` and `classify_layer_type()` (used by both `extractor` and `visualization`)
- `viz/_mesh_helpers.py` — shared Delaunay triangulation (`triangulate_polygon_with_holes`) and batch prism geometry (`collect_triangular_prism_geometry`), used by both PyVista and Open3D renderers

### Visualization pipeline

3D backends: PyVista (desktop) and Open3D+Plotly (Jupyter, with view buttons for Iso/Top/Front/Right). Three.js backend removed.

```
Simulation.plot_2d()
  -> viz.build_geometry_model(component, stack, domain_config)
       -> LayeredComponentBase + extract_geometry_model()
       -> crop_geometry_model() if stack is z-cropped
  -> viz.build_overlay(gm, component, stack, domain_config)
       -> overlay.build_sim_overlay()
  -> common/viz/render2d.plot_prism_slices(gm, overlay=overlay)
```

- **Z slices** -- draw actual prism polygon outlines at the slice plane (XY view)
- **X/Y slices** -- compute Shapely line-polygon intersections for accurate cross-sections (XZ/YZ views); each intersection segment becomes a rectangle from z_base to z_top
- **Overlay** -- sim cell boundary, PML regions, port markers, dielectric backgrounds

---

## Key Design Decisions

- **GDS-file approach** -- Send raw GDS to cloud (not polygon coords in JSON). Complex geometries have 1000s of nodes; DerivedLayers need gdsfactory to resolve.
- **Port z-center = highest refractive index** -- Photonic ports center on waveguide core (not conductor layer like RF).
- **Z-crop** -- Auto-crops stack around core layer. Full UBC stack is 15um; cropped to ~1.5um (massive compute savings).
- **Port extension into PML** -- `gf.components.extend_ports()` at `write_config()` time. Original bbox stored in `SimConfig.component_bbox` for correct cell sizing.
- **Symmetries disabled for S-params** -- `add_mode_monitor` uses `use_symmetry=false` internally; `get_eigenmode_coefficients` doesn't apply `S.multiplicity()`. Source port coefficients underestimated ~2x. gplugins also never uses `mp.Mirror`.
- **`field_decay` default stopping** -- Monitors |component|² at a point and stops when it decays by `decay_by` from peak. Runner uses `mp.stop_when_fields_decayed()` + `_make_time_cap(max_time)` in an `until_after_sources` list (OR logic — first condition wins). `energy_decay` is similar but monitors total cell energy (more robust for ring couplers where DFTs can falsely converge). `stop_after_sources(time)` runs for a fixed sim-time after sources turn off.
- **Wall-clock safety net** -- `stop_after_walltime(seconds)` sets `wall_time_max` on `FDTD`, an orthogonal real-time cap. Runner's `_make_wall_time_cap()` uses `time.time()` deadline and is appended to the `until_after_sources` list in all 4 mode branches when `wall_time_max > 0`. A warning is logged when the wall-clock limit likely triggered (elapsed >= 95% of limit).
- **`dft_min_run_time` is absolute sim time** -- Not time-after-sources. With a broadband source turning off at ~t=78 and min_run_time=100, DFT checking starts at t=100 (only ~22 units after source ends). Must exceed `device_length * n_group`.
- **API models are the single source of truth for defaults** -- Config models (`models/config.py`) and the cloud runner (`script.py`) have no default values for fields that `build_config()` always populates. `model_dump(by_alias=True)` serializes all fields explicitly, so config defaults and script.py `.get()` fallbacks were dead code that could silently drift. Config sub-models are strict (missing fields = `ValidationError`); `script.py` uses direct `dict["key"]` access.
- **Stack resolved lazily** -- `_ensure_stack()` falls back to `get_stack()` (active PDK defaults) when no explicit stack kwargs are set. Same path for `write_config()` and viz.

---

## TODO

### Near-term

- [ ] **Cloud end-to-end test** -- Full round-trip (upload -> run meep -> download results) hasn't been tested with real cloud instance.
- [ ] **Monitor z-span touches PML** -- After z-crop, layer stack z-extent exactly matches PML inner boundary. At coarse resolution, outermost monitor pixels may straddle PML. Fix: shrink monitor z-span by ~`2/resolution` inset.
- [ ] **Empty `core2` layer in config** -- UBC PDK's ebeam_y_1550 includes a `core2` entry (GDS layer 31) with zero geometry. Harmless but clutters config. Consider filtering empty layers.

### Medium-term

- [ ] **Custom monitors** -- Monitors are port-name strings only. Add `FieldMonitor(center, size)` and `FluxMonitor` for custom measurement locations.
- [ ] **Per-port margin/mode_index** -- `port_margin` is global in `Domain`. Could add per-port control for margin and higher-order modes.
- [ ] **Typed boundaries** -- Only PML today. Add `Periodic`, `Bloch`, `PEC`, `PMC` per-axis boundary specs.
- [ ] **Mesh refinement regions** -- Single global `resolution`. Add local refinement for thin features.
- [ ] **Dispersive materials** -- `Material(n, k)` is non-dispersive. Add Sellmeier/Lorentz/Drude support.
- [ ] **`updated_copy()` for parameter sweeps** -- Pydantic `model_copy(update={...})` makes this nearly free.

### Deferred

- [ ] **JSON schema v2** -- Restructure JSON groups (`"solver.wavelength"`, per-port margins, etc.) with version field for runner backward compat. Requires atomic client + runner update.
- [ ] **Port symmetries** -- gplugins-style: run fewer source ports, copy S-params between symmetric port pairs (e.g. S31=S21 for Y-branch).
- [ ] **`add_flux` + `eig_parity` path** -- Alternative to `add_mode_monitor` that works correctly with `mp.Mirror` symmetry. Needs auto-detection of correct parity from symmetry config and waveguide polarization.

---

## Docker / Cloud

- **Base:** `continuumio/miniconda3` -> conda env with `pymeep=*=mpi_mpich_*`, `gdsfactory`, `nlopt`
- **Entrypoint:** downloads input -> `mpirun -np $NP python run_meep.py` -> uploads outputs
- **`meep_np` hardcoded to 2** -- auto-compute logic exists (`total_voxels // 200k`) but `lscpu` reports host cores inside containers, not Batch-allocated vCPUs. TODO: pass Batch vCPU allocation as `MEEP_NP` env var from job submission, then restore auto-compute with proper clamping
- **Instance recommendation:** `c7a.16xlarge` (AMD EPYC Genoa, DDR5, ~460 GB/s mem BW) for typical photonics; `hpc7a.96xlarge` for large problems
- **Local testing:** `./nbs/test_meep_local.sh` (regenerate -> Docker build -> run -> results)
