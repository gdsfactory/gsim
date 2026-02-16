# gsim.meep — Overview & Roadmap

## Architecture

```
GDS + PDK  ──►  3D Geometry  ──►  Client Viz  ──►  SimConfig JSON  ──►  Cloud Runner
(common/)       (GeometryModel)   (no meep)        (no meep)            (meep in Docker)
```

**Key constraint:** gsim is the client SDK — no meep dependency. MEEP runs only in Docker/cloud.

**Upload package:** `layout.gds` + `sim_config.json` + `run_meep.py` → Docker → `s_parameters.csv` + `meep_debug.json` + diagnostic PNGs.

---

## API: Declarative `Simulation` (current)

```python
from gsim import meep

sim = meep.Simulation()
sim.geometry.component = ybranch
sim.geometry.z_crop = "auto"
sim.materials = {"si": 3.47, "SiO2": 1.44}       # float shorthand → Material(n=...)
sim.source.port = "o1"
sim.source.wavelength = 1.55
sim.source.bandwidth = 0.1
sim.source.num_freqs = 11
sim.monitors = ["o1", "o2"]                          # just port names
sim.domain.pml = 1.0
sim.domain.margin = 0.5
sim.solver.resolution = 32
sim.solver.stopping = meep.DFTDecay(threshold=1e-3, min_time=100)
sim.solver.simplify_tol = 0.01
sim.solver.save_geometry = True
sim.solver.save_fields = True
sim.plot_2d(slices="xyz")
result = sim.run("./meep-sim")
```

### Design principles

- **6 typed physics objects** — `Geometry`, `Material`, `ModeSource`, `Domain`, `FDTD` + `monitors: list[str]` — assigned to a `Simulation` container. No ordering dependencies.
- **Attribute style or constructor style** — `sim.source.port = "o1"` (attribute, preferred) or `sim.source = ModeSource(port="o1")` (constructor). Both work via Pydantic `validate_assignment=True`.
- **Float shorthand for materials** — `{"si": 3.47}` auto-normalizes to `Material(n=3.47)` via validator.
- **Typed stopping variants** — `FixedTime`, `FieldDecay`, `DFTDecay` instead of string `mode`.
- **Source defines spectral window** — `WavelengthConfig` derived from `ModeSource.wavelength/bandwidth/num_freqs`. Monitors are just port name strings.
- **JSON contract unchanged** — `write_config()` translates new API → existing `SimConfig` → JSON. Runner template untouched.
- **Legacy `MeepSim`** kept for backward compat (imperative builder pattern with `set_*()` methods).

### What the new API fixed from the old review

| Old issue | Resolution |
|---|---|
| `FDTDConfig` misnamed | `WavelengthConfig` (renamed in Step 1), monitors now define wavelength directly |
| Sources/monitors invisible | Explicit `ModeSource` (with spectral window) + `monitors: list[str]` |
| `set_accuracy()` conflates concerns | `FDTD` holds solver numerics + diagnostics (subpixel, simplify, save, verbose) |
| `verbose_interval` homeless | Lives in `FDTD` |
| Builder ordering dependencies | Declarative — assign in any order, resolved at `write_config()` |
| Stopping uses MEEP-internal names | `DFTDecay(threshold=, min_time=)` — physicist-friendly |
| `set_domain(0.5)` positional arg | `Domain(margin=0.5)` — all keyword |

---

## Module Structure

### `gsim.meep` — Public API

| Module | Purpose |
|---|---|
| `__init__.py` | Exports: `Simulation`, `MeepSim`, all model classes |
| `models/api.py` | Declarative models: `Geometry`, `Material`, `ModeSource`, `Domain`, `FixedTime`, `FieldDecay`, `DFTDecay`, `FDTD` |
| `simulation.py` | `Simulation` container — `write_config(output_dir)` (translates to `SimConfig`), `run(output_dir)`, `validate_config()`, `plot_2d()`/`plot_3d()` |
| `sim.py` | Legacy `MeepSim` — imperative builder, backward compat |
| `base.py` | `MeepSimMixin` — shared viz, stack resolution, z-crop, material helpers |
| `models/config.py` | `SimConfig` + all sub-configs (JSON serialization layer) |
| `models/results.py` | `SParameterResult` — CSV + debug JSON + diagnostic PNGs |
| `ports.py` | `extract_port_info()` — port center/direction/normal from gdsfactory |
| `materials.py` | `resolve_materials()` — material names → (n, k) via common DB |
| `script.py` | `generate_meep_script()` — cloud runner template (string in Python) |
| `overlay.py` | `SimOverlay` + `PortOverlay` + `DielectricOverlay` — viz metadata |

### `gsim.common` — Shared infrastructure

Solver-agnostic: `LayeredComponentBase`, `GeometryModel`/`Prism`, `LayerStack`, `MaterialProperties`, `viz/` (Matplotlib 2D, PyVista/Open3D/Three.js 3D).

---

## Key Design Decisions

- **GDS-file approach** — Send raw GDS to cloud (not polygon coords in JSON). Complex geometries have 1000s of nodes; DerivedLayers need gdsfactory to resolve.
- **Port z-center = highest refractive index** — Photonic ports center on waveguide core (not conductor layer like RF).
- **Z-crop** — Auto-crops stack around core layer. Full UBC stack is 15um; cropped to ~1.5um (massive compute savings).
- **Port extension into PML** — `gf.components.extend_ports()` at `write_config()` time. Original bbox stored in `SimConfig.component_bbox` for correct cell sizing.
- **Symmetries disabled for S-params** — `add_mode_monitor` uses `use_symmetry=false` internally; `get_eigenmode_coefficients` doesn't apply `S.multiplicity()`. Source port coefficients underestimated ~2x. gplugins also never uses `mp.Mirror`.
- **`dft_min_run_time` default 100** — Prevents false convergence before pulse traverses device. Must exceed `device_length × n_group`.

---

## TODO

### Near-term

- [ ] **Cloud end-to-end test** — Full round-trip (upload → run meep → download results) hasn't been tested with real cloud instance.
- [ ] **Monitor z-span touches PML** — After z-crop, layer stack z-extent exactly matches PML inner boundary. At coarse resolution, outermost monitor pixels may straddle PML. Fix: shrink monitor z-span by ~`2/resolution` inset.
- [ ] **Empty `core2` layer in config** — UBC PDK's ebeam_y_1550 includes a `core2` entry (GDS layer 31) with zero geometry. Harmless but clutters config. Consider filtering empty layers.
- [ ] **`Simulation.set_stack()` / `set_z_crop()` convenience** — Currently stack is resolved lazily from PDK when `geometry.stack` is `None`. Could add explicit methods or make `z_crop="auto"` the default.

### Medium-term

- [ ] **Custom monitors** — Monitors are port-name strings only. Add `FieldMonitor(center, size)` and `FluxMonitor` for custom measurement locations.
- [ ] **Per-port margin/mode_index** — `port_margin` is global in `Domain`. Could add per-port control for margin and higher-order modes.
- [ ] **Typed boundaries** — Only PML today. Add `Periodic`, `Bloch`, `PEC`, `PMC` per-axis boundary specs.
- [ ] **Mesh refinement regions** — Single global `resolution`. Add local refinement for thin features.
- [ ] **Dispersive materials** — `Material(n, k)` is non-dispersive. Add Sellmeier/Lorentz/Drude support.
- [ ] **`updated_copy()` for parameter sweeps** — Pydantic `model_copy(update={...})` makes this nearly free.
- [ ] **`SParameterSim` high-level layer** — One-liner convenience API on top of `Simulation`.

### Deferred

- [ ] **JSON schema v2** — Restructure JSON groups (`"solver.wavelength"`, per-port margins, etc.) with version field for runner backward compat. Requires atomic client + runner update.
- [ ] **Port symmetries** — gplugins-style: run fewer source ports, copy S-params between symmetric port pairs (e.g. S31=S21 for Y-branch).
- [ ] **`add_flux` + `eig_parity` path** — Alternative to `add_mode_monitor` that works correctly with `mp.Mirror` symmetry. Needs auto-detection of correct parity from symmetry config and waveguide polarization.

---

## Docker / Cloud

- **Base:** `continuumio/miniconda3` → conda env with `pymeep=*=mpi_mpich_*`, `gdsfactory`, `nlopt`
- **Entrypoint:** downloads input → `mpirun -np $NP python run_meep.py` → uploads outputs
- **`meep_np` auto-computed** from problem size (`total_voxels // 200k`, clamped to physical cores)
- **Instance recommendation:** `c7a.16xlarge` (AMD EPYC Genoa, DDR5, ~460 GB/s mem BW) for typical photonics; `hpc7a.96xlarge` for large problems
- **Local testing:** `./nbs/test_meep_local.sh` (regenerate → Docker build → run → results)
