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
| **3. Visualization** | Done   | `common/viz/` — PyVista, Open3D+Plotly, Three.js, Matplotlib                  |
| **4. Sim config**    | Done   | `SimConfig` JSON with layer_stack, ports, materials, fdtd, resolution, margin |
| **5. Cloud runner**  | Done   | `run_meep.py` reads GDS via gdsfactory, Delaunay triangulation for holes      |
| **6. Docker**        | Done   | `simulation-engines/meep/` — MPI-enabled pymeep, follows palace conventions   |

Tests: 62 meep-specific, 119 total passing.

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
| `meep/__init__.py`       | Public API: `MeepSim`, `FDTDConfig`, `MarginConfig`, `ResolutionConfig`, `SimConfig`, `SParameterResult`                                                                                                          |
| `meep/base.py`           | `MeepSimMixin` — fluent API (`set_geometry`, `set_stack`, `set_z_crop`, `set_wavelength`, `set_resolution`, `set_margin`, `set_material`, `set_source_port`). Builds `GeometryModel` + `SimOverlay` for viz.       |
| `meep/sim.py`            | `MeepSim` — main class. `write_config()` exports `layout.gds` + `sim_config.json` + `run_meep.py`. `simulate()` calls gcloud. `plot_2d()`/`plot_3d()` for client-side viz.                                        |
| `meep/models/config.py`  | Pydantic models: `FDTDConfig`, `ResolutionConfig`, `MarginConfig`, `SimConfig`, `PortData`, `LayerStackEntry`, `MaterialData`                                                                                     |
| `meep/models/results.py` | `SParameterResult` — parses CSV output, complex S-params from mag+phase.                                                                                                                                          |
| `meep/overlay.py`        | `SimOverlay` + `PortOverlay` dataclasses, `build_sim_overlay()` — visualization metadata for sim cell, PML regions, port markers.                                                                                 |
| `meep/ports.py`          | `extract_port_info()` — port center/direction/normal from gdsfactory ports. `_get_z_center()` uses highest refractive index layer.                                                                                |
| `meep/materials.py`      | `resolve_materials()` — resolves layer material names to (n, k) via common DB.                                                                                                                                    |
| `meep/script.py`         | `generate_meep_script()` — cloud runner template. Reads GDS + config (incl. margin), builds `mp.Prism`, handles holes via Delaunay, runs FDTD with `until_after_sources`, saves S-params CSV.                     |

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
  sim_config.json    # layer_stack, ports, materials, fdtd, resolution, margin
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

### Margin / PML configuration

**Decision:** Add a `MarginConfig` model with `pml_thickness`, `margin_xy`, `margin_z` fields, exposed via `sim.set_margin()`.

**Rationale:**

- PML thickness was hardcoded to 1.0 um in the runner script — not configurable
- Extra margin between geometry and PML is useful for avoiding evanescent field interaction with PML
- Margin values are serialized into `sim_config.json` and read by the cloud runner

### Simulation overlay visualization

**Decision:** Add `SimOverlay` / `PortOverlay` dataclasses and draw them on `plot_2d()` cross-sections.

**Rationale:**

- `plot_2d()` previously only showed geometry prisms and a dashed bbox matching geometry bounds
- Users need to see the actual simulation cell (geometry + margin + PML), PML regions, and port locations
- Overlay rendering is solver-agnostic — `render2d.py` draws from generic overlay metadata
- Source ports shown in red with arrow, monitor ports in blue, PML as semi-transparent orange

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
- Default: 2um padding above and below core
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

```bash
cd simulation-engines/meep
docker build -f Dockerfile.local -t meep-gsim:local .
docker run --rm -e MEEP_NP=4 -v /tmp/meep-out:/app/data meep-gsim:local
ls /tmp/meep-out/s_parameters.csv
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
sim.set_z_crop()  # crop to 2um above/below core (removes metals, substrate)
sim.set_material("si", refractive_index=3.47)
sim.set_wavelength(wavelength=1.55, bandwidth=0.1, run_after_sources=100)
sim.set_resolution(pixels_per_um=40)
sim.set_margin(pml_thickness=1.0)
sim.set_output_dir("./meep-sim-test")

# Visualize before running
sim.plot_2d(slices="xyz")

# Write config files
sim.write_config()

# Run on cloud
result = sim.simulate()
result.plot()
```
