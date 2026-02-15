# gsim.meep API Review — Physicist/EE Perspective

## What works well

1. **Config-as-JSON architecture** — The definition/execution split (local Python SDK → JSON + GDS → cloud MEEP runner) is exactly right. Tidy3D does the same thing and it's what makes cloud execution, reproducibility, and caching possible.

2. **Pydantic-based validation** — Type-checking at construction time, computed fields (`fcen`, `df`), and `validate_config()` before submission. This catches errors early.

3. **Sensible defaults** — A user can get a working sim with minimal calls. The auto source bandwidth (`3x monitor bw`), auto port extension, and `dft_decay` stopping mode are all good physics-informed defaults.

4. **Good docstrings and warnings** — The symmetry warning is a great example of domain knowledge baked into the API.

---

## Issues — Naming

### 1. `FDTDConfig` is misnamed — it's a wavelength/frequency config

This class contains `wavelength`, `bandwidth`, `num_freqs`, `fcen`, `df`. None of these are "FDTD settings." FDTD settings would be Courant factor, time step, boundary conditions, etc. This is like naming a class `NewtonSolverConfig` when it only holds mass and velocity.

**Suggestion**: Rename to `WavelengthConfig` or `FrequencyConfig`. The user calls `set_wavelength()` — the internal name should match.

### 2. `set_stopping()` parameter names diverge from config field names

| User-facing param | Internal config field |
|---|---|
| `max_time` | `run_after_sources` |
| `threshold` | `decay_by` |

The user-facing names are better (more physics-meaningful), but the config field names leak MEEP internals. A physicist doesn't know what "run after sources" means. Also `decay_by` reads awkwardly — "decay by 1e-3" vs "decay to 1e-3" vs "decay threshold 1e-3."

**Suggestion**: Use consistent names on both sides. `max_time` / `threshold` everywhere, or `run_time_limit` / `decay_threshold`.

### 3. `verbose_interval` is homeless

It's a top-level field on `SimConfig`, set via `set_accuracy()`. A physicist looking for "how do I see progress output" would never think to look under accuracy settings. It's not an accuracy concern — it's a logging/output concern.

**Suggestion**: Move to `DiagnosticsConfig` (or a new `OutputConfig`). It naturally belongs with `save_geometry`, `save_fields`, etc.

### 4. `DomainConfig.port_margin` and `extend_ports` don't belong in domain config

`port_margin` controls the transverse extent of mode monitors — it's a monitor concern, not a domain sizing concern. `extend_ports` modifies geometry (extends waveguides into PML) — it's a geometry concern. Both are shoehorned into "domain" because they relate to spatial layout, but conceptually they're distinct.

**Suggestion**: `port_margin` → part of a `MonitorConfig` or `PortConfig`. `extend_ports` → part of geometry config or a dedicated boundary/PML config.

---

## Issues — Separation of Concerns

### 5. Sources and monitors are invisible — biggest conceptual gap

In every major FDTD tool (Tidy3D, Lumerical, COMSOL), sources and monitors are explicit, first-class objects that the user defines. In gsim.meep, they're auto-generated from ports. The user never sees a `Source` or `Monitor` object.

This is fine for the "S-parameter extraction from a photonic component" use case, but:
- It's **opaque**: the user can't inspect what source/monitor geometry was created
- It's **inflexible**: what if they want a field monitor at a custom location? A power flux monitor? A time monitor?
- It's **non-standard**: every physicist expects to explicitly define what they're exciting and what they're measuring

Compare Tidy3D:
```python
sources=[td.ModeSource(center=(-1.5, 0, 0), size=(0, 2, 2), ...)]
monitors=[td.ModeMonitor(center=(1.5, 0, 0), size=(0, 2, 2), name="output", ...)]
```

Even if you auto-generate from ports by default, the user should be able to see and override the source/monitor list.

### 6. `set_accuracy()` conflates three unrelated concerns

Currently bundles:
- **Subpixel averaging** (`eps_averaging`, `subpixel_maxeval`, `subpixel_tol`) — a solver/mesh concern
- **Polygon simplification** (`simplify_tol`) — a geometry pre-processing concern
- **Progress printing** (`verbose_interval`) — a logging concern

These have nothing to do with each other. A physicist adjusting mesh accuracy shouldn't need to think about polygon simplification or log verbosity in the same call.

### 7. Materials are weakly typed strings

The material binding chain is: GDS layer → `LayerStackEntry.material` (a string like `"si"`) → `MaterialData` (n, k). Materials are just strings that get resolved at `write_config()` time. There's no way to:
- Define a dispersive material (Sellmeier, Lorentz, Drude)
- Use anisotropic materials
- Import a material from a database by name with wavelength-dependent data
- Know at set-time whether a material name will resolve

Compare Tidy3D where `td.Medium(permittivity=11.7)` is a typed object you can inspect, compose, and validate immediately.

---

## Issues — API Style

### 8. Imperative builder pattern creates ordering dependencies

The current API requires calls in a specific order:
```python
sim.set_stack()       # must come before set_z_crop
sim.set_z_crop()      # must come after set_stack
sim.set_wavelength()  # must come before write_config (for fwidth computation)
```

This is the Lumerical pattern. The user has to know the dependency graph. Tidy3D avoids this entirely — everything goes into one constructor call, and computed fields are derived lazily.

### 9. No `updated_copy()` pattern for creating variants

A common physics workflow is: run a simulation, then sweep a parameter. Currently there's no way to create a modified copy:
```python
# Desired (Tidy3D-style):
sim_v2 = sim.updated_copy(resolution=ResolutionConfig(pixels_per_um=64))

# Current: start from scratch or manually re-set
```

Since `MeepSim` is a Pydantic model, you could get `model_copy(update={...})` almost for free.

### 10. `set_domain(0.5)` — inconsistent positional arg

`margin` is the only positional argument across all `set_*` methods — every other parameter is keyword-only (good). This one inconsistency breaks the pattern.

---

## Issues — Missing Concepts

### 11. No boundary conditions beyond PML

There's no way to set periodic, Bloch, PEC, or PMC boundaries. PML thickness is buried in `DomainConfig`. Compare Tidy3D:
```python
td.BoundarySpec(
    x=td.Boundary.pml(),
    y=td.Boundary.periodic(),
    z=td.Boundary.pml(),
)
```

### 12. No mesh refinement

A single `pixels_per_um` for the entire domain. No per-axis control, no local refinement regions. For practical photonic devices (e.g., thin slabs, sharp features), local mesh refinement is essential for accuracy without blowing up memory.

### 13. No run_time as a direct concept

A physicist thinks "I want to simulate for 500 femtoseconds." Instead they have to think in "MEEP time units after sources turn off" which is not physical time in seconds, and is relative to when sources end, not absolute. The `StoppingConfig` is solver-internal language, not physics language.

---

## Suggested Concern Mapping

| Concern | Current | Suggested |
|---|---|---|
| **Wavelength/Frequency** | `FDTDConfig` via `set_wavelength()` | `WavelengthConfig` |
| **Geometry** | `set_geometry()` + `set_stack()` + `set_z_crop()` | Keep, but make `set_z_crop()` auto on `set_stack()` |
| **Materials** | `set_material()` (string-based) | Typed `Medium` objects |
| **Source** | `set_source()` (implicit) | Explicit `Source` list (auto-generated from ports by default) |
| **Monitors** | Hidden (auto from ports) | Explicit `Monitor` list (auto-generated from ports by default) |
| **Mesh/Grid** | `ResolutionConfig` via `set_resolution()` | `GridConfig` with global + override regions |
| **Boundaries** | PML only, in `DomainConfig` | `BoundaryConfig` per-axis |
| **Domain/Cell** | `DomainConfig` (overloaded) | `DomainConfig` (margins only) |
| **Solver** | `StoppingConfig` via `set_stopping()` | `SolverConfig` (time, convergence) |
| **Output** | `DiagnosticsConfig` + stray `verbose_interval` | `OutputConfig` (all output settings) |

---

## Priority Recommendations

If prioritizing changes that give the most "aha, this makes sense" improvement for a new user:

1. **Rename `FDTDConfig` → `WavelengthConfig`** — Immediate clarity gain, no behavior change
2. **Move `verbose_interval` into `DiagnosticsConfig`** — Quick fix, correct grouping
3. **Make `set_domain()` keyword-only** — Consistency with all other `set_*` methods
4. **Align `StoppingConfig` field names** with user-facing params — `max_time`/`threshold` everywhere
5. **Split `set_accuracy()`** — separate subpixel settings from geometry simplification
6. **Expose auto-generated sources/monitors** — at minimum, a `sim.show_sources()` / `sim.show_monitors()` for inspection; ideally, explicit `Source` / `Monitor` objects

These are roughly in order of effort (1-4 are trivial renames, 5 is a small refactor, 6 is a design change).

---

## Chosen Direction: Option B (Improved Builder with Concern Groups)

After evaluating three design options (see `docs/design/`), we chose **Option B** — keep the familiar `set_*()` builder pattern but fix the concern groupings. The builder pattern fits the iterative Jupyter notebook workflow better than a fully declarative approach (Option A), and the concern-group cleanup is the real high-value change.

Option C's `SParameterSim` high-level layer can be added on top later.

### Design docs

- `docs/design/option-a-declarative-api.md` — Tidy3D-style fully declarative (reference, not chosen)
- `docs/design/option-b-builder-with-concern-groups.md` — **Chosen**: COMSOL-inspired concern groups
- `docs/design/option-c-layered-api.md` — Two-layer progressive disclosure (future addition)

---

## Implementation Plan

### Dependency chain

Every change touches up to 4 files in a chain:

```
models/config.py  →  sim.py  →  script.py (runner template)  →  generate_meep_config.py
   (Pydantic)       (set_*())     (reads JSON in Docker)          (example/test)
```

JSON restructuring requires updating both client models AND the runner template atomically. We phase the work to minimize risk.

### Step 1 — Safe renames, no JSON change ✅ Done

Use Pydantic `serialization_alias` to rename Python classes/fields while keeping JSON output identical. The runner template does not change at all.

| Change | Detail | Status |
|---|---|---|
| `FDTDConfig` → `WavelengthConfig` | Rename class. `SimConfig` field `fdtd` → `wavelength` with `serialization_alias="fdtd"`. `FDTDConfig` kept as backward-compat alias. | ✅ |
| `StoppingConfig.run_after_sources` → `max_time` | Rename field, `serialization_alias="run_after_sources"` for JSON. | ✅ |
| `StoppingConfig.decay_by` → `threshold` | Same alias pattern, `serialization_alias="decay_by"`. | ✅ |
| `verbose_interval` → into `DiagnosticsConfig` | Added to `DiagnosticsConfig`. `set_diagnostics(verbose_interval=)` is the new home. `set_accuracy(verbose_interval=)` still works but emits `DeprecationWarning`. `SimConfig` keeps top-level `verbose_interval` for the runner, populated from `diagnostics_config` in `write_config()`. | ✅ |
| `set_domain(margin, ...)` → `set_domain(*, margin, ...)` | Made keyword-only. | ✅ |
| `set_material(name, refractive_index=)` → also accept `n=`, `k=` | Added short alias params with mutual exclusion check. | ✅ |
| `to_json()` uses `by_alias=True` | One-line change to make `serialization_alias` effective. | ✅ |

**Files touched:** `models/config.py`, `models/__init__.py`, `meep/__init__.py`, `sim.py`, `base.py`, `generate_meep_config.py`, `tests/meep/test_meep_sim.py`
**Files NOT touched:** `script.py` (runner) — JSON is identical.

### Step 2 — Add `inspect_ports()` and `set_source_port()`

Purely additive — new methods, no breaking changes.

| Change | Detail |
|---|---|
| `inspect_ports()` → `list[PortEntry]` | Returns auto-generated port list after `set_geometry()`. |
| `set_source_port(port, *, bandwidth)` | Combines current `set_source()` with clearer name. |
| `PortEntry` gets `margin` and `mode_index` | Per-port fields with defaults matching current behavior (0.5, 1). |
| Deprecate `set_source()` | Thin wrapper calling `set_source_port()`, emits warning. |

**Files touched:** `sim.py`, `models/config.py` (add per-port fields to `PortData`)
**Files NOT touched:** `script.py` — runner already reads `port_margin` from domain config; per-port margin is a future runner change.

### Step 3 — Split `set_accuracy()`

New methods alongside old one, with deprecation wrapper.

| Change | Detail |
|---|---|
| New `set_subpixel(eps_averaging, maxeval, tol)` | Subpixel averaging settings (solver concern). |
| `simplify_tol` moves to `set_geometry()` or stays in `set_subpixel()` | Geometry pre-processing concern. TBD — may keep together since both affect mesh quality. |
| `verbose_interval` already moved in Step 1 | Into `DiagnosticsConfig`, set via `set_diagnostics()`. |
| Deprecate `set_accuracy()` | Wrapper that calls `set_subpixel()` + `set_diagnostics(verbose_interval=)`. |

### Step 4 — JSON restructuring + runner update (bigger change)

Atomic change: restructure JSON groups AND update runner template in one commit.

| Change | Detail |
|---|---|
| JSON: `"fdtd"` → `"solver.wavelength"` | Nest wavelength inside solver group. |
| JSON: `"domain.port_margin"` → `"ports.entries[].margin"` | Per-port margin. |
| JSON: `"domain.extend_ports"` → `"ports.extend_into_pml"` | Move to ports group. |
| JSON: top-level `"verbose_interval"` → `"output.verbose_interval"` | Consolidate output. |
| JSON: add `"config_version": 2` | Version field for runner backward compat. |
| Runner: update all `config["fdtd"]` → `config["solver"]["wavelength"]` reads | Matching path changes. |
| Runner: support `config_version` dispatch | Read v1 or v2 format. |

**Risk:** This is the only step where a mismatch between client and runner breaks Docker runs. Must be tested end-to-end with `test_meep_local.sh`.

### Future steps (not in this round)

- **Deduplicate `set_*()` defaults** — `set_*()` method signatures duplicate defaults already defined in Pydantic config models (e.g. `eps_averaging=False` in both `AccuracyConfig` and `set_accuracy()`). Make config models the single source of truth, e.g. `def set_accuracy(self, **kwargs): self.accuracy_config = AccuracyConfig(**kwargs)`. Tradeoff: loses explicit parameter names in method signature, but IDE autocomplete still works from the config class.
- **`SParameterSim` high-level layer** (Option C) — one-liner API on top of `MeepSim`
- **Custom monitors** — `add_monitor(FieldMonitor(...))` on `MeepSim`
- **Mesh refinement regions** — `add_mesh_region(center, size, resolution)`
- **Typed boundaries** — `BoundarySpec(x=PML(), y=Periodic())`
- **`get_sparameters()` convenience function**
