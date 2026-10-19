# Cloud execution guide

Submit simulations, inspect job status, and load results through [`gsim.gcloud`][gsim.gcloud].

- [`run_simulation`][gsim.gcloud.run_simulation]
- [`get_status`][gsim.gcloud.get_status]
- [`wait_for_results`][gsim.gcloud.wait_for_results]
- [`RunResult`][gsim.gcloud.RunResult]

## Sizing metadata

FDTD and Palace inputs include `metadata.json`. gsim extracts problem-size counts
and sends `sizing_requirements` to DataLab's prepare endpoint. This works with the
existing SDK; no SDK upgrade is required. DataLab chooses the smallest resource
profile covering the counts and rejects requirements exceeding its largest profile.
gsim does not choose CPU counts, RAM, or instance types.

The request contains one solver-specific object:

```json
{"sizing_requirements": {"fdtd": {"cells": 60480960}}}
```

```json
{"sizing_requirements": {"palace": {"tetrahedra": 34862, "field_dofs": 223152}}}
```

gsim prepares and uploads sized jobs directly, reusing the SDK's configured API
host, authentication, and job handle. Start, polling, and result downloads continue
through the SDK. ZIP creation uses a temporary directory and works on Windows
with older SDKs. Missing metadata, and Palace problems without a supported 3D
field-DOF estimate, retain the existing fixed-resource submission path. Malformed
counts and server rejections are reported as errors without retrying at a smaller size.

FDTD metadata includes measured mesh bounds, the requested `non_pml_bounds`,
cell size, PML thickness, and `grid.cell_counts`. These client estimates follow
the engine's `fdtd-z16-v1` layout: round each non-PML extent upward to whole
cells, align the Z interior and each Z PML face to 16 cells, and include one
guard cell per X/Y face and 16 per Z face. `total_cells_xyz` and `total_cells`
include PML, alignment, and guards; their product supplies `fdtd.cells`. These
remain client estimates tied to that engine layout.

The current Palace resource tiers are calibrated with MUMPS. Other linear
solvers can have different memory needs and convergence behavior; these tiers
are not validated sizing guarantees for every solver. The full diagnostic sidecar,
including solver settings, remains in the input ZIP. The deployed prepare API reads
the compact counts, not `sizing_metadata` or `input_manifest`. Palace field-DOF
estimates describe the mesh before Palace preprocessing. With AMR and an explicit
`MaxSize`, gsim sends the larger of that bound and the input estimate. Unbounded
refinement can grow beyond the initial estimate; the compact API does not validate
its peak resource needs.

## Palace size limits and out-of-memory failures

There is no single tetrahedron or field-DOF count that guarantees a Palace model
will fit in memory. The active cloud resource profiles determine which problem
sizes the prepare endpoint accepts. Supported sized submissions are checked before
gsim creates or uploads the input ZIP; a rejection includes DataLab's explanation.
Contact support for the current supported capacity if your model exceeds it.

Inspect `sim.mesh()`'s **Estimated Field DOFs**, or
`result.metadata["mesh"]["field_dofs"]["estimated_field_dofs"]`, before submission.
The estimate uses the configured field order, rather than mesh file size alone.
Acceptance is not a memory guarantee: solver settings, mesh preprocessing and
refinement can change peak memory beyond the sizing estimate.

`wait_for_results()` raises `RuntimeError` for a failed job, including in
`verbose="quiet"` mode. Exit 137 means SIGKILL, often an out-of-memory kill;
the error includes the backend reason, available solver logs and recovery advice.
Coarsen the mesh, reduce the model size, or lower Palace's field order with
`sim.solver.order` and recheck convergence. Cloud resources are selected
automatically; contact support with the job ID if more memory is needed.
Jobs without supported sizing metadata use fixed resources and do not receive
the prepare endpoint's problem-size check.

## Result caching

Passing `check_cache=True` to `sim.run()` looks for a completed cloud job with
byte-identical inputs and reuses its results instead of submitting a new job:

```python
sp = sim.run(check_cache=True)
```

The cache key is derived from the files written by `write_config()` — the same
bytes the solver consumes — rather than from the simulation object, so it also
covers changes to the generated solver script. FDTD and Palace exclude the
root-level diagnostic `metadata.json` from physics cache keys; it is still
included in the uploaded ZIP. A lookup failure is never fatal:
it degrades to a normal submit.

- [`check_cache`][gsim.gcloud.check_cache]
- [`check_cache_for_dir`][gsim.gcloud.check_cache_for_dir]
- [`compute_input_hash`][gsim.hashing.compute_input_hash]
