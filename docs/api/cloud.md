# Cloud execution guide

Submit simulations, inspect job status, and load results through [`gsim.gcloud`][gsim.gcloud].

- [`run_simulation`][gsim.gcloud.run_simulation]
- [`get_status`][gsim.gcloud.get_status]
- [`wait_for_results`][gsim.gcloud.wait_for_results]
- [`RunResult`][gsim.gcloud.RunResult]

## Sizing metadata

FDTD and Palace inputs include `metadata.json`. When the installed SDK supports
`sizing_metadata` and `input_manifest`, gsim forwards the metadata and per-file
SHA-256 hashes during submission. Otherwise, it warns and submits with fixed
resources. The server decides admission and resource allocation.

FDTD metadata includes measured mesh bounds, the requested `non_pml_bounds`,
cell size, PML thickness, and `grid.cell_counts`. These client estimates follow
the engine's `fdtd-z16-v1` layout: round each non-PML extent upward to whole
cells, align the Z interior and each Z PML face to 16 cells, and include one
guard cell per X/Y face and 16 per Z face. `total_cells_xyz` and `total_cells`
include PML, alignment, and guards. The server must recompute the layout for
its engine version and compare counts before applying cell-count limits.

The current Palace resource tiers are calibrated with MUMPS. Other linear
solvers can have different memory needs and convergence behavior; these tiers
are not validated sizing guarantees for every solver. The metadata includes
the selected linear solver so the server can account for it. Palace field-DOF
estimates describe the mesh before Palace preprocessing; validate them against
completed solver jobs before enforcing admission boundaries.

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
included in the upload manifest. A lookup failure is never fatal:
it degrades to a normal submit.

- [`check_cache`][gsim.gcloud.check_cache]
- [`check_cache_for_dir`][gsim.gcloud.check_cache_for_dir]
- [`compute_input_hash`][gsim.hashing.compute_input_hash]
