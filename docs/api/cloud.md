# Cloud execution guide

Submit simulations, inspect job status, and load results through [`gsim.gcloud`][gsim.gcloud].

- [`run_simulation`][gsim.gcloud.run_simulation]
- [`get_status`][gsim.gcloud.get_status]
- [`wait_for_results`][gsim.gcloud.wait_for_results]
- [`RunResult`][gsim.gcloud.RunResult]

## Result caching

Passing `check_cache=True` to `sim.run()` looks for a completed cloud job with
byte-identical inputs and reuses its results instead of submitting a new job:

```python
sp = sim.run(check_cache=True)
```

The cache key is derived from the files written by `write_config()` — the same
bytes the solver consumes — rather than from the simulation object, so it also
covers changes to the generated solver script. A lookup failure is never fatal:
it degrades to a normal submit.

- [`check_cache`][gsim.gcloud.check_cache]
- [`check_cache_for_dir`][gsim.gcloud.check_cache_for_dir]
- [`compute_input_hash`][gsim.hashing.compute_input_hash]
