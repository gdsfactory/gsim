# Recorded PDK validation runs

Each `2d_<case>/` directory is one fresh validation run recorded with the canonical PDK-authoritative stack.
`sim_config.json` records the PDK name, overlay, stack hash, resolved layer/material table, resolution and provenance.

Fresh quantitative cases: UBC straight, coupler, ring coupling and Bragg; Cornerstone coupler and ring coupling;
generic-photonic-PDK SiEPIC coupler. All use the 2-D effective-index plane. The measured MZI is runtime-excluded
(see `_excluded_runtime/`), not part of the quantitative validation set.

The legacy `_stale_inferred_stack/` runs predate the PDK-authoritative stack and are retained only for history.
