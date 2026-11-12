# RF benchmark: stored results

Case files for the RF benchmark of the gsim documentation. Each `*.json` file describes one device, what it is compared
with, and one row per compared quantity. `schema.json` is the JSON Schema every case file is checked against.

## Rules

- **Generic data only.** Open PDKs (gpdk, IHP Open PDK) and published papers. No foundry data, no customer data, no
  prices.
- **Every row records where it comes from.** Solver and version, gsim version and commit, element order, mesh,
  convergence, hardware, run time, date, and the notebook cell that produced it.
- **Status of a row.**
  - `published`: the row has its full provenance (`provenance_missing` is empty).
  - `reported`: the number was quoted from an earlier run whose solver version, hardware or mesh were not recorded. The
    row lists the missing fields in `provenance_missing` and a page prints them as "not recorded". They are filled in
    after the run is repeated with the provenance cell.
- **Status of a case file.** `published` when every row is published, `reported` when at least one row is reported, and
  `planned` when the case has an owner and a list of quantities but no rows.
- **Agreement is computed, never stored.** A reader takes `value` and `reference_value` and computes the relative error
  (for a complex impedance the distance in the complex plane over the magnitude of the reference). The files hold the
  two numbers and nothing derived from them.
- **`origin`.** `local` rows are recomputed by the notebook itself (the check cell at the end of each details notebook
  compares them with the file). `cloud` rows are copied from a cloud run and are not recomputed.
- **`role`.** `prediction`, `calibration` (a parameter was fitted to this reference), `consistency` (two solvers, or a
  solver and a closed form, on the same model) or `not-scored`. A calibration row only shows that the fit worked.
- **`reference_key`, `reference_key_imag`, `reference_row`.** Where the reference value is a number of
  `nbs/data/bj2012_reference.json`, its dotted path in `reference_key` (a list index is a number in the path; two paths
  joined by `|` mean the average of the two values, used for the mean of the paper's two arms); the imaginary part of a
  complex reference has `reference_key_imag`. Where the reference is the value of another row of the same file (one
  solver against another), that row's `id` is in `reference_row`. The tests check both copies.
- **`run_seconds`** is the wall time of the notebook cell that produced the row on the machine named in `hardware`.
  **`element_order`** is the element order of the solver that produced the number; for DEVSIM rows it is the order of
  the mesh elements (linear), and for closed-form rows it is the order of the solve that supplied their input.
- **`bias_v`** is the reverse bias of a row of a bias sweep.
- A row that quotes a number from a **file of the repository** instead of a notebook run (the IHP and the GSSG cases)
  names that file in `source`.

## Files

| File                            | Status                                                     | Source                                    |
| ------------------------------- | ---------------------------------------------------------- | ----------------------------------------- |
| `bj2012_electrode.json`         | reported (femwell rows published, Palace rows reported)    | `palace_bj2012_electrode_details.ipynb`   |
| `bj2012_modulator.json`         | published (DEVSIM, femwell, closed form)                   | `tcad_bj2012_modulator_details.ipynb`     |
| `ihp_l2n0.json`                 | reported (Palace against measurements: not validated)      | `nbs/data/ihp_validation/`                |
| `cpw_gssg_tied.json`            | reported (uniform-line consistency of terminal wave ports) | `docs/terminal_wave_ports.md`, section 12 |
| `cst_cpw.json`, `gfem_cpw.json` | planned (no rows)                                          | to be agreed                              |

After a cloud batch of the electrode notebook, `bj2012_electrode/` holds the raw S-parameters (`J2.s2p`, `J3.s2p`) and a
`provenance.json`. They are written by the notebook's storing cell when `RUN_CLOUD` is on.

## Adding a case

1. Copy the closest case file, set `case`, `status`, `device` and `reference`.
1. One row per quantity. `value` and `reference_value` are in `unit`; a complex impedance uses `value_imag` and
   `reference_value_imag`.
1. Fill the provenance fields from the run, or list what is missing in `provenance_missing` and set the row's status to
   `reported` (and the file's).
1. Validate: `uv run pytest tests/test_bj2012_notebooks.py` checks every case file against `schema.json`.
