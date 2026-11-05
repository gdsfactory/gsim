# Validate Palace result evidence

[`validate_solution`][gsim.palace.validation.validate_solution] reports core solver convergence, optional error
estimation, adaptive sampling, process exit and OOM evidence, mode checks, and input provenance independently. A zero
exit code does not establish physical validity. The function reads supplied evidence and never changes tolerances or
reruns a simulation.

```python
import json
from pathlib import Path
from gsim.palace.validation import validate_solution

output = Path("run/output/palace")
config = json.loads(Path("run/config.json").read_text())
report = validate_solution(
    (output / "palace.log").read_text(encoding="utf-8"),
    metadata=json.loads((output / "palace.json").read_text()),
    estimator_disabled=config["Solver"]["Linear"].get("EstimatorMaxIts") == 0,
    expected_excitations=[1, 2],  # Require these adaptive training outcomes.
    input_files={"config.json": Path("run/config.json")},
)
report.core.status  # "passed", "failed", or "unknown"
report.estimator.status  # Also permits "skipped" when explicitly disabled.
[(item.excitation, item.sample_count) for item in report.adaptive]
report.provenance.solver_revision  # Recorded GitTag, or None if unavailable.
report.provenance.input_check.status  # "unknown": no saved manifest was supplied.
report.warnings  # Suspicious but unproven, for example a jump of the port mode.
```

Only set `estimator_disabled=True` when the actual input disables the estimator. The parser recognizes the paired
zero-iteration PCG and unit-residual warning inside estimator sections or adaptive offline training. It keeps field
correction, eigen, and driven failures separate. An enabled estimator failure remains a failure of that check. A skipped
estimator supplies no evidence of mesh accuracy. gsim never writes `EstimatorMaxIts` by default, so on a config gsim
wrote, the lookup above is true only when you asked for 0. For a config that gsim did not write, read the resolved
`EstimatorMaxIts` from Palace's own `config_resolved.json` in the output directory.

A converged estimator prints nothing, so a normal run reports `estimator` as `unknown`; that is expected and is not a
problem. A non-zero `EstimatorMaxIts` that stops the estimator early is reported as a failure: of the `estimator` check
where Palace prints an estimator header (electrostatic, driven uniform sweeps, eigenmode, boundary mode), and of the
`core` check during adaptive offline training, which has no header.

Adaptive checks retain excitation IDs and complete training-frequency lists in GHz. Missing or truncated outcomes are
`unknown`; reaching the sample limit is `failed`. Successful adaptive sampling cannot clear an earlier core failure.
Explicit `expected_excitations` also exposes an entirely missing excitation. An outcome without an excitation header
retains `excitation=None`; it cannot satisfy an explicitly requested excitation ID.

## Check a selected mode

A selected mode number alone does not establish port identity. Supply reference values and tolerances from your own
application. The following self-contained example rejects opposite voltage parity even though the linear solver and
process exit succeeded:

```python
from gsim.palace.validation import ModeExpectation, ModeSample, validate_solution

report = validate_solution(
    "GMRES solver converged in 11 iterations",
    exit_code=0,
    mode_samples=[ModeSample(
        port=1, mode=2, wave_number=50 - 1j, voltage_ratio=-1,
    )],
    mode_expectations=[ModeExpectation(
        port=1, mode=2,
        wave_number=50 - 1j, wave_number_rtol=0.01,
        voltage_ratio=1, voltage_ratio_atol=0.02,
    )],
)
assert report.exit_status.status == report.core.status == "passed"
assert report.mode_checks[0].status == "failed"
assert report.estimator.status == "unknown"
```

Wave numbers are complex values in 1/m, including attenuation and its sign. Without explicit `mode_samples`,
observations come from driven wave-port log lines. Every matching sample, including repeated frequency/excitation
solves, must satisfy the expectation. `frequency_ghz` can restrict a reference to one frequency; its default relative
matching tolerance of `5e-4` accommodates the four-significant-digit log output. Physical comparison tolerances default
to zero and should be selected by the caller.

For BoundaryMode results, construct `ModeSample` records from the selected mode's tables. `voltage_ratio` is the complex
ratio of consistently oriented voltage integrals whose parity you want to check. The API does not infer that ratio from
a log or prescribe a CPW convention. A requested but absent ratio is `unknown`. Explicit samples replace log
observations; pass all observations you want checked.

## Check wave-port mode continuity

Palace selects the wave-port mode only by its index. If the port eigensolve returns a different solution at some
frequency (spurious solutions in exact sweeps on old Palace builds, gsim#375, or a closed-box mode with
`waveport_boundary="pec"`, gsim#408), the sweep still finishes and the S-parameters are silently wrong. A physical port
mode changes slowly with frequency, so `validate_solution` compares the effective index `n_eff = Re(k_n) / k0` of every
port solve in a driven log with the median of its five nearest neighbours in frequency. This runs automatically whenever
the log has wave-port solves with a frequency, and the result is in `report.port_mode_checks`:

```python
report = validate_solution(log_text)
for check in report.warnings:
    print(check.name, check.status)
    print(*check.evidence, sep="\n")
```

For a log with spurious solves the evidence names each flagged point (port, excitation, frequency, `n_eff`, neighbour
median):

```text
port_mode_continuity warning
3 of 124 port solves flagged.
port 1, mode 1, excitation 2: n_eff = 2.0087 at 1 GHz, neighbour median 1.7116 (17.36 % > 1 %)
port 2, mode 1, excitation 2: n_eff = 2.2669 at 1 GHz, neighbour median 1.7117 (32.44 % > 1 %)
port 2, mode 1, excitation 2: n_eff = 3.3796 at 1.3 GHz, neighbour median 1.7117 (97.44 % > 1 %)
```

- `passed`: every series with at least `MIN_DISTINCT_FREQUENCIES` (6) distinct frequencies stays within `CONTINUITY_RTOL`
  (1 %) of its neighbour median. Series that are too short are listed in the evidence as not judged.
- `warning`: at least one point is outside the tolerance, or a wave number is not finite. A warning is not a failure and
  is not part of `report.failures`: a mode that really changes quickly (close to cutoff, very coarse sweeps) is flagged
  too. Inspect the listed frequencies and the port fields before trusting or discarding the sweep.
- `unknown`: no series is long enough to judge, for example a single-frequency run. Unknown is never a pass.
- No check is added when the log has no frequency-tagged wave-port solve (eigenmode, electrostatic, lumped-port runs).

The check only sees the real part of the wave number. It cannot find a wrong mode that is smooth in frequency, for
example a sweep that is wrong at every frequency; compare against a BoundaryMode solve with `ModeExpectation` for that.

Ports are compared with each other only on request, because the log does not record the port geometry and ports of
different width or stack legitimately have different modes. When every wave port has the same cross-section, pass
`ports_share_cross_section=True`; ports solved at the same frequency must then agree within `PORT_AGREEMENT_RTOL`
(0.5 %), reported as the `port_mode_ports` check. To run the check on your own observations, call
`check_port_mode_continuity(samples, compare_ports=..., rtol=..., port_rtol=...)` with `ModeSample` records that carry
`frequency_ghz`.

## Compare meshes between runs

Gmsh gives the same mesh twice only for some settings. With the default `algorithm_3d="delaunay"` and
`surface_threads=1`, the mesh does not depend on `threads`. `surface_threads` above 1 is faster but gives a different
mesh on every run, and with `algorithm_3d="hxt"` the mesh depends on the thread count. This is a Gmsh limitation
(gmsh#2255): `Mesh.Reproducible` only covers HXT's initial Delaunay step, and a fixed random seed cannot fix the order in
which threads finish. gsim logs a warning for both non-reproducible settings.

`sim.mesh()` records an ID-independent mesh hash, the Gmsh options used and the Gmsh and gsim versions:

```python
result = sim.mesh(threads=4, surface_threads=1)
result.mesh_stats["mesh_hash"]
result.mesh_stats["versions"]["gmsh"]
result.mesh_stats["gmsh_options"]["Mesh.MaxNumThreads2D"]
```

Equal hashes mean the same nodes (to 1e-4 um) and the same elements in each physical group. Compare hashes only between
runs with the same Gmsh version on the same kind of host; the stats do not record the host. A different hash under
reproducible settings points at a change in the geometry, the mesh settings or the Gmsh build.

## Evidence and limits

Statuses describe supplied evidence, not a complete physical validation. In particular, a passed core check means at
least one recognized convergence message and no recognized core failure. It cannot establish complete log coverage, mesh
convergence, or convergence of unreported eigenpairs. Unrecognized log formats and absent metrics are not proof of
success. The log fixtures cover driven logs from archived `cfa430a` and `cfa430a-dirty` builds, and electrostatic,
eigenmode, boundary-mode and magnetostatic logs from Palace `v0.18.1`. Transient log coverage is not established. A
magnetostatic solve that did not converge still exits 0 and writes a plausible inductance matrix, so `load_inductance`
checks `palace.log` the same way and refuses a run whose core check failed; the cloud build's `Current source solve did
not converge for port N!` counts as a core failure. The standard driven `It N/M: Index = ...` header resets the
estimator section, but abridged or reformatted logs may omit that boundary. Later Palace formats may require additional
parsing rules.

`report.failures` contains explicit failures; an empty tuple can still leave unknown or omitted checks. Mode checks are
omitted unless requested; the automatic port-mode continuity checks are separate (`report.port_mode_checks`, listed in
`report.warnings` when they warn). Exit and OOM checks remain unknown unless independently recorded values are supplied.
No resource limits, passivity limits, attenuation limits, or application-specific mode thresholds are assumed beyond the
documented continuity tolerances.

`input_files` records current file hashes. To compare against independently saved run provenance, pass
`expected_input_sha256={"config.json": saved_digest}`. Only named manifest entries are compared; a mismatch fails and a
missing or unreadable file (including a directory) is unknown, with the OS error recorded as evidence. Other input
checks still complete. Current hashes alone cannot prove which files a remote solver used.
