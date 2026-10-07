# Validate Palace result evidence

`validate_solution` reports core solver convergence, optional error estimation, adaptive sampling, process exit and OOM
evidence, mode checks, and input provenance independently. A zero exit code does not establish physical validity. The
function reads supplied evidence and never changes tolerances or reruns a simulation.

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
```

Only set `estimator_disabled=True` when the actual input disables the estimator. The parser recognizes the paired
zero-iteration PCG and unit-residual warning inside estimator sections or adaptive offline training. It keeps field
correction, eigen, and driven failures separate. An enabled estimator failure remains a failure of that check. A skipped
estimator supplies no evidence of mesh accuracy.

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

## Evidence and limits

Statuses describe supplied evidence, not a complete physical validation. In particular, a passed core check means at
least one recognized convergence message and no recognized core failure. It cannot establish complete log coverage, mesh
convergence, or convergence of unreported eigenpairs. Unrecognized log formats and absent metrics are not proof of
success. Archived `cfa430a` and `cfa430a-dirty` coverage is limited to driven and eigenmode logs; electrostatic and
magnetostatic log coverage is not established. Their standard `It N/M: Index = ...` header resets the estimator section,
but abridged or reformatted logs may omit that boundary. Later Palace formats may require additional parsing rules.

`report.failures` contains explicit failures; an empty tuple can still leave unknown or omitted checks. Mode checks are
omitted unless requested. Exit and OOM checks remain unknown unless independently recorded values are supplied. No
resource limits, passivity limits, attenuation limits, or application-specific mode thresholds are assumed.

`input_files` records current file hashes. To compare against independently saved run provenance, pass
`expected_input_sha256={"config.json": saved_digest}`. Only named manifest entries are compared; a mismatch fails and a
missing or unreadable file (including a directory) is unknown, with the OS error recorded as evidence. Other input
checks still complete. Current hashes alone cannot prove which files a remote solver used.
