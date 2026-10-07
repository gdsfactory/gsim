"""Regressions for independent Palace result evidence and mode checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gsim.palace.validation import ModeExpectation, ModeSample, validate_solution

FIXTURES = Path(__file__).parent / "fixtures" / "validation"
CORE_SUCCESS = "GMRES solver converged in 11 iterations"
ZERO_PAIR = (
    "PCG solver did NOT converge in 0 iterations\n"
    "\n\x1b[38;2;255;255;000m--> Warning!\x1b[0m\n"
    "Linear solver did not converge, norm(Ax-b)/norm(b) = 1.000e+00 "
    "(norm(b) = 6.313e-01)!\n"
)


def test_disabled_estimator_and_wrong_mode_are_independent():
    report = validate_solution(
        (FIXTURES / "disabled-estimator.txt").read_text(encoding="utf-8"),
        estimator_disabled=True,
        exit_code=0,
        metadata={"GitTag": "cfa430a"},
        mode_expectations=[
            ModeExpectation(
                port=1,
                mode=1,
                frequency_ghz=1,
                wave_number=50 - 1j,
                wave_number_rtol=0.05,
            )
        ],
    )
    assert report.core.status == report.exit_status.status == "passed"
    assert report.estimator.status == "skipped"
    assert report.mode_checks[0].status == "failed"
    assert report.mode_samples[0].wave_number == complex(157.3, -19980)
    assert report.provenance.solver_revision == "cfa430a"
    assert report.oom.status == report.provenance.input_check.status == "unknown"


def test_disabled_requires_explicit_evidence():
    log = (FIXTURES / "disabled-estimator.txt").read_text(encoding="utf-8")
    report = validate_solution(log)
    assert report.core.status == "passed"
    assert report.estimator.status == "failed"
    assert validate_solution(ZERO_PAIR, estimator_disabled=True).core.status == "failed"


def test_offline_estimator_pair_and_missing_excitation():
    log = (FIXTURES / "offline-disabled-estimator.txt").read_text(encoding="utf-8")
    report = validate_solution(
        log, estimator_disabled=True, expected_excitations=[1, 2]
    )
    assert report.core.status == "passed"
    assert report.estimator.status == "skipped"
    assert [item.check.status for item in report.adaptive] == ["passed", "unknown"]
    assert report.adaptive[0].sample_count == 9
    assert len(report.adaptive[0].frequencies_ghz) == 9


def test_adaptive_convergence_does_not_hide_core_failure():
    report = validate_solution(
        (FIXTURES / "adaptive-core-failure.txt").read_text(encoding="utf-8")
    )
    assert report.core.status == "failed"
    assert [item.excitation for item in report.adaptive] == [1, 2]
    assert [item.sample_count for item in report.adaptive] == [7, 9]
    assert all(item.check.status == "passed" for item in report.adaptive)
    assert [len(item.frequencies_ghz) for item in report.adaptive] == [7, 9]
    assert all(sample.excitation is None for sample in report.mode_samples)


@pytest.mark.parametrize(
    "failure",
    [
        "GMRES solver did NOT converge in 40 iterations",
        "Linear eigensolve failed to converge due to DIVERGED_ITS",
        "Correcting magnetic field\nPCG solver did NOT converge in 10 iterations",
        "Field correction\n" + ZERO_PAIR,
    ],
)
def test_core_failures_after_estimator_are_not_suppressed(failure):
    log = CORE_SUCCESS + "\nUpdating solution error estimates\n" + ZERO_PAIR + failure
    report = validate_solution(log, estimator_disabled=True)
    assert report.core.status == "failed"
    assert report.estimator.status == "skipped"


def test_nonunit_residual_and_incomplete_pair_are_not_skipped():
    for warning in (
        ZERO_PAIR.replace("1.000e+00", "2.000e+00"),
        ZERO_PAIR.splitlines()[0],
    ):
        report = validate_solution(
            "Beginning PROM construction offline phase:\n" + warning,
            estimator_disabled=True,
        )
        assert report.core.status == "failed"


def test_absent_metrics_and_exit_zero_do_not_pass():
    report = validate_solution("", exit_code=0, expected_excitations=[3])
    assert report.core.status == report.estimator.status == "unknown"
    assert report.adaptive[0].excitation == 3
    assert report.adaptive[0].check.status == "unknown"
    assert report.mode_checks == ()
    assert report.provenance.solver_revision is None
    assert report.failures == ()
    assert validate_solution("", exit_code=9, oom_killed=True).failures


def test_estimator_success_does_not_establish_core_convergence():
    report = validate_solution(
        "Updating solution error estimates\nPCG solver converged in 4 iterations"
    )
    assert report.estimator.status == "passed"
    assert report.core.status == "unknown"


def test_eigensolve_success():
    report = validate_solution(
        "Linear eigensolve converged (2 eigenpairs) due to CONVERGED_TOL; iterations 1"
    )
    assert report.core.status == "passed"


def test_adaptive_incomplete_and_maximum_samples():
    report = validate_solution(
        "Adding excitation index 7 (1/2):\n"
        "Adaptive sampling converged with 3 frequency samples:\n"
        "Sampled frequencies (GHz): 1, 2\nSample errors: inf, inf\n"
        "Adding excitation index 9 (2/2):\n"
        "Adaptive sampling reached maximum number of samples!\n",
        expected_excitations=[7, 9, 12],
    )
    assert [item.excitation for item in report.adaptive] == [7, 9, 12]
    assert [item.check.status for item in report.adaptive] == [
        "unknown",
        "failed",
        "unknown",
    ]


def test_repeated_modes_and_frequency_rewinds():
    log = (
        "Sweeping excitation index 1 (1/2):\n"
        "Calculating boundary modes at wave ports for \u03c9/2\u03c0 = 1e0 GHz\n"
        "Port 3, mode 2: k\u2099 = 40-0.5i\n"
        "Sweeping excitation index 2 (2/2):\n"
        "Calculating boundary modes at wave ports for \u03c9/2\u03c0 = 1e0 GHz\n"
        "Port 3, mode 2: k\u2099 = 40+9i\n"
    )
    report = validate_solution(
        log,
        mode_expectations=[
            ModeExpectation(
                port=3,
                mode=2,
                frequency_ghz=1,
                wave_number=40 - 0.5j,
                wave_number_atol=0.1,
            )
        ],
    )
    assert len(report.mode_samples) == 2
    assert [sample.excitation for sample in report.mode_samples] == [1, 2]
    assert report.mode_checks[0].status == "failed"


@pytest.mark.parametrize(
    ("ratio", "status"), [(1 + 0j, "passed"), (-1 + 0j, "failed"), (None, "unknown")]
)
def test_voltage_parity_is_optional_and_caller_defined(ratio, status):
    sample = ModeSample(port=1, mode=2, wave_number=50 - 1j, voltage_ratio=ratio)
    expectation = ModeExpectation(
        port=1, mode=2, voltage_ratio=1, voltage_ratio_atol=0.02
    )
    report = validate_solution(
        CORE_SUCCESS, mode_samples=[sample], mode_expectations=[expectation]
    )
    assert report.mode_checks[0].status == status
    wrong_mode = validate_solution(
        "", mode_samples=[replace(sample, mode=1)], mode_expectations=[expectation]
    )
    assert wrong_mode.mode_checks[0].status == "failed"


def test_missing_mode_or_nonfinite_evidence():
    expectation = ModeExpectation(port=1, mode=1)
    assert (
        validate_solution("", mode_expectations=[expectation]).mode_checks[0].status
        == "unknown"
    )
    report = validate_solution(
        "",
        mode_samples=[ModeSample(1, 1, complex(float("nan")))],
        mode_expectations=[expectation],
    )
    assert report.mode_checks[0].status == "failed"
    logged = validate_solution(
        "Port 1, mode 1: k\u2099 = nan+0i",
        mode_expectations=[expectation],
    )
    assert logged.mode_checks[0].status == "failed"
    with pytest.raises(ValueError, match="nonnegative"):
        validate_solution(
            "", mode_expectations=[replace(expectation, wave_number_atol=-1)]
        )


def test_input_provenance_requires_an_independent_manifest(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"Solver": {}}\n')
    initial = validate_solution("", input_files={"config.json": config})
    assert initial.provenance.input_check.status == "unknown"
    manifest = initial.provenance.input_sha256
    verified = validate_solution(
        "", input_files={"config.json": config}, expected_input_sha256=manifest
    )
    assert verified.provenance.input_check.status == "passed"
    config.write_text("{}\n")
    changed = validate_solution(
        "", input_files={"config.json": config}, expected_input_sha256=manifest
    )
    assert changed.provenance.input_check.status == "failed"
    absent = validate_solution("", expected_input_sha256=manifest)
    assert absent.provenance.input_check.status == "unknown"


def test_directory_input_preserves_other_checks(tmp_path):
    valid = tmp_path / "config.json"
    valid.write_text("{}")
    report = validate_solution(
        CORE_SUCCESS,
        input_files={"directory": tmp_path, "config": valid},
    )
    assert report.core.status == "passed"
    assert report.provenance.input_check.status == "unknown"
    assert set(report.provenance.input_sha256) == {"config"}
    assert "directory" in report.provenance.input_check.evidence[0]


def test_unreadable_input_is_unknown(tmp_path, monkeypatch):
    path = tmp_path / "unreadable.json"
    path.write_text("{}")

    def denied(*_args, **_kwargs):
        raise PermissionError("test access denied")

    monkeypatch.setattr(Path, "open", denied)
    report = validate_solution(CORE_SUCCESS, input_files={"config": path})
    assert report.core.status == "passed"
    assert report.provenance.input_check.status == "unknown"
    assert (
        "PermissionError: test access denied"
        in report.provenance.input_check.evidence[0]
    )


def test_crlf_log_preserves_evidence():
    log = (FIXTURES / "disabled-estimator.txt").read_text(encoding="utf-8")
    assert validate_solution(log.replace("\n", "\r\n")) == validate_solution(log)


def test_unidentified_adaptive_outcome_does_not_satisfy_expected_excitation():
    report = validate_solution(
        "Adaptive sampling converged with 2 frequency samples:\n"
        "Sampled frequencies (GHz): 1, 2\nSample errors: 0, 0\n",
        expected_excitations=[1],
    )
    assert [(check.excitation, check.check.status) for check in report.adaptive] == [
        (None, "passed"),
        (1, "unknown"),
    ]


def test_terminal_iteration_header_ends_estimator_section():
    # Header form from Palace ElectrostaticSolver::Solve; synthetic regression.
    report = validate_solution(
        CORE_SUCCESS + "\nUpdating solution error estimates\n"
        "PCG solver converged in 3 iterations\n"
        "It 2/2: Index = 2 (elapsed time = 1.00e+00 s)\n"
        "PCG solver did NOT converge in 20 iterations\n"
    )
    assert report.core.status == "failed"
    assert report.estimator.status == "passed"
