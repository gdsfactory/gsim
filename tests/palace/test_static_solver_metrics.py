"""Separate static field convergence from auxiliary error-estimator solves."""

import pytest

from gsim.palace.results import load_static_solver_metrics


def static_log(index=1, *, initial="3e2", final="1.5e-7", converged=True):
    outcome = (
        "CG solver converged in 25 iterations"
        if converged
        else "CG solver did NOT converge in 25 iterations"
    )
    return (
        f"It {index}/2: Index = {index}, V = 1.0e+00 V\n"
        f"  0 KSP residual norm ||r||_B = {initial}\n"
        f"  25 KSP residual norm ||r||_B = {final}\n"
        f"{outcome}\nUpdating solution error estimates\n"
        "  0 KSP residual norm ||r||_B = 1.0e+00\n"
        "  3 KSP residual norm ||r||_B = 1.0e-04\n"
        "PCG solver converged in 3 iterations\n"
    )


def test_metrics_preserve_indices_and_exclude_estimator():
    log = static_log() + static_log(2, final="2.4e-7") + static_log()
    metrics = load_static_solver_metrics("\x1b[31m" + log + "\x1b[0m")
    assert [item["index"] for item in metrics] == [1, 2, 1]
    assert [item["iterations"] for item in metrics] == [25, 25, 25]
    assert metrics[0]["initial_residual"] == 300
    assert metrics[0]["final_residual"] == 1.5e-7
    assert metrics[1]["relative_residual"] == pytest.approx(8e-10)
    assert load_static_solver_metrics("GMRES solver converged in 1 iteration") == []


@pytest.mark.parametrize(
    "log",
    [
        static_log(converged=False),
        static_log(final="nan"),
        static_log(initial="inf"),
        static_log(initial="0"),
        static_log(final="-1"),
        static_log().replace("  25 KSP residual norm ||r||_B = 1.5e-7\n", ""),
        static_log().replace("CG solver converged in 25 iterations", ""),
        static_log().replace(
            "Updating solution error estimates", "GMRES solver converged in 1 iteration"
        ),
        static_log().replace(
            "CG solver converged in 25 iterations",
            "CG solver did NOT converge\nCG solver converged in 25 iterations",
        ),
    ],
)
def test_incomplete_or_invalid_indexed_solve_is_rejected(log):
    with pytest.raises(ValueError, match="static solver convergence"):
        load_static_solver_metrics(log)


def test_overflowed_ratio_is_rejected():
    with pytest.raises(ValueError, match="Nonfinite residual ratio"):
        load_static_solver_metrics(static_log(initial="1e-320", final="1"))


def test_intermediate_samples_do_not_stand_in_for_the_convergence_iteration():
    log = static_log().replace(
        "  25 KSP residual norm ||r||_B = 1.5e-7\n",
        "  1 KSP residual norm ||r||_B = 2.0e2\n",
    )
    with pytest.raises(ValueError, match="static solver convergence"):
        load_static_solver_metrics(log)


def test_first_sample_must_be_iteration_zero():
    log = static_log().replace(
        "  0 KSP residual norm ||r||_B = 3e2\n",
        "  4 KSP residual norm ||r||_B = 3e2\n",
    )
    with pytest.raises(ValueError, match="static solver convergence"):
        load_static_solver_metrics(log)


def test_gmres_monitor_lines_without_the_norm_label_are_parsed():
    log = (
        "It 1/1: Index = 1 (elapsed time = 1.00e+00 s)\n"
        "Residual norms for GMRES solve\n"
        "  0 (restart 0) KSP residual norm 4.489804e+02\n"
        "  9 (restart 0) KSP residual norm 1.870905e-06\n"
        "GMRES solver converged in 9 iterations (avg. reduction factor: 1.8e-01)\n"
        "Updating solution error estimates\n"
        "  0 KSP residual norm ||r||_B = 1.0e+00\n"
        "PCG solver converged in 1 iteration\n"
    )
    metrics = load_static_solver_metrics(log)

    assert metrics[0]["iterations"] == 9
    assert metrics[0]["initial_residual"] == pytest.approx(4.489804e2)
    assert metrics[0]["relative_residual"] == pytest.approx(1.870905e-06 / 4.489804e2)


def test_magnetostatic_excitations_are_reported_as_unsupported():
    log = (
        "It 1/1: Current Index = 1 (elapsed time = 1.00e+00 s)\n"
        "  0 KSP residual norm ||r||_B = 3.0e+02\n"
        "  5 KSP residual norm ||r||_B = 1.0e-08\n"
        "PCG solver converged in 5 iterations\n"
    )
    with pytest.raises(ValueError, match="Magnetostatic"):
        load_static_solver_metrics(log)
