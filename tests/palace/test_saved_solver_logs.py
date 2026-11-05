"""Retain solver output on success and failure without requiring console output."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gsim.palace import ElectrostaticSim
from gsim.palace.runtime import PalaceLaunch, run_palace


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("returncode", [0, 7])
def test_saved_log_contains_both_streams(
    tmp_path: Path, verbose: bool, returncode: int
) -> None:
    script = tmp_path / "solver.py"
    script.write_text(
        "import sys\nprint('field solve', flush=True)\n"
        "print('estimator solve', file=sys.stderr, flush=True)\n"
        f"sys.exit({returncode})\n"
    )
    launch = PalaceLaunch((sys.executable, str(script)), dict(os.environ))
    target = tmp_path / "logs" / "solver.log"
    options = dict(cwd=tmp_path, num_processes=1, verbose=verbose, log_path=target)
    if returncode:
        with pytest.raises(subprocess.CalledProcessError) as error:
            run_palace(launch, **options)
        assert error.value.returncode == returncode
        assert "field solve" in error.value.stderr
    else:
        run_palace(launch, **options)
    assert target.read_text() == "field solve\nestimator solve\n"


def test_saved_log_preserves_utf8_solver_output(tmp_path: Path) -> None:
    script = tmp_path / "solver.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write('\\u03c9/2\\u03c0 k\\u2099\\n'.encode('utf-8'))\n"
        "sys.stdout.buffer.flush()\n"
    )
    launch = PalaceLaunch((sys.executable, str(script)), dict(os.environ))
    target = tmp_path / "solver.log"
    run_palace(launch, cwd=tmp_path, num_processes=1, verbose=False, log_path=target)

    assert target.read_bytes() == "\u03c9/2\u03c0 k\u2099\n".encode()


@pytest.mark.parametrize("initial_guess", [None, False, True])
def test_initial_guess_is_optional_and_reaches_the_config(
    initial_guess: bool | None,
) -> None:
    sim = ElectrostaticSim()
    sim.set_solver(initial_guess=initial_guess)
    linear = sim.solver.linear.to_palace_config()
    if initial_guess is None:
        assert "InitialGuess" not in linear
    else:
        assert linear["InitialGuess"] is initial_guess
