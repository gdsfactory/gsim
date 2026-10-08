"""Migration notices must reach users with Python's default warning filters."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

import gsim.palace as pa


@pytest.mark.parametrize("kwargs", [{}, {"tolerance": 1e-8}])
def test_omitted_legacy_order_warns_and_uses_two(kwargs):
    sim = pa.DrivenSim()
    with pytest.warns(FutureWarning, match="order=2.*order=1.*Pass order=1"):
        sim.set_numerical(**kwargs)
    assert sim.solver.order == 2
    assert sim.solver.linear.tolerance == kwargs.get("tolerance", 1e-6)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_explicit_order_only_emits_api_deprecation(order):
    sim = pa.DrivenSim()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        sim.set_numerical(order=order)
    assert sim.solver.order == order
    assert len(notices) == 1
    assert notices[0].category is DeprecationWarning


def test_explicit_none_is_not_treated_as_an_omitted_order():
    with pytest.raises(ValidationError):
        pa.DrivenSim().set_numerical(order=None)


@pytest.mark.parametrize(
    "statement",
    [
        "pa.DrivenSim(numerical={'order': 1})",
        "pa.EigenmodeSim(eigenmode={'target': 4e9})",
        "sim.numerical = {'order': 1}",
        "sim.eigenmode = {'target': 4e9}",
        "sim.solver.linear.tolerance = 1e-8; sim.solver.tolerance = 1e-7",
    ],
)
def test_warnings_point_to_user_statement(statement):
    sim = pa.EigenmodeSim()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        exec(statement, {"pa": pa, "sim": sim})  # noqa: S102
    assert notices
    assert all(notice.filename == "<string>" for notice in notices)
    assert all(notice.lineno == 1 for notice in notices)


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("pa.DrivenSim().set_numerical()", "FutureWarning"),
        ("pa.DrivenSim().set_numerical(tolerance=1e-8)", "FutureWarning"),
        ("pa.DrivenSim(numerical={'order': 1})", "DeprecationWarning"),
        ("pa.EigenmodeSim(eigenmode={'target': 4e9})", "DeprecationWarning"),
        ("sim.numerical = {'order': 1}", "DeprecationWarning"),
        ("sim.eigenmode = {'target': 4e9}", "DeprecationWarning"),
        ("sim.set_solver(); sim.solver.linear.tolerance = 1e-8", None),
    ],
)
def test_notices_visible_under_default_script_filters(tmp_path, statement, expected):
    script = tmp_path / "user_simulation.py"
    script.write_text(
        "import gsim.palace as pa\nsim = pa.EigenmodeSim()\n" + statement + "\n"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONWARNINGS", None)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    if expected is None:
        assert "DeprecationWarning:" not in result.stderr
        assert "FutureWarning:" not in result.stderr
    else:
        assert expected in result.stderr
        assert f"{script}:3:" in result.stderr
        assert "pydantic/main.py" not in result.stderr
        if expected == "FutureWarning":
            assert "order=2" in result.stderr
            assert "order=1" in result.stderr
