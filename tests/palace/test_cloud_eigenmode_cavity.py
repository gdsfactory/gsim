"""Cloud EigenmodeSim on a short CPW resonator — compares eigenfrequencies
against reference.

Eigenmode sims don't return S-params; the cloud ``run()`` produces a
``dict[str, Path]`` of output files. This test parses the Palace eigenvalue
CSV and regression-checks the real and imaginary parts of each mode's
complex frequency.

Reference data lives next to this file at
``test_cloud_eigenmode_cavity/test_eigenmode_cavity.npz``. Run
``uv run pytest tests/palace/test_cloud_eigenmode_cavity.py --force-regen``
to regenerate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from tests._cloud_fixtures import (
    make_eigenmode_cavity_sim,
    require_regression_reference,
)

pytestmark = pytest.mark.cloud


def _find_eig_csv(result_files: dict[str, Path]) -> Path:
    """Return the first eigenvalue CSV found in the cloud result files."""
    matches = [
        Path(p)
        for name, p in result_files.items()
        if name.endswith(".csv") and "eig" in name.lower()
    ]
    if not matches:
        msg = f"No eigenvalue CSV in result files: {list(result_files)}"
        raise FileNotFoundError(msg)
    return matches[0]


def _parse_eigenvalues(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (real_ghz, imag_ghz) arrays parsed from Palace's eig CSV."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    re_col = next(
        (c for c in df.columns if c.lower().startswith("re") or "re{f" in c.lower()),
        df.columns[1],
    )
    im_col = next(
        (c for c in df.columns if c.lower().startswith("im") or "im{f" in c.lower()),
        df.columns[2] if len(df.columns) > 2 else None,
    )
    real = df[re_col].to_numpy(dtype=float)
    imag = df[im_col].to_numpy(dtype=float) if im_col else np.zeros_like(real)
    return real, imag


def test_eigenmode_cavity(
    ndarrays_regression: NDArraysRegressionFixture,
    tmp_path,
    request: pytest.FixtureRequest,
) -> None:
    require_regression_reference(request)
    sim = make_eigenmode_cavity_sim(tmp_path / "palace-sim")
    sim.mesh(preset="default", margin=0)
    result = sim.run(parent_dir=tmp_path, verbose="quiet")

    csv_path = _find_eig_csv(result)
    freq_real, freq_imag = _parse_eigenvalues(csv_path)

    ndarrays_regression.check(
        {"freq_real_ghz": freq_real, "freq_imag_ghz": freq_imag},
        default_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )
