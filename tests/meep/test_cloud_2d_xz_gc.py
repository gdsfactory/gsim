"""Cloud 2D XZ grating-coupler MEEP test — compares S-params against reference.

Reference data lives next to this file at
``test_cloud_2d_xz_gc/test_2d_xz_gc.npz``. Run
``uv run pytest tests/meep/test_cloud_2d_xz_gc.py --force-regen`` to
regenerate.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from tests._cloud_fixtures import make_2d_xz_gc_sim, require_regression_reference

pytestmark = pytest.mark.cloud


def test_2d_xz_gc(
    ndarrays_regression: NDArraysRegressionFixture,
    tmp_path,
    request: pytest.FixtureRequest,
) -> None:
    require_regression_reference(request)
    sim = make_2d_xz_gc_sim()
    result = sim.run(parent_dir=tmp_path, verbose="quiet")

    arrays: dict[str, np.ndarray] = {
        "wavelengths": np.asarray(result.wavelengths, dtype=float),
    }
    for name, values in sorted(result.s_params.items()):
        arr = np.asarray(values, dtype=complex)
        arrays[f"{name}_real"] = arr.real
        arrays[f"{name}_imag"] = arr.imag

    ndarrays_regression.check(
        arrays,
        default_tolerance={"atol": 1e-6, "rtol": 1e-6},
    )
