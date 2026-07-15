"""Cloud S-bend MEEP test — compares S-params against stored reference.

Reference data lives next to this file at
``test_cloud_sbend/test_sbend.npz``.  On the first run the file is generated
automatically; use ``--force-regen`` to overwrite it.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from tests._cloud_fixtures import make_sbend_sim

pytestmark = [pytest.mark.cloud, pytest.mark.sim_smoke_test]


def test_sbend(
    ndarrays_regression: NDArraysRegressionFixture,
    tmp_path,
) -> None:
    sim = make_sbend_sim()
    result = sim.run(parent_dir=tmp_path, verbose="quiet")

    arrays: dict[str, np.ndarray] = {
        "wavelengths": np.asarray(result.wavelengths, dtype=float),
    }
    for name, values in sorted(result.s_params.items()):
        arr = np.asarray(values, dtype=complex)
        arrays[f"{name}_real"] = arr.real
        arrays[f"{name}_imag"] = arr.imag

    # S-params are bounded in [-1, 1]; compare with an absolute floor rather
    # than rtol (rtol explodes on components that pass through zero). Baseline
    # regenerated against current cloud solver output.
    ndarrays_regression.check(
        arrays,
        default_tolerance={"atol": 0.01, "rtol": 0.0},
    )
