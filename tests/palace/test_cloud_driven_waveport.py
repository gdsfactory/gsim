"""Cloud DrivenSim with wave ports — compares S-params against reference.

Reference data lives next to this file at
``test_cloud_driven_waveport/test_driven_cpw_waveport.npz``. Run
``uv run pytest tests/palace/test_cloud_driven_waveport.py --force-regen``
to regenerate.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from tests._cloud_fixtures import (
    make_driven_cpw_waveport_sim,
    require_regression_reference,
)

pytestmark = pytest.mark.cloud


def test_driven_cpw_waveport(
    ndarrays_regression: NDArraysRegressionFixture,
    tmp_path,
    request: pytest.FixtureRequest,
) -> None:
    require_regression_reference(request)
    sim = make_driven_cpw_waveport_sim(tmp_path / "palace-sim")
    sim.mesh(
        refined_mesh_size=2.0,
        max_mesh_size=40.0,
        fmax=150e9,
        margin_x=0,
        margin_y=50.0,
    )
    sp = sim.run(parent_dir=tmp_path, verbose="quiet")

    arrays: dict[str, np.ndarray] = {"freq_ghz": np.asarray(sp.freq, dtype=float)}
    for to_port in sp.port_names:
        for from_port in sp.port_names:
            try:
                entry = sp[to_port, from_port]
            except KeyError:
                continue
            arr = entry.complex
            key = f"s_{to_port}_{from_port}"
            arrays[f"{key}_real"] = np.real(arr)
            arrays[f"{key}_imag"] = np.imag(arr)

    ndarrays_regression.check(
        arrays,
        default_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )
