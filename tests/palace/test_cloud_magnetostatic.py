"""Cloud MagnetostaticSim on two side-by-side IHP coils: inductance matrix smoke test.

End to end on GDSFactory+ cloud: mesh, config, the cloud's ``Magnetostatic``
run, download, and ``load_inductance`` with the convergence check on. The
geometry is the validation pair of ``nbs/palace_magnetostatic.ipynb`` on a
coarse Order-1 mesh, so the numbers are not accurate; they only have to stay
put.

Reference data lives next to this file at
``test_cloud_magnetostatic/test_magnetostatic_pair.npz``. On the first run the
file is generated automatically; use ``--force-regen`` to overwrite it.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from tests._cloud_fixtures import make_magnetostatic_pair_sim

pytestmark = [pytest.mark.cloud, pytest.mark.sim_smoke_test]

#: Coarse mesh: half the 2 um trace width at the metal, coarse air.
MESH = {"refined_mesh_size": 1.0, "max_mesh_size": 30.0}


def test_magnetostatic_pair(
    ndarrays_regression: NDArraysRegressionFixture,
    tmp_path,
) -> None:
    sim = make_magnetostatic_pair_sim(tmp_path / "palace-sim")
    sim.set_solver(order=1)
    sim.mesh(**MESH, verbose=False)
    results = sim.run(parent_dir=tmp_path, verbose="quiet")

    inductance = sim.load_inductance(results)
    assert inductance.sources == ("A", "B")
    assert inductance.problems() == []
    # Side by side, both coils wound the same way: the return flux of one
    # threads the other backwards.
    assert inductance.mutual_inductance("A", "B") < 0

    # The downloaded directory reads the same as the results dict.
    (sim_data,) = tmp_path.glob("sim-data-*")
    from_dir = sim.load_inductance(sim_data)
    for field in ("matrix", "inverse", "excitation_current", "stored_energy"):
        expected = getattr(inductance, field)
        assert np.array_equal(getattr(from_dir, field), expected), field

    # 2 % leaves room for drift of the cloud's Palace build and of Gmsh.
    ndarrays_regression.check(
        {"inductance_H": inductance.matrix},
        default_tolerance={"rtol": 2e-2, "atol": 0.0},
    )
