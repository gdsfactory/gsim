"""Cloud slab eigenmode solver test — uploads mode solver config and
parses the returned ``ModeSweepResult``.

Does NOT require a GDS file — the 1D slab cell is built purely from
the layer stack and resolved material data.
"""

from __future__ import annotations

import pytest
from gdsfactory import gpdk

from gsim.meep import ModeSweepResult, Simulation

pytestmark = [pytest.mark.cloud, pytest.mark.sim_smoke_test]


def test_slab_mode_cloud(tmp_path) -> None:
    gpdk.PDK.activate()

    sim = Simulation()
    sim.materials = {"si": 12.0, "SiO2": 2.1}
    sim.source.port = "o1"
    sim.source.wavelength = 1.55
    sim.source.wavelength_span = 0.05
    sim.num_freqs = 1
    sim.monitors = []
    sim.domain.pml = 1.0
    sim.domain.margin_z = 0.5
    sim.solver.resolution = 10
    sim.solver.stop_when_energy_decayed(dt=15.0, decay_by=0.05)

    sim.mode_solver.where = "slab"
    sim.mode_solver.wavelengths = [1.55]
    sim.mode_solver.num_bands = 2
    sim.mode_solver.n_field_z = 80

    config_dir = tmp_path / "config"
    sim.write_mode_solver_config(config_dir)

    from gsim import gcloud

    job_id = gcloud.upload(config_dir, "meep", verbose=False)
    gcloud.start(job_id, verbose=False)
    result = gcloud.wait_for_results(job_id, verbose="quiet", parent_dir=tmp_path)

    assert isinstance(result, ModeSweepResult), f"Got {type(result)}"
    assert len(result.results) > 0, "Expected at least one ModeResult"
    for r in result.results:
        assert r.n_eff > 0, f"n_eff={r.n_eff} is not positive"
        assert r.fields, f"Band {r.band_num} at {r.wavelength} um has no fields"
