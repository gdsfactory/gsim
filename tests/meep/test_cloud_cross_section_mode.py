"""Cloud cross-section eigenmode solver test — uploads mode solver config
with pre-computed cross-section geometry and parses ``ModeSweepResult``.

Requires a GDS component with a port and layer stack.  The geometry is
pre-computed client-side so the cloud runner only needs MEEP + numpy.
"""

from __future__ import annotations

import pytest
from gdsfactory import gpdk

from gsim.meep import ModeSweepResult, Simulation

pytestmark = [pytest.mark.cloud, pytest.mark.sim_smoke_test]


@pytest.fixture
def cross_section_component():
    """A simple straight waveguide component with a port and layer stack."""
    import gdsfactory as gf

    from gsim.common.stack import Layer, LayerStack

    gpdk.PDK.activate()

    c = gf.components.straight(length=10, width=0.5)

    si_zmin = 0.0
    si_zmax = 0.22

    stack = LayerStack(
        pdk_name="generic",
        units="um",
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=si_zmin,
                zmax=si_zmax,
                thickness=si_zmax - si_zmin,
                material="si",
                layer_type="dielectric",
            ),
        },
        materials={},
        dielectrics=[
            {"name": "box", "zmin": -2.0, "zmax": 0.0, "material": "SiO2"},
            {"name": "clad", "zmin": si_zmax, "zmax": 2.0, "material": "SiO2"},
        ],
        simulation={},
    )
    return c, stack


def test_cross_section_mode_cloud(tmp_path, cross_section_component) -> None:
    gpdk.PDK.activate()

    component, stack = cross_section_component

    sim = Simulation()
    sim.geometry.component = component
    sim.geometry.stack = stack
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

    sim.mode_solver.where = "cross_section"
    sim.mode_solver.wavelengths = [1.55]
    sim.mode_solver.num_bands = 2
    sim.mode_solver.port = "o1"
    sim.mode_solver.y_span = 3.0
    sim.mode_solver.n_field_z = 80
    sim.mode_solver.n_field_y = 50

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
        assert r.cross_section_plane == "yz", (
            f"Expected YZ plane, got {r.cross_section_plane}"
        )
