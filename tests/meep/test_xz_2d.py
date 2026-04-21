"""Integration test for XZ 2D grating coupler simulation."""

from __future__ import annotations

import os

import pytest


def _build_minimal_gc_sim():
    """Grating-coupler-ish stub: straight + 5 teeth on the core layer."""
    import gdsfactory as gf

    from gsim.common.stack import Layer, LayerStack
    from gsim.meep.simulation import Simulation

    c = gf.Component()
    # Feed waveguide from x=-8..0 at y=0, width 0.5
    c.add_polygon(
        [(-8.0, -0.25), (0.0, -0.25), (0.0, 0.25), (-8.0, 0.25)],
        layer=(1, 0),
    )
    # Grating teeth
    pitch = 0.62
    tooth_w = 0.3
    for i in range(5):
        x0 = i * pitch
        c.add_polygon(
            [(x0, -0.25), (x0 + tooth_w, -0.25), (x0 + tooth_w, 0.25), (x0, 0.25)],
            layer=(1, 0),
        )
    # Waveguide port at the back of the straight.
    c.add_port(
        name="o1",
        center=(-8.0, 0.0),
        orientation=180.0,
        width=0.5,
        layer=(1, 0),
    )

    stack = LayerStack(
        pdk_name="test",
        units="um",
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=0.0,
                zmax=0.22,
                thickness=0.22,
                material="si",
                layer_type="dielectric",
            ),
        },
        materials={},
        dielectrics=[
            {"name": "box", "zmin": -2.0, "zmax": 0.0, "material": "SiO2"},
            {"name": "clad", "zmin": 0.22, "zmax": 1.0, "material": "SiO2"},
        ],
        simulation={},
    )

    sim = Simulation()
    sim.geometry.component = c
    sim.geometry.stack = stack
    sim.materials = {"si": 3.47, "SiO2": 1.44}
    sim.solver.is_3d = False
    sim.solver.plane = "xz"
    sim.solver.resolution = 15
    sim.solver.stop_when_energy_decayed()
    sim.source_fiber(
        x=1.2,
        z_offset=1.0,
        angle_deg=14.5,
        waist=5.4,
        wavelength=1.55,
        wavelength_span=0.04,
        num_freqs=5,
    )
    sim.monitors = ["o1"]
    sim.domain.pml = 1.0
    sim.domain.margin = 0.5
    return sim


def test_xz_build_config_produces_expected_shape():
    sim = _build_minimal_gc_sim()
    result = sim.build_config()

    cfg = result.config
    assert cfg.plane == "xz"
    assert cfg.is_3d is False
    assert cfg.fiber_source is not None
    assert cfg.y_cut == pytest.approx(0.0, abs=1e-6)

    # Feed waveguide port should survive the filter.
    assert any(p.name == "o1" for p in cfg.ports)


@pytest.mark.skipif(
    os.environ.get("GSIM_RUN_CLOUD_TESTS") != "1",
    reason="Cloud integration test (set GSIM_RUN_CLOUD_TESTS=1 to enable)",
)
def test_xz_end_to_end_cloud():
    import numpy as np

    sim = _build_minimal_gc_sim()
    result = sim.run()

    assert result.s_params is not None
    s = np.asarray(result.s_params.get("o1@fiber") or result.s_params.get("o1"))
    assert s.size == 5
    assert np.all(np.isfinite(s))
    assert (np.abs(s) ** 2 <= 1.01).all()
