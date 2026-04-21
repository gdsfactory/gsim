"""Visualization tests for XZ 2D preview (``plot_2d(slices='y')``)."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt


def _xz_sim_for_viz():
    """Build a minimal XZ simulation suitable for plot_2d tests."""
    import gdsfactory as gf

    from gsim.common.stack import Layer, LayerStack
    from gsim.meep.simulation import Simulation

    c = gf.Component()
    c.add_polygon(
        [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
        layer=(1, 0),
    )
    c.add_port(
        name="o1",
        center=(5.0, 0.0),
        orientation=0.0,
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
    sim.source_fiber(x=0.0, z_offset=1.0, waist=5.4)
    return sim


class TestPlot2DXZ:
    """Smoke tests that plot_2d(slices='y') works for XZ sims."""

    def test_slices_y_returns_axes(self):
        sim = _xz_sim_for_viz()
        fig, ax = plt.subplots()
        result = sim.plot_2d(slices="y", ax=ax)
        assert result is ax
        plt.close(fig)

    def test_default_slice_when_plane_xz(self):
        sim = _xz_sim_for_viz()
        fig, ax = plt.subplots()
        # Default: plane='xz' → slices='y'.
        result = sim.plot_2d(ax=ax)
        assert result is ax
        plt.close(fig)
