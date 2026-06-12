"""Tests for interactive (Plotly) 2D visualization."""

from __future__ import annotations


def _xz_sim_for_viz():
    """Build a minimal XZ simulation suitable for interactive plot tests."""
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
    sim.materials = {"si": 12.0, "SiO2": 2.1}
    sim.solver.is_3d = False
    sim.solver.plane = "xz"
    sim.source_fiber(x=0.0, z=1.22, waist=5.4)
    return sim


class TestPlot2DInteractive:
    """Smoke tests for plot_2d_interactive."""

    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go

        sim = _xz_sim_for_viz()
        fig = sim.plot_2d_interactive()
        assert isinstance(fig, go.Figure)

    def test_has_traces(self):
        sim = _xz_sim_for_viz()
        fig = sim.plot_2d_interactive()
        assert len(fig.data) > 0

    def test_z_slice(self):
        import plotly.graph_objects as go

        sim = _xz_sim_for_viz()
        fig = sim.plot_2d_interactive(slices="z", z="core")
        assert isinstance(fig, go.Figure)
        assert "XY cross section" in fig.layout.title.text

    def test_multi_slice_raises(self):
        import pytest

        sim = _xz_sim_for_viz()
        with pytest.raises(ValueError, match="exactly one slice"):
            sim.plot_2d_interactive(slices="xy")
