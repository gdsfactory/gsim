"""``sim.plot_stack()`` and ``sim.show_stack()`` on gsim's own LayerStack."""

from __future__ import annotations

import gdsfactory as gf
import plotly.graph_objects as go
import pytest

from gsim import palace
from gsim.common.stack import get_stack


@pytest.fixture
def _gpdk_active():
    """Activate the generic PDK for the test and restore the previous PDK."""
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    gf.gpdk.PDK.activate()
    try:
        yield
    finally:
        if previous is not None:
            previous.activate()


@pytest.fixture
def sim(_gpdk_active) -> palace.DrivenSim:
    sim = palace.DrivenSim()
    sim.set_stack(get_stack())
    return sim


def test_plot_stack_returns_a_figure_with_a_shape_per_layer(sim) -> None:
    figure = sim.plot_stack()
    assert isinstance(figure, go.Figure)
    assert sim.stack is not None
    assert len(sim.stack.layers) > 0
    assert len(figure.layout.shapes) == len(sim.stack.layers)


def test_show_stack_prints_every_layer(sim, capsys) -> None:
    sim.show_stack()
    out = capsys.readouterr().out
    assert sim.stack is not None
    for name in sim.stack.layers:
        assert name in out
