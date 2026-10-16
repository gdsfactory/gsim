"""Check geometric slices and overlays without opening a display."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely.geometry import Polygon

from gsim.common.geometry_model import GeometryModel, Prism
from gsim.common.viz.render2d import _plot_single_prism_slice, plot_prism_slices
from gsim.common.viz.render2d_interactive import plot_prism_slices_interactive
from gsim.meep.overlay import DielectricOverlay, PortOverlay, SimOverlay


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("empty", [False, True])
def test_slice_overlays_and_disconnected_intersections(monkeypatch, axis, empty):
    shape = Polygon([(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)])
    prism = Prism(np.array(shape.exterior.coords[:-1]), 0, 1, original_polygon=shape)
    model = GeometryModel({"core": [] if empty else [prism]}, ((0, 0, 0), (3, 3, 1)))
    overlay = SimOverlay(
        (-1, -1, -1),
        (4, 4, 2),
        0.25,
        ports=[
            PortOverlay(
                f"p{normal}{source}{offset}",
                (1.5 + offset, 1.5, 0.5),
                1,
                normal,
                "-",
                source,
                0.5,
            )
            for normal in (0, 1)
            for source in (False, True)
            for offset in (0, 10)
        ],
        dielectrics=[
            DielectricOverlay("oxide", "SiO2", -1, 2),
            DielectricOverlay("empty", "air", 1, 1),
        ],
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    _figure, ax = plt.subplots()
    try:
        plot_prism_slices(
            model,
            slices=axis,
            ax=ax,
            x=1.5,
            y=1.5,
            z=0.5,
            overlay=overlay,
            legend=False,
            aspect="auto",
        )
        assert ax.get_xlim() == pytest.approx((-1, 4))
        assert len(ax.patches) >= 5
        interactive = plot_prism_slices_interactive(
            model, **{axis: 0.5 if axis == "z" else 1.5}, overlay=overlay
        )
        assert "Sim cell" in [trace.name for trace in interactive.data]
        plot_prism_slices(model, slices=axis, overlay=overlay, legend=False)
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    ("options", "message"),
    [({"aspect": "invalid"}, "aspect"), ({"slices": "xy"}, "multiple slices")],
)
def test_invalid_slice_options(options, message):
    _figure, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match=message):
            plot_prism_slices(
                GeometryModel({}, ((0, 0, 0), (1, 1, 1))), ax=ax, **options
            )
        with pytest.raises(ValueError, match="exactly one"):
            _plot_single_prism_slice(
                GeometryModel({}, ((0, 0, 0), (1, 1, 1))), x=None, y=None, z=None
            )
    finally:
        plt.close("all")
