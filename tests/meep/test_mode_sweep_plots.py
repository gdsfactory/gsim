"""Unit tests for ModeSweepResult.plot_mode() and plot_index() delegation."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")
import matplotlib.pyplot as plt

from gsim.meep.models.results import ModeResult
from gsim.meep.results import ModeSweepResult


def _make_result(n_eff=2.5, wavelength=1.55, band_num=1, **kwargs):
    z = np.linspace(-2, 2, 50)
    rng = np.random.default_rng(42)
    fields = {"Ey": rng.random(50) + 1j * rng.random(50)}
    return ModeResult(
        n_eff=n_eff,
        wavelength=wavelength,
        frequency=1.0 / wavelength,
        band_num=band_num,
        fields=fields,
        z_grid=z,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _find_result
# ---------------------------------------------------------------------------


class TestFindResult:
    def test_find_by_wavelength(self):
        r1 = _make_result(n_eff=2.5, wavelength=1.55)
        r2 = _make_result(n_eff=2.3, wavelength=1.60)
        sweep = ModeSweepResult([r1, r2])
        found = sweep._find_result(1.55)
        assert found is r1

    def test_find_by_wavelength_and_band(self):
        r1 = _make_result(n_eff=2.5, wavelength=1.55, band_num=1)
        r2 = _make_result(n_eff=2.3, wavelength=1.55, band_num=2)
        sweep = ModeSweepResult([r1, r2])
        found = sweep._find_result(1.55, band=2)
        assert found is r2

    def test_find_no_match_wavelength_raises(self):
        sweep = ModeSweepResult([_make_result(wavelength=1.55)])
        with pytest.raises(ValueError, match="No mode result found"):
            sweep._find_result(1.60)

    def test_find_no_match_band_raises(self):
        sweep = ModeSweepResult([_make_result(wavelength=1.55, band_num=1)])
        with pytest.raises(ValueError, match="No mode result found"):
            sweep._find_result(1.55, band=5)


# ---------------------------------------------------------------------------
# plot_mode delegation
# ---------------------------------------------------------------------------


class TestSweepPlotMode:
    def test_plot_mode_delegates(self):
        r1 = _make_result(n_eff=2.5, wavelength=1.55)
        sweep = ModeSweepResult([r1])
        fig, ax = sweep.plot_mode(wavelength=1.55, show=False)
        assert isinstance(fig, plt.Figure)
        assert ax is not None
        assert "n_eff=2.5000" in ax.get_title()
        plt.close(fig)

    def test_plot_mode_with_band(self):
        r1 = _make_result(n_eff=2.3, wavelength=1.55, band_num=2)
        sweep = ModeSweepResult([r1])
        fig, _ax = sweep.plot_mode(wavelength=1.55, band=2, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_mode_forwards_kwargs(self):
        r1 = _make_result(n_eff=2.5, wavelength=1.55)
        sweep = ModeSweepResult([r1])
        fig, _ax = sweep.plot_mode(
            wavelength=1.55,
            components="Ey",
            norm="real",
            show=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_index delegation
# ---------------------------------------------------------------------------


class TestSweepPlotIndex:
    def test_plot_index_raises_without_stack(self):
        r1 = _make_result(n_eff=2.5, wavelength=1.55)
        sweep = ModeSweepResult([r1])
        with pytest.raises(ValueError, match="No stack"):
            sweep.plot_index(wavelength=1.55)
