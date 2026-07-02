"""Unit tests for ModeResult.plot_mode() and ModeResult.plot_index()."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")
import matplotlib.pyplot as plt

from gsim.meep.models.results import ModeResult


def _make_1d_result(**overrides):
    z = np.linspace(-2, 2, 50)
    rng_e = np.random.default_rng(42)
    rng_x = np.random.default_rng(43)
    defaults: dict[str, object] = {
        "n_eff": 2.5,
        "wavelength": 1.55,
        "frequency": 1.0 / 1.55,
        "fields": {
            "Ey": rng_e.random(50) + 1j * rng_x.random(50),
            "Ex": np.linspace(0, 1, 50) + 0j,
        },
        "z_grid": z,
        "x_grid": None,
        "y_grid": None,
        "stack": None,
        "component": None,
        "port_or_position": None,
        "cross_section_plane": None,
        "kdom": [],
        "n_group": None,
        "band_num": 1,
        "parity": "NO_PARITY",
    }
    defaults.update(overrides)
    return ModeResult(**defaults)  # type: ignore[arg-type]


def _make_2d_result(**overrides):
    z = np.linspace(-2, 2, 30)
    y = np.linspace(-1, 1, 20)
    rng = np.random.default_rng(42)
    defaults: dict[str, object] = {
        "n_eff": 2.5,
        "wavelength": 1.55,
        "frequency": 1.0 / 1.55,
        "fields": {
            "Ey": rng.random((30, 20)) + 1j * rng.random((30, 20)),
            "Ex": np.linspace(0, 1, 30)[:, None] * np.ones(20) + 0j,
            "Ez": rng.random((30, 20)) + 1j * rng.random((30, 20)),
        },
        "z_grid": z,
        "y_grid": y,
        "x_grid": None,
        "stack": None,
        "component": None,
        "port_or_position": None,
        "cross_section_plane": "yz",
        "kdom": [],
        "n_group": None,
        "band_num": 1,
        "parity": "NO_PARITY",
    }
    defaults.update(overrides)
    return ModeResult(**defaults)  # type: ignore[arg-type]


def _make_2d_result(**overrides):
    z = np.linspace(-2, 2, 30)
    y = np.linspace(-1, 1, 20)
    rng = np.random.default_rng(42)
    fields = {
        "Ey": rng.random((30, 20)) + 1j * rng.random((30, 20)),
        "Ex": np.linspace(0, 1, 30)[:, None] * np.ones(20) + 0j,
        "Ez": rng.random((30, 20)) + 1j * rng.random((30, 20)),
    }
    kwargs = dict(
        n_eff=2.5,
        wavelength=1.55,
        frequency=1.0 / 1.55,
        fields=fields,
        z_grid=z,
        y_grid=y,
        cross_section_plane="yz",
    )
    kwargs.update(overrides)
    return ModeResult(**kwargs)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestModeResultPlotErrors:
    def test_plot_mode_empty_fields_raises(self):
        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
        )
        with pytest.raises(ValueError, match="no field data"):
            result.plot_mode()

    def test_plot_mode_invalid_norm_raises(self):
        result = _make_1d_result()
        with pytest.raises(ValueError, match="Unknown norm"):
            result.plot_mode(norm="invalid_norm")

    def test_plot_mode_unknown_component_raises(self):
        result = _make_1d_result()
        with pytest.raises(ValueError, match="Unknown component"):
            result.plot_mode(components="Hz")

    def test_plot_mode_ax_with_multi_component_raises(self):
        result = _make_1d_result()
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="ax may only be passed"):
            result.plot_mode(components="all", ax=ax)
        plt.close(fig)

    def test_plot_mode_both_x_and_y_grid_raises(self):
        result = _make_1d_result(
            x_grid=np.array([0.0]),
            y_grid=np.array([0.0]),
        )
        with pytest.raises(ValueError, match="Ambiguous geometry"):
            result.plot_mode()

    def test_plot_index_no_stack_raises(self):
        result = _make_1d_result()
        with pytest.raises(ValueError, match="No stack"):
            result.plot_index()

    def test_plot_index_no_z_grid_raises(self):
        from unittest.mock import MagicMock

        mock_stack = MagicMock()
        result = _make_1d_result(z_grid=None, stack=mock_stack)
        with pytest.raises(ValueError, match="No z_grid"):
            result.plot_index()


# ---------------------------------------------------------------------------
# Auto-component detection
# ---------------------------------------------------------------------------


class TestAutoComponent:
    def test_auto_component_picks_largest_amplitude(self):
        z = np.linspace(-2, 2, 50)
        fields = {
            "Ex": np.ones(50) * 0.1,
            "Ey": np.ones(50) * 5.0,
            "Ez": np.ones(50) * 0.05,
        }
        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
            fields=fields,
            z_grid=z,
        )
        assert result._auto_component() == "Ey"

    def test_auto_component_complex_magnitude(self):
        z = np.linspace(-2, 2, 50)
        fields = {
            "Ex": np.ones(50) * (1 + 1j),
            "Ey": np.ones(50) * (2 - 1j),
        }
        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
            fields=fields,
            z_grid=z,
        )
        assert result._auto_component() == "Ey"


# ---------------------------------------------------------------------------
# norm transforms
# ---------------------------------------------------------------------------


class TestNormTransforms:
    def test_abs_norm(self):
        field = np.array([1 + 2j, -3 - 4j])
        out = ModeResult._apply_norm(field, "abs")
        np.testing.assert_allclose(out, np.array([np.sqrt(5), 5.0]))

    def test_real_norm(self):
        field = np.array([1 + 2j, -3 - 4j])
        out = ModeResult._apply_norm(field, "real")
        np.testing.assert_allclose(out, np.array([1.0, -3.0]))

    def test_imag_norm(self):
        field = np.array([1 + 2j, -3 - 4j])
        out = ModeResult._apply_norm(field, "imag")
        np.testing.assert_allclose(out, np.array([2.0, -4.0]))

    def test_phase_norm(self):
        field = np.array([1 + 1j])
        out = ModeResult._apply_norm(field, "phase")
        np.testing.assert_allclose(out, np.array([np.pi / 4]))


# ---------------------------------------------------------------------------
# 1D plot_mode smoke tests
# ---------------------------------------------------------------------------


class TestPlotMode1D:
    def test_plot_mode_auto_1d(self):
        result = _make_1d_result()
        fig, ax = result.plot_mode(show=False)
        assert isinstance(fig, plt.Figure)
        assert ax is not None
        assert "n_eff=2.5000" in ax.get_title()
        plt.close(fig)

    def test_plot_mode_specific_component_1d(self):
        result = _make_1d_result()
        fig, ax = result.plot_mode(components="Ex", show=False)
        assert isinstance(fig, plt.Figure)
        assert "Ex" in ax.get_title()
        plt.close(fig)

    def test_plot_mode_all_1d(self):
        result = _make_1d_result()
        fig, axes = result.plot_mode(components="all", show=False)
        assert isinstance(fig, plt.Figure)
        assert axes.shape == (2, 1)
        plt.close(fig)

    def test_plot_mode_list_components_1d(self):
        result = _make_1d_result()
        fig, _ax = result.plot_mode(components=["Ey"], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_mode_with_ax_1d(self):
        result = _make_1d_result()
        fig, ax = plt.subplots()
        fig2, ax2 = result.plot_mode(components="Ey", ax=ax, show=False)
        assert fig is fig2
        assert ax is ax2
        plt.close(fig)

    def test_plot_mode_title_none_1d(self):
        result = _make_1d_result()
        fig, ax = result.plot_mode(title=None, show=False)
        assert ax.get_title() == ""
        plt.close(fig)

    def test_plot_mode_custom_title_1d(self):
        result = _make_1d_result()
        fig, ax = result.plot_mode(title="Custom Title", show=False)
        assert ax.get_title() == "Custom Title"
        plt.close(fig)


# ---------------------------------------------------------------------------
# 2D plot_mode smoke tests
# ---------------------------------------------------------------------------


class TestPlotMode2D:
    def test_plot_mode_auto_2d(self):
        result = _make_2d_result()
        fig, ax = result.plot_mode(show=False)
        assert isinstance(fig, plt.Figure)
        assert ax is not None
        assert "n_eff=2.5000" in ax.get_title()
        plt.close(fig)

    def test_plot_mode_all_2d(self):
        result = _make_2d_result()
        fig, axes = result.plot_mode(components="all", show=False)
        assert isinstance(fig, plt.Figure)
        assert axes.shape[0] * axes.shape[1] >= 3
        plt.close(fig)

    def test_plot_mode_shared_colorbar_2d(self):
        result = _make_2d_result()
        fig, _axes = result.plot_mode(
            components="all", shared_colorbar=True, show=False
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_mode_xz_grid_2d(self):
        x = np.linspace(-1, 1, 10)
        z = np.linspace(-2, 2, 15)
        rng = np.random.default_rng(42)
        fields = {"Ey": rng.random((15, 10)) + 1j * rng.random((15, 10))}
        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
            fields=fields,
            z_grid=z,
            x_grid=x,
            cross_section_plane="xz",
        )
        fig, ax = result.plot_mode(show=False)
        assert ax.get_xlabel() == "x (µm)"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_index smoke tests
# ---------------------------------------------------------------------------


class TestPlotIndex:
    def test_plot_index_1d_raises_without_stack(self):
        result = _make_1d_result()
        with pytest.raises(ValueError, match="No stack"):
            result.plot_index()

    def test_plot_index_2d_raises_without_stack(self):
        result = _make_2d_result()
        with pytest.raises(ValueError, match="No stack"):
            result.plot_index()
