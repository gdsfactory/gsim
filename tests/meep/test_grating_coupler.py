"""Tests for grating coupler simulation configuration and spectrum validation.

Config tests verify the XZ 2D setup: vertical port detection, z-normal
monitors, PML overlap warnings, and port overrides.

The spectrum test validates that a grating coupler simulation produces
a Gaussian-like response with ~5 dB loss and ~40 nm bandwidth, matching
expected photonic design specifications.
"""

from __future__ import annotations

import warnings

import gdsfactory as gf
import numpy as np
import pytest

from gsim.meep import Simulation

gf.gpdk.PDK.activate()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gc_component():
    """Elliptical grating coupler from generic PDK."""
    return gf.components.grating_coupler_elliptical()


@pytest.fixture
def gc_sim(gc_component):
    """Configured grating coupler simulation in XZ 2D mode."""
    sim = Simulation()
    sim.geometry(component=gc_component, z_crop="auto")
    sim.materials = {"si": 3.47, "SiO2": 1.44}
    sim.source(port="o1", wavelength=1.55, wavelength_span=0.04, num_freqs=51)
    sim.monitors = ["o1", "o2"]
    sim.port_overrides = {"o2": {"width": 20}}
    sim.domain(pml=1.0, margin=0.5)
    sim.solver(resolution=20, simulation_plane="xz")
    sim.solver.stop_when_energy_decayed()
    return sim


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestGratingCouplerConfig:
    """Test grating coupler XZ 2D configuration."""

    def test_build_config_succeeds(self, gc_sim):
        result = gc_sim.build_config()
        assert result.config is not None
        assert result.config.simulation_plane == "xz"

    def test_vertical_port_detected(self, gc_sim):
        """o2 (vertical_te) should become a z-normal port."""
        result = gc_sim.build_config()
        ports = {p.name: p for p in result.config.ports}
        assert ports["o2"].normal_axis == 2
        assert ports["o2"].direction == "+"

    def test_waveguide_port_is_x_normal(self, gc_sim):
        """o1 (optical) should remain an x-normal port."""
        result = gc_sim.build_config()
        ports = {p.name: p for p in result.config.ports}
        assert ports["o1"].normal_axis == 0

    def test_port_override_applied(self, gc_sim):
        """width_override should be set on o2."""
        result = gc_sim.build_config()
        ports = {p.name: p for p in result.config.ports}
        assert ports["o2"].width_override == 20

    def test_z_crop_applied(self, gc_sim):
        """Stack should be z-cropped (meaningful in XZ mode)."""
        result = gc_sim.build_config()
        z_vals = [ls.zmin for ls in result.config.layer_stack] + [
            ls.zmax for ls in result.config.layer_stack
        ]
        # After z-crop, z-range should be reasonable (not full PDK stack)
        assert max(z_vals) - min(z_vals) < 5.0

    def test_no_pml_overlap_default(self, gc_sim):
        """Default config should not warn about PML overlap."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gc_sim.build_config()
            pml_warnings = [x for x in w if "overlap" in str(x.message).lower()]
            assert len(pml_warnings) == 0, (
                f"Unexpected PML overlap: {[str(x.message) for x in pml_warnings]}"
            )

    def test_pml_overlap_warning_when_forced(self, gc_component):
        """Forcing offset into PML should trigger a warning."""
        sim = Simulation()
        sim.geometry(component=gc_component, z_crop="auto")
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source(port="o1", wavelength=1.55, wavelength_span=0.04, num_freqs=51)
        sim.monitors = ["o1", "o2"]
        # Force monitor into PML via offset override
        sim.port_overrides = {"o2": {"offset": 1.0}}
        sim.domain(pml=1.0, margin=0.5)
        sim.solver(resolution=20, simulation_plane="xz")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim.build_config()
            pml_warnings = [x for x in w if "overlap" in str(x.message).lower()]
            assert len(pml_warnings) > 0, "Expected PML overlap warning for o2"

    def test_interactive_plot_returns_figure(self, gc_sim):
        """plot_2d_interactive should return a plotly Figure."""
        fig = gc_sim.plot_2d_interactive()
        assert hasattr(fig, "data")
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# Spectrum validation (reference data)
# ---------------------------------------------------------------------------


def _gaussian_db(wavelength, peak_wl, peak_db, bandwidth_nm):
    """Gaussian in dB scale for spectrum fitting."""
    sigma = bandwidth_nm * 1e-3 / (2 * np.sqrt(2 * np.log(2)))
    return peak_db * np.exp(-((wavelength - peak_wl) ** 2) / (2 * sigma**2))


def _fit_gaussian_spectrum(wavelengths, s21_db):
    """Fit a Gaussian to the S21 spectrum, return (peak_wl, peak_db, fwhm_nm)."""
    peak_idx = np.argmax(s21_db)
    peak_wl = wavelengths[peak_idx]
    peak_db = s21_db[peak_idx]

    # Find FWHM: half-maximum in dB is peak_db / 2 (since dB is already log-scale)
    half_max = peak_db / 2
    above_half = s21_db >= half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        wl_range = wavelengths[indices[-1]] - wavelengths[indices[0]]
        fwhm_nm = wl_range * 1e3
    else:
        fwhm_nm = 0.0

    return peak_wl, peak_db, fwhm_nm


@pytest.mark.slow
class TestGratingCouplerSpectrum:
    """Validate grating coupler spectrum against expected performance.

    These tests require a cloud simulation run. Mark with ``@pytest.mark.slow``
    and skip in CI. Run locally with ``pytest -m slow``.

    Expected: ~5 dB peak coupling loss, ~40 nm 3dB bandwidth, Gaussian shape.
    """

    @pytest.fixture
    def gc_result(self, gc_sim):
        """Run grating coupler simulation on cloud."""
        return gc_sim.run()

    def test_peak_loss_within_range(self, gc_result):
        """Peak |S21|^2 should be between -3 dB and -8 dB."""
        sp = gc_result.sparameters
        s21_db = 20 * np.log10(np.abs(sp["S21"]))
        peak_db = np.max(s21_db)
        assert -8 < peak_db < -3, f"Peak loss {peak_db:.1f} dB outside [-8, -3] range"

    def test_bandwidth_within_range(self, gc_result):
        """3dB bandwidth should be between 20 nm and 60 nm."""
        sp = gc_result.sparameters
        wavelengths = sp["wavelength"]
        s21_db = 20 * np.log10(np.abs(sp["S21"]))
        _, _, fwhm_nm = _fit_gaussian_spectrum(wavelengths, s21_db)
        assert 20 < fwhm_nm < 60, f"Bandwidth {fwhm_nm:.0f} nm outside [20, 60] range"

    def test_gaussian_shape(self, gc_result):
        """Spectrum should be approximately Gaussian (R^2 > 0.8)."""
        sp = gc_result.sparameters
        wavelengths = sp["wavelength"]
        s21_db = 20 * np.log10(np.abs(sp["S21"]))
        peak_wl, peak_db, fwhm_nm = _fit_gaussian_spectrum(wavelengths, s21_db)

        # Generate Gaussian fit and compute R^2
        fitted = _gaussian_db(wavelengths, peak_wl, peak_db, fwhm_nm)
        ss_res = np.sum((s21_db - fitted) ** 2)
        ss_tot = np.sum((s21_db - np.mean(s21_db)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        assert r_squared > 0.8, f"Spectrum R^2={r_squared:.2f}, not Gaussian enough"
