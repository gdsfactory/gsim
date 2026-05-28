"""Tests for Palace material resolution with frequency-dependent dispersion."""

from __future__ import annotations

import pytest
from scipy.constants import c as C0  # noqa: N812

from gsim.common.stack.materials import MATERIALS_DB
from gsim.palace.materials import resolve_palace_materials_at_frequency
from gsim.palace.models import DrivenConfig


class TestDrivenConfigCenterFrequency:
    def test_center_frequency_band(self):
        d = DrivenConfig(fmin=1e9, fmax=100e9)
        assert d.center_frequency == pytest.approx(50.5e9)

    def test_center_frequency_single_point(self):
        d = DrivenConfig(fmin=10e9, fmax=10e9)
        assert d.center_frequency == pytest.approx(10e9)

    def test_single_freq_mode(self):
        d = DrivenConfig(fmin=50e9, fmax=50e9, num_points=1)
        assert d.fmin == d.fmax
        assert d.center_frequency == pytest.approx(50e9)


class TestSetDrivenSingleFreq:
    def test_f_sets_fmin_fmax(self):
        from gsim.palace import DrivenSim

        sim = DrivenSim()
        sim.set_driven(f=50e9)
        assert sim.driven.fmin == 50e9
        assert sim.driven.fmax == 50e9
        assert sim.driven.num_points == 1
        assert sim.driven.center_frequency == pytest.approx(50e9)

    def test_fmin_fmax_override_f(self):
        from gsim.palace import DrivenSim

        sim = DrivenSim()
        sim.set_driven(f=50e9, fmin=1e9, fmax=100e9, num_points=40)
        assert sim.driven.fmin == 1e9
        assert sim.driven.fmax == 100e9
        assert sim.driven.num_points == 40


class TestResolvePalaceMaterialsAtFrequency:
    def test_sio2_at_optical_frequency(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        freq_hz = C0 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "SiO2" in resolved
        assert resolved["SiO2"]["permittivity"] == pytest.approx(2.085, abs=0.01)

    def test_sio2_at_rf_frequency(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        freq_hz = 5e9
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "SiO2" in resolved
        assert resolved["SiO2"]["permittivity"] == pytest.approx(4.1)

    def test_silicon_at_optical_frequency(self):
        materials = {"silicon": MATERIALS_DB["silicon"].to_dict()}
        freq_hz = C0 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "silicon" in resolved
        n_sq = resolved["silicon"]["permittivity"]
        n = n_sq**0.5
        assert 3.4 < n < 3.6

    def test_conductor_unchanged(self):
        materials = {"aluminum": MATERIALS_DB["aluminum"].to_dict()}
        resolved = resolve_palace_materials_at_frequency(materials, 5e9)
        assert "aluminum" in resolved
        assert resolved["aluminum"]["conductivity"] == 3.77e7

    def test_unknown_material_preserved(self):
        materials = {"custom_mat": {"permittivity": 5.0}}
        resolved = resolve_palace_materials_at_frequency(materials, 5e9)
        assert "custom_mat" in resolved
        assert resolved["custom_mat"]["permittivity"] == 5.0

    def test_empty_materials(self):
        resolved = resolve_palace_materials_at_frequency({}, 5e9)
        assert resolved == {}

    def test_preserves_nonoptical_fields(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        freq_hz = C0 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert resolved["SiO2"]["permittivity"] is not None

    def test_does_not_mutate_input(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        original_permittivity = materials["SiO2"]["permittivity"]
        resolve_palace_materials_at_frequency(materials, 5e9)
        assert materials["SiO2"]["permittivity"] == original_permittivity

    def test_sapphire_anisotropic_resolved(self):
        materials = {"sapphire": MATERIALS_DB["sapphire"].to_dict()}
        freq_hz = C0 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "sapphire" in resolved
        assert isinstance(resolved["sapphire"]["permittivity"], list)
