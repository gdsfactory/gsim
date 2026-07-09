"""Tests for PDK overlay loading and merging."""

from __future__ import annotations

import pytest

from gsim.common.stack.materials import MaterialProperties
from gsim.common.stack.overlays import load_overlay, merge_overlay


class TestLoadOverlay:
    def test_load_valid_yaml(self, tmp_path):
        yaml_content = """
materials:
  SiO2:
    type: dielectric
    permittivity: 4.0
    loss_tangent: 0.001
    dispersion_models:
      - type: constant
        permittivity: 4.0
        source: "Custom PDK"
"""
        overlay_path = tmp_path / "overlay.yaml"
        overlay_path.write_text(yaml_content)

        result = load_overlay(overlay_path)
        assert "SiO2" in result
        assert result["SiO2"].permittivity == 4.0
        assert result["SiO2"].loss_tangent == 0.001

    def test_load_with_frequency_validity(self, tmp_path):
        yaml_content = """
materials:
  SiO2:
    type: dielectric
    permittivity: 4.1
    dispersion_models:
      - type: constant
        permittivity: 4.1
        validity_frequency: [0, 10e9]
        source: "IHP SG13G2 PDK"
"""
        overlay_path = tmp_path / "overlay.yaml"
        overlay_path.write_text(yaml_content)

        result = load_overlay(overlay_path)
        assert "SiO2" in result
        dm = result["SiO2"].dispersion_models
        assert len(dm) == 1
        assert dm[0].validity.valid_frequency is not None

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_overlay("/nonexistent/overlay.yaml")

    def test_load_empty_yaml(self, tmp_path):
        yaml_content = "# empty\n"
        overlay_path = tmp_path / "empty.yaml"
        overlay_path.write_text(yaml_content)

        result = load_overlay(overlay_path)
        assert result == {}

    def test_load_with_anisotropic(self, tmp_path):
        yaml_content = """
materials:
  sapphire:
    type: dielectric
    permittivity: [9.3, 9.3, 11.5]
    material_axes: [[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]
"""
        overlay_path = tmp_path / "overlay.yaml"
        overlay_path.write_text(yaml_content)

        result = load_overlay(overlay_path)
        assert "sapphire" in result
        assert result["sapphire"].permittivity == [9.3, 9.3, 11.5]

    def test_load_with_wavelength_validity(self, tmp_path):
        yaml_content = """
materials:
  Si3N4:
    type: dielectric
    permittivity: 7.5
    dispersion_models:
      - type: sellmeier
        source: "Luke et al. 2015"
        validity_wavelength: [0.31, 5.5]
"""
        overlay_path = tmp_path / "overlay.yaml"
        overlay_path.write_text(yaml_content)

        result = load_overlay(overlay_path)
        assert "Si3N4" in result


class TestMergeOverlay:
    def test_merge_adds_new_material(self):
        overlay = {
            "custom_mat": MaterialProperties(permittivity=5.0),
        }
        merged = merge_overlay(overlay)
        assert "custom_mat" in merged
        assert merged["custom_mat"].permittivity == 5.0

    def test_merge_overrides_existing(self):
        overlay = {
            "SiO2": MaterialProperties(permittivity=3.9, loss_tangent=0.001),
        }
        merged = merge_overlay(overlay)
        assert merged["SiO2"].permittivity == 3.9
        assert merged["SiO2"].loss_tangent == 0.001

    def test_merge_preserves_non_overlaid(self):
        overlay = {
            "custom": MaterialProperties(permittivity=5.0),
        }
        merged = merge_overlay(overlay)
        assert "aluminum" in merged
        assert "copper" in merged

    def test_merge_does_not_mutate_base(self):
        from gsim.common.stack.materials import MATERIALS_DB

        original_count = len(MATERIALS_DB)
        overlay = {
            "custom": MaterialProperties(permittivity=5.0),
        }
        merge_overlay(overlay)
        assert len(MATERIALS_DB) == original_count

    def test_merge_case_insensitive_override(self):
        """merge_overlay matches overlay keys case-insensitively to base."""
        overlay = {
            "sio2": MaterialProperties(permittivity=3.8),
        }
        merged = merge_overlay(overlay)
        assert merged["SiO2"].permittivity == 3.8

    def test_merge_case_insensitive_preserves_base(self):
        """Case-insensitive merge preserves non-overlaid base materials."""
        overlay = {
            "custom_material": MaterialProperties(permittivity=5.0),
        }
        merged = merge_overlay(overlay)
        assert "aluminum" in merged
        assert merged["custom_material"].permittivity == 5.0
