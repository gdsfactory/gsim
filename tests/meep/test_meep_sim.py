"""Tests for MEEP simulation module."""

from __future__ import annotations

import ast
import json

import pytest

from gsim.meep import FDTDConfig, MeepSim, ResolutionConfig, SimConfig, SParameterResult
from gsim.meep.models.config import LayerStackEntry, MaterialData, PortData

# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------


class TestFDTDConfig:
    """Test FDTDConfig frequency/wavelength conversion."""

    def test_defaults(self):
        cfg = FDTDConfig()
        assert cfg.wavelength == 1.55
        assert cfg.bandwidth == 0.1
        assert cfg.num_freqs == 21

    def test_fcen(self):
        cfg = FDTDConfig(wavelength=1.55)
        assert abs(cfg.fcen - 1.0 / 1.55) < 1e-10

    def test_df(self):
        cfg = FDTDConfig(wavelength=1.55, bandwidth=0.1)
        wl_min = 1.55 - 0.05
        wl_max = 1.55 + 0.05
        expected_df = 1.0 / wl_min - 1.0 / wl_max
        assert abs(cfg.df - expected_df) < 1e-10

    def test_to_dict(self):
        cfg = FDTDConfig()
        d = cfg.to_dict()
        assert "wavelength" in d
        assert "fcen" in d
        assert "df" in d
        assert "num_freqs" in d


class TestResolutionConfig:
    """Test ResolutionConfig presets."""

    def test_default(self):
        cfg = ResolutionConfig.default()
        assert cfg.pixels_per_um == 32

    def test_coarse(self):
        cfg = ResolutionConfig.coarse()
        assert cfg.pixels_per_um == 16

    def test_fine(self):
        cfg = ResolutionConfig.fine()
        assert cfg.pixels_per_um == 64

    def test_custom(self):
        cfg = ResolutionConfig(pixels_per_um=48)
        assert cfg.pixels_per_um == 48

    def test_to_dict(self):
        cfg = ResolutionConfig()
        d = cfg.to_dict()
        assert d["pixels_per_um"] == 32


class TestSimConfig:
    """Test SimConfig serialization."""

    def test_to_json(self, tmp_path):
        cfg = SimConfig(
            gds_filename="layout.gds",
            layer_stack=[
                {
                    "layer_name": "core",
                    "gds_layer": [1, 0],
                    "zmin": 0.0,
                    "zmax": 0.22,
                    "material": "si",
                    "sidewall_angle": 0.0,
                }
            ],
            ports=[
                {
                    "name": "o1",
                    "center": [0, 0, 0.11],
                    "orientation": 0,
                    "width": 0.5,
                    "normal_axis": 0,
                    "direction": "-",
                    "is_source": True,
                }
            ],
            materials={"si": {"refractive_index": 3.47, "extinction_coeff": 0.0}},
            fdtd=FDTDConfig().to_dict(),
            resolution=ResolutionConfig().to_dict(),
        )
        path = tmp_path / "config.json"
        cfg.to_json(path)
        assert path.exists()

        data = json.loads(path.read_text())
        assert "layer_stack" in data
        assert "ports" in data
        assert "materials" in data
        assert data["gds_filename"] == "layout.gds"
        assert data["materials"]["si"]["refractive_index"] == 3.47
        assert data["layer_stack"][0]["layer_name"] == "core"


class TestPortData:
    """Test PortData model."""

    def test_creation(self):
        p = PortData(
            name="o1",
            center=[0.0, 0.0, 0.11],
            orientation=0.0,
            width=0.5,
            normal_axis=0,
            direction="-",
            is_source=True,
        )
        assert p.name == "o1"
        assert p.is_source

    def test_to_dict(self):
        p = PortData(
            name="o1",
            center=[0.0, 0.0, 0.11],
            orientation=180.0,
            width=0.5,
            normal_axis=0,
            direction="+",
        )
        d = p.to_dict()
        assert d["name"] == "o1"
        assert d["direction"] == "+"


class TestLayerStackEntry:
    """Test LayerStackEntry model."""

    def test_creation(self):
        entry = LayerStackEntry(
            layer_name="core",
            gds_layer=[1, 0],
            zmin=0.0,
            zmax=0.22,
            material="si",
        )
        assert entry.layer_name == "core"
        assert entry.gds_layer == [1, 0]
        assert entry.sidewall_angle == 0.0

    def test_with_sidewall_angle(self):
        entry = LayerStackEntry(
            layer_name="core",
            gds_layer=[1, 0],
            zmin=0.0,
            zmax=0.22,
            material="si",
            sidewall_angle=10.0,
        )
        assert entry.sidewall_angle == 10.0

    def test_to_dict(self):
        entry = LayerStackEntry(
            layer_name="clad",
            gds_layer=[2, 0],
            zmin=-0.5,
            zmax=0.5,
            material="SiO2",
        )
        d = entry.to_dict()
        assert d["layer_name"] == "clad"
        assert d["gds_layer"] == [2, 0]


class TestMaterialData:
    """Test MaterialData model."""

    def test_creation(self):
        m = MaterialData(refractive_index=3.47)
        assert m.refractive_index == 3.47
        assert m.extinction_coeff == 0.0

    def test_with_extinction(self):
        m = MaterialData(refractive_index=3.47, extinction_coeff=0.01)
        assert m.extinction_coeff == 0.01


# ---------------------------------------------------------------------------
# MeepSim validation tests
# ---------------------------------------------------------------------------


class TestMeepSimValidation:
    """Test MeepSim validation logic."""

    def test_missing_geometry(self):
        sim = MeepSim()
        result = sim.validate_config()
        assert not result.valid
        assert any("No component set" in e for e in result.errors)

    def test_no_stack_warning(self):
        sim = MeepSim()
        result = sim.validate_config()
        assert any("No stack configured" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Mixin method tests
# ---------------------------------------------------------------------------


class TestMeepSimMixin:
    """Test mixin methods on MeepSim."""

    def test_set_output_dir(self, tmp_path):
        sim = MeepSim()
        sim.set_output_dir(tmp_path / "test-meep")
        assert sim.output_dir == tmp_path / "test-meep"
        assert sim.output_dir.exists()

    def test_set_stack(self):
        sim = MeepSim()
        sim.set_stack(air_above=2.0)
        assert sim._stack_kwargs["air_above"]

    def test_set_stack_default_air_above(self):
        """Photonic default air_above should be 1.0 (not 200 like RF)."""
        sim = MeepSim()
        sim.set_stack()
        assert sim._stack_kwargs["air_above"]

    def test_set_material(self):
        sim = MeepSim()
        sim.set_material("si", refractive_index=3.47)
        assert "si" in sim.materials
        assert sim.materials["si"].refractive_index == 3.47

    def test_set_material_with_extinction(self):
        sim = MeepSim()
        sim.set_material("custom", refractive_index=2.0, extinction_coeff=0.01)
        assert sim.materials["custom"].extinction_coeff == 0.01

    def test_set_wavelength(self):
        sim = MeepSim()
        sim.set_wavelength(wavelength=1.31, bandwidth=0.05, num_freqs=11)
        assert sim.fdtd_config.wavelength == 1.31
        assert sim.fdtd_config.bandwidth == 0.05
        assert sim.fdtd_config.num_freqs == 11

    def test_set_resolution_direct(self):
        sim = MeepSim()
        sim.set_resolution(pixels_per_um=48)
        assert sim.resolution_config.pixels_per_um == 48

    def test_set_resolution_preset(self):
        sim = MeepSim()
        sim.set_resolution(preset="fine")
        assert sim.resolution_config.pixels_per_um == 64

    def test_set_source_port(self):
        sim = MeepSim()
        sim.set_source_port("o2")
        assert sim.source_port == "o2"

    def test_write_config_requires_output_dir(self):
        sim = MeepSim()
        with pytest.raises(ValueError, match="Output directory not set"):
            sim.write_config()


# ---------------------------------------------------------------------------
# Port extraction tests
# ---------------------------------------------------------------------------


class TestPortExtraction:
    """Test port info extraction."""

    def test_get_port_normal(self):
        from gsim.meep.ports import get_port_normal

        assert get_port_normal(0) == (0, "-")
        assert get_port_normal(90) == (1, "-")
        assert get_port_normal(180) == (0, "+")
        assert get_port_normal(270) == (1, "+")

    def test_get_port_normal_invalid(self):
        from gsim.meep.ports import get_port_normal

        with pytest.raises(ValueError, match="Invalid port orientation"):
            get_port_normal(45)


# ---------------------------------------------------------------------------
# Port z-center tests
# ---------------------------------------------------------------------------


class TestPortZCenter:
    """Test photonic port z-center logic."""

    def test_z_center_uses_highest_n(self):
        """Port z-center should use layer with highest refractive index."""
        from gsim.common.stack.extractor import Layer, LayerStack

        stack = LayerStack(
            layers={
                "clad": Layer(
                    name="clad",
                    gds_layer=(2, 0),
                    zmin=-1.0,
                    zmax=1.0,
                    thickness=2.0,
                    material="SiO2",
                    layer_type="dielectric",
                ),
                "core": Layer(
                    name="core",
                    gds_layer=(1, 0),
                    zmin=0.0,
                    zmax=0.22,
                    thickness=0.22,
                    material="si",
                    layer_type="dielectric",
                ),
            }
        )

        from gsim.meep.ports import _get_z_center

        z = _get_z_center(stack)
        # Si (n=3.47) has highest index, so z-center should be midpoint of core
        assert abs(z - 0.11) < 1e-6

    def test_z_center_fallback(self):
        """Falls back to midpoint of all layers when no optical data."""
        from gsim.common.stack.extractor import Layer, LayerStack

        stack = LayerStack(
            layers={
                "unknown_layer": Layer(
                    name="unknown_layer",
                    gds_layer=(99, 0),
                    zmin=0.0,
                    zmax=1.0,
                    thickness=1.0,
                    material="unknown_material",
                    layer_type="dielectric",
                ),
            }
        )

        from gsim.meep.ports import _get_z_center

        z = _get_z_center(stack)
        assert abs(z - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Materials resolution tests
# ---------------------------------------------------------------------------


class TestMaterialsResolution:
    """Test material resolution from common DB."""

    def test_resolve_known_material(self):
        from gsim.common.stack.materials import get_material_properties

        props = get_material_properties("silicon")
        assert props is not None
        assert props.refractive_index == 3.47

    def test_resolve_sio2(self):
        from gsim.common.stack.materials import get_material_properties

        props = get_material_properties("SiO2")
        assert props is not None
        assert props.refractive_index == 1.44

    def test_optical_classmethod(self):
        from gsim.common.stack.materials import MaterialProperties

        mat = MaterialProperties.optical(refractive_index=2.5)
        assert mat.refractive_index == 2.5
        assert mat.extinction_coeff == 0.0
        assert mat.type == "dielectric"


# ---------------------------------------------------------------------------
# Script generation tests
# ---------------------------------------------------------------------------


class TestScriptGeneration:
    """Test MEEP runner script generation."""

    def test_generates_valid_python(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        ast.parse(script)

    def test_contains_config_filename(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script("my_config.json")
        assert "my_config.json" in script

    def test_has_main_function(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "def main():" in script
        assert 'if __name__ == "__main__":' in script

    def test_has_gds_loading(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "import gdsfactory" in script
        assert "import_gds" in script

    def test_has_triangulation(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "triangulate_polygon_with_holes" in script
        assert "Delaunay" in script


# ---------------------------------------------------------------------------
# SParameterResult tests
# ---------------------------------------------------------------------------


class TestSParameterResult:
    """Test S-parameter result parsing."""

    def test_from_csv(self, tmp_path):
        csv_path = tmp_path / "s_parameters.csv"
        csv_path.write_text(
            "wavelength,S11_mag,S11_phase,S21_mag,S21_phase\n"
            "1.500000,0.100000,-30.0000,0.900000,45.0000\n"
            "1.550000,0.150000,-25.0000,0.850000,50.0000\n"
        )

        result = SParameterResult.from_csv(csv_path)
        assert len(result.wavelengths) == 2
        assert "S11" in result.s_params
        assert "S21" in result.s_params
        assert len(result.s_params["S11"]) == 2
        assert abs(abs(result.s_params["S11"][0]) - 0.1) < 1e-6

    def test_empty_result(self):
        result = SParameterResult()
        assert result.wavelengths == []
        assert result.s_params == {}


# ---------------------------------------------------------------------------
# Layer sidewall_angle test
# ---------------------------------------------------------------------------


class TestLayerSidewallAngle:
    """Test sidewall_angle field on Layer model."""

    def test_default_zero(self):
        from gsim.common.stack.extractor import Layer

        layer = Layer(
            name="test",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=0.22,
            thickness=0.22,
            material="si",
            layer_type="dielectric",
        )
        assert layer.sidewall_angle == 0.0

    def test_custom_angle(self):
        from gsim.common.stack.extractor import Layer

        layer = Layer(
            name="test",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=0.22,
            thickness=0.22,
            material="si",
            layer_type="dielectric",
            sidewall_angle=10.0,
        )
        assert layer.sidewall_angle == 10.0

    def test_to_dict_includes_nonzero_angle(self):
        from gsim.common.stack.extractor import Layer

        layer = Layer(
            name="test",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=0.22,
            thickness=0.22,
            material="si",
            layer_type="dielectric",
            sidewall_angle=5.0,
        )
        d = layer.to_dict()
        assert d["sidewall_angle"] == 5.0

    def test_to_dict_omits_zero_angle(self):
        from gsim.common.stack.extractor import Layer

        layer = Layer(
            name="test",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=0.22,
            thickness=0.22,
            material="si",
            layer_type="dielectric",
        )
        d = layer.to_dict()
        assert "sidewall_angle" not in d


# ---------------------------------------------------------------------------
# Integration test: import without meep
# ---------------------------------------------------------------------------


class TestImportWithoutMeep:
    """Verify module imports work without meep installed."""

    def test_import_meep_sim(self):
        from gsim.meep import MeepSim

        sim = MeepSim()
        assert sim.geometry is None

    def test_import_all_public_api(self):
        from gsim.meep import (
            FDTDConfig,
            MeepSim,
            ResolutionConfig,
            SimConfig,
            SParameterResult,
        )

        assert FDTDConfig is not None
        assert MeepSim is not None
        assert ResolutionConfig is not None
        assert SParameterResult is not None
        assert SimConfig is not None
