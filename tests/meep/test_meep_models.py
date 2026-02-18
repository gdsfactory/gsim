"""Tests for MEEP config models, ports, materials, script generation, and overlays."""

from __future__ import annotations

import ast
import json

import pytest
from pydantic import ValidationError

from gsim.meep import (
    DomainConfig,
    ResolutionConfig,
    SimConfig,
    SourceConfig,
    SParameterResult,
    WavelengthConfig,
)
from gsim.meep.models.config import (
    LayerStackEntry,
    MaterialData,
    PortData,
    StoppingConfig,
    SymmetryEntry,
)

# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------


class TestWavelengthConfig:
    """Test WavelengthConfig frequency/wavelength conversion."""

    def test_requires_all_fields(self):
        with pytest.raises(ValidationError):
            WavelengthConfig()

    def test_fcen(self):
        cfg = WavelengthConfig(wavelength=1.55, bandwidth=0.1, num_freqs=11)
        assert abs(cfg.fcen - 1.0 / 1.55) < 1e-10

    def test_df(self):
        cfg = WavelengthConfig(wavelength=1.55, bandwidth=0.1, num_freqs=11)
        wl_min = 1.55 - 0.05
        wl_max = 1.55 + 0.05
        expected_df = 1.0 / wl_min - 1.0 / wl_max
        assert abs(cfg.df - expected_df) < 1e-10

    def test_model_dump(self):
        cfg = WavelengthConfig(wavelength=1.55, bandwidth=0.1, num_freqs=11)
        d = cfg.model_dump()
        assert "wavelength" in d
        assert "fcen" in d
        assert "df" in d
        assert "num_freqs" in d
        # stopping and run_after_sources are no longer in fdtd dict
        assert "stopping" not in d
        assert "run_after_sources" not in d


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

    def test_model_dump(self):
        cfg = ResolutionConfig(pixels_per_um=32)
        d = cfg.model_dump()
        assert d["pixels_per_um"] == 32


class TestSimConfig:
    """Test SimConfig serialization."""

    def test_to_json(self, tmp_path):
        wl_cfg = WavelengthConfig(wavelength=1.55, bandwidth=0.1, num_freqs=11)
        fwidth = SourceConfig().compute_fwidth(wl_cfg.fcen, wl_cfg.df)
        source_cfg = SourceConfig(fwidth=fwidth)
        stopping_cfg = StoppingConfig(
            mode="dft_decay",
            max_time=200.0,
            decay_dt=50.0,
            decay_component="Ey",
            threshold=0.05,
            dft_min_run_time=100,
        )
        from gsim.meep.models.config import (
            AccuracyConfig,
            DiagnosticsConfig,
        )

        cfg = SimConfig(
            gds_filename="layout.gds",
            verbose_interval=0,
            layer_stack=[  # ty: ignore[invalid-argument-type]
                {
                    "layer_name": "core",
                    "gds_layer": [1, 0],
                    "zmin": 0.0,
                    "zmax": 0.22,
                    "material": "si",
                    "sidewall_angle": 0.0,
                }
            ],
            ports=[  # ty: ignore[invalid-argument-type]
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
            materials={"si": {"refractive_index": 3.47, "extinction_coeff": 0.0}},  # ty: ignore[invalid-argument-type]
            wavelength=wl_cfg,
            source=source_cfg,
            stopping=stopping_cfg,
            resolution=ResolutionConfig(pixels_per_um=32),
            domain=DomainConfig(
                dpml=1.0,
                margin_xy=0.5,
                margin_z_above=0.5,
                margin_z_below=0.5,
                port_margin=0.5,
                extend_ports=0.0,
                source_port_offset=0.1,
                distance_source_to_monitors=0.2,
            ),
            accuracy=AccuracyConfig(
                eps_averaging=False,
                subpixel_maxeval=0,
                subpixel_tol=1e-4,
                simplify_tol=0.0,
            ),
            diagnostics=DiagnosticsConfig(
                save_geometry=True,
                save_fields=True,
                save_epsilon_raw=False,
                save_animation=False,
                animation_interval=0.5,
                preview_only=False,
                verbose_interval=0,
            ),
            dielectrics=[],
            symmetries=[],
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
        # JSON keys use serialization aliases (fdtd, run_after_sources, decay_by)
        assert "source" in data
        assert data["source"]["fwidth"] > data["fdtd"]["df"]
        assert "stopping" in data
        assert data["stopping"]["mode"] == "dft_decay"
        assert data["stopping"]["run_after_sources"] == 200.0


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

    def test_model_dump(self):
        p = PortData(
            name="o1",
            center=[0.0, 0.0, 0.11],
            orientation=180.0,
            width=0.5,
            normal_axis=0,
            direction="+",
        )
        d = p.model_dump()
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

    def test_model_dump(self):
        entry = LayerStackEntry(
            layer_name="clad",
            gds_layer=[2, 0],
            zmin=-0.5,
            zmax=0.5,
            material="SiO2",
        )
        d = entry.model_dump()
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
# Import test
# ---------------------------------------------------------------------------


class TestImportWithoutMeep:
    """Verify module imports work without meep installed."""

    def test_import_all_public_api(self):
        from gsim.meep import (
            DomainConfig,
            ResolutionConfig,
            SimConfig,
            Simulation,
            SourceConfig,
            SParameterResult,
            WavelengthConfig,
        )

        assert WavelengthConfig is not None
        assert DomainConfig is not None
        assert ResolutionConfig is not None
        assert SParameterResult is not None
        assert SimConfig is not None
        assert SourceConfig is not None
        assert Simulation is not None


# ---------------------------------------------------------------------------
# DomainConfig tests
# ---------------------------------------------------------------------------


class TestDomainConfig:
    """Test DomainConfig model."""

    def test_requires_all_fields(self):
        with pytest.raises(ValidationError):
            DomainConfig()

    def test_custom(self):
        cfg = DomainConfig(
            dpml=0.5,
            margin_xy=0.2,
            margin_z_above=0.3,
            margin_z_below=0.4,
            port_margin=0.5,
            extend_ports=0.0,
            source_port_offset=0.1,
            distance_source_to_monitors=0.2,
        )
        assert cfg.dpml == 0.5
        assert cfg.margin_xy == 0.2
        assert cfg.margin_z_above == 0.3
        assert cfg.margin_z_below == 0.4

    def test_extend_ports_custom(self):
        cfg = DomainConfig(
            dpml=1.0,
            margin_xy=0.5,
            margin_z_above=0.5,
            margin_z_below=0.5,
            port_margin=0.5,
            extend_ports=2.5,
            source_port_offset=0.1,
            distance_source_to_monitors=0.2,
        )
        assert cfg.extend_ports == 2.5

    def test_extend_ports_serialization(self):
        cfg = DomainConfig(
            dpml=1.0,
            margin_xy=0.5,
            margin_z_above=0.5,
            margin_z_below=0.5,
            port_margin=0.5,
            extend_ports=3.0,
            source_port_offset=0.1,
            distance_source_to_monitors=0.2,
        )
        d = cfg.model_dump()
        assert d["extend_ports"] == 3.0

    def test_model_dump(self):
        cfg = DomainConfig(
            dpml=2.0,
            margin_xy=0.5,
            margin_z_above=1.0,
            margin_z_below=1.5,
            port_margin=0.5,
            extend_ports=0.0,
            source_port_offset=0.1,
            distance_source_to_monitors=0.2,
        )
        d = cfg.model_dump()
        assert d["dpml"] == 2.0
        assert d["margin_xy"] == 0.5
        assert d["margin_z_above"] == 1.0
        assert d["margin_z_below"] == 1.5


class TestSimConfigComponentBbox:
    """Test SimConfig.component_bbox field."""

    @pytest.fixture
    def _sim_kwargs(self):
        from gsim.meep.models.config import AccuracyConfig, DiagnosticsConfig

        return dict(
            gds_filename="layout.gds",
            verbose_interval=0,
            layer_stack=[],
            dielectrics=[],
            ports=[],
            materials={},
            wavelength=WavelengthConfig(wavelength=1.55, bandwidth=0.1, num_freqs=11),
            source=SourceConfig(),
            stopping=StoppingConfig(
                mode="fixed",
                max_time=100.0,
                decay_dt=50.0,
                decay_component="Ey",
                threshold=0.05,
                dft_min_run_time=100,
            ),
            resolution=ResolutionConfig(pixels_per_um=32),
            domain=DomainConfig(
                dpml=1.0,
                margin_xy=0.5,
                margin_z_above=0.5,
                margin_z_below=0.5,
                port_margin=0.5,
                extend_ports=0.0,
                source_port_offset=0.1,
                distance_source_to_monitors=0.2,
            ),
            accuracy=AccuracyConfig(
                eps_averaging=False,
                subpixel_maxeval=0,
                subpixel_tol=1e-4,
                simplify_tol=0.0,
            ),
            diagnostics=DiagnosticsConfig(
                save_geometry=True,
                save_fields=True,
                save_epsilon_raw=False,
                save_animation=False,
                animation_interval=0.5,
                preview_only=False,
                verbose_interval=0,
            ),
            symmetries=[],
        )

    def test_default_none(self, _sim_kwargs):
        cfg = SimConfig(**_sim_kwargs)
        assert cfg.component_bbox is None

    def test_with_bbox(self, _sim_kwargs):
        cfg = SimConfig(**_sim_kwargs, component_bbox=[-5.0, -2.0, 5.0, 2.0])
        assert cfg.component_bbox == [-5.0, -2.0, 5.0, 2.0]

    def test_json_roundtrip(self, tmp_path, _sim_kwargs):
        cfg = SimConfig(**_sim_kwargs, component_bbox=[-1.0, -0.5, 1.0, 0.5])
        path = tmp_path / "config.json"
        cfg.to_json(path)
        data = json.loads(path.read_text())
        assert data["component_bbox"] == [-1.0, -0.5, 1.0, 0.5]

    def test_json_roundtrip_none(self, tmp_path, _sim_kwargs):
        cfg = SimConfig(**_sim_kwargs)
        path = tmp_path / "config.json"
        cfg.to_json(path)
        data = json.loads(path.read_text())
        assert data["component_bbox"] is None


class TestDomainInSimConfig:
    """Test domain serialization in SimConfig JSON."""

    def test_domain_in_sim_config_json(self, tmp_path):
        from gsim.meep.models.config import AccuracyConfig, DiagnosticsConfig

        cfg = SimConfig(
            gds_filename="layout.gds",
            verbose_interval=0,
            layer_stack=[],
            dielectrics=[],
            ports=[],
            materials={},
            wavelength=WavelengthConfig(wavelength=1.55, bandwidth=0.1, num_freqs=11),
            source=SourceConfig(),
            stopping=StoppingConfig(
                mode="fixed",
                max_time=100.0,
                decay_dt=50.0,
                decay_component="Ey",
                threshold=0.05,
                dft_min_run_time=100,
            ),
            resolution=ResolutionConfig(pixels_per_um=32),
            domain=DomainConfig(
                dpml=0.5,
                margin_xy=0.2,
                margin_z_above=0.5,
                margin_z_below=0.5,
                port_margin=0.5,
                extend_ports=0.0,
                source_port_offset=0.1,
                distance_source_to_monitors=0.2,
            ),
            accuracy=AccuracyConfig(
                eps_averaging=False,
                subpixel_maxeval=0,
                subpixel_tol=1e-4,
                simplify_tol=0.0,
            ),
            diagnostics=DiagnosticsConfig(
                save_geometry=True,
                save_fields=True,
                save_epsilon_raw=False,
                save_animation=False,
                animation_interval=0.5,
                preview_only=False,
                verbose_interval=0,
            ),
            symmetries=[],
        )
        path = tmp_path / "config.json"
        cfg.to_json(path)

        data = json.loads(path.read_text())
        assert "domain" in data
        assert data["domain"]["dpml"] == 0.5
        assert data["domain"]["margin_xy"] == 0.2


# ---------------------------------------------------------------------------
# SymmetryEntry tests
# ---------------------------------------------------------------------------


class TestSymmetryEntry:
    """Test SymmetryEntry model."""

    def test_creation(self):
        s = SymmetryEntry(direction="X", phase=-1)
        assert s.direction == "X"
        assert s.phase == -1

    def test_requires_phase(self):
        with pytest.raises(ValidationError):
            SymmetryEntry(direction="Y")

    def test_model_dump(self):
        s = SymmetryEntry(direction="Z", phase=-1)
        d = s.model_dump()
        assert d == {"direction": "Z", "phase": -1}

    def test_invalid_direction(self):
        with pytest.raises(ValidationError):
            SymmetryEntry(direction="W")  # ty: ignore[invalid-argument-type]

    def test_invalid_phase(self):
        with pytest.raises(ValidationError):
            SymmetryEntry(direction="X", phase=2)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# StoppingConfig tests
# ---------------------------------------------------------------------------


class TestStoppingConfig:
    """Test StoppingConfig model."""

    def test_requires_all_fields(self):
        with pytest.raises(ValidationError):
            StoppingConfig()

    def test_field_decay_mode(self):
        cfg = StoppingConfig(
            mode="field_decay",
            threshold=1e-4,
            decay_dt=25.0,
            max_time=100.0,
            decay_component="Ey",
            dft_min_run_time=100,
        )
        assert cfg.mode == "field_decay"
        assert cfg.threshold == 1e-4
        assert cfg.decay_dt == 25.0

    def test_energy_decay_mode(self):
        cfg = StoppingConfig(
            mode="energy_decay",
            decay_dt=100,
            threshold=1e-4,
            max_time=100.0,
            decay_component="Ey",
            dft_min_run_time=100,
        )
        assert cfg.mode == "energy_decay"
        assert cfg.decay_dt == 100
        assert cfg.threshold == 1e-4

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            StoppingConfig(  # ty: ignore[invalid-argument-type]
                mode="invalid",
                max_time=100.0,
                decay_dt=50.0,
                decay_component="Ey",
                threshold=0.05,
                dft_min_run_time=100,
            )

    def test_threshold_bounds(self):
        with pytest.raises(ValidationError):
            StoppingConfig(
                mode="fixed",
                max_time=100.0,
                decay_dt=50.0,
                decay_component="Ey",
                threshold=0,
                dft_min_run_time=100,
            )
        with pytest.raises(ValidationError):
            StoppingConfig(
                mode="fixed",
                max_time=100.0,
                decay_dt=50.0,
                decay_component="Ey",
                threshold=1.0,
                dft_min_run_time=100,
            )

    def test_wall_time_max_default(self):
        cfg = StoppingConfig(
            mode="fixed",
            max_time=100.0,
            decay_dt=50.0,
            decay_component="Ey",
            threshold=0.05,
            dft_min_run_time=100,
        )
        assert cfg.wall_time_max == 0.0

    def test_wall_time_max_set(self):
        cfg = StoppingConfig(
            mode="field_decay",
            max_time=200.0,
            decay_dt=50.0,
            decay_component="Ey",
            threshold=0.05,
            dft_min_run_time=100,
            wall_time_max=3600,
        )
        assert cfg.wall_time_max == 3600

    def test_wall_time_max_serialized(self):
        cfg = StoppingConfig(
            mode="fixed",
            max_time=100.0,
            decay_dt=50.0,
            decay_component="Ey",
            threshold=0.05,
            dft_min_run_time=100,
            wall_time_max=1800,
        )
        data = cfg.model_dump()
        assert data["wall_time_max"] == 1800

    def test_wall_time_max_negative_rejected(self):
        with pytest.raises(ValidationError):
            StoppingConfig(
                mode="fixed",
                max_time=100.0,
                decay_dt=50.0,
                decay_component="Ey",
                threshold=0.05,
                dft_min_run_time=100,
                wall_time_max=-1,
            )


class TestWavelengthConfigStopping:
    """Test that WavelengthConfig no longer embeds StoppingConfig."""

    def test_model_dump_excludes_stopping(self):
        cfg = WavelengthConfig(wavelength=1.55, bandwidth=0.1, num_freqs=11)
        d = cfg.model_dump()
        assert "stopping" not in d
        assert "run_after_sources" not in d


# ---------------------------------------------------------------------------
# SourceConfig tests
# ---------------------------------------------------------------------------


class TestSourceConfig:
    """Test SourceConfig model."""

    def test_defaults(self):
        cfg = SourceConfig()
        assert cfg.bandwidth is None
        assert cfg.port is None

    def test_auto_fwidth(self):
        """Auto fwidth should be max(3*monitor_df, 0.2*fcen)."""
        cfg = SourceConfig()
        fcen = 1.0 / 1.55
        monitor_df = 0.042  # typical for 0.1um bandwidth at 1550nm
        fwidth = cfg.compute_fwidth(fcen, monitor_df)
        expected = max(3 * monitor_df, 0.2 * fcen)
        assert abs(fwidth - expected) < 1e-10
        assert fwidth > monitor_df  # always wider than monitor

    def test_explicit_bandwidth(self):
        """Explicit wavelength bandwidth converted to frequency."""
        cfg = SourceConfig(bandwidth=0.3)
        fcen = 1.0 / 1.55
        fwidth = cfg.compute_fwidth(fcen, 0.042)
        # Should convert 0.3um wavelength bw to freq bw
        wl_min = 1.55 - 0.15
        wl_max = 1.55 + 0.15
        expected = 1.0 / wl_min - 1.0 / wl_max
        assert abs(fwidth - expected) < 1e-10

    def test_model_dump(self):
        cfg = SourceConfig(port="o1")
        fcen = 1.0 / 1.55
        fwidth = cfg.compute_fwidth(fcen, 0.042)
        cfg_with_fwidth = cfg.model_copy(update={"fwidth": fwidth})
        d = cfg_with_fwidth.model_dump()
        assert "fwidth" in d
        assert "port" in d
        assert d["port"] == "o1"
        assert d["fwidth"] > 0

    def test_auto_fwidth_wider_than_monitor(self):
        """Auto source fwidth should always be wider than monitor df."""
        cfg = SourceConfig()
        fcen = 1.0 / 1.55
        wl = WavelengthConfig(wavelength=1.55, bandwidth=0.1, num_freqs=11)
        fwidth = cfg.compute_fwidth(fcen, wl.df)
        assert fwidth > wl.df


# ---------------------------------------------------------------------------
# Script symmetry tests
# ---------------------------------------------------------------------------


class TestScriptSymmetry:
    """Test that the runner script includes symmetry support."""

    def test_script_has_build_symmetries(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "build_symmetries" in script

    def test_script_has_mp_mirror(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "mp.Mirror" in script

    def test_script_has_split_chunks_evenly(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "split_chunks_evenly" in script


# ---------------------------------------------------------------------------
# Script decay tests
# ---------------------------------------------------------------------------


class TestScriptDecay:
    """Test that the runner script includes decay stopping logic."""

    def test_script_has_stop_when_fields_decayed(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "stop_when_fields_decayed" in script

    def test_script_has_component_map(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "_COMPONENT_MAP" in script
        assert "mp.Ez" in script
        assert "mp.Ey" in script

    def test_script_has_stop_when_energy_decayed(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "stop_when_energy_decayed" in script
        assert "energy_decay" in script

    def test_script_valid_python(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        ast.parse(script)


# ---------------------------------------------------------------------------
# Overlay tests
# ---------------------------------------------------------------------------


class TestOverlay:
    """Test SimOverlay + PortOverlay dataclasses and build_sim_overlay."""

    def test_port_overlay_creation(self):
        from gsim.meep.overlay import PortOverlay

        p = PortOverlay(
            name="o1",
            center=(0.0, 0.0, 0.11),
            width=0.5,
            normal_axis=0,
            direction="-",
            is_source=True,
            z_span=0.22,
        )
        assert p.name == "o1"
        assert p.is_source
        assert p.z_span == 0.22

    def test_sim_overlay_creation(self):
        from gsim.meep.overlay import SimOverlay

        ov = SimOverlay(
            cell_min=(-5.0, -2.0, -1.0),
            cell_max=(5.0, 2.0, 1.0),
            dpml=1.0,
            ports=[],
        )
        assert ov.cell_min == (-5.0, -2.0, -1.0)
        assert ov.dpml == 1.0

    def test_dielectric_overlay_creation(self):
        from gsim.meep.overlay import DielectricOverlay

        d = DielectricOverlay(
            name="oxide",
            material="SiO2",
            zmin=-1.0,
            zmax=0.0,
        )
        assert d.name == "oxide"
        assert d.material == "SiO2"
        assert d.zmin == -1.0

    def test_build_sim_overlay(self):
        import numpy as np

        from gsim.common.geometry_model import GeometryModel, Prism
        from gsim.meep.overlay import build_sim_overlay

        gm = GeometryModel(
            prisms={
                "core": [
                    Prism(
                        vertices=np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]]),
                        z_base=0.0,
                        z_top=0.22,
                        layer_name="core",
                    )
                ]
            },
            bbox=((-2.0, -1.0, 0.0), (2.0, 1.0, 0.22)),
        )

        domain_cfg = DomainConfig(
            dpml=1.0,
            margin_xy=0.5,
            margin_z_above=0.0,
            margin_z_below=0.0,
            port_margin=0.5,
            extend_ports=0.0,
            source_port_offset=0.1,
            distance_source_to_monitors=0.2,
        )

        port_data = [
            PortData(
                name="o1",
                center=[-2.0, 0.0, 0.11],
                orientation=0.0,
                width=0.5,
                normal_axis=0,
                direction="-",
                is_source=True,
            ),
            PortData(
                name="o2",
                center=[2.0, 0.0, 0.11],
                orientation=180.0,
                width=0.5,
                normal_axis=0,
                direction="+",
                is_source=False,
            ),
        ]

        overlay = build_sim_overlay(gm, domain_cfg, port_data)

        # cell_min = geo_min - (margin_xy + dpml) for xy, - dpml for z
        assert overlay.cell_min[0] == pytest.approx(-3.5)  # -2 - (0.5 + 1.0)
        assert overlay.cell_min[1] == pytest.approx(-2.5)  # -1 - (0.5 + 1.0)
        assert overlay.cell_min[2] == pytest.approx(-1.0)  # 0 - 1.0
        assert overlay.cell_max[0] == pytest.approx(3.5)
        assert overlay.cell_max[1] == pytest.approx(2.5)
        assert overlay.cell_max[2] == pytest.approx(1.22)  # 0.22 + 1.0

        assert len(overlay.ports) == 2
        assert overlay.ports[0].is_source
        assert not overlay.ports[1].is_source
        assert overlay.dpml == 1.0

    def test_build_sim_overlay_with_dielectrics(self):
        import numpy as np

        from gsim.common.geometry_model import GeometryModel, Prism
        from gsim.meep.overlay import build_sim_overlay

        gm = GeometryModel(
            prisms={
                "core": [
                    Prism(
                        vertices=np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]]),
                        z_base=0.0,
                        z_top=0.22,
                        layer_name="core",
                    )
                ]
            },
            bbox=((-2.0, -1.0, -1.0), (2.0, 1.0, 1.22)),
        )

        domain_cfg = DomainConfig(
            dpml=1.0,
            margin_xy=0.5,
            margin_z_above=0.5,
            margin_z_below=0.5,
            port_margin=0.5,
            extend_ports=0.0,
            source_port_offset=0.1,
            distance_source_to_monitors=0.2,
        )
        dielectrics = [
            {"name": "substrate", "material": "silicon", "zmin": -1.0, "zmax": 0.0},
            {"name": "oxide", "material": "SiO2", "zmin": 0.0, "zmax": 1.22},
        ]

        overlay = build_sim_overlay(gm, domain_cfg, [], dielectrics=dielectrics)

        assert len(overlay.dielectrics) == 2
        assert overlay.dielectrics[0].name == "substrate"
        assert overlay.dielectrics[0].material == "silicon"
        assert overlay.dielectrics[1].name == "oxide"
        assert overlay.dielectrics[1].material == "SiO2"


# ---------------------------------------------------------------------------
# Script domain config tests
# ---------------------------------------------------------------------------


class TestScriptDomainConfig:
    """Test that the runner script reads domain config."""

    def test_script_reads_domain(self):
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "domain" in script
        assert "dpml" in script
        assert "margin_xy" in script

    def test_script_uses_dpml_from_config(self):
        """Verify the script no longer hardcodes padding = 1.0."""
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "padding = 1.0" not in script

    def test_script_has_background_slabs(self):
        """Verify the script has build_background_slabs function."""
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "build_background_slabs" in script
        assert "mp.Block" in script
        assert "mp.inf" in script

    def test_script_handles_component_bbox(self):
        """Verify the runner script uses component_bbox for cell sizing."""
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "component_bbox" in script
        assert "bbox_left" in script
        assert "bbox_right" in script
        assert "bbox_top" in script
        assert "bbox_bottom" in script

    def test_script_contains_wall_time_cap(self):
        """Verify runner script has wall-clock time cap logic."""
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        assert "_make_wall_time_cap" in script
        assert "wall_time_max" in script

    def test_script_wall_time_all_modes(self):
        """Wall-clock cap should be wired into all 4 stopping modes."""
        from gsim.meep.script import generate_meep_script

        script = generate_meep_script()
        # Each mode branch should reference _make_wall_time_cap
        assert script.count("_make_wall_time_cap") >= 4


# ---------------------------------------------------------------------------
# Render2d overlay tests
# ---------------------------------------------------------------------------


class TestRender2dOverlay:
    """Test overlay drawing in render2d."""

    def test_plot_prism_slices_with_overlay(self):
        """Verify plot_prism_slices accepts overlay kwarg without error."""
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from gsim.common.geometry_model import GeometryModel, Prism
        from gsim.common.viz.render2d import plot_prism_slices
        from gsim.meep.overlay import PortOverlay, SimOverlay

        gm = GeometryModel(
            prisms={
                "core": [
                    Prism(
                        vertices=np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]]),
                        z_base=0.0,
                        z_top=0.22,
                        layer_name="core",
                    )
                ]
            },
            bbox=((-2.0, -1.0, 0.0), (2.0, 1.0, 0.22)),
        )

        overlay = SimOverlay(
            cell_min=(-3.0, -2.0, -1.0),
            cell_max=(3.0, 2.0, 1.22),
            dpml=1.0,
            ports=[
                PortOverlay(
                    name="o1",
                    center=(-2.0, 0.0, 0.11),
                    width=0.5,
                    normal_axis=0,
                    direction="-",
                    is_source=True,
                    z_span=0.22,
                ),
            ],
        )

        fig, ax = plt.subplots()
        result = plot_prism_slices(gm, z=0.11, ax=ax, overlay=overlay)
        assert result is ax

        # Check that sim cell boundary was drawn (check patches)
        patch_labels = [p.get_label() for p in ax.patches]
        assert "Sim cell" in patch_labels

        plt.close(fig)

    def test_plot_without_overlay_fallback(self):
        """Without overlay, should still draw geometry bbox."""
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from gsim.common.geometry_model import GeometryModel, Prism
        from gsim.common.viz.render2d import plot_prism_slices

        gm = GeometryModel(
            prisms={
                "core": [
                    Prism(
                        vertices=np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]]),
                        z_base=0.0,
                        z_top=0.22,
                        layer_name="core",
                    )
                ]
            },
            bbox=((-2.0, -1.0, 0.0), (2.0, 1.0, 0.22)),
        )

        fig, ax = plt.subplots()
        result = plot_prism_slices(gm, z=0.11, ax=ax)
        assert result is ax

        patch_labels = [p.get_label() for p in ax.patches]
        assert "Simulation" in patch_labels

        plt.close(fig)

    def test_plot_with_dielectric_overlay(self):
        """Verify dielectric overlays render without error."""
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from gsim.common.geometry_model import GeometryModel, Prism
        from gsim.common.viz.render2d import plot_prism_slices
        from gsim.meep.overlay import DielectricOverlay, SimOverlay

        gm = GeometryModel(
            prisms={
                "core": [
                    Prism(
                        vertices=np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]]),
                        z_base=0.0,
                        z_top=0.22,
                        layer_name="core",
                    )
                ]
            },
            bbox=((-2.0, -1.0, -1.0), (2.0, 1.0, 1.22)),
        )

        overlay = SimOverlay(
            cell_min=(-3.0, -2.0, -2.0),
            cell_max=(3.0, 2.0, 2.22),
            dpml=1.0,
            ports=[],
            dielectrics=[
                DielectricOverlay(
                    name="substrate", material="silicon", zmin=-1.0, zmax=0.0
                ),
                DielectricOverlay(name="oxide", material="SiO2", zmin=0.0, zmax=1.22),
            ],
        )

        # Test XZ view (side view — should draw horizontal bands)
        fig, ax = plt.subplots()
        result = plot_prism_slices(gm, y=0.0, ax=ax, slices="y", overlay=overlay)
        assert result is ax

        patch_labels = [p.get_label() for p in ax.patches]
        assert "substrate" in patch_labels
        assert "oxide" in patch_labels

        plt.close(fig)

        # Test XY view (top view — should draw background fill for active z)
        fig, ax = plt.subplots()
        result = plot_prism_slices(gm, z=0.11, ax=ax, slices="z", overlay=overlay)
        assert result is ax

        patch_labels = [p.get_label() for p in ax.patches]
        # z=0.11 is within oxide (0.0 to 1.22) but not substrate (-1.0 to 0.0)
        assert "oxide" in patch_labels

        plt.close(fig)
