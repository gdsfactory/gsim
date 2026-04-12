"""Integration tests for the Meep simulation workflow.

These tests verify the full local workflow: configure -> validate ->
build_config -> write_config, stopping before cloud submission.
"""

from __future__ import annotations

import json

import gdsfactory as gf
import pytest

from gsim.meep import Simulation

gf.gpdk.PDK.activate()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def straight_component():
    """Simple straight waveguide with two ports."""
    return gf.components.straight(length=10, width=0.5)


@pytest.fixture
def configured_sim(straight_component):
    """Fully configured Simulation ready for build_config/write_config."""
    sim = Simulation()
    sim.geometry.component = straight_component
    sim.materials = {"si": 3.47, "SiO2": 1.44}
    sim.source.port = "o1"
    sim.source.wavelength = 1.55
    sim.source.wavelength_span = 0.1
    sim.source.num_freqs = 11
    sim.monitors = ["o2"]
    sim.domain.pml = 1.0
    sim.domain.margin = 0.5
    sim.solver.resolution = 16
    sim.solver.stop_when_energy_decayed(dt=20, decay_by=0.01)
    return sim


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Test configuration validation catches errors and warnings."""

    def test_missing_component_is_invalid(self):
        sim = Simulation()
        result = sim.validate_config()
        assert not result.valid
        assert any("No component" in e for e in result.errors)

    def test_invalid_source_port(self, straight_component):
        sim = Simulation()
        sim.geometry.component = straight_component
        sim.source.port = "nonexistent"
        result = sim.validate_config()
        assert not result.valid
        assert any("nonexistent" in e for e in result.errors)

    def test_invalid_monitor_port(self, straight_component):
        sim = Simulation()
        sim.geometry.component = straight_component
        sim.monitors = ["o1", "bad_port"]
        result = sim.validate_config()
        assert not result.valid
        assert any("bad_port" in e for e in result.errors)

    def test_no_stack_warning(self, straight_component):
        sim = Simulation()
        sim.geometry.component = straight_component
        result = sim.validate_config()
        assert result.valid
        assert any("No stack" in w for w in result.warnings)

    def test_valid_config_passes(self, configured_sim):
        result = configured_sim.validate_config()
        assert result.valid


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------


class TestBuildConfig:
    """Test the build_config pipeline produces correct output."""

    def test_returns_build_result(self, configured_sim):
        result = configured_sim.build_config()
        assert result.config is not None
        assert result.component is not None
        assert result.original_component is not None

    def test_config_has_ports(self, configured_sim):
        result = configured_sim.build_config()
        assert len(result.config.ports) >= 2
        port_names = [p.name for p in result.config.ports]
        assert "o1" in port_names
        assert "o2" in port_names

    def test_source_port_flagged(self, configured_sim):
        result = configured_sim.build_config()
        source_ports = [p for p in result.config.ports if p.is_source]
        assert len(source_ports) == 1
        assert source_ports[0].name == "o1"

    def test_config_has_layer_stack(self, configured_sim):
        result = configured_sim.build_config()
        assert len(result.config.layer_stack) > 0

    def test_config_has_materials(self, configured_sim):
        result = configured_sim.build_config()
        assert len(result.config.materials) > 0

    def test_wavelength_config(self, configured_sim):
        result = configured_sim.build_config()
        assert abs(result.config.wavelength.wavelength - 1.55) < 1e-6
        assert result.config.wavelength.num_freqs == 11
        assert result.config.wavelength.fcen > 0
        assert result.config.wavelength.df > 0

    def test_source_fwidth_positive(self, configured_sim):
        result = configured_sim.build_config()
        assert result.config.source.fwidth > 0

    def test_stopping_config(self, configured_sim):
        result = configured_sim.build_config()
        assert result.config.stopping.mode == "energy_decay"
        assert result.config.stopping.threshold == 0.01

    def test_resolution_config(self, configured_sim):
        result = configured_sim.build_config()
        assert result.config.resolution.pixels_per_um == 16

    def test_domain_config(self, configured_sim):
        result = configured_sim.build_config()
        assert result.config.domain.dpml == 1.0
        assert result.config.domain.margin_xy == 0.5

    def test_extended_component_differs(self, configured_sim):
        """Ports should be extended into PML, making the component longer."""
        result = configured_sim.build_config()
        orig_bbox = result.original_component.dbbox()
        ext_bbox = result.component.dbbox()
        # Extended component should be wider in x (port direction)
        assert ext_bbox.right >= orig_bbox.right
        assert ext_bbox.left <= orig_bbox.left

    def test_invalid_config_raises(self):
        sim = Simulation()
        with pytest.raises(ValueError, match="Invalid configuration"):
            sim.build_config()


# ---------------------------------------------------------------------------
# write_config
# ---------------------------------------------------------------------------


class TestWriteConfig:
    """Test write_config produces all required output files."""

    def test_creates_output_files(self, configured_sim, tmp_path):
        output_dir = configured_sim.write_config(tmp_path / "sim_output")
        assert (output_dir / "layout.gds").exists()
        assert (output_dir / "sim_config.json").exists()
        assert (output_dir / "run_meep.py").exists()

    def test_config_json_valid(self, configured_sim, tmp_path):
        output_dir = configured_sim.write_config(tmp_path / "sim_output")
        data = json.loads((output_dir / "sim_config.json").read_text())

        assert data["gds_filename"] == "layout.gds"
        assert "layer_stack" in data
        assert "ports" in data
        assert "materials" in data
        assert "fdtd" in data  # wavelength config serialized as "fdtd"
        assert "source" in data
        assert "stopping" in data
        assert "resolution" in data
        assert "domain" in data

    def test_config_json_ports(self, configured_sim, tmp_path):
        output_dir = configured_sim.write_config(tmp_path / "sim_output")
        data = json.loads((output_dir / "sim_config.json").read_text())

        port_names = [p["name"] for p in data["ports"]]
        assert "o1" in port_names
        assert "o2" in port_names
        source_ports = [p for p in data["ports"] if p.get("is_source")]
        assert len(source_ports) == 1

    def test_runner_script_is_valid_python(self, configured_sim, tmp_path):
        import ast

        output_dir = configured_sim.write_config(tmp_path / "sim_output")
        script = (output_dir / "run_meep.py").read_text()
        ast.parse(script)
        assert "def main():" in script

    def test_creates_parent_dirs(self, configured_sim, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "sim"
        output_dir = configured_sim.write_config(deep_path)
        assert output_dir.exists()
        assert (output_dir / "sim_config.json").exists()


# ---------------------------------------------------------------------------
# Stopping modes
# ---------------------------------------------------------------------------


class TestStoppingModes:
    """Test different stopping criteria produce correct configs."""

    def test_dft_decay(self, straight_component, tmp_path):
        sim = Simulation()
        sim.geometry.component = straight_component
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source.port = "o1"
        sim.monitors = ["o2"]
        sim.solver.resolution = 16
        sim.solver.stop_when_dft_decayed(tol=1e-3, min_time=50)

        output_dir = sim.write_config(tmp_path / "dft")
        data = json.loads((output_dir / "sim_config.json").read_text())
        assert data["stopping"]["mode"] == "dft_decay"
        assert data["stopping"]["decay_by"] == 1e-3

    def test_field_decay(self, straight_component, tmp_path):
        sim = Simulation()
        sim.geometry.component = straight_component
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source.port = "o1"
        sim.monitors = ["o2"]
        sim.solver.resolution = 16
        sim.solver.stop_when_fields_decayed(dt=50, component="Hz", decay_by=0.05)

        output_dir = sim.write_config(tmp_path / "field_decay")
        data = json.loads((output_dir / "sim_config.json").read_text())
        assert data["stopping"]["mode"] == "field_decay"
        assert data["stopping"]["decay_component"] == "Hz"

    def test_fixed_time(self, straight_component, tmp_path):
        sim = Simulation()
        sim.geometry.component = straight_component
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source.port = "o1"
        sim.monitors = ["o2"]
        sim.solver.resolution = 16
        sim.solver.stop_after_sources(time=500)

        output_dir = sim.write_config(tmp_path / "fixed")
        data = json.loads((output_dir / "sim_config.json").read_text())
        assert data["stopping"]["mode"] == "fixed"
        assert data["stopping"]["run_after_sources"] == 500


# ---------------------------------------------------------------------------
# Different component types
# ---------------------------------------------------------------------------


class TestDifferentComponents:
    """Test workflow with various gdsfactory components."""

    def test_mmi_component(self):
        component = gf.components.mmi1x2(length_mmi=5.5, width_mmi=2.5)
        sim = Simulation()
        sim.geometry.component = component
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source.port = "o1"
        sim.monitors = ["o2", "o3"]
        sim.solver.resolution = 16

        result = sim.build_config()
        port_names = [p.name for p in result.config.ports]
        assert "o1" in port_names
        assert "o2" in port_names
        assert "o3" in port_names

    def test_bend_component(self, tmp_path):
        component = gf.components.bend_euler(radius=10)
        sim = Simulation()
        sim.geometry.component = component
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source.port = "o1"
        sim.monitors = ["o2"]
        sim.solver.resolution = 16

        output_dir = sim.write_config(tmp_path / "bend")
        assert (output_dir / "sim_config.json").exists()
        data = json.loads((output_dir / "sim_config.json").read_text())
        assert len(data["ports"]) == 2
