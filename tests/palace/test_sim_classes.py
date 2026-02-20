"""Tests for Palace simulation classes."""

from __future__ import annotations

import pytest

from gsim.palace import DrivenSim, EigenmodeSim, ElectrostaticSim


class TestDrivenSimValidation:
    """Test DrivenSim validation logic."""

    def test_missing_geometry(self):
        """Test validation catches missing geometry."""
        sim = DrivenSim()
        result = sim.validate_config()
        assert not result.valid
        assert any("No component set" in e for e in result.errors)

    def test_inplane_port_requires_layer(self):
        """Test that add_port raises for inplane port without layer."""
        sim = DrivenSim()
        # PortConfig validates eagerly at creation time
        with pytest.raises(ValueError):
            sim.add_port("o1", geometry="inplane")  # No layer specified

    def test_via_port_requires_layers(self):
        """Test that add_port raises for via port without layers."""
        sim = DrivenSim()
        # PortConfig validates eagerly at creation time
        with pytest.raises(ValueError):
            sim.add_port("o1", geometry="via")  # No from_layer/to_layer

    def test_cpw_port_requires_layer(self):
        """Test validation catches CPW port without layer."""
        sim = DrivenSim()
        sim.add_cpw_port("P1", "P2", layer="", length=5.0)  # Empty layer
        result = sim.validate_config()
        assert not result.valid
        assert any("'layer' is required" in e for e in result.errors)

    def test_no_ports_warning(self):
        """Test validation warns when no ports configured."""
        sim = DrivenSim()
        result = sim.validate_config()
        # Should have warning about no ports (but this is not an error)
        assert any("No ports configured" in w for w in result.warnings)

    def test_invalid_excitation_port(self):
        """Test validation catches invalid excitation port."""
        sim = DrivenSim()
        sim.add_port("o1", layer="metal1", length=5.0)
        sim.set_driven(excitation_port="nonexistent")
        result = sim.validate_config()
        assert not result.valid
        assert any(
            "Excitation port 'nonexistent' not found" in e for e in result.errors
        )


class TestEigenSimValidation:
    """Test EigenmodeSim validation logic."""

    def test_missing_geometry(self):
        """Test validation catches missing geometry."""
        sim = EigenmodeSim()
        result = sim.validate_config()
        assert not result.valid
        assert any("No component set" in e for e in result.errors)

    def test_no_ports_is_warning_not_error(self):
        """Test that no ports is a warning, not an error for eigenmode."""
        sim = EigenmodeSim()
        result = sim.validate_config()
        # Eigenmode can work without ports (finds all modes)
        assert any("No ports configured" in w for w in result.warnings)

    def test_inplane_port_requires_layer(self):
        """Test that add_port raises for inplane port without layer."""
        sim = EigenmodeSim()
        # PortConfig validates eagerly at creation time
        with pytest.raises(ValueError):
            sim.add_port("o1", geometry="inplane")  # No layer


class TestElectrostaticSimValidation:
    """Test ElectrostaticSim validation logic."""

    def test_missing_geometry(self):
        """Test validation catches missing geometry."""
        sim = ElectrostaticSim()
        result = sim.validate_config()
        assert not result.valid
        assert any("No component set" in e for e in result.errors)

    def test_requires_two_terminals(self):
        """Test validation requires at least 2 terminals."""
        sim = ElectrostaticSim()
        sim.add_terminal("T1", layer="metal1")  # Only one terminal
        result = sim.validate_config()
        assert not result.valid
        assert any("at least 2 terminals" in e for e in result.errors)

    def test_two_terminals_valid(self):
        """Test validation passes with 2 terminals (but missing geometry)."""
        sim = ElectrostaticSim()
        sim.add_terminal("T1", layer="metal1")
        sim.add_terminal("T2", layer="metal1")
        result = sim.validate_config()
        # Still invalid due to missing geometry, but terminal count is OK
        assert any("No component set" in e for e in result.errors)
        assert not any("at least 2 terminals" in e for e in result.errors)


class TestMixinMethods:
    """Test mixin methods work on all simulation classes."""

    def test_set_output_dir(self, tmp_path):
        """Test set_output_dir works on all sim classes."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim]:
            sim = cls()
            sim.set_output_dir(tmp_path / "test")
            assert sim.output_dir == tmp_path / "test"
            assert sim.output_dir is not None
            assert sim.output_dir.exists()

    def test_set_stack(self):
        """Test set_stack works on all sim classes."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim]:
            sim = cls()
            sim.set_stack(air_above=500.0)
            assert sim._stack_kwargs["air_above"] == 500.0

    def test_set_material(self):
        """Test set_material works on all sim classes."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim]:
            sim = cls()
            sim.set_material(
                "custom_metal", material_type="conductor", conductivity=1e7
            )
            assert "custom_metal" in sim.materials
            assert sim.materials["custom_metal"].conductivity == 1e7

    def test_set_numerical(self):
        """Test set_numerical works on all sim classes."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim]:
            sim = cls()
            sim.set_numerical(order=3, tolerance=1e-8)
            assert sim.numerical.order == 3
            assert sim.numerical.tolerance == 1e-8

    def test_mesh_requires_output_dir(self):
        """Test mesh() raises if output_dir not set."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim]:
            sim = cls()
            with pytest.raises(ValueError, match="Output directory not set"):
                sim.mesh()
