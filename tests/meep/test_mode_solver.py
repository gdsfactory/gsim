"""Tests for the standalone Meep mode solver."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError


def _fake_import_error():
    raise ImportError("pymeep")


class TestModeResult:
    """ModeResult model validation — no meep required."""

    def test_minimal_construction(self):
        """ModeResult can be constructed with minimal required fields."""
        from gsim.meep.models.results import ModeResult

        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
            band_num=1,
            parity="NO_PARITY",
        )
        assert result.n_eff == 2.5
        assert result.wavelength == 1.55
        assert result.band_num == 1
        assert result.parity == "NO_PARITY"
        assert result.n_group is None
        assert result.kdom == []
        assert result.fields == {}

    def test_requires_n_eff(self):
        """ModeResult requires n_eff."""
        from gsim.meep.models.results import ModeResult

        with pytest.raises(ValidationError):
            ModeResult(
                wavelength=1.55,
                frequency=1.0 / 1.55,
                band_num=1,
                parity="NO_PARITY",
            )

    def test_requires_wavelength(self):
        """ModeResult requires wavelength."""
        from gsim.meep.models.results import ModeResult

        with pytest.raises(ValidationError):
            ModeResult(
                n_eff=2.5,
                frequency=1.0 / 1.55,
                band_num=1,
                parity="NO_PARITY",
            )

    def test_full_construction_with_fields(self):
        """ModeResult stores field arrays by component name."""
        from gsim.meep.models.results import ModeResult

        ex = np.array([[1.0, 0.0], [0.0, 1.0]])
        ey = np.array([[0.0, 1.0], [-1.0, 0.0]])
        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
            fields={"Ex": ex, "Ey": ey},
            kdom=[1.0, 0.0, 0.0],
            n_group=4.2,
            band_num=2,
            parity="EVEN_Y",
        )
        assert result.n_group == 4.2
        assert result.band_num == 2
        assert result.parity == "EVEN_Y"
        assert list(result.kdom) == [1.0, 0.0, 0.0]
        np.testing.assert_array_equal(result.fields["Ex"], ex)
        np.testing.assert_array_equal(result.fields["Ey"], ey)

    def test_defaults(self):
        """Optional fields have correct defaults."""
        from gsim.meep.models.results import ModeResult

        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
        )
        assert result.band_num == 1
        assert result.parity == "NO_PARITY"
        assert result.n_group is None
        assert result.kdom == []
        assert result.fields == {}

    def test_model_dump(self):
        """ModeResult serializes to dict including numpy arrays."""
        from gsim.meep.models.results import ModeResult

        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
            band_num=1,
            parity="NO_PARITY",
        )
        d = result.model_dump()
        assert d["n_eff"] == 2.5
        assert d["wavelength"] == 1.55
        assert d["band_num"] == 1
        assert d["parity"] == "NO_PARITY"
        assert d["n_group"] is None


class TestResolveStackAndMaterials:
    """_resolve_stack_and_materials() — no meep required."""

    def test_resolves_stack_and_materials(self):
        """Resolves a layer stack and material data from a Simulation."""
        import gdsfactory as gf

        from gsim.meep import Simulation

        c = gf.components.straight(length=10, width=0.5)

        sim = Simulation()
        sim.geometry.component = c
        sim.materials = {
            "si": 12.0,
            "SiO2": 2.1,
        }

        stack, materials = sim._resolve_stack_and_materials(wavelength=1.55)
        assert stack is not None
        assert len(stack.layers) > 0
        assert isinstance(materials, dict)
        assert len(materials) > 0

    def test_materials_contain_epsilon(self):
        """Resolved materials have epsilon_diag entries."""
        import gdsfactory as gf

        from gsim.meep import Simulation

        c = gf.components.straight(length=10, width=0.5)

        sim = Simulation()
        sim.geometry.component = c
        sim.materials = {
            "si": 12.0,
            "SiO2": 2.1,
        }

        _stack, materials = sim._resolve_stack_and_materials(wavelength=1.55)
        for name, mat in materials.items():
            assert mat.epsilon_diag is not None, f"{name} missing epsilon_diag"

    def test_wavelength_affects_dispersive_materials(self):
        """Different wavelengths produce different epsilon for dispersive materials."""
        import gdsfactory as gf

        from gsim.meep import Simulation

        c = gf.components.straight(length=10, width=0.5)

        sim = Simulation()
        sim.geometry.component = c
        sim.materials = {}

        _stack, mats_1550 = sim._resolve_stack_and_materials(wavelength=1.55)
        _stack, mats_1310 = sim._resolve_stack_and_materials(wavelength=1.31)

        # si should be present in both (from default PDK stack)
        if "si" in mats_1550 and "si" in mats_1310:
            eps_1550 = mats_1550["si"].epsilon_diag
            eps_1310 = mats_1310["si"].epsilon_diag
            # si has material dispersion — eps at 1.31 != eps at 1.55
            if isinstance(eps_1550, list):
                eps_1550 = eps_1550[0]
            if isinstance(eps_1310, list):
                eps_1310 = eps_1310[0]
            assert eps_1550 != eps_1310, (
                f"Expected different epsilon at different wavelengths, "
                f"got {eps_1550} at 1.55 and {eps_1310} at 1.31"
            )


class TestModeSolverImport:
    """Lazy import behavior — no meep required."""

    def test_solve_slab_mode_import_error_without_meep(self, monkeypatch):
        """Raises ImportError with clear message when meep not installed."""
        from gsim.meep import mode_solver

        monkeypatch.setattr(mode_solver, "_import_meep", _fake_import_error)

        with pytest.raises(ImportError, match="pymeep"):
            mode_solver.solve_slab_mode(
                stack=cast(Any, None),
                wavelength=1.55,
            )

    def test_solve_cross_section_mode_import_error_without_meep(self, monkeypatch):
        """Raises ImportError with clear message when meep not installed."""
        from gsim.meep import mode_solver

        monkeypatch.setattr(mode_solver, "_import_meep", _fake_import_error)

        with pytest.raises(ImportError, match="pymeep"):
            mode_solver.solve_cross_section_mode(
                component=cast(Any, None),
                stack=cast(Any, None),
                wavelength=1.55,
                x_span=2.0,
            )


class TestModeSolverUnit:
    """Unit tests for mode solver — no meep required."""

    def test_mode_solver_module_exports(self):
        """Public functions are importable from mode_solver module."""
        from gsim.meep.mode_solver import (
            solve_cross_section_mode,
            solve_slab_mode,
        )

        assert callable(solve_slab_mode)
        assert callable(solve_cross_section_mode)


class TestModeZGrid:
    """Unit tests for mode_z_grid — no meep required."""

    def test_z_grid_bounds_and_length(self):
        """Z-grid spans stack extent in absolute coordinates."""
        from gsim.common.stack.extractor import Layer, LayerStack
        from gsim.meep.mode_solver import mode_z_grid

        layers = {
            "box": Layer(
                name="box",
                gds_layer=(0, 0),
                zmin=-2.0,
                zmax=0.0,
                thickness=2.0,
                material="sio2",
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
        stack = LayerStack(layers=layers)
        z_grid = mode_z_grid(stack, n_points=100)

        assert len(z_grid) == 100
        assert z_grid[0] < -1.9  # near z_min = -2.0
        assert z_grid[-1] > 0.2  # near z_max = 0.22

    def test_z_grid_absolute_coords(self):
        """Z-grid values match absolute layer extents + margins."""

        from gsim.common.stack.extractor import Layer, LayerStack
        from gsim.meep.mode_solver import mode_z_grid

        layers = {
            "box": Layer(
                name="box",
                gds_layer=(0, 0),
                zmin=-2.0,
                zmax=0.0,
                thickness=2.0,
                material="sio2",
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
        stack = LayerStack(layers=layers)
        z_grid = mode_z_grid(stack, n_points=200, z_margin=(0.5, 0.5))

        z_min = -2.0
        z_max = 0.22
        zm_bottom, zm_top = 0.5, 0.5
        expected_bottom = z_min - zm_bottom
        expected_top = z_max + zm_top
        assert z_grid[0] < -2.4  # below z_min - zm_bottom
        assert z_grid[-1] > 0.6  # above z_max + zm_top
        assert abs(z_grid[0] - expected_bottom) < 0.05
        assert abs(z_grid[-1] - expected_top) < 0.05

    def test_asymmetric_z_margin_grid_coverage(self):
        """Asymmetric z_margin gives correct start/end bounds."""

        from gsim.common.stack.extractor import Layer, LayerStack
        from gsim.meep.mode_solver import mode_z_grid

        layers = {
            "box": Layer(
                name="box",
                gds_layer=(0, 0),
                zmin=-2.0,
                zmax=0.0,
                thickness=2.0,
                material="sio2",
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
        stack = LayerStack(layers=layers)
        z_grid = mode_z_grid(stack, n_points=200, z_margin=(0, 0.5))

        z_min, z_max = -2.0, 0.22
        expected_bottom = z_min  # margin_bottom = 0
        expected_top = z_max + 0.5
        assert abs(z_grid[0] - expected_bottom) < 0.05
        assert abs(z_grid[-1] - expected_top) < 0.05


class TestRefractiveIndexProfile:
    """Unit tests for refractive_index_profile — no meep required."""

    def test_profile_length_matches_grid(self):
        """Profile has same length as input z_grid."""
        import numpy as np

        from gsim.common.stack.extractor import Layer, LayerStack
        from gsim.meep.mode_solver import (
            mode_z_grid,
            refractive_index_profile,
        )

        layers = {
            "box": Layer(
                name="box",
                gds_layer=(0, 0),
                zmin=-2.0,
                zmax=0.0,
                thickness=2.0,
                material="sio2",
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
        stack = LayerStack(layers=layers)
        z_grid = mode_z_grid(stack, n_points=50)
        n_profile = refractive_index_profile(stack, 1.55, z_grid=z_grid)

        assert len(n_profile) == 50
        assert isinstance(n_profile, np.ndarray)

    def test_si_layer_higher_index(self):
        """Si core region has higher index than SiO2 cladding."""
        import numpy as np

        from gsim.common.stack.extractor import Layer, LayerStack
        from gsim.meep.mode_solver import (
            mode_z_grid,
            refractive_index_profile,
        )

        layers = {
            "box": Layer(
                name="box",
                gds_layer=(0, 0),
                zmin=-2.0,
                zmax=0.0,
                thickness=2.0,
                material="sio2",
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
        stack = LayerStack(layers=layers)
        z_grid = mode_z_grid(stack, n_points=200)
        n_profile = refractive_index_profile(stack, 1.55, z_grid=z_grid)

        core_mask = (z_grid >= 0.0) & (z_grid < 0.22)
        box_mask = ~core_mask

        assert np.mean(n_profile[core_mask]) > np.mean(n_profile[box_mask])


class TestModeXGrid:
    """Unit tests for mode_x_grid — no meep required."""

    def test_x_grid_bounds_and_length(self):
        """X-grid spans cell extent and has correct length."""

        from gsim.meep.mode_solver import mode_x_grid

        x_grid = mode_x_grid(n_points=100, x_span=4.0)

        assert len(x_grid) == 100
        assert x_grid[0] > -2.1
        assert x_grid[0] < -1.9
        assert x_grid[-1] > 1.9
        assert x_grid[-1] < 2.1

    def test_x_grid_centred(self):
        """X-grid is centred on origin (zero-mean)."""
        import numpy as np

        from gsim.meep.mode_solver import mode_x_grid

        x_grid = mode_x_grid(n_points=200, x_span=6.0)

        assert abs(np.mean(x_grid)) < 0.1


class TestSimulationSolveMode:
    """Simulation.solve_mode() wrapper tests."""

    def test_method_exists(self):
        """solve_mode method exists on Simulation."""
        import gdsfactory as gf

        from gsim.meep import Simulation

        c = gf.components.straight(length=10, width=0.5)
        sim = Simulation()
        sim.geometry.component = c

        assert hasattr(sim, "solve_mode")
        assert callable(sim.solve_mode)

    def test_requires_wavelength(self):
        """solve_mode raises TypeError without wavelength."""
        import gdsfactory as gf

        from gsim.meep import Simulation

        c = gf.components.straight(length=10, width=0.5)
        sim = Simulation()
        sim.geometry.component = c

        with pytest.raises(TypeError):
            sim.solve_mode(port="o1")

    @pytest.mark.meep_local
    def test_delegates_to_mode_solver(self):
        """solve_mode delegates to cross-section solver, returns ModeResult."""
        import gdsfactory as gf

        from gsim.meep import Simulation
        from gsim.meep.models.results import ModeResult

        c = gf.components.straight(length=10, width=0.5)
        sim = Simulation()
        sim.geometry.component = c

        result = sim.solve_mode(port="o1", y_span=2.5, wavelength=1.55)
        assert isinstance(result, ModeResult)
        assert result.n_eff > 1.0
        assert result.wavelength == 1.55


class TestModeSolverIntegration:
    """Integration tests requiring a local MEEP installation.

    These are deselected by default. Run with::

        pytest -m meep_local tests/meep/test_mode_solver.py
    """

    @pytest.mark.meep_local
    def test_solve_slab_mode_si_sio2(self):
        """solve_slab_mode returns n_eff for a simple Si/SiO2 slab."""
        from gsim.common.stack import get_stack
        from gsim.meep.mode_solver import solve_slab_mode

        stack = get_stack()
        result = solve_slab_mode(
            stack=stack,
            wavelength=1.55,
            band_num=1,
            parity="NO_PARITY",
        )
        assert result.n_eff > 1.0
        assert result.wavelength == 1.55
        assert result.band_num == 1
        assert len(result.kdom) == 3

    @pytest.mark.meep_local
    def test_solve_slab_mode_compute_group_index(self):
        """solve_slab_mode returns n_group (always computed if available)."""
        from gsim.common.stack import get_stack
        from gsim.meep.mode_solver import solve_slab_mode

        stack = get_stack()
        result = solve_slab_mode(
            stack=stack,
            wavelength=1.55,
            band_num=1,
            parity="NO_PARITY",
        )
        assert result.n_eff > 1.0
        assert result.n_group is not None
        assert result.n_group > 0.0

    @pytest.mark.meep_local
    def test_solve_slab_mode_returns_fields(self):
        """solve_slab_mode returns field profile arrays when field_z_grid is given."""

        from gsim.common.stack import get_stack
        from gsim.meep.mode_solver import mode_z_grid, solve_slab_mode

        stack = get_stack()
        z_grid = mode_z_grid(stack, n_points=135)
        result = solve_slab_mode(
            stack=stack,
            wavelength=1.55,
            field_z_grid=z_grid,
        )
        assert len(result.fields) > 0
        assert len(result.fields) >= 1
        for arr in result.fields.values():
            assert arr is not None
            assert arr.ndim == 2
            assert arr.shape[0] == len(z_grid)
            assert arr.shape[1] == 1

    @pytest.mark.meep_local
    def test_solve_cross_section_mode_straight_wg(self):
        """solve_cross_section_mode returns n_eff for a straight waveguide."""
        import gdsfactory as gf

        from gsim.common.stack import get_stack
        from gsim.meep.mode_solver import (
            mode_x_grid,
            mode_z_grid,
            solve_cross_section_mode,
        )

        c = gf.components.straight(length=10, width=0.5)
        stack = get_stack()

        x_span = 2.5
        z_grid = mode_z_grid(stack, n_points=135)
        x_grid = mode_x_grid(n_points=80, x_span=x_span)
        result = solve_cross_section_mode(
            component=c,
            stack=stack,
            port="o1",
            y_span=x_span,
            wavelength=1.55,
            band_num=1,
            parity="NO_PARITY",
            field_x_grid=x_grid,
            field_z_grid=z_grid,
        )
        assert result.n_eff > 1.0
        assert result.wavelength == 1.55
        assert len(result.fields) > 0

    @pytest.mark.meep_local
    def test_solve_cross_section_mode_position(self):
        """solve_cross_section_mode works with explicit position + x_span."""
        import gdsfactory as gf

        from gsim.common.stack import get_stack
        from gsim.meep.mode_solver import solve_cross_section_mode

        c = gf.components.straight(length=10, width=0.5)
        stack = get_stack()

        result = solve_cross_section_mode(
            component=c,
            stack=stack,
            position=(5.0, 0.0),
            x_span=2.0,
            wavelength=1.55,
        )
        assert result.n_eff > 1.0

    @pytest.mark.meep_local
    def test_simulation_solve_mode_slab(self):
        """Simulation.solve_mode() without port falls back to slab mode."""
        from gsim.common.stack import get_stack
        from gsim.meep import Simulation

        stack = get_stack()
        sim = Simulation()
        sim.geometry.stack = stack

        result = sim.solve_mode(wavelength=1.55)
        assert result.n_eff > 1.0

    @pytest.mark.meep_local
    def test_simulation_solve_mode_cross_section(self):
        """Simulation.solve_mode() with port delegates to cross-section mode."""
        import gdsfactory as gf

        from gsim.common.stack import get_stack
        from gsim.meep import Simulation

        c = gf.components.straight(length=10, width=0.5)
        stack = get_stack()

        sim = Simulation()
        sim.geometry.component = c
        sim.geometry.stack = stack

        result = sim.solve_mode(port="o1", y_span=2.5, wavelength=1.55)
        assert result.n_eff > 1.0
