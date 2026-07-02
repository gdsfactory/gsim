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

    def test_new_grid_fields_default_to_none(self):
        """x_grid, y_grid, z_grid default to None."""
        from gsim.meep.models.results import ModeResult

        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
        )
        assert result.x_grid is None
        assert result.y_grid is None
        assert result.z_grid is None

    def test_new_context_fields_default(self):
        """stack, component, port_or_position, cross_section_plane default."""
        from gsim.meep.models.results import ModeResult

        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
        )
        assert result.stack is None
        assert result.component is None
        assert result.port_or_position is None
        assert result.cross_section_plane is None

    def test_grid_fields_stored(self):
        """x_grid, y_grid, z_grid are stored when provided."""
        from gsim.meep.models.results import ModeResult

        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        z = np.array([5.0, 6.0])
        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
            x_grid=x,
            y_grid=y,
            z_grid=z,
            cross_section_plane="xz",
        )
        np.testing.assert_array_equal(result.x_grid, x)
        np.testing.assert_array_equal(result.y_grid, y)
        np.testing.assert_array_equal(result.z_grid, z)
        assert result.cross_section_plane == "xz"

    def test_model_dump_excludes_stack_and_component(self):
        """stack and component are excluded from model_dump."""
        from unittest.mock import MagicMock

        from gsim.meep.models.results import ModeResult

        mock_stack = MagicMock()
        mock_comp = MagicMock()
        result = ModeResult(
            n_eff=2.5,
            wavelength=1.55,
            frequency=1.0 / 1.55,
            stack=mock_stack,
            component=mock_comp,
        )
        d = result.model_dump()
        assert "stack" not in d
        assert "component" not in d
        assert "n_eff" in d


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


class TestModeSolver:
    """ModeSolver model validation — no meep required."""

    def test_default_wavelengths_empty(self):
        """ModeSolver defaults to empty wavelengths list."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver()
        assert ms.wavelengths == []

    def test_minimal_construction(self):
        """ModeSolver with wavelengths uses correct defaults."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55])
        assert ms.wavelengths == [1.55]
        assert ms.where == "auto"
        assert ms.num_bands == 1
        assert ms.band is None
        assert ms.parity == "NO_PARITY"
        assert ms.eigensolver_tol == 1e-6
        assert ms.port is None
        assert ms.position is None
        assert ms.x_span is None
        assert ms.y_span is None
        assert ms.n_field_x == 0
        assert ms.n_field_y == 0
        assert ms.n_field_z == 0

    def test_slab_forbids_port(self):
        """where='slab' forbids port."""
        from gsim.meep.models.api import ModeSolver

        with pytest.raises(ValidationError, match="slab"):
            ModeSolver(wavelengths=[1.55], where="slab", port="o1")

    def test_slab_forbids_position(self):
        """where='slab' forbids position."""
        from gsim.meep.models.api import ModeSolver

        with pytest.raises(ValidationError, match="slab"):
            ModeSolver(wavelengths=[1.55], where="slab", position=(0, 0))

    def test_slab_forbids_x_span(self):
        """where='slab' forbids x_span."""
        from gsim.meep.models.api import ModeSolver

        with pytest.raises(ValidationError, match="slab"):
            ModeSolver(wavelengths=[1.55], where="slab", x_span=5.0)

    def test_slab_forbids_y_span(self):
        """where='slab' forbids y_span."""
        from gsim.meep.models.api import ModeSolver

        with pytest.raises(ValidationError, match="slab"):
            ModeSolver(wavelengths=[1.55], where="slab", y_span=5.0)

    def test_cross_section_without_port_or_position_is_allowed(self):
        """where='cross_section' without port/position allowed at model level.

        Port/position validation for cross-section is deferred to
        solve_modes() call time (Issue 2).
        """
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55], where="cross_section")
        assert ms.where == "cross_section"
        assert ms.port is None
        assert ms.position is None

    def test_cross_section_allows_port(self):
        """where='cross_section' with port is valid."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55], where="cross_section", port="o1")
        assert ms.where == "cross_section"
        assert ms.port == "o1"

    def test_cross_section_allows_position(self):
        """where='cross_section' with position is valid."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55], where="cross_section", position=(0, 0))
        assert ms.where == "cross_section"
        assert ms.position == (0, 0)

    def test_cross_section_forbids_both_port_and_position(self):
        """where='cross_section' forbids both port and position."""
        from gsim.meep.models.api import ModeSolver

        with pytest.raises(ValidationError, match="port"):
            ModeSolver(
                wavelengths=[1.55],
                where="cross_section",
                port="o1",
                position=(0, 0),
            )

    def test_parity_rejects_invalid(self):
        """parity rejects invalid strings."""
        from gsim.meep.models.api import ModeSolver

        with pytest.raises(ValidationError):
            ModeSolver(wavelengths=[1.55], parity="EVEN_Z")  # ty: ignore[invalid-argument-type]

    def test_fundamental_sets_bands(self):
        """fundamental() sets num_bands=1, band=None."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55], num_bands=5, band=3)
        result = ms.fundamental()
        assert result is ms
        assert ms.num_bands == 1
        assert ms.band is None

    def test_first_sets_num_bands(self):
        """first(n) sets num_bands=n, band=None."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55], band=3)
        result = ms.first(4)
        assert result is ms
        assert ms.num_bands == 4
        assert ms.band is None

    def test_call_sets_band(self):
        """__call__(band=n) sets band."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55])
        result = ms(band=2)
        assert result is ms
        assert ms.band == 2

    def test_at_port_sets_port(self):
        """at_port(name) sets port."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55])
        result = ms.at_port("o2")
        assert result is ms
        assert ms.port == "o2"

    def test_at_slab_sets_where(self):
        """at_slab() sets where='slab'."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55])
        result = ms.at_slab()
        assert result is ms
        assert ms.where == "slab"

    def test_at_cross_section_sets_where(self):
        """at_cross_section() sets where='cross_section'."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55])
        result = ms.at_cross_section()
        assert result is ms
        assert ms.where == "cross_section"

    def test_sweep_wavelength_populates(self):
        """sweep_wavelength(start, stop, num) populates wavelengths."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55])
        result = ms.sweep_wavelength(1.50, 1.60, 5)
        assert result is ms
        assert len(ms.wavelengths) == 5
        assert ms.wavelengths[0] == 1.50
        assert ms.wavelengths[-1] == 1.60

    def test_call_updates_fields(self):
        """__call__ updates fields in place, returns self."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55])
        result = ms(num_bands=3, port="o1")
        assert result is ms
        assert ms.num_bands == 3
        assert ms.port == "o1"

    def test_fluent_chaining(self):
        """Fluent chaining produces correct field values."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55]).first(3).at_port("o1")
        assert ms.num_bands == 3
        assert ms.band is None
        assert ms.port == "o1"

    def test_band_zero_raises(self):
        """band=0 raises ValidationError — bands are 1-indexed."""
        from gsim.meep.models.api import ModeSolver

        with pytest.raises(ValidationError):
            ModeSolver(wavelengths=[1.55], band=0)

    def test_num_bands_zero_raises(self):
        """num_bands=0 raises ValidationError — must be at least 1."""
        from gsim.meep.models.api import ModeSolver

        with pytest.raises(ValidationError):
            ModeSolver(wavelengths=[1.55], num_bands=0)

    def test_port_and_position_both_set_raises(self):
        """Both port and position set raises ValidationError."""
        from gsim.meep.models.api import ModeSolver

        with pytest.raises(ValidationError, match="port"):
            ModeSolver(wavelengths=[1.55], port="o1", position=(0, 0))

    def test_sweep_wavelength_single_point(self):
        """sweep_wavelength with num=1 produces a single-element wavelengths list."""
        from gsim.meep.models.api import ModeSolver

        ms = ModeSolver(wavelengths=[1.55])
        ms.sweep_wavelength(1.50, 1.60, 1)
        assert len(ms.wavelengths) == 1
        assert ms.wavelengths[0] == 1.50


class TestModeSweepResult:
    """ModeSweepResult wrapper tests — no meep required."""

    @staticmethod
    def _make_result(
        n_eff=2.5, wavelength=1.55, frequency=1.0, band_num=1, parity="NO_PARITY"
    ):
        from gsim.meep.models.results import ModeResult

        return ModeResult(
            n_eff=n_eff,
            wavelength=wavelength,
            frequency=frequency,
            band_num=band_num,
            parity=parity,
        )

    def test_construction(self):
        """ModeSweepResult stores list of ModeResult."""
        from gsim.meep.results import ModeSweepResult

        r1 = self._make_result(n_eff=2.5, wavelength=1.55)
        r2 = self._make_result(n_eff=2.3, wavelength=1.60)
        result = ModeSweepResult([r1, r2])
        assert len(result.results) == 2
        assert result.results[0] is r1

    def test_at_filters_by_wavelength(self):
        """at(wavelength) returns new ModeSweepResult filtered to that wavelength."""
        from gsim.meep.results import ModeSweepResult

        r1 = self._make_result(n_eff=2.5, wavelength=1.55)
        r2 = self._make_result(n_eff=2.3, wavelength=1.60)
        result = ModeSweepResult([r1, r2])
        filtered = result.at(1.55)
        assert isinstance(filtered, ModeSweepResult)
        assert len(filtered.results) == 1
        assert filtered.results[0].n_eff == 2.5

    def test_at_no_match_returns_empty(self):
        """at(wavelength) with no match returns empty ModeSweepResult."""
        from gsim.meep.results import ModeSweepResult

        r1 = self._make_result(n_eff=2.5, wavelength=1.55)
        result = ModeSweepResult([r1])
        filtered = result.at(1.60)
        assert len(filtered.results) == 0

    def test_band_lookup(self):
        """band(n) returns first ModeResult matching band_num."""
        from gsim.meep.results import ModeSweepResult

        r1 = self._make_result(n_eff=2.5, wavelength=1.55, band_num=1)
        r2 = self._make_result(n_eff=2.3, wavelength=1.55, band_num=2)
        result = ModeSweepResult([r1, r2])
        found = result.band(2)
        assert found is not None
        assert found.n_eff == 2.3
        assert found.band_num == 2

    def test_band_no_match_returns_none(self):
        """band(n) returns None when no match."""
        from gsim.meep.results import ModeSweepResult

        r1 = self._make_result(n_eff=2.5, wavelength=1.55, band_num=1)
        result = ModeSweepResult([r1])
        assert result.band(5) is None

    def test_to_dict_format(self):
        """to_dict() returns dict of lists."""
        from gsim.meep.results import ModeSweepResult

        r1 = self._make_result(n_eff=2.5, wavelength=1.55, band_num=1)
        r2 = self._make_result(n_eff=2.3, wavelength=1.60, band_num=1)
        result = ModeSweepResult([r1, r2])
        d = result.to_dict()
        assert "wavelength" in d
        assert "band_num" in d
        assert "n_eff" in d
        assert "n_group" in d
        assert "parity" in d
        assert d["wavelength"] == [1.55, 1.60]
        assert d["n_eff"] == [2.5, 2.3]
        assert d["band_num"] == [1, 1]
        assert d["n_group"] == [None, None]

    def test_at_chaining(self):
        """at(wavelength) can be chained for further queries."""
        from gsim.meep.results import ModeSweepResult

        r1 = self._make_result(n_eff=2.5, wavelength=1.55, band_num=1)
        r2 = self._make_result(n_eff=2.3, wavelength=1.55, band_num=2)
        result = ModeSweepResult([r1, r2])
        found = result.at(1.55).band(1)
        assert found is not None
        assert found.n_eff == 2.5

    def test_empty_at_returns_empty_wrapper(self):
        """at() on empty ModeSweepResult returns empty wrapper."""
        from gsim.meep.results import ModeSweepResult

        result = ModeSweepResult([])
        filtered = result.at(1.55)
        assert isinstance(filtered, ModeSweepResult)
        assert len(filtered.results) == 0

    def test_empty_band_returns_none(self):
        """band() on empty ModeSweepResult returns None."""
        from gsim.meep.results import ModeSweepResult

        result = ModeSweepResult([])
        assert result.band(1) is None

    def test_empty_to_dict_empty_lists(self):
        """to_dict() on empty ModeSweepResult returns dict of empty lists."""
        from gsim.meep.results import ModeSweepResult

        result = ModeSweepResult([])
        d = result.to_dict()
        assert d["wavelength"] == []
        assert d["n_eff"] == []
        assert d["band_num"] == []


class TestSimulationSolveMode:
    """Simulation.solve_mode() replaced by solve_modes()."""

    def test_solve_mode_gone(self):
        """solve_mode is removed from Simulation."""
        import gdsfactory as gf

        from gsim.meep import Simulation

        c = gf.components.straight(length=10, width=0.5)
        sim = Simulation()
        sim.geometry.component = c

        assert not hasattr(sim, "solve_mode")


class TestSimulationModeSolverField:
    """Simulation.mode_solver field — no meep required."""

    def test_mode_solver_field_exists(self):
        """sim.mode_solver exists and is a ModeSolver instance by default."""
        from gsim.meep import Simulation
        from gsim.meep.models.api import ModeSolver

        sim = Simulation()
        assert hasattr(sim, "mode_solver")
        assert isinstance(sim.mode_solver, ModeSolver)

    def test_mode_solver_configurable(self):
        """sim.mode_solver can be configured and re-read."""
        from gsim.meep import Simulation

        sim = Simulation()
        sim.mode_solver(wavelengths=[1.55], num_bands=3, port="o1")
        assert sim.mode_solver.wavelengths == [1.55]
        assert sim.mode_solver.num_bands == 3
        assert sim.mode_solver.port == "o1"


class TestSimulationSolveModes:
    """Simulation.solve_modes() tests — no meep required for error paths."""

    def test_solve_modes_exists(self):
        """sim.solve_modes exists and is callable."""
        from gsim.meep import Simulation

        sim = Simulation()
        assert hasattr(sim, "solve_modes")
        assert callable(sim.solve_modes)

    def test_solve_modes_requires_wavelengths(self):
        """sim.solve_modes raises ValueError when wavelengths is empty."""
        from gsim.meep import Simulation

        sim = Simulation()
        sim.mode_solver.wavelengths = []

        with pytest.raises(ValueError, match="wavelengths"):
            sim.solve_modes()

    def test_solve_modes_cross_section_no_component(self):
        """sim.solve_modes raises ValueError when cross_section but no component."""
        from gsim.meep import Simulation

        sim = Simulation()
        sim.mode_solver(wavelengths=[1.55], where="cross_section", port="o1")

        with pytest.raises(ValueError, match="component"):
            sim.solve_modes()

    def test_solve_modes_cross_section_no_port_or_position(self):
        """solve_modes raises ValueError for cross_section without port/position."""
        import gdsfactory as gf

        from gsim.meep import Simulation

        c = gf.components.straight(length=10, width=0.5)
        sim = Simulation()
        sim.geometry.component = c
        sim.mode_solver(wavelengths=[1.55], where="cross_section")

        with pytest.raises(ValueError, match="port"):
            sim.solve_modes()

    def test_band_overrides_num_bands(self):
        """band field overrides num_bands — correct band_nums computed."""
        import gdsfactory as gf

        from gsim.meep import Simulation

        c = gf.components.straight(length=10, width=0.5)
        sim = Simulation()
        sim.geometry.component = c
        sim.mode_solver(wavelengths=[1.55], num_bands=3, band=2)

        assert sim.mode_solver.band == 2
        assert sim.mode_solver.num_bands == 3


class TestSimulationSolveModesIntegration:
    """Simulation.solve_modes() integration tests (meep_local)."""

    @pytest.mark.meep_local
    def test_solve_modes_slab_single(self):
        """solve_modes() slab: single wavelength, single band."""
        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        stack = get_stack()
        sim = Simulation()
        sim.geometry.stack = stack
        sim.mode_solver(wavelengths=[1.55])

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) >= 1
        assert result.results[0].n_eff > 1.0

    @pytest.mark.meep_local
    def test_solve_modes_multi_band_slab(self):
        """solve_modes() multi-band slab: num_bands=3, single wavelength."""
        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        stack = get_stack()
        sim = Simulation()
        sim.geometry.stack = stack
        sim.mode_solver(wavelengths=[1.55], num_bands=3)

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) == 3
        for r in result.results:
            assert r.n_eff > 1.0

    @pytest.mark.meep_local
    def test_solve_modes_multi_wavelength_slab(self):
        """solve_modes() multi-wavelength slab: 2 wavelengths, single band."""
        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        stack = get_stack()
        sim = Simulation()
        sim.geometry.stack = stack
        sim.mode_solver(wavelengths=[1.50, 1.55])

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) == 2
        n_effs = [r.n_eff for r in result.results]
        assert n_effs[0] > n_effs[1]

    @pytest.mark.meep_local
    def test_solve_modes_cross_section_at_port(self):
        """solve_modes() cross-section at port."""
        import gdsfactory as gf

        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        c = gf.components.straight(length=10, width=0.5)
        stack = get_stack()
        sim = Simulation()
        sim.geometry.component = c
        sim.geometry.stack = stack
        sim.mode_solver(wavelengths=[1.55], port="o1", y_span=2.5)

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) >= 1
        assert result.results[0].n_eff > 1.0

    @pytest.mark.meep_local
    def test_solve_modes_cross_section_at_position(self):
        """solve_modes() cross-section at position."""
        import gdsfactory as gf

        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        c = gf.components.straight(length=10, width=0.5)
        stack = get_stack()
        sim = Simulation()
        sim.geometry.component = c
        sim.geometry.stack = stack
        sim.mode_solver(wavelengths=[1.55], position=(5.0, 0.0), x_span=4.0)

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) >= 1

    @pytest.mark.meep_local
    def test_solve_modes_with_fields(self):
        """solve_modes() with n_field_z returns populated fields."""
        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        stack = get_stack()
        sim = Simulation()
        sim.geometry.stack = stack
        sim.mode_solver(wavelengths=[1.55], n_field_z=200)

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) >= 1
        assert result.results[0].n_eff > 1.0
        assert len(result.results[0].fields) > 0

    @pytest.mark.meep_local
    def test_solve_modes_first_at_port(self):
        """mode_solver.first(3).at_port('o1') -> 3 results."""
        import gdsfactory as gf

        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        c = gf.components.straight(length=10, width=0.5)
        stack = get_stack()
        sim = Simulation()
        sim.geometry.component = c
        sim.geometry.stack = stack
        sim.mode_solver.first(3).at_port("o1")
        sim.mode_solver(wavelengths=[1.55], y_span=2.5)

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) == 3

    @pytest.mark.meep_local
    def test_solve_modes_uses_solver_resolution(self):
        """solve_modes() reads resolution from sim.solver."""
        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        stack = get_stack()
        sim = Simulation()
        sim.geometry.stack = stack
        sim.solver.resolution = 16
        sim.mode_solver(wavelengths=[1.55])

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) >= 1
        assert result.results[0].n_eff > 1.0

    @pytest.mark.meep_local
    def test_solve_modes_cartesian_product(self):
        """solve_modes() slab: 2 wavelengths x 2 bands = 4 results."""
        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        stack = get_stack()
        sim = Simulation()
        sim.geometry.stack = stack
        sim.mode_solver(wavelengths=[1.50, 1.55], num_bands=2)

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) == 4

        wavelengths = sorted({r.wavelength for r in result.results})
        assert wavelengths == [1.50, 1.55]

        band_nums = sorted({r.band_num for r in result.results})
        assert band_nums == [1, 2]


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
    def test_simulation_solve_modes_slab(self):
        """Simulation.solve_modes() slab mode without port."""
        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        stack = get_stack()
        sim = Simulation()
        sim.geometry.stack = stack
        sim.mode_solver(wavelengths=[1.55])

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) >= 1
        assert result.results[0].n_eff > 1.0

    @pytest.mark.meep_local
    def test_simulation_solve_modes_cross_section(self):
        """Simulation.solve_modes() with port -> cross-section mode."""
        import gdsfactory as gf

        from gsim.common.stack import get_stack
        from gsim.meep import Simulation
        from gsim.meep.results import ModeSweepResult

        c = gf.components.straight(length=10, width=0.5)
        stack = get_stack()

        sim = Simulation()
        sim.geometry.component = c
        sim.geometry.stack = stack
        sim.mode_solver(wavelengths=[1.55], port="o1", y_span=2.5)

        result = sim.solve_modes()
        assert isinstance(result, ModeSweepResult)
        assert len(result.results) >= 1
        assert result.results[0].n_eff > 1.0
