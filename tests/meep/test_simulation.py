"""Tests for the declarative Simulation API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gsim.meep import (
    FDTD,
    Domain,
    Geometry,
    Material,
    ModeSource,
    Simulation,
    Symmetry,
)

# ---------------------------------------------------------------------------
# Model construction & defaults
# ---------------------------------------------------------------------------


class TestMaterial:
    """Tests for the Material model."""

    def test_basic(self):
        m = Material(n=3.47)
        assert m.n == 3.47
        assert m.k == 0.0

    def test_with_extinction(self):
        m = Material(n=3.47, k=0.01)
        assert m.k == 0.01

    def test_n_must_be_positive(self):
        with pytest.raises(ValidationError):
            Material(n=0)

    def test_k_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            Material(n=1.5, k=-0.1)


class TestModeSource:
    """Tests for the ModeSource model."""

    def test_defaults(self):
        s = ModeSource()
        assert s.port is None
        assert s.wavelength == 1.55
        assert s.wavelength_span == 0.1
        assert s.num_freqs == 11

    def test_custom(self):
        s = ModeSource(port="o1", wavelength=1.31, wavelength_span=0.05, num_freqs=21)
        assert s.port == "o1"
        assert s.wavelength == 1.31
        assert s.wavelength_span == 0.05
        assert s.num_freqs == 21


class TestDomain:
    """Tests for the Domain model."""

    def test_defaults(self):
        d = Domain()
        assert d.pml == 1.0
        assert d.margin == 0.5
        assert d.margin_z_above == 0.5
        assert d.margin_z_below == 0.5
        assert d.port_margin == 0.5
        assert d.extend_ports == 0.0
        assert d.source_port_offset == 0.1
        assert d.distance_source_to_monitors == 0.2
        assert d.symmetries == []

    def test_custom(self):
        d = Domain(pml=0.5, margin=0.2, port_margin=0.3)
        assert d.pml == 0.5
        assert d.margin == 0.2
        assert d.port_margin == 0.3

    def test_symmetries(self):
        d = Domain(symmetries=[Symmetry(direction="Y", phase=-1)])
        assert len(d.symmetries) == 1
        assert d.symmetries[0].direction == "Y"
        assert d.symmetries[0].phase == -1


class TestStoppingFields:
    """Tests for the FDTD stopping criteria."""

    def test_defaults(self):
        f = FDTD()
        assert f.stopping == "field_decay"
        assert f.max_time == 2000.0
        assert f.stopping_threshold == 0.05
        assert f.stopping_min_time == 100.0
        assert f.stopping_component == "Ey"
        assert f.stopping_dt == 50.0
        assert f.stopping_monitor_port is None

    def test_fixed_mode(self):
        f = FDTD(stopping="fixed", max_time=100)
        assert f.stopping == "fixed"
        assert f.max_time == 100

    def test_field_decay_mode(self):
        f = FDTD(stopping="field_decay", stopping_threshold=1e-4, stopping_dt=25)
        assert f.stopping == "field_decay"
        assert f.stopping_threshold == 1e-4
        assert f.stopping_dt == 25

    def test_dft_decay_custom(self):
        f = FDTD(
            stopping="dft_decay",
            max_time=300,
            stopping_threshold=1e-4,
            stopping_min_time=80,
        )
        assert f.stopping == "dft_decay"
        assert f.max_time == 300
        assert f.stopping_threshold == 1e-4
        assert f.stopping_min_time == 80

    def test_energy_decay_mode(self):
        f = FDTD(stopping="energy_decay", stopping_dt=100, stopping_threshold=1e-4)
        assert f.stopping == "energy_decay"
        assert f.stopping_dt == 100
        assert f.stopping_threshold == 1e-4

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            FDTD(stopping="invalid")  # ty: ignore[invalid-argument-type]


class TestFDTD:
    """Tests for the FDTD model."""

    def test_defaults(self):
        f = FDTD()
        assert f.resolution == 32
        assert f.stopping == "field_decay"
        assert f.max_time == 2000.0
        assert f.subpixel is False
        assert f.simplify_tol == 0.0

    def test_with_custom_stopping(self):
        f = FDTD(stopping="field_decay", stopping_threshold=1e-4)
        assert f.stopping == "field_decay"
        assert f.stopping_threshold == 1e-4

    def test_resolution_custom(self):
        f = FDTD(resolution=64)
        assert f.resolution == 64


class TestFDTDDiagnostics:
    """Tests for the FDTD diagnostics settings."""

    def test_defaults(self):
        f = FDTD()
        assert f.save_geometry is True
        assert f.save_fields is True
        assert f.save_animation is False
        assert f.preview_only is False
        assert f.verbose_interval == 0

    def test_custom(self):
        f = FDTD(save_animation=True, verbose_interval=5.0, preview_only=True)
        assert f.save_animation is True
        assert f.verbose_interval == 5.0
        assert f.preview_only is True


# ---------------------------------------------------------------------------
# Material normalization
# ---------------------------------------------------------------------------


class TestMaterialNormalization:
    """Tests for material shorthand and normalization."""

    def test_float_shorthand(self):
        sim = Simulation(materials={"si": 3.47, "sio2": 1.44})
        assert isinstance(sim.materials["si"], Material)
        assert sim.materials["si"].n == 3.47
        assert isinstance(sim.materials["sio2"], Material)
        assert sim.materials["sio2"].n == 1.44

    def test_material_object(self):
        sim = Simulation(materials={"si": Material(n=3.47, k=0.01)})
        assert sim.materials["si"].n == 3.47  # ty: ignore[unresolved-attribute]
        assert sim.materials["si"].k == 0.01  # ty: ignore[unresolved-attribute]

    def test_dict_shorthand(self):
        sim = Simulation(materials={"si": {"n": 3.47, "k": 0.01}})  # ty: ignore[invalid-argument-type]
        assert isinstance(sim.materials["si"], Material)
        assert sim.materials["si"].n == 3.47

    def test_assignment_normalization(self):
        sim = Simulation()
        sim.materials = {"si": 3.47}
        assert isinstance(sim.materials["si"], Material)
        assert sim.materials["si"].n == 3.47


# ---------------------------------------------------------------------------
# Field-by-field assignment
# ---------------------------------------------------------------------------


class TestFieldAssignment:
    """Tests for field-by-field assignment in Simulation."""

    def test_source_port(self):
        sim = Simulation()
        sim.source.port = "o1"
        assert sim.source.port == "o1"

    def test_source_wavelength(self):
        sim = Simulation()
        sim.source.wavelength = 1.31
        assert sim.source.wavelength == 1.31

    def test_domain_pml(self):
        sim = Simulation()
        sim.domain.pml = 0.5
        assert sim.domain.pml == 0.5

    def test_solver_resolution(self):
        sim = Simulation()
        sim.solver.resolution = 64
        assert sim.solver.resolution == 64

    def test_solver_stopping_replace(self):
        sim = Simulation()
        sim.solver.stopping = "field_decay"
        sim.solver.stopping_threshold = 1e-4
        assert sim.solver.stopping == "field_decay"
        assert sim.solver.stopping_threshold == 1e-4

    def test_stop_when_energy_decayed(self):
        f = FDTD()
        result = f.stop_when_energy_decayed(dt=100, decay_by=1e-4)
        assert result is f
        assert f.stopping == "energy_decay"
        assert f.stopping_dt == 100
        assert f.stopping_threshold == 1e-4

    def test_stop_when_dft_decayed(self):
        f = FDTD()
        result = f.stop_when_dft_decayed(tol=1e-4, min_time=200)
        assert result is f
        assert f.stopping == "dft_decay"
        assert f.stopping_threshold == 1e-4
        assert f.stopping_min_time == 200

    def test_stop_when_fields_decayed(self):
        f = FDTD()
        result = f.stop_when_fields_decayed(
            dt=25, component="Hz", decay_by=1e-4, monitor_port="o2"
        )
        assert result is f
        assert f.stopping == "field_decay"
        assert f.stopping_dt == 25
        assert f.stopping_component == "Hz"
        assert f.stopping_threshold == 1e-4
        assert f.stopping_monitor_port == "o2"

    def test_stop_after_sources(self):
        f = FDTD()
        result = f.stop_after_sources(time=500)
        assert result is f
        assert f.stopping == "fixed"
        assert f.max_time == 500

    def test_wall_time_max_default(self):
        f = FDTD()
        assert f.wall_time_max == 0.0

    def test_stop_after_walltime(self):
        f = FDTD()
        result = f.stop_after_walltime(seconds=3600)
        assert result is f
        assert f.wall_time_max == 3600

    def test_wall_time_max_orthogonal(self):
        """wall_time_max can be combined with any stopping mode."""
        f = FDTD()
        f.stop_when_fields_decayed(dt=50, decay_by=0.05)
        f.stop_after_walltime(seconds=1800)
        assert f.stopping == "field_decay"
        assert f.wall_time_max == 1800

    def test_geometry_component(self):
        sim = Simulation()
        sim.geometry.component = "placeholder"
        assert sim.geometry.component == "placeholder"


class TestCallableAPI:
    """Tests for the callable (method-like) API in Simulation."""

    def test_source_callable(self):
        sim = Simulation()
        result = sim.source(port="o1", wavelength=1.31, wavelength_span=0.05)
        assert result is sim.source
        assert sim.source.port == "o1"
        assert sim.source.wavelength == 1.31
        assert sim.source.wavelength_span == 0.05

    def test_domain_callable(self):
        sim = Simulation()
        sim.domain(pml=0.5, margin=0.2)
        assert sim.domain.pml == 0.5
        assert sim.domain.margin == 0.2

    def test_geometry_callable(self):
        sim = Simulation()
        sim.geometry(component="placeholder", z_crop="auto")
        assert sim.geometry.component == "placeholder"
        assert sim.geometry.z_crop == "auto"

    def test_solver_callable(self):
        sim = Simulation()
        sim.solver(resolution=64, simplify_tol=0.01)
        assert sim.solver.resolution == 64
        assert sim.solver.simplify_tol == 0.01

    def test_callable_validation(self):
        sim = Simulation()
        with pytest.raises(ValidationError):
            sim.solver(resolution=1)  # ge=4

    def test_callable_invalid_field(self):
        sim = Simulation()
        with pytest.raises(ValidationError):
            sim.source(nonexistent_field="value")


class TestWholeObjectAssignment:
    """Tests for whole object assignment in Simulation."""

    def test_source(self):
        sim = Simulation()
        sim.source = ModeSource(port="o2", wavelength=1.31)
        assert sim.source.port == "o2"
        assert sim.source.wavelength == 1.31

    def test_domain(self):
        sim = Simulation()
        sim.domain = Domain(pml=0.5, margin=0.2)
        assert sim.domain.pml == 0.5
        assert sim.domain.margin == 0.2

    def test_solver(self):
        sim = Simulation()
        sim.solver = FDTD(resolution=64, stopping="dft_decay")
        assert sim.solver.resolution == 64
        assert sim.solver.stopping == "dft_decay"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for simulation configuration validation."""

    def test_missing_component(self):
        sim = Simulation()
        result = sim.validate_config()
        assert not result.valid
        assert any("No component" in e for e in result.errors)

    def test_no_stack_warning(self):
        sim = Simulation()
        result = sim.validate_config()
        assert any("No stack" in w for w in result.warnings)

    def test_string_monitors(self):
        sim = Simulation()
        sim.monitors = ["o1", "o2", "o3"]
        assert sim.monitors == ["o1", "o2", "o3"]

    def test_write_config_requires_output_dir(self):
        sim = Simulation()
        with pytest.raises(TypeError):
            sim.write_config()


# ---------------------------------------------------------------------------
# Wavelength derivation
# ---------------------------------------------------------------------------


class TestWavelengthDerivation:
    """Tests for wavelength configuration derivation."""

    def test_from_source_defaults(self):
        sim = Simulation()
        wl = sim._wavelength_config()
        assert wl.wavelength == 1.55
        assert wl.bandwidth == 0.1
        assert wl.num_freqs == 11

    def test_from_source_custom(self):
        sim = Simulation()
        sim.source = ModeSource(wavelength=1.31, wavelength_span=0.05, num_freqs=21)
        wl = sim._wavelength_config()
        assert wl.wavelength == 1.31
        assert wl.bandwidth == 0.05
        assert wl.num_freqs == 21


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------


class TestConfigTranslation:
    """Tests for translation to low-level config objects."""

    def test_stopping_fixed(self):
        sim = Simulation()
        sim.solver.stopping = "fixed"
        sim.solver.max_time = 200
        cfg = sim._stopping_config()
        assert cfg.mode == "fixed"
        assert cfg.max_time == 200

    def test_stopping_field_decay(self):
        sim = Simulation()
        sim.solver.stopping = "field_decay"
        sim.solver.stopping_threshold = 1e-4
        sim.solver.stopping_dt = 25
        sim.solver.stopping_monitor_port = "o2"
        cfg = sim._stopping_config()
        assert cfg.mode == "field_decay"
        assert cfg.threshold == 1e-4
        assert cfg.decay_dt == 25
        assert cfg.decay_monitor_port == "o2"

    def test_stopping_dft_decay(self):
        sim = Simulation()
        sim.solver.stopping = "dft_decay"
        sim.solver.max_time = 200
        sim.solver.stopping_threshold = 1e-4
        sim.solver.stopping_min_time = 80
        cfg = sim._stopping_config()
        assert cfg.mode == "dft_decay"
        assert cfg.max_time == 200
        assert cfg.threshold == 1e-4
        assert cfg.dft_min_run_time == 80

    def test_stopping_energy_decay(self):
        sim = Simulation()
        sim.solver.stop_when_energy_decayed(dt=100, decay_by=1e-4)
        cfg = sim._stopping_config()
        assert cfg.mode == "energy_decay"
        assert cfg.decay_dt == 100
        assert cfg.threshold == 1e-4

    def test_domain_translation(self):
        sim = Simulation()
        sim.domain = Domain(pml=0.5, margin=0.3, margin_z_above=1.0, margin_z_below=0.8)
        cfg = sim._domain_config()
        assert cfg.dpml == 0.5
        assert cfg.margin_xy == 0.3
        assert cfg.margin_z_above == 1.0
        assert cfg.margin_z_below == 0.8
        assert cfg.source_port_offset == 0.1
        assert cfg.distance_source_to_monitors == 0.2

    def test_resolution_translation(self):
        sim = Simulation()
        sim.solver.resolution = 64
        cfg = sim._resolution_config()
        assert cfg.pixels_per_um == 64

    def test_accuracy_translation(self):
        sim = Simulation()
        sim.solver.subpixel = True
        sim.solver.simplify_tol = 0.01
        cfg = sim._accuracy_config()
        assert cfg.eps_averaging is True
        assert cfg.simplify_tol == 0.01

    def test_source_translation(self):
        sim = Simulation()
        sim.source = ModeSource(port="o1")
        cfg = sim._source_config()
        assert cfg.port == "o1"
        assert cfg.bandwidth is None  # always auto fwidth

    def test_diagnostics_translation(self):
        sim = Simulation()
        sim.solver.save_animation = True
        sim.solver.preview_only = True
        cfg = sim._diagnostics_config()
        assert cfg.save_animation is True
        assert cfg.preview_only is True


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Tests for importing the new declarative API."""

    def test_import_all_new_api(self):
        from gsim.meep import (
            FDTD,
            Domain,
            Material,
            ModeSource,
            Simulation,
            Symmetry,
        )

        assert all(
            cls is not None
            for cls in [
                Domain,
                FDTD,
                Geometry,
                Material,
                ModeSource,
                Simulation,
                Symmetry,
            ]
        )

    def test_simulation_instantiation(self):
        sim = Simulation()
        assert sim.geometry.component is None
        assert sim.materials == {}
        assert sim.monitors == []
        assert sim.solver.resolution == 32
