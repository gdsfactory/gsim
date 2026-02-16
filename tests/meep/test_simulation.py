"""Tests for the declarative Simulation API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gsim.meep import (
    FDTD,
    DFTDecay,
    Domain,
    FieldDecay,
    FixedTime,
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
    def test_defaults(self):
        s = ModeSource()
        assert s.port is None
        assert s.wavelength == 1.55
        assert s.bandwidth == 0.1
        assert s.num_freqs == 11

    def test_custom(self):
        s = ModeSource(port="o1", wavelength=1.31, bandwidth=0.05, num_freqs=21)
        assert s.port == "o1"
        assert s.wavelength == 1.31
        assert s.bandwidth == 0.05
        assert s.num_freqs == 21


class TestDomain:
    def test_defaults(self):
        d = Domain()
        assert d.pml == 1.0
        assert d.margin == 0.5
        assert d.margin_z_above == 0.5
        assert d.margin_z_below == 0.5
        assert d.port_margin == 0.5
        assert d.extend_ports == 0.0
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


class TestStoppingVariants:
    def test_fixed_time_defaults(self):
        s = FixedTime()
        assert s.max_time == 100.0

    def test_field_decay_defaults(self):
        s = FieldDecay()
        assert s.max_time == 100.0
        assert s.threshold == 1e-3
        assert s.component == "Ey"
        assert s.dt == 50.0
        assert s.monitor_port is None

    def test_dft_decay_defaults(self):
        s = DFTDecay()
        assert s.max_time == 100.0
        assert s.threshold == 1e-3
        assert s.min_time == 100.0

    def test_dft_decay_custom(self):
        s = DFTDecay(max_time=200, threshold=1e-4, min_time=80)
        assert s.max_time == 200
        assert s.threshold == 1e-4
        assert s.min_time == 80


class TestFDTD:
    def test_defaults(self):
        f = FDTD()
        assert f.resolution == 32
        assert isinstance(f.stopping, FixedTime)
        assert f.subpixel is False
        assert f.simplify_tol == 0.0

    def test_with_dft_decay(self):
        f = FDTD(stopping=DFTDecay(threshold=1e-4))
        assert isinstance(f.stopping, DFTDecay)
        assert f.stopping.threshold == 1e-4

    def test_resolution_custom(self):
        f = FDTD(resolution=64)
        assert f.resolution == 64


class TestFDTDDiagnostics:
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
    def test_float_shorthand(self):
        sim = Simulation(materials={"si": 3.47, "sio2": 1.44})
        assert isinstance(sim.materials["si"], Material)
        assert sim.materials["si"].n == 3.47
        assert isinstance(sim.materials["sio2"], Material)
        assert sim.materials["sio2"].n == 1.44

    def test_material_object(self):
        sim = Simulation(materials={"si": Material(n=3.47, k=0.01)})
        assert sim.materials["si"].n == 3.47
        assert sim.materials["si"].k == 0.01

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
        sim.solver.stopping = DFTDecay(threshold=1e-4, min_time=80)
        assert isinstance(sim.solver.stopping, DFTDecay)
        assert sim.solver.stopping.threshold == 1e-4

    def test_geometry_component(self):
        sim = Simulation()
        sim.geometry.component = "placeholder"
        assert sim.geometry.component == "placeholder"


class TestWholeObjectAssignment:
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
        sim.solver = FDTD(resolution=64, stopping=DFTDecay())
        assert sim.solver.resolution == 64
        assert isinstance(sim.solver.stopping, DFTDecay)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
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
    def test_from_source_defaults(self):
        sim = Simulation()
        wl = sim._wavelength_config()
        assert wl.wavelength == 1.55
        assert wl.bandwidth == 0.1
        assert wl.num_freqs == 11

    def test_from_source_custom(self):
        sim = Simulation()
        sim.source = ModeSource(wavelength=1.31, bandwidth=0.05, num_freqs=21)
        wl = sim._wavelength_config()
        assert wl.wavelength == 1.31
        assert wl.bandwidth == 0.05
        assert wl.num_freqs == 21


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------


class TestConfigTranslation:
    def test_stopping_fixed(self):
        sim = Simulation()
        sim.solver.stopping = FixedTime(max_time=200)
        cfg = sim._stopping_config()
        assert cfg.mode == "fixed"
        assert cfg.max_time == 200

    def test_stopping_field_decay(self):
        sim = Simulation()
        sim.solver.stopping = FieldDecay(threshold=1e-4, dt=25, monitor_port="o2")
        cfg = sim._stopping_config()
        assert cfg.mode == "decay"
        assert cfg.threshold == 1e-4
        assert cfg.decay_dt == 25
        assert cfg.decay_monitor_port == "o2"

    def test_stopping_dft_decay(self):
        sim = Simulation()
        sim.solver.stopping = DFTDecay(max_time=200, threshold=1e-4, min_time=80)
        cfg = sim._stopping_config()
        assert cfg.mode == "dft_decay"
        assert cfg.max_time == 200
        assert cfg.threshold == 1e-4
        assert cfg.dft_min_run_time == 80

    def test_domain_translation(self):
        sim = Simulation()
        sim.domain = Domain(pml=0.5, margin=0.3, margin_z_above=1.0, margin_z_below=0.8)
        cfg = sim._domain_config()
        assert cfg.dpml == 0.5
        assert cfg.margin_xy == 0.3
        assert cfg.margin_z_above == 1.0
        assert cfg.margin_z_below == 0.8

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
        sim.source = ModeSource(port="o1", bandwidth=0.05)
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
    def test_import_all_new_api(self):
        from gsim.meep import (
            FDTD,
            DFTDecay,
            Domain,
            FieldDecay,
            FixedTime,
            Material,
            ModeSource,
            Simulation,
            Symmetry,
        )

        assert all(
            cls is not None
            for cls in [
                DFTDecay,
                Domain,
                FDTD,
                FieldDecay,
                FixedTime,
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
