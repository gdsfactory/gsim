"""Tests for the declarative Simulation API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gsim.meep import (
    FDTD,
    Domain,
    Material,
    ModeSource,
    Simulation,
)

# ---------------------------------------------------------------------------
# Model construction & defaults
# ---------------------------------------------------------------------------


class TestMaterial:
    """Tests for the Material model."""

    def test_n_must_be_positive(self):
        with pytest.raises(ValidationError):
            Material(n=0)

    def test_k_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            Material(n=1.5, k=-0.1)


class TestStoppingFields:
    """Tests for the FDTD stopping criteria."""

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

    def test_resolution_custom(self):
        f = FDTD(resolution=64)
        assert f.resolution == 64


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
        sim.source = ModeSource(wavelength=1.31, wavelength_span=0.05)
        sim.num_freqs = 21
        wl = sim._wavelength_config()
        assert wl.wavelength == 1.31
        assert wl.bandwidth == 0.05
        assert wl.num_freqs == 21

    def test_num_freqs_propagates_with_mode_source(self):
        sim = Simulation()
        sim.num_freqs = 51
        wl = sim._wavelength_config()
        assert wl.num_freqs == 51

    def test_num_freqs_propagates_with_fiber_source(self):
        # Regression: sim.source_fiber(num_freqs=51) previously produced
        # a WavelengthConfig with num_freqs=11 because _wavelength_config
        # read self.source.num_freqs, ignoring the fiber source.
        sim = Simulation()
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source_fiber(
            x=0.0,
            z=2.0,
            waist=5.2,
            wavelength=1.55,
            wavelength_span=0.04,
        )
        sim.num_freqs = 51
        wl = sim._wavelength_config()
        assert wl.num_freqs == 51
        assert wl.wavelength == 1.55
        assert wl.bandwidth == 0.04

    def test_wavelength_config_picks_fiber_source_when_active(self):
        sim = Simulation()
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source.wavelength = 1.31
        sim.source.wavelength_span = 0.02
        sim.source_fiber(
            x=0.0,
            z=2.0,
            waist=5.2,
            wavelength=1.55,
            wavelength_span=0.04,
        )
        wl = sim._wavelength_config()
        assert wl.wavelength == 1.55
        assert wl.bandwidth == 0.04

    def test_both_sources_rejected_at_validate(self):
        sim = Simulation()
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source.port = "o1"
        sim.source_fiber(x=0.0, z=2.0, waist=5.2)
        result = sim.validate_config()
        assert not result.valid
        assert any(("Both" in e) or ("one source" in e.lower()) for e in result.errors)


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
# 2D mode tests
# ---------------------------------------------------------------------------


class Test2DMode:
    """Tests for 2D simulation mode (is_3d=False)."""

    def test_fdtd_is_3d_default_true(self):
        f = FDTD()
        assert f.is_3d is True

    def test_fdtd_is_3d_false(self):
        f = FDTD(is_3d=False)
        assert f.is_3d is False

    def test_solver_assignment(self):
        sim = Simulation()
        sim.solver.is_3d = False
        assert sim.solver.is_3d is False

    def test_solver_callable(self):
        sim = Simulation()
        sim.solver(is_3d=False)
        assert sim.solver.is_3d is False

    def test_build_config_2d_port_z_zero(self):
        """In 2D mode, build_config produces port z-centers at 0."""
        import gdsfactory as gf

        c = gf.components.straight(length=10, width=0.5)

        sim = Simulation()
        sim.geometry.component = c
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source.port = "o1"
        sim.monitors = ["o1", "o2"]
        sim.solver.is_3d = False

        result = sim.build_config()
        for port in result.config.ports:
            assert port.center[2] == 0.0

    def test_build_config_2d_is_3d_false_in_config(self):
        """build_config with is_3d=False sets is_3d=False in SimConfig."""
        import gdsfactory as gf

        c = gf.components.straight(length=10, width=0.5)

        sim = Simulation()
        sim.geometry.component = c
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source.port = "o1"
        sim.monitors = ["o1", "o2"]
        sim.solver.is_3d = False

        result = sim.build_config()
        assert result.config.is_3d is False

    def test_build_config_3d_default(self):
        """Default build_config should be 3D."""
        import gdsfactory as gf

        c = gf.components.straight(length=10, width=0.5)

        sim = Simulation()
        sim.geometry.component = c
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source.port = "o1"
        sim.monitors = ["o1", "o2"]

        result = sim.build_config()
        assert result.config.is_3d is True

    def test_write_config_2d_json(self, tmp_path):
        """write_config with is_3d=False produces JSON with is_3d=false."""
        import json

        import gdsfactory as gf

        c = gf.components.straight(length=10, width=0.5)

        sim = Simulation()
        sim.geometry.component = c
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source.port = "o1"
        sim.monitors = ["o1", "o2"]
        sim.solver.is_3d = False

        out = sim.write_config(tmp_path / "sim2d")
        config_data = json.loads((out / "sim_config.json").read_text())
        assert config_data["is_3d"] is False
        # Ports should have z=0
        for port in config_data["ports"]:
            assert port["center"][2] == 0.0


# ---------------------------------------------------------------------------
# XZ 2D build_config wiring
# ---------------------------------------------------------------------------


def _xz_straight_component():
    """Build a straight-waveguide component with a port on +X."""
    import gdsfactory as gf

    c = gf.Component()
    c.add_polygon(
        [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
        layer=(1, 0),
    )
    c.add_port(
        name="o1",
        center=(5.0, 0.0),
        orientation=0.0,
        width=0.5,
        layer=(1, 0),
    )
    return c


def _xz_trivial_stack():
    """Build a trivial 3-layer stack (substrate / core / clad) for XZ tests."""
    from gsim.common.stack import Layer, LayerStack

    return LayerStack(
        pdk_name="test",
        units="um",
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=0.0,
                zmax=0.22,
                thickness=0.22,
                material="si",
                layer_type="dielectric",
            ),
        },
        materials={},
        dielectrics=[
            {"name": "box", "zmin": -2.0, "zmax": 0.0, "material": "SiO2"},
            {"name": "clad", "zmin": 0.22, "zmax": 1.0, "material": "SiO2"},
        ],
        simulation={},
    )


class TestXZBuildConfig:
    """Tests for XZ 2D fields wired through Simulation.build_config()."""

    def test_y_cut_defaults_to_bbox_center(self):
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = _xz_straight_component()
        sim.geometry.stack = _xz_trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source_fiber(x=0.0, z=1.22, waist=5.4)

        result = sim.build_config()

        # Straight is centered on y=0 -> bbox center is 0.
        assert result.config.y_cut == pytest.approx(0.0, abs=1e-6)

    def test_y_cut_explicit_override(self):
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = _xz_straight_component()
        sim.geometry.y_cut = 0.1
        sim.geometry.stack = _xz_trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source_fiber(x=0.0, z=1.22, waist=5.4)

        result = sim.build_config()
        assert result.config.y_cut == pytest.approx(0.1)

    def test_xz_plane_serializes(self):
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = _xz_straight_component()
        sim.geometry.stack = _xz_trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source_fiber(x=0.0, z=1.22, waist=5.4)

        result = sim.build_config()
        assert result.config.plane == "xz"
        assert result.config.is_3d is False

    def test_fiber_source_serialized_with_k_direction(self):
        import math

        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = _xz_straight_component()
        sim.geometry.stack = _xz_trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source_fiber(x=0.0, z=1.22, angle_deg=14.5, waist=5.4)

        result = sim.build_config()
        fs = result.config.fiber_source
        assert fs is not None
        theta = math.radians(14.5)
        assert fs.k_direction[0] == pytest.approx(math.sin(theta))
        assert fs.k_direction[1] == pytest.approx(0.0)
        assert fs.k_direction[2] == pytest.approx(-math.cos(theta))
        # Absolute z passed through unchanged.
        assert fs.z == pytest.approx(1.22)


class TestXZValidation:
    """Tests for XZ-only validation in build_config."""

    def test_xz_without_monitors_or_fiber_errors(self):
        import gdsfactory as gf

        from gsim.common.stack import Layer, LayerStack
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        c = gf.Component()
        c.add_polygon(
            [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
            layer=(1, 0),
        )
        sim.geometry.component = c
        sim.geometry.stack = LayerStack(
            pdk_name="test",
            units="um",
            layers={
                "core": Layer(
                    name="core",
                    gds_layer=(1, 0),
                    zmin=0.0,
                    zmax=0.22,
                    thickness=0.22,
                    material="si",
                    layer_type="dielectric",
                ),
            },
            materials={},
            dielectrics=[],
            simulation={},
        )
        sim.materials = {"si": 3.47}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"

        with pytest.raises(ValueError, match="no valid monitors and no fiber source"):
            sim.build_config()


class TestXZAutoCrop:
    """Tests for XZ 2D auto z-crop + fiber-aware margin."""

    def test_xz_defaults_z_crop_auto(self):
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = _xz_straight_component()
        sim.geometry.stack = _xz_trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source_fiber(x=0.0, z=1.22, waist=5.4)

        sim.build_config()

        # build_config resolves z_crop to "auto", applies it, then clears it.
        assert sim.geometry.z_crop is None
        # Stack should have been cropped around the core layer.
        assert sim.geometry.stack is not None
        z_min = min(l.zmin for l in sim.geometry.stack.layers.values())
        z_max = max(l.zmax for l in sim.geometry.stack.layers.values())
        # Core is at [0.0, 0.22]; cropped range should be a few um at most.
        assert z_max - z_min < 10.0

    def test_xz_fiber_expands_margin_z_above(self):
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = _xz_straight_component()
        sim.geometry.stack = _xz_trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        # Clad zmax is 1.0 (see _xz_trivial_stack), so absolute z=1.42 is
        # 0.42 um above the top of the physical stack.
        sim.source_fiber(x=0.0, z=1.42, waist=5.4, angle_deg=14.5)

        initial_margin = sim.domain.margin_z_above
        sim.build_config()
        # Margin should grow to at least (z - stack_top) + waist/2 so the
        # fiber beam plane sits inside the simulation cell.
        assert sim.domain.margin_z_above >= 0.42 + 5.4 / 2
        assert sim.domain.margin_z_above > initial_margin

    def test_xz_auto_crop_preserves_full_box(self):
        """Auto z-crop must keep the full BOX dielectric intact.

        Regression for the GC notebook where margin_z_below=0.5 µm was
        chopping a real 2 µm SOI BOX down to 0.5 µm because the crop was
        referenced to the core layer.
        """
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = _xz_straight_component()
        sim.geometry.stack = _xz_trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        # Default margin_z_below=0.5 would previously crop the 2 µm BOX
        # to 0.5 µm. Option A references the full non-air stack extent,
        # so the BOX stays full thickness.
        sim.source_fiber(x=0.0, z=1.22, waist=5.4)
        sim.build_config()

        assert sim.geometry.stack is not None
        box = next(d for d in sim.geometry.stack.dielectrics if d["name"] == "box")
        assert box["zmin"] == -2.0
        assert box["zmax"] == 0.0
