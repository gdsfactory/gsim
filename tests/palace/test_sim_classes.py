"""Tests for Palace simulation classes and the Palace binary resolver.

Covers DrivenSim/EigenmodeSim/ElectrostaticSim validation and config, plus
the ``gsim.palace.runtime`` binary resolver (``resolve_palace_binary`` etc.).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsim.palace import BoundaryModeSim, DrivenSim, EigenmodeSim, ElectrostaticSim
from gsim.palace.models import MeshConfig


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
        sim.add_cpw_port("P1", layer="", s_width=10, gap_width=6, length=5.0)
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

    def test_floquet_requires_target_frequency(self):
        """Floquet setup must include eigenmode target frequency."""
        sim = EigenmodeSim()
        with pytest.raises(ValueError, match="Floquet requires target frequency"):
            sim.set_eigenmode(floquet=True)

    def test_floquet_options_are_stored(self):
        """Floquet options should propagate into eigenmode config."""
        sim = EigenmodeSim()
        sim.set_eigenmode(target=40e9, floquet=True, phi_target=1.2, n_eff_guess=2.4)
        assert sim.eigenmode.floquet is True
        assert sim.eigenmode.phi_target == pytest.approx(1.2)
        assert sim.eigenmode.n_eff_guess == pytest.approx(2.4)


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


class TestBoundaryModeSimValidation:
    """Test BoundaryModeSim validation and API behavior."""

    def test_missing_geometry(self):
        """Test validation catches missing geometry."""
        sim = BoundaryModeSim()
        result = sim.validate_config()
        assert not result.valid
        assert any("No component set" in e for e in result.errors)

    def test_requires_explicit_cross_section(self):
        """Boundary mode requires explicit cross-section selection."""
        sim = BoundaryModeSim()
        result = sim.validate_config()
        assert not result.valid
        assert any("explicit cross-section plane" in e for e in result.errors)

    def test_set_cross_section_from_string(self):
        """String plane spec should parse into axis/value fields."""
        sim = BoundaryModeSim()
        sim.set_cross_section("y=100")
        assert sim.cross_section is not None
        assert sim.cross_section.axis == "y"
        assert sim.cross_section.value == pytest.approx(100.0)

    def test_set_cross_section_rejects_invalid_spec(self):
        """Invalid plane spec should raise ValueError."""
        sim = BoundaryModeSim()
        with pytest.raises(ValueError, match="Invalid plane spec"):
            sim.set_cross_section("foo")

    def test_z_cross_section_rejected_for_native_2d(self):
        """Native 2D BoundaryMode currently supports only x/y sections."""
        sim = BoundaryModeSim()
        sim.set_cross_section("z=0")
        result = sim.validate_config()
        assert not result.valid
        assert any("supports only x/y" in e for e in result.errors)

    def test_port_api_rejected_for_boundarymode(self):
        """BoundaryMode native 2D does not accept explicit port definitions."""
        sim = BoundaryModeSim()
        sim.set_cross_section("x=0")
        sim.add_wave_port("o1", layer="metal1")
        result = sim.validate_config()
        assert not result.valid
        assert any("cross_section-only native 2D" in e for e in result.errors)

    def test_set_boundary_mode_updates_model(self):
        """set_boundary_mode should populate BoundaryModeConfig fields."""
        sim = BoundaryModeSim()
        sim.set_boundary_mode(
            freq=8e9,
            num_modes=3,
            save=2,
            target=2.2,
            tolerance=1e-8,
            max_size=60,
            solver_type="SLEPc",
        )
        cfg = sim.boundary_mode
        assert cfg.freq == pytest.approx(8e9)
        assert cfg.num_modes == 3
        assert cfg.save == 2
        assert cfg.target == pytest.approx(2.2)
        assert cfg.tolerance == pytest.approx(1e-8)
        assert cfg.max_size == 60
        assert cfg.solver_type == "SLEPc"


class TestMixinMethods:
    """Test mixin methods work on all simulation classes."""

    def test_set_output_dir(self, tmp_path):
        """Test set_output_dir works on all sim classes."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim, BoundaryModeSim]:
            sim = cls()
            sim.set_output_dir(tmp_path / "test")
            assert sim.output_dir == tmp_path / "test"
            assert sim.output_dir is not None
            assert sim.output_dir.exists()

    def test_set_stack(self):
        """Test set_stack works on all sim classes."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim, BoundaryModeSim]:
            sim = cls()
            sim.set_stack(air_above=500.0, air_below=25.0)
            assert "air_above" not in sim._stack_kwargs
            assert "air_below" not in sim._stack_kwargs

    def test_set_stack_default_air_above_zero(self):
        """Default stack setup should not pass legacy air kwargs at all."""
        sim = DrivenSim()
        sim.set_stack()
        assert "air_above" not in sim._stack_kwargs
        assert "air_below" not in sim._stack_kwargs

    def test_set_airbox(self):
        """Test set_airbox stores explicit airbox margins and z extents."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim, BoundaryModeSim]:
            sim = cls()
            sim.set_airbox(margin_x=50.0, margin_y=30.0, z_above=100.0, z_below=80.0)
            assert sim._airbox_config == {
                "margin_x": 50.0,
                "margin_y": 30.0,
                "z_above": 100.0,
                "z_below": 80.0,
                "material": "air",
            }

    def test_set_airbox_material(self):
        """set_airbox(material=...) stores a custom background material."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim, BoundaryModeSim]:
            sim = cls()
            sim.set_airbox(material="sio2", margin_y=2.0)
            assert sim._airbox_config["material"] == "sio2"
            assert sim._airbox_config["margin_y"] == 2.0
            with pytest.raises(ValueError, match="material"):
                sim.set_airbox(material="")
            with pytest.raises(ValueError, match="material"):
                sim.set_airbox(material=123)  # ty: ignore[invalid-argument-type]

    def test_set_airbox_defaults_to_zero(self):
        """Unassigned set_airbox arguments should default to 0.0."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim]:
            sim = cls()
            sim.set_airbox()
            assert sim._airbox_config == {
                "margin_x": 0.0,
                "margin_y": 0.0,
                "z_above": 0.0,
                "z_below": 0.0,
                "material": "air",
            }

    def test_set_airbox_partial_defaults(self):
        """Any omitted set_airbox field should still become 0.0."""
        sim = DrivenSim()
        sim.set_airbox(margin_x=50.0)
        assert sim._airbox_config == {
            "margin_x": 50.0,
            "margin_y": 0.0,
            "z_above": 0.0,
            "z_below": 0.0,
            "material": "air",
        }

    def test_set_airbox_invalid(self):
        """set_airbox should reject negative margins/extents."""
        sim = DrivenSim()
        with pytest.raises(ValueError):
            sim.set_airbox(margin_x=-1.0, z_above=100.0, z_below=100.0)

    def test_mesh_routes_airbox_kwargs_through_set_airbox(self, monkeypatch, tmp_path):
        """mesh() air-region kwargs should be applied via set_airbox()."""
        captured: dict[str, object] = {}

        def _fake_set_airbox(
            _self,
            *,
            margin_x=None,
            margin_y=None,
            z_above=None,
            z_below=None,
            material="air",
        ):
            captured["margin_x"] = margin_x
            captured["margin_y"] = margin_y
            captured["z_above"] = z_above
            captured["z_below"] = z_below
            captured["material"] = material

        def _fake_generate_mesh_internal(_self, **_kwargs):
            return SimpleNamespace(mesh_stats={}, mesh_path=tmp_path / "palace.msh")

        sim = DrivenSim()
        sim.set_output_dir(tmp_path / "sim")

        monkeypatch.setattr(DrivenSim, "set_airbox", _fake_set_airbox)

        def _fake_validate_config(_self):
            return SimpleNamespace(valid=True, errors=[])

        monkeypatch.setattr(
            DrivenSim,
            "validate_config",
            _fake_validate_config,
        )
        monkeypatch.setattr(DrivenSim, "_resolve_stack", lambda _self: object())
        monkeypatch.setattr(
            DrivenSim,
            "_configure_ports_on_component",
            lambda _self, _stack: None,
        )
        monkeypatch.setattr(
            "gsim.palace.ports.extract_ports",
            lambda _component, _stack: [],
        )
        monkeypatch.setattr(
            DrivenSim,
            "_generate_mesh_internal",
            _fake_generate_mesh_internal,
        )

        sim.mesh(
            margin_x=50.0,
            margin_y=10.0,
            z_above=120.0,
            z_below=80.0,
            verbose=False,
        )

        assert captured == {
            "margin_x": 50.0,
            "margin_y": 10.0,
            "z_above": 120.0,
            "z_below": 80.0,
            "material": "air",
        }

    def test_mesh_preserves_background_material(self, monkeypatch, tmp_path):
        """mesh() margin overrides must not reset the background material to air."""
        captured: dict[str, object] = {}

        def _fake_set_airbox(_self, **kwargs):
            captured.update(kwargs)

        def _fake_generate_mesh_internal(_self, **_kwargs):
            return SimpleNamespace(mesh_stats={}, mesh_path=tmp_path / "palace.msh")

        sim = DrivenSim()
        sim.set_output_dir(tmp_path / "sim")
        sim.set_airbox(material="sio2", margin_y=3.0)

        monkeypatch.setattr(DrivenSim, "set_airbox", _fake_set_airbox)
        monkeypatch.setattr(
            DrivenSim,
            "validate_config",
            lambda _self: SimpleNamespace(valid=True, errors=[]),
        )
        monkeypatch.setattr(DrivenSim, "_resolve_stack", lambda _self: object())
        monkeypatch.setattr(
            DrivenSim,
            "_configure_ports_on_component",
            lambda _self, _stack: None,
        )
        monkeypatch.setattr(
            "gsim.palace.ports.extract_ports",
            lambda _component, _stack: [],
        )
        monkeypatch.setattr(
            DrivenSim,
            "_generate_mesh_internal",
            _fake_generate_mesh_internal,
        )

        sim.mesh(margin_y=50.0, verbose=False)

        assert captured["material"] == "sio2"
        assert captured["margin_y"] == 50.0

    def test_set_airbox_margin_y_zero_reaches_generate_mesh(
        self, monkeypatch, tmp_path
    ):
        """set_airbox(margin_y=0) must propagate to meshing domain extents."""
        captured: dict[str, float] = {}

        def _fake_generate_mesh(**kwargs):
            captured["margin_x"] = kwargs["margin_x"]
            captured["margin_y"] = kwargs["margin_y"]
            return SimpleNamespace(
                mesh_path=tmp_path / "palace.msh",
                config_path=None,
                port_info=[],
                mesh_stats={},
                groups={},
            )

        monkeypatch.setattr(
            "gsim.palace.mesh.generator.generate_mesh", _fake_generate_mesh
        )
        monkeypatch.setattr(DrivenSim, "_resolve_stack", lambda _self: object())
        monkeypatch.setattr(
            DrivenSim,
            "_configure_ports_on_component",
            lambda _self, _stack: None,
        )
        monkeypatch.setattr(
            "gsim.palace.ports.extract_ports",
            lambda _component, _stack: [],
        )

        sim = DrivenSim()
        sim.set_output_dir(tmp_path / "sim")
        sim.set_airbox(margin_x=50.0, margin_y=0.0, z_above=100.0, z_below=100.0)

        sim._generate_mesh_internal(
            output_dir=tmp_path / "sim",
            mesh_config=MeshConfig.default(),
            ports=[],
            driven_config=sim.driven,
            model_name="palace",
            verbose=False,
            write_config=False,
        )

        assert captured["margin_x"] == 50.0
        assert captured["margin_y"] == 0.0

    def test_curved_mesh_options_reach_generate_mesh(self, monkeypatch, tmp_path):
        """Curve-fit, decimation, and verbosity options must be forwarded."""
        captured: dict[str, object] = {}

        def _fake_generate_mesh(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                mesh_path=tmp_path / "palace.msh",
                config_path=None,
                port_info=[],
                mesh_stats={},
                groups={},
            )

        monkeypatch.setattr(
            "gsim.palace.mesh.generator.generate_mesh", _fake_generate_mesh
        )
        monkeypatch.setattr(DrivenSim, "_resolve_stack", lambda _self: object())

        sim = DrivenSim()
        sim.set_output_dir(tmp_path / "sim")

        mesh_config = MeshConfig.default(
            curve_fit_mode="bspline",
            curve_fit_layers=["core"],
            curve_fit_tolerance_um=0.02,
            curve_fit_min_points=12,
            curve_fit_corner_angle_deg=30.0,
        )

        sim._generate_mesh_internal(
            output_dir=tmp_path / "sim",
            mesh_config=mesh_config,
            ports=[],
            driven_config=sim.driven,
            model_name="palace",
            verbose=False,
            write_config=False,
            decimate_tolerance=0.005,
            gmsh_verbosity=7,
        )

        assert captured["curve_fit_mode"] == "bspline"
        assert captured["curve_fit_layers"] == ["core"]
        assert captured["curve_fit_tolerance_um"] == 0.02
        assert captured["curve_fit_min_points"] == 12
        assert captured["curve_fit_corner_angle_deg"] == 30.0
        assert captured["decimate_tolerance"] == 0.005
        assert captured["verbosity"] == 7

    def test_set_material(self):
        """Test set_material works on all sim classes."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim, BoundaryModeSim]:
            sim = cls()
            sim.set_material(
                "custom_metal", material_type="conductor", conductivity=1e7
            )
            assert "custom_metal" in sim.materials
            assert sim.materials["custom_metal"].conductivity == 1e7

    def test_set_numerical(self):
        """Test set_numerical works on all sim classes."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim, BoundaryModeSim]:
            sim = cls()
            sim.set_numerical(
                order=3,
                tolerance=1e-8,
                max_iterations=1000,
                solver_type="MUMPS",
                preconditioner="AMS",
                device="CPU",
            )
            assert sim.numerical.order == 3
            assert sim.numerical.tolerance == 1e-8
            assert sim.numerical.max_iterations == 1000
            assert sim.numerical.solver_type == "MUMPS"
            assert sim.numerical.preconditioner == "AMS"
            assert sim.numerical.device == "CPU"

    def test_mesh_requires_output_dir(self):
        """Test mesh() raises if output_dir not set."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim, BoundaryModeSim]:
            sim = cls()
            with pytest.raises(ValueError, match="Output directory not set"):
                sim.mesh()


class TestAddPec:
    """Test add_pec() on all simulation classes."""

    def test_add_pec_stores_config(self):
        """add_pec() stores PECBlockConfig on all simulation classes."""
        for cls in [DrivenSim, EigenmodeSim, ElectrostaticSim, BoundaryModeSim]:
            sim = cls()
            sim.add_pec(gds_layer=(65000, 0), from_layer="metal1", to_layer="topmetal2")
            assert len(sim._pec_blocks) == 1
            cfg = sim._pec_blocks[0]
            assert cfg.from_layer == "metal1"
            assert cfg.to_layer == "topmetal2"
            assert cfg.gds_layer == (65000, 0)

    def test_add_pec_custom_gds_layer(self):
        """Custom gds_layer is respected."""
        sim = DrivenSim()
        sim.add_pec(gds_layer=(200, 0), from_layer="metal1", to_layer="topmetal2")
        assert sim._pec_blocks[0].gds_layer == (200, 0)

    def test_add_pec_accumulates(self):
        """Multiple add_pec() calls accumulate."""
        sim = DrivenSim()
        sim.add_pec(gds_layer=(65000, 0), from_layer="metal1", to_layer="topmetal2")
        sim.add_pec(gds_layer=(65000, 0), from_layer="metal2", to_layer="topmetal2")
        assert len(sim._pec_blocks) == 2
        assert sim._pec_blocks[0].from_layer == "metal1"
        assert sim._pec_blocks[1].from_layer == "metal2"


# ---------------------------------------------------------------------------
# Palace binary resolver tests (gsim.palace.runtime)
# ---------------------------------------------------------------------------


@pytest.fixture
def _mock_gcloud(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock gsim.gcloud so we can import gsim.palace.runtime without gdsfactoryplus."""
    import types

    gcloud = types.ModuleType("gsim.gcloud")
    for name in (
        "get_status",
        "wait_for_results",
        "register_result_parser",
        "print_job_summary",
        "run_simulation",
    ):
        setattr(gcloud, name, lambda *a, **kw: None)
    gcloud.RunResult = type("RunResult", (), {})  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "gsim.gcloud", gcloud)


@pytest.fixture
def _no_local_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate gsim's own cached/downloaded runtime during resolver tests."""
    import gsim.palace.runtime as rt

    monkeypatch.setattr(rt, "_cached_binary", lambda: None)
    monkeypatch.setattr(rt, "_cached_library_dir", lambda: None)
    monkeypatch.setattr(rt, "_is_linux_x86_64", lambda: False)
    monkeypatch.setattr(rt, "_auto_download_enabled", lambda: False)
    monkeypatch.setattr(rt, "_palace_cpu_available", lambda: False)
    monkeypatch.setattr(rt, "_palace_toolkit_available", lambda: False)


class TestResolvePalaceBinary:
    @pytest.mark.usefixtures("_mock_gcloud", "_no_local_runtime")
    def test_returns_none_when_nothing_found(self) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        with pytest.MonkeyPatch().context() as mp:
            mp.delenv("PALACE_BIN", raising=False)
            mp.delenv("PALACE_EXECUTABLE", raising=False)
            result = resolve_palace_binary()
            assert result is None

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_uses_palace_bin_env(self) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        fake_bin = Path("/usr/local/bin/palace")

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("PALACE_BIN", str(fake_bin))
            mp.setattr("gsim.palace.runtime._binary_is_runnable", lambda *a, **k: True)
            mp.setattr("pathlib.Path.is_file", lambda _: True)
            result = resolve_palace_binary()
            assert result is not None

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_delegates_to_palacetoolkit(self) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        fake_ptk_bin = Path("/opt/palacetoolkit/bin/palace")

        class _FakePalaceCPU:
            @staticmethod
            def palace_binary_path() -> Path:
                return fake_ptk_bin

        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "palacetoolkit_palace_cpu", _FakePalaceCPU())
            mp.setattr("gsim.palace.runtime._palace_cpu_available", lambda: True)
            mp.setattr("gsim.palace.runtime._palace_toolkit_available", lambda: False)
            mp.setattr("gsim.palace.runtime._cached_binary", lambda: None)
            mp.setattr("gsim.palace.runtime._binary_is_runnable", lambda *a, **k: True)
            mp.setattr("pathlib.Path.is_file", lambda _: True)
            result = resolve_palace_binary()
            assert result is not None

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_delegates_to_palacetoolkit_package(self) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        fake_ptk_bin = Path("/opt/palacetoolkit/runtime/bin/palace")

        import types

        ptk = types.ModuleType("palacetoolkit")
        ptk.__path__ = []  # type: ignore[attr-defined]
        ptk_runtime = types.ModuleType("palacetoolkit.palace_runtime")
        setattr(ptk_runtime, "resolve_palace_binary", lambda: fake_ptk_bin)  # noqa: B010

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("gsim.palace.runtime._palace_cpu_available", lambda: False)
            mp.setattr("gsim.palace.runtime._palace_toolkit_available", lambda: True)
            mp.setattr("gsim.palace.runtime._cached_binary", lambda: None)
            mp.setattr("gsim.palace.runtime._binary_is_runnable", lambda *a, **k: True)
            mp.setattr("pathlib.Path.is_file", lambda _: True)
            mp.setitem(sys.modules, "palacetoolkit", ptk)
            mp.setitem(sys.modules, "palacetoolkit.palace_runtime", ptk_runtime)
            result = resolve_palace_binary()
            assert result is not None

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_uses_gsim_cached_runtime(self) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        fake_bin = Path("/home/user/.cache/palacetoolkit/runtime/bin/palace")

        with pytest.MonkeyPatch().context() as mp:
            mp.delenv("PALACE_BIN", raising=False)
            mp.delenv("PALACE_EXECUTABLE", raising=False)
            mp.setattr("gsim.palace.runtime._palace_cpu_available", lambda: False)
            mp.setattr("gsim.palace.runtime._palace_toolkit_available", lambda: False)
            mp.setattr("gsim.palace.runtime._cached_binary", lambda: fake_bin)
            mp.setattr("gsim.palace.runtime._binary_is_runnable", lambda *a, **k: True)
            result = resolve_palace_binary()
            assert result is not None

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_downloads_runtime_when_missing(self) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        fake_downloaded = Path("/home/user/.cache/palacetoolkit/runtime/bin/palace")

        with pytest.MonkeyPatch().context() as mp:
            mp.delenv("PALACE_BIN", raising=False)
            mp.delenv("PALACE_EXECUTABLE", raising=False)
            mp.setattr("gsim.palace.runtime._palace_cpu_available", lambda: False)
            mp.setattr("gsim.palace.runtime._palace_toolkit_available", lambda: False)
            mp.setattr("gsim.palace.runtime._cached_binary", lambda: None)
            mp.setattr("gsim.palace.runtime._is_linux_x86_64", lambda: True)
            mp.setattr("gsim.palace.runtime._auto_download_enabled", lambda: True)
            mp.setattr(
                "gsim.palace.runtime.install_palace_runtime",
                lambda **k: fake_downloaded,
            )
            mp.setattr("gsim.palace.runtime._cached_library_dir", lambda: None)
            mp.setattr("gsim.palace.runtime._binary_is_runnable", lambda *a, **k: True)
            result = resolve_palace_binary()
            assert result == fake_downloaded.resolve()

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_prefer_bundled_skips_env(self) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("PALACE_BIN", "/usr/bin/palace")
            mp.setattr("gsim.palace.runtime._palace_cpu_available", lambda: False)
            mp.setattr("gsim.palace.runtime._palace_toolkit_available", lambda: False)
            mp.setattr("gsim.palace.runtime._cached_binary", lambda: None)
            mp.setattr("gsim.palace.runtime._auto_download_enabled", lambda: False)
            result = resolve_palace_binary(prefer_bundled=True)
            assert result is None


class TestInstallPalaceRuntime:
    @pytest.mark.usefixtures("_mock_gcloud")
    def test_returns_cached_binary_when_present(self, tmp_path: Path) -> None:
        import gsim.palace.runtime as rt

        tag = "0.17.0"
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(rt, "_runtime_cache_dir", lambda: tmp_path)
            mp.setattr(rt, "_binary_tag", lambda: tag)
            prefix = tmp_path / f"palace-cpu-v{tag}"
            (prefix / "bin").mkdir(parents=True)
            (prefix / "lib").mkdir(parents=True)
            bin_palace = prefix / "bin" / "palace"
            bin_palace.write_text("#!/bin/sh\nexit 0\n")
            bin_palace.chmod(0o755)
            result = rt.install_palace_runtime(force=False)
            assert result == bin_palace

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_raises_on_non_linux_x86_64(self, tmp_path: Path) -> None:
        import gsim.palace.runtime as rt

        with pytest.MonkeyPatch().context() as mp:
            # Use an empty cache dir so the fallthrough to the platform guard
            # is deterministic regardless of what is cached on the host.
            mp.setattr(rt, "_runtime_cache_dir", lambda: tmp_path)
            mp.setattr(rt, "_is_linux_x86_64", lambda: False)
            with pytest.raises(RuntimeError):
                rt.install_palace_runtime()

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_downloads_and_extracts_runtime(self, tmp_path: Path) -> None:
        import io
        import zipfile

        import gsim.palace.runtime as rt

        tag = "0.9.9"
        cache_dir = tmp_path / "cache"

        # Build a fake wheel in memory: payload with bin/palace and lib/libfoo.so
        wheel_buf = io.BytesIO()
        with zipfile.ZipFile(wheel_buf, "w") as zf:
            zf.writestr("palacetoolkit_palace_cpu/bin/palace", "#!/bin/sh\nexit 0\n")
            zf.writestr("palacetoolkit_palace_cpu/bin/palace-x86_64.bin", "x")
            zf.writestr("palacetoolkit_palace_cpu/lib/libfoo.so", "libdata")
        wheel_buf.seek(0)

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return wheel_buf.getvalue()

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(rt, "_runtime_cache_dir", lambda: cache_dir)
            mp.setattr(rt, "_binary_tag", lambda: tag)
            mp.setattr(rt, "_is_linux_x86_64", lambda: True)
            mp.setattr(
                rt, "_binary_wheel_url", lambda t: "https://example.invalid/x.whl"
            )
            mp.setattr(rt, "_binary_wheel_url_from_release", lambda t, timeout: None)
            mp.setattr(rt, "urlopen", lambda *a, **k: _FakeResponse())

            result = rt.install_palace_runtime(force=False)

            prefix = cache_dir / f"palace-cpu-v{tag}"
            assert result == prefix / "bin" / "palace"
            assert (prefix / "bin" / "palace").is_file()
            assert (prefix / "lib" / "libfoo.so").is_file()
            assert os.access(result, os.X_OK)


class TestResolvePalaceLibraryDir:
    @pytest.mark.usefixtures("_mock_gcloud", "_no_local_runtime")
    def test_returns_none_without_runtime(self) -> None:
        from gsim.palace.runtime import resolve_palace_library_dir

        assert resolve_palace_library_dir() is None

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_delegates_to_palacetoolkit(self) -> None:
        from gsim.palace.runtime import resolve_palace_library_dir

        fake_lib = Path("/opt/palacetoolkit/lib")

        class _FakePalaceCPU:
            @staticmethod
            def palace_library_path() -> Path:
                return fake_lib

        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "palacetoolkit_palace_cpu", _FakePalaceCPU())
            mp.setattr("gsim.palace.runtime._palace_cpu_available", lambda: True)
            mp.setattr("gsim.palace.runtime._palace_toolkit_available", lambda: False)
            mp.setattr("pathlib.Path.is_dir", lambda _: True)
            result = resolve_palace_library_dir()
            assert result is not None

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_uses_gsim_cached_library_dir(self) -> None:
        from gsim.palace.runtime import resolve_palace_library_dir

        fake_lib = Path("/home/user/.cache/palacetoolkit/runtime/lib")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("gsim.palace.runtime._palace_cpu_available", lambda: False)
            mp.setattr("gsim.palace.runtime._palace_toolkit_available", lambda: False)
            mp.setattr("gsim.palace.runtime._cached_library_dir", lambda: fake_lib)
            result = resolve_palace_library_dir()
            assert result == fake_lib.resolve()

    @pytest.mark.usefixtures("_mock_gcloud")
    def test_delegates_to_palacetoolkit_package(self) -> None:
        from gsim.palace.runtime import resolve_palace_library_dir

        fake_lib = Path("/opt/palacetoolkit/runtime/lib")

        import types

        ptk = types.ModuleType("palacetoolkit")
        ptk.__path__ = []  # type: ignore[attr-defined]
        ptk_runtime = types.ModuleType("palacetoolkit.palace_runtime")
        setattr(ptk_runtime, "resolve_palace_library_dir", lambda: fake_lib)  # noqa: B010

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("gsim.palace.runtime._palace_cpu_available", lambda: False)
            mp.setattr("gsim.palace.runtime._palace_toolkit_available", lambda: True)
            mp.setattr("pathlib.Path.is_dir", lambda _: True)
            mp.setitem(sys.modules, "palacetoolkit", ptk)
            mp.setitem(sys.modules, "palacetoolkit.palace_runtime", ptk_runtime)
            result = resolve_palace_library_dir()
            assert result is not None


class TestPalacetoolkitAvailable:
    @pytest.mark.usefixtures("_mock_gcloud")
    def test_true_when_installed(self) -> None:
        from gsim.palace.runtime import (
            _palace_cpu_available,
            _palace_toolkit_available,
        )

        # If either package is actually installed on the system, this will be
        # True. We can't force it via mock here without patching importlib,
        # which is fragile. Instead we just verify the functions run.
        assert isinstance(_palace_cpu_available(), bool)
        assert isinstance(_palace_toolkit_available(), bool)
