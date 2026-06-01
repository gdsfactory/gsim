"""Unit tests for curved meshing pipeline behavior."""

from __future__ import annotations

import builtins
import json
from types import SimpleNamespace

from gsim.common import Layer, LayerStack
from gsim.palace.mesh import generator as mesh_generator
from gsim.palace.mesh.config_generator import generate_palace_config


class _FakeOption:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def setNumber(self, name: str, value: float) -> None:  # noqa: N802 (gmsh API)
        self.calls.append((name, value))


class _FakeMeshOps:
    def __init__(self) -> None:
        self.generated_dim: int | None = None

    def generate(self, dim: int) -> None:
        self.generated_dim = dim

    def setOrder(self, _order: int) -> None:  # noqa: N802 (gmsh API)
        return

    def optimize(self, _method: str) -> None:
        return


class _FakeModel:
    def __init__(self) -> None:
        self.occ = object()
        self.mesh = _FakeMeshOps()
        self._models: list[str] = []

    def list(self) -> builtins.list[str]:
        return list(self._models)

    def setCurrent(self, _name: str) -> None:  # noqa: N802 (gmsh API)
        return

    def remove(self) -> None:
        self._models.clear()

    def add(self, name: str) -> None:
        self._models.append(name)


class _FakeFltk:
    def run(self) -> None:
        return


class _FakeGmsh:
    def __init__(self) -> None:
        self.option = _FakeOption()
        self.model = _FakeModel()
        self.fltk = _FakeFltk()
        self.cleared = False
        self.finalized = False
        self.writes: list[str] = []

    def initialize(self) -> None:
        return

    def clear(self) -> None:
        self.cleared = True

    def finalize(self) -> None:
        self.finalized = True

    def write(self, path: str) -> None:
        self.writes.append(path)


def test_generate_mesh_forwards_curve_fit_and_decimation(monkeypatch, tmp_path) -> None:
    """Curved meshing settings propagate through generate_mesh internals."""
    captured: dict[str, object] = {}
    fake_gmsh = _FakeGmsh()

    monkeypatch.setattr(mesh_generator, "gmsh", fake_gmsh)

    def _fake_extract_geometry(_component, _stack, decimate_tolerance=None):
        captured["decimate_tolerance"] = decimate_tolerance
        return SimpleNamespace(polygons=[object()], bbox=(0.0, 0.0, 10.0, 10.0))

    monkeypatch.setattr(mesh_generator, "extract_geometry", _fake_extract_geometry)
    monkeypatch.setattr(mesh_generator, "add_metals", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        mesh_generator,
        "add_ports",
        lambda *_args, **_kwargs: ({"P1": [11]}, []),
    )
    monkeypatch.setattr(
        mesh_generator,
        "add_dielectrics",
        lambda *_args, **_kwargs: {"air": [31]},
    )

    def _fake_add_patterned_dielectrics(
        _kernel,
        _geometry,
        _stack,
        *,
        curve_fit_mode,
        curve_fit_layers,
        curve_fit_tolerance_um,
        curve_fit_min_points,
        curve_fit_corner_angle_deg,
    ):
        captured["curve_fit"] = {
            "curve_fit_mode": curve_fit_mode,
            "curve_fit_layers": curve_fit_layers,
            "curve_fit_tolerance_um": curve_fit_tolerance_um,
            "curve_fit_min_points": curve_fit_min_points,
            "curve_fit_corner_angle_deg": curve_fit_corner_angle_deg,
        }
        return {"core": [21]}

    monkeypatch.setattr(
        mesh_generator,
        "add_patterned_dielectrics",
        _fake_add_patterned_dielectrics,
    )

    def _fake_build_entities(
        metal_tags,
        dielectric_tags,
        patterned_dielectric_tags,
        port_tags,
        port_info,
        pec_block_tags,
        stack,
    ):
        captured["build_entities_args"] = {
            "dielectric_tags": dielectric_tags,
            "patterned_dielectric_tags": patterned_dielectric_tags,
            "port_tags": port_tags,
            "port_info": port_info,
            "pec_block_tags": pec_block_tags,
            "stack": stack,
            "metal_tags": metal_tags,
        }
        return []

    monkeypatch.setattr(mesh_generator, "build_entities", _fake_build_entities)
    monkeypatch.setattr(
        mesh_generator.gmsh_utils,
        "run_boolean_pipeline",
        lambda _entities: {},
    )

    def _fake_assign_physical_groups(
        _kernel,
        _metal_tags,
        all_dielectric_tags,
        _port_tags,
        _port_info,
        _entities,
        _pg_map,
        _stack,
        pec_block_tags=None,
    ):
        captured["all_dielectric_tags"] = all_dielectric_tags
        captured["assign_pec_block_tags"] = pec_block_tags
        return {
            "volumes": {},
            "conductor_surfaces": {},
            "pec_surfaces": {},
            "port_surfaces": {},
            "boundary_surfaces": {},
        }

    monkeypatch.setattr(
        mesh_generator,
        "assign_physical_groups",
        _fake_assign_physical_groups,
    )
    monkeypatch.setattr(
        mesh_generator, "_setup_mesh_fields", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(mesh_generator, "collect_mesh_stats", lambda: {"nodes": 1})

    stack = LayerStack()
    result = mesh_generator.generate_mesh(
        component=object(),
        stack=stack,
        ports=[],
        output_dir=tmp_path,
        curve_fit_mode="bspline",
        curve_fit_layers=["core", "core2"],
        curve_fit_tolerance_um=0.02,
        curve_fit_min_points=12,
        curve_fit_corner_angle_deg=30.0,
        decimate_tolerance=0.005,
        verbosity=7,
        write_config=False,
    )

    assert captured["decimate_tolerance"] == 0.005
    assert captured["curve_fit"] == {
        "curve_fit_mode": "bspline",
        "curve_fit_layers": ["core", "core2"],
        "curve_fit_tolerance_um": 0.02,
        "curve_fit_min_points": 12,
        "curve_fit_corner_angle_deg": 30.0,
    }
    assert captured["all_dielectric_tags"] == {"air": [31], "core": [21]}
    assert captured["assign_pec_block_tags"] is None
    assert ("General.Verbosity", 7) in fake_gmsh.option.calls
    assert result.mesh_path == tmp_path / "palace.msh"
    assert fake_gmsh.cleared is True
    assert fake_gmsh.finalized is True


def test_generate_palace_config_shaped_dielectric_layer_material(tmp_path) -> None:
    """Shaped dielectrics resolve material properties via the stack layer map."""
    stack = LayerStack()
    stack.layers["CORE"] = Layer(
        name="CORE",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=0.22,
        thickness=0.22,
        material="silicon",
        layer_type="dielectric",
    )
    stack.materials = {
        "silicon": {
            "permittivity": 12.1,
            "loss_tangent": 0.002,
        }
    }

    groups = {
        "volumes": {
            "CORE": {
                "phys_group": 101,
                "is_shaped_dielectric": True,
            }
        },
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }

    config_path = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=stack,
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type="driven",
        absorbing_boundary=False,
    )

    config = json.loads(config_path.read_text())
    materials = config["Domains"]["Materials"]

    core_mat = next(
        (entry for entry in materials if 101 in entry.get("Attributes", [])),
        None,
    )
    assert core_mat is not None
    assert core_mat["Permittivity"] == 12.1
    assert core_mat["LossTan"] == 0.002
