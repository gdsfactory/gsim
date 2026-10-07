"""Unit tests for curved meshing pipeline behavior."""

from __future__ import annotations

import builtins
import json
from types import SimpleNamespace

import gmsh
import pytest

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
    monkeypatch.setattr(
        mesh_generator, "collect_mesh_stats", lambda **_kwargs: {"nodes": 1}
    )

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


class _FakeKernel:
    """Minimal OCC-ish kernel recording polygon surface creation calls."""

    def __init__(self) -> None:
        self.surface_calls: list[dict[str, object]] = []
        self.wire_loop_calls: list[dict[str, object]] = []

    def synchronize(self) -> None:
        return

    def removeAllDuplicates(self) -> None:  # noqa: N802 (gmsh API)
        return

    def getBoundingBox(self, dim: int, tag: int) -> tuple:  # noqa: N802 (gmsh API)
        del dim, tag
        return (0.0, 0.0, 0.0, 1.0, 1.0, 0.0)

    def getEntities(self, dim: int):  # noqa: N802 (gmsh API)
        del dim
        return []

    def remove(self, *_args, **_kwargs) -> None:
        return


def test_add_metals_forwards_curve_fit_for_selected_conductor_layers(
    monkeypatch,
) -> None:
    """Conductor layers in curve_fit_layers get spline/bspline surface boundaries."""
    from gsim.palace.mesh import geometry as mesh_geometry
    from gsim.palace.mesh.geometry import GeometryData

    stack = LayerStack()
    stack.layers["CORE"] = Layer(
        name="CORE",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=0.0,
        thickness=0.0,
        material="aluminum",
        layer_type="conductor",
    )
    stack.layers["OTHER"] = Layer(
        name="OTHER",
        gds_layer=(2, 0),
        zmin=0.0,
        zmax=0.0,
        thickness=0.0,
        material="aluminum",
        layer_type="conductor",
    )
    stack.materials = {"aluminum": {"conductivity": 3.77e7}}

    geometry = GeometryData(
        polygons=[
            (1, [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], []),
            (2, [0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 2.0, 2.0], []),
        ],
        bbox=(0.0, 0.0, 2.0, 2.0),
        layer_bboxes={},
    )

    monkeypatch.setattr(
        mesh_geometry, "_detect_shaped_dielectric_layers", lambda *_a, **_k: set()
    )
    monkeypatch.setattr(mesh_geometry, "_merge_via_polygons", lambda polys, _d: polys)

    kernel = _FakeKernel()
    captured: list[dict[str, object]] = []

    def _fake_create_polygon_surface(_kernel, _x, _y, _z, **kwargs):
        captured.append(kwargs)
        return len(captured)

    monkeypatch.setattr(
        mesh_geometry.gmsh_utils,
        "create_polygon_surface",
        _fake_create_polygon_surface,
    )
    monkeypatch.setattr(
        mesh_geometry.gmsh_utils,
        "_create_wire_loop",
        lambda _k, *_a, **kw: captured.append({**kw, "_wire": True}) or 1,
    )

    mesh_geometry.add_metals(
        kernel,
        geometry,
        stack,
        planar_conductors=True,
        curve_fit_mode="bspline",
        curve_fit_layers=["CORE"],
        curve_fit_tolerance_um=0.02,
        curve_fit_min_points=12,
        curve_fit_corner_angle_deg=30.0,
    )

    surface_modes = [c.get("loop_mode") for c in captured if not c.get("_wire")]
    # CORE (layer 1) uses bspline; OTHER (layer 2) falls back to straight lines.
    assert surface_modes == ["bspline", "line"]
    for call in captured:
        if call.get("_wire"):
            assert call["point_merge_tol"] == 0.02
            assert call["corner_turn_threshold_deg"] == 30.0


@pytest.mark.parametrize(("geometry_order", "element_type"), [(1, 4), (2, 11), (3, 29)])
def test_collect_mesh_stats_handles_high_order_tetrahedra(
    geometry_order, element_type
) -> None:
    """Tetrahedra of every order retain quality and field DOF statistics."""
    from gsim.palace.mesh import config_generator

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("high_order_stats")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
        gmsh.model.mesh.generate(3)
        linear_stats = config_generator.collect_mesh_stats(field_order=3)
        gmsh.model.mesh.setOrder(geometry_order)

        stats = config_generator.collect_mesh_stats(field_order=3)

        assert stats["tetrahedra"] == linear_stats["tetrahedra"] > 0
        assert stats["element_type"] == element_type
        assert stats["geometry_orders"] == [geometry_order]
        assert stats["quality"]["mean"] > 0
        assert stats["sicn"]["invalid"] == 0
        assert stats["edge_length"]["min"] > 0
        assert stats["topology"] == linear_stats["topology"]
        assert stats["field_dofs"] == linear_stats["field_dofs"]
    finally:
        gmsh.finalize()
