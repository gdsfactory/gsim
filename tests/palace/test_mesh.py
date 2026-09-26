"""Tests for mesh configuration."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from gsim.common.stack import LayerStack
from gsim.common.stack.extractor import Layer
from gsim.palace.mesh import MeshConfig
from gsim.palace.mesh import validation as mesh_validation
from gsim.palace.mesh.geometry import (
    GeometryData,
    _snap_via_z_range,
    add_dielectrics,
    add_metals,
    add_ports,
    get_layer_info,
    get_layer_infos,
)
from gsim.palace.ports.config import PalacePort, PortGeometry, PortType


class TestMeshConfig:
    """Test MeshConfig class."""

    def test_default_config(self):
        """Test default MeshConfig values."""
        config = MeshConfig()
        assert config.refined_mesh_size == 5.0
        assert config.max_mesh_size == 300.0
        assert config.margin == 0.0
        assert config.fmax == 100e9
        assert config.show_gui is False
        assert config.boundary_conditions is not None
        assert len(config.boundary_conditions) == 6
        assert config.curve_fit_mode == "line"
        assert config.curve_fit_layers == ["core", "core2"]
        assert config.curve_fit_tolerance_um == 0.0
        assert config.curve_fit_min_points == 8
        assert config.curve_fit_corner_angle_deg == 45.0
        assert config.high_order_elements is False
        assert config.high_order_order == 2
        assert config.high_order_optimize is True

    def test_coarse_preset(self):
        """Test coarse mesh preset."""
        config = MeshConfig.coarse()
        assert config.refined_mesh_size == 10.0
        assert config.max_mesh_size == 600.0
        assert config.cells_per_wavelength == 5

    def test_default_preset(self):
        """Test default mesh preset."""
        config = MeshConfig.default()
        assert config.refined_mesh_size == 5.0
        assert config.max_mesh_size == 300.0
        assert config.cells_per_wavelength == 10

    def test_fine_preset(self):
        """Test fine mesh preset."""
        config = MeshConfig.fine()
        assert config.refined_mesh_size == 2.0
        assert config.max_mesh_size == 70.0
        assert config.cells_per_wavelength == 20

    def test_preset_with_overrides(self):
        """Test preset with custom overrides."""
        config = MeshConfig.coarse(margin=100.0, fmax=50e9)
        assert config.refined_mesh_size == 10.0  # From preset
        assert config.margin == 100.0  # Override
        assert config.fmax == 50e9  # Override

    def test_custom_config(self):
        """Test fully custom config."""
        config = MeshConfig(
            refined_mesh_size=3.0,
            max_mesh_size=200.0,
            margin=75.0,
        )
        assert config.refined_mesh_size == 3.0
        assert config.max_mesh_size == 200.0
        assert config.margin == 75.0

    def test_curve_fit_overrides(self):
        """Test custom curve-fit settings."""
        config = MeshConfig(
            curve_fit_mode="bspline",
            curve_fit_layers=["core"],
            curve_fit_tolerance_um=0.02,
            curve_fit_min_points=12,
            curve_fit_corner_angle_deg=30.0,
        )
        assert config.curve_fit_mode == "bspline"
        assert config.curve_fit_layers == ["core"]
        assert config.curve_fit_tolerance_um == 0.02
        assert config.curve_fit_min_points == 12
        assert config.curve_fit_corner_angle_deg == 30.0

    def test_high_order_overrides(self):
        """Test custom high-order mesh settings."""
        config = MeshConfig(
            high_order_elements=True,
            high_order_order=3,
            high_order_optimize=False,
        )
        assert config.high_order_elements is True
        assert config.high_order_order == 3
        assert config.high_order_optimize is False


def test_add_dielectrics_margin_applies_only_to_airlike(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_box(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_box", _fake_create_box
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        dielectrics=[
            {"name": "oxide", "zmin": -2.0, "zmax": 0.5, "material": "SiO2"},
            {"name": "air", "zmin": 0.5, "zmax": 8.0, "material": "air"},
            {"name": "passivation", "zmin": 8.0, "zmax": 9.0, "material": "passive"},
        ],
        materials={
            "SiO2": {"permittivity": 4.1},
            "air": {"permittivity": 1.0},
            "passive": {"permittivity": 6.6},
        },
    )

    add_dielectrics(
        _Kernel(), geometry, stack, margin_x=5.0, margin_y=7.0, air_margin=0.0
    )

    # Only air-like regions expand by margins.
    # Non-air dielectrics keep original design footprint.
    assert calls[0] == (0.0, 0.0, -2.0, 10.0, 20.0, 0.5)
    assert calls[1] == (-5.0, -7.0, 0.5, 15.0, 27.0, 8.0)
    assert calls[2] == (0.0, 0.0, 8.0, 10.0, 20.0, 9.0)


def test_add_dielectrics_explicit_airbox_z_extents(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_box(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_box", _fake_create_box
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        dielectrics=[
            {"name": "oxide", "zmin": -2.0, "zmax": 0.5, "material": "SiO2"},
            {"name": "air", "zmin": 0.5, "zmax": 8.0, "material": "air"},
        ],
        materials={
            "SiO2": {"permittivity": 4.1},
            "air": {"permittivity": 1.0},
        },
    )

    tags = add_dielectrics(
        _Kernel(),
        geometry,
        stack,
        margin_x=5.0,
        margin_y=7.0,
        air_margin=0.0,
        airbox_z_above=100.0,
        airbox_z_below=100.0,
    )

    assert "airbox" in tags
    # Air layer is skipped when explicit airbox is built.
    assert list(tags.keys()) == ["SiO2", "airbox"]


def test_add_dielectrics_explicit_airbox_skips_named_air_regions(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_box(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_box", _fake_create_box
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        dielectrics=[
            {"name": "oxide", "zmin": -2.0, "zmax": 0.5, "material": "SiO2"},
            {
                "name": "air_box_top",
                "zmin": 0.5,
                "zmax": 8.0,
                "material": "ambient",
            },
        ],
        materials={
            "SiO2": {"type": "dielectric", "permittivity": 4.1},
            # Intentionally non-air metadata to force name-based detection.
            "ambient": {"type": "unknown"},
        },
    )

    tags = add_dielectrics(
        _Kernel(),
        geometry,
        stack,
        margin_x=5.0,
        margin_y=7.0,
        air_margin=0.0,
        airbox_z_above=100.0,
        airbox_z_below=100.0,
    )

    assert list(tags.keys()) == ["SiO2", "airbox"]
    # Oxide + explicit airbox only.
    assert len(calls) == 2


def test_add_ports_waveport_max_size_uses_3d_domain_bounds(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_port_rectangle(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return 77

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_port_rectangle",
        _fake_create_port_rectangle,
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    stack = LayerStack(
        layers={
            "metal1": _mk_layer("metal1", 1.0, 2.0, "conductor"),
        }
    )
    port = PalacePort(
        name="o1",
        port_type=PortType.WAVEPORT,
        geometry=PortGeometry.INPLANE,
        center=(5.0, 0.0),
        width=4.0,
        orientation=0.0,
        layer="metal1",
        max_size=True,
    )

    port_tags, port_info = add_ports(
        _Kernel(),
        [port],
        stack,
        domain_bounds=(-100.0, -60.0, -20.0, 120.0, 80.0, 40.0),
    )

    assert port_tags == {"P1": [77]}
    assert calls == [(5.0, -60.0, -20.0, 5.0, 80.0, 40.0)]
    assert port_info[0]["zmin"] == -20.0
    assert port_info[0]["zmax"] == 40.0
    assert port_info[0]["ymin"] == -60.0
    assert port_info[0]["ymax"] == 80.0


def _mk_layer(
    name: str,
    zmin: float,
    zmax: float,
    ltype: Literal["conductor", "via", "dielectric", "substrate"] = "conductor",
) -> Layer:
    return Layer(
        name=name,
        gds_layer=(0, 0),
        zmin=zmin,
        zmax=zmax,
        thickness=zmax - zmin,
        material="aluminum",
        layer_type=ltype,
    )


def _mk_mim_stack() -> LayerStack:
    """Return a stack where one GDS layer backs two LayerLevels (IHP MIM).

    Both ``mim_diel`` (40 nm high-k dielectric) and ``mim`` (top-plate
    conductor) live on GDS layer 10 (``MIMdrawing``). Each must be meshed at
    its own z-height, which is the reason ``get_layer_infos`` exists.
    """
    return LayerStack(
        layers={
            "mim_diel": Layer(
                name="mim_diel",
                gds_layer=(10, 0),
                zmin=5.0,
                zmax=5.04,
                thickness=0.04,
                material="HIK",
                layer_type="dielectric",
            ),
            "mim": Layer(
                name="mim",
                gds_layer=(10, 0),
                zmin=5.04,
                zmax=5.14,
                thickness=0.1,
                material="aluminum",
                layer_type="conductor",
            ),
            "topmetal1": Layer(
                name="topmetal1",
                gds_layer=(11, 0),
                zmin=6.0,
                zmax=8.0,
                thickness=2.0,
                material="aluminum",
                layer_type="conductor",
            ),
        }
    )


class TestGetLayerInfos:
    """A single GDS layer may back multiple stack LayerLevels."""

    def test_returns_all_matching_layers_in_stack_order(self):
        stack = _mk_mim_stack()
        infos = get_layer_infos(stack, 10)
        assert [info["name"] for info in infos] == ["mim_diel", "mim"]
        assert infos[0]["type"] == "dielectric"
        assert infos[0]["zmin"] == pytest.approx(5.0)
        assert infos[0]["zmax"] == pytest.approx(5.04)
        assert infos[0]["thickness"] == pytest.approx(0.04)
        assert infos[1]["type"] == "conductor"
        assert infos[1]["zmin"] == pytest.approx(5.04)
        assert infos[1]["zmax"] == pytest.approx(5.14)
        assert infos[1]["thickness"] == pytest.approx(0.1)

    def test_legacy_single_lookup_returns_only_first(self):
        stack = _mk_mim_stack()
        info = get_layer_info(stack, 10)
        assert info is not None
        assert info["name"] == "mim_diel"
        assert info["type"] == "dielectric"

    def test_unmatched_layer_returns_empty_list(self):
        stack = _mk_mim_stack()
        assert get_layer_infos(stack, 999) == []
        assert get_layer_info(stack, 999) is None

    def test_layers_on_distinct_gds_numbers_are_isolated(self):
        stack = _mk_mim_stack()
        infos = get_layer_infos(stack, 11)
        assert [info["name"] for info in infos] == ["topmetal1"]


class TestAddMetalsSharedGdsLayer:
    """Both LayerLevels on a shared GDS layer must be meshed at own z."""

    def test_meshes_dielectric_and_conductor_on_same_gds_layer(self):
        import gmsh

        stack = _mk_mim_stack()
        geometry = GeometryData(
            polygons=[(10, [0.0, 10.0, 10.0, 0.0], [0.0, 0.0, 10.0, 10.0], [])],
            bbox=(0.0, 0.0, 10.0, 10.0),
            layer_bboxes={},
        )

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.model.add("mim_shared_layer")
            kernel = gmsh.model.occ

            tags = add_metals(kernel, geometry, stack)
            kernel.synchronize()

            assert "mim_diel" in tags
            assert "mim" in tags

            # mim_diel is a shaped dielectric -> solid 3D volume
            mim_diel_vols = tags["mim_diel"]["volumes"]
            assert mim_diel_vols
            for vol in mim_diel_vols:
                bbox = kernel.getBoundingBox(3, vol)
                assert bbox[2] == pytest.approx(5.0, abs=0.01)
                assert bbox[5] == pytest.approx(5.04, abs=0.01)

            # mim conductor -> shell volume (tuple tag) at its own z-height
            mim_vols = tags["mim"]["volumes"]
            assert mim_vols
            assert isinstance(mim_vols[0], tuple)
            # shell loop surfaces bound the conductor z-range [5.04, 5.14]
            loop = mim_vols[0][1]
            zmin_surfs = [kernel.getBoundingBox(2, s) for s in loop]
            assert any(
                abs(bbox[2] - 5.04) < 0.01 and abs(bbox[5] - 5.04) < 0.01
                for bbox in zmin_surfs
            )
            assert any(
                abs(bbox[2] - 5.14) < 0.01 and abs(bbox[5] - 5.14) < 0.01
                for bbox in zmin_surfs
            )
        finally:
            gmsh.clear()
            gmsh.finalize()

    def test_unrelated_gds_layer_untouched(self):
        import gmsh

        stack = _mk_mim_stack()
        geometry = GeometryData(
            polygons=[
                (10, [0.0, 10.0, 10.0, 0.0], [0.0, 0.0, 10.0, 10.0], []),
                (11, [0.0, 5.0, 5.0, 0.0], [0.0, 0.0, 5.0, 5.0], []),
            ],
            bbox=(0.0, 0.0, 10.0, 10.0),
            layer_bboxes={},
        )

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.model.add("mim_shared_layer")
            kernel = gmsh.model.occ

            tags = add_metals(kernel, geometry, stack)
            kernel.synchronize()

            assert "topmetal1" in tags
            assert tags["topmetal1"]["volumes"]
        finally:
            gmsh.clear()
            gmsh.finalize()


class TestSnapViaZRange:
    """Snap via z-range to avoid sliver volumes against adjacent conductors."""

    def test_snaps_top_overlap_with_conductor(self):
        # Mimics IHP vmim z=[5.58, 6.24] vs topmetal1 z=[6.23, 8.23]
        stack = LayerStack(
            layers={
                "vmim": _mk_layer("vmim", 5.58, 6.24, "via"),
                "topmetal1": _mk_layer("topmetal1", 6.23, 8.23, "conductor"),
            }
        )
        new_zmin, new_zmax = _snap_via_z_range(stack, "vmim", 5.58, 6.24)
        assert new_zmin == pytest.approx(5.58)
        assert new_zmax == pytest.approx(6.23)

    def test_does_not_snap_large_overlap(self):
        # Overlap > tol must be left alone (likely a real geometric intersection).
        stack = LayerStack(
            layers={
                "via": _mk_layer("via", 0.0, 1.0, "via"),
                "metal": _mk_layer("metal", 0.5, 2.0, "conductor"),
            }
        )
        new_zmin, new_zmax = _snap_via_z_range(stack, "via", 0.0, 1.0, tol=0.05)
        assert (new_zmin, new_zmax) == (0.0, 1.0)

    def test_no_overlap_unchanged(self):
        stack = LayerStack(
            layers={
                "via": _mk_layer("via", 1.0, 2.0, "via"),
                # touches, no overlap
                "metal": _mk_layer("metal", 2.0, 3.0, "conductor"),
            }
        )
        new_zmin, new_zmax = _snap_via_z_range(stack, "via", 1.0, 2.0)
        assert (new_zmin, new_zmax) == (1.0, 2.0)

    def test_snaps_bottom_overlap_with_conductor(self):
        stack = LayerStack(
            layers={
                "via": _mk_layer("via", 1.99, 3.0, "via"),
                "metal": _mk_layer("metal", 1.0, 2.0, "conductor"),
            }
        )
        new_zmin, new_zmax = _snap_via_z_range(stack, "via", 1.99, 3.0)
        assert new_zmin == pytest.approx(2.0)
        assert new_zmax == pytest.approx(3.0)


class TestValidationHelpers:
    """Test helper routines used by mesh validation."""

    def test_parse_direction(self):
        """Direction parser returns normalized vectors and rejects unknown keys."""
        vec = mesh_validation._parse_direction("+X")
        assert np.allclose(vec, np.array([1.0, 0.0, 0.0]))

        with pytest.raises(ValueError, match="Unknown port direction"):
            mesh_validation._parse_direction("north")

    def test_perp_dist(self):
        """Perpendicular distance to an axis-aligned line is computed correctly."""
        v = np.array([0.0, 3.0, 4.0])
        origin = np.zeros(3)
        normals = [np.array([1.0, 0.0, 0.0])]
        dist = mesh_validation._perp_dist(v, normals, origin)
        assert dist == pytest.approx(5.0)

    def test_palace_obb_planar_rectangle(self):
        """OBB helper handles a simple planar rectangle point cloud."""
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        _center, axes, planar = mesh_validation._palace_obb(pts)
        lengths = [2.0 * float(np.linalg.norm(ax)) for ax in axes]
        assert planar is True
        assert max(lengths) == pytest.approx(2.0)
        assert min(lengths) == pytest.approx(0.0)
