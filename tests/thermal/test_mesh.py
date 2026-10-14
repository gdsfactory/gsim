"""Tests for the thermal cross-section mesher (gsim.thermal.mesh)."""

from __future__ import annotations

import gdsfactory as gf
import meshio
import numpy as np
import pytest

from gsim.common.mesh_regions import element_regions, group_tags
from gsim.common.stack.extractor import Layer, LayerStack
from gsim.common.stack.materials import MATERIALS_DB
from gsim.femwell.adapter import epsilon_by_region, solve_modes

pytest.importorskip("skfem")

from gsim.thermal.mesh import build_cross_section_mesh

CORE_W = 0.5
CORE_T = 0.22
HEATER_W = 2.0
HEATER_ZMIN = 1.5
HEATER_ZMAX = 1.65
SUBSTRATE_THICKNESS = 10.0


@pytest.fixture(autouse=True)
def _generic_pdk_active() -> None:
    gf.gpdk.PDK.activate()


@pytest.fixture
def stack() -> LayerStack:
    """SOI stack: silicon substrate, BOX, cladding, Si core and TiN heater."""
    layers = {
        "core": Layer(
            name="core",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=CORE_T,
            thickness=CORE_T,
            material="silicon",
            layer_type="dielectric",
        ),
        "heater": Layer(
            name="heater",
            gds_layer=(47, 0),
            zmin=HEATER_ZMIN,
            zmax=HEATER_ZMAX,
            thickness=HEATER_ZMAX - HEATER_ZMIN,
            material="TiN",
            layer_type="conductor",
        ),
    }
    materials = {m: MATERIALS_DB[m].to_dict() for m in ("silicon", "SiO2", "TiN")}
    dielectrics = [
        {
            "name": "substrate",
            "zmin": -2.0 - SUBSTRATE_THICKNESS,
            "zmax": -2.0,
            "material": "silicon",
        },
        {"name": "box", "zmin": -2.0, "zmax": 0.0, "material": "SiO2"},
        {"name": "clad", "zmin": 0.0, "zmax": 3.0, "material": "SiO2"},
    ]
    return LayerStack(
        pdk_name="test", layers=layers, materials=materials, dielectrics=dielectrics
    )


@pytest.fixture
def phase_shifter() -> gf.Component:
    """A Si strip under a 2 um TiN heater strip, both running along x."""
    c = gf.Component()
    c.add_polygon(
        [(-5, -CORE_W / 2), (5, -CORE_W / 2), (5, CORE_W / 2), (-5, CORE_W / 2)],
        layer=(1, 0),
    )
    c.add_polygon(
        [
            (-5, -HEATER_W / 2),
            (5, -HEATER_W / 2),
            (5, HEATER_W / 2),
            (-5, HEATER_W / 2),
        ],
        layer=(47, 0),
    )
    return c


def test_every_layer_and_slab_is_a_region_with_named_window_edges(
    phase_shifter, stack, tmp_path
):
    mesh = build_cross_section_mesh(
        phase_shifter, stack, "x=0", output=tmp_path / "thermal.msh"
    )

    assert set(mesh.region_material) == {"core", "heater", "silicon", "SiO2"}
    assert mesh.region_material["heater"] == "TiN"
    assert mesh.region_material["core"] == "silicon"
    assert set(mesh.boundary_facets) == {"bottom", "top", "left", "right"}
    assert mesh.msh_path == tmp_path / "thermal.msh"
    assert mesh.msh_path.exists()
    assert mesh.plane == "x=0"


@pytest.fixture
def thermal_mesh(phase_shifter, stack, tmp_path):
    return build_cross_section_mesh(
        phase_shifter, stack, "x=0", output=tmp_path / "thermal.msh"
    )


def test_region_areas_match_the_drawn_rectangles(thermal_mesh):
    # Default Window: drawn extent (the 2 um heater) plus 20 um each side,
    # from the substrate's own thickness below it to the cladding top.
    width = HEATER_W + 2 * 20.0
    assert thermal_mesh.window == pytest.approx((-21.0, 21.0))
    assert thermal_mesh.z_range == pytest.approx((-12.0, 3.0))

    core = CORE_W * CORE_T
    heater = HEATER_W * (HEATER_ZMAX - HEATER_ZMIN)
    assert thermal_mesh.region_area("core") == pytest.approx(core, rel=1e-6)
    assert thermal_mesh.region_area("heater") == pytest.approx(heater, rel=1e-6)
    assert thermal_mesh.region_area("silicon") == pytest.approx(
        width * SUBSTRATE_THICKNESS, rel=1e-6
    )
    assert thermal_mesh.region_area("SiO2") == pytest.approx(
        width * 5.0 - core - heater, rel=1e-6
    )


def test_region_mask_selects_the_region_triangles(thermal_mesh):
    mask = thermal_mesh.region_mask("heater")
    assert mask.shape == (thermal_mesh.mesh.t.shape[1],)
    assert 0 < mask.sum() < mask.size
    with pytest.raises(ValueError, match="heater"):
        thermal_mesh.region_mask("nope")


def test_boundary_facets_cover_each_window_edge(thermal_mesh):
    p = thermal_mesh.mesh.p
    facets = thermal_mesh.mesh.facets
    (h0, h1), (v0, v1) = thermal_mesh.window, thermal_mesh.z_range
    edges = {
        "bottom": (1, v0, h1 - h0),
        "top": (1, v1, h1 - h0),
        "left": (0, h0, v1 - v0),
        "right": (0, h1, v1 - v0),
    }
    boundary = set(thermal_mesh.mesh.boundary_facets().tolist())
    for name, (coord, at, length) in edges.items():
        idx = thermal_mesh.boundary_facets[name]
        assert set(idx.tolist()) <= boundary
        ends = p[:, facets[:, idx]]  # (2 coords, 2 ends, n facets)
        np.testing.assert_allclose(ends[coord], at, atol=1e-9)
        lengths = np.hypot(*(ends[:, 1] - ends[:, 0]))
        assert lengths.sum() == pytest.approx(length, rel=1e-9)


def test_depth_extends_the_substrate_below_the_box(phase_shifter, stack, tmp_path):
    mesh = build_cross_section_mesh(
        phase_shifter, stack, "x=0", depth=60.0, output=tmp_path / "deep.msh"
    )
    width = mesh.window[1] - mesh.window[0]
    assert mesh.z_range == pytest.approx((-62.0, 3.0))
    assert mesh.region_area("silicon") == pytest.approx(width * 60.0, rel=1e-6)


def test_stack_without_substrate_gets_a_100um_silicon_one(
    phase_shifter, stack, tmp_path, caplog
):
    stack.dielectrics = [d for d in stack.dielectrics if d["name"] != "substrate"]
    mesh = build_cross_section_mesh(
        phase_shifter, stack, "x=0", output=tmp_path / "synthetic.msh"
    )
    width = mesh.window[1] - mesh.window[0]
    assert mesh.z_range == pytest.approx((-102.0, 3.0))
    assert mesh.region_material["silicon"] == "silicon"
    assert mesh.region_area("silicon") == pytest.approx(width * 100.0, rel=1e-6)
    assert "substrate" in caplog.text


def test_small_optical_window_cuts_through_the_slabs(phase_shifter, stack, tmp_path):
    mesh = build_cross_section_mesh(
        phase_shifter,
        stack,
        "x=0",
        window=(-1.5, 1.5),
        z_range=(-1.0, 1.0),
        output=tmp_path / "optical.msh",
    )
    # The heater sits above the Window; the substrate below it.
    assert set(mesh.regions) == {"core", "SiO2"}
    assert mesh.region_area("core") == pytest.approx(CORE_W * CORE_T, rel=1e-6)
    assert mesh.region_area("SiO2") == pytest.approx(
        3.0 * 2.0 - CORE_W * CORE_T, rel=1e-6
    )


def test_window_area_no_layer_claims_is_the_background(phase_shifter, stack, tmp_path):
    mesh = build_cross_section_mesh(
        phase_shifter,
        stack,
        "x=0",
        window=(-3.0, 3.0),
        z_range=(-1.0, 5.0),
        output=tmp_path / "air.msh",
    )
    assert mesh.region_material["air"] == "air"
    assert mesh.region_area("air") == pytest.approx(6.0 * 2.0, rel=1e-6)


@pytest.fixture
def optical_mesh(phase_shifter, stack, tmp_path):
    return build_cross_section_mesh(
        phase_shifter,
        stack,
        "x=0",
        window=(-1.5, 1.5),
        z_range=(-1.0, 1.0),
        output=tmp_path / "optical.msh",
    )


def test_femwell_adapter_reads_the_same_regions_in_the_same_order(optical_mesh, stack):
    msh = meshio.read(str(optical_mesh.msh_path))
    assert set(group_tags(msh, dim=2)) == {"core", "SiO2"}
    assert set(group_tags(msh, dim=1)) == {"bottom", "top", "left", "right"}
    _tris, names = element_regions(msh)
    assert names == optical_mesh.element_region.tolist()

    eps = epsilon_by_region(optical_mesh.msh_path, stack, wavelength_um=1.55)
    assert eps["core"].real == pytest.approx(3.476**2, rel=5e-3)
    assert eps["SiO2"].real == pytest.approx(1.444**2, rel=5e-3)


def test_femwell_solves_the_strip_mode_on_the_optical_window(optical_mesh, stack):
    pytest.importorskip("femwell")
    eps = epsilon_by_region(optical_mesh.msh_path, stack, wavelength_um=1.55)
    modes = solve_modes(optical_mesh.msh_path, epsilon=eps, wavelength_um=1.55)
    # 500 x 220 nm Si strip in oxide, TE0 at 1550 nm.
    assert 2.3 < modes[0].n_eff.real < 2.6


def _longest_edge(mesh, region: str) -> float:
    m = mesh.mesh
    t = m.t[:, mesh.region_mask(region)]
    p = m.p
    return max(
        float(np.hypot(*(p[:, t[i]] - p[:, t[(i + 1) % 3]])).max()) for i in range(3)
    )


def test_deep_window_stays_small_but_fine_at_the_features(
    phase_shifter, stack, tmp_path
):
    mesh = build_cross_section_mesh(
        phase_shifter, stack, "x=0", depth=500.0, output=tmp_path / "deep.msh"
    )
    assert mesh.mesh.t.shape[1] < 50_000
    # A quarter of the smallest feature (the 0.15 um heater, 0.22 um core),
    # with slack for gmsh's sizing tolerance.
    assert _longest_edge(mesh, "heater") <= 1.5 * 0.15 / 4
    assert _longest_edge(mesh, "core") <= 1.5 * CORE_T / 4


def test_max_size_overrides_a_region_default(phase_shifter, stack, tmp_path):
    mesh = build_cross_section_mesh(
        phase_shifter,
        stack,
        "x=0",
        max_size={"silicon": 1.0},
        output=tmp_path / "sized.msh",
    )
    assert _longest_edge(mesh, "silicon") <= 1.5 * 1.0


def test_horizontal_plane_is_rejected(phase_shifter, stack):
    with pytest.raises(ValueError, match="vertical"):
        build_cross_section_mesh(phase_shifter, stack, "z=0.1")


def test_a_boundary_line_that_is_no_mesh_edge_is_an_error(tmp_path):
    # Two triangles of a unit square split along the 1-3 diagonal, and a
    # "bottom" line joining corners 0 and 2, which no triangle edge does.
    from gsim.thermal.mesh import _load

    skfem = pytest.importorskip("skfem")
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    path = tmp_path / "bad.msh"
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [
                ("triangle", np.array([[0, 1, 3], [1, 2, 3]])),
                ("line", np.array([[0, 2]])),
            ],
            cell_data={
                "gmsh:physical": [np.array([1, 1]), np.array([2])],
                "gmsh:geometrical": [np.array([1, 1]), np.array([2])],
            },
            field_data={"slab": np.array([1, 2]), "bottom": np.array([2, 1])},
        ),
        file_format="gmsh22",
        binary=False,
    )
    with pytest.raises(RuntimeError, match="bottom"):
        _load(skfem, path, "x=0", (0.0, 1.0), (0.0, 1.0), {"slab": "oxide"})
