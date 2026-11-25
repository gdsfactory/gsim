"""Zero-thickness conductors in the native 2D cross-section are PEC sheets.

A metal section no thicker than the 3D planar threshold is meshed as a line
on the dielectric interface: the strip and the grounds of a CPW on a
substrate are interior PEC curves that both dielectrics share, as Palace's
own ``cpw2d_thin`` example does. Thicker metal keeps its outline as a hole.
"""

from __future__ import annotations

import json

import gdsfactory as gf
import meshio
import numpy as np
import pytest

from gsim.common.modes import Conductor
from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import BoundaryModeSim
from gsim.palace.line_impedance import line_impedance_paths
from gsim.palace.terminal_ports import TerminalPortSpec, terminal_paths

STRIP_W = 10.0
GAP = 6.0
GROUND_W = 20.0
STRIP_H = (-STRIP_W / 2, STRIP_W / 2)
GROUND_H = (STRIP_W / 2 + GAP, STRIP_W / 2 + GAP + GROUND_W)
TOL = 1e-6


def _component():
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    comp.add_polygon(
        [(-50, STRIP_H[0]), (50, STRIP_H[0]), (50, STRIP_H[1]), (-50, STRIP_H[1])],
        layer=(1, 0),
    )
    for lo, hi in (GROUND_H, (-GROUND_H[1], -GROUND_H[0])):
        comp.add_polygon([(-50, lo), (50, lo), (50, hi), (-50, hi)], layer=(1, 0))
    return comp


def _stack(film_thickness: float, conductivity: float = 0.0) -> LayerStack:
    stack = LayerStack(pdk_name="cpw_sheet", units="um")
    stack.materials = {
        "sub": {"permittivity": 11.45},
        "film": {"conductivity": conductivity},
    }
    stack.layers = {
        "film": Layer(
            name="film",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=film_thickness,
            thickness=film_thickness,
            material="film",
            layer_type="conductor",
        )
    }
    stack.dielectrics = [
        {"name": "sub", "zmin": -300.0, "zmax": 0.0, "material": "sub"}
    ]
    return stack


def _sim(
    tmp_path, film_thickness=0.0, conductivity=0.0, window_z=None
) -> BoundaryModeSim:
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_geometry(_component())
    sim.set_stack(_stack(film_thickness, conductivity))
    sim.set_airbox(margin_x=10.0, margin_y=10.0, z_above=200.0, z_below=0.0)
    sim.set_cross_section("x=0", window_z=window_z)
    sim.set_boundary_mode(freq=5e9, num_modes=2)
    sim.mesh(preset="coarse", verbose=False)
    return sim


def _names(mesh: meshio.Mesh) -> dict[int, str]:
    return {int(tag): name for name, (tag, _dim) in mesh.field_data.items()}


def _lines(mesh: meshio.Mesh, name: str) -> np.ndarray:
    """Endpoints ``(n, 2, 2)`` of the line elements in one physical group."""
    tag = int(mesh.field_data[name][0])
    cells = mesh.cells_dict["line"]
    physical = mesh.cell_data_dict["gmsh:physical"]["line"]
    return mesh.points[cells[physical == tag]][:, :, :2]


def _lengths(segments: np.ndarray) -> np.ndarray:
    return np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)


def test_strip_and_grounds_are_interior_curves_with_their_extent(tmp_path):
    sim = _sim(tmp_path)
    segments = _lines(meshio.read(sim.mesh_path), "film_pec")
    h = segments[..., 0]
    assert np.allclose(segments[..., 1], 0.0)
    assert _lengths(segments).sum() == pytest.approx(STRIP_W + 2 * GROUND_W)

    in_strip = (h >= STRIP_H[0] - TOL).all(axis=1) & (h <= STRIP_H[1] + TOL).all(axis=1)
    assert _lengths(segments[in_strip]).sum() == pytest.approx(STRIP_W)
    for lo, hi in (GROUND_H, (-GROUND_H[1], -GROUND_H[0])):
        in_ground = (h >= min(lo, hi) - TOL).all(axis=1) & (h <= max(lo, hi) + TOL).all(
            axis=1
        )
        assert _lengths(segments[in_ground]).sum() == pytest.approx(GROUND_W)


def test_sheets_are_pec_in_the_config_and_not_conductive(tmp_path):
    sim = _sim(tmp_path, conductivity=4.1e7)
    sim.write_config()
    mesh = meshio.read(sim.mesh_path)
    config = json.loads((tmp_path / "config.json").read_text())
    tag = int(mesh.field_data["film_pec"][0])
    assert tag in config["Boundaries"]["PEC"]["Attributes"]
    assert "Conductivity" not in config["Boundaries"]


def test_each_sheet_edge_is_shared_by_substrate_and_vacuum(tmp_path):
    sim = _sim(tmp_path)
    mesh = meshio.read(sim.mesh_path)
    names = _names(mesh)
    triangles = mesh.cells_dict["triangle"]
    triangle_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    sheet_nodes = mesh.cells_dict["line"][
        mesh.cell_data_dict["gmsh:physical"]["line"] == mesh.field_data["film_pec"][0]
    ]
    assert len(sheet_nodes) > 0
    for a, b in sheet_nodes:
        touching = [i for i, t in enumerate(triangles) if a in t and b in t]
        assert len(touching) == 2
        assert {names[int(triangle_tags[i])] for i in touching} == {"sub", "air"}


@pytest.mark.parametrize("film_thickness", [0.0, 0.01])
def test_thin_metal_is_a_sheet_at_its_bottom_face(tmp_path, film_thickness):
    sim = _sim(tmp_path, film_thickness=film_thickness)
    segments = _lines(meshio.read(sim.mesh_path), "film_pec")
    assert np.allclose(segments[..., 1], 0.0)
    assert _lengths(segments).sum() == pytest.approx(STRIP_W + 2 * GROUND_W)


@pytest.mark.parametrize("film_thickness", [0.02, 0.5])
def test_metal_at_the_threshold_or_thicker_keeps_its_outline_as_a_hole(
    tmp_path, film_thickness
):
    sim = _sim(tmp_path, film_thickness=film_thickness)
    segments = _lines(meshio.read(sim.mesh_path), "film_pec")
    vertical = np.isclose(segments[..., 0], STRIP_H[0]).all(axis=1) | np.isclose(
        segments[..., 0], STRIP_H[1]
    ).all(axis=1)
    assert vertical.any()
    top = segments[vertical][:, :, 1]
    assert top.min() == pytest.approx(0.0)
    assert top.max() == pytest.approx(film_thickness)


def test_a_sheet_on_the_domain_wall_is_pec_and_not_absorbing(tmp_path):
    sim = _sim(tmp_path, window_z=(-300.0, 0.0))
    mesh = meshio.read(sim.mesh_path)
    sheet = _lines(mesh, "film_pec")
    assert np.allclose(sheet[..., 1], 0.0)
    assert _lengths(sheet).sum() == pytest.approx(STRIP_W + 2 * GROUND_W)
    wall = _lines(mesh, "absorbing")
    floor = wall[(wall[..., 1] == 0.0).all(axis=1)]
    midpoints = floor[..., 0].mean(axis=1)
    metal = (STRIP_H, GROUND_H, (-GROUND_H[1], -GROUND_H[0]))
    for h in midpoints:
        assert not any(lo < h < hi for lo, hi in metal)


def test_impedance_paths_for_a_sheet_signal_enclose_it(tmp_path):
    sim = _sim(tmp_path)
    paths = line_impedance_paths(
        signal=(STRIP_H, (0.0, 0.0)),
        ground=(GROUND_H, (0.0, 0.0)),
        domain=sim.mesh_extent,
    )
    (h_start, v_start), (h_end, v_end) = paths.voltage
    assert v_start == v_end == 0.0
    for h in (h_start, h_end):
        assert STRIP_H[1] < h < GROUND_H[0]
    corners = np.asarray(paths.current)
    assert corners[:, 1].min() < 0.0 < corners[:, 1].max()
    assert corners[:, 0].min() < STRIP_H[0]
    assert corners[:, 0].max() > STRIP_H[1]
    on_metal = (corners[:, 1] == 0.0) & (np.abs(corners[:, 0]) <= STRIP_H[1])
    assert not on_metal.any()


def test_terminal_paths_stay_in_the_gap_between_sheets():
    signal = Conductor("S", (STRIP_H, (0.0, 0.0)))
    ground = Conductor("G", (GROUND_H, (0.0, 0.0)))
    spec = TerminalPortSpec("p", terminals=[signal], reference=ground)
    ((_, points),) = terminal_paths(spec, ((-41.0, 41.0), (-300.0, 200.0))).items()
    for h, v in points:
        assert v == 0.0
        assert STRIP_H[1] < h < GROUND_H[0]


def test_sheet_impedance_replaces_pec_on_2d_sheets(tmp_path):
    sim = _sim(tmp_path)
    sim.add_sheet_impedance("film", inductance=20e-12)
    sim.write_config()
    mesh = meshio.read(sim.mesh_path)
    config = json.loads((tmp_path / "config.json").read_text())
    tag = int(mesh.field_data["film_pec"][0])
    boundaries = config["Boundaries"]
    assert boundaries["Impedance"] == [{"Attributes": [tag], "Ls": 20e-12}]
    assert tag not in boundaries.get("PEC", {}).get("Attributes", [])
