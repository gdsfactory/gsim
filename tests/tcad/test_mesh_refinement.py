"""A refinement box holds the mesh to a size over an area, not along lines."""

from __future__ import annotations

import pytest

from gsim.tcad import Simulation
from tests._helpers import longest_edge_in_box

from .conftest import build_contacted_diode

#: Around the diode's Junction, the height of the slab: (h, h, z, z, size) in um.
BOX = (-20.15, -19.85, 0.0, 0.22, 0.01)


def _longest_edge_in_box(tmp_path, **solver_settings) -> float:
    comp, stack = build_contacted_diode()
    sim = Simulation()
    sim.geometry(comp, stack=stack)
    sim.domain(
        plane="x=0",
        window=(-21.1, -18.9),
        margin_x=2.0,
        margin_y=2.0,
        z_above=1.5,
        z_below=1.0,
    )
    sim.contacts.add(name="cathode", semiconductor="n_pad", electrode="cathode_metal")
    sim.contacts.add(name="anode", semiconductor="p_pad", electrode="anode_metal")
    sim.interfaces.add(name="junction", region_a="n_rib", region_b="p_rib")
    sim.solver(
        mesh_preset="coarse",
        refined_mesh_size_um=0.05,
        max_mesh_size_um=40.0,
        **solver_settings,
    )
    mesh = sim.mesh(output_dir=tmp_path)
    return longest_edge_in_box(mesh.mesh_path, BOX[:2], BOX[2:4])


def test_elements_inside_a_refinement_box_keep_to_its_size(tmp_path):
    # Refinement lines size the elements on them, and the size grows at
    # once with the distance: mid-slab, 0.1 um from any line, a 0.05 um
    # mesh is several times its refined size. A depletion edge moves there.
    without = _longest_edge_in_box(tmp_path / "lines")
    within = _longest_edge_in_box(tmp_path / "box", refinement_boxes=[BOX])

    # gmsh takes a size as a target: an edge runs up to about twice it.
    assert without > 5.0 * BOX[4]
    assert within < 2.5 * BOX[4]


def test_a_refinement_box_needs_a_positive_size(tmp_path):
    with pytest.raises(ValueError, match="refinement"):
        _longest_edge_in_box(tmp_path, refinement_boxes=[(*BOX[:4], 0.0)])


def test_a_refinement_box_is_a_mesh_setting():
    """A box changes the mesh, so it is one of the solver's mesh fields."""
    from gsim.tcad.api import MESH_FIELDS

    assert "refinement_boxes" in MESH_FIELDS


class TestTheOptionOnTheMeshConfig:
    """The option is a setting like the others: validated, and kept."""

    @staticmethod
    def _build(sim, **overrides):
        settings = {
            "preset": "coarse",
            "refined_mesh_size": None,
            "max_mesh_size": None,
            "margin": None,
            "airbox_margin": None,
            "fmax": None,
            "planar_conductors": None,
            "show_gui": False,
        }
        return sim._build_mesh_config(**(settings | overrides))

    def test_a_box_is_refused_where_it_is_configured(self):
        from gsim.palace.models import MeshConfig

        with pytest.raises(ValueError, match="refinement"):
            MeshConfig(refinement_boxes=[(*BOX[:4], -1.0)])
        with pytest.raises(ValueError, match="refinement"):
            MeshConfig(refinement_boxes=[(BOX[1], BOX[0], *BOX[2:])])

    def test_boxes_on_the_sims_mesh_config_survive_a_mesh_call(self):
        from gsim.palace import BoundaryModeSim
        from gsim.palace.models import MeshConfig

        sim = BoundaryModeSim()
        sim.mesh_config = MeshConfig(refinement_boxes=[BOX])

        assert self._build(sim).refinement_boxes == [BOX]

    def test_boxes_given_to_the_call_replace_them(self):
        from gsim.palace import BoundaryModeSim
        from gsim.palace.models import MeshConfig

        sim = BoundaryModeSim()
        sim.mesh_config = MeshConfig(refinement_boxes=[BOX])

        assert self._build(sim, refinement_boxes=[]).refinement_boxes == []
