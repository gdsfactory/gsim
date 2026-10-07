"""A refinement box holds the mesh to a size over an area, not along lines."""

from __future__ import annotations

import pytest

from gsim.modulator.demo import demo_phase_shifter
from gsim.tcad import ChargeTransportSim
from tests._helpers import longest_edge_in_box

#: Around the demo Junction, the height of the slab: (h, h, z, z, size) in um.
BOX = (-20.15, -19.85, 0.0, 0.22, 0.01)


def _longest_edge_in_box(tmp_path, **mesh_kwargs) -> float:
    demo = demo_phase_shifter()
    sim = ChargeTransportSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_stack(demo.stack)
    sim.set_airbox(margin_x=2.0, margin_y=2.0, z_above=1.5, z_below=1.0)
    sim.set_geometry(demo.component)
    sim.set_cross_section("x=0", window=(-21.1, -18.9))
    sim.add_contact(name="cathode", layer_a="n_pad", layer_b="cathode_metal")
    sim.add_contact(name="anode", layer_a="p_pad", layer_b="anode_metal")
    sim.add_interface(name="junction", layer_a="n_rib", layer_b="p_rib")
    sim.mesh(
        preset="coarse",
        refined_mesh_size=0.05,
        max_mesh_size=40.0,
        verbose=False,
        **mesh_kwargs,
    )
    return longest_edge_in_box(sim.mesh_path, BOX[:2], BOX[2:4])


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
