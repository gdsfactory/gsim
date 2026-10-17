"""Tests for named contact and interface line groups in the native-2D mesh.

Contacts and interfaces are declared as layer pairs; the shared curves
between the two layers' meshed regions become a dim-1 physical group
carrying the declared name, so DEVSIM's ``add_gmsh_contact`` and
``add_gmsh_interface`` can bind to them. The two are recorded apart.
"""

from __future__ import annotations

import meshio
import numpy as np
import pytest
from pydantic import ValidationError

from gsim.palace import BoundaryModeSim
from gsim.palace.models import ContactSpec, InterfaceSpec
from tests._helpers import draw_pn_rib


def _build_pn_device():
    """Rib with adjacent P/N doped regions (touching halves)."""
    return draw_pn_rib()


def _make_sim(tmp_path, contacts, interfaces=()):
    comp, stack = _build_pn_device()
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    sim.set_geometry(comp)
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=50e9, num_modes=1)
    for contact in contacts:
        sim.add_contact(**contact)
    for interface in interfaces:
        sim.add_interface(**interface)
    sim.mesh(preset="coarse", refined_mesh_size=0.05, max_mesh_size=40.0, verbose=False)
    return sim


class TestContactSpecModel:
    def test_fields(self):
        spec = ContactSpec(name="anode", layer_a="metal1", layer_b="p_rib")
        assert spec.name == "anode"

    def test_rejects_same_layer(self):
        with pytest.raises(ValidationError):
            ContactSpec(name="bad", layer_a="p_rib", layer_b="p_rib")

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            ContactSpec(name="", layer_a="a", layer_b="b")


class TestInterfaceSpecModel:
    def test_fields(self):
        spec = InterfaceSpec(name="junction", layer_a="p_rib", layer_b="n_rib")
        assert spec.name == "junction"

    def test_rejects_same_layer(self):
        with pytest.raises(ValidationError):
            InterfaceSpec(name="bad", layer_a="p_rib", layer_b="p_rib")


class TestInterfaceLineGroups:
    def test_an_interface_is_tagged_apart_from_the_contacts(self, tmp_path):
        sim = _make_sim(
            tmp_path,
            [
                {"name": "anode", "layer_a": "p_rib", "layer_b": "sio2"},
                {"name": "cathode", "layer_a": "n_rib", "layer_b": "sio2"},
            ],
            [{"name": "junction", "layer_a": "p_rib", "layer_b": "n_rib"}],
        )
        groups = sim.mesh_groups
        assert set(groups["contact_lines"]) == {"anode", "cathode"}
        assert set(groups["interface_lines"]) == {"junction"}
        assert groups["interface_lines"]["junction"]["tags"]
        field_data = meshio.read(sim.mesh_path).field_data
        assert int(np.asarray(field_data["junction"])[1]) == 1

    def test_a_nontouching_interface_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"Interface 'far'.*do not touch"):
            _make_sim(
                tmp_path,
                [{"name": "anode", "layer_a": "p_rib", "layer_b": "sio2"}],
                [{"name": "far", "layer_a": "p_rib", "layer_b": "air"}],
            )


class TestContactLineGroups:
    def test_contact_group_in_mesh(self, tmp_path):
        sim = _make_sim(
            tmp_path,
            [{"name": "anode", "layer_a": "p_rib", "layer_b": "n_rib"}],
        )
        contact_lines = sim.mesh_groups["contact_lines"]
        assert "anode" in contact_lines
        assert contact_lines["anode"]["tags"]

        mesh = meshio.read(sim.mesh_path)
        field_data = mesh.field_data
        assert "anode" in field_data
        dim = int(np.asarray(field_data["anode"])[1])
        assert dim == 1
        # Lines with that physical tag actually exist in the mesh.
        tag = int(np.asarray(field_data["anode"])[0])
        line_tags = np.concatenate(
            [
                arr
                for cell_block, arr in zip(
                    mesh.cells, mesh.cell_data["gmsh:physical"], strict=True
                )
                if cell_block.type == "line"
            ]
        )
        assert (line_tags == tag).sum() > 0

    def test_multiple_contacts(self, tmp_path):
        sim = _make_sim(
            tmp_path,
            [
                {"name": "anode", "layer_a": "p_rib", "layer_b": "sio2"},
                {"name": "cathode", "layer_a": "n_rib", "layer_b": "sio2"},
            ],
        )
        contact_lines = sim.mesh_groups["contact_lines"]
        assert {"anode", "cathode"} <= set(contact_lines)

    def test_nontouching_pair_raises(self, tmp_path):
        with pytest.raises(ValueError, match="anode"):
            _make_sim(
                tmp_path,
                [{"name": "anode", "layer_a": "p_rib", "layer_b": "substrate"}],
            )

    def test_unknown_layer_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no_such_layer"):
            _make_sim(
                tmp_path,
                [{"name": "anode", "layer_a": "p_rib", "layer_b": "no_such_layer"}],
            )
