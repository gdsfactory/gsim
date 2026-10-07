"""Hermetic tests for insulating Regions in the charge-transport solve.

DEVSIM is faked (see conftest). The device is the modulator demo Phase
shifter, whose Contacts sit between a doped pad and its electrode, so the
oxide around the doped slab is free to join the solve. The fake reports
every mesh node for every Region, which is all the node-by-node Interface
binding needs to find its pairs.
"""

from __future__ import annotations

import warnings

import meshio
import numpy as np
import pytest

from gsim.modulator.demo import demo_phase_shifter
from gsim.tcad import ChargeTransportSim, Insulator, StepDoping
from gsim.tcad.mesh import line_group_points
from gsim.tcad.sim import VACUUM_PERMITTIVITY_F_PER_CM

DOPED = {"n_pad": "donor", "n_rib": "donor", "p_rib": "acceptor", "p_pad": "acceptor"}
OXIDE_PERMITTIVITY = 3.9


def _make_meshed_sim(tmp_path, *, contact_against="electrode"):
    demo = demo_phase_shifter()
    sim = ChargeTransportSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_stack(demo.stack)
    sim.set_airbox(margin_x=2.0, margin_y=2.0, z_above=1.5, z_below=1.0)
    sim.set_geometry(demo.component)
    sim.set_cross_section("x=0", window=(-21.1, -18.9))
    sides = {
        "electrode": {"cathode": "cathode_metal", "anode": "anode_metal"},
        "oxide": {"cathode": "sio2", "anode": "sio2"},
    }[contact_against]
    sim.add_contact(name="cathode", layer_a="n_pad", layer_b=sides["cathode"])
    sim.add_contact(name="anode", layer_a="p_pad", layer_b=sides["anode"])
    sim.add_interface(name="n_link", layer_a="n_pad", layer_b="n_rib")
    sim.add_interface(name="junction", layer_a="n_rib", layer_b="p_rib")
    sim.add_interface(name="p_link", layer_a="p_rib", layer_b="p_pad")
    for region, dopant_type in DOPED.items():
        sim.add_doping(
            StepDoping(region=region, dopant_type=dopant_type, concentration_cm3=1e18)
        )
    sim.mesh(preset="coarse", refined_mesh_size=0.05, max_mesh_size=40.0, verbose=False)
    return sim


@pytest.fixture(scope="module")
def meshed_sim(tmp_path_factory):
    return _make_meshed_sim(tmp_path_factory.mktemp("tcad-insulators"))


@pytest.fixture
def oxide_sim(meshed_sim, fake_devsim):
    """The meshed sim with the oxide declared, over a fake DEVSIM."""
    devsim, _sp = fake_devsim
    points = meshio.read(str(meshed_sim.devsim_mesh_path)).points
    devsim.node_coords = {"x": list(points[:, 0]), "y": list(points[:, 1])}
    meshed_sim.reset_device()
    meshed_sim.insulators = []
    meshed_sim.add_insulator(region="sio2", relative_permittivity=OXIDE_PERMITTIVITY)
    yield meshed_sim
    meshed_sim.insulators = []
    meshed_sim.reset_device()


def _node_coordinates(devsim, call, side):
    """Coordinates (cm) of one side's nodes of a node-built interface."""
    nodes = call[f"nodes{side}"]
    return np.column_stack(
        [np.asarray(devsim.node_coords[axis])[nodes] for axis in ("x", "y")]
    )


class TestDeclaration:
    def test_no_insulator_is_solved_by_default(self):
        assert ChargeTransportSim().insulators == []

    def test_an_insulator_names_its_region_and_permittivity(self):
        sim = ChargeTransportSim()
        sim.add_insulator(region="sio2", relative_permittivity=3.9)
        assert sim.insulators == [Insulator(region="sio2", relative_permittivity=3.9)]

    def test_a_permittivity_must_be_positive(self):
        with pytest.raises(ValueError, match="relative_permittivity"):
            ChargeTransportSim().add_insulator(region="sio2", relative_permittivity=0.0)


class TestPotentialOnlyRegion:
    def test_the_insulator_is_a_region_of_the_device(self, oxide_sim, fake_devsim):
        devsim, _sp = fake_devsim
        oxide_sim.setup_device()
        materials = {
            c["region"]: c["material"] for c in devsim.called("add_gmsh_region")
        }
        assert materials["sio2"] == "Oxide"
        assert {r for r, m in materials.items() if m == "Silicon"} == set(DOPED)

    def test_it_carries_the_potential_and_its_own_permittivity(
        self, oxide_sim, fake_devsim
    ):
        devsim, sp = fake_devsim
        oxide_sim.setup_device()
        assert [args[1] for args in sp.called("CreateOxidePotentialOnly")] == ["sio2"]
        [permittivity] = [
            c for c in devsim.called("set_parameter") if c["name"] == "Permittivity"
        ]
        assert permittivity["region"] == "sio2"
        assert permittivity["value"] == pytest.approx(
            OXIDE_PERMITTIVITY * VACUUM_PERMITTIVITY_F_PER_CM
        )

    def test_it_holds_no_doping_mobility_or_carriers(self, oxide_sim, fake_devsim):
        devsim, sp = fake_devsim
        oxide_sim.solve(0.5, contact="cathode")
        assert "sio2" not in {c["region"] for c in devsim.called("node_solution")}
        for model in ("CreateSiliconPotentialOnly", "CreateSiliconDriftDiffusion"):
            assert {args[1] for args in sp.called(model)} == set(DOPED)


class TestInterfaces:
    def test_every_doped_region_touching_the_oxide_is_tied_to_it(
        self, oxide_sim, fake_devsim
    ):
        devsim, _sp = fake_devsim
        oxide_sim.setup_device()
        calls = devsim.called("create_interface_from_nodes")
        assert {c["region0"] for c in calls} == set(DOPED)
        assert {c["region1"] for c in calls} == {"sio2"}
        for call in calls:
            # Each pair is one point of the mesh seen from its two Regions.
            np.testing.assert_allclose(
                _node_coordinates(devsim, call, 0), _node_coordinates(devsim, call, 1)
            )
            assert len(call["nodes0"]) > 2

    def test_the_potential_alone_is_continuous_across_them(
        self, oxide_sim, fake_devsim
    ):
        devsim, _sp = fake_devsim
        oxide_sim.solve(0.5, contact="cathode")
        oxide_interfaces = {
            c["name"] for c in devsim.called("create_interface_from_nodes")
        }
        continuous = {
            (c["interface"], c["name"]) for c in devsim.called("interface_equation")
        }
        for name in oxide_interfaces:
            assert (name, "PotentialEquation") in continuous
            assert (name, "ElectronContinuityEquation") not in continuous
            assert (name, "HoleContinuityEquation") not in continuous

    def test_a_node_already_carrying_a_contact_or_an_interface_is_left_out(
        self, oxide_sim, fake_devsim
    ):
        """DEVSIM assembles a node on two Interfaces wrongly; see the sim."""
        devsim, _sp = fake_devsim
        oxide_sim.setup_device()
        taken = np.vstack(
            [
                line_group_points(oxide_sim.devsim_mesh_path, name)
                for name in ("cathode", "anode", "n_link", "junction", "p_link")
            ]
        )
        for call in devsim.called("create_interface_from_nodes"):
            tied = _node_coordinates(devsim, call, 1)
            distance = np.linalg.norm(tied[:, None, :] - taken[None, :, :], axis=-1)
            assert distance.min() > 1e-9

    def test_a_mesh_node_the_regions_do_not_hold_is_reported(
        self, oxide_sim, fake_devsim
    ):
        # The pairs are found by coordinate. Should DEVSIM's nodes not sit
        # where the mesh file puts them, continuity would be lost with
        # nothing said.
        devsim, _sp = fake_devsim
        devsim.node_coords = {
            axis: [value + 3e-9 for value in values]
            for axis, values in devsim.node_coords.items()
        }
        with pytest.warns(UserWarning, match="match no node"):
            oxide_sim.setup_device()
        assert devsim.called("create_interface_from_nodes") == []

    def test_a_matched_mesh_raises_no_such_warning(self, oxide_sim):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            oxide_sim.setup_device()

    def test_no_insulator_means_no_node_built_interface(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        meshed_sim.reset_device()
        meshed_sim.setup_device()
        assert devsim.called("create_interface_from_nodes") == []
        assert {c["region"] for c in devsim.called("add_gmsh_region")} == set(DOPED)


class TestCarrierMap:
    def test_it_holds_the_doped_regions_only(self, oxide_sim):
        point = oxide_sim.solve(0.5, contact="cathode")
        assert set(point.carriers.region) == set(DOPED)


class TestRefusals:
    @pytest.mark.usefixtures("fake_devsim")
    def test_an_insulator_the_mesh_does_not_hold_is_named(self, meshed_sim):
        sim = meshed_sim.model_copy()
        sim.insulators = [Insulator(region="nitride", relative_permittivity=7.5)]
        with pytest.raises(ValueError, match="nitride"):
            sim.setup_device()

    @pytest.mark.usefixtures("fake_devsim")
    def test_a_doped_region_cannot_also_be_insulating(self, meshed_sim):
        sim = meshed_sim.model_copy()
        sim.insulators = [Insulator(region="n_rib", relative_permittivity=11.9)]
        with pytest.raises(ValueError, match="n_rib"):
            sim.setup_device()

    @pytest.mark.usefixtures("fake_devsim")
    def test_a_contact_against_the_insulator_is_refused(self, tmp_path):
        sim = _make_meshed_sim(tmp_path, contact_against="oxide")
        sim.add_insulator(region="sio2", relative_permittivity=OXIDE_PERMITTIVITY)
        with pytest.raises(ValueError, match=r"cathode.*sio2"):
            sim.setup_device()
