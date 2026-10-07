"""Hermetic tests for the ChargeTransportSim device setup and solve wiring.

DEVSIM is faked (see conftest); assertions target the external seams:
configuration in, DEVSIM setup calls and generated artifacts out.
"""

from __future__ import annotations

import sys

import meshio
import numpy as np
import pytest

from gsim.tcad import (
    CallableDoping,
    ChargeTransportSim,
    StepDoping,
    TableDoping,
)
from gsim.tcad.mesh import UM_TO_CM

from .conftest import FakeDevsim, build_pn_device


def _make_meshed_sim(tmp_path):
    comp, stack = build_pn_device()
    sim = ChargeTransportSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    sim.set_geometry(comp)
    sim.set_cross_section("x=0")
    sim.add_contact(name="anode", layer_a="p_rib", layer_b="sio2")
    sim.add_contact(name="cathode", layer_a="n_rib", layer_b="sio2")
    sim.add_interface(name="junction", layer_a="p_rib", layer_b="n_rib")
    sim.add_doping(
        StepDoping(region="p_rib", dopant_type="acceptor", concentration_cm3=1e18)
    )
    sim.add_doping(
        StepDoping(region="n_rib", dopant_type="donor", concentration_cm3=1e18)
    )
    sim.mesh(preset="coarse", refined_mesh_size=0.05, max_mesh_size=40.0, verbose=False)
    return sim


@pytest.fixture(scope="module")
def meshed_sim(tmp_path_factory):
    return _make_meshed_sim(tmp_path_factory.mktemp("tcad"))


@pytest.fixture(autouse=True)
def _fresh_device(request, monkeypatch):
    """Solver-side state must not leak between tests sharing the mesh."""
    if "meshed_sim" not in request.fixturenames:
        return
    # The release goes to a stand-in of its own: the real DEVSIM is never
    # imported for it, and the recorder the test asks for does not see the
    # device an earlier test left on the module-scoped sim.
    with monkeypatch.context() as release:
        release.setitem(sys.modules, "devsim", FakeDevsim())
        request.getfixturevalue("meshed_sim").reset_device()


class TestMeshing:
    def test_interfaces_are_declared_apart_from_contacts(self, meshed_sim):
        """An interface carries no terminal voltage, and is never a contact."""
        assert {spec.name for spec in meshed_sim.contact_specs} == {"anode", "cathode"}
        assert {spec.name for spec in meshed_sim.interface_specs} == {"junction"}
        groups = meshed_sim.mesh_groups
        assert set(groups["contact_lines"]) == {"anode", "cathode"}
        assert set(groups["interface_lines"]) == {"junction"}
        # The junction is tagged on the mesh under its own name all the same.
        assert "junction" in meshio.read(str(meshed_sim.mesh_path)).field_data

    def test_shared_mesh_and_scaled_copy(self, meshed_sim):
        # The shared native-2D mesh is the geometry source ...
        assert meshed_sim.mesh_path is not None
        assert meshed_sim.mesh_path.exists()
        # ... and the DEVSIM copy is the same mesh with cm coordinates.
        assert meshed_sim.devsim_mesh_path is not None
        assert meshed_sim.devsim_mesh_path.exists()
        um_mesh = meshio.read(str(meshed_sim.mesh_path))
        cm_mesh = meshio.read(str(meshed_sim.devsim_mesh_path))
        np.testing.assert_allclose(
            cm_mesh.points, um_mesh.points * UM_TO_CM, atol=1e-12
        )
        assert set(um_mesh.field_data) == set(cm_mesh.field_data)
        assert {"p_rib", "n_rib", "anode", "cathode"} <= set(cm_mesh.field_data)

    def test_mesh_requires_contacts(self, tmp_path):
        comp, stack = build_pn_device()
        sim = ChargeTransportSim()
        sim.set_output_dir(str(tmp_path))
        sim.set_stack(stack)
        sim.set_geometry(comp)
        sim.set_cross_section("x=0")
        with pytest.raises(ValueError, match="contact"):
            sim.mesh(preset="coarse", verbose=False)


class TestDeviceSetup:
    def test_devsim_setup_sequence(self, meshed_sim, fake_devsim):
        devsim, sp = fake_devsim
        meshed_sim.setup_device("pn")

        [create] = devsim.called("create_gmsh_mesh")
        assert create["file"] == str(meshed_sim.devsim_mesh_path)

        regions = {c["region"] for c in devsim.called("add_gmsh_region")}
        assert regions == {"p_rib", "n_rib"}
        for call in devsim.called("add_gmsh_region"):
            assert call["gmsh_name"] == call["region"]
            assert call["material"] == "Silicon"

        contacts = {c["name"]: c["region"] for c in devsim.called("add_gmsh_contact")}
        assert contacts == {"anode": "p_rib", "cathode": "n_rib"}

        assert devsim.called("finalize_mesh")
        [created] = devsim.called("create_device")
        assert created["device"] == "pn"

        # Potential-only physics on every region, contact BCs at 0 V.
        assert {args[1] for args in sp.called("CreateSiliconPotentialOnly")} == {
            "p_rib",
            "n_rib",
        }
        assert {args[2] for args in sp.called("CreateSiliconPotentialOnlyContact")} == {
            "anode",
            "cathode",
        }
        # Each electrical contact is driven through a circuit source at 0 V.
        assert devsim.circuit["V_anode"] == 0.0
        assert devsim.circuit["V_cathode"] == 0.0
        sources = {c["name"]: c for c in devsim.called("circuit_element")}
        assert set(sources) == {"V_anode", "V_cathode"}
        assert sources["V_cathode"]["n1"] == "cathode_bias"

        # The P/N junction is a region-region interface, not a contact,
        # with potential continuity from the potential-only stage.
        [iface] = devsim.called("add_gmsh_interface")
        assert iface["name"] == "junction"
        assert {iface["region0"], iface["region1"]} == {"p_rib", "n_rib"}
        potential_continuity = [
            c
            for c in devsim.called("interface_equation")
            if c["name"] == "PotentialEquation"
        ]
        assert len(potential_continuity) == 1
        assert potential_continuity[0]["type"] == "continuous"

    def test_the_mobility_follows_the_doping(self, meshed_sim, fake_devsim):
        from gsim.common.carriers import MobilityModel

        devsim, _sp = fake_devsim
        meshed_sim.setup_device("pn")

        model = MobilityModel.masetti_silicon()
        n_nodes = len(devsim.node_coords["x"])
        np.testing.assert_allclose(
            devsim.node_values[("p_rib", "HoleMobility")],
            [float(model.holes_cm2(1e18))] * n_nodes,
        )
        np.testing.assert_allclose(
            devsim.node_values[("n_rib", "ElectronMobility")],
            [float(model.electrons_cm2(1e18))] * n_nodes,
        )
        averaged = {
            (c["region"], c["node_model"], c["edge_model"])
            for c in devsim.called("edge_average_model")
        }
        assert ("p_rib", "HoleMobility", "HoleMobilityEdge") in averaged
        assert ("n_rib", "ElectronMobility", "ElectronMobilityEdge") in averaged

    def test_the_mobility_reads_both_dopants_where_they_compensate(
        self, meshed_sim, fake_devsim
    ):
        """A graded Junction carries donors into the p Region. They cancel
        acceptors in the net doping, and add to them as scatterers."""
        from gsim.common.carriers import MobilityModel

        devsim, _sp = fake_devsim
        # The fake's nodes sit at x = 0, 1, 2 um.
        donor_tail = TableDoping(
            region="p_rib",
            dopant_type="donor",
            x_um=[0.0, 1.0, 2.0],
            values_cm3=[1e18, 4e17, 0.0],
        )
        meshed_sim.doping.append(donor_tail)
        try:
            meshed_sim.setup_device("pn")
        finally:
            meshed_sim.doping.remove(donor_tail)

        model = MobilityModel.masetti_silicon()
        total = np.array([2e18, 1.4e18, 1e18])
        np.testing.assert_allclose(
            devsim.node_values[("p_rib", "HoleMobility")], model.holes_cm2(total)
        )
        np.testing.assert_allclose(
            devsim.node_values[("p_rib", "ElectronMobility")],
            model.electrons_cm2(total),
        )
        # Fully compensated at x = 0, and the mobility is still the doped one.
        np.testing.assert_allclose(
            np.array(devsim.node_values[("p_rib", "Donors")])
            - np.array(devsim.node_values[("p_rib", "Acceptors")]),
            [0.0, -6e17, -1e18],
        )

    def test_doping_node_models(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        meshed_sim.setup_device("pn")

        # NetDoping = Donors - Acceptors on every region.
        net_models = {
            c["region"]: c["equation"]
            for c in devsim.called("node_model")
            if c["name"] == "NetDoping"
        }
        assert net_models == {
            "p_rib": "Donors - Acceptors",
            "n_rib": "Donors - Acceptors",
        }

        # Node values match the analytic profiles evaluated at the node
        # coordinates (fake coords are in cm; profiles take um).
        n_nodes = len(devsim.node_coords["x"])
        np.testing.assert_allclose(
            devsim.node_values[("p_rib", "Acceptors")], [1e18] * n_nodes
        )
        np.testing.assert_allclose(
            devsim.node_values[("p_rib", "Donors")], [0.0] * n_nodes
        )
        np.testing.assert_allclose(
            devsim.node_values[("n_rib", "Donors")], [1e18] * n_nodes
        )

    def test_a_table_and_a_function_reach_the_node_values(
        self, meshed_sim, fake_devsim
    ):
        """Every union member goes through the same validated seam.

        ``ChargeTransportSim.doping`` is a discriminated union under
        ``validate_assignment``, so a shape that is not a member never
        reaches DEVSIM at all.
        """
        devsim, _sp = fake_devsim
        sim = meshed_sim.model_copy()
        sim.doping = []
        sim.add_doping(
            TableDoping(
                region="p_rib",
                dopant_type="acceptor",
                y_um=[-10.0, 10.0],
                values_cm3=[2e17, 2e17],
            )
        )
        sim.add_doping(
            CallableDoping(
                region="n_rib",
                dopant_type="donor",
                function=lambda x, y: 6e17,
            )
        )
        assert [type(p).__name__ for p in sim.doping] == [
            "TableDoping",
            "CallableDoping",
        ]

        sim.setup_device("pn")

        n_nodes = len(devsim.node_coords["x"])
        np.testing.assert_allclose(
            devsim.node_values[("p_rib", "Acceptors")], [2e17] * n_nodes
        )
        np.testing.assert_allclose(
            devsim.node_values[("n_rib", "Donors")], [6e17] * n_nodes
        )

    @pytest.mark.usefixtures("fake_devsim")
    def test_unknown_doping_region_raises(self, meshed_sim):
        sim = meshed_sim.model_copy()
        sim.doping = [
            StepDoping(
                region="no_such_region",
                dopant_type="donor",
                concentration_cm3=1e18,
            )
        ]
        with pytest.raises(ValueError, match="no_such_region"):
            sim.setup_device()

    def test_setup_before_mesh_raises(self):
        sim = ChargeTransportSim()
        sim.add_doping(
            StepDoping(region="p_rib", dopant_type="acceptor", concentration_cm3=1e18)
        )
        with pytest.raises(ValueError, match="mesh"):
            sim.setup_device()

    @pytest.mark.usefixtures("fake_devsim")
    def test_no_doping_raises(self, meshed_sim):
        sim = meshed_sim.model_copy()
        sim.doping = []
        with pytest.raises(ValueError, match="doping"):
            sim.setup_device()


class TestSolveWiring:
    def test_solve_returns_bias_point_with_small_signal_capacitance(
        self, meshed_sim, fake_devsim
    ):
        devsim, _sp = fake_devsim
        point = meshed_sim.solve(-1.0, contact="cathode")

        assert point.bias_v == -1.0
        # Fake device charge is linear in bias: C = charge_per_volt exactly.
        assert point.capacitance_f_per_cm == pytest.approx(
            devsim.charge_per_volt, rel=1e-6
        )
        assert point.capacitance_f_per_m == pytest.approx(
            devsim.charge_per_volt * 1e2, rel=1e-6
        )
        assert set(point.currents_a_per_cm) == {"anode", "cathode"}
        # The swept contact's circuit source carries the requested bias.
        assert devsim.circuit["V_cathode"] == pytest.approx(-1.0)

        # Carrier maps concatenate both regions; coords are back in um.
        n_nodes = len(devsim.node_coords["x"])
        assert point.carriers.x_um.size == 2 * n_nodes
        assert set(point.carriers.region) == {"p_rib", "n_rib"}
        np.testing.assert_allclose(
            point.carriers.x_um[:n_nodes],
            np.asarray(devsim.node_coords["x"]) / UM_TO_CM,
        )

    def test_sweep_orders_points(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        biases = [0.0, -0.5, -1.0]
        result = meshed_sim.sweep(biases, contact="cathode")
        assert result.contact == "cathode"
        np.testing.assert_allclose(result.voltages, biases)
        np.testing.assert_allclose(
            result.capacitance_f_per_cm, devsim.charge_per_volt, rtol=1e-6
        )
        assert result.capacitance_f_per_m.shape == (3,)

    @pytest.mark.usefixtures("fake_devsim")
    def test_devsim_output_silent_by_default(self, meshed_sim, capsys):
        meshed_sim.sweep([0.0, -0.5], contact="cathode")
        assert "Iteration" not in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_devsim")
    def test_verbose_streams_devsim_output(self, meshed_sim, capsys):
        meshed_sim.solve(0.0, contact="cathode", verbose=True)
        assert "Iteration: dc" in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_devsim")
    def test_unknown_sweep_contact_raises(self, meshed_sim):
        with pytest.raises(ValueError, match="gate"):
            meshed_sim.solve(0.0, contact="gate")

    def test_drift_diffusion_initialized_once(self, meshed_sim, fake_devsim):
        _devsim, sp = fake_devsim
        meshed_sim.solve(0.0, contact="cathode")
        meshed_sim.solve(-0.2, contact="cathode")
        # DD assembly happens once per region despite two solves.
        assert len(sp.called("CreateSiliconDriftDiffusion")) == 2
        assert {args[1] for args in sp.called("CreateSiliconDriftDiffusion")} == {
            "p_rib",
            "n_rib",
        }
        # The currents run on the doping-dependent mobilities, not on
        # simple_physics' two constants.
        assert {args[2:] for args in sp.called("CreateSiliconDriftDiffusion")} == {
            ("ElectronMobilityEdge", "HoleMobilityEdge")
        }

    def test_carrier_continuity_across_interface(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        meshed_sim.solve(0.0, contact="cathode")
        equations = {c["name"] for c in devsim.called("interface_equation")}
        assert {
            "PotentialEquation",
            "ElectronContinuityEquation",
            "HoleContinuityEquation",
        } <= equations


class TestDevsimNamespace:
    """DEVSIM's mesh/device namespace is process-wide, so names must differ."""

    def test_each_setup_claims_its_own_mesh_and_device(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        first = meshed_sim.setup_device()
        first_mesh = devsim.called("create_gmsh_mesh")[0]["mesh"]

        meshed_sim.reset_device()
        second = meshed_sim.setup_device()
        second_mesh = devsim.called("create_gmsh_mesh")[1]["mesh"]

        assert first != second
        assert first_mesh != second_mesh

    def test_reset_releases_the_device_and_its_mesh(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        device = meshed_sim.setup_device()
        mesh_name = devsim.called("create_gmsh_mesh")[0]["mesh"]

        meshed_sim.reset_device()

        assert devsim.called("delete_device") == [{"device": device}]
        assert devsim.called("delete_mesh") == [{"mesh": mesh_name}]

    def test_the_mesh_name_is_recognisable(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        meshed_sim.setup_device()
        name = devsim.called("create_gmsh_mesh")[0]["mesh"]
        assert name.startswith("gsim_tcad_mesh")
