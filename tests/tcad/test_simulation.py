"""What a tcad.Simulation does when its concerns change, and what it refuses.

Mesh inputs (geometry, domain, contacts, interfaces, the solver's mesh
fields) make the mesh stale: a solve raises until mesh() runs again.
Device inputs (materials, doping, the other solver fields) rebuild the
DEVSIM device on the existing mesh at the next solve. Contacts and
Interfaces state the role of each Region, and the device checks it.

DEVSIM is faked (see conftest).
"""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim import tcad
from gsim.tcad import Simulation, StepDoping

from .conftest import build_contacted_diode

DOPED = {"n_pad": "donor", "n_rib": "donor", "p_rib": "acceptor", "p_pad": "acceptor"}


def _configured_sim() -> Simulation:
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
    sim.interfaces.add(name="n_link", region_a="n_pad", region_b="n_rib")
    sim.interfaces.add(name="junction", region_a="n_rib", region_b="p_rib")
    sim.interfaces.add(name="p_link", region_a="p_rib", region_b="p_pad")
    for region, dopant_type in DOPED.items():
        sim.doping.add(
            StepDoping(region=region, dopant_type=dopant_type, concentration_cm3=1e18)
        )
    sim.solver(mesh_preset="coarse", refined_mesh_size_um=0.05, max_mesh_size_um=40.0)
    return sim


@pytest.fixture(scope="module")
def meshed(tmp_path_factory):
    sim = _configured_sim()
    sim.mesh(output_dir=tmp_path_factory.mktemp("tcad-simulation"))
    return sim


@pytest.fixture
def sim(meshed, fake_devsim):  # noqa: ARG001 - the device talks to the fake
    """The module's meshed sim over a fake DEVSIM, its concerns put back."""
    geometry = dict(meshed.geometry)
    domain = dict(meshed.domain)
    solver = dict(meshed.solver)
    materials = meshed.materials.model_copy(deep=True)
    doping = list(meshed.doping)
    contacts = list(meshed.contacts)
    interfaces = list(meshed.interfaces)
    meshed._reset_device()
    yield meshed
    meshed.geometry = geometry
    meshed.domain(**domain)
    meshed.solver(**solver)
    meshed.materials = materials
    meshed.doping = doping
    meshed.contacts = contacts
    meshed.interfaces = interfaces
    meshed._reset_device()


def _created_devices(devsim) -> list[str]:
    return [call["device"] for call in devsim.called("create_device")]


class TestMeshPreconditions:
    def test_the_first_mesh_needs_an_output_directory(self):
        with pytest.raises(ValueError, match="output_dir"):
            _configured_sim().mesh()

    def test_a_mesh_needs_a_component(self, tmp_path):
        sim = _configured_sim()
        sim.geometry.component = None
        with pytest.raises(ValueError, match="geometry"):
            sim.mesh(output_dir=tmp_path)

    def test_a_mesh_needs_a_plane(self, tmp_path):
        sim = _configured_sim()
        sim.domain.plane = None
        with pytest.raises(ValueError, match="plane"):
            sim.mesh(output_dir=tmp_path)

    def test_a_contact_and_an_interface_cannot_share_a_name(self, tmp_path):
        sim = _configured_sim()
        sim.interfaces.add(name="anode", region_a="n_pad", region_b="p_pad")
        with pytest.raises(ValueError, match="anode"):
            sim.mesh(output_dir=tmp_path)

    @pytest.mark.usefixtures("fake_devsim")
    def test_a_solve_needs_a_mesh(self):
        with pytest.raises(ValueError, match=r"mesh\(\)"):
            _configured_sim().solve(0.0)

    def test_the_mesh_reports_its_paths_and_groups(self, meshed):
        mesh = meshed._current_mesh()
        assert mesh.mesh_path.exists()
        assert mesh.devsim_mesh_path.parent == mesh.mesh_path.parent
        assert {"cathode", "anode"} <= set(mesh.groups["contact_lines"])
        assert {"n_link", "junction", "p_link"} <= set(mesh.groups["interface_lines"])


class TestStaleMesh:
    @pytest.mark.parametrize(
        "change",
        [
            lambda sim: setattr(sim.domain, "window", (-21.0, -19.0)),
            lambda sim: sim.domain(margin_x=3.0),
            lambda sim: sim.contacts.remove("anode"),
            lambda sim: sim.interfaces.remove("junction"),
            lambda sim: setattr(sim.solver, "refined_mesh_size_um", 0.04),
            lambda sim: setattr(sim.solver, "refinement_boxes", [(0, 1, 0, 1, 0.1)]),
            lambda sim: sim.geometry(stack="other.yaml"),
        ],
        ids=["window", "airbox", "contact", "interface", "size", "box", "stack"],
    )
    def test_a_changed_mesh_input_refuses_the_solve(self, sim, change):
        change(sim)
        with pytest.raises(ValueError, match=r"changed since mesh\(\)"):
            sim.solve(0.0, contact="cathode")
        with pytest.raises(ValueError, match=r"changed since mesh\(\)"):
            sim.sweep([0.0], contact="cathode")

    def test_putting_the_input_back_clears_the_refusal(self, sim):
        window = sim.domain.window
        sim.domain.window = (-21.0, -19.0)
        sim.domain.window = window
        sim.solve(0.0, contact="cathode")

    def test_a_new_component_makes_the_mesh_stale(self, sim):
        comp, _stack = build_contacted_diode()
        sim.geometry(comp)
        with pytest.raises(ValueError, match=r"changed since mesh\(\)"):
            sim.solve(0.0, contact="cathode")

    @pytest.mark.parametrize(
        "change",
        [
            lambda sim: sim.solver(relative_error=1e-8),
            lambda sim: sim.materials(temperature_k=350.0),
            lambda sim: (
                sim.doping.clear()
                or sim.doping.add(
                    StepDoping(
                        region="n_pad", dopant_type="donor", concentration_cm3=1e18
                    )
                )
            ),
        ],
        ids=["solver", "materials", "doping"],
    )
    def test_a_device_input_is_not_a_mesh_input(self, sim, change):
        change(sim)
        sim._current_mesh()


class TestDeviceRebuild:
    def test_an_unchanged_configuration_keeps_its_device(self, sim, fake_devsim):
        devsim, sp = fake_devsim
        sim.solve(0.0, contact="cathode")
        sim.solve(-0.2, contact="cathode")
        assert len(_created_devices(devsim)) == 1
        assert len(sp.called("CreateSiliconDriftDiffusion")) == len(DOPED)

    def test_a_materials_change_rebuilds_the_device(self, sim, fake_devsim):
        devsim, sp = fake_devsim
        sim.solve(0.0, contact="cathode")
        sim.materials.temperature_k = 350.0
        sim.solve(0.0, contact="cathode")

        first, second = _created_devices(devsim)
        assert first != second
        assert {"device": first} in devsim.called("delete_device")
        temperatures = [args[2] for args in sp.called("SetSiliconParameters")]
        assert temperatures == [300.0] * len(DOPED) + [350.0] * len(DOPED)

    def test_a_doping_change_rebuilds_the_device(self, sim, fake_devsim):
        devsim, _sp = fake_devsim
        sim.solve(0.0, contact="cathode")
        sim.doping = [
            StepDoping(region=region, dopant_type=dopant_type, concentration_cm3=5e17)
            for region, dopant_type in DOPED.items()
        ]
        sim.solve(0.0, contact="cathode")
        assert len(_created_devices(devsim)) == 2
        assert devsim.node_values[("p_rib", "Acceptors")][0] == 5e17

    def test_a_doping_profile_changed_in_place_rebuilds_the_device(
        self, sim, fake_devsim
    ):
        devsim, _sp = fake_devsim
        sim.solve(0.0, contact="cathode")
        profile = next(iter(sim.doping))
        sim.doping.remove(profile)
        sim.doping.add(profile.model_copy(update={"concentration_cm3": 2e18}))
        sim.solve(0.0, contact="cathode")
        assert len(_created_devices(devsim)) == 2

    def test_a_solver_setting_change_rebuilds_the_device(self, sim, fake_devsim):
        devsim, _sp = fake_devsim
        sim.solve(0.0, contact="cathode")
        sim.solver.relative_error = 1e-9
        sim.solve(0.0, contact="cathode")
        assert len(_created_devices(devsim)) == 2
        assert devsim.called("solve")[-3]["relative_error"] == 1e-9

    def test_a_new_mesh_drops_the_device(self, fake_devsim, tmp_path):
        devsim, _sp = fake_devsim
        sim = _configured_sim()
        sim.mesh(output_dir=tmp_path)
        sim.solve(0.0, contact="cathode")
        sim.mesh()
        sim.solve(0.0, contact="cathode")
        first, second = _created_devices(devsim)
        assert first != second


class TestContactRoles:
    """A Contact states which side is the semiconductor and which the metal."""

    def test_the_contact_binds_to_its_semiconductor(self, sim, fake_devsim):
        devsim, _sp = fake_devsim
        sim._setup_device()
        contacts = {c["name"]: c["region"] for c in devsim.called("add_gmsh_contact")}
        assert contacts == {"cathode": "n_pad", "anode": "p_pad"}

    def test_an_undoped_semiconductor_is_refused(self, sim):
        sim.doping = [p for p in sim.doping if p.region != "n_pad"]
        with pytest.raises(ValueError, match=r"semiconductor 'n_pad'.*no doping"):
            sim._setup_device()

    def test_an_insulating_semiconductor_is_refused(self, sim):
        sim.doping = [p for p in sim.doping if p.region != "n_pad"]
        sim.materials.insulators = {"n_pad": 11.9}
        with pytest.raises(ValueError, match=r"semiconductor 'n_pad'.*insulating"):
            sim._setup_device()

    def test_a_doped_electrode_is_refused(self, sim):
        original = list(sim.contacts)
        sim.contacts.remove("cathode")
        sim.contacts.add(name="cathode", semiconductor="n_pad", electrode="sio2")
        sim.mesh()
        try:
            sim.doping.add(
                StepDoping(region="sio2", dopant_type="donor", concentration_cm3=1e15)
            )
            with pytest.raises(ValueError, match=r"electrode 'sio2'.*doping"):
                sim._setup_device()
        finally:
            sim.contacts = original
            sim.mesh()

    def test_a_contact_declared_the_wrong_way_round_is_refused(self, sim):
        original = list(sim.contacts)
        sim.contacts.remove("cathode")
        sim.contacts.add(
            name="cathode", semiconductor="cathode_metal", electrode="n_pad"
        )
        sim.mesh()
        try:
            with pytest.raises(ValueError, match="semiconductor 'cathode_metal'"):
                sim._setup_device()
        finally:
            sim.contacts = original
            sim.mesh()

    def test_an_interface_joins_two_doped_regions(self, sim):
        sim.doping = [p for p in sim.doping if p.region != "n_rib"]
        with pytest.raises(ValueError, match=r"n_link.*two doped regions"):
            sim._setup_device()

    def test_a_region_is_doped_or_insulating(self, sim):
        sim.materials.insulators = {"p_rib": 11.9}
        with pytest.raises(ValueError, match=r"p_rib.*one or the other"):
            sim._setup_device()


class TestSweptContact:
    def test_the_first_contact_is_swept_by_default(self, sim):
        assert sim.sweep([0.0]).contact == "cathode"

    def test_an_unknown_contact_is_refused(self, sim):
        with pytest.raises(ValueError, match="gate"):
            sim.solve(0.0, contact="gate")


class TestReviewRegressions:
    def test_a_failed_setup_is_run_again_by_the_next_solve(self, sim, fake_devsim):
        devsim, _sp = fake_devsim
        state = {"fail": True}

        def donors(_x, _y):
            if state["fail"]:
                raise RuntimeError("profile failed")
            return 1e18

        sim.doping = [
            *(p for p in sim.doping if p.region != "n_rib"),
            tcad.CallableDoping(region="n_rib", dopant_type="donor", function=donors),
        ]
        with pytest.raises(RuntimeError, match="profile failed"):
            sim.solve(0.0, contact="cathode")
        state["fail"] = False
        sim.solve(0.0, contact="cathode")

        first, second = _created_devices(devsim)
        assert first != second
        assert {"device": first} in devsim.called("delete_device")
        assert {c["name"] for c in devsim.called("circuit_element")} == {
            "V_cathode",
            "V_anode",
        }

    def test_a_device_another_simulation_deleted_is_built_again(self, sim, fake_devsim):
        from gsim.tcad import device as device_module

        devsim, _sp = fake_devsim
        sim.solve(0.0, contact="cathode")
        device_module._LIVE_DEVICES.clear()  # what another sim's claim does
        sim.solve(0.0, contact="cathode")
        assert len(_created_devices(devsim)) == 2

    def test_a_removed_contact_reports_the_stale_mesh(self, sim):
        sim.contacts.remove("cathode")
        with pytest.raises(ValueError, match=r"changed since mesh\(\)"):
            sim.solve(0.0, contact="cathode")

    def test_the_snapshot_holds_the_component_it_compares(self, sim):
        before = sim._mesh_inputs()
        assert sim._mesh_inputs() == before
        sim.geometry.component = gf.Component()
        assert sim._mesh_inputs() != before

    @pytest.mark.usefixtures("fake_devsim")
    def test_a_component_edited_in_place_makes_the_mesh_stale(self, tmp_path):
        sim = _configured_sim()
        sim.mesh(output_dir=tmp_path)
        component = sim.geometry.component
        assert component is not None
        component.add_polygon(
            [(-5.0, -30.0), (5.0, -30.0), (5.0, -29.0), (-5.0, -29.0)], layer=(1, 0)
        )
        with pytest.raises(ValueError, match=r"changed since mesh\(\)"):
            sim.solve(0.0, contact="cathode")


class TestReviewOfPR370:
    def test_the_other_contacts_go_back_to_zero(self, sim, fake_devsim):
        devsim, _sp = fake_devsim
        sim.solve(2.0, contact="cathode")
        assert devsim.circuit["V_cathode"] == pytest.approx(2.0)
        sim.solve(0.0, contact="anode")
        assert devsim.circuit["V_cathode"] == pytest.approx(0.0)
        assert devsim.circuit["V_anode"] == pytest.approx(0.0)

    def test_a_sweep_starts_from_the_other_contacts_at_zero(self, sim, fake_devsim):
        devsim, _sp = fake_devsim
        sim.solve(1.0, contact="anode")
        sim.sweep([0.0, 0.5], contact="cathode")
        assert devsim.circuit["V_anode"] == pytest.approx(0.0)
        assert devsim.circuit["V_cathode"] == pytest.approx(0.5)

    @pytest.mark.parametrize("temperature_k", [250.0, 300.0, 350.0])
    def test_the_intrinsic_density_follows_the_temperature(
        self, sim, fake_devsim, temperature_k
    ):
        from gsim.tcad.silicon import intrinsic_density_cm3

        devsim, _sp = fake_devsim
        sim.materials.temperature_k = temperature_k
        sim._setup_device()
        expected = intrinsic_density_cm3(temperature_k)
        for name in ("n_i", "n1", "p1"):
            values = {
                c["region"]: c["value"]
                for c in devsim.called("set_parameter")
                if c["name"] == name
            }
            assert values == dict.fromkeys(DOPED, pytest.approx(expected))

    @pytest.mark.parametrize("extended", [True, False])
    def test_extended_precision_is_set_either_way(self, sim, fake_devsim, extended):
        devsim, _sp = fake_devsim
        sim.solver.extended_precision = extended
        sim._setup_device()
        for name in ("extended_solver", "extended_model", "extended_equation"):
            assert devsim.parameters[name] is extended

    def test_a_missing_interface_between_doped_regions_is_refused(self, sim):
        original = list(sim.interfaces)
        sim.interfaces.remove("n_link")
        sim.mesh()
        try:
            with pytest.raises(ValueError, match=r"'n_pad' and 'n_rib' touch"):
                sim._setup_device()
        finally:
            sim.interfaces = original
            sim.mesh()

    @pytest.mark.parametrize("bias", [float("nan"), float("inf"), -float("inf")])
    def test_a_bias_must_be_finite(self, sim, bias):
        with pytest.raises(ValueError, match="finite"):
            sim.solve(bias, contact="cathode")
        with pytest.raises(ValueError, match="finite"):
            sim.sweep([0.0, bias], contact="cathode")
