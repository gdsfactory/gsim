"""The concern models and collections that configure a tcad.Simulation.

Pure configuration: no mesh, no DEVSIM.
"""

from __future__ import annotations

import gdsfactory as gf
import pytest
from pydantic import ValidationError

from gsim import tcad
from gsim.tcad import (
    Contact,
    Contacts,
    Domain,
    Doping,
    Geometry,
    Insulator,
    Interfaces,
    Materials,
    Simulation,
    Solver,
    StepDoping,
)


def _p_doping(region: str = "p_rib") -> StepDoping:
    return StepDoping(region=region, dopant_type="acceptor", concentration_cm3=1e18)


class TestBatchUpdate:
    def test_a_call_applies_several_settings(self):
        solver = Solver()
        solver(relative_error=1e-8, max_iterations=50)
        assert solver.relative_error == 1e-8
        assert solver.max_iterations == 50

    def test_a_failed_call_leaves_the_model_unchanged(self):
        solver = Solver()
        with pytest.raises(ValidationError):
            solver(relative_error=1e-8, max_iterations=0)
        assert solver.relative_error == 1e-6
        assert solver.max_iterations == 100

    def test_a_single_attribute_can_be_assigned_and_is_validated(self):
        solver = Solver()
        solver.relative_error = 1e-9
        assert solver.relative_error == 1e-9
        with pytest.raises(ValidationError):
            solver.relative_error = -1.0

    def test_an_unknown_setting_is_refused(self):
        with pytest.raises(ValidationError, match="extra"):
            Solver()(relative_eror=1e-8)

    def test_the_geometry_keeps_its_component_through_a_stack_change(self):
        component = gf.Component()
        geometry = Geometry()
        geometry(component)
        geometry(stack="stack.yaml")
        assert geometry.component is component
        assert geometry.stack == "stack.yaml"

    def test_an_explicit_none_stack_means_the_active_pdk(self):
        geometry = Geometry()
        geometry(gf.Component(), stack="stack.yaml")
        geometry(stack=None)
        assert geometry.stack is None
        geometry(stack="stack.yaml")
        geometry(gf.Component())
        assert geometry.stack == "stack.yaml"


class TestDefaults:
    """The defaults are the ones the setter-style sim had."""

    def test_materials(self):
        materials = Materials()
        assert materials.temperature_k == 300.0
        assert materials.semiconductor == "Silicon"
        assert materials.mobility == tcad.MobilityModel.masetti_silicon()
        assert materials.insulators == {}

    def test_solver(self):
        solver = Solver()
        assert solver.mesh_preset is None
        assert solver.bias_step_v == 0.1
        assert solver.small_signal_freq_hz == 1.0
        assert solver.junction_freq_hz == 1e9
        assert solver.absolute_error == 1e10
        assert solver.relative_error == 1e-6
        assert solver.max_iterations == 100
        assert solver.extended_precision is True
        assert solver.claim_devsim_namespace is True


class TestValidation:
    def test_a_temperature_must_be_positive(self):
        with pytest.raises(ValidationError, match="temperature_k"):
            Materials(temperature_k=0.0)

    def test_an_insulator_permittivity_must_be_positive(self):
        with pytest.raises(ValidationError, match="insulators"):
            Materials(insulators={"sio2": -3.9})

    def test_insulators_are_read_from_a_mapping_or_models(self):
        insulator = Insulator(region="sio2", relative_permittivity=3.9)
        assert Materials(insulators={"sio2": 3.9}).insulators == {"sio2": 3.9}
        assert Materials(insulators=[insulator]).insulators == {"sio2": 3.9}
        assert Materials(insulators=insulator).insulators == {"sio2": 3.9}
        assert Materials(insulators={"sio2": insulator}).insulators == {"sio2": 3.9}

    @pytest.mark.parametrize("plane", ["x=0", "y=-12.5", " X = 3 "])
    def test_a_plane_is_cut_along_x_or_y(self, plane):
        assert Domain(plane=plane).plane == plane

    @pytest.mark.parametrize("plane", ["z=0", "x=", "0", "w=1"])
    def test_any_other_plane_is_refused(self, plane):
        with pytest.raises(ValidationError, match="plane"):
            Domain(plane=plane)

    def test_a_window_must_ascend(self):
        with pytest.raises(ValidationError, match="window"):
            Domain(plane="x=0", window=(1.0, -1.0))
        with pytest.raises(ValidationError, match="window_z"):
            Domain(plane="x=0", window_z=(0.5, 0.5))

    def test_a_margin_cannot_be_negative(self):
        with pytest.raises(ValidationError, match="margin_x"):
            Domain(margin_x=-1.0)

    def test_a_mesh_size_must_be_positive(self):
        with pytest.raises(ValidationError, match="refined_mesh_size_um"):
            Solver(refined_mesh_size_um=0.0)

    def test_a_stack_is_a_layer_stack_a_path_or_none(self):
        with pytest.raises(ValidationError, match="stack"):
            Geometry(stack=42)

    def test_a_contact_joins_two_regions(self):
        with pytest.raises(ValidationError, match="itself"):
            Contact(name="anode", semiconductor="p_rib", electrode="p_rib")

    def test_an_interface_joins_two_regions(self):
        with pytest.raises(ValidationError, match="itself"):
            Interfaces().add(name="junction", region_a="p_rib", region_b="p_rib")


class TestContactsAndInterfaces:
    def test_items_are_kept_in_order_and_found_by_name(self):
        contacts = Contacts()
        anode = contacts.add(name="anode", semiconductor="p_rib", electrode="metal_p")
        contacts.add(name="cathode", semiconductor="n_rib", electrode="metal_n")
        assert [c.name for c in contacts] == ["anode", "cathode"]
        assert len(contacts) == 2
        assert contacts["anode"] is anode
        assert "cathode" in contacts

    def test_an_item_can_be_removed(self):
        interfaces = Interfaces()
        interfaces.add(name="junction", region_a="p_rib", region_b="n_rib")
        removed = interfaces.remove("junction")
        assert removed.region_b == "n_rib"
        assert len(interfaces) == 0

    def test_a_name_is_declared_once(self):
        contacts = Contacts()
        contacts.add(name="anode", semiconductor="p_rib", electrode="metal_p")
        with pytest.raises(ValueError, match="anode"):
            contacts.add(name="anode", semiconductor="p_pad", electrode="metal_p")
        interfaces = Interfaces()
        interfaces.add(name="junction", region_a="p_rib", region_b="n_rib")
        with pytest.raises(ValueError, match="junction"):
            interfaces.add(name="junction", region_a="p_pad", region_b="n_pad")

    def test_a_declared_contact_is_frozen(self):
        contacts = Contacts()
        contact = contacts.add(name="anode", semiconductor="p_rib", electrode="m")
        with pytest.raises(ValidationError):
            contact.electrode = "other"  # ty: ignore[invalid-assignment]


class TestDoping:
    def test_profiles_are_kept_in_order(self):
        doping = Doping()
        first = doping.add(_p_doping("p_rib"))
        doping.add(_p_doping("p_pad"))
        doping.add(_p_doping("p_rib"))
        assert len(doping) == 3
        assert next(iter(doping)) is first
        assert doping.regions() == ["p_rib", "p_pad"]

    def test_a_profile_mapping_is_validated_into_its_shape(self):
        doping = Doping()
        profile = doping.add(
            {
                "kind": "step",
                "region": "n_rib",
                "dopant_type": "donor",
                "concentration_cm3": 1e18,
            }
        )
        assert isinstance(profile, StepDoping)

    def test_a_shape_that_is_not_a_profile_is_refused(self):
        with pytest.raises(ValidationError):
            Doping().add({"kind": "sphere", "region": "n_rib"})

    def test_remove_and_clear(self):
        doping = Doping([_p_doping("p_rib"), _p_doping("p_pad")])
        doping.remove(next(iter(doping)))
        assert doping.regions() == ["p_pad"]
        doping.clear()
        assert len(doping) == 0


class TestSimulationConstructor:
    def test_concerns_are_taken_as_mappings_and_lists(self):
        sim = Simulation(
            domain={"plane": "x=0", "window": (-25.0, -15.0)},
            materials={"temperature_k": 350.0, "insulators": {"sio2": 3.9}},
            doping=[_p_doping()],
            contacts=[{"name": "anode", "semiconductor": "p_rib", "electrode": "m"}],
            interfaces=[{"name": "junction", "region_a": "p_rib", "region_b": "n"}],
            solver={"mesh_preset": "coarse"},
        )
        assert sim.domain.window == (-25.0, -15.0)
        assert sim.materials.temperature_k == 350.0
        assert sim.doping.regions() == ["p_rib"]
        assert sim.contacts["anode"].electrode == "m"
        assert sim.interfaces["junction"].region_a == "p_rib"
        assert sim.solver.mesh_preset == "coarse"

    def test_concerns_are_taken_as_models(self):
        solver = Solver(relative_error=1e-8)
        sim = Simulation(solver=solver)
        assert sim.solver is solver

    def test_a_replaced_concern_is_validated(self):
        sim = Simulation()
        sim.materials = {"temperature_k": 77.0}
        assert isinstance(sim.materials, Materials)
        assert sim.materials.temperature_k == 77.0
        with pytest.raises(ValidationError):
            sim.solver = {"max_iterations": 0}
        sim.doping = [_p_doping()]
        assert isinstance(sim.doping, Doping)

    def test_an_unknown_concern_is_refused(self):
        with pytest.raises(TypeError):
            Simulation(source={})  # ty: ignore[unknown-argument]
