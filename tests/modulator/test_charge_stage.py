"""The charge Stage of a Study: configuration, run, and the missing extra."""

from __future__ import annotations

import sys

import pytest

from gsim.tcad.results import BiasSweepResult


class TestConfiguration:
    def test_defaults_are_readable(self, study):
        assert study.charge.biases == [0.0]
        assert study.charge.window is None
        assert study.charge.has_run is False

    def test_the_section_is_callable(self, study):
        assert study.charge(biases=[0.0, -1.0]) is study.charge
        assert study.charge.biases == [0.0, -1.0]

    def test_unknown_setting_is_rejected(self, study):
        with pytest.raises(ValueError, match="nope"):
            study.charge(nope=1)


class TestSimulationAssembly:
    def test_the_sim_carries_the_derived_contacts_and_interfaces(self, study):
        """Contacts are Contacts and Interfaces are Interfaces, all the way."""
        sim = study.charge.simulation()
        assert {spec.name for spec in sim.contact_specs} == {"anode", "cathode"}
        assert {spec.name for spec in sim.interface_specs} == {
            "junction",
            "n_pad_n_rib",
            "p_rib_p_pad",
        }

    def test_the_sim_carries_one_doping_profile_per_doped_region(self, study):
        sim = study.charge.simulation()
        by_region = {profile.region: profile for profile in sim.doping}
        assert set(by_region) == {"p_rib", "p_pad", "n_rib", "n_pad"}
        assert by_region["p_rib"].dopant_type == "acceptor"
        assert by_region["n_rib"].dopant_type == "donor"
        assert by_region["p_rib"].concentration_cm3 == 1e18

    def test_the_derived_window_reaches_the_sim(self, study):
        sim = study.charge.simulation()
        assert sim.cross_section.window == pytest.approx(study.layout.window)

    def test_an_explicit_window_reaches_the_sim(self, study):
        study.charge(window=(-21.0, -19.0))
        sim = study.charge.simulation()
        assert sim.cross_section.window == pytest.approx((-21.0, -19.0))


class TestTheDepletionEdgeIsResolved:
    """The mesh is held fine where the depletion edge moves, not only on lines."""

    def _mesh_kwargs_of_a_run(self, study, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh",
            lambda _self, **kwargs: seen.update(kwargs),
        )
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.sweep",
            lambda _self, _biases, **_kwargs: BiasSweepResult(
                contact="cathode", points=[]
            ),
        )
        monkeypatch.setattr("gsim.modulator.charge.require_devsim", lambda: None)
        study.charge.run()
        return seen

    def test_the_junction_regions_are_held_to_the_lines_size(self, study, monkeypatch):
        seen = self._mesh_kwargs_of_a_run(study, monkeypatch)
        span = study.layout.junction_span
        assert seen["refinement_boxes"] == [
            (*span.h, *span.z, 0.5 * study.charge.mesh["refined_mesh_size"])
        ]

    def test_boxes_given_in_the_mesh_settings_are_kept(self, study, monkeypatch):
        study.charge(mesh=study.charge.mesh | {"refinement_boxes": []})
        assert self._mesh_kwargs_of_a_run(study, monkeypatch)["refinement_boxes"] == []


class TestTheOxideAroundTheJunction:
    """Ticket 05: the electrostatic solve can reach into the surrounding oxide."""

    def test_the_oxide_is_in_the_solve_by_default(self, study):
        assert study.charge.oxide is True

    def test_the_solve_is_silicon_only_when_asked(self, study):
        study.charge(oxide=False)
        assert study.charge.simulation().insulators == []

    def test_the_stacks_oxide_joins_the_solve_as_an_insulator(self, study):
        sim = study.charge.simulation()
        [oxide] = sim.insulators
        assert oxide.region == "sio2"
        assert oxide.relative_permittivity == pytest.approx(
            study.stack.materials["sio2"]["permittivity"]
        )

    def test_no_doped_region_is_declared_insulating(self, study):
        # A background slab drawn in a doped Region's own material.
        slab = dict(study.stack.dielectrics[0])
        study.stack.dielectrics.append(
            {**slab, "name": "doped_slab", "material": "p_rib"}
        )
        study.charge(oxide=True)
        sim = study.charge.simulation()
        doped = {profile.region for profile in sim.doping}
        assert "p_rib" in doped
        assert [oxide.region for oxide in sim.insulators] == ["sio2"]

    def test_a_dielectric_that_conducts_is_not_declared_insulating(self, study):
        # A silicon substrate is a background slab too, and no insulator.
        slab = dict(study.stack.dielectrics[0])
        study.stack.materials["substrate_si"] = {
            "permittivity": 11.9,
            "conductivity": 2.0,
        }
        study.stack.dielectrics.append(
            {**slab, "name": "substrate", "material": "substrate_si"}
        )
        study.charge(oxide=True)
        sim = study.charge.simulation()
        assert [oxide.region for oxide in sim.insulators] == ["sio2"]


class TestMissingExtra:
    def test_running_without_devsim_names_the_extra(self, study, monkeypatch):
        monkeypatch.setitem(sys.modules, "devsim", None)
        with pytest.raises(ImportError, match=r"gsim\[tcad\]"):
            study.charge.run()

    def test_the_check_happens_before_any_meshing(self, study, monkeypatch):
        monkeypatch.setitem(sys.modules, "devsim", None)
        meshed = []
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh",
            lambda self, **kwargs: meshed.append(kwargs),
        )
        with pytest.raises(ImportError):
            study.charge.run()
        assert meshed == []


class TestSweptContact:
    def test_defaults_to_the_contact_on_the_n_side(self, study, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh", lambda s, **k: None
        )
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.sweep",
            lambda self, biases, contact=None, verbose=False: (
                calls.append(contact)
                or BiasSweepResult(contact=str(contact), points=[])
            ),
        )
        monkeypatch.setattr("gsim.modulator.charge.require_devsim", lambda: None)

        assert study.charge.contact is None
        study.charge.run()

        assert calls == ["cathode"]


class TestDevsimRelease:
    def test_a_re_run_releases_the_previous_devsim_device(self, study, monkeypatch):
        released = []

        def record_reset(sim):
            released.append(sim)

        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh", lambda s, **k: None
        )
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.sweep",
            lambda self, biases, contact=None, verbose=False: BiasSweepResult(
                contact=str(contact), points=[]
            ),
        )
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.reset_device",
            record_reset,
        )
        monkeypatch.setattr("gsim.modulator.charge.require_devsim", lambda: None)

        study.charge.run()
        assert released == []

        study.charge(biases=[0.0, 0.5])
        study.charge.run()
        assert len(released) == 1


class TestRun:
    def test_returns_the_bias_sweep_result_type(self, study, monkeypatch):
        sweep = BiasSweepResult(contact="cathode", points=[])
        calls = []

        def fake_sweep(_self, biases, *, contact=None, verbose=False):  # noqa: ARG001
            calls.append((list(biases), contact))
            return sweep

        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh", lambda s, **k: None
        )
        monkeypatch.setattr("gsim.tcad.sim.ChargeTransportSim.sweep", fake_sweep)
        monkeypatch.setattr("gsim.modulator.charge.require_devsim", lambda: None)

        study.charge(biases=[0.0, -1.0], contact="cathode")
        result = study.charge.run()

        assert result is sweep
        assert calls == [([0.0, -1.0], "cathode")]
        assert study.charge.contact == "cathode"
        assert study.charge.run() is sweep
        assert len(calls) == 1

    def test_verbose_reports_the_stage(
        self, phase_shifter, device, tmp_path, monkeypatch, capsys
    ):
        from gsim.modulator import Study

        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh", lambda s, **k: None
        )
        streamed: list[bool] = []

        def fake_sweep(_self, _biases, *, contact=None, verbose=False):  # noqa: ARG001
            streamed.append(verbose)
            return BiasSweepResult(contact="c", points=[])

        monkeypatch.setattr("gsim.tcad.sim.ChargeTransportSim.sweep", fake_sweep)
        monkeypatch.setattr("gsim.modulator.charge.require_devsim", lambda: None)

        component, stack = phase_shifter
        study = Study(
            component=component,
            stack=stack,
            device=device,
            output_dir=tmp_path,
            verbose=True,
        )
        study.charge.run()

        out = capsys.readouterr().out.splitlines()
        assert len(out) == 2
        assert all("charge" in line for line in out)
        # The Study's verbosity reaches DEVSIM's own output.
        assert streamed == [True]


class TestJunctionModelExport:
    """The sweep's junction branch leaves the Study as a model file."""

    @pytest.fixture
    def swept(self, study, monkeypatch):
        """A Study whose charge Stage answers from the canned sweep."""
        from .conftest import junction_sweep

        solves = {"charge": 0}

        def solve(_stage):
            solves["charge"] += 1
            return junction_sweep()

        monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", solve)
        study.solves = solves
        return study

    def test_the_file_matches_the_stage_exactly(self, swept, tmp_path):
        from gsim.common.circuit import read_junction_model

        path = swept.charge.export_junction_model(tmp_path / "junction.json")
        model = read_junction_model(path)

        branch = swept.charge.run().junction_branch()
        assert model.bias_v.tolist() == swept.charge.run().voltages.tolist()
        assert model.r_s_ohm_m.tolist() == list(branch.r_s_ohm_m)
        assert model.c_j_f_per_m.tolist() == list(branch.c_j_f_per_m)

    def test_the_contact_and_settings_are_recorded(self, swept, tmp_path):
        from gsim.common.circuit import read_junction_model

        from .conftest import JUNCTION_FREQ_HZ

        model = read_junction_model(
            swept.charge.export_junction_model(tmp_path / "junction.json")
        )

        assert model.contact == "cathode"
        assert model.freq_hz == JUNCTION_FREQ_HZ
        assert model.provenance["temperature_k"] == swept.charge.temperature
        assert model.provenance["generator"].startswith("gsim ")

    def test_without_a_path_it_lands_in_the_charge_stage_directory(self, swept):
        path = swept.charge.export_junction_model()

        assert path == swept.stage_dir("charge") / "junction.json"
        assert path.exists()

    def test_the_export_runs_the_stage_first(self, swept, tmp_path):
        assert swept.charge.has_run is False

        swept.charge.export_junction_model(tmp_path / "junction.json")

        assert swept.charge.has_run is True
        assert swept.solves == {"charge": 1}

    def test_a_sweep_without_admittances_is_an_actionable_error(self, biased, tmp_path):
        # The canned `biased` fixture predates the small-signal solve.
        with pytest.raises(ValueError, match="admittance"):
            biased.charge.export_junction_model(tmp_path / "junction.json")

    def test_mixed_fit_frequencies_are_refused(self, swept, tmp_path):
        sweep = swept.charge.run()
        sweep.points[-1].admittance_freq_hz = 2e9

        with pytest.raises(ValueError, match="one fit frequency"):
            swept.charge.export_junction_model(tmp_path / "junction.json")
