"""The carriers Stage: plasma dispersion, mobilities, and the coupling seam."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from gsim.common.carriers import (
    MobilityModel,
    PlasmaDispersionModel,
    carrier_absorption_cm,
    carrier_conductivity,
    carrier_index_shift,
)
from gsim.modulator.carriers import CarrierResponseSweep
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

N_CM3 = np.asarray([1e15, 1e17, 5e18])
P_CM3 = np.asarray([2e18, 3e16, 1e15])


def carrier_map(scale: float = 1.0) -> CarrierMap:
    """A three-node Carrier map with known concentrations."""
    return CarrierMap(
        x_um=np.asarray([0.0, 0.0, 0.0]),
        y_um=np.asarray([-20.2, -20.0, -19.8]),
        region=["n_rib", "n_rib", "p_rib"],
        electrons_cm3=N_CM3 * scale,
        holes_cm3=P_CM3 * scale,
    )


def fake_sweep() -> BiasSweepResult:
    """A two-point Bias sweep standing in for a charge solve."""
    return BiasSweepResult(
        contact="cathode",
        points=[
            BiasPoint(bias_v=0.0, carriers=carrier_map()),
            BiasPoint(bias_v=-1.0, carriers=carrier_map(scale=0.5)),
        ],
    )


@pytest.fixture
def solved_charge(study, monkeypatch):
    """The Study with its charge Stage answering from a canned sweep."""
    solves = []

    def record_solve(_stage):
        solves.append(1)
        return fake_sweep()

    monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", record_solve)
    return study, solves


class TestConfiguration:
    def test_the_published_defaults_are_readable(self, study):
        assert study.carriers.dispersion.wavelength_um == 1.55
        assert study.carriers.mobility is None
        assert study.carriers.effective_mobility() == MobilityModel.masetti_silicon()
        assert study.carriers.has_run is False

    def test_the_section_is_callable(self, study):
        constant = MobilityModel.constant(mu_n_cm2=1000.0)
        assert study.carriers(mobility=constant) is study.carriers
        assert study.carriers.mobility == constant

    def test_unknown_setting_is_rejected(self, study):
        with pytest.raises(ValueError, match="nope"):
            study.carriers(nope=1)

    def test_negative_mobility_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.carriers(mobility={"mu_max_n": 1417.0, "mu_max_p": -1.0})


class TestSubstitutedCoefficients:
    def test_a_foundry_model_replaces_the_published_fit(self, study):
        foundry = PlasmaDispersionModel.nedeljkovic_1550().model_copy(
            update={"dn_hole_coeff": 2.0e-18}
        )
        study.carriers(dispersion=foundry)

        response = study.carriers.response(N_CM3, P_CM3)

        assert study.carriers.dispersion.dn_hole_coeff == 2.0e-18
        assert response.index_shift == pytest.approx(
            carrier_index_shift(N_CM3, P_CM3, model=foundry)
        )

    def test_coefficients_can_be_given_as_a_mapping(self, study):
        study.carriers(dispersion=PlasmaDispersionModel.soref_1550().model_dump())
        assert study.carriers.dispersion.dn_electron_coeff == 8.8e-22

    def test_another_wavelength_is_a_configuration_change(self, study):
        study.carriers(dispersion=PlasmaDispersionModel.nedeljkovic_1310())
        assert study.carriers.dispersion.wavelength_um == 1.31


class TestResponseSeam:
    def test_the_coupling_matches_the_published_relationships(self, study):
        response = study.carriers.response(N_CM3, P_CM3)

        assert response.index_shift == pytest.approx(
            carrier_index_shift(N_CM3, P_CM3, model=study.carriers.dispersion)
        )
        assert response.absorption_cm == pytest.approx(
            carrier_absorption_cm(N_CM3, P_CM3, model=study.carriers.dispersion)
        )
        assert response.conductivity_s_per_m == pytest.approx(
            carrier_conductivity(N_CM3, P_CM3, mobility=MobilityModel.masetti_silicon())
        )

    def test_the_charge_stage_mobility_reaches_the_conductivity(self, study):
        # One model for the transport solve and the RF conductivity.
        study.charge(mobility=MobilityModel.constant(mu_n_cm2=100.0, mu_p_cm2=50.0))

        response = study.carriers.response(N_CM3, P_CM3)

        assert response.conductivity_s_per_m == pytest.approx(
            carrier_conductivity(N_CM3, P_CM3, mu_n_cm2=100.0, mu_p_cm2=50.0)
        )

    def test_the_configured_mobilities_reach_the_conductivity(self, study):
        study.carriers(mobility=MobilityModel.constant(mu_n_cm2=100.0, mu_p_cm2=50.0))

        response = study.carriers.response(N_CM3, P_CM3)

        assert response.conductivity_s_per_m == pytest.approx(
            carrier_conductivity(N_CM3, P_CM3, mu_n_cm2=100.0, mu_p_cm2=50.0)
        )

    def test_it_works_on_carriers_from_any_mesh(self, study):
        """The seam the EM Stages use on their own, transferred carriers."""
        response = study.carriers.response([1e17] * 5, [1e17] * 5)
        assert response.index_shift.shape == (5,)


class TestRun:
    def test_running_triggers_the_charge_stage_first(self, solved_charge):
        study, solves = solved_charge

        study.carriers.run()

        assert solves == [1]
        assert study.charge.has_run is True

    def test_an_already_solved_charge_stage_is_not_re_run(self, solved_charge):
        study, solves = solved_charge
        study.charge.run()

        study.carriers.run()

        assert solves == [1]

    def test_every_bias_point_gets_its_material_response(self, solved_charge):
        study, _ = solved_charge

        result = study.carriers.run()

        assert isinstance(result, CarrierResponseSweep)
        assert result.contact == "cathode"
        assert result.voltages == pytest.approx([0.0, -1.0])
        first = result.points[0]
        assert first.bias_v == 0.0
        assert first.index_shift == pytest.approx(
            carrier_index_shift(N_CM3, P_CM3, model=study.carriers.dispersion)
        )
        assert first.absorption_cm == pytest.approx(
            carrier_absorption_cm(N_CM3, P_CM3, model=study.carriers.dispersion)
        )
        assert first.conductivity_s_per_m == pytest.approx(
            carrier_conductivity(N_CM3, P_CM3, mobility=MobilityModel.masetti_silicon())
        )

    def test_the_response_spans_the_cross_section_nodes(self, solved_charge):
        study, _ = solved_charge

        point = study.carriers.run().points[1]

        assert point.carriers.x_um == pytest.approx([0.0, 0.0, 0.0])
        assert point.carriers.y_um == pytest.approx([-20.2, -20.0, -19.8])
        assert point.carriers.region == ["n_rib", "n_rib", "p_rib"]
        assert point.carriers.electrons_cm3 == pytest.approx(N_CM3 * 0.5)
        assert point.index_shift.shape == (3,)

    def test_running_twice_couples_once(self, solved_charge):
        study, _ = solved_charge
        assert study.carriers.run() is study.carriers.run()

    def test_no_solver_runtime_is_needed_to_couple(self, solved_charge, monkeypatch):
        from gsim.modulator import CarriersStage

        study, _ = solved_charge
        for name in ("devsim", "femwell", "skfem"):
            monkeypatch.setitem(sys.modules, name, None)

        assert study.carriers.run().points
        # And with no Study at all: the coupling is pure numpy.
        assert CarriersStage().response([1e17], [1e17]).index_shift.size == 1

    def test_verbose_reports_the_stage(self, solved_charge, capsys):
        study, _ = solved_charge
        study.verbose = True

        study.carriers.run()

        out = capsys.readouterr().out.splitlines()
        assert any("carriers" in line for line in out)


class TestResultIntegrity:
    def test_a_response_disagreeing_with_its_carrier_map_is_rejected(self):
        from gsim.modulator.carriers import CarrierResponse

        with pytest.raises(ValueError, match="3 nodes"):
            CarrierResponse(
                bias_v=0.0,
                carriers=carrier_map(),
                index_shift=np.zeros(2),
                absorption_cm=np.zeros(2),
                conductivity_s_per_m=np.zeros(2),
            )

    def test_quantities_disagreeing_in_length_are_rejected(self):
        from gsim.modulator.carriers import MaterialResponse

        with pytest.raises(ValueError, match="disagree in length"):
            MaterialResponse(
                index_shift=np.zeros(3),
                absorption_cm=np.zeros(2),
                conductivity_s_per_m=np.zeros(3),
            )


class TestInvalidation:
    def test_a_charge_change_drops_the_coupling(self, solved_charge):
        study, _ = solved_charge
        study.carriers.run()

        study.charge(biases=[0.0, -2.0])

        assert study.carriers.has_run is False

    def test_a_carriers_change_leaves_the_charge_solve_alone(self, solved_charge):
        study, solves = solved_charge
        study.carriers.run()

        study.carriers(mobility=MobilityModel.constant(mu_n_cm2=900.0))

        assert study.carriers.has_run is False
        assert study.charge.has_run is True
        study.carriers.run()
        assert solves == [1]

    def test_every_stage_is_wired_downstream_of_what_it_reads(self, study):
        """A Stage is cleared by its own upstream, and by nothing else."""
        cleared_by = {
            name: {
                other
                for other in study.stages
                if study.stages[name] in study.stages[other]._downstream
            }
            for name in study.stages
        }

        assert cleared_by == {
            "charge": set(),
            "carriers": {"charge"},
            # Both EM stages read the carriers stage, and neither reads
            # the other, so re-configuring one leaves the other alone.
            "optical": {"charge", "carriers"},
            "rf": {"charge", "carriers"},
            # The line stage reads both EM stages, so everything upstream
            # of either clears the device report.
            "line": {"charge", "carriers", "optical", "rf"},
        }

    def test_the_charge_stage_carries_carriers_as_downstream(self, study):
        assert study.carriers in study.charge._downstream
