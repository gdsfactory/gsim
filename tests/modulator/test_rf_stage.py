"""The RF Stage: its Staircase, its signal conductor, its Window.

Nothing here needs gmsh or femwell. The Staircase is built from a canned
Bias sweep, so what is under test is the derivation the Stage does — the
Strips tiling the Junction extent, the electrode carrying the RF signal,
and the Window defaulting to the whole Cross-section — plus the error a
user without the femwell extra gets. The real solve lives in
``test_rf_stage_runtime.py``.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from gsim.common.carriers import MobilityModel
from gsim.modulator.staircase import ElectrodeSpec

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, SLAB


class TestConfiguration:
    def test_defaults_are_readable(self, biased):
        assert biased.rf.frequencies_hz == [10e9, 40e9]
        assert biased.rf.n_strips == 21
        assert biased.rf.bias_v is None
        assert biased.rf.window is None
        assert biased.rf.has_run is False

    def test_the_section_is_callable(self, biased):
        assert biased.rf(frequencies_hz=[20e9], n_strips=3) is biased.rf
        assert biased.rf.frequencies_hz == [20e9]
        assert biased.rf.n_strips == 3

    def test_unknown_setting_is_rejected(self, biased):
        with pytest.raises(ValueError, match="nope"):
            biased.rf(nope=1)

    def test_a_frequency_list_that_is_empty_is_rejected(self, biased):
        with pytest.raises(ValueError):
            biased.rf(frequencies_hz=[])

    def test_a_nonpositive_frequency_is_rejected(self, biased):
        with pytest.raises(ValueError, match="positive"):
            biased.rf(frequencies_hz=[0.0])

    def test_fewer_than_one_strip_is_rejected(self, biased):
        with pytest.raises(ValueError):
            biased.rf(n_strips=0)


class TestBiasPoint:
    def test_the_last_point_of_the_sweep_is_the_default(self, biased):
        assert biased.rf.bias_point().bias_v == 2.0

    def test_a_chosen_bias_selects_its_point(self, biased):
        biased.rf(bias_v=0.0)
        assert biased.rf.bias_point().bias_v == 0.0

    def test_a_bias_the_sweep_never_visited_is_reported(self, biased):
        biased.rf(bias_v=-3.0)
        with pytest.raises(ValueError, match=r"-3|0\.0, 2\.0"):
            biased.rf.bias_point()


class TestStaircase:
    def test_the_strips_tile_the_junction_extent(self, biased):
        """The RF staircase stands on its own, so the rib is its default."""
        biased.rf(n_strips=4)

        staircase = biased.rf.staircase()

        assert len(staircase.strip_names) == 4
        edges = np.asarray(staircase.strips.edges_um, dtype=float)
        assert edges[0] == pytest.approx(CENTER_Y - HALF_WIDTH)
        assert edges[-1] == pytest.approx(CENTER_Y + HALF_WIDTH)

    def test_the_strip_span_is_overridable(self, biased):
        """A wider span carries the pads, and their resistance, into RF."""
        slab = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)
        biased.rf(n_strips=4, strip_span=slab)

        edges = np.asarray(biased.rf.staircase().strips.edges_um, dtype=float)

        assert edges[0] == pytest.approx(slab[0])
        assert edges[-1] == pytest.approx(slab[1])

    def test_the_junction_takes_n_strips_and_each_other_region_its_own(self, biased):
        """Across the slab: the rib at the configured count, each pad in
        strips_per_region Strips of its own, none straddling a boundary."""
        biased.rf(n_strips=4, strips_per_region=2, strip_span=SLAB)

        edges = np.asarray(biased.rf.staircase().strips.edges_um, dtype=float)

        rib = (CENTER_Y - HALF_WIDTH, CENTER_Y + HALF_WIDTH)
        expected = np.concatenate(
            (
                np.linspace(SLAB[0], rib[0], 3),
                np.linspace(rib[0], rib[1], 5)[1:],
                np.linspace(rib[1], SLAB[1], 3)[1:],
            )
        )
        np.testing.assert_allclose(edges, expected)

    def test_strips_that_resolve_the_rib_are_not_reported(self, biased):
        biased.rf(n_strips=21, strip_span=SLAB)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            biased.rf.staircase()

    def test_strips_that_average_the_depletion_away_are_reported(self, biased):
        """No dielectric Strip left: the slab shunts the two electrodes."""
        biased.rf(n_strips=2)
        with pytest.warns(UserWarning, match="no depleted strip"):
            biased.rf.staircase()

    def test_a_depleted_strip_is_not_reported(self, biased):
        biased.rf(n_strips=21)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            biased.rf.staircase()

    def test_the_unloaded_staircase_is_not_reported(self, biased):
        """Carriers switched off, every Strip is a dielectric."""
        biased.rf(n_strips=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            biased.rf.unloaded_staircase()

    def test_the_strips_carry_the_carrier_derived_conductivity(self, biased):
        staircase = biased.rf.staircase()

        sigma = np.asarray(staircase.strips.conductivity_s_per_m, dtype=float)
        assert np.all(sigma > 0.0)
        # mu_n > mu_p, and the n side is the low-h one on this device.
        assert sigma[0] > sigma[-1]

    def test_the_mobilities_come_from_the_carriers_stage(self, biased):
        biased.carriers(mobility=MobilityModel.constant(mu_n_cm2=1.0, mu_p_cm2=1.0))
        slow = np.asarray(
            biased.rf.staircase().strips.conductivity_s_per_m, dtype=float
        )

        biased.carriers(
            mobility=MobilityModel.constant(mu_n_cm2=1000.0, mu_p_cm2=1000.0)
        )
        fast = np.asarray(
            biased.rf.staircase().strips.conductivity_s_per_m, dtype=float
        )

        assert np.all(fast > slow)

    def test_the_electrodes_flank_the_strip_extent(self, biased):
        biased.rf(electrodes=ElectrodeSpec(width_um=3.0, gap_um=0.5))

        staircase = biased.rf.staircase()

        low, high = staircase.electrode_spans
        assert low == pytest.approx((CENTER_Y - HALF_WIDTH - 3.5, CENTER_Y - 0.8))
        assert high == pytest.approx((CENTER_Y + 0.8, CENTER_Y + HALF_WIDTH + 3.5))

    def test_the_caller_never_assembles_a_second_component(self, biased):
        staircase = biased.rf.staircase()

        assert staircase.component is not biased.component
        assert set(staircase.strip_names) <= set(staircase.stack().layers)


class TestConductorModel:
    """Which model of the electrode metal a run takes (ADR 0003)."""

    def test_the_femwell_route_meshes_the_metal_as_a_region(self, biased):
        """femwell can carry the metal's own loss, so it does."""
        assert biased.rf.effective_conductor_model() == "volume"
        staircase = biased.rf.staircase()
        assert staircase.conductor_model == "volume"
        stack = staircase.stack()
        assert stack.layers[staircase.electrode_names[0]].layer_type == "dielectric"

    def test_the_palace_route_meshes_the_metal_as_a_perfect_conductor(self, biased):
        """A metal region takes palace's eigenvalue search over; an
        outline does not."""
        biased.rf(route="palace")
        assert biased.rf.effective_conductor_model() == "pec"
        staircase = biased.rf.staircase()
        assert staircase.conductor_model == "pec"
        stack = staircase.stack()
        assert stack.layers[staircase.electrode_names[0]].layer_type == "conductor"

    @pytest.mark.parametrize("route", ["femwell", "palace"])
    @pytest.mark.parametrize("model", ["volume", "pec"])
    def test_an_explicit_model_overrides_the_route_default(self, biased, route, model):
        """What makes the two routes comparable: one cross-section, both."""
        biased.rf(route=route, conductor_model=model)
        assert biased.rf.effective_conductor_model() == model
        assert biased.rf.staircase().conductor_model == model

    def test_changing_the_model_invalidates_the_result(self, biased):
        biased.rf.seed(object())
        biased.rf(conductor_model="pec")
        assert not biased.rf.has_run

    def test_the_default_is_read_off_the_adapter(self, biased, fake_route):
        """Whatever route answers the Route, its model is the default."""
        biased.rf(route="palace")
        assert biased.rf.effective_conductor_model() == fake_route.conductor_model
        fake_route.conductor_model = "volume"
        assert biased.rf.effective_conductor_model() == "volume"


class TestPerfectElectrodesNeedTheWall:
    """femwell has one perfect-conductor condition, for the whole boundary."""

    def test_a_pec_electrode_without_the_wall_is_refused(self):
        """Off, the electrode hole would come out as an open slot."""
        from gsim.modulator.femwell_route import FemwellRoute

        with pytest.raises(ValueError, match="open slots"):
            FemwellRoute().check_line_settings(
                conductor_model="pec",
                metallic_boundaries=False,
                order=2,
                stage_name="rf",
            )

    def test_a_pec_electrode_with_the_wall_is_fine(self):
        from gsim.modulator.femwell_route import FemwellRoute

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FemwellRoute().check_line_settings(
                conductor_model="pec",
                metallic_boundaries=True,
                order=2,
                stage_name="rf",
            )

    def test_a_volume_electrode_does_not_need_the_wall(self, biased):
        """A metal region is a conductor whatever the boundary is."""
        from gsim.modulator.femwell_route import FemwellRoute

        biased.rf(conductor_model="volume", metallic_boundaries=False)
        assert biased.rf.effective_conductor_model() == "volume"
        FemwellRoute().check_line_settings(
            conductor_model="volume",
            metallic_boundaries=False,
            order=1,
            stage_name="rf",
        )

    def test_the_refusal_comes_before_anything_is_meshed(self, study, monkeypatch):
        """Settings the Route cannot honour cost no charge solve and no mesh."""

        def fail(*_args, **_kwargs):
            raise AssertionError("the charge stage must not run")

        monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", fail)
        study.rf(route="femwell", conductor_model="pec", metallic_boundaries=False)
        with pytest.raises(ValueError, match="open slots"):
            study.rf.run()

    def test_the_stage_hands_its_settings_to_the_adapter_first(
        self, biased, fake_route, monkeypatch
    ):
        """Whatever the Route makes of them, the Stage asks before it meshes."""

        def fail(*_args, **_kwargs):
            raise AssertionError("nothing may be meshed before the check")

        monkeypatch.setattr("gsim.palace.BoundaryModeSim.mesh", fail)
        biased.rf(route="palace", conductor_model="pec", order=2)
        with pytest.raises(AssertionError, match="before the check"):
            biased.rf.run()

        (checked,) = fake_route.made("check_line_settings")
        assert checked["conductor_model"] == "pec"
        assert checked["metallic_boundaries"] is True
        assert checked["order"] == 2
        assert fake_route.made("require") == [{"stage_name": "rf"}]


class TestMetallicWall:
    """Both Routes put the same condition on the Window's outer wall."""

    def test_the_simulation_carries_the_wall_by_default(self, biased):
        assert biased.rf.metallic_boundaries is True
        assert biased.rf.simulation().metallic_boundaries is True

    def test_turning_the_wall_off_reaches_the_simulation(self, biased):
        biased.rf(conductor_model="volume", metallic_boundaries=False)
        assert biased.rf.simulation().metallic_boundaries is False


class TestContourOrder:
    """A perfect conductor's current is read off the field around it."""

    def test_a_first_order_solve_of_a_pec_staircase_is_reported(self):
        from gsim.modulator.femwell_route import FemwellRoute

        with pytest.warns(UserWarning, match="biased high by tens of percent"):
            FemwellRoute().check_line_settings(
                conductor_model="pec",
                metallic_boundaries=True,
                order=1,
                stage_name="rf",
            )

    def test_a_second_order_solve_is_not(self):
        from gsim.modulator.femwell_route import FemwellRoute

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FemwellRoute().check_line_settings(
                conductor_model="pec",
                metallic_boundaries=True,
                order=2,
                stage_name="rf",
            )


class TestStripMaterials:
    def test_the_strips_are_valid_up_to_the_highest_frequency_solved(self, biased):
        biased.rf(frequencies_hz=[10e9, 90e9])

        from gsim.common.stack.materials import MaterialProperties

        props = biased.rf.staircase().stack().materials["strip_0"]
        material = (
            props
            if isinstance(props, MaterialProperties)
            else MaterialProperties.model_validate(props)
        )
        model = material.dispersion_models[0]

        assert model.validity.valid_frequency == (0, 90e9)


class TestSignalConductor:
    def test_the_signal_conductor_is_the_swept_contacts_electrode(self, biased):
        # The charge sweep drives the n-side contact, which is the low-h
        # side of this device, so the low electrode carries the signal.
        assert biased.charge.swept_contact() == "cathode"
        assert biased.rf.signal_contact_name() == "cathode"
        assert biased.rf.signal_electrode() == "electrode_low"

    def test_the_other_contact_selects_the_other_electrode(self, biased):
        biased.rf(signal_contact="anode")
        assert biased.rf.signal_electrode() == "electrode_high"

    def test_a_contact_the_device_does_not_have_is_reported(self, biased):
        biased.rf(signal_contact="gate")
        with pytest.raises(ValueError, match="gate"):
            biased.rf.signal_electrode()

    def test_renamed_electrodes_are_followed(self, biased):
        biased.rf(electrodes=ElectrodeSpec(names=("ground", "signal")))
        biased.rf(signal_contact="anode")
        assert biased.rf.signal_electrode() == "signal"


def run_rf(study, fake_route, *, modes_at, **settings):
    """Run the RF Stage on the fake route, with the solves scripted."""
    fake_route.modes_at = modes_at
    study.rf(route="palace", **settings)
    return study.rf.run()


class TestModeTracking:
    """Where each frequency's eigenvalue search is aimed, and what it finds."""

    def test_the_first_frequency_is_aimed_at_the_configured_guess(
        self, biased, fake_route
    ):
        run_rf(biased, fake_route, modes_at=[3.9 - 0.2j], n_guess=2.5)

        assert fake_route.made("solve")[0]["target"] == pytest.approx(2.5)

    def test_later_frequencies_follow_the_mode_they_just_solved(
        self, biased, fake_route
    ):
        run_rf(
            biased,
            fake_route,
            modes_at=[3.9 - 0.2j],
            n_guess=2.5,
            frequencies_hz=[10e9, 20e9],
        )

        targets = [call["target"] for call in fake_route.made("solve")]
        assert targets == pytest.approx([2.5, 3.9])

    def test_tracking_is_switchable_off(self, biased, fake_route):
        run_rf(
            biased,
            fake_route,
            modes_at=[3.9 - 0.2j],
            n_guess=2.5,
            frequencies_hz=[10e9, 20e9],
            track_modes=False,
        )

        targets = [call["target"] for call in fake_route.made("solve")]
        assert targets == pytest.approx([2.5, 2.5])

    def test_a_dense_sweep_is_judged_on_its_own_spacing(self, biased, fake_route):
        """A step that is fine over a doubling is a jump over 1%."""
        indices = {40e9: [3.90 + 0j], 40.4e9: [3.70 + 0j]}
        with pytest.warns(UserWarning, match="jumps across the sweep"):
            run_rf(
                biased,
                fake_route,
                modes_at=lambda f: indices[f],
                frequencies_hz=[40e9, 40.4e9],
            )

    def test_a_dispersing_index_is_not_reported_as_a_jump(self, biased, fake_route):
        indices = {10e9: [3.90 + 0j], 20e9: [3.85 + 0j], 40e9: [3.80 + 0j]}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_rf(
                biased,
                fake_route,
                modes_at=lambda f: indices[f],
                frequencies_hz=[10e9, 20e9, 40e9],
            )

    def test_an_index_that_steps_between_frequencies_is_reported(
        self, biased, fake_route
    ):
        indices = {10e9: [3.90 + 0j], 20e9: [3.85 + 0j], 40e9: [1.4 + 0j]}
        with pytest.warns(UserWarning, match=r"20 -> 40 GHz"):
            run_rf(
                biased,
                fake_route,
                modes_at=lambda f: indices[f],
                frequencies_hz=[10e9, 20e9, 40e9],
            )


class TestSolveLoop:
    """One loop for both Routes: prepare, solve, select, read, per frequency."""

    def test_the_line_is_prepared_once_with_both_electrodes(self, biased, fake_route):
        run_rf(biased, fake_route, modes_at=[3.0 - 0.01j], frequencies_hz=[10e9, 20e9])

        (prepared,) = fake_route.made("prepare_line")
        assert prepared["signal"].name == "electrode_low"
        assert prepared["return_"].name == "electrode_high"
        assert prepared["signal"].model == "pec"
        assert prepared["stage_name"] == "rf"

    def test_every_frequency_is_solved_and_read_once(self, biased, fake_route):
        line = run_rf(
            biased, fake_route, modes_at=[3.0 - 0.01j], frequencies_hz=[10e9, 20e9]
        )

        assert [c["freq_hz"] for c in fake_route.made("solve")] == [10e9, 20e9]
        assert [c["freq_hz"] for c in fake_route.made("read_line")] == [10e9, 20e9]
        np.testing.assert_allclose(line.n_rf, [3.0, 3.0])
        np.testing.assert_allclose(line.z0_ohm, [50.0, 50.0])

    def test_the_selected_mode_is_the_one_read(self, biased, fake_route):
        run_rf(
            biased,
            fake_route,
            modes_at=[0.4 - 0.001j, 3.2 - 0.01j, 2.1 - 0.005j],
            frequencies_hz=[10e9],
        )

        (read,) = fake_route.made("read_line")
        assert read["mode"].n_eff == 3.2 - 0.01j

    def test_a_lossy_loaded_line_is_selected_over_the_wall_mode(
        self, biased, fake_route
    ):
        """The demo's depleted line at 10 GHz on 61 strips, beside its wall
        Mode: it loses 0.74 of a radian per radian, and is still the line."""
        wall, line = 2.2486 - 0.7121j, 6.4802 - 4.7918j
        run_rf(biased, fake_route, modes_at=[wall, line], frequencies_hz=[10e9])

        (read,) = fake_route.made("read_line")
        assert read["mode"].n_eff == line

    def test_the_reading_is_what_the_result_carries(self, biased, fake_route):
        from gsim.common.modes import LineReading

        fake_route.reading = lambda mode, freq_hz: LineReading(
            n_eff=mode.n_eff, z0_ohm=complex(40.0 + freq_hz / 1e9), wall_mode=False
        )
        line = run_rf(
            biased, fake_route, modes_at=[3.0 - 0.01j], frequencies_hz=[10e9, 20e9]
        )
        np.testing.assert_allclose(line.z0_ohm, [50.0, 60.0])

    def test_a_squeezed_mode_warns_through_the_stage(self, biased, fake_route):
        fake_route.boundary = 0.5
        with pytest.warns(UserWarning, match="rf stage's mode.*window boundary"):
            run_rf(biased, fake_route, modes_at=[3.0 - 0.01j])

    def test_an_unmeasured_boundary_ratio_does_not_warn(self, biased, fake_route):
        fake_route.boundary = float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_rf(biased, fake_route, modes_at=[3.0 - 0.01j])


class TestWindow:
    def test_the_window_defaults_to_the_full_cross_section(self, biased):
        cross_section = biased.rf.simulation().cross_section

        assert cross_section.window is None
        assert cross_section.window_z is None

    def test_an_explicit_window_overrides_the_default(self, biased):
        biased.rf(window=(CENTER_Y - 4.0, CENTER_Y + 4.0), window_z=(-1.0, 1.0))

        cross_section = biased.rf.simulation().cross_section

        assert cross_section.window == pytest.approx((CENTER_Y - 4.0, CENTER_Y + 4.0))
        assert cross_section.window_z == pytest.approx((-1.0, 1.0))

    def test_the_rf_stage_writes_into_its_own_directory(self, biased):
        sim = biased.rf.simulation()

        assert sim.output_dir == biased.stage_dir("rf")
        assert sim.output_dir != biased.stage_dir("optical")


class TestInvalidation:
    def test_the_rf_stage_is_downstream_of_the_carriers_stage(self, study):
        assert study.rf in study.carriers._downstream
        assert study.rf in study.charge._downstream

    def test_the_optical_and_rf_stages_do_not_invalidate_each_other(self, study):
        assert study.rf not in study.optical._downstream
        assert study.optical not in study.rf._downstream


class TestMissingExtra:
    def test_running_without_femwell_names_the_extra(self, biased, monkeypatch):
        monkeypatch.setitem(sys.modules, "femwell", None)
        with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
            biased.rf.run()

    def test_the_extra_is_checked_before_anything_is_meshed(self, study, monkeypatch):
        """A user without femwell pays for no mesh and no charge solve."""
        monkeypatch.setitem(sys.modules, "femwell", None)

        def fail(*_args, **_kwargs):
            raise AssertionError("the charge stage must not run")

        monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", fail)
        with pytest.raises(ImportError):
            study.rf.run()


class TestUnloadedStaircase:
    def test_the_carriers_are_switched_off(self, biased):
        staircase = biased.rf.unloaded_staircase()

        sigma = np.asarray(staircase.strips.conductivity_s_per_m, dtype=float)
        assert np.all(sigma == 0.0)

    def test_the_geometry_is_the_loaded_solves(self, biased):
        loaded = biased.rf.staircase()
        unloaded = biased.rf.unloaded_staircase()

        assert unloaded.strip_names == loaded.strip_names
        assert unloaded.electrode_spans == loaded.electrode_spans
        assert np.asarray(unloaded.strips.edges_um, dtype=float) == pytest.approx(
            np.asarray(loaded.strips.edges_um, dtype=float)
        )


class TestJunctionBranchLookup:
    def rc_sweep(self, study, r_s_ohm_m=8e-4, c_j_f_per_m=2.4e-10, freq_hz=1e6):
        """Give the charge Stage's points an exact series-RC admittance."""
        from gsim.tcad.results import BiasSweepResult
        from tests._helpers import series_rc_admittance

        y_per_m = series_rc_admittance(freq_hz, r_s_ohm_m, c_j_f_per_m)
        sweep: BiasSweepResult = study.charge.result
        study.charge.seed(
            sweep.model_copy(
                update={
                    "points": [
                        point.model_copy(
                            update={
                                "admittance_s_per_cm": complex(y_per_m) / 1e2,
                                "admittance_freq_hz": freq_hz,
                            }
                        )
                        for point in sweep.points
                    ]
                }
            )
        )

    def test_the_branch_is_read_at_the_stages_bias_point(self, biased):
        self.rc_sweep(biased)
        biased.rf(bias_v=2.0)

        r_s, c_j = biased.rf.junction_branch()

        assert r_s == pytest.approx(8e-4, rel=1e-9)
        assert c_j == pytest.approx(2.4e-10, rel=1e-9)

    def test_a_sweep_without_admittances_says_how_to_get_them(self, biased):
        with pytest.raises(ValueError, match="admittance"):
            biased.rf.junction_branch()

    def test_the_whole_sweeps_branches_are_read_in_sweep_order(self, biased):
        self.rc_sweep(biased)

        bias_v, branch = biased.rf.junction_branches()

        assert bias_v.tolist() == [0.0, 2.0]
        np.testing.assert_allclose(branch.r_s_ohm_m, [8e-4, 8e-4], rtol=1e-9)
        np.testing.assert_allclose(branch.c_j_f_per_m, [2.4e-10, 2.4e-10], rtol=1e-9)


class TestWallModeWarning:
    """The Stage says when the Route's reading is the wall Mode's (ADR 0005).

    A shielded line has two propagating Modes, and a lossy loaded line
    at an undepleted Bias can leave the default rule with the wrong one:
    both electrodes at one potential, returning through the Window wall.
    Which Route caught it, and how, is the reading's business; the
    warning is the Stage's, and there is one of it.
    """

    @staticmethod
    def _reading(wall_mode, diagnostic=""):
        from gsim.common.modes import LineReading

        return lambda mode, freq_hz: LineReading(
            n_eff=mode.n_eff,
            z0_ohm=182.0 + 0j,
            wall_mode=wall_mode,
            diagnostic=diagnostic,
        )

    def test_a_wall_mode_reading_is_reported_with_the_way_out(self, biased, fake_route):
        fake_route.reading = self._reading(
            True,
            "whose two electrodes carry currents 100% in common: the mode "
            "between them and the window wall",
        )
        with pytest.warns(UserWarning, match="window wall") as record:
            run_rf(biased, fake_route, modes_at=[2.0262 - 6.6e-7j])
        message = str(record[0].message)
        assert "rf stage" in message
        assert "10 GHz" in message
        assert "n_guess" in message
        assert "bias_v" in message

    def test_a_line_mode_reading_is_not(self, biased, fake_route):
        fake_route.reading = self._reading(False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_rf(biased, fake_route, modes_at=[2.9 - 1.4e-3j])

    def test_a_reading_that_could_not_tell_is_not_a_wall_mode(self, biased, fake_route):
        """No reading is no reading, and no warning."""
        fake_route.reading = self._reading(None)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_rf(biased, fake_route, modes_at=[2.0 + 0j])

    def test_the_way_out_names_the_stage_and_its_settings(self, biased, fake_route):
        """Every way out the warning offers names a setting of this Stage."""
        fake_route.reading = self._reading(True, "whose two electrodes agree")
        with pytest.warns(UserWarning, match="selected a mode") as record:
            run_rf(biased, fake_route, modes_at=[2.0262 - 6.6e-7j])
        message = str(record[0].message)
        assert "study.rf(bias_v=...)" in message
        assert "n_guess" in message
        assert "max_loss_ratio" in message
        assert "rule=" in message
