"""The ``pn_phase_shifter`` preset and the demo device builder.

The preset turns a caller's component, stack and region map into a Study
with every Stage configured; the demo builder draws a rib device for
examples and tests. Neither of them solves anything, so nothing here
needs a solver runtime.
"""

from __future__ import annotations

import warnings

import pytest

from gsim.common.carriers import MobilityModel, PlasmaDispersionModel
from gsim.modulator import (
    Device,
    Study,
    demo_phase_shifter,
    pn_phase_shifter,
)


def _study_over(demo, **kwargs):
    """A Study over a demo device, through the preset."""
    return pn_phase_shifter(
        component=demo.component,
        stack=demo.stack,
        device=demo.device,
        **kwargs,
    )


class TestDemoBuilder:
    def test_it_draws_the_regions_its_device_description_names(self, demo):
        assert set(demo.device.doped_regions) <= set(demo.stack.layers)

    def test_it_draws_an_electrode_over_each_pad(self, demo):
        conductors = set(demo.stack.get_conductor_layers())

        assert {"anode_metal", "cathode_metal"} <= conductors

    def test_its_device_description_is_interpretable(self, demo):
        study = Study(component=demo.component, stack=demo.stack, device=demo.device)

        assert {c.name for c in study.layout.contacts} == {"anode", "cathode"}
        assert set(study.layout.junction.regions) == {"n_rib", "p_rib"}

    def test_its_geometry_is_configurable(self):
        wider = demo_phase_shifter(half_width_um=0.5, rib_height_um=0.3)
        study = Study(component=wider.component, stack=wider.stack, device=wider.device)
        span = study.layout.junction_span

        assert span.h[1] - span.h[0] == pytest.approx(1.0)
        assert span.z == pytest.approx((0.0, 0.3))

    def test_it_reports_every_dimension_it_drew(self, demo):
        drawn = demo_phase_shifter(half_width_um=0.5, waveguide_width_um=0.6)

        assert drawn.half_width_um == pytest.approx(0.5)
        assert drawn.waveguide_width_um == pytest.approx(0.6)
        assert drawn.rib_height_um == pytest.approx(demo.rib_height_um)


class TestPresetConfiguresEveryStage:
    def test_one_call_configures_all_five_stages(self, demo):
        study = _study_over(
            demo,
            biases=[0.0, 1.0],
            wavelength_um=1.55,
            frequencies_hz=[10e9, 40e9],
            n_strips=4,
            length_um=2000.0,
            n_group=3.8,
        )

        assert isinstance(study, Study)
        assert study.charge.biases == [0.0, 1.0]
        assert study.carriers.dispersion.wavelength_um == pytest.approx(1.55)
        assert study.optical.wavelength_um == pytest.approx(1.55)
        assert study.rf.frequencies_hz == [10e9, 40e9]
        assert study.rf.n_strips == 4
        assert study.line.length_um == pytest.approx(2000.0)
        assert study.line.n_group == pytest.approx(3.8)

    def test_it_configures_without_solving_anything(self, demo):
        study = _study_over(demo)

        assert not any(stage.has_run for stage in study.stages.values())

    def test_its_defaults_are_workable(self, demo):
        study = _study_over(demo)

        assert len(study.charge.biases) > 1
        assert len(study.rf.frequencies_hz) > 1
        assert study.rf.n_strips >= 1
        assert study.line.length_um > 0.0

    def test_the_rf_staircase_spans_the_doped_slab(self, demo):
        """The pads carry series resistance, so the RF strips include them."""
        study = _study_over(demo)

        assert study.rf.strip_span == pytest.approx(study.layout.doped_span)
        assert study.rf.strip_span[0] < study.layout.junction_span.h[0]
        assert study.rf.strip_span[1] > study.layout.junction_span.h[1]

    def test_the_default_rf_strips_resolve_a_depleted_junction(self, demo):
        """Across the whole doped slab, and still one Strip is depleted:
        the line Mode is between the electrodes, not shunted by the slab."""
        from gsim.tcad.results import BiasPoint, BiasSweepResult

        from .conftest import carriers_at

        study = _study_over(demo)
        study.charge.seed(
            BiasSweepResult(
                contact="cathode",
                points=[BiasPoint(bias_v=2.0, carriers=carriers_at(2.0))],
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            study.rf.staircase()

    def test_the_optical_staircase_states_its_extent_too(self, demo):
        """Both EM stages declare their strip extent in the same place."""
        study = _study_over(demo, route="palace", n_strips=3)

        assert study.optical.strip_span == pytest.approx(study.layout.doped_span)
        assert study.optical.strip_span == pytest.approx(study.rf.strip_span)

    def test_the_route_reaches_both_em_stages(self, demo):
        study = _study_over(demo, route="palace", n_strips=3)

        assert study.optical.route == "palace"
        assert study.rf.route == "palace"
        assert study.optical.effective_n_strips() == 3
        assert study.rf.n_strips == 3

    def test_the_femwell_optical_stage_keeps_the_continuous_profile(self, demo):
        study = _study_over(demo, n_strips=3)

        assert study.optical.effective_n_strips() is None
        assert study.rf.n_strips == 3

    def test_the_output_directory_is_passed_through(self, demo, tmp_path):
        study = _study_over(demo, output_dir=tmp_path / "preset")

        assert study.output_dir == tmp_path / "preset"

    def test_verbose_is_passed_through(self, demo):
        assert _study_over(demo, verbose=True).verbose is True


class TestPresetGeneratesNoGeometry:
    def test_it_studies_the_component_and_stack_it_was_given(self, demo):
        study = _study_over(demo)

        assert study.component is demo.component
        assert study.stack is demo.stack

    def test_it_accepts_a_region_map_as_a_plain_mapping(self, demo):
        by_mapping = pn_phase_shifter(
            component=demo.component,
            stack=demo.stack,
            device={"p_regions": ["p_rib", "p_pad"], "n_regions": ["n_rib", "n_pad"]},
        )

        assert by_mapping.device.doped_regions == demo.device.doped_regions


class TestPresetSetsDefaultsWithoutLocking:
    def test_every_stage_stays_reconfigurable(self, demo):
        study = _study_over(demo)

        study.charge(biases=[0.0, 3.0])
        study.carriers(mobility=MobilityModel.constant(mu_n_cm2=1000.0))
        study.optical(wavelength_um=1.31, num_modes=2)
        study.rf(frequencies_hz=[5e9], n_strips=2)
        study.line(length_um=1000.0)

        assert study.charge.biases == [0.0, 3.0]
        assert study.carriers.mobility == MobilityModel.constant(mu_n_cm2=1000.0)
        assert study.optical.wavelength_um == pytest.approx(1.31)
        assert study.optical.num_modes == 2
        assert study.rf.frequencies_hz == [5e9]
        assert study.rf.n_strips == 2
        assert study.line.length_um == pytest.approx(1000.0)

    def test_the_device_description_stays_replaceable(self, demo):
        study = _study_over(demo)

        study.device = Device(p_regions=["p_rib", "p_pad"], n_regions=["n_rib"])

        assert study.device.n_regions == ["n_rib"]


class TestPresetDispersionFollowsTheWavelength:
    def test_a_fitted_wavelength_selects_its_own_coefficients(self, demo):
        study = _study_over(demo, wavelength_um=1.31)

        assert study.carriers.dispersion.wavelength_um == pytest.approx(1.31)

    def test_an_unfitted_wavelength_warns_and_names_the_fits(self, demo):
        with pytest.warns(UserWarning, match=r"1\.4"):
            study = _study_over(demo, wavelength_um=1.4)

        assert study.optical.wavelength_um == pytest.approx(1.4)

    def test_an_unfitted_wavelength_falls_back_to_the_nearest_fit(self, demo):
        with pytest.warns(UserWarning, match="nearest"):
            study = _study_over(demo, wavelength_um=1.32)

        assert study.carriers.dispersion.wavelength_um == pytest.approx(1.31)

    def test_an_explicit_model_is_kept_and_does_not_warn(self, demo):
        model = PlasmaDispersionModel.soref_1550()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            study = _study_over(demo, wavelength_um=1.4, dispersion=model)

        assert study.carriers.dispersion == model


class TestPresetRejectsWhatItCannotInterpret:
    def test_an_undrawn_doped_region_is_named_at_preset_time(self, demo):
        with pytest.raises(ValueError, match="p_slab"):
            pn_phase_shifter(
                component=demo.component,
                stack=demo.stack,
                device=Device(p_regions=["p_slab"], n_regions=["n_rib", "n_pad"]),
            )

    def test_an_undrawn_electrode_is_named_at_preset_time(self, demo):
        with pytest.raises(ValueError, match="top_metal"):
            pn_phase_shifter(
                component=demo.component,
                stack=demo.stack,
                device=Device(
                    p_regions=["p_rib", "p_pad"],
                    n_regions=["n_rib", "n_pad"],
                    electrodes=["top_metal"],
                ),
            )

    def test_an_electrode_landing_on_nothing_is_named_at_preset_time(self, demo):
        # The cathode metal is drawn over the n pad, so a description that
        # calls only that pad n-doped leaves the p side without a contact.
        with pytest.raises(ValueError, match="p-side contact"):
            pn_phase_shifter(
                component=demo.component,
                stack=demo.stack,
                device=Device(
                    p_regions=["p_rib", "p_pad"],
                    n_regions=["n_rib", "n_pad"],
                    electrodes=["cathode_metal"],
                ),
            )

    def test_the_error_names_the_preset_that_could_not_interpret_it(self, demo):
        with pytest.raises(ValueError, match="pn_phase_shifter"):
            pn_phase_shifter(
                component=demo.component,
                stack=demo.stack,
                device=Device(p_regions=["p_slab"], n_regions=["n_rib", "n_pad"]),
            )


@pytest.mark.tcad_local
class TestPresetDefaultsSolve:
    """The preset's defaults are workable, not merely well-typed.

    Gated on DEVSIM: the charge Stage is the one whose settings the
    preset can most easily get wrong, since its Window and its Contacts
    are the derivation everything downstream stands on.
    """

    def test_the_charge_stage_runs_on_the_presets_settings(
        self, demo, tmp_path_factory
    ):
        study = _study_over(
            demo,
            biases=[0.0, 1.0],
            output_dir=tmp_path_factory.mktemp("preset-charge"),
        )

        sweep = study.charge.run()

        assert [point.bias_v for point in sweep.points] == [0.0, 1.0]
        assert study.charge.has_run

    def test_the_carriers_stage_follows_it(self, demo, tmp_path_factory):
        study = _study_over(
            demo,
            biases=[0.0, 1.0],
            output_dir=tmp_path_factory.mktemp("preset-carriers"),
        )

        response = study.carriers.run()

        assert len(response.points) == 2
        assert study.charge.has_run
