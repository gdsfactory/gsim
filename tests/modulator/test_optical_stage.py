"""The optical Stage: its Window, its configuration, and the missing extra.

Everything here runs without femwell, skfem or DEVSIM: the Window
derivation and the Stage's configuration are pure derivation, and the
missing-extra path is exactly the one a user without the extra hits.
The real solve lives in ``test_optical_stage_runtime.py``.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from gsim.tcad.results import BiasPoint, BiasSweepResult

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, RIB_HEIGHT, carriers_at

JUNCTION_Y = CENTER_Y


class TestConfiguration:
    def test_defaults_are_readable(self, study):
        assert study.optical.wavelength_um == 1.55
        assert study.optical.num_modes == 1
        assert study.optical.window is None
        assert study.optical.has_run is False

    def test_the_section_is_callable(self, study):
        assert study.optical(wavelength_um=1.31, num_modes=2) is study.optical
        assert study.optical.wavelength_um == 1.31
        assert study.optical.num_modes == 2

    def test_unknown_setting_is_rejected(self, study):
        with pytest.raises(ValueError, match="nope"):
            study.optical(nope=1)

    def test_a_nonsense_wavelength_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.optical(wavelength_um=0.0)

    def test_at_least_one_mode_must_be_asked_for(self, study):
        with pytest.raises(ValueError):
            study.optical(num_modes=0)


class TestDerivedWindow:
    def test_the_window_is_a_box_around_the_rib_not_the_doped_slab(self, study):
        window = study.optical.simulation().cross_section.window

        # Centred on the junction, and wider than the charge window, which
        # is clipped to the doped slab between the contacts (ADR 0002).
        assert window == pytest.approx((JUNCTION_Y - 2.0, JUNCTION_Y + 2.0))
        assert window != pytest.approx(study.layout.window)

    def test_the_vertical_window_clears_the_guiding_layer(self, study):
        window_z = study.optical.simulation().cross_section.window_z

        assert window_z == pytest.approx((-1.0, RIB_HEIGHT + 1.0))

    def test_the_margin_sizes_the_window(self, study):
        study.optical(mode_margin_um=3.0, z_above_um=0.5, z_below_um=0.25)

        cross_section = study.optical.simulation().cross_section

        assert cross_section.window == pytest.approx(
            (JUNCTION_Y - 3.0, JUNCTION_Y + 3.0)
        )
        assert cross_section.window_z == pytest.approx((-0.25, RIB_HEIGHT + 0.5))

    def test_an_explicit_window_overrides_the_derivation(self, study):
        study.optical(window=(-21.0, -19.0), window_z=(-0.5, 0.8))

        cross_section = study.optical.simulation().cross_section

        assert cross_section.window == pytest.approx((-21.0, -19.0))
        assert cross_section.window_z == pytest.approx((-0.5, 0.8))

    def test_the_junction_is_where_the_doped_regions_meet(self, study):
        assert study.layout.junction_position == pytest.approx(JUNCTION_Y)

    def test_the_charge_window_spans_the_whole_doped_slab(self, study):
        """The Window the optical Stage is deliberately not reusing."""
        assert study.layout.window == pytest.approx(
            (
                CENTER_Y - HALF_WIDTH - PAD_WIDTH - 0.5,
                CENTER_Y + HALF_WIDTH + PAD_WIDTH + 0.5,
            )
        )


class TestOwnMesh:
    def test_the_optical_stage_writes_into_its_own_directory(self, study):
        sim = study.optical.simulation()
        assert sim.output_dir == study.stage_dir("optical")
        assert sim.output_dir != study.stage_dir("charge")

    def test_the_perturbed_regions_default_to_the_doped_ones(self, study):
        assert study.optical.perturbed_region_names() == study.device.doped_regions

    def test_the_perturbed_regions_are_overridable(self, study):
        study.optical(perturbed_regions=["p_rib", "n_rib"])
        assert study.optical.perturbed_region_names() == ["p_rib", "n_rib"]


class TestMissingExtra:
    def test_running_without_femwell_names_the_extra(self, study, monkeypatch):
        monkeypatch.setitem(sys.modules, "femwell", None)
        with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
            study.optical.run()

    def test_the_extra_is_checked_before_anything_is_meshed(self, study, monkeypatch):
        """A user without femwell pays for no mesh and no charge solve."""
        monkeypatch.setitem(sys.modules, "femwell", None)

        def fail(*_args, **_kwargs):
            raise AssertionError("the charge stage must not run")

        monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", fail)
        with pytest.raises(ImportError):
            study.optical.run()


class TestInvalidation:
    def test_a_carriers_change_drops_the_optical_result(self, study):
        order = list(study.stages)
        assert order.index("carriers") < order.index("optical")
        assert study.optical in study.carriers._downstream
        assert study.optical in study.charge._downstream

    def test_the_optical_stage_feeds_the_line_stage(self, study):
        assert study.optical._downstream == [study.line]


class TestStripExtent:
    """What the Strips tile, and what happens when they cannot reach it."""

    def test_the_default_is_the_doped_slab_not_the_rib(self, biased):
        """Ticket 19: strips on the rib alone solve the wrong waveguide."""
        extent = biased.optical.strip_extent()

        assert extent == pytest.approx(biased.layout.doped_span)
        assert extent[0] < biased.layout.junction_span.h[0]
        assert extent[1] > biased.layout.junction_span.h[1]

    def test_a_chosen_span_is_taken_as_given(self, biased):
        biased.optical(strip_span=(CENTER_Y - 0.1, CENTER_Y + 0.1))

        assert biased.optical.strip_extent() == pytest.approx(
            (CENTER_Y - 0.1, CENTER_Y + 0.1)
        )

    def test_a_chosen_span_is_narrowed_by_the_map_too(self, biased):
        """The preset chooses the span, so the clamp has to reach it."""
        biased.optical(strip_span=biased.layout.doped_span)
        carriers = biased.charge.result.points[0].carriers
        carriers.x_um = np.clip(carriers.x_um, CENTER_Y - 0.2, CENTER_Y + 0.2)

        with pytest.warns(UserWarning, match="the extent asked for"):
            extent = biased.optical.strip_extent(carriers)

        assert extent == pytest.approx((CENTER_Y - 0.2, CENTER_Y + 0.2))

    def test_a_carrier_map_narrower_than_the_slab_narrows_the_default(self, biased):
        """Strips cannot outrun the map they average, and say so."""
        carriers = biased.charge.result.points[0].carriers
        carriers.x_um = np.clip(carriers.x_um, CENTER_Y - 0.2, CENTER_Y + 0.2)

        with pytest.warns(UserWarning, match="derived for this stage"):
            extent = biased.optical.strip_extent(carriers)

        assert extent == pytest.approx((CENTER_Y - 0.2, CENTER_Y + 0.2))

    def test_a_map_covering_the_slab_narrows_nothing_and_says_nothing(self, biased):
        carriers = biased.charge.result.points[0].carriers

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            extent = biased.optical.strip_extent(carriers)

        assert extent == pytest.approx(biased.layout.doped_span)


class TestUnperturbedIndex:
    """What the Strips are before the carriers move them."""

    def test_it_is_the_drawn_junction_material_not_a_textbook_value(self, study):
        """The continuous route perturbs this index; so must the strips."""
        from gsim.modulator.staircase import DEFAULT_SI_INDEX

        index = study.optical.unperturbed_index()

        # The demo draws its doped silicon at eps = 11.9, which is not the
        # database's silicon at 1.55 um.
        assert index == pytest.approx(11.9**0.5)
        assert index != pytest.approx(DEFAULT_SI_INDEX)

    def test_a_chosen_index_is_taken_as_given(self, study):
        study.optical(strip_index=3.5)

        assert study.optical.unperturbed_index() == 3.5


class TestSurroundings:
    """The drawn device, redrawn around the Strips."""

    def test_the_drawn_slab_reaches_the_staircase(self, biased):
        names = {region.name for region in biased.optical.surroundings()}

        assert any(name.startswith("slab90") for name in names)

    def test_the_drawn_metal_reaches_it_as_a_conductor(self, biased):
        """The electrodes are what omitting cost 0.18 on this device."""
        regions = {region.name: region for region in biased.optical.surroundings()}

        assert regions["cathode_metal"].layer_type == "conductor"
        assert regions["anode_metal"].layer_type == "conductor"

    def test_the_doped_silicon_the_strips_replace_does_not(self, biased):
        names = {region.name for region in biased.optical.surroundings()}

        assert not names & {"n_pad", "n_rib", "p_rib", "p_pad"}

    def test_strips_on_the_rib_leave_the_pads_as_drawn_silicon(self, biased):
        """Narrow the strips and the pads come back, unperturbed."""
        biased.optical(strip_span=(CENTER_Y - HALF_WIDTH, CENTER_Y + HALF_WIDTH))

        names = {region.name for region in biased.optical.surroundings()}

        assert "n_pad" in names
        assert "p_pad" in names


class TestConductorClearance:
    """What the Palace Route cannot mesh, refused rather than crashed.

    A drawn conductor is meshed as an outline with its interior left out
    of the domain (ADR 0003). When the Window cuts one, that outline runs
    along the Window's own wall and the Palace binary aborts with no
    message at all — deterministically, on this geometry. femwell meshes
    it, so the refusal is the Palace route's and not the Stage's.
    """

    WINDOW = (CENTER_Y - 2.0, CENTER_Y + 2.0)
    WINDOW_Z = (-1.0, 1.0)

    @staticmethod
    def _metal(h, z):
        from gsim.modulator.staircase import SurroundingRegion

        return SurroundingRegion(
            name="pad_metal", h=h, z=z, material="aluminum", layer_type="conductor"
        )

    def _clear(self, surroundings, window=WINDOW, window_z=WINDOW_Z):
        from gsim.modulator.palace_route import conductor_clearance

        conductor_clearance(
            surroundings, window=window, window_z=window_z, stage_name="optical"
        )

    def test_a_conductor_inside_the_window_is_fine(self):
        self._clear([self._metal((-20.6, -20.3), (0.22, 0.72))])

    def test_a_conductor_outside_it_is_fine_too(self):
        self._clear([self._metal((-20.6, -20.3), (1.1, 1.8))])

    def test_a_conductor_the_vertical_window_cuts_is_refused(self):
        with pytest.raises(ValueError, match=r"window_z.*cuts through|pad_metal"):
            self._clear([self._metal((-20.6, -20.3), (0.5, 1.5))])

    def test_a_conductor_the_in_plane_window_cuts_is_refused(self):
        with pytest.raises(ValueError, match="pad_metal"):
            self._clear(
                [self._metal((-21.0, -20.3), (0.22, 0.72))],
                window=(CENTER_Y - 0.5, CENTER_Y + 0.5),
            )

    def test_a_dielectric_the_window_cuts_is_not_its_business(self):
        from gsim.modulator.staircase import SurroundingRegion

        self._clear(
            [
                SurroundingRegion(
                    name="slab", h=(-30.0, -10.0), z=(0.0, 0.09), material="si"
                )
            ]
        )

    def test_the_palace_adapter_refuses_through_its_staircase_check(self):
        from types import SimpleNamespace

        from gsim.modulator.palace_route import PalaceRoute

        with pytest.raises(ValueError, match="pad_metal"):
            PalaceRoute().check_staircase(
                SimpleNamespace(surroundings=[self._metal((-20.6, -20.3), (0.5, 1.5))]),
                window=self.WINDOW,
                window_z=self.WINDOW_Z,
                stage_name="optical",
            )

    def test_the_stage_hands_every_staircase_to_the_adapter(self, biased, fake_route):
        """With its Window, before meshing it: the check is the route's."""
        biased.optical(route="palace", n_strips=2, n_guess=2.9)

        biased.optical.run()

        checks = fake_route.made("check_staircase")
        assert len(checks) == len(biased.charge.result.points)
        assert checks[0]["window"] == pytest.approx(biased.optical.mode_window())
        assert checks[0]["window_z"] == pytest.approx(biased.optical.mode_window_z())
        assert "cathode_metal" in {
            region.name for region in checks[0]["staircase"].surroundings
        }
        assert checks[0]["stage_name"] == "optical"


class TestBoundaryCondition:
    """Both Routes have to read the drawn metal the same way."""

    def test_the_domain_boundary_is_metallic_by_default(self, study):
        assert study.optical.metallic_boundaries is True
        assert study.optical.simulation().metallic_boundaries is True

    def test_turning_it_off_reaches_the_simulation(self, study):
        study.optical(metallic_boundaries=False)

        assert study.optical.simulation().metallic_boundaries is False


class TestStaircaseWavelength:
    """The Staircase and the continuous profile solve the one problem.

    Both paths turn the same carriers into a complex permittivity, and
    the only wavelength either may use for that is the one the Stage is
    solving at. The plasma-dispersion model's own wavelength is where its
    coefficients were fitted; using it instead inflates every Strip's
    free-carrier absorption whenever the Stage solves somewhere else.
    """

    @staticmethod
    def _strips(study, *, wavelength_um: float):
        """The Strip response of a single-Bias Staircase at a wavelength."""
        import numpy as np

        from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

        h = np.linspace(JUNCTION_Y - HALF_WIDTH, JUNCTION_Y + HALF_WIDTH, 41)
        z = np.linspace(0.0, RIB_HEIGHT, 5)
        hh, zz = (a.ravel() for a in np.meshgrid(h, z))
        n_side = hh < JUNCTION_Y
        study.charge.seed(
            BiasSweepResult(
                contact="cathode",
                points=[
                    BiasPoint(
                        bias_v=0.0,
                        carriers=CarrierMap(
                            x_um=hh,
                            y_um=zz,
                            region=["n_rib" if side else "p_rib" for side in n_side],
                            electrons_cm3=np.where(n_side, 1e18, 1e10),
                            holes_cm3=np.where(n_side, 1e10, 1e18),
                        ),
                    )
                ],
            )
        )
        study.optical(route="femwell", n_strips=4, wavelength_um=wavelength_um)
        return study.optical.staircase(study.carriers.run().points[0]).strips

    def test_the_strips_take_the_extinction_of_the_solve_wavelength(self, study):
        """What the Staircase carries is what the continuous path builds."""
        from gsim.common.carriers import permittivity_perturbation

        wavelength_um = 1.31
        assert study.carriers.dispersion.wavelength_um != wavelength_um
        strips = self._strips(study, wavelength_um=wavelength_um)

        for i, eps in enumerate(strips.permittivity):
            expected = permittivity_perturbation(
                n0=study.optical.unperturbed_index(),
                dn=float(strips.index_shift[i]),
                dalpha_cm=float(strips.absorption_cm[i]),
                wavelength_um=wavelength_um,
            )
            assert eps == pytest.approx(expected)

    def test_moving_the_solve_wavelength_moves_the_strip_loss(self, study):
        """And it is the solve wavelength that moves it, not the fit."""
        at_fit = self._strips(study, wavelength_um=1.55).permittivity
        at_solve = self._strips(study, wavelength_um=1.31).permittivity

        for fitted, solved in zip(at_fit, at_solve, strict=True):
            assert solved.imag == pytest.approx(fitted.imag * 1.31 / 1.55)


def _wavelength_um(freq_hz: float) -> float:
    from scipy.constants import speed_of_light as c0

    return c0 / freq_hz * 1e6


def _dispersive_modes(*, index=2.4, slope=-0.9, cubic=0.0, around=1.55):
    """A Route script whose Mode index moves with the wavelength solved at."""

    def modes(freq_hz: float) -> list[complex]:
        offset = _wavelength_um(freq_hz) - around
        return [index + slope * offset + cubic * offset**3 - 1e-5j]

    return modes


def _solved_wavelengths(fake_route) -> list[float]:
    return [_wavelength_um(call["freq_hz"]) for call in fake_route.made("solve")]


@pytest.fixture
def unmeshed(monkeypatch):
    """Skip the meshing: the fake Route reads no mesh, and gmsh is the cost."""
    monkeypatch.setattr("gsim.palace.BoundaryModeSim.mesh", lambda _sim, **_kw: None)


@pytest.mark.usefixtures("unmeshed")
@pytest.mark.filterwarnings("ignore:.*no material dispersion")
class TestGroupIndex:
    """The group index the Velocity mismatch is measured against.

    Solver-free: the fake Route answers an index that moves with the
    wavelength it is asked at, so what is under test is what the Stage
    asks for — which Bias point, which wavelengths — and what it does
    with the answers.
    """

    @pytest.fixture
    def staircased(self, biased, fake_route):
        """A Study whose optical Stage solves through the fake Route."""
        fake_route.modes_at = _dispersive_modes()
        biased.optical(route="palace", n_strips=2, n_guess=2.9)
        return biased

    def test_it_is_the_index_less_the_wavelength_times_its_slope(self, staircased):
        assert staircased.optical.group_index() == pytest.approx(2.4 + 0.9 * 1.55)

    def test_it_costs_two_solves_either_side_of_the_wavelength(
        self, staircased, fake_route
    ):
        staircased.optical.run()
        assert _solved_wavelengths(fake_route) == pytest.approx([1.55, 1.55])

        staircased.optical.group_index()

        assert _solved_wavelengths(fake_route)[2:] == pytest.approx([1.54, 1.56])

    def test_the_extra_modes_are_solved_at_the_reference_bias(self, biased, fake_route):
        """The sweep need not start at the bias the shift is measured from."""
        biased.charge.seed(
            BiasSweepResult(
                contact="cathode",
                points=[
                    BiasPoint(bias_v=v, carriers=carriers_at(v)) for v in (2.0, 0.0)
                ],
            )
        )
        fake_route.modes_at = _dispersive_modes()
        biased.optical(route="palace", n_strips=2, n_guess=2.9)

        biased.optical.group_index()

        record = biased.optical.result.group_index
        assert record.bias_v == 0.0 == biased.optical.result.reference_bias_v
        electrons = [
            call["staircase"].strips.electrons_cm3
            for call in fake_route.made("check_staircase")
        ]
        assert not np.array_equal(electrons[0], electrons[1])
        assert np.array_equal(electrons[2], electrons[1])
        assert np.array_equal(electrons[3], electrons[1])

    def test_asking_twice_solves_once(self, staircased, fake_route):
        first = staircased.optical.group_index()

        assert staircased.optical.group_index() == first
        assert len(fake_route.made("solve")) == 4

    def test_it_is_kept_with_the_sweep_it_was_read_off(self, staircased):
        n_group = staircased.optical.group_index()

        record = staircased.optical.result.group_index
        assert record.n_group == n_group
        assert record.wavelength_um == 1.55
        assert record.wavelengths_um == pytest.approx((1.54, 1.56))
        assert record.n_eff == pytest.approx((2.4 + 0.009, 2.4 - 0.009))

    def test_the_sweep_itself_is_not_solved_again(self, staircased):
        sweep = staircased.optical.run()

        staircased.optical.group_index()

        assert staircased.optical.result is sweep

    def test_re_configuring_the_stage_drops_it_with_the_sweep(
        self, staircased, fake_route
    ):
        staircased.optical.group_index()

        staircased.optical(wavelength_um=1.31)

        assert staircased.optical.has_run is False
        staircased.optical.group_index()
        assert _solved_wavelengths(fake_route)[4:] == pytest.approx(
            [1.31, 1.31, 1.30, 1.32]
        )

    def test_a_change_upstream_drops_it_too(self, staircased, fake_route):
        staircased.optical.group_index()

        staircased.carriers.invalidate()
        staircased.optical.group_index()

        assert len(fake_route.made("solve")) == 8

    def test_a_forced_run_drops_it_with_the_sweep_it_replaces(self, staircased):
        staircased.optical.group_index()

        assert staircased.optical.run(force=True).group_index is None

    def test_the_extra_meshes_land_beside_the_sweep_not_on_it(
        self, staircased, fake_route
    ):
        staircased.optical.group_index()

        directories = {call["sim"].output_dir for call in fake_route.made("solve")}
        assert len(directories) == 4


@pytest.mark.usefixtures("unmeshed")
@pytest.mark.filterwarnings("ignore:.*no material dispersion")
class TestGroupIndexStep:
    """The finite-difference step is a setting, and a forgiving one."""

    def test_the_default_is_ten_nanometres(self, study):
        assert study.optical.group_index_step_um == 0.01

    def test_a_step_reaching_past_zero_wavelength_is_rejected(self, study):
        with pytest.raises(ValueError, match="group_index_step_um"):
            study.optical(group_index_step_um=1.55)

    def test_the_step_sets_the_wavelengths_solved(self, biased, fake_route):
        fake_route.modes_at = _dispersive_modes()
        biased.optical(
            route="palace", n_strips=2, n_guess=2.9, group_index_step_um=0.02
        )

        biased.optical.group_index()

        assert biased.optical.result.group_index.wavelengths_um == pytest.approx(
            (1.53, 1.57)
        )

    def test_halving_it_does_not_move_the_answer(self, biased, fake_route):
        """A central difference: the error is the index's third derivative."""
        # Far more of one than a silicon guide has around 1.55 um.
        fake_route.modes_at = _dispersive_modes(cubic=-2.0)
        biased.optical(route="palace", n_strips=2, n_guess=2.9)
        full = biased.optical.group_index()

        biased.optical(group_index_step_um=0.005)
        halved = biased.optical.group_index()

        assert full == pytest.approx(2.4 + 0.9 * 1.55, abs=1e-3)
        assert halved == pytest.approx(full, abs=1e-3)


@pytest.mark.usefixtures("unmeshed")
class TestMaterialDispersion:
    """Every material is resolved again at each wavelength solved."""

    @pytest.fixture
    def dispersive(self, phase_shifter, device, tmp_path, fake_route):
        """The phase shifter, its doped silicon given silicon's dispersion."""
        from gsim.common.stack.materials import MATERIALS_DB, MaterialProperties
        from gsim.modulator import Study

        component, stack = phase_shifter
        stack = stack.model_copy(deep=True)
        sellmeier = next(
            model
            for model in MATERIALS_DB["silicon"].dispersion_models
            if model.type == "sellmeier"
        )
        for name in device.doped_regions:
            stack.materials[name] = MaterialProperties(
                dispersion_models=[sellmeier]
            ).to_dict()
        study = Study(
            component=component,
            stack=stack,
            device=device,
            plane="x=0",
            output_dir=tmp_path / "dispersive",
        )
        study.charge.seed(
            BiasSweepResult(
                contact="cathode",
                points=[
                    BiasPoint(bias_v=v, carriers=carriers_at(v)) for v in (0.0, 2.0)
                ],
            )
        )
        fake_route.modes_at = _dispersive_modes()
        study.optical(route="palace", n_strips=2, n_guess=2.9)
        return study

    def test_the_strips_start_from_the_core_index_of_each_wavelength(
        self, dispersive, fake_route
    ):
        dispersive.optical.group_index()

        record = dispersive.optical.result.group_index
        # Silicon's index falls with wavelength around 1.55 um.
        assert record.core_index[0] > dispersive.optical.unperturbed_index()
        assert dispersive.optical.unperturbed_index() > record.core_index[1]
        staircases = [
            call["staircase"] for call in fake_route.made("check_staircase")[2:]
        ]
        for staircase, index in zip(staircases, record.core_index, strict=True):
            assert np.sqrt(np.max(staircase.strips.permittivity.real)) == (
                pytest.approx(index, abs=1e-3)
            )

    def test_a_dispersive_core_warns_about_nothing(self, dispersive):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dispersive.optical.group_index()

        assert dispersive.optical.result.group_index.material_dispersion is True

    def test_a_core_that_does_not_move_with_wavelength_is_named(
        self, biased, fake_route
    ):
        """The demo draws its silicon at a constant eps = 11.9."""
        fake_route.modes_at = _dispersive_modes()
        biased.optical(route="palace", n_strips=2, n_guess=2.9)

        with pytest.warns(UserWarning, match="no material dispersion"):
            biased.optical.group_index()

        assert biased.optical.result.group_index.material_dispersion is False

    def test_a_chosen_strip_index_is_a_core_that_does_not_move(self, dispersive):
        dispersive.optical(strip_index=3.5)

        with pytest.warns(UserWarning, match="strip_index"):
            dispersive.optical.group_index()
