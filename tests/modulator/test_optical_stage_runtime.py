"""The optical Stage on the real pipeline: its own mesh, and a real solve.

The solve is gated on gmsh and the femwell runtime, and stands on a
synthetic Carrier map so it needs no DEVSIM: what it proves is the Stage's
own chain — derive the Window, mesh it, carry the carriers onto that mesh,
perturb the permittivity and solve. The end-to-end run off a real charge
solve is the ``tcad_local`` test at the bottom.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gsim.modulator import Device, OpticalSweep, Study
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap
from tests._helpers import longest_edge_in_box, skip_without_devsim

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, RIB_HEIGHT, build_demo

pytest.importorskip("gmsh")
pytest.importorskip("femwell")
pytest.importorskip("skfem")

DOPING_CM3 = 1e18
DEPLETED_CM3 = 1e10
SLAB = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)


def depletion_carriers(bias_v: float) -> CarrierMap:
    """A Carrier map whose depletion region widens with reverse bias.

    Not a charge solve — a monotone stand-in for one, sampled across the
    doped slab so the transfer onto the optical mesh has something to
    interpolate.
    """
    y = np.linspace(SLAB[0], SLAB[1], 121)
    z = np.linspace(0.0, RIB_HEIGHT, 9)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()

    half_width = 0.05 * np.sqrt(1.0 + abs(bias_v))
    depleted = np.abs(yy - CENTER_Y) < half_width
    n_side = yy < CENTER_Y

    electrons = np.where(n_side, DOPING_CM3, DEPLETED_CM3)
    holes = np.where(n_side, DEPLETED_CM3, DOPING_CM3)
    electrons = np.where(depleted, DEPLETED_CM3, electrons)
    holes = np.where(depleted, DEPLETED_CM3, holes)

    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["p_rib" if side else "n_rib" for side in ~n_side],
        electrons_cm3=electrons,
        holes_cm3=holes,
    )


def canned_sweep(biases) -> BiasSweepResult:
    """A Bias sweep of synthetic Carrier maps."""
    return BiasSweepResult(
        contact="cathode",
        points=[
            BiasPoint(bias_v=bias, carriers=depletion_carriers(bias)) for bias in biases
        ],
    )


def build_study(output_dir):
    """A Study over the phase shifter, writing into ``output_dir``."""
    demo = build_demo()
    component, stack = demo.component, demo.stack
    return Study(
        component=component,
        stack=stack,
        device=Device(p_regions=["p_rib", "p_pad"], n_regions=["n_rib", "n_pad"]),
        output_dir=output_dir,
    )


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """The optical Stage run across a synthetic Bias sweep."""
    study = build_study(tmp_path_factory.mktemp("modulator-optical"))
    biases = [0.0, 2.0]
    study.charge.seed(canned_sweep(biases))
    return study, study.optical.run()


class TestItsOwnMesh:
    def test_the_optical_mesh_is_not_the_charge_mesh(self, solved):
        """ADR 0002: the optical solve meshes its own Window."""
        study, _ = solved
        charge_sim = study.charge.simulation()
        charge_sim.mesh(**study.charge.mesh)

        optical_mesh = study.stage_dir("optical") / "palace.msh"

        assert optical_mesh.exists()
        assert optical_mesh != charge_sim.mesh_path

    def test_the_optical_mesh_spans_the_derived_window(self, solved):
        import meshio

        study, _ = solved
        points = np.asarray(
            meshio.read(str(study.stage_dir("optical") / "palace.msh")).points
        )
        window = study.layout.window_around_junction(
            margin_um=study.optical.mode_margin_um
        )
        window_z = study.layout.window_z_around_guide(above_um=1.0, below_um=1.0)

        assert points[:, 0].min() == pytest.approx(window[0], abs=0.05)
        assert points[:, 0].max() == pytest.approx(window[1], abs=0.05)
        assert points[:, 1].min() == pytest.approx(window_z[0], abs=0.05)
        assert points[:, 1].max() == pytest.approx(window_z[1], abs=0.05)

    def test_the_continuous_mesh_resolves_the_junction_as_the_charge_mesh_did(
        self, solved
    ):
        # The Carrier map comes off a charge mesh that resolves the
        # depletion edge. A transfer onto elements several times coarser
        # smears it: measured on a real charge solve, the index shift read
        # 15 % low at this Stage's own 0.05 um.
        study, _ = solved
        span = study.layout.junction_span
        longest = longest_edge_in_box(
            study.stage_dir("optical") / "palace.msh", span.h, span.z
        )
        [(*_extent, size)] = study.charge.junction_boxes()

        # gmsh takes a size as a target: an edge runs up to about twice it.
        assert longest < 2.5 * size


class TestSolvedModes:
    def test_the_sweep_reports_one_mode_per_bias_point(self, solved):
        _, sweep = solved

        assert isinstance(sweep, OpticalSweep)
        assert sweep.contact == "cathode"
        assert sweep.wavelength_um == 1.55
        assert sweep.voltages == pytest.approx([0.0, 2.0])

    def test_the_mode_is_guided_by_the_rib(self, solved):
        _, sweep = solved
        # Between the oxide cladding and bulk silicon.
        assert all(1.444 < n.real < 3.48 for n in sweep.n_eff)

    def test_the_index_shift_is_measured_from_zero_bias(self, solved):
        _, sweep = solved

        assert sweep.reference_bias_v == 0.0
        assert sweep.index_shift[0] == 0.0
        # Reverse bias depletes the rib: fewer carriers, less negative
        # plasma-dispersion shift, so the effective index rises.
        assert sweep.index_shift[1] > 0.0

    def test_depleting_the_rib_lowers_the_loss(self, solved):
        _, sweep = solved

        assert all(loss > 0.0 for loss in sweep.loss_db_cm)
        assert sweep.loss_db_cm[1] < sweep.loss_db_cm[0]

    def test_the_mode_is_contained_by_its_window(self, solved):
        _, sweep = solved
        assert all(p.boundary_field_ratio < 0.01 for p in sweep.points)

    def test_the_carriers_stage_ran_first(self, solved):
        study, _ = solved
        assert study.carriers.has_run is True

    def test_running_twice_solves_once(self, solved):
        study, sweep = solved
        assert study.optical.run() is sweep


class TestWindowTooSmall:
    def test_a_clipped_mode_warns_naming_the_stage(self, tmp_path):
        study = build_study(tmp_path)
        study.charge.seed(canned_sweep([0.0]))
        # A window barely wider than the rib cannot hold the mode's tails.
        study.optical(mode_margin_um=0.45, z_above_um=0.15, z_below_um=0.15)

        with pytest.warns(UserWarning, match="optical stage"):
            sweep = study.optical.run()

        assert sweep.points[0].boundary_field_ratio > 0.01


class TestRegionsOffTheWindow:
    def test_a_pad_clipped_out_of_the_window_is_not_an_error(self, tmp_path):
        """The optical Window is a box around the rib, not the doped slab."""
        study = build_study(tmp_path)
        study.charge.seed(canned_sweep([0.0]))
        # Tight enough that both contact pads fall outside the mesh.
        study.optical(mode_margin_um=0.25, z_above_um=0.15, z_below_um=0.15)

        with pytest.warns(UserWarning, match="optical stage"):
            sweep = study.optical.run()

        assert sweep.points[0].n_eff.real > 1.444

    def test_a_named_region_off_the_window_is_reported(self, tmp_path):
        study = build_study(tmp_path)
        study.charge.seed(canned_sweep([0.0]))
        study.optical(
            mode_margin_um=0.25,
            z_above_um=0.15,
            z_below_um=0.15,
            perturbed_regions=["p_rib", "p_pad"],
        )

        with pytest.raises(ValueError, match=r"p_pad"):
            study.optical.run()

    def test_a_window_holding_no_doped_region_is_reported(self, tmp_path):
        study = build_study(tmp_path)
        study.charge.seed(canned_sweep([0.0]))
        # A box in the cladding, well above the rib.
        study.optical(window=(CENTER_Y - 1.0, CENTER_Y + 1.0), window_z=(1.0, 2.0))

        with pytest.raises(ValueError, match="perturb nothing"):
            study.optical.run()


@pytest.mark.tcad_local
class TestEndToEnd:
    def test_a_real_charge_solve_reaches_the_optical_mode(self, tmp_path):
        skip_without_devsim()
        study = build_study(tmp_path)
        study.charge(biases=[0.0, 2.0])

        sweep = study.optical.run()

        assert study.charge.has_run is True
        assert len(sweep.points) == 2
        assert all(1.444 < n.real < 3.48 for n in sweep.n_eff)
        assert sweep.index_shift[1] > 0.0


class TestStaircaseWavelength:
    """The Staircase reads its loss at the wavelength being solved at.

    A plasma-dispersion model's wavelength is where its coefficients were
    fitted: it says what ``dalpha_cm`` a carrier concentration means, not
    where anyone is solving. Turning that absorption into an extinction
    coefficient — ``kappa = alpha lambda / 4 pi`` — is the step that needs
    the solve's own wavelength, and the continuous path has always used
    it. Building the Strips at the fit wavelength instead inflated every
    Strip's loss by ``1.55 / 1.31``, about 18%, at 1.31 um.
    """

    @staticmethod
    def _staircase_loss(output_dir, wavelength_um: float) -> float:
        """Modal loss (dB/cm) of the Staircase solved at one wavelength."""
        study = build_study(output_dir)
        study.charge.seed(canned_sweep([0.0]))
        study.optical(
            route="femwell",
            wavelength_um=wavelength_um,
            n_strips=8,
            num_modes=1,
        )
        # No index guess: femwell's own tracks the largest permittivity,
        # which is the guided mode at either wavelength. A fixed guess
        # picks different branches at 1.31 um and 1.55 um, and the
        # comparison would then be between two different modes.
        return float(study.optical.run().points[0].loss_db_cm)

    def test_the_staircase_loss_is_the_carriers_not_the_wavelengths(self, tmp_path):
        """Fixed coefficients, so dB/cm is a property of the carriers.

        The material absorption does not move between 1.31 um and 1.55 um
        here — the same 1.55 um fit answers both — so the modal loss may
        only move by the confinement change, which on this guide is
        about 6%: the shorter wavelength is held tighter in the rib and
        so overlaps the doped strips more. The defect moved it by the
        wavelength ratio itself, 1.55 / 1.31, and in the other direction.
        """
        dispersion_um = 1.55
        at_fit = self._staircase_loss(tmp_path / "at-fit", dispersion_um)
        away = self._staircase_loss(tmp_path / "away", 1.31)

        assert at_fit > 0.0
        assert away == pytest.approx(at_fit, rel=0.10)
        # And nowhere near the value the defect produced, which is what
        # the loose-looking tolerance still has to separate.
        assert away < at_fit * dispersion_um / 1.31 * 0.95


class TestStripsTooNarrowForTheMode:
    """Ticket 19: a Staircase whose Strips cannot hold the Mode says so.

    Only the Strips carry the carrier response, so a Mode that mostly
    lives outside them is answered by the surrounding regions — which are
    the drawn materials, unperturbed. The index shift then belongs to
    whatever fraction of the Mode the Strips do hold, and nothing in the
    result says so unless the Stage does.
    """

    def test_strips_narrower_than_the_mode_warn_naming_the_extent(self, tmp_path):
        study = build_study(tmp_path)
        study.charge.seed(canned_sweep([0.0]))
        # A tenth of the guide: the mode is overwhelmingly outside it.
        study.optical(
            route="femwell",
            n_strips=2,
            strip_span=(CENTER_Y - 0.05, CENTER_Y + 0.05),
            n_guess=2.9,
        )

        with pytest.warns(UserWarning, match="outside the strip extent"):
            study.optical.run()

    def test_the_default_extent_does_not_warn(self, tmp_path):
        study = build_study(tmp_path)
        study.charge.seed(canned_sweep([0.0]))
        study.optical(route="femwell", n_strips=4, n_guess=2.9)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            study.optical.run()

        assert not [w for w in caught if "strip extent" in str(w.message)]


def _material_index(name: str, wavelength_um: float) -> float:
    """Index of a database material at one wavelength."""
    from gsim.common.stack.materials import resolve_material_at_wavelength

    resolved = resolve_material_at_wavelength(name, wavelength_um)
    assert resolved is not None
    assert resolved.permittivity_scalar is not None
    return float(np.sqrt(resolved.permittivity_scalar))


def slab_te0_index(wavelength_um: float, *, core: float, cladding: float) -> float:
    """Effective index of the TE0 Mode of a symmetric slab, in closed form.

    The root of ``kappa d/2 tan(kappa d/2) = gamma d/2`` on the branch
    ``kappa d/2 < pi/2``, for a slab ``RIB_HEIGHT`` thick.
    """
    from scipy.optimize import brentq

    k0 = 2.0 * np.pi / wavelength_um
    half = RIB_HEIGHT / 2.0

    def mismatch(n_eff: float) -> float:
        kappa = k0 * np.sqrt(core**2 - n_eff**2)
        gamma = k0 * np.sqrt(n_eff**2 - cladding**2)
        return float(kappa * half * np.tan(kappa * half) - gamma * half)

    branch = np.sqrt(max(core**2 - (np.pi / (2.0 * half * k0)) ** 2, 0.0))
    return float(brentq(mismatch, max(cladding, branch) + 1e-9, core - 1e-9))


def slab_group_index(wavelength_um: float, *, dispersive: bool) -> float:
    """Group index of that slab, its materials dispersive or frozen."""

    def n_eff(at_um: float) -> float:
        materials_um = at_um if dispersive else wavelength_um
        return slab_te0_index(
            at_um,
            core=_material_index("silicon", materials_um),
            cladding=_material_index("SiO2", materials_um),
        )

    step = 1e-4
    slope = (n_eff(wavelength_um + step) - n_eff(wavelength_um - step)) / (2.0 * step)
    return n_eff(wavelength_um) - wavelength_um * slope


def build_slab_study(output_dir):
    """A slab guide: the doped slab, seen through a Window narrower than it.

    The four doped Regions stand at one height, so a Window inside the
    rib sees a uniform silicon layer in oxide, and its metallic side
    walls are what the TE slab Mode — uniform along them, its field
    normal to them — satisfies exactly. The doped silicon is given
    silicon's own Sellmeier model in place of the demo's constant, and
    the carriers are intrinsic, so nothing perturbs the slab. A Mode
    uniform along the side walls peaks on them, which is the one thing
    the Window-containment check is switched off for here.
    """
    from gsim.common.stack.materials import MATERIALS_DB, MaterialProperties

    demo = build_demo()
    stack = demo.stack.model_copy(deep=True)
    sellmeier = next(
        model
        for model in MATERIALS_DB["silicon"].dispersion_models
        if model.type == "sellmeier"
    )
    for name in demo.device.doped_regions:
        stack.materials[name] = MaterialProperties(
            dispersion_models=[sellmeier]
        ).to_dict()
    study = Study(
        component=demo.component,
        stack=stack,
        device=demo.device,
        output_dir=output_dir,
    )

    y = np.linspace(SLAB[0], SLAB[1], 41)
    z = np.linspace(0.0, RIB_HEIGHT, 5)
    yy, zz = (a.ravel() for a in np.meshgrid(y, z, indexing="ij"))
    intrinsic = CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in yy < CENTER_Y],
        electrons_cm3=np.full(yy.shape, DEPLETED_CM3),
        holes_cm3=np.full(yy.shape, DEPLETED_CM3),
    )
    study.charge.seed(
        BiasSweepResult(
            contact="cathode", points=[BiasPoint(bias_v=0.0, carriers=intrinsic)]
        )
    )
    study.optical(
        window=(CENTER_Y - 0.2, CENTER_Y + 0.2),
        window_z=(-1.5, RIB_HEIGHT + 1.5),
        boundary_field_tol=2.0,
    )
    return study


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    """The slab guide's group index, solved once with second-order elements."""
    study = build_slab_study(tmp_path_factory.mktemp("modulator-slab"))
    study.optical(order=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        study.optical.group_index()
    return study


class TestGroupIndexOfASlab:
    """The group index against a guide whose group index is known.

    The closed form is the symmetric slab's TE0 dispersion relation with
    the same Sellmeier materials, differentiated at a step a hundred
    times finer than the Stage's. Second-order elements land the Stage
    within 1e-3 of it — it measured 1e-5 — where first-order ones on the
    default mesh sit 0.4% low, the index itself being 0.6% low there.
    """

    TOLERANCE = 1e-3

    def test_the_mode_solved_is_the_slab_mode(self, slab):
        expected = slab_te0_index(
            1.55,
            core=_material_index("silicon", 1.55),
            cladding=_material_index("SiO2", 1.55),
        )

        assert slab.optical.result.n_eff[0].real == pytest.approx(expected, abs=1e-3)

    def test_the_group_index_lands_on_the_closed_form(self, slab):
        expected = slab_group_index(1.55, dispersive=True)

        assert slab.optical.group_index() == pytest.approx(expected, abs=self.TOLERANCE)

    def test_material_dispersion_is_in_the_answer(self, slab):
        """Freezing the materials moves the closed form by 0.13."""
        frozen = slab_group_index(1.55, dispersive=False)

        assert slab.optical.result.group_index.material_dispersion is True
        assert slab.optical.group_index() - frozen > 100 * self.TOLERANCE

    def test_halving_the_step_does_not_move_it(self, tmp_path):
        study = build_slab_study(tmp_path)
        full = study.optical.group_index()

        study.optical(group_index_step_um=study.optical.group_index_step_um / 2.0)
        halved = study.optical.group_index()

        assert halved == pytest.approx(full, abs=1e-4)

    def test_the_staircase_carries_the_same_dispersion(self, tmp_path):
        """Strips re-read the core index at each wavelength, as the mesh does."""
        continuous = build_slab_study(tmp_path / "continuous")
        staircase = build_slab_study(tmp_path / "staircase")
        staircase.optical(n_strips=2)

        assert staircase.optical.group_index() == pytest.approx(
            continuous.optical.group_index(), abs=5e-3
        )
