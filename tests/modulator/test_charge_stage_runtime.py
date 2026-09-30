"""The charge Stage on the real pipeline: meshing, and the DEVSIM solve.

The mesh test needs gmsh only; the solve test is gated on DEVSIM
(deselected by default, run with ``pytest -m tcad_local``).
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.modulator import Device, Study, pn_phase_shifter, rib_phase_shifter
from gsim.tcad.results import BiasSweepResult
from tests._helpers import skip_without_devsim

from .conftest import build_demo


@pytest.fixture(scope="module")
def meshed(tmp_path_factory):
    """The charge Stage's simulation, actually meshed."""
    demo = build_demo()
    component, stack = demo.component, demo.stack
    study = Study(
        component=component,
        stack=stack,
        device=Device(
            p_regions=["p_rib", "p_pad"],
            n_regions=["n_rib", "n_pad"],
        ),
        output_dir=tmp_path_factory.mktemp("modulator-charge"),
    )
    sim = study.charge.simulation()
    sim.mesh(**study.charge.mesh)
    return study, sim


class TestDerivedNamesReachTheMesh:
    def test_every_derived_contact_and_interface_is_tagged(self, meshed):
        """Each under its own kind of line group, so nothing is told apart later."""
        study, sim = meshed
        contact_lines = set(sim.mesh_groups["contact_lines"])
        interface_lines = set(sim.mesh_groups["interface_lines"])
        assert {c.name for c in study.layout.contacts} == contact_lines
        assert {i.name for i in study.layout.interfaces} == interface_lines

    def test_every_doped_region_is_a_mesh_volume(self, meshed):
        study, sim = meshed
        volumes = set(sim.mesh_groups["volumes"])
        assert set(study.device.doped_regions) <= volumes

    def test_the_mesh_is_clipped_to_the_derived_window(self, meshed):
        import meshio

        study, sim = meshed
        points = np.asarray(meshio.read(str(sim.mesh_path)).points)
        low, high = study.layout.window
        # The airbox margin extends the domain past the window on both
        # sides, but the meshed slab is far smaller than the component.
        assert points[:, 0].min() > low - 5.0
        assert points[:, 0].max() < high + 5.0


@pytest.mark.tcad_local
class TestSolve:
    def test_the_study_solves_the_bias_sweep(self, tmp_path):
        skip_without_devsim()
        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(
                p_regions=["p_rib", "p_pad"],
                n_regions=["n_rib", "n_pad"],
            ),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0, 0.5])
        sweep = study.charge.run()

        assert isinstance(sweep, BiasSweepResult)
        assert sweep.contact == "cathode"
        assert len(sweep.points) == 2
        assert np.all(sweep.points[0].carriers.electrons_cm3 > 0.0)
        # Reverse bias widens the depletion region: capacitance falls.
        assert sweep.capacitance_f_per_cm[1] < sweep.capacitance_f_per_cm[0]
        assert study.charge.run() is sweep

    def test_re_running_after_a_change_solves_again(self, tmp_path):
        """DEVSIM's global device/mesh/circuit namespace survives a re-run."""
        skip_without_devsim()
        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(
                p_regions=["p_rib", "p_pad"],
                n_regions=["n_rib", "n_pad"],
            ),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0])
        first = study.charge.run()

        study.charge(biases=[0.5])
        second = study.charge.run()

        assert second is not first
        assert [point.bias_v for point in second.points] == [0.5]


@pytest.mark.tcad_local
class TestJunctionBranch:
    def test_the_sweep_carries_a_fittable_junction_branch(self, tmp_path):
        """Ticket: the shunt branch per unit length, sane on the demo device.

        C_j lands in the fF/um decade range (1e-10..1e-8 F/m), R_s in the
        ohm*mm range (1e-5..1e-1 ohm*m), and reverse bias widens the
        depletion region so C_j falls.
        """
        skip_without_devsim()
        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(
                p_regions=["p_rib", "p_pad"],
                n_regions=["n_rib", "n_pad"],
            ),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0, 1.0, 2.0])
        sweep = study.charge.run()

        for point in sweep.points:
            assert point.admittance_freq_hz > 0.0
            assert point.admittance_s_per_cm.imag > 0.0

        r_s, c_j = sweep.junction_branch()

        assert np.all(c_j > 1e-11)
        assert np.all(c_j < 1e-7)
        assert np.all(r_s > 0.0)
        assert np.all(r_s < 1e0)
        # The fit's capacitance agrees with the existing |Im(I)|/omega
        # extraction at the quasi-static frequency, where R_s barely bites.
        assert c_j == pytest.approx(sweep.capacitance_f_per_m, rel=0.05)
        # Reverse bias (positive on the cathode) depletes the junction.
        assert np.all(np.diff(c_j) < 0.0)


@pytest.mark.tcad_local
class TestAGradedJunction:
    def test_grading_lowers_the_capacitance_at_every_bias_point(self, tmp_path):
        """Ticket: compensation thins the doping either side of the Junction,
        so the depletion region is wider and its capacitance lower, and
        reverse bias still widens it."""
        skip_without_devsim()
        biases = [0.0, 1.0, 2.0]
        sweeps = {}
        for label, straggle_um in (("abrupt", 0.0), ("graded", 0.03)):
            shifter = rib_phase_shifter(lateral_straggle_um=straggle_um)
            study = pn_phase_shifter(
                component=shifter.component,
                stack=shifter.stack,
                device=shifter.device,
                electrodes=shifter.electrodes,
                biases=biases,
                output_dir=tmp_path / label,
            )
            sweeps[label] = study.charge.run()

        abrupt = sweeps["abrupt"].capacitance_f_per_m
        graded = sweeps["graded"].capacitance_f_per_m
        assert np.all(graded > 0.0)
        assert np.all(graded < abrupt)
        assert np.all(np.diff(graded) < 0.0)


@pytest.mark.tcad_local
class TestTheCapacitanceConvergesWithTheMesh:
    """Modulator-realism ticket 06: the mesh resolves the depletion region."""

    @staticmethod
    def _capacitance_pf_per_m(output_dir, **mesh) -> float:
        shifter = rib_phase_shifter()
        study = pn_phase_shifter(
            component=shifter.component,
            stack=shifter.stack,
            device=shifter.device,
            electrodes=shifter.electrodes,
            biases=[0.0],
            output_dir=output_dir,
        )
        study.charge(mesh=study.charge.mesh | mesh)
        return float(study.charge.run().capacitance_f_per_m[0]) * 1e12

    def test_halving_the_mesh_size_moves_the_capacitance_by_under_two_percent(
        self, tmp_path
    ):
        """With the oxide in the solve, where the node-built Interfaces are.

        An insulator Interface leaves out the nodes that already carry a
        Contact or an Interface, an error that scales with the element
        size: measured, 417.1 against 415.5 pF/m at 10 and 5 nm.
        """
        skip_without_devsim()
        default = self._capacitance_pf_per_m(tmp_path / "default")
        halved = self._capacitance_pf_per_m(tmp_path / "halved", refined_mesh_size=0.01)

        assert default == pytest.approx(halved, rel=0.02)

    def test_refining_the_lines_alone_does_not_get_there(self, tmp_path):
        """What the box is for: 452.6 pF/m without it against 417.1 with."""
        skip_without_devsim()
        boxed = self._capacitance_pf_per_m(tmp_path / "boxed")
        lines_only = self._capacitance_pf_per_m(tmp_path / "lines", refinement_boxes=[])

        assert lines_only > 1.05 * boxed


@pytest.mark.tcad_local
class TestJunctionModelExport:
    def test_the_demo_sweep_round_trips_through_the_model_file(self, tmp_path):
        """Ticket: the exported file reconstructs the sweep's fit exactly."""
        skip_without_devsim()
        from gsim.common.circuit import read_junction_model

        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(
                p_regions=["p_rib", "p_pad"],
                n_regions=["n_rib", "n_pad"],
            ),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0, 1.0, 2.0])

        path = study.charge.export_junction_model()
        model = read_junction_model(path)

        sweep = study.charge.run()
        r_s, c_j = sweep.junction_branch()
        assert path == study.stage_dir("charge") / "junction.json"
        assert model.contact == "cathode"
        assert model.freq_hz == sweep.points[0].admittance_freq_hz
        assert model.bias_v.tolist() == sweep.voltages.tolist()
        assert model.r_s_ohm_m.tolist() == list(r_s)
        assert model.c_j_f_per_m.tolist() == list(c_j)


@pytest.fixture(scope="module")
def sweeps(tmp_path_factory):
    """The demo device's Bias sweep, silicon alone and with its oxide."""
    skip_without_devsim()
    solved = {}
    for oxide in (False, True):
        demo = build_demo()
        study = Study(
            component=demo.component,
            stack=demo.stack,
            device=demo.device,
            output_dir=tmp_path_factory.mktemp(f"modulator-oxide-{oxide}"),
        )
        study.charge(biases=[0.0, 0.5, 1.0, 2.0], oxide=oxide)
        solved[oxide] = study.charge.run()
    return solved


@pytest.mark.tcad_local
class TestTheOxideAroundTheJunction:
    """Ticket 05: Poisson is solved in the oxide too, carriers stay in silicon."""

    def test_the_fringing_field_adds_capacitance_at_every_bias(self, sweeps):
        gain_pf_per_m = (
            sweeps[True].capacitance_f_per_cm - sweeps[False].capacitance_f_per_cm
        ) * 1e14
        # A path in parallel with the depleted silicon, and nearly blind
        # to the depletion width: much the same gain across the sweep.
        assert np.all(gain_pf_per_m > 60.0)
        assert np.all(gain_pf_per_m < 130.0)

    def test_the_junction_branch_carries_it(self, sweeps):
        with_oxide = np.asarray(sweeps[True].junction_branch().c_j_f_per_m)
        silicon_only = np.asarray(sweeps[False].junction_branch().c_j_f_per_m)
        assert np.all(with_oxide > silicon_only)

    def test_the_device_is_still_at_equilibrium_at_zero_bias(self, sweeps):
        """A node tied into two Interfaces drives a current at 0 V; none is."""
        carriers = sweeps[True].points[0].carriers
        product = carriers.electrons_cm3 * carriers.holes_cm3
        assert product.max() == pytest.approx(product.min(), rel=1e-6)
        assert abs(sweeps[True].points[0].currents_a_per_cm["cathode"]) < 1e-20

    def test_the_carrier_map_holds_the_doped_regions_only(self, sweeps):
        for with_oxide, silicon_only in zip(
            sweeps[True].points, sweeps[False].points, strict=True
        ):
            assert set(with_oxide.carriers.region) == {
                "n_pad",
                "n_rib",
                "p_rib",
                "p_pad",
            }
            assert with_oxide.carriers.x_um.size == silicon_only.carriers.x_um.size

    def test_the_depletion_formula_is_the_silicon_path_alone(self, sweeps):
        """The analytic comparison, its tolerance revisited for the oxide.

        The depletion approximation is a parallel plate through the
        depleted silicon. The silicon-only solve is that plate, and sits
        within 20 % of it (worst at 0 V, where the depletion edge is
        softest). With the oxide the solve counts a second, fringing path
        the formula does not have — about 90 pF/m beside a plate of
        260 to 470 pF/m — so it stands 30 to 40 % above the formula, by
        design rather than by error: the bound is 45 %, and the silicon
        path inside it is what the formula still checks.
        """
        from gsim.common.stack.pn_junction import PNJunctionConfig
        from gsim.tcad.validation import compare_capacitance

        junction = PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18)
        silicon_only = compare_capacitance(junction, sweeps[False], height_um=0.22)
        with_oxide = compare_capacitance(junction, sweeps[True], height_um=0.22)

        assert silicon_only.within(0.20)
        assert with_oxide.within(0.45)
        assert not with_oxide.within(0.25)
        assert np.all(with_oxide.c_tcad_f_per_cm > with_oxide.c_analytic_f_per_cm)
