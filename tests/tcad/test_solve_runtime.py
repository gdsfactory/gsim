"""Runtime-gated drift-diffusion solve tests (require DEVSIM).

Deselected by default (see pyproject addopts); run with
``pytest -m tcad_local``, mirroring the ``meep_local`` convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.stack.pn_junction import PNJunctionConfig
from gsim.tcad import Simulation, StepDoping
from gsim.tcad.validation import compare_capacitance, estimate_depletion_width_um
from tests._helpers import skip_without_devsim

from .conftest import build_padded_diode

skip_without_devsim()

pytestmark = pytest.mark.tcad_local

_NA_CM3 = 1e18
_ND_CM3 = 1e18
_RIB_HEIGHT_UM = 0.22
# Positive cathode (n-side) bias reverse-biases the junction.
_REVERSE_BIASES = [0.0, 0.5, 1.0]


@pytest.fixture(scope="module")
def meshed_diode(tmp_path_factory):
    comp, stack, _names = build_padded_diode(zmax=_RIB_HEIGHT_UM)
    sim = Simulation()
    sim.geometry(comp, stack=stack)
    sim.domain(plane="x=0", margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    # Contacts on the outer pads, away from the junction; interfaces make
    # the four doped regions one continuous device.
    sim.contacts.add(name="anode", semiconductor="p_pad", electrode="sio2")
    sim.contacts.add(name="cathode", semiconductor="n_pad", electrode="sio2")
    sim.interfaces.add(name="junction", region_a="p_rib", region_b="n_rib")
    sim.interfaces.add(name="p_link", region_a="p_pad", region_b="p_rib")
    sim.interfaces.add(name="n_link", region_a="n_pad", region_b="n_rib")
    for region in ("p_rib", "p_pad"):
        sim.doping.add(
            StepDoping(region=region, dopant_type="acceptor", concentration_cm3=_NA_CM3)
        )
    for region in ("n_rib", "n_pad"):
        sim.doping.add(
            StepDoping(region=region, dopant_type="donor", concentration_cm3=_ND_CM3)
        )
    sim.solver(mesh_preset="coarse", refined_mesh_size_um=0.02, max_mesh_size_um=40.0)
    sim.mesh(output_dir=tmp_path_factory.mktemp("tcad-runtime"))
    return sim


@pytest.fixture(scope="module")
def solved_sweep(meshed_diode):
    meshed_diode.materials.temperature_k = 300.0
    return meshed_diode.sweep(_REVERSE_BIASES, contact="cathode")


class TestDriftDiffusionSolve:
    def test_carrier_maps_physical(self, solved_sweep):
        point = solved_sweep.points[0]
        assert np.all(point.carriers.electrons_cm3 > 0.0)
        assert np.all(point.carriers.holes_cm3 > 0.0)
        # Majority carriers approach the doping levels somewhere on each side.
        assert point.carriers.holes_cm3.max() == pytest.approx(_NA_CM3, rel=0.5)
        assert point.carriers.electrons_cm3.max() == pytest.approx(_ND_CM3, rel=0.5)

    def test_capacitance_positive_and_decreasing_in_reverse_bias(self, solved_sweep):
        c = solved_sweep.capacitance_f_per_cm
        assert np.all(c > 0.0)
        # Depletion widens under reverse bias: C(V) decreases.
        assert c[-1] < c[0]

    def test_reverse_leakage_is_small(self, solved_sweep):
        currents = solved_sweep.points[-1].currents_a_per_cm
        # Reverse-biased diode: terminal currents balance and are tiny.
        assert abs(currents["anode"] + currents["cathode"]) <= 1e-6 * max(
            abs(currents["anode"]), abs(currents["cathode"]), 1e-30
        )
        assert abs(currents["cathode"]) < 1e-6


class TestAnalyticCrossCheck:
    def test_capacitance_tracks_sze_model(self, solved_sweep):
        junction = PNJunctionConfig(na_cm3=_NA_CM3, nd_cm3=_ND_CM3)
        comparison = compare_capacitance(
            junction, solved_sweep, height_um=_RIB_HEIGHT_UM
        )
        # Fully depleted regime on an abrupt symmetric junction: the numeric
        # small-signal C(V) must track the depletion approximation.
        assert comparison.within(0.35)

    def test_carrier_profile_edges_match_depletion_extents(self, solved_sweep):
        junction = PNJunctionConfig(
            na_cm3=_NA_CM3, nd_cm3=_ND_CM3, v_reverse=_REVERSE_BIASES[-1]
        )
        carriers = solved_sweep.points[-1].carriers
        # 1D cut across the junction at mid rib height.
        band = np.abs(carriers.y_um - _RIB_HEIGHT_UM / 2) < 0.03
        order = np.argsort(carriers.x_um[band])
        width = estimate_depletion_width_um(
            carriers.x_um[band][order],
            carriers.electrons_cm3[band][order],
            carriers.holes_cm3[band][order],
            na_cm3=_NA_CM3,
            nd_cm3=_ND_CM3,
        )
        assert width == pytest.approx(junction.w_um, rel=0.5)


class TestParityWithChargeTransportSim:
    """The concern-grouped API solves the same device as the setter API did.

    Reference numbers from ``ChargeTransportSim`` on #298
    (``feat/tcad-twmzm``), same device, same mesh settings, same biases.
    """

    C_F_PER_CM = (5.457119288045399e-12, 4.105770096334684e-12, 3.3610442251991344e-12)
    I_CATHODE_A_PER_CM = (0.0, -4.1014068148460336e-15, -4.869532720050845e-15)
    Y_S_PER_CM = (
        7.371280401253742e-05 + 0.03428791396284675j,
        3.988263646248467e-05 + 0.025797243202074j,
        2.6081073943502278e-05 + 0.021118024977971262j,
    )
    N_NODES = 639
    ELECTRONS_SUM_CM3 = (
        2.7794164626005203e20,
        2.7416658783555882e20,
        2.714170863614466e20,
    )
    HOLES_SUM_CM3 = (
        2.7884171689711665e20,
        2.7512049656009238e20,
        2.7241141392524362e20,
    )

    def test_capacitance(self, solved_sweep):
        np.testing.assert_allclose(
            solved_sweep.capacitance_f_per_cm, self.C_F_PER_CM, rtol=1e-6
        )

    def test_terminal_currents(self, solved_sweep):
        currents = [p.currents_a_per_cm["cathode"] for p in solved_sweep.points]
        np.testing.assert_allclose(
            currents, self.I_CATHODE_A_PER_CM, rtol=1e-3, atol=1e-20
        )

    def test_small_signal_admittance(self, solved_sweep):
        admittance = [p.admittance_s_per_cm for p in solved_sweep.points]
        np.testing.assert_allclose(admittance, self.Y_S_PER_CM, rtol=1e-6)

    def test_carrier_maps(self, solved_sweep):
        for point, n_sum, p_sum in zip(
            solved_sweep.points,
            self.ELECTRONS_SUM_CM3,
            self.HOLES_SUM_CM3,
            strict=True,
        ):
            assert point.carriers.x_um.size == self.N_NODES
            assert point.carriers.electrons_cm3.sum() == pytest.approx(n_sum, rel=1e-6)
            assert point.carriers.holes_cm3.sum() == pytest.approx(p_sum, rel=1e-6)


class TestSiliconConstants:
    def test_the_constants_are_devsims(self):
        from gsim.tcad.runtime import import_simple_physics
        from gsim.tcad.silicon import (
            INTRINSIC_DENSITY_300K_CM3,
            SILICON_RELATIVE_PERMITTIVITY,
        )

        sp = import_simple_physics()
        assert sp.eps_si == SILICON_RELATIVE_PERMITTIVITY
        assert sp.n_i == INTRINSIC_DENSITY_300K_CM3

    def test_the_built_in_voltage_falls_with_temperature(self, meshed_diode):
        from gsim.tcad.silicon import intrinsic_density_cm3

        built_in = {}
        try:
            for temperature_k in (250.0, 300.0, 350.0):
                meshed_diode.materials.temperature_k = temperature_k
                potential = meshed_diode.solve(
                    0.0, contact="cathode"
                ).carriers.potential_v
                built_in[temperature_k] = float(potential.max() - potential.min())
                expected = PNJunctionConfig(
                    na_cm3=_NA_CM3,
                    nd_cm3=_ND_CM3,
                    temperature_k=temperature_k,
                    ni_cm3=intrinsic_density_cm3(temperature_k),
                ).v_bi
                assert built_in[temperature_k] == pytest.approx(expected, rel=0.01)
        finally:
            meshed_diode.materials.temperature_k = 300.0
        assert built_in[350.0] < built_in[300.0] < built_in[250.0]


class TestBiasHistory:
    def test_a_solve_does_not_depend_on_the_contact_swept_before(self, meshed_diode):
        meshed_diode._reset_device()
        fresh = meshed_diode.solve(0.0, contact="anode").capacitance_f_per_cm
        meshed_diode.solve(1.0, contact="cathode")
        after = meshed_diode.solve(0.0, contact="anode").capacitance_f_per_cm
        assert after == pytest.approx(fresh, rel=1e-6)
