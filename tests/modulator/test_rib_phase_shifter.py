"""The realistic rib Phase shifter, and the Staircases that follow it.

A rib on a thinner slab, a lightly doped core, moderately doped plus
Regions and heavily doped contact Regions under the metal. Nothing here
solves: the charge Stage is seeded with a synthetic Carrier map, so what
is under test is what the builder draws and describes, and that the RF
Staircase draws each Strip at the height of the silicon it stands for.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gsim.common.carriers import MobilityModel
from gsim.modulator import pn_phase_shifter, rib_phase_shifter
from gsim.tcad.doping import (
    StepDoping,
    acceptor_donor_concentrations,
    net_doping_cm3,
)
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

RIB_HEIGHT = 0.22
SLAB_HEIGHT = 0.09
RIB_REGIONS = {"n_rib", "p_rib"}
STRAGGLE = 0.05


@pytest.fixture(scope="module")
def device():
    """The rib Phase shifter at its published defaults, drawn once."""
    return rib_phase_shifter()


@pytest.fixture
def study(device, tmp_path):
    """The preset over the rib device, with its own electrodes."""
    return pn_phase_shifter(
        component=device.component,
        stack=device.stack,
        device=device.device,
        electrodes=device.electrodes,
        output_dir=tmp_path,
    )


def depleted_map(study, bias_v: float = 2.0) -> CarrierMap:
    """A Carrier map across the doped slab: doped, depleted at the Junction."""
    low, high = study.layout.doped_span
    centre = study.layout.junction_position
    y = np.linspace(low, high, 801)
    z = np.linspace(0.0, RIB_HEIGHT, 12)
    yy, zz = (a.ravel() for a in np.meshgrid(y, z, indexing="ij"))
    depleted = np.abs(yy - centre) < 0.07 * np.sqrt(1.0 + bias_v)
    n_side = yy < centre
    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in n_side],
        electrons_cm3=np.where(n_side & ~depleted, 3e17, 1e5),
        holes_cm3=np.where(~n_side & ~depleted, 5e17, 1e5),
    )


@pytest.fixture
def seeded(study):
    """The Study with a depleted 2 V Bias point in its charge Stage."""
    study.charge.seed(
        BiasSweepResult(
            contact="cathode",
            points=[BiasPoint(bias_v=2.0, carriers=depleted_map(study))],
        )
    )
    return study


class TestTheDrawnDevice:
    def test_the_rib_stands_taller_than_the_slab_beside_it(self, study):
        spans = study.layout.region_spans
        for name in study.device.doped_regions:
            expected = RIB_HEIGHT if name in RIB_REGIONS else SLAB_HEIGHT
            assert spans[name].z == pytest.approx((0.0, expected)), name

    def test_the_junction_is_at_the_centre_of_the_rib(self, study, device):
        assert set(study.layout.junction.regions) == RIB_REGIONS
        assert study.layout.junction_position == pytest.approx(device.center_um)

    def test_the_metal_lands_on_the_heavily_doped_regions(self, study):
        landed = {contact.name: contact.region for contact in study.layout.contacts}
        assert landed == {"cathode": "n_contact", "anode": "p_contact"}

    def test_each_region_is_doped_at_its_own_level(self, study, device):
        profiles = {p.region: p for p in study.charge.simulation().doping}

        assert profiles["p_rib"].concentration_cm3 == pytest.approx(5e17)
        assert profiles["n_rib"].concentration_cm3 == pytest.approx(3e17)
        assert profiles["p_contact"].concentration_cm3 == pytest.approx(1e20)
        assert profiles["n_contact"].dopant_type == "donor"
        assert device.doping_cm3 == {
            name: profile.concentration_cm3 for name, profile in profiles.items()
        }

    def test_the_rf_line_takes_the_devices_electrodes(self, study, device):
        assert study.rf.electrodes == device.electrodes


def region_spans(shifter) -> dict[str, tuple[float, float]]:
    """Extent of each doped Region on the Cross-section the Study derives."""
    layout = pn_phase_shifter(
        component=shifter.component,
        stack=shifter.stack,
        device=shifter.device,
        electrodes=shifter.electrodes,
    ).layout
    return {name: layout.region_spans[name].h for name in layout.doped_regions}


def profiles_of(shifter, region: str) -> list:
    """The doping profiles the device description gives one Region."""
    return [p for p in shifter.device.doping or [] if p.region == region]


def doping_along_the_junction_axis(shifter, x_um):
    """Acceptors and donors (cm^-3) along the junction axis, Region by Region.

    A profile reaches the solve on the nodes of the Region it names, so
    each position is read off the profiles of the Region drawn there.
    """
    x = np.asarray(x_um, dtype=np.float64)
    acceptors = np.zeros_like(x)
    donors = np.zeros_like(x)
    for name, (low, high) in region_spans(shifter).items():
        inside = (x >= low) & (x <= high)
        a, d = acceptor_donor_concentrations(profiles_of(shifter, name), x[inside], 0.0)
        acceptors[inside], donors[inside] = a, d
    return acceptors, donors


class TestAGradedJunction:
    def test_zero_straggle_is_the_step_doping_exactly(self, device):
        abrupt = rib_phase_shifter(lateral_straggle_um=0.0)

        assert abrupt.device == device.device
        assert abrupt.device.doping == [
            StepDoping(
                region=name,
                dopant_type="donor" if name.startswith("n_") else "acceptor",
                concentration_cm3=level,
            )
            for name, level in device.doping_cm3.items()
        ]

    def test_a_straggle_hands_the_study_graded_profiles(self, tmp_path):
        graded = rib_phase_shifter(lateral_straggle_um=STRAGGLE)
        study = pn_phase_shifter(
            component=graded.component,
            stack=graded.stack,
            device=graded.device,
            electrodes=graded.electrodes,
            output_dir=tmp_path,
        )

        assert graded.lateral_straggle_um == STRAGGLE
        assert study.charge.simulation().doping == graded.device.doping
        assert graded.device.doping
        assert not any(isinstance(p, StepDoping) for p in graded.device.doping)
        # The drawn levels are still what the device says of itself.
        assert graded.doping_cm3 == rib_phase_shifter().doping_cm3

    def test_a_negative_straggle_is_refused(self):
        with pytest.raises(ValueError, match="lateral_straggle_um"):
            rib_phase_shifter(lateral_straggle_um=-0.01)

    def test_donors_and_acceptors_overlap_across_the_junction(self):
        graded = rib_phase_shifter(lateral_straggle_um=STRAGGLE)
        x = graded.center_um + np.linspace(-STRAGGLE, STRAGGLE, 41)

        acceptors, donors = doping_along_the_junction_axis(graded, x)

        assert np.all(acceptors > 1e16)
        assert np.all(donors > 1e16)
        # Each falls monotonically into the other side.
        assert np.all(np.diff(donors) < 0.0)
        assert np.all(np.diff(acceptors) > 0.0)

    def test_the_levels_are_the_drawn_ones_away_from_every_step(self):
        graded = rib_phase_shifter(lateral_straggle_um=STRAGGLE)
        for name, (low, high) in region_spans(graded).items():
            if high - low < 12 * STRAGGLE:
                continue
            net = net_doping_cm3(profiles_of(graded, name), 0.5 * (low + high), 0.0)
            sign = 1.0 if name.startswith("n_") else -1.0
            assert net == pytest.approx(sign * graded.doping_cm3[name], rel=1e-6), name

    def test_the_net_doping_changes_sign_at_the_drawn_junction(self):
        """Equal straggle either side of equally doped cores."""
        graded = rib_phase_shifter(
            lateral_straggle_um=STRAGGLE, p_core_cm3=4e17, n_core_cm3=4e17
        )
        offsets = np.array([-0.2, -1e-3, 1e-3, 0.2]) * STRAGGLE

        acceptors, donors = doping_along_the_junction_axis(
            graded, graded.center_um + offsets
        )

        np.testing.assert_array_equal(np.sign(donors - acceptors), [1, 1, -1, -1])

    def test_unequal_cores_move_the_junction_into_the_lighter_side(self):
        """The heavier side's tail wins out to where the two tails cross."""
        graded = rib_phase_shifter(lateral_straggle_um=STRAGGLE)
        x = graded.center_um + np.linspace(-2.0, 2.0, 4001) * STRAGGLE

        acceptors, donors = doping_along_the_junction_axis(graded, x)
        crossing = x[np.argmin(np.abs(donors - acceptors))]

        # p core 5e17 against n core 3e17: the sign change is on the n side,
        # within a straggle of the drawn Junction.
        assert graded.center_um - STRAGGLE < crossing < graded.center_um

    def test_the_mobility_reads_the_total_doping_where_it_compensates(self):
        """Where the dopants cancel, the impurities the carriers scatter off
        do not: the mobility reads their sum, not their difference."""
        graded = rib_phase_shifter(
            lateral_straggle_um=STRAGGLE, p_core_cm3=4e17, n_core_cm3=4e17
        )
        model = MobilityModel.masetti_silicon()
        x = graded.center_um + np.linspace(-3.0, 3.0, 121) * STRAGGLE

        acceptors, donors = doping_along_the_junction_axis(graded, x)
        total = acceptors + donors

        # Half of each core at the Junction: 4e17 in total across the zone,
        # where the net doping passes through zero.
        np.testing.assert_allclose(total, 4e17, rtol=1e-3)
        np.testing.assert_allclose(
            model.electrons_cm2(total), model.electrons_cm2(4e17), rtol=1e-3
        )
        at_the_junction = 60
        net = abs(donors[at_the_junction] - acceptors[at_the_junction])
        assert net < 1e-3 * total[at_the_junction]
        assert model.electrons_cm2(total[at_the_junction]) < 0.5 * model.electrons_cm2(
            net
        )

    def test_the_rf_conductivity_stays_finite_and_positive_there(self, tmp_path):
        graded = rib_phase_shifter(lateral_straggle_um=STRAGGLE)
        study = pn_phase_shifter(
            component=graded.component,
            stack=graded.stack,
            device=graded.device,
            electrodes=graded.electrodes,
            output_dir=tmp_path,
        )
        x = graded.center_um + np.linspace(-3.0, 3.0, 601) * STRAGGLE
        acceptors, donors = doping_along_the_junction_axis(graded, x)
        # Neutral silicon at equilibrium: the majority carriers number the
        # net doping, down to the intrinsic density where it vanishes.
        net = donors - acceptors
        intrinsic = 1e10
        majority = 0.5 * (np.abs(net) + np.sqrt(net**2 + 4.0 * intrinsic**2))
        minority = intrinsic**2 / majority
        electrons = np.where(net > 0.0, majority, minority)
        holes = np.where(net > 0.0, minority, majority)

        sigma = study.carriers.response(electrons, holes).conductivity_s_per_m

        assert np.all(np.isfinite(sigma))
        assert np.all(sigma > 0.0)
        # It dips where the dopants compensate, and only there.
        assert sigma.argmin() not in (0, sigma.size - 1)


class TestTheRFStaircaseFollowsIt:
    def test_each_strip_stands_at_the_height_of_its_region(self, seeded):
        seeded.rf(n_strips=10, strips_per_region=2)
        strips = seeded.rf.staircase().strips

        centres = 0.5 * (strips.edges_um[1:] + strips.edges_um[:-1])
        rib = seeded.layout.junction_span.h
        in_rib = (centres > rib[0]) & (centres < rib[1])
        np.testing.assert_allclose(strips.zmax_um[in_rib], RIB_HEIGHT)
        np.testing.assert_allclose(strips.zmax_um[~in_rib], SLAB_HEIGHT)
        # Ten across the rib, two across each of the six slab Regions.
        assert strips.count == 10 + 6 * 2

    def test_the_default_strips_leave_one_depleted(self, seeded):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            seeded.rf.staircase()


class TestTheOpticalStaircase:
    def test_a_staircase_drawn_at_the_rib_height_says_so(self, seeded):
        """The optical Staircase still draws every Strip at the Junction's
        height, which is the slab's only when the slab is the rib."""
        seeded.optical(route="palace", n_strips=5)
        point = seeded.carriers.run().points[0]

        with pytest.warns(UserWarning, match="rib's height"):
            seeded.optical.staircase(point)
