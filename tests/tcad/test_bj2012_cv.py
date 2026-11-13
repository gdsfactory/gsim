"""Regression test: junction C(V) of the Baehr-Jones 2012 modulator (DEVSIM).

Baehr-Jones et al., Opt. Express 20(11) 12014 (2012),
https://doi.org/10.1364/OE.20.012014. The cross-section is the generic one of
``nbs/tcad_bj2012_modulator.ipynb``: a 0.5 um wide, 220 nm tall rib on a 90 nm
slab with 0.95 um slab either side, 2.6 um contact regions and an aluminum
landing on each, and the calibrated erfc-graded junction (lateral straggle
0.20 um) of that notebook. The straggle is a calibrated unknown (the paper does not
give the grading length): the least-squares fit of C(V) to the ten points of Fig. 2d
on the notebook's mesh gives 0.204 um (see the section "How sigma was fitted" of the
details notebook and ``nbs/data/rf_benchmark/bj2012_modulator/sigma_fit.json``).
The paper's implants are striated (10 um period, 1 um gaps), so the capacitance
per unit length of device is 0.9 times that of the simulated segment; the
comparison with Fig. 2d applies that factor.

A reduced version of the notebook's sweep, to fit the budget of the
``tcad_local`` workflow (about a minute): the coarse notebook mesh, three
reverse biases (0, 1 and 3 V) and 0.5 V ramp steps. Two checks:

- the simulated C(V) against stored values of this reduced run (+-3 %), which
  catches changes of the solver or the mesher;
- against the paper's Fig. 2d (+-10 %), read by eye from the figure
  (``nbs/data/bj2012_reference.json``). The graded profile is a fit to that
  figure, so this agreement shows the fit holds, not that the physics is
  predicted.

Deselected by default; run with ``pytest -m tcad_local``.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import gdsfactory as gf
import numpy as np
import pytest
from scipy.special import erfc

from gsim.common.carriers import MobilityModel
from gsim.common.cross_section import build_doped_cross_section
from gsim.common.stack.extractor import Layer
from gsim.common.stack.materials import make_doped_materials
from gsim.common.stack.pn_junction import PNJunctionConfig
from gsim.tcad import CallableDoping, Simulation, StepDoping
from gsim.tcad.validation import analytic_capacitance_f_per_cm, compare_capacitance
from tests._helpers import draw_rectangle, skip_without_devsim

skip_without_devsim()

pytestmark = pytest.mark.tcad_local

REFERENCE = json.loads(
    (
        Path(__file__).resolve().parents[2] / "nbs" / "data" / "bj2012_reference.json"
    ).read_text(encoding="utf-8")
)
GEOMETRY = REFERENCE["geometry"]
DOPING = REFERENCE["doping"]

CENTER_Y = -20.0  # junction position along y (um)
RIB_HEIGHT = GEOMETRY["rib_height_um"]
SLAB_HEIGHT = GEOMETRY["slab_height_um"]
RIB_HALF = GEOMETRY["rib_width_um"] / 2
SLAB_WIDTH = GEOMETRY["contact_start_from_rib_edge_um"]
CONTACT_WIDTH = GEOMETRY["rib_edge_to_via_side_um"] - SLAB_WIDTH
METAL_THICKNESS = GEOMETRY["lower_metal_thickness_um"]
SIGMA_UM = 0.20  # calibrated lateral straggle of the notebook (um), fitted to Fig. 2d
#: Fraction of the device length that carries a junction (Fig. 1a inset).
FILL = (
    1 - GEOMETRY["implant_striation_gap_um"] / GEOMETRY["implant_striation_period_um"]
)
#: Slab doping (cm^-3) that gives the paper's sheet resistances in the 90 nm slab,
#: with Masetti mobility; ``test_slab_doping_gives_the_sheet_resistance`` checks it.
SLAB_P_CM3 = 2.79e17
SLAB_N_CM3 = 1.60e17

# Regions from the n-side contact to the p-side one:
# name: (dopant, width, height) in um.
REGIONS = {
    "n_contact": ("donor", CONTACT_WIDTH, SLAB_HEIGHT),
    "n_slab": ("donor", SLAB_WIDTH, SLAB_HEIGHT),
    "n_rib": ("donor", RIB_HALF, RIB_HEIGHT),
    "p_rib": ("acceptor", RIB_HALF, RIB_HEIGHT),
    "p_slab": ("acceptor", SLAB_WIDTH, SLAB_HEIGHT),
    "p_contact": ("acceptor", CONTACT_WIDTH, SLAB_HEIGHT),
}
ELECTRODES = {"n_contact": "cathode_metal", "p_contact": "anode_metal"}

#: Reverse biases (V on the cathode) of the reduced sweep, on Fig. 2d's grid.
BIASES = [0.0, 1.0, 3.0]
#: Simulated C (fF/um) of this reduced run, at sigma = 0.20 um (the refit of
#: ``nbs/data/rf_benchmark/bj2012_modulator/sigma_fit.json``). History of the stored
#: values: first 0.2253, 0.1837 and 0.1530 at sigma = 0.27 um; gsim commit 6ef78bc8
#: (#480, signed finite-box edge couples) lowered them by 5.8 to 6.4 % to 0.2123, 0.1720
#: and 0.1438 (the same case on the sources of the parent commit gives the old values
#: and with #480 the new ones, and no other commit in between moves C); the refit to
#: sigma = 0.20 um, which brings the curve back onto Fig. 2d, gives the values below.
#: DEVSIM's own |cot| couples overweight obtuse triangles (``test_edge_couples.py``
#: against an independent P1 solve); with the signed ones the 0 V capacitance settles as
#: the mesh is refined (0.2123, 0.2111, 0.2108, 0.2103 fF/um at 20, 10, 5 and 2.5 nm in
#: the rib, at sigma = 0.27 um), with the old ones it did not (0.2253 to 0.2162 and back
#: to 0.2194).
STORED_C_FF_PER_UM = {0.0: 0.2275, 1.0: 0.1846, 3.0: 0.1540}
#: 1 F/cm of depth = 1e15 fF / 1e4 um.
F_PER_CM_TO_FF_PER_UM = 1e11


def _draw_device():
    """The doped regions and the two aluminum landings: component, stack, spans."""
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    layers: dict[str, Layer] = {}
    spans: dict[str, tuple[float, float]] = {}
    y = CENTER_Y - sum(width for _, width, _ in REGIONS.values()) / 2
    for i, (name, (_, width, height)) in enumerate(REGIONS.items()):
        spans[name] = (y, y + width)
        draw_rectangle(comp, (30, i), width=10.0, height=width, x0=0.0, y=y + width / 2)
        layers[name] = Layer(
            name=name,
            gds_layer=(30, i),
            zmin=0.0,
            zmax=height,
            thickness=height,
            material=name,
            layer_type="dielectric",
            mesh_resolution="medium" if "contact" in name else "fine",
        )
        y += width
    for i, (region, metal) in enumerate(ELECTRODES.items()):
        y0, y1 = spans[region]
        draw_rectangle(
            comp, (46, i), width=10.0, height=y1 - y0, x0=0.0, y=(y0 + y1) / 2
        )
        layers[metal] = Layer(
            name=metal,
            gds_layer=(46, i),
            zmin=SLAB_HEIGHT,
            zmax=SLAB_HEIGHT + METAL_THICKNESS,
            thickness=METAL_THICKNESS,
            material="aluminum",
            layer_type="conductor",
            mesh_resolution="fine",
        )
    materials = make_doped_materials(
        [(name, 0.0) for name in REGIONS], permittivity=11.9
    )
    stack, _ = build_doped_cross_section(
        comp,
        axis="x",
        value=0.0,
        substrate_thickness=2.0,
        doping={"layer_specs": layers, "materials": materials},
        verbose=False,
    )
    return comp, stack, spans


def _graded_profiles():
    """Erfc tails of both implants across the rib; slab and contact doping."""
    n_peak, p_peak = DOPING["n_peak_cm3"], DOPING["p_peak_cm3"]
    plateau = {  # region: (donor plateau, acceptor plateau) in cm^-3
        "n_rib": (n_peak, p_peak),
        "p_rib": (n_peak, p_peak),
        "n_slab": (SLAB_N_CM3, SLAB_P_CM3),
        "p_slab": (SLAB_N_CM3, SLAB_P_CM3),
    }
    scale = np.sqrt(2) * SIGMA_UM
    profiles = []
    for region, (n_plateau, p_plateau) in plateau.items():
        profiles.append(
            CallableDoping(
                region=region,
                dopant_type="donor",
                function=lambda y, _z, nd=n_plateau: (
                    nd * 0.5 * erfc((y - CENTER_Y) / scale)
                ),
            )
        )
        profiles.append(
            CallableDoping(
                region=region,
                dopant_type="acceptor",
                function=lambda y, _z, na=p_plateau: (
                    na * 0.5 * erfc(-(y - CENTER_Y) / scale)
                ),
            )
        )
    profiles.extend(
        StepDoping(
            region=region,
            dopant_type="donor" if region[0] == "n" else "acceptor",
            concentration_cm3=1e20,
        )
        for region in ("n_contact", "p_contact")
    )
    return profiles


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    comp, stack, spans = _draw_device()
    rib = (spans["n_rib"][0], spans["p_rib"][1])
    window = (spans["n_contact"][0] - 0.5, spans["p_contact"][1] + 0.5)
    sim = Simulation()
    sim.geometry(comp, stack=stack)
    sim.domain(
        plane="x=0", window=window, margin_x=2.0, margin_y=2.0, z_above=1.5, z_below=1.0
    )
    sim.materials(insulators={"sio2": stack.materials["sio2"]["permittivity"]})
    for profile in _graded_profiles():
        sim.doping.add(profile)
    sim.contacts.add(
        name="cathode", semiconductor="n_contact", electrode="cathode_metal"
    )
    sim.contacts.add(name="anode", semiconductor="p_contact", electrode="anode_metal")
    for left, right in pairwise(REGIONS):
        junction = (left, right) == ("n_rib", "p_rib")
        sim.interfaces.add(
            name="junction" if junction else f"{left}_{right}",
            region_a=left,
            region_b=right,
        )
    # The notebook's coarse mesh (0.04 um elements, 0.02 um across the rib).
    sim.solver(
        mesh_preset="coarse",
        refined_mesh_size_um=0.04,
        max_mesh_size_um=40.0,
        refinement_boxes=[(*rib, 0.0, RIB_HEIGHT, 0.02)],
        bias_step_v=0.5,
    )
    sim.mesh(output_dir=tmp_path_factory.mktemp("bj2012-cv"))
    return sim.sweep(BIASES, contact="cathode")


@pytest.fixture(scope="module")
def c_ff_per_um(sweep):
    junction = PNJunctionConfig(
        na_cm3=DOPING["p_peak_cm3"], nd_cm3=DOPING["n_peak_cm3"]
    )
    comparison = compare_capacitance(junction, sweep, height_um=RIB_HEIGHT)
    # per unit length of device: the segment's capacitance times the fill
    return dict(
        zip(
            BIASES,
            FILL * comparison.c_tcad_f_per_cm * F_PER_CM_TO_FF_PER_UM,
            strict=True,
        )
    )


class TestJunctionCapacitance:
    def test_matches_the_stored_run(self, c_ff_per_um):
        for bias, expected in STORED_C_FF_PER_UM.items():
            assert c_ff_per_um[bias] == pytest.approx(expected, rel=0.03), bias

    def test_is_within_ten_percent_of_fig_2d(self, c_ff_per_um):
        paper = np.interp(
            BIASES,
            REFERENCE["capacitance"]["bias_v_reverse"],
            REFERENCE["capacitance"]["c_fF_per_um"],
        )
        for bias, expected in zip(BIASES, paper, strict=True):
            assert c_ff_per_um[bias] == pytest.approx(expected, rel=0.10), bias

    def test_falls_with_reverse_bias(self, c_ff_per_um):
        values = [c_ff_per_um[bias] for bias in BIASES]
        assert values == sorted(values, reverse=True)

    def test_the_stated_uniform_doping_would_be_far_off(self):
        """The reason for the graded profile, from the closed form alone.

        The abrupt-junction formula at the paper's quoted peaks (times the
        striation fill) is already about 40 % above Fig. 2d at 0 V; uniform
        doping in DEVSIM is about 1.9 times as high (see the notebook). Needs no
        solve.
        """
        junction = PNJunctionConfig(
            na_cm3=DOPING["p_peak_cm3"], nd_cm3=DOPING["n_peak_cm3"]
        )
        (c_abrupt,) = analytic_capacitance_f_per_cm(
            junction, [0.0], height_um=RIB_HEIGHT
        )
        paper_at_zero = np.interp(
            0.0,
            REFERENCE["capacitance"]["bias_v_reverse"],
            REFERENCE["capacitance"]["c_fF_per_um"],
        )
        assert c_abrupt * FILL * F_PER_CM_TO_FF_PER_UM > 1.3 * paper_at_zero


def test_slab_doping_gives_the_sheet_resistance():
    """The slab doping constants reproduce the paper's 10 / 7 kohm/sq (no solve)."""
    q = 1.602176634e-19
    mobility = MobilityModel.masetti_silicon()
    thickness_cm = SLAB_HEIGHT * 1e-4
    sheet_p = 1.0 / (
        q * float(mobility.holes_cm2(SLAB_P_CM3)) * SLAB_P_CM3 * thickness_cm
    )
    sheet_n = 1.0 / (
        q * float(mobility.electrons_cm2(SLAB_N_CM3)) * SLAB_N_CM3 * thickness_cm
    )
    assert sheet_p == pytest.approx(
        DOPING["slab_sheet_resistance_p_kohm_per_sq"] * 1e3, rel=0.02
    )
    assert sheet_n == pytest.approx(
        DOPING["slab_sheet_resistance_n_kohm_per_sq"] * 1e3, rel=0.02
    )
