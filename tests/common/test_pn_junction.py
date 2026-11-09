"""The PN-junction depletion model, and its capacitance on a Palace solve.

Part 1 checks the Sze formulas against hand values; part 2 checks that
the model's capacitance reaches a ``BoundaryModeSim`` as the lumped
Impedance boundary ``set_pn_junction`` applies between the two doped
halves of a rib.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError
from scipy.constants import Boltzmann as KB  # noqa: N814
from scipy.constants import elementary_charge as Q  # noqa: N812
from scipy.constants import epsilon_0 as EPS0  # noqa: N812

from gsim.common.stack.pn_junction import (
    PNJunctionConfig,
    built_in_voltage,
    depletion_extents,
    depletion_width,
    junction_capacitance_per_area,
)
from gsim.palace import BoundaryModeSim
from tests._helpers import draw_pn_rib

# ---------------------------------------------------------------------------
# Part 1: depletion model.
# ---------------------------------------------------------------------------


VT_300 = KB * 300.0 / Q


class TestBuiltInVoltage:
    def test_symmetric_silicon_value(self):
        v_bi = built_in_voltage(1e19, 1e19)
        expected = VT_300 * math.log(1e38 / (1.5e10) ** 2)
        assert v_bi == pytest.approx(expected, rel=1e-12)
        assert v_bi == pytest.approx(1.05, abs=0.03)

    def test_temperature_dependence(self):
        cold = built_in_voltage(1e18, 1e18, temperature_k=250.0)
        hot = built_in_voltage(1e18, 1e18, temperature_k=350.0)
        expected_cold = KB * 250.0 / Q * math.log(1e36 / (1.5e10) ** 2)
        expected_hot = KB * 350.0 / Q * math.log(1e36 / (1.5e10) ** 2)
        assert cold == pytest.approx(expected_cold, rel=1e-12)
        assert hot == pytest.approx(expected_hot, rel=1e-12)

    def test_rejects_nonphysical_inputs(self):
        with pytest.raises(ValueError):
            built_in_voltage(-1e18, 1e18)
        with pytest.raises(ValueError):
            built_in_voltage(1e18, 0.0)
        with pytest.raises(ValueError):
            built_in_voltage(1e18, 1e18, temperature_k=0.0)

    def test_rejects_degenerate_doping(self):
        with pytest.raises(ValueError, match="ni"):
            built_in_voltage(1e9, 1e9)


class TestDepletionWidthAbrupt:
    def test_symmetric_hand_check(self):
        w_um = depletion_width(1e18, 1e18)
        na_m3 = nd_m3 = 1e18 * 1e6
        eps_s = 11.9 * EPS0
        expected_m = math.sqrt(
            2
            * eps_s
            * VT_300
            * math.log(1e36 / (1.5e10) ** 2)
            / Q
            * (na_m3 + nd_m3)
            / (na_m3 * nd_m3)
        )
        assert w_um == pytest.approx(expected_m * 1e6, rel=1e-12)

    def test_reverse_bias_sqrt_scaling(self):
        w0 = depletion_width(1e19, 5e17)
        vbi = built_in_voltage(1e19, 5e17)
        w_r = depletion_width(1e19, 5e17, v_reverse=2.0)
        assert w_r / w0 == pytest.approx(math.sqrt((vbi + 2.0) / vbi), rel=1e-12)

    def test_one_sided_limit(self):
        # NA >> ND: nearly all the depletion spills into the lightly doped side.
        w = depletion_width(1e20, 1e17)
        xp, xn = depletion_extents(1e20, 1e17, w_um=w)
        assert xn == pytest.approx(w, rel=1e-3)
        assert xp == pytest.approx(w * 1e-3, rel=1e-2)

    def test_forward_bias_below_flatband(self):
        vbi = built_in_voltage(1e18, 1e18)
        w_eq = depletion_width(1e18, 1e18)
        w_fw = depletion_width(1e18, 1e18, v_reverse=-vbi / 2)
        assert w_fw < w_eq
        with pytest.raises(ValueError, match="flat-band"):
            depletion_width(1e18, 1e18, v_reverse=-(vbi + 0.01))


class TestDepletionWidthGraded:
    def test_cubic_root_law(self):
        a_cm4 = 1e21
        vbi = built_in_voltage(1e18, 1e18)
        w = depletion_width(1e18, 1e18, grading="linear", grade_const_cm4=a_cm4)
        eps_s = 11.9 * EPS0
        expected_m = (12 * eps_s * vbi / (Q * a_cm4 * 1e6)) ** (1 / 3)
        assert w == pytest.approx(expected_m * 1e6, rel=1e-12)

    def test_graded_bias_scaling(self):
        kwargs = dict(grading="linear", grade_const_cm4=1e21)
        w0 = depletion_width(1e18, 1e18, **kwargs)
        w_r = depletion_width(1e18, 1e18, v_reverse=1.0, **kwargs)
        vbi = built_in_voltage(1e18, 1e18)
        assert w_r / w0 == pytest.approx(((vbi + 1.0) / vbi) ** (1 / 3), rel=1e-12)

    def test_graded_is_symmetric(self):
        w = depletion_width(1e18, 1e19, grading="linear", grade_const_cm4=1e20)
        xp, xn = depletion_extents(1e18, 1e19, w_um=w, grading="linear")
        assert xp == pytest.approx(w / 2)
        assert xn == pytest.approx(w / 2)

    def test_requires_grade_constant(self):
        with pytest.raises(ValueError, match="grade_const"):
            depletion_width(1e18, 1e18, grading="linear")

    def test_unknown_grading(self):
        with pytest.raises(ValueError, match="grading"):
            depletion_width(1e18, 1e18, grading="exponential")  # type: ignore[arg-type]


class TestCapacitance:
    def test_per_area_inverse_w(self):
        eps_r = 11.9
        for w_um in (0.01, 0.05, 0.2):
            c = junction_capacitance_per_area(eps_r, w_um)
            assert c == pytest.approx(eps_r * EPS0 / (w_um * 1e-6), rel=1e-12)

    def test_absolute_capacitance(self):
        junc = PNJunctionConfig(na_cm3=1e19, nd_cm3=1e19)
        c = junc.capacitance(length_um=10.0, height_um=0.22)
        area_m2 = 10.0 * 0.22 * 1e-12
        assert c == pytest.approx(junc.c_per_area * area_m2, rel=1e-12)
        # Same order as typical TW-MZM junction caps (~fF per 10 um).
        assert 1e-15 < c < 1e-13

    def test_capacitance_scales_with_bias(self):
        junc0 = PNJunctionConfig(na_cm3=1e19, nd_cm3=1e19)
        junc_r = PNJunctionConfig(na_cm3=1e19, nd_cm3=1e19, v_reverse=3.0)
        assert junc_r.capacitance(10.0, 0.22) < junc0.capacitance(10.0, 0.22)


class TestPNJunctionConfig:
    def test_derived_quantities_consistent(self):
        cfg = PNJunctionConfig(na_cm3=2e18, nd_cm3=8e18, v_reverse=0.5)
        assert cfg.v_bi == pytest.approx(built_in_voltage(2e18, 8e18))
        assert cfg.w_um == pytest.approx(
            depletion_width(2e18, 8e18, v_reverse=0.5), rel=1e-12
        )
        total = cfg.xp_um + cfg.xn_um
        assert total == pytest.approx(cfg.w_um, rel=1e-12)
        # Asymmetric split: more depletion on the lighter-doped side.
        assert cfg.xp_um > cfg.xn_um

    def test_dict_construction(self):
        cfg = PNJunctionConfig.model_validate({"na_cm3": 1e19, "nd_cm3": 1e19})
        assert cfg.na_cm3 == 1e19

    def test_linear_requires_grade_const(self):
        with pytest.raises(ValidationError, match="grade_const"):
            PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18, grading="linear")

    def test_rejects_beyond_flatband(self):
        vbi = built_in_voltage(1e18, 1e18)
        with pytest.raises(ValidationError):
            PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18, v_reverse=-vbi - 0.05)

    def test_rejects_bad_concentrations(self):
        with pytest.raises(ValidationError):
            PNJunctionConfig(na_cm3=0.0, nd_cm3=1e18)

    def test_to_metadata_keys(self):
        meta = PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18).to_metadata()
        for key in (
            "na_cm3",
            "nd_cm3",
            "v_bi",
            "w_um",
            "xp_um",
            "xn_um",
            "c_per_area_f_m2",
            "grading",
        ):
            assert key in meta


# ---------------------------------------------------------------------------
# Part 2: the capacitance as a lumped Palace boundary.
# ---------------------------------------------------------------------------


F_RF = 50e9


@pytest.fixture(scope="module")
def capacitance_boundary(tmp_path_factory):
    """A rib with touching p/n halves, the Sze capacitance applied between them."""
    junction = PNJunctionConfig(na_cm3=1e19, nd_cm3=1e19)
    comp, stack = draw_pn_rib()
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path_factory.mktemp("cap") / "palace-sim-pn"))
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    sim.set_geometry(comp)
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=F_RF, num_modes=1, save=0)
    sim.mesh(preset="coarse", refined_mesh_size=0.05, max_mesh_size=40.0)
    applied = sim.set_pn_junction(
        junction, layer_p="p_rib", layer_n="n_rib", length_um=10.0, height_um=0.22
    )
    sim.write_config()
    config = json.loads((Path(sim.output_dir) / "config.json").read_text())
    return sim, config, junction, applied


class TestCapacitanceBoundary:
    def test_the_applied_capacitance_is_the_models(self, capacitance_boundary):
        _sim, _config, junction, applied = capacitance_boundary
        assert applied == pytest.approx(junction.capacitance(10.0, 0.22))

    def test_a_dict_junction_is_accepted(self, capacitance_boundary):
        sim, _config, _junction, applied = capacitance_boundary
        from_dict = sim.set_pn_junction(
            {"na_cm3": 1e19, "nd_cm3": 1e19},
            layer_p="p_rib",
            layer_n="n_rib",
            length_um=10.0,
            height_um=0.22,
        )
        assert from_dict == pytest.approx(applied)

    def test_impedance_boundary_in_config(self, capacitance_boundary):
        _sim, config, _junction, _applied = capacitance_boundary
        impedance = config.get("Boundaries", {}).get("Impedance", [])
        assert len(impedance) == 1
        assert "Cs" in impedance[0]
        assert impedance[0]["Cs"] > 0

    def test_cs_value_matches_computed_capacitance(self, capacitance_boundary):
        _sim, config, _junction, applied = capacitance_boundary
        # Interface p_rib|n_rib is the vertical rib edge; its curve length is
        # the 0.22 um rib height, so Cs = C / 0.22um.
        cs = config["Boundaries"]["Impedance"][0]["Cs"]
        assert cs == pytest.approx(applied / (0.22 * 1e-6), rel=1e-9)

    def test_the_doped_halves_are_domains_and_nothing_sits_between(
        self, capacitance_boundary
    ):
        sim, _config, _junction, _applied = capacitance_boundary
        volumes = sim.mesh_groups["volumes"]
        assert {"p_rib", "n_rib"} <= set(volumes)
        assert "junction" not in volumes
