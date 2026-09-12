"""Tests for the consolidated PN-junction stack module.

Covers ``gsim.common.stack.pn_junction`` end to end:

- Part 1: depletion model (Sze ch. 2 formulas, validated against textbook
  expressions recomputed here with scipy.constants).
- Part 2: ``make_pn_junction_profile`` geometry, materials and mode selection.
- Part 3: Palace capacitance vs high-res mesh representation.
- Part 4: 1D free-carrier plasma dispersion (complex optical permittivity).
- Part 5: ``make_segmented_junction_profile`` strips and their Palace config.
"""

from __future__ import annotations

import json
import logging
import math
from itertools import pairwise
from pathlib import Path
from typing import cast

import gdsfactory as gf
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.constants import Boltzmann as KB  # noqa: N814
from scipy.constants import elementary_charge as Q  # noqa: N812
from scipy.constants import epsilon_0 as EPS0  # noqa: N812

from gsim.common.cross_section import (
    RectYZ2D,
    build_doped_cross_section,
    build_optical_cross_section,
    extract_plane_section,
)
from gsim.common.stack.extractor import LayerStack
from gsim.common.stack.pn_junction import (
    NI_SI_300K_CM3,
    PNJunctionConfig,
    built_in_voltage,
    carrier_profile_1d,
    depletion_extents,
    depletion_width,
    drude_relaxation_times,
    epsilon_eff_relative,
    junction_capacitance_per_area,
    junction_epsilon_profile,
    make_pn_junction_profile,
    make_segmented_junction_profile,
    optical_params,
    refractive_index,
    select_junction_mode,
)
from gsim.palace import BoundaryModeSim

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


class TestSelectJunctionMode:
    def test_comparable_width_selects_high_res(self):
        assert select_junction_mode(0.05, 0.2, 0.2) == "high_res"
        assert select_junction_mode(0.0401, 0.2, 0.2) == "high_res"

    def test_too_thin_selects_capacitance(self):
        assert select_junction_mode(0.0166, 0.2, 0.2) == "capacitance"
        assert select_junction_mode(0.0399, 0.2, 0.2) == "capacitance"

    def test_threshold_is_fraction_of_smaller_flank(self):
        assert select_junction_mode(0.0099, 0.05, 0.4, fraction=0.2) == "capacitance"
        assert select_junction_mode(0.0101, 0.05, 0.4, fraction=0.2) == "high_res"

    def test_custom_fraction(self):
        assert select_junction_mode(0.09, 0.2, 0.2, fraction=0.5) == "capacitance"
        assert select_junction_mode(0.11, 0.2, 0.2, fraction=0.5) == "high_res"

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            select_junction_mode(0.0, 0.2, 0.2)
        with pytest.raises(ValueError):
            select_junction_mode(0.1, 0.0, 0.2)
        with pytest.raises(ValueError):
            select_junction_mode(0.1, 0.2, 0.2, fraction=1.5)


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
# Part 2: PN-junction profile geometry and mode selection.
# ---------------------------------------------------------------------------


CY = -20.0
RIB_WIDTH = 0.4
LENGTH = 10.0

P_REGION = ("p_rib", (21, 0), 1.6e3)
N_REGION = ("n_rib", (20, 0), 1.6e3)
JUNCTION_REGION = ("junction", (22, 0))


def _thin_junction() -> PNJunctionConfig:
    """Na=Nd=1e19 cm^-3 at zero bias -> W ~ 17 nm < threshold."""
    return PNJunctionConfig(na_cm3=1e19, nd_cm3=1e19)


def _wide_junction() -> PNJunctionConfig:
    """Light doping + reverse bias -> W ~ 71 nm > threshold (40 nm)."""
    return PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18, v_reverse=1.0)


def _build(junction, **kwargs):
    comp = gf.Component()
    kwargs.setdefault("p_region", P_REGION)
    kwargs.setdefault("n_region", N_REGION)
    kwargs.setdefault("zmin", 0.0)
    kwargs.setdefault("zmax", 0.22)
    result = make_pn_junction_profile(
        comp,
        length=LENGTH,
        center_y=CY,
        rib_width=RIB_WIDTH,
        junction=junction,
        **kwargs,
    )
    return comp, result


def _section_rects(comp, result):
    """Extract the x=0 plane section from a profile-built component."""
    stack = LayerStack(pdk_name="test")
    stack.layers.update(result["layer_specs"])
    for name, mat in result["materials"].items():
        stack.materials[name] = mat.to_dict()
    rects = extract_plane_section(comp.copy(), stack, axis="x", value=0.0)
    # axis="x" always yields YZ rectangles; narrow the union for attribute access.
    return sorted(cast("list[RectYZ2D]", rects), key=lambda r: r.y0)


class TestAutoModeSelection:
    def test_thin_junction_selects_capacitance(self):
        _comp, res = _build(_thin_junction())
        assert res["junction"]["mode"] == "capacitance"
        assert "threshold" in res["junction"]["selection_reason"]

    def test_wide_junction_selects_high_res(self):
        _comp, res = _build(_wide_junction(), junction_region=JUNCTION_REGION)
        assert res["junction"]["mode"] == "high_res"

    def test_auto_logs_selection_reason(self, caplog):
        with caplog.at_level(logging.INFO, logger="gsim.common.stack.pn_junction"):
            _comp, _res = _build(_thin_junction())
        assert any("capacitance" in rec.message for rec in caplog.records)

    def test_forced_mode_overrides_auto(self):
        _comp, res = _build(
            _thin_junction(), mode="high_res", junction_region=JUNCTION_REGION
        )
        assert res["junction"]["mode"] == "high_res"
        assert "forced" in res["junction"]["selection_reason"]
        _comp, res = _build(_wide_junction(), mode="capacitance")
        assert res["junction"]["mode"] == "capacitance"


class TestCapacitanceModeGeometry:
    def test_no_junction_polygon_or_spec(self):
        comp, res = _build(_thin_junction())
        assert "junction" not in res["layer_specs"]
        assert "junction" not in res["materials"]
        # No polygon may exist on the junction GDS layer.
        polys = comp.get_polygons(layers=(JUNCTION_REGION[1],))
        assert not any(v for v in polys.values())

    def test_p_n_adjacent_halves(self):
        comp, res = _build(_thin_junction())
        rects = _section_rects(comp, res)
        names = [r.layer_name for r in rects]
        assert set(names) == {"p_rib", "n_rib"}
        by_name = {r.layer_name: r for r in rects}
        assert by_name["p_rib"].y0 == pytest.approx(CY)
        assert by_name["n_rib"].y1 == pytest.approx(CY)

    def test_junction_metadata_present(self):
        junc = _thin_junction()
        _comp, res = _build(junc)
        meta = res["junction"]
        assert meta["w_um"] == pytest.approx(junc.w_um)
        assert meta["c_f"] == pytest.approx(junc.capacitance(LENGTH, 0.22))
        assert meta["xp_um"] + meta["xn_um"] == pytest.approx(meta["w_um"])


class TestHighResModeGeometry:
    def test_three_contiguous_regions(self):
        comp, res = _build(_wide_junction(), junction_region=JUNCTION_REGION)
        rects = _section_rects(comp, res)
        names = [r.layer_name for r in rects]
        assert names == ["n_rib", "junction", "p_rib"]

        n_r, j_r, p_r = rects
        # Contiguity with no gaps or overlaps.
        assert n_r.y1 == pytest.approx(j_r.y0)
        assert j_r.y1 == pytest.approx(p_r.y0)

        junc = _wide_junction()
        # Depletion strip spans [cy - xn, cy + xp] (within layout DBU rounding).
        assert j_r.y0 == pytest.approx(CY - junc.xn_um, abs=2e-3)
        assert j_r.y1 == pytest.approx(CY + junc.xp_um, abs=2e-3)
        assert (j_r.y1 - j_r.y0) == pytest.approx(junc.w_um, abs=4e-3)
        # Flanks fill the rest of the rib.
        assert (p_r.y1 - p_r.y0) == pytest.approx(RIB_WIDTH / 2 - junc.xp_um, abs=4e-3)
        assert (n_r.y1 - n_r.y0) == pytest.approx(RIB_WIDTH / 2 - junc.xn_um, abs=4e-3)
        # Full rib span is covered exactly once.
        assert p_r.y1 - n_r.y0 == pytest.approx(RIB_WIDTH)

    def test_material_models(self):
        _comp, res = _build(_wide_junction(), junction_region=JUNCTION_REGION)
        # Doped regions carry Drude conductivity.
        for name in ("p_rib", "n_rib"):
            mat = res["materials"][name]
            assert mat.conductivity == pytest.approx(1.6e3)
            assert mat.permittivity == pytest.approx(11.9)
        # Junction strip: depleted silicon -> pure real permittivity, no carriers.
        jmat = res["materials"]["junction"]
        assert jmat.permittivity == pytest.approx(11.9)
        assert jmat.conductivity is None
        assert jmat.dispersion_models == []

    def test_high_res_requires_junction_region(self):
        with pytest.raises(ValueError, match="junction_region"):
            _build(_wide_junction())

    def test_layer_specs_reference_materials(self):
        _comp, res = _build(_wide_junction(), junction_region=JUNCTION_REGION)
        for name in ("p_rib", "n_rib", "junction"):
            spec = res["layer_specs"][name]
            assert spec.material == name
            assert spec.zmin == 0.0
            assert spec.zmax == 0.22


class TestValidation:
    def test_depletion_wider_than_rib_rejected(self):
        big = PNJunctionConfig(na_cm3=1e16, nd_cm3=1e16, v_reverse=5.0)
        if big.w_um <= RIB_WIDTH:
            pytest.skip("picked parameters do not exceed the rib width")
        with pytest.raises(ValueError, match="fit"):
            _build(big)

    def test_invalid_zmax_rejected(self):
        with pytest.raises(ValueError):
            _build(_thin_junction(), zmax=-1.0)

    def test_accepts_dict_junction_config(self):
        _comp, res = _build(
            {"na_cm3": 1e18, "nd_cm3": 1e18, "v_reverse": 1.0},
            junction_region=JUNCTION_REGION,
        )
        assert res["junction"]["w_um"] == pytest.approx(_wide_junction().w_um)
        assert res["junction"]["mode"] == "high_res"


# ---------------------------------------------------------------------------
# Part 3: Palace capacitance vs high-res representation.
# ---------------------------------------------------------------------------


F_RF = 50e9


def _build_device(junction: PNJunctionConfig, **profile_kwargs):
    """Build the rib+slab+doping device and return (comp, stack)."""
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    wg = comp << gf.c.rectangle((10.0, 0.4), centered=True, layer=(1, 0))
    wg.y = -20.0
    slab = comp << gf.c.rectangle((10.0, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0

    pn = make_pn_junction_profile(
        comp,
        length=10.0,
        center_y=-20.0,
        rib_width=0.4,
        junction=junction,
        p_region=("p_rib", (21, 0), 1.6e3),
        n_region=("n_rib", (20, 0), 1.6e3),
        junction_region=("junction", (22, 0)),
        zmin=0.0,
        zmax=0.22,
        **profile_kwargs,
    )
    stack, _section = build_doped_cross_section(
        comp,
        axis="x",
        value=0.0,
        substrate_thickness=2.0,
        doping=pn,
        verbose=False,
    )
    return comp, stack, pn


def _make_sim(junction: PNJunctionConfig, tmp_path: Path, apply_capacitance: bool):
    comp, stack, pn = _build_device(junction)
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path / "palace-sim-pn"))
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    sim.set_geometry(comp)
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=F_RF, num_modes=1, save=0)
    sim.mesh(preset="coarse", refined_mesh_size=0.05, max_mesh_size=40.0)
    if apply_capacitance:
        applied = sim.set_pn_junction(
            junction,
            layer_p="p_rib",
            layer_n="n_rib",
            length_um=10.0,
            height_um=0.22,
        )
        assert applied == pytest.approx(junction.capacitance(10.0, 0.22))
    sim.write_config()
    config_path = Path(sim.output_dir) / "config.json"
    return sim, json.loads(config_path.read_text()), pn


@pytest.fixture(scope="module")
def cap_mode(tmp_path_factory):
    """Thin depletion: auto-selected capacitance mode with lumped C."""
    return _make_sim(_thin_junction(), tmp_path_factory.mktemp("cap"), True)


@pytest.fixture(scope="module")
def hires_mode(tmp_path_factory):
    """Wide depletion: auto-selected high-res mode, no lumped C."""
    return _make_sim(_wide_junction(), tmp_path_factory.mktemp("hires"), False)


class TestCapacitanceMode:
    def test_no_junction_domain_on_mesh(self, cap_mode):
        sim, _config, _pn = cap_mode
        groups = sim._last_mesh_result.groups
        assert "junction" not in groups["volumes"]

    def test_impedance_boundary_in_config(self, cap_mode):
        _sim, config, _pn = cap_mode
        impedance = config.get("Boundaries", {}).get("Impedance", [])
        assert len(impedance) == 1
        assert "Cs" in impedance[0]
        assert impedance[0]["Cs"] > 0

    def test_cs_value_matches_computed_capacitance(self, cap_mode):
        _sim, config, pn = cap_mode
        # Interface p_rib|n_rib is the vertical rib edge; its curve length is
        # the 0.22 um rib height, so Cs = C / 0.22um.
        expected_cs = pn["junction"]["c_f"] / (0.22 * 1e-6)
        cs = config["Boundaries"]["Impedance"][0]["Cs"]
        assert cs == pytest.approx(expected_cs, rel=1e-9)

    def test_doped_domains_present(self, cap_mode):
        sim, _config, _pn = cap_mode
        groups = sim._last_mesh_result.groups
        assert {"p_rib", "n_rib"} <= set(groups["volumes"])


class TestHighResMode:
    def test_junction_dielectric_domain_on_mesh(self, hires_mode):
        sim, _config, _pn = hires_mode
        groups = sim._last_mesh_result.groups
        assert "junction" in groups["volumes"]
        assert groups["volumes"]["junction"].get("is_shaped_dielectric") is True

    def test_no_impedance_boundary(self, hires_mode):
        _sim, config, _pn = hires_mode
        assert not config.get("Boundaries", {}).get("Impedance")

    def test_junction_material_is_pure_dielectric(self, hires_mode):
        sim, config, _pn = hires_mode
        groups = sim._last_mesh_result.groups
        junc_attr = groups["volumes"]["junction"]["phys_group"]
        materials = config["Domains"]["Materials"]
        entries = [m for m in materials if junc_attr in m.get("Attributes", [])]
        assert len(entries) == 1, f"Expected one material for attr {junc_attr}"
        entry = entries[0]
        assert abs(float(entry["Permittivity"]) - 11.9) < 1e-6
        assert not entry.get("Conductivity"), (
            "Depleted silicon must have zero conductivity"
        )

    def test_p_n_junction_strip_contiguous(self, hires_mode):
        """All three regions survive as separate domains."""
        sim, _config, _pn = hires_mode
        volumes = sim._last_mesh_result.groups["volumes"]
        assert {"p_rib", "n_rib", "junction"} <= set(volumes)


# ---------------------------------------------------------------------------
# Part 4: 1D free-carrier plasma dispersion (complex optical permittivity).
# ---------------------------------------------------------------------------

LAMBDA_1550_UM = 1.55
EPS_BG_SI_1550 = 12.0946  # Si Sellmeier at 1.55 um (reference value)


class TestCarrierProfile1D:
    def test_bulk_and_depleted_values(self):
        na = nd = 1e18
        junc = PNJunctionConfig(na_cm3=na, nd_cm3=nd)
        y = np.array([-0.3, -junc.xn_um / 2, 0.0, junc.xp_um / 2, 0.3])
        n, p = carrier_profile_1d(
            y,
            center_um=0.0,
            xp_um=junc.xp_um,
            xn_um=junc.xn_um,
            na_cm3=na,
            nd_cm3=nd,
            ni_cm3=junc.ni_cm3,
        )
        assert n[0] == pytest.approx(nd)
        assert p[0] == pytest.approx(junc.ni_cm3**2 / nd)
        assert p[-1] == pytest.approx(na)
        assert n[-1] == pytest.approx(junc.ni_cm3**2 / na)
        assert n[1] == pytest.approx(junc.ni_cm3)
        assert p[2] == pytest.approx(junc.ni_cm3)

    def test_mass_action_holds_everywhere(self):
        junc = PNJunctionConfig(na_cm3=3e17, nd_cm3=2e18, v_reverse=1.0)
        y = np.linspace(-0.3, 0.3, 601)
        n, p = carrier_profile_1d(
            y,
            center_um=0.0,
            xp_um=junc.xp_um,
            xn_um=junc.xn_um,
            na_cm3=3e17,
            nd_cm3=2e18,
            ni_cm3=junc.ni_cm3,
        )
        assert n * p / junc.ni_cm3**2 == pytest.approx(np.ones_like(y))

    def test_asymmetric_split_respected(self):
        # NA >> ND: depletion lies almost entirely on the N side.
        junc = PNJunctionConfig(na_cm3=1e20, nd_cm3=1e17)
        assert junc.xn_um > 0.9 * junc.w_um
        n, p = carrier_profile_1d(
            np.array([-junc.w_um, junc.w_um]),
            center_um=0.0,
            xp_um=junc.xp_um,
            xn_um=junc.xn_um,
            na_cm3=1e20,
            nd_cm3=1e17,
        )
        assert n[0] == pytest.approx(1e17)
        assert p[1] == pytest.approx(1e20)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            carrier_profile_1d(
                [0.0], center_um=0.0, xp_um=0.1, xn_um=0.1, na_cm3=0.0, nd_cm3=1e18
            )
        with pytest.raises(ValueError):
            carrier_profile_1d(
                [0.0], center_um=0.0, xp_um=-0.1, xn_um=0.1, na_cm3=1e18, nd_cm3=1e18
            )


class TestDrudeOptics:
    def test_relaxation_times_in_relaxation_regime(self):
        tau_e, tau_h = drude_relaxation_times()
        assert tau_e > tau_h > 0
        omega = 2 * math.pi * 299792458 / (LAMBDA_1550_UM * 1e-6)
        assert omega * tau_e > 100
        assert omega * tau_h > 100
        with pytest.raises(ValueError):
            drude_relaxation_times(mu_n_cm2_vs=0.0)

    def test_intrinsic_returns_background(self):
        eps = epsilon_eff_relative(
            NI_SI_300K_CM3,
            NI_SI_300K_CM3,
            wavelength_um=LAMBDA_1550_UM,
            eps_bg_rel=EPS_BG_SI_1550,
        )
        assert float(np.real(eps)) == pytest.approx(EPS_BG_SI_1550, rel=1e-6)
        assert abs(float(np.imag(eps))) < 1e-8

    def test_electron_shift_matches_soref_scale(self):
        # Soref-Bennett at 1550 nm: dn_e = -8.8e-22 * dN (N in cm^-3).
        eps = epsilon_eff_relative(
            2e16, 0.0, wavelength_um=LAMBDA_1550_UM, eps_bg_rel=EPS_BG_SI_1550
        )
        n, _k = refractive_index(eps)
        dn = float(n) - math.sqrt(EPS_BG_SI_1550)
        assert dn == pytest.approx(-8.8e-22 * 2e16, rel=0.5)

    def test_heavy_doping_shift_and_loss(self):
        eps = epsilon_eff_relative(
            1e18, 1e18, wavelength_um=LAMBDA_1550_UM, eps_bg_rel=EPS_BG_SI_1550
        )
        n, k = refractive_index(eps)
        dn = float(n) - math.sqrt(EPS_BG_SI_1550)
        assert -3e-3 < dn < -5e-4
        assert 0.0 < float(k) < 1e-4

    def test_optical_params_mapping(self):
        eps = epsilon_eff_relative(
            1e18, 1e18, wavelength_um=LAMBDA_1550_UM, eps_bg_rel=EPS_BG_SI_1550
        )
        prime, sigma = optical_params(eps, LAMBDA_1550_UM)
        omega = 2 * math.pi * 299792458 / (LAMBDA_1550_UM * 1e-6)

        assert isinstance(prime, float)
        assert isinstance(sigma, float)
        assert prime == pytest.approx(float(np.real(eps)), rel=1e-12)
        assert sigma == pytest.approx(omega * EPS0 * float(np.imag(eps)), rel=1e-12)
        assert sigma > 0
        # Array input stays array.
        ep_arr, _sg_arr = optical_params([eps, eps], LAMBDA_1550_UM)
        assert isinstance(ep_arr, np.ndarray)
        assert ep_arr.shape == (2,)

    def test_junction_profile_depletion_dip(self):
        junc = PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18)
        y = np.linspace(-0.2, 0.2, 401)
        prof = junction_epsilon_profile(y, junc, wavelength_um=LAMBDA_1550_UM)
        assert set(prof) >= {
            "y_um",
            "n_cm3",
            "p_cm3",
            "eps_rel",
            "eps_prime",
            "sigma_Sm",
            "n_index",
            "k_index",
            "junction",
        }
        # Quasi-neutral wings carry the plasma shift; the depletion dip does not.
        assert prof["eps_prime"][0] < prof["eps_bg_rel"]
        mid = prof["eps_prime"][200]
        assert mid == pytest.approx(prof["eps_bg_rel"], rel=1e-6)
        assert prof["sigma_Sm"][200] < 1e-6 < prof["sigma_Sm"][0]

    def test_reverse_bias_widens_depleted_dip(self):
        y = np.linspace(-0.2, 0.2, 401)
        wide = junction_epsilon_profile(
            y,
            {"na_cm3": 1e17, "nd_cm3": 1e17, "v_reverse": 3.0},
            wavelength_um=LAMBDA_1550_UM,
        )
        narrow = junction_epsilon_profile(
            y,
            {"na_cm3": 1e17, "nd_cm3": 1e17},
            wavelength_um=LAMBDA_1550_UM,
        )
        assert wide["junction"]["w_um"] > 2 * narrow["junction"]["w_um"]
        depleted = np.asarray(wide["sigma_Sm"]) < 1e-6
        assert depleted.sum() > (np.asarray(narrow["sigma_Sm"]) < 1e-6).sum()


# ---------------------------------------------------------------------------
# Part 5: segmented-junction strips and their Palace representation.
# ---------------------------------------------------------------------------

SEG_CY = -20.0
SEG_RIB = 0.4
SEG_LENGTH = 10.0
SEG_JUNCTION = {"na_cm3": 1e18, "nd_cm3": 1e18}


def _build_segmented(n_p=8, n_n=8, **kwargs):
    comp = gf.Component()
    result = make_segmented_junction_profile(
        comp,
        length=SEG_LENGTH,
        center_y=SEG_CY,
        rib_width=SEG_RIB,
        junction=SEG_JUNCTION,
        n_p=n_p,
        n_n=n_n,
        zmin=0.0,
        zmax=0.22,
        **kwargs,
    )
    return comp, result


class TestSegmentedProfileGeometry:
    def test_sixteen_strips_cover_rib(self):
        _comp, res = _build_segmented()
        assert len(res["layer_specs"]) == 16
        assert len(res["materials"]) == 16
        rects = sorted((s["y0_um"], s["y1_um"]) for s in res["segments"].values())
        assert rects[0][0] == pytest.approx(SEG_CY - SEG_RIB / 2)
        assert rects[-1][1] == pytest.approx(SEG_CY + SEG_RIB / 2)
        for (_y0, y1), (y0_next, _y1) in pairwise(rects):
            assert y1 == pytest.approx(y0_next)
        total = sum(y1 - y0 for y0, y1 in rects)
        assert total == pytest.approx(SEG_RIB)

    def test_numbering_runs_junction_outward(self):
        _comp, res = _build_segmented()
        assert res["segments"]["p_1"]["y0_um"] == pytest.approx(SEG_CY)
        assert res["segments"]["n_1"]["y1_um"] == pytest.approx(SEG_CY)
        assert res["segments"]["p_8"]["y1_um"] == pytest.approx(SEG_CY + SEG_RIB / 2)
        assert res["segments"]["n_8"]["y0_um"] == pytest.approx(SEG_CY - SEG_RIB / 2)

    def test_junction_strips_depleted_outer_strips_bulk(self):
        _comp, res = _build_segmented()
        bg = res["junction"]["eps_bg_rel"]
        for name in ("p_1", "n_1"):
            seg = res["segments"][name]
            assert seg["n_cm3"] == pytest.approx(NI_SI_300K_CM3, rel=0.01)
            assert seg["eps_prime"] == pytest.approx(bg, rel=1e-9)
            assert res["materials"][name].conductivity is None
        for name in ("p_8", "n_8"):
            seg = res["segments"][name]
            assert seg["eps_prime"] < res["segments"]["p_1"]["eps_prime"]
            assert res["materials"][name].conductivity is not None
            assert res["materials"][name].conductivity > 0

    def test_materials_match_center_sampling(self):
        _comp, res = _build_segmented()
        cfg = PNJunctionConfig.model_validate(SEG_JUNCTION)
        for name, seg in res["segments"].items():
            prof = junction_epsilon_profile(
                [seg["yc_um"]],
                cfg,
                center_um=SEG_CY,
                wavelength_um=res["junction"]["wavelength_um"],
            )
            assert res["materials"][name].permittivity == pytest.approx(
                float(prof["eps_prime"][0]), rel=1e-12
            )

    def test_gds_layers_unique(self):
        _comp, res = _build_segmented()
        layers = [s["gds_layer"] for s in res["segments"].values()]
        assert len(set(layers)) == 16

    def test_section_extraction_yields_strips(self):
        comp, res = _build_segmented()
        stack = LayerStack(pdk_name="test")
        stack.layers.update(res["layer_specs"])
        for name, mat in res["materials"].items():
            stack.materials[name] = mat.to_dict()
        rects = extract_plane_section(comp.copy(), stack, axis="x", value=0.0)
        names = sorted(r.layer_name for r in cast("list[RectYZ2D]", rects))
        assert names == sorted(res["segments"])

    def test_rejects_bad_counts(self):
        with pytest.raises(ValueError, match="positive integers"):
            _build_segmented(n_p=0)


class TestSegmentedOpticalConfig:
    F_OPT = 193.4e12

    def _optical_sim(self, tmp_path, res, comp):
        gf.gpdk.PDK.activate()
        device_layers = {
            "core": ((1, 0), 0.0, 0.22),
            "slab": ((3, 0), 0.0, 0.09),
        }
        device_materials = {}
        extra_materials = {}
        for name, spec in res["layer_specs"].items():
            device_layers[name] = (tuple(spec.gds_layer), 0.0, 0.22)
            device_materials[name] = name
            extra_materials[name] = res["materials"][name]
        stack, _section = build_optical_cross_section(
            comp,
            axis="x",
            value=0.0,
            device_layers=device_layers,
            device_materials=device_materials,
            extra_materials=extra_materials,
            substrate_thickness=2.0,
            cladding_top=2.0,
            verbose=False,
        )
        sim = BoundaryModeSim()
        sim.set_output_dir(str(tmp_path / "palace-sim-seg-opt"))
        sim.set_stack(stack)
        sim.set_airbox(
            material="sio2", margin_x=1.0, margin_y=1.0, z_above=1.0, z_below=1.0
        )
        sim.set_geometry(comp)
        sim.set_cross_section("x=0")
        sim.set_boundary_mode(freq=self.F_OPT, num_modes=1, save=0)
        sim.mesh(preset="coarse", refined_mesh_size=0.02, max_mesh_size=1.0)
        sim.write_config()
        config = json.loads((Path(sim.output_dir) / "config.json").read_text())
        return sim, config

    def test_strip_domains_carry_permittivity_and_conductivity(self, tmp_path):
        comp, res = _build_segmented()
        sim, config = self._optical_sim(tmp_path, res, comp)
        volumes = sim._last_mesh_result.groups["volumes"]
        assert {f"p_{i}" for i in range(1, 9)} <= set(volumes)
        assert {f"n_{i}" for i in range(1, 9)} <= set(volumes)
        materials = config["Domains"]["Materials"]
        bg = res["junction"]["eps_bg_rel"]
        for name, seg in res["segments"].items():
            attr = volumes[name]["phys_group"]
            entries = [m for m in materials if attr in m.get("Attributes", [])]
            assert len(entries) == 1, f"Expected one material for {name}"
            entry = entries[0]
            assert float(entry["Permittivity"]) == pytest.approx(
                seg["eps_prime"], rel=1e-9
            )
            if seg["sigma_Sm"] >= 1e-6:
                assert float(entry["Conductivity"]) == pytest.approx(
                    seg["sigma_Sm"], rel=1e-9
                )
                assert float(entry["Permittivity"]) < bg
            else:
                assert not entry.get("Conductivity")
                assert float(entry["Permittivity"]) == pytest.approx(bg, rel=1e-9)
