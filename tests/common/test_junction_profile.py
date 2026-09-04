"""Tests for ``make_pn_junction_profile`` geometry, materials and mode selection."""

from __future__ import annotations

import logging
from typing import cast

import gdsfactory as gf
import pytest

from gsim.common.cross_section import RectYZ2D, extract_plane_section
from gsim.common.stack.doping import make_pn_junction_profile
from gsim.common.stack.extractor import LayerStack
from gsim.common.stack.junction import PNJunctionConfig

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
        with caplog.at_level(logging.INFO, logger="gsim.common.stack.doping"):
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
