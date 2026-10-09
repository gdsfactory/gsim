"""MEEP material resolution for Cauchy models (silicon, Li 1980).

Needs no MEEP install.
"""

from __future__ import annotations

import warnings

import pytest

from gsim.common.materials.si_li_293k import _li_index
from gsim.common.stack.materials import (
    DispersionModel,
    MaterialProperties,
    ResolvedMaterial,
    SellmeierTerm,
    ValidityRange,
)
from gsim.meep.materials import (
    _resolved_to_material_data,
    dispersion_model_to_meep_poles,
    resolve_materials,
    resolve_materials_with_dispersion,
)

CAUCHY = DispersionModel(
    type="cauchy",
    cauchy_terms=[0.9],
    epsilon_inf=11.4,
    validity=ValidityRange(valid_wavelength=(1.0, 2.0)),
    source="test cauchy",
)
SELLMEIER = DispersionModel(
    type="sellmeier",
    sellmeier_terms=[SellmeierTerm(B=1.0, C=0.09)],
    epsilon_inf=11.0,
    validity=ValidityRange(valid_wavelength=(1.4, 1.8)),
    source="test sellmeier",
)


def _record(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, caught


def test_cauchy_has_no_poles():
    assert dispersion_model_to_meep_poles(CAUCHY) == []


class TestSiliconOBand:
    @pytest.mark.parametrize("mode", ["true", "auto", "false"])
    def test_constant_eps_at_1p31(self, mode):
        # bandwidth chosen small enough that "auto" stays constant as well
        mats, caught = _record(
            resolve_materials_with_dispersion,
            {"silicon"},
            wavelength_um=1.31,
            bandwidth_um=0.001,
            dispersion=mode,
        )
        si = mats["silicon"]
        assert si.epsilon_susceptibilities is None
        assert si.D_conductivity is None
        assert si.D_conductivity_diag is None
        assert si.epsilon_diag == pytest.approx(
            [_li_index(1.31, 293.0) ** 2] * 3, abs=1e-9
        )
        cauchy_warnings = [w for w in caught if "Cauchy" in str(w.message)]
        if mode == "true":
            assert len(cauchy_warnings) == 1
            assert "non-dispersive" in str(cauchy_warnings[0].message)
            assert "1.31" in str(cauchy_warnings[0].message)
        else:
            assert not cauchy_warnings

    def test_auto_wide_band_at_1p31_warns_and_is_constant(self):
        mats, caught = _record(
            resolve_materials_with_dispersion,
            {"silicon"},
            wavelength_um=1.31,
            bandwidth_um=0.2,
            dispersion="auto",
        )
        si = mats["silicon"]
        assert si.epsilon_susceptibilities is None
        assert si.epsilon_diag == pytest.approx(
            [_li_index(1.31, 293.0) ** 2] * 3, abs=1e-9
        )
        assert any("non-dispersive" in str(w.message) for w in caught)

    def test_1p55_uses_covering_sellmeier_poles(self):
        mats, caught = _record(
            resolve_materials_with_dispersion,
            {"silicon"},
            wavelength_um=1.55,
            bandwidth_um=0.5,
            dispersion="true",
        )
        si = mats["silicon"]
        assert si.epsilon_susceptibilities is not None
        assert len(si.epsilon_susceptibilities) == 3
        assert si.D_conductivity is None
        assert not [w for w in caught if "Cauchy" in str(w.message)]

    def test_plain_resolve_materials_uses_li_constant(self):
        mats, caught = _record(resolve_materials, {"silicon"}, wavelength_um=1.31)
        si = mats["silicon"]
        assert si.epsilon_diag == pytest.approx(
            [_li_index(1.31, 293.0) ** 2] * 3, abs=1e-9
        )
        assert si.D_conductivity is None
        assert not caught

    def test_rf_silicon_not_affected(self):
        # 10 GHz = 29979.2458 um: constant model, base conductivity kept.
        mats, caught = _record(resolve_materials, {"silicon"}, wavelength_um=29979.2458)
        assert mats["silicon"].epsilon_diag == pytest.approx([11.9] * 3)
        assert mats["silicon"].D_conductivity == 2.0
        assert not caught


class TestCauchyMaterialSelection:
    def _props(self, with_sellmeier: bool) -> MaterialProperties:
        models = [CAUCHY] + ([SELLMEIER] if with_sellmeier else [])
        return MaterialProperties(permittivity=12.0, dispersion_models=models)

    def test_sellmeier_covering_wavelength_gives_poles(self):
        mats, caught = _record(
            resolve_materials_with_dispersion,
            {"x"},
            overrides={"x": self._props(True)},
            wavelength_um=1.5,
            bandwidth_um=0.2,
            dispersion="true",
        )
        data = mats["x"]
        assert data.epsilon_susceptibilities is not None
        assert data.epsilon_diag == [11.0] * 3  # the Sellmeier model's eps_inf
        assert data.valid_freq_range == pytest.approx([1 / 1.8, 1 / 1.4])
        assert not [w for w in caught if "Cauchy" in str(w.message)]

    def test_sellmeier_not_covering_falls_back_to_constant(self):
        # 1.2 um: Cauchy covers (1.0-2.0), Sellmeier (1.4-1.8) does not.
        mats, caught = _record(
            resolve_materials_with_dispersion,
            {"x"},
            overrides={"x": self._props(True)},
            wavelength_um=1.2,
            bandwidth_um=0.2,
            dispersion="true",
        )
        data = mats["x"]
        assert data.epsilon_susceptibilities is None
        assert data.epsilon_diag == pytest.approx(
            [CAUCHY.evaluate_permittivity(1.2)] * 3, rel=1e-14
        )
        assert any("non-dispersive" in str(w.message) for w in caught)

    def test_cauchy_only_falls_back_to_constant(self):
        mats, caught = _record(
            resolve_materials_with_dispersion,
            {"x"},
            overrides={"x": self._props(False)},
            wavelength_um=1.5,
            bandwidth_um=0.2,
            dispersion="true",
        )
        assert mats["x"].epsilon_susceptibilities is None
        assert any("non-dispersive" in str(w.message) for w in caught)

    def test_resolved_to_material_data_guard(self):
        resolved = ResolvedMaterial(permittivity=12.25)
        data, caught = _record(
            _resolved_to_material_data, resolved, 1.31, dispersive_model=CAUCHY
        )
        assert data.epsilon_diag == [12.25] * 3
        assert data.epsilon_susceptibilities is None
        assert data.valid_freq_range is None
        assert len(caught) == 1
