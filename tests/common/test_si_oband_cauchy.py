"""Silicon in the O-band: Cauchy model from Li 1980, conductivity, coverage warning."""

from __future__ import annotations

import math
import warnings

import pytest
from pydantic import ValidationError
from scipy.constants import c as C0  # noqa: N812

from gsim.common.materials.si_li_293k import _li_index, li_coefficients
from gsim.common.stack.materials import (
    DispersionCoverageWarning,
    DispersionModel,
    MaterialProperties,
    ValidityRange,
    get_material_properties,
    resolve_material_at_wavelength,
)

OBAND_UM = (1.26, 1.31, 1.36)


def _silicon() -> MaterialProperties:
    props = get_material_properties("silicon")
    assert props is not None
    return props


class TestCauchyModel:
    def test_requires_terms(self):
        with pytest.raises(ValidationError, match="cauchy_terms"):
            DispersionModel(type="cauchy")
        with pytest.raises(ValidationError, match="cauchy_terms"):
            DispersionModel(type="cauchy", cauchy_terms=[])

    def test_series_evaluation(self):
        dm = DispersionModel(type="cauchy", cauchy_terms=[0.5, 0.02], epsilon_inf=10.0)
        wl = 2.0
        expected = 10.0 + 0.5 / wl**2 + 0.02 / wl**4
        assert dm.evaluate_permittivity(wl) == pytest.approx(expected, rel=1e-14)
        assert dm.evaluate_n(wl) == pytest.approx(math.sqrt(expected), rel=1e-14)

    def test_negative_eps_raises(self):
        dm = DispersionModel(type="cauchy", cauchy_terms=[-5.0], epsilon_inf=1.0)
        with pytest.raises(ValueError, match="n\\^2<0"):
            dm.evaluate_n(0.5)

    def test_dump_roundtrip(self):
        dm = DispersionModel(
            type="cauchy",
            cauchy_terms=[0.5],
            epsilon_inf=10.0,
            validity=ValidityRange(valid_wavelength=(1.0, 2.0)),
        )
        again = DispersionModel(**dm.model_dump())
        assert again == dm
        props = MaterialProperties(dispersion_models=[dm])
        dumped = props.to_dict()["dispersion_models"]
        assert isinstance(dumped, list)
        assert isinstance(dumped[0], dict)
        assert dumped[0]["cauchy_terms"] == [0.5]


class TestSiliconLi:
    @pytest.mark.parametrize("wl", [1.31, 1.55])
    def test_index_matches_li_function(self, wl):
        resolved = resolve_material_at_wavelength("silicon", wl)
        assert resolved is not None
        n = math.sqrt(resolved.permittivity_scalar)
        assert n == pytest.approx(_li_index(wl, 293.0), abs=1e-9)
        assert resolved.model_type == "cauchy"
        assert "Li 1980" in resolved.model_source
        assert resolved.within_validity

    def test_reference_values(self):
        # Values quoted for Li 1980 at 293 K (repository's own function).
        assert _li_index(1.31, 293.0) == pytest.approx(3.50027, abs=1e-5)
        assert _li_index(1.55, 293.0) == pytest.approx(3.47569, abs=1e-5)

    def test_coefficients_single_source(self):
        (model,) = [m for m in _silicon().dispersion_models if m.type == "cauchy"]
        eps, a = li_coefficients(293.0)
        assert model.epsilon_inf == eps
        assert model.cauchy_terms == [a]
        assert model.validity.valid_wavelength == (1.2, 14)
        assert model.source == "Li 1980, J. Phys. Chem. Ref. Data 9, 561 (293 K)"

    @pytest.mark.parametrize("wl", [*OBAND_UM, 1.55, 1.2, 14.0])
    def test_no_conductivity_in_validity_range(self, wl):
        resolved = resolve_material_at_wavelength("silicon", wl)
        assert resolved is not None
        assert resolved.conductivity is None
        assert resolved.conductivity_scalar is None
        assert resolved.behavior == "dielectric"

    def test_li_wins_over_salzberg_where_both_apply(self):
        props = _silicon()
        types = [m.type for m in props.dispersion_models]
        assert types == ["cauchy", "sellmeier", "constant"]
        salzberg = props.dispersion_models[1]
        for wl in (1.4, 1.55, 3.0):
            resolved = props.evaluate_at_wavelength(wl)
            assert resolved.model_type == "cauchy"
            # The two published models agree to better than 0.2 % in n.
            n_li = math.sqrt(resolved.permittivity_scalar)
            assert salzberg.evaluate_n(wl) == pytest.approx(n_li, rel=2e-3)
        # Salzberg alone is still evaluable where it is valid.
        assert salzberg.validity.covers_wavelength(1.55)
        assert not salzberg.validity.covers_wavelength(1.31)

    @pytest.mark.parametrize("freq_hz", [10e9, 5e9, 1e9])
    def test_rf_constant_unchanged_no_warning(self, freq_hz):
        wl_um = C0 / freq_hz * 1e6
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved = resolve_material_at_wavelength("silicon", wl_um)
            direct = _silicon().evaluate_at_frequency(freq_hz)
        assert resolved is not None
        assert resolved.permittivity == 11.9
        assert resolved.conductivity == 2.0
        assert resolved.model_type == "constant"
        assert resolved.within_validity
        assert direct.permittivity == 11.9
        assert direct.conductivity == 2.0

    def test_index_variation_still_positive_and_quiet(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            variation = _silicon().index_variation(1.55, 0.1)
        assert 0 < variation < 0.05


class TestNoCoverageWarning:
    def test_silicon_outside_all_models(self):
        with pytest.warns(DispersionCoverageWarning, match="no dispersion model"):
            resolved = resolve_material_at_wavelength("silicon", 0.5)
        assert resolved is not None
        assert not resolved.within_validity
        assert "no dispersion model covers" in resolved.validity_note
        # Base values are still returned, but flagged.
        assert resolved.permittivity == 11.9

    @pytest.mark.parametrize("freq_hz", [25.5e9, 50e9, 100e9])
    def test_rf_outside_constant_range_is_quiet_but_flagged(self, freq_hz):
        # RF sweeps beyond the PDK constant's 0-10 GHz range: no warning (it would
        # fire on every RF run), but the result is still flagged.
        wl_um = C0 / freq_hz * 1e6
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved = resolve_material_at_wavelength("silicon", wl_um)
            direct = _silicon().evaluate_at_frequency(freq_hz)
        assert resolved is not None
        assert resolved.permittivity == 11.9
        assert not resolved.within_validity
        assert not direct.within_validity

    def test_far_infrared_still_warns(self):
        with pytest.warns(DispersionCoverageWarning):
            _silicon().evaluate_at_wavelength(50.0)

    def test_exactly_one_warning_through_resolver(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_material_at_wavelength("silicon", 0.5)
        assert len(caught) == 1
        assert issubclass(caught[0].category, DispersionCoverageWarning)
        assert issubclass(DispersionCoverageWarning, UserWarning)

    def test_direct_evaluation_warns(self):
        with pytest.warns(DispersionCoverageWarning):
            _silicon().evaluate_at_wavelength(0.5)

    def test_gap_between_models_warns(self):
        # Salzberg alone left a gap below 1.36 um; a test material shows the gap
        # case without relying on the silicon entry.
        mat = MaterialProperties(
            permittivity=12.0,
            conductivity=2.0,
            dispersion_models=[
                DispersionModel(
                    type="cauchy",
                    cauchy_terms=[0.9],
                    epsilon_inf=11.4,
                    validity=ValidityRange(valid_wavelength=(1.36, 11)),
                ),
            ],
        )
        with pytest.warns(DispersionCoverageWarning):
            resolved = mat.evaluate_at_wavelength(1.31)
        assert not resolved.within_validity
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ok = mat.evaluate_at_wavelength(1.5)
        assert ok.within_validity
        assert ok.conductivity is None

    def test_old_silicon_gap_is_now_reported(self):
        # The previous silicon entry (Salzberg & Villa + RF constant only) left
        # 1.26-1.36 um uncovered and silently returned eps=11.9, sigma=2 S/m.
        old = _silicon().model_copy(
            update={"dispersion_models": _silicon().dispersion_models[1:]}
        )
        with pytest.warns(DispersionCoverageWarning):
            resolved = old.evaluate_at_wavelength(1.31)
        assert not resolved.within_validity
        assert resolved.conductivity == 2.0

    def test_no_models_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved = MaterialProperties(permittivity=12.0).evaluate_at_wavelength(
                1.55
            )
            empty = MaterialProperties().evaluate_at_wavelength(1.55)
        assert resolved.permittivity == 12.0
        assert resolved.within_validity
        assert not empty.within_validity

    def test_unspecified_validity_model_not_flagged(self):
        mat = MaterialProperties(
            dispersion_models=[DispersionModel(type="constant", permittivity=4.1)]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mat.evaluate_at_wavelength(1.55)
        assert len(caught) == 1
        assert "unspecified" in str(caught[0].message).lower()
        assert not issubclass(caught[0].category, DispersionCoverageWarning)

    @pytest.mark.parametrize("name", ["aluminum", "copper", "gold", "tungsten"])
    def test_conductors_exempt(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolved = resolve_material_at_wavelength(name, 1.55)
        assert resolved is not None
        assert resolved.behavior == "conductive"

    @pytest.mark.parametrize("name", ["SiO2", "Si3N4", "sapphire", "air"])
    def test_other_materials_in_range_quiet(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolve_material_at_wavelength(name, 1.55)
