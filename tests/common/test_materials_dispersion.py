"""Tests for dispersion models, validity ranges, and frequency-aware
material resolution."""

from __future__ import annotations

import warnings

import pytest

from gsim.common.stack.materials import (
    DispersionModel,
    LorentzianTerm,
    MaterialProperties,
    ResolvedMaterial,
    SellmeierTerm,
    ValidityRange,
    get_material_properties,
    resolve_material_at_wavelength,
    should_enable_dispersion,
)


class TestValidityRange:
    def test_unspecified(self):
        vr = ValidityRange()
        assert vr.is_unspecified
        assert not vr.covers_frequency(1e9)
        assert not vr.covers_wavelength(1.55)

    def test_frequency_range(self):
        vr = ValidityRange(valid_frequency=(1e9, 10e9))
        assert not vr.is_unspecified
        assert vr.covers_frequency(5e9)
        assert vr.covers_frequency(1e9)
        assert vr.covers_frequency(10e9)
        assert not vr.covers_frequency(100e9)
        assert not vr.covers_frequency(0.1e9)

    def test_wavelength_range(self):
        vr = ValidityRange(valid_wavelength=(0.21, 3.71))
        assert not vr.is_unspecified
        assert vr.covers_wavelength(1.55)
        assert vr.covers_wavelength(0.21)
        assert vr.covers_wavelength(3.71)
        assert not vr.covers_wavelength(0.1)
        assert not vr.covers_wavelength(5.0)

    def test_frequency_covers_wavelength(self):
        vr = ValidityRange(valid_frequency=(1e9, 10e9))
        wl_um = 299_792_458 / 5e9 * 1e6
        assert vr.covers_wavelength(wl_um)

    def test_wavelength_covers_frequency(self):
        vr = ValidityRange(valid_wavelength=(0.21, 3.71))
        freq_hz = 299_792_458 / (1.55 * 1e-6)
        assert vr.covers_frequency(freq_hz)


class TestSellmeierTerm:
    def test_n_squared_contribution(self):
        t = SellmeierTerm(B=0.696, C=0.0684**2)
        result = t.n_squared_contribution(1.55)
        assert result > 0

    def test_sellmeier_at_long_wavelength(self):
        t = SellmeierTerm(B=0.696, C=0.0684**2)
        result = t.n_squared_contribution(10.0)
        assert result > 0
        assert result < 1.0

    def test_anisotropic_sigma_diagonal(self):
        t = SellmeierTerm(B=0.696, C=0.0684**2, sigma_diagonal=[1.0, 1.0, 2.0])
        assert t.sigma_diagonal == [1.0, 1.0, 2.0]

    def test_default_sigma_diagonal_none(self):
        t = SellmeierTerm(B=0.696, C=0.0684**2)
        assert t.sigma_diagonal is None


class TestLorentzianTerm:
    def test_creation(self):
        pole = LorentzianTerm(frequency=0.5, gamma=0.01, sigma=1.0)
        assert pole.frequency == 0.5
        assert pole.gamma == 0.01
        assert pole.sigma == 1.0

    def test_anisotropic_sigma(self):
        pole = LorentzianTerm(
            frequency=0.5, gamma=0.01, sigma=1.0, sigma_diagonal=[1.0, 1.0, 0.5]
        )
        assert pole.sigma_diagonal == [1.0, 1.0, 0.5]

    def test_rejects_zero_frequency(self):
        with pytest.raises(ValueError):
            LorentzianTerm(frequency=0, gamma=0.01, sigma=1.0)


class TestDispersionModel:
    def test_constant_from_permittivity(self):
        dm = DispersionModel(type="constant", permittivity=4.1)
        assert dm.evaluate_n(1.55) == pytest.approx(2.0248, rel=1e-3)

    def test_sellmeier_sio2(self):
        dm = DispersionModel(
            type="sellmeier",
            sellmeier_terms=[
                SellmeierTerm(B=0.696, C=0.0684**2),
                SellmeierTerm(B=0.408, C=0.1162**2),
                SellmeierTerm(B=0.897, C=9.896**2),
            ],
            validity=ValidityRange(valid_wavelength=(0.21, 3.71)),
            source="Malitson 1965",
        )
        n = dm.evaluate_n(1.55)
        assert n == pytest.approx(1.4442, rel=1e-3)

    def test_sellmeier_at_wavelength_range_edge(self):
        dm = DispersionModel(
            type="sellmeier",
            sellmeier_terms=[
                SellmeierTerm(B=0.696, C=0.0684**2),
                SellmeierTerm(B=0.408, C=0.1162**2),
                SellmeierTerm(B=0.897, C=9.896**2),
            ],
        )
        n = dm.evaluate_n(0.21)
        assert n > 1.44

    def test_lorentzian_model(self):
        dm = DispersionModel(
            type="lorentzian",
            epsilon_inf=11.68,
            lorentzian_terms=[
                LorentzianTerm(frequency=5.0, gamma=0.1, sigma=1.0),
            ],
        )
        n = dm.evaluate_n(1.55)
        assert n > 0

    def test_evaluate_permittivity(self):
        dm = DispersionModel(type="constant", permittivity=4.1)
        assert dm.evaluate_permittivity(1.55) == 4.1

    def test_sellmeier_no_terms_errors(self):
        dm = DispersionModel(type="sellmeier")
        with pytest.raises(ValueError, match="no terms"):
            dm.evaluate_n(1.55)

    def test_constant_no_values_errors(self):
        dm = DispersionModel(type="constant")
        with pytest.raises(ValueError, match="no permittivity"):
            dm.evaluate_n(1.55)

    def test_validity(self):
        dm = DispersionModel(
            type="constant",
            permittivity=4.1,
            validity=ValidityRange(valid_wavelength=(0.21, 3.71)),
        )
        assert dm.validity.covers_wavelength(1.55)
        assert not dm.validity.covers_wavelength(10.0)

    def test_source(self):
        dm = DispersionModel(type="constant", permittivity=4.1, source="Malitson 1965")
        assert dm.source == "Malitson 1965"


class TestMaterialPropertiesEvaluation:
    def test_evaluate_at_wavelength_with_sellmeier(self):
        mat = get_material_properties("SiO2")
        assert mat is not None
        resolved = mat.evaluate_at_wavelength(1.55)
        assert resolved.permittivity is not None
        assert resolved.within_validity

    def test_evaluate_at_wavelength_rf(self):
        mat = get_material_properties("SiO2")
        assert mat is not None
        freq_hz = 5e9
        wl_um = 299_792_458 / freq_hz * 1e6
        resolved = mat.evaluate_at_wavelength(wl_um)
        assert resolved.permittivity is not None
        assert resolved.permittivity == pytest.approx(4.1, rel=1e-2)

    def test_evaluate_at_wavelength_permittivity_only(self):
        mat = MaterialProperties(permittivity=12.0)
        resolved = mat.evaluate_at_wavelength(1.55)
        assert resolved.permittivity == pytest.approx(12.0, rel=1e-3)
        assert "constant permittivity" in resolved.validity_note

    def test_evaluate_at_wavelength_permittivity_fallback(self):
        mat = MaterialProperties(permittivity=4.1)
        resolved = mat.evaluate_at_wavelength(1.55)
        assert resolved.permittivity is not None
        assert resolved.permittivity == 4.1

    def test_evaluate_at_frequency(self):
        mat = get_material_properties("SiO2")
        assert mat is not None
        resolved = mat.evaluate_at_frequency(193e12)
        assert resolved.permittivity is not None
        assert resolved.permittivity is not None

    def test_evaluate_no_data(self):
        mat = MaterialProperties()
        resolved = mat.evaluate_at_wavelength(1.55)
        assert not resolved.within_validity

    def test_index_variation_with_dispersion(self):
        mat = get_material_properties("silicon")
        assert mat is not None
        variation = mat.index_variation(1.55, 0.1)
        assert 0 < variation < 0.05

    def test_index_variation_wide_bandwidth(self):
        mat = get_material_properties("silicon")
        assert mat is not None
        variation_narrow = mat.index_variation(1.55, 0.1)
        variation_wide = mat.index_variation(1.55, 0.5)
        assert variation_wide > variation_narrow

    def test_index_variation_no_dispersive_model(self):
        mat = MaterialProperties()
        assert mat.index_variation(1.55, 0.1) == 0.0

    def test_unspecified_validity_warns(self):
        mat = MaterialProperties(
            dispersion_models=[
                DispersionModel(type="constant", permittivity=4.1),
            ],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mat.evaluate_at_wavelength(1.55)
            assert len(w) == 1
            assert "unspecified" in str(w[0].message).lower()

    def test_conductor_type_preserved(self):
        mat = MaterialProperties(conductivity=5.8e7)
        resolved = mat.evaluate_at_wavelength(1.55)
        assert resolved.conductivity == 5.8e7

    def test_to_dict_includes_dispersion_models(self):
        mat = MaterialProperties(
            dispersion_models=[
                DispersionModel(
                    type="constant",
                    permittivity=4.1,
                    validity=ValidityRange(valid_wavelength=(0.21, 3.71)),
                ),
            ],
        )
        d = mat.to_dict()
        assert "dispersion_models" in d
        assert len(d["dispersion_models"]) == 1  # ty: ignore[invalid-argument-type]

    def test_to_dict_includes_tensor_conductivity(self):
        mat = MaterialProperties(conductivity=[2.0, 2.0, 0.5])
        d = mat.to_dict()
        assert d["conductivity"] == [2.0, 2.0, 0.5]


class TestResolvedMaterial:
    def test_creation(self):
        rm = ResolvedMaterial(permittivity=2.0736, within_validity=True)
        assert rm.permittivity == pytest.approx(2.0736)
        assert rm.within_validity

    def test_anisotropic_fields(self):
        rm = ResolvedMaterial(
            permittivity=[9.3, 9.3, 11.5],
            material_axes=[[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        )
        assert isinstance(rm.permittivity, list)
        assert rm.material_axes is not None


class TestResolveMaterialAtWavelength:
    def test_resolve_sio2_at_optical(self):
        resolved = resolve_material_at_wavelength("SiO2", 1.55)
        assert resolved is not None
        assert resolved.permittivity is not None

    def test_resolve_unknown_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_material_at_wavelength("nonexistent_xyz", 1.55)
            assert result is None
            assert len(w) == 1

    def test_resolve_with_override(self):
        override = MaterialProperties(permittivity=6.25)
        resolved = resolve_material_at_wavelength(
            "SiO2", 1.55, overrides={"SiO2": override}
        )
        assert resolved is not None
        assert resolved.permittivity == pytest.approx(6.25, rel=1e-3)


class TestShouldEnableDispersion:
    def test_silicon_narrow_band(self):
        result = should_enable_dispersion("silicon", 1.55, 0.1)
        assert isinstance(result, bool)

    def test_sio2_narrow_band(self):
        result = should_enable_dispersion("SiO2", 1.55, 0.1)
        assert not result

    def test_unknown_material(self):
        result = should_enable_dispersion("nonexistent", 1.55, 0.1)
        assert not result


class TestMaterialsDB:
    def test_get_aluminum(self):
        props = get_material_properties("aluminum")
        assert props is not None
        assert props.conductivity == 3.77e7

    def test_get_copper(self):
        props = get_material_properties("copper")
        assert props is not None
        assert props.conductivity == 5.8e7

    def test_get_sio2(self):
        props = get_material_properties("SiO2")
        assert props is not None
        assert props.permittivity == 4.1

    def test_sio2_has_dispersion_models(self):
        props = get_material_properties("SiO2")
        assert props is not None
        assert len(props.dispersion_models) >= 2
        model_types = [m.type for m in props.dispersion_models]
        assert "sellmeier" in model_types
        assert "constant" in model_types

    def test_silicon_has_dispersion_models(self):
        props = get_material_properties("silicon")
        assert props is not None
        assert len(props.dispersion_models) >= 1

    def test_sio2_sellmeier_validity(self):
        props = get_material_properties("SiO2")
        assert props is not None
        sellmeier = next(m for m in props.dispersion_models if m.type == "sellmeier")
        assert sellmeier.validity.valid_wavelength is not None
        assert sellmeier.validity.valid_wavelength[0] == 0.21
        assert sellmeier.validity.valid_wavelength[1] == 6.7
        assert "Malitson 1965" in sellmeier.source

    def test_sio2_rf_model_validity(self):
        props = get_material_properties("SiO2")
        assert props is not None
        rf_model = next(m for m in props.dispersion_models if m.type == "constant")
        assert rf_model.validity.valid_frequency is not None
        assert rf_model.source == "IHP SG13G2 PDK"

    def test_get_by_alias(self):
        props = get_material_properties("al")
        assert props is not None
        assert props.conductivity is not None

    def test_case_insensitive(self):
        props = get_material_properties("ALUMINUM")
        assert props is not None
        assert props.conductivity is not None

    def test_unknown_material(self):
        props = get_material_properties("unknown_material_xyz")
        assert props is None

    def test_material_behavior_conductor_at_rf(self):
        resolved = resolve_material_at_wavelength("aluminum", 60000)
        assert resolved is not None
        assert resolved.behavior == "conductive"

    def test_material_behavior_dielectric(self):
        resolved = resolve_material_at_wavelength("SiO2", 1.55)
        assert resolved is not None
        assert resolved.behavior == "dielectric"

    def test_material_behavior_silicon_optical(self):
        resolved = resolve_material_at_wavelength("silicon", 1.55)
        assert resolved is not None
        assert resolved.behavior == "dielectric"

    def test_material_behavior_silicon_rf(self):
        resolved = resolve_material_at_wavelength("silicon", 60000)
        assert resolved is not None
        assert resolved.behavior == "dielectric"
        assert resolved.conductivity_scalar is not None

    def test_sapphire_anisotropic(self):
        props = get_material_properties("sapphire")
        assert props is not None
        assert isinstance(props.permittivity, list)
        assert props.permeability is not None
        assert isinstance(props.loss_tangent, list)
        assert props.material_axes is not None

    def test_dielectric_classmethod(self):
        mat = MaterialProperties.dielectric(permittivity=6.25)
        assert mat.permittivity == 6.25
        resolved = mat.evaluate_at_wavelength(1.55)
        assert resolved.behavior == "dielectric"

    def test_conductor_classmethod(self):
        mat = MaterialProperties.conductor(3.77e7)
        assert mat.conductivity == 3.77e7
        resolved = mat.evaluate_at_wavelength(1.55)
        assert resolved.behavior == "conductive"

    def test_dielectric_with_loss(self):
        mat = MaterialProperties.dielectric(permittivity=4.1, loss_tangent=0.001)
        assert mat.permittivity == 4.1
        assert mat.loss_tangent == 0.001


class TestOverlayInResolution:
    """Tests for three-tier resolution: override > overlay > built-in."""

    def test_overlay_overrides_builtin(self):
        overlay_si = MaterialProperties(
            permittivity=12.5,
            dispersion_models=[
                DispersionModel(
                    type="constant", permittivity=12.5, source="test overlay"
                ),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resolved = resolve_material_at_wavelength(
                "silicon", 1.55, overlay={"silicon": overlay_si}
            )
        assert resolved is not None
        assert resolved.permittivity == pytest.approx(12.5)

    def test_user_override_wins_over_overlay(self):
        override_si = MaterialProperties(permittivity=99.0)
        overlay_si = MaterialProperties(permittivity=12.5)
        resolved = resolve_material_at_wavelength(
            "silicon",
            1.55,
            overrides={"silicon": override_si},
            overlay={"silicon": overlay_si},
        )
        assert resolved is not None
        assert resolved.permittivity == 99.0

    def test_overlay_provides_missing_material(self):
        custom = MaterialProperties(permittivity=5.0)
        resolved = resolve_material_at_wavelength(
            "custom_foundry_mat", 1.55, overlay={"custom_foundry_mat": custom}
        )
        assert resolved is not None
        assert resolved.permittivity is not None

    def test_overlay_with_sellmeier(self):
        overlay_sio2 = MaterialProperties(
            permittivity=4.0,
            dispersion_models=[
                DispersionModel(
                    type="sellmeier",
                    sellmeier_terms=[
                        SellmeierTerm(B=0.696, C=0.0684**2),
                        SellmeierTerm(B=0.408, C=0.1162**2),
                        SellmeierTerm(B=0.897, C=9.896**2),
                    ],
                    validity=ValidityRange(valid_wavelength=(0.21, 3.71)),
                    source="test overlay",
                ),
            ],
        )
        resolved = resolve_material_at_wavelength(
            "SiO2", 1.55, overlay={"SiO2": overlay_sio2}
        )
        assert resolved is not None
        assert resolved.model_source == "test overlay"

    def test_should_enable_dispersion_with_overlay(self):
        overlay_si = MaterialProperties(
            dispersion_models=[
                DispersionModel(
                    type="sellmeier",
                    sellmeier_terms=[
                        SellmeierTerm(B=10.357, C=0.8832**2),
                        SellmeierTerm(B=0.860, C=6.004**2),
                    ],
                    validity=ValidityRange(valid_wavelength=(1.36, 11)),
                    source="test overlay",
                ),
            ],
        )
        result = should_enable_dispersion(
            "silicon", 1.55, 0.5, overlay={"silicon": overlay_si}
        )
        assert result is True

    def test_no_overlay_falls_to_builtin(self):
        resolved = resolve_material_at_wavelength("SiO2", 1.55, overlay=None)
        assert resolved is not None
        assert resolved.permittivity is not None
