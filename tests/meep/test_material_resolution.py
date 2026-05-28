"""Tests for MEEP material resolution with anisotropic tensors and dispersion."""

from __future__ import annotations

import math

import pytest

from gsim.common.stack.materials import (
    DispersionModel,
    LorentzianTerm,
    MaterialProperties,
    ResolvedMaterial,
    SellmeierTerm,
    ValidityRange,
)
from gsim.meep.materials import (
    _find_dispersion_model,
    _is_identity_axes,
    _resolved_to_material_data,
    _rotate_diagonal_tensor,
    _validity_to_freq_range,
    dispersion_model_to_meep_poles,
    lorentzian_to_meep_poles,
    loss_tangent_to_conductivity,
    resolve_materials,
    resolve_materials_with_dispersion,
    sellmeier_to_lorentzian_poles,
)


class TestLossTangentToConductivity:
    def test_zero_loss_tangent(self):
        assert loss_tangent_to_conductivity(0.0, 4.1, 5e9) == 0.0

    def test_conversion_at_5ghz(self):
        sigma = loss_tangent_to_conductivity(0.001, 4.1, 5e9)
        expected = 2.0 * math.pi * 5e9 * 8.854187817e-12 * 4.1 * 0.001
        assert sigma == pytest.approx(expected, rel=1e-6)

    def test_conversion_at_10ghz(self):
        sigma = loss_tangent_to_conductivity(0.01, 9.3, 10e9)
        expected = 2.0 * math.pi * 10e9 * 8.854187817e-12 * 9.3 * 0.01
        assert sigma == pytest.approx(expected, rel=1e-6)

    def test_higher_frequency_higher_conductivity(self):
        s1 = loss_tangent_to_conductivity(0.01, 4.0, 1e9)
        s2 = loss_tangent_to_conductivity(0.01, 4.0, 10e9)
        assert s2 > s1


class TestIsIdentityAxes:
    def test_none_is_identity(self):
        assert _is_identity_axes(None) is True

    def test_identity_matrix(self):
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert _is_identity_axes(identity) is True

    def test_rotated_is_not_identity(self):
        rotated = [[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]
        assert _is_identity_axes(rotated) is False


class TestRotateDiagonalTensor:
    def test_identity_rotation_preserves_zero_offdiag(self):
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        offdiag = _rotate_diagonal_tensor([9.3, 9.3, 11.5], identity)
        for v in offdiag:
            assert abs(v) < 1e-10

    def test_90_degree_rotation(self):
        axes = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        offdiag = _rotate_diagonal_tensor([1.0, 2.0, 3.0], axes)
        for v in offdiag:
            assert abs(v) < 1e-10

    def test_sapphire_rotation_produces_offdiag(self):
        axes = [[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]
        offdiag = _rotate_diagonal_tensor([9.3, 9.3, 11.5], axes)
        assert abs(offdiag[0]) < 1e-10
        assert abs(offdiag[2]) < 1e-10


class TestResolvedToMaterialData:
    def test_scalar_material(self):
        resolved = ResolvedMaterial(permittivity=3.47**2)
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.epsilon_diag == pytest.approx([3.47**2] * 3)

    def test_anisotropic_permittivity(self):
        resolved = ResolvedMaterial(permittivity=[9.3, 9.3, 11.5])
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.epsilon_diag == pytest.approx([9.3, 9.3, 11.5])

    def test_permeability(self):
        resolved = ResolvedMaterial(
            permittivity=1.77**2, permeability=[0.99999975, 0.99999975, 0.99999979]
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.mu_diag is not None
        assert len(data.mu_diag) == 3

    def test_loss_tangent_to_conductivity_conversion(self):
        resolved = ResolvedMaterial(permittivity=1.77**2, loss_tangent=3e-5)
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.D_conductivity is not None
        assert data.D_conductivity > 0

    def test_conductivity_direct(self):
        resolved = ResolvedMaterial(permittivity=1.0, conductivity=1e5)
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.D_conductivity == 1e5

    def test_tensor_conductivity(self):
        resolved = ResolvedMaterial(permittivity=1.0, conductivity=[1e4, 1e4, 1e5])
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.D_conductivity_diag == pytest.approx([1e4, 1e4, 1e5])

    def test_anisotropic_loss_tangent(self):
        resolved = ResolvedMaterial(
            permittivity=[9.3, 9.3, 11.5], loss_tangent=[3e-5, 3e-5, 8.6e-5]
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.D_conductivity_diag is not None
        assert len(data.D_conductivity_diag) == 3
        assert data.D_conductivity_diag[0] > 0
        assert data.D_conductivity_diag[2] > data.D_conductivity_diag[0]

    def test_material_axes_rotation(self):
        resolved = ResolvedMaterial(
            permittivity=[9.3, 9.3, 11.5],
            material_axes=[[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.epsilon_offdiag is not None

    def test_identity_axes_no_offdiag(self):
        resolved = ResolvedMaterial(
            permittivity=[9.3, 9.3, 11.5],
            material_axes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.epsilon_offdiag is None

    def test_no_permittivity_raises(self):
        resolved = ResolvedMaterial()
        with pytest.raises(ValueError, match="no permittivity"):
            _resolved_to_material_data(resolved, 1.55)


class TestResolveMaterialsWithTensors:
    def test_sapphire_resolves_with_tensors(self):
        materials = resolve_materials({"sapphire"}, wavelength_um=1.55)
        assert "sapphire" in materials
        sapph = materials["sapphire"]
        assert sapph.epsilon_diag is not None
        assert sapph.mu_diag is not None

    def test_conductor_skipped(self):
        materials = resolve_materials({"aluminum"}, wavelength_um=1.55)
        assert "aluminum" not in materials

    def test_isotropic_material_no_tensors(self):
        materials = resolve_materials({"SiO2"}, wavelength_um=1.55)
        assert "SiO2" in materials
        sio2 = materials["SiO2"]
        assert sio2.epsilon_diag is not None


class TestSellmeierToLorentzian:
    def test_sio2_sellmeier_produces_poles(self):
        model = DispersionModel(
            type="sellmeier",
            sellmeier_terms=[
                SellmeierTerm(B=0.696, C=0.0684**2),
                SellmeierTerm(B=0.408, C=0.1162**2),
                SellmeierTerm(B=0.897, C=9.896**2),
            ],
            validity=ValidityRange(valid_wavelength=(0.21, 3.71)),
        )
        poles = sellmeier_to_lorentzian_poles(model)
        assert len(poles) == 3
        for pole in poles:
            assert pole.gamma == 0.0
            assert pole.frequency > 0
            assert pole.sigma > 0

    def test_sellmeier_pole_frequencies(self):
        model = DispersionModel(
            type="sellmeier",
            sellmeier_terms=[
                SellmeierTerm(B=0.696, C=0.0684**2),
            ],
        )
        poles = sellmeier_to_lorentzian_poles(model)
        assert len(poles) == 1
        expected_w0 = 1.0 / math.sqrt(0.0684**2)
        assert poles[0].frequency == pytest.approx(expected_w0, rel=1e-6)

    def test_sellmeier_sigma(self):
        model = DispersionModel(
            type="sellmeier",
            sellmeier_terms=[
                SellmeierTerm(B=0.696, C=0.0684**2),
            ],
        )
        poles = sellmeier_to_lorentzian_poles(model)
        w0 = 1.0 / math.sqrt(0.0684**2)
        expected_sigma = 0.696 * w0**2
        assert poles[0].sigma == pytest.approx(expected_sigma, rel=1e-6)

    def test_sellmeier_anisotropic_sigma(self):
        model = DispersionModel(
            type="sellmeier",
            sellmeier_terms=[
                SellmeierTerm(B=0.696, C=0.0684**2, sigma_diagonal=[1.0, 1.0, 2.0]),
            ],
        )
        poles = sellmeier_to_lorentzian_poles(model)
        assert poles[0].sigma_diagonal == [1.0, 1.0, 2.0]

    def test_constant_model_no_poles(self):
        model = DispersionModel(type="constant", permittivity=4.1)
        poles = dispersion_model_to_meep_poles(model)
        assert poles == []

    def test_lorentzian_direct_mapping(self):
        model = DispersionModel(
            type="lorentzian",
            lorentzian_terms=[
                LorentzianTerm(frequency=0.5, gamma=0.01, sigma=1.0),
            ],
            epsilon_inf=1.5,
        )
        poles = lorentzian_to_meep_poles(model)
        assert len(poles) == 1
        assert poles[0].frequency == 0.5
        assert poles[0].gamma == 0.01
        assert poles[0].sigma == 1.0

    def test_zero_c_skipped(self):
        model = DispersionModel(
            type="sellmeier",
            sellmeier_terms=[
                SellmeierTerm(B=0.5, C=0.0),
            ],
        )
        poles = sellmeier_to_lorentzian_poles(model)
        assert poles == []


class TestValidityToFreqRange:
    def test_wavelength_validity(self):
        model = DispersionModel(
            type="sellmeier",
            sellmeier_terms=[SellmeierTerm(B=1.0, C=0.1)],
            validity=ValidityRange(valid_wavelength=(0.5, 2.0)),
        )
        freq_range = _validity_to_freq_range(model)
        assert freq_range is not None
        assert len(freq_range) == 2
        assert freq_range[0] == pytest.approx(1.0 / 2.0)
        assert freq_range[1] == pytest.approx(1.0 / 0.5)

    def test_unspecified_validity(self):
        model = DispersionModel(
            type="constant", permittivity=4.1, validity=ValidityRange()
        )
        freq_range = _validity_to_freq_range(model)
        assert freq_range is None


class TestFindDispersionModel:
    def test_finds_covering_model(self):
        props = MaterialProperties(
            permittivity=4.1,
            dispersion_models=[
                DispersionModel(
                    type="sellmeier",
                    sellmeier_terms=[SellmeierTerm(B=1.0, C=0.1)],
                    validity=ValidityRange(valid_wavelength=(0.5, 2.0)),
                ),
                DispersionModel(
                    type="constant",
                    permittivity=4.1,
                    validity=ValidityRange(valid_frequency=(0, 10e9)),
                ),
            ],
        )
        model = _find_dispersion_model(props, 1.55)
        assert model is not None
        assert model.type == "sellmeier"

    def test_falls_back_to_unspecified(self):
        props = MaterialProperties(
            permittivity=4.1,
            dispersion_models=[
                DispersionModel(
                    type="constant",
                    permittivity=4.1,
                    validity=ValidityRange(),
                    source="unspecified",
                ),
            ],
        )
        model = _find_dispersion_model(props, 1.55)
        assert model is not None
        assert model.source == "unspecified"

    def test_no_model_at_wavelength(self):
        props = MaterialProperties(
            permittivity=4.1,
            dispersion_models=[
                DispersionModel(
                    type="constant",
                    permittivity=4.1,
                    validity=ValidityRange(valid_frequency=(0, 10e9)),
                ),
            ],
        )
        model = _find_dispersion_model(props, 1.55)
        assert model is None

    def test_no_dispersion_models(self):
        props = MaterialProperties(permittivity=4.1)
        model = _find_dispersion_model(props, 1.55)
        assert model is None


class TestDispersiveMaterialData:
    def test_dispersive_rendering_populates_susceptibilities(self):
        model = DispersionModel(
            type="sellmeier",
            sellmeier_terms=[
                SellmeierTerm(B=0.696, C=0.0684**2),
                SellmeierTerm(B=0.408, C=0.1162**2),
                SellmeierTerm(B=0.897, C=9.896**2),
            ],
            validity=ValidityRange(valid_wavelength=(0.21, 3.71)),
            epsilon_inf=1.0,
        )
        resolved = ResolvedMaterial(permittivity=1.44**2)
        data = _resolved_to_material_data(resolved, 1.55, dispersive_model=model)
        assert data.epsilon_susceptibilities is not None
        assert len(data.epsilon_susceptibilities) == 3
        assert data.valid_freq_range is not None

    def test_constant_model_no_susceptibilities(self):
        model = DispersionModel(type="constant", permittivity=4.1)
        resolved = ResolvedMaterial(permittivity=1.44**2)
        data = _resolved_to_material_data(resolved, 1.55, dispersive_model=model)
        assert data.epsilon_susceptibilities is None

    def test_epsilon_inf_in_dispersive_mode(self):
        model = DispersionModel(
            type="sellmeier",
            sellmeier_terms=[SellmeierTerm(B=1.0, C=0.1)],
            epsilon_inf=2.0,
        )
        resolved = ResolvedMaterial(permittivity=2.25)
        data = _resolved_to_material_data(resolved, 1.55, dispersive_model=model)
        assert data.epsilon_diag == [2.0, 2.0, 2.0]


class TestResolveMaterialsWithDispersion:
    def test_auto_silicon_wide_bandwidth_gets_dispersion(self):
        materials = resolve_materials_with_dispersion(
            {"silicon"}, wavelength_um=1.55, bandwidth_um=0.5, dispersion="auto"
        )
        assert "silicon" in materials
        si = materials["silicon"]
        assert si.epsilon_susceptibilities is not None

    def test_auto_sio2_narrow_bandwidth_no_dispersion(self):
        materials = resolve_materials_with_dispersion(
            {"SiO2"}, wavelength_um=1.55, bandwidth_um=0.1, dispersion="auto"
        )
        assert "SiO2" in materials
        sio2 = materials["SiO2"]
        assert sio2.epsilon_susceptibilities is None

    def test_force_true_dispersion(self):
        materials = resolve_materials_with_dispersion(
            {"SiO2"}, wavelength_um=1.55, bandwidth_um=0.1, dispersion="true"
        )
        assert "SiO2" in materials
        sio2 = materials["SiO2"]
        assert sio2.epsilon_susceptibilities is not None

    def test_force_false_no_dispersion(self):
        materials = resolve_materials_with_dispersion(
            {"silicon"}, wavelength_um=1.55, bandwidth_um=0.5, dispersion="false"
        )
        assert "silicon" in materials
        si = materials["silicon"]
        assert si.epsilon_susceptibilities is None

    def test_conductor_skipped(self):
        materials = resolve_materials_with_dispersion(
            {"aluminum"}, wavelength_um=1.55, dispersion="true"
        )
        assert "aluminum" not in materials

    def test_auto_germanium_gets_dispersion(self):
        materials = resolve_materials_with_dispersion(
            {"germanium"}, wavelength_um=3.0, bandwidth_um=1.0, dispersion="auto"
        )
        assert "germanium" in materials
        ge = materials["germanium"]
        assert ge.epsilon_susceptibilities is not None
