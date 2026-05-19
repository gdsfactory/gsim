"""Tests for MEEP material resolution with anisotropic tensors."""

from __future__ import annotations

import math

import pytest

from gsim.common.stack.materials import (
    DispersionModel,
    MaterialProperties,
    ResolvedMaterial,
    SellmeierTerm,
    ValidityRange,
)
from gsim.meep.materials import (
    _is_identity_axes,
    _resolved_to_material_data,
    _rotate_diagonal_tensor,
    loss_tangent_to_conductivity,
    resolve_materials,
)
from gsim.meep.models.config import MaterialData


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
        resolved = ResolvedMaterial(
            refractive_index=3.47,
            permittivity=3.47**2,
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.refractive_index == pytest.approx(3.47)
        assert data.extinction_coeff == 0.0
        assert data.epsilon_diag == pytest.approx([3.47**2] * 3)

    def test_anisotropic_permittivity(self):
        resolved = ResolvedMaterial(
            refractive_index=1.77,
            permittivity=1.77**2,
            permittivity_diagonal=[9.3, 9.3, 11.5],
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.epsilon_diag == pytest.approx([9.3, 9.3, 11.5])

    def test_permeability(self):
        resolved = ResolvedMaterial(
            refractive_index=1.77,
            permittivity=1.77**2,
            permeability=[0.99999975, 0.99999975, 0.99999979],
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.mu_diag is not None
        assert len(data.mu_diag) == 3

    def test_loss_tangent_to_conductivity_conversion(self):
        resolved = ResolvedMaterial(
            refractive_index=1.77,
            permittivity=1.77**2,
            loss_tangent=3e-5,
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.D_conductivity is not None
        assert data.D_conductivity > 0

    def test_conductivity_direct(self):
        resolved = ResolvedMaterial(
            refractive_index=1.0,
            permittivity=1.0,
            conductivity=1e5,
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.D_conductivity == 1e5

    def test_conductivity_diagonal(self):
        resolved = ResolvedMaterial(
            refractive_index=1.0,
            permittivity=1.0,
            conductivity_diagonal=[1e4, 1e4, 1e5],
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.D_conductivity_diag == pytest.approx([1e4, 1e4, 1e5])

    def test_anisotropic_loss_tangent(self):
        resolved = ResolvedMaterial(
            refractive_index=1.77,
            permittivity=1.77**2,
            loss_tangent_diagonal=[3e-5, 3e-5, 8.6e-5],
            permittivity_diagonal=[9.3, 9.3, 11.5],
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.D_conductivity_diag is not None
        assert len(data.D_conductivity_diag) == 3
        assert data.D_conductivity_diag[0] > 0
        assert data.D_conductivity_diag[2] > data.D_conductivity_diag[0]

    def test_material_axes_rotation(self):
        resolved = ResolvedMaterial(
            refractive_index=1.77,
            permittivity=1.77**2,
            permittivity_diagonal=[9.3, 9.3, 11.5],
            material_axes=[[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.epsilon_offdiag is not None

    def test_identity_axes_no_offdiag(self):
        resolved = ResolvedMaterial(
            refractive_index=1.77,
            permittivity=1.77**2,
            permittivity_diagonal=[9.3, 9.3, 11.5],
            material_axes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        data = _resolved_to_material_data(resolved, 1.55)
        assert data.epsilon_offdiag is None

    def test_no_refractive_index_raises(self):
        resolved = ResolvedMaterial(permittivity=4.1)
        with pytest.raises(ValueError, match="no refractive_index"):
            _resolved_to_material_data(resolved, 1.55)


class TestResolveMaterialsWithTensors:
    def test_sapphire_resolves_with_tensors(self):
        materials = resolve_materials(
            {"sapphire"}, wavelength_um=1.55
        )
        assert "sapphire" in materials
        sapph = materials["sapphire"]
        assert sapph.epsilon_diag is not None
        assert sapph.mu_diag is not None

    def test_conductor_skipped(self):
        materials = resolve_materials(
            {"aluminum"}, wavelength_um=1.55
        )
        assert "aluminum" not in materials

    def test_isotropic_material_no_tensors(self):
        materials = resolve_materials(
            {"SiO2"}, wavelength_um=1.55
        )
        assert "SiO2" in materials
        sio2 = materials["SiO2"]
        assert sio2.refractive_index is not None
