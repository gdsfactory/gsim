"""Tests for project-first MEEP material resolution."""

from __future__ import annotations

import math

import pytest
from scipy.constants import c as C0  # noqa: N812

from gsim.common.materials import MaterialNotFoundError
from gsim.common.stack.materials import MaterialProperties, ResolvedMaterial
from gsim.meep.material_cards import MeepMaterialCompatibilityError
from gsim.meep.materials import (
    _is_identity_axes,
    _resolved_to_material_data,
    _rotate_diagonal_tensor,
    loss_tangent_to_conductivity,
    resolve_fdtd_materials,
    resolve_materials,
)


def test_loss_tangent_converts_to_meep_d_conductivity() -> None:
    frequency_hz = C0 / 1.55e-6

    conductivity = loss_tangent_to_conductivity(0.001, 4.1, frequency_hz)

    expected = 2 * math.pi * (1 / 1.55) * 0.001
    assert conductivity == pytest.approx(expected)


def test_identity_material_axes_are_detected() -> None:
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    assert _is_identity_axes(None)
    assert _is_identity_axes(identity)
    assert not _is_identity_axes([[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])


def test_rotated_tensor_produces_meep_off_diagonal_terms() -> None:
    axes = [[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]

    off_diagonal = _rotate_diagonal_tensor([9.0, 4.0, 2.0], axes)

    assert off_diagonal[0] != 0
    assert off_diagonal[1:] == pytest.approx([0.0, 0.0])


def test_explicit_override_supports_tensor_permittivity() -> None:
    resolved = ResolvedMaterial(
        permittivity=[9.0, 4.0, 2.0],
        material_axes=[[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
    )

    material = _resolved_to_material_data(resolved, 1.55)

    assert material.epsilon_diag == [9.0, 4.0, 2.0]
    assert material.epsilon_offdiag is not None


def test_center_wavelength_resolution_uses_builtin_cards() -> None:
    materials = resolve_materials({"Si", "SiO2"}, wavelength_um=1.55)

    assert materials["Si"].epsilon_diag == pytest.approx([12.0945625246] * 3)
    assert materials["SiO2"].epsilon_diag == pytest.approx([1.4440236217**2] * 3)
    assert materials["Si"].epsilon_susceptibilities is None


def test_fdtd_resolution_preserves_sellmeier_poles() -> None:
    materials = resolve_fdtd_materials(
        {"Si", "SiO2"},
        wavelength_um=1.55,
        wavelength_span_um=0.1,
        resolution=32,
    )

    silicon = materials["Si"]
    silica = materials["SiO2"]
    assert silicon.epsilon_diag == [1.0, 1.0, 1.0]
    assert silica.epsilon_diag == [1.0, 1.0, 1.0]
    assert len(silicon.epsilon_susceptibilities or []) == 3
    assert len(silica.epsilon_susceptibilities or []) == 3
    assert {term.kind for term in silica.epsilon_susceptibilities or []} == {
        "lorentzian"
    }


def test_low_resolution_warns_for_high_sellmeier_pole() -> None:
    with pytest.warns(RuntimeWarning, match="resolution >= 23"):
        resolve_fdtd_materials(
            {"SiO2"},
            wavelength_um=1.55,
            wavelength_span_um=0.1,
            resolution=20,
        )


def test_explicit_override_remains_nondispersive() -> None:
    materials = resolve_fdtd_materials(
        {"Si"},
        overrides={"si": MaterialProperties(permittivity=12.0)},
        wavelength_um=1.55,
        wavelength_span_um=0.1,
        resolution=20,
    )

    assert materials["Si"].epsilon_diag == [12.0, 12.0, 12.0]
    assert materials["Si"].epsilon_susceptibilities is None


def test_override_keys_are_case_insensitive() -> None:
    materials = resolve_materials(
        {"SiO2"},
        overrides={"  sio2  ": MaterialProperties(permittivity=3.9)},
        wavelength_um=1.55,
    )

    assert materials["SiO2"].epsilon_diag == [3.9, 3.9, 3.9]


def test_duplicate_override_keys_warn() -> None:
    overrides = {
        "SiO2": MaterialProperties(permittivity=3.9),
        "sio2": MaterialProperties(permittivity=4.0),
    }

    with pytest.warns(UserWarning, match="Duplicate material entries"):
        resolve_materials({"SiO2"}, overrides=overrides, wavelength_um=1.55)


def test_missing_material_card_fails_strictly() -> None:
    with pytest.raises(MaterialNotFoundError, match="No MaterialCard"):
        resolve_fdtd_materials(
            {"unobtainium"},
            wavelength_um=1.55,
            wavelength_span_um=0.1,
            resolution=32,
        )


def test_fdtd_resolution_runs_meep_compatibility_validator() -> None:
    with pytest.raises(MeepMaterialCompatibilityError, match="causal Lorentz/Drude"):
        resolve_fdtd_materials(
            {"Si-Li-293K"},
            wavelength_um=1.55,
            wavelength_span_um=0.1,
            resolution=32,
        )
