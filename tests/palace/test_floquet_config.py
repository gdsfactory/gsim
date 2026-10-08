"""Floquet phases use actual mesh translations, independently of eigensolver shifts."""

from __future__ import annotations

import json
import math

import pytest

from gsim.common import LayerStack
from gsim.palace import EigenmodeSim
from gsim.palace.mesh.config_generator import generate_palace_config, write_config
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.models import EigenmodeConfig


def _groups():
    return {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {
            "periodic_donor": {"phys_group": [3, 2]},
            "periodic_receiver": {"phys_group": [5, 4]},
        },
    }


def _generate(tmp_path, eigenmode=None, *, axis="x", translation=(100.25, 0.0, 0.0)):
    config_path = generate_palace_config(
        groups=_groups(),
        ports=[],
        port_info=[],
        stack=LayerStack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type="eigenmode",
        eigenmode_config=eigenmode or EigenmodeConfig(target=40e9, floquet=True),
        periodic_axis=axis,
        periodic_translation=translation,
    )
    return json.loads(config_path.read_text())


@pytest.mark.parametrize("phase", [-math.pi, -0.4, 0.0, 0.4, math.pi])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_generated_json_has_exact_signed_phase(tmp_path, phase, axis):
    translation = [0.0, 0.0, 0.0]
    translation["xyz".index(axis)] = 100.25
    config = _generate(
        tmp_path,
        EigenmodeConfig(target=40e9, floquet=True, phi_target=phase),
        axis=axis,
        translation=tuple(translation),
    )
    periodic = config["Boundaries"]["Periodic"]
    pair = periodic["BoundaryPairs"][0]
    assert config["Model"]["L0"] == 1e-6
    assert pair == {
        "DonorAttributes": [2, 3],
        "ReceiverAttributes": [4, 5],
        "Translation": translation,
    }
    wave_vector = periodic["FloquetWaveVector"]
    assert wave_vector["xyz".index(axis)] == pytest.approx(phase / 100.25)
    assert sum(k * d for k, d in zip(wave_vector, translation, strict=True)) == (
        pytest.approx(phase)
    )


def test_target_and_legacy_index_do_not_change_phase(tmp_path):
    blocks = [
        _generate(
            tmp_path,
            EigenmodeConfig(target=target, n_eff_guess=index, floquet=True),
        )["Boundaries"]["Periodic"]
        for target, index in [(40e9, 2.0), (80e9, 4.0)]
    ]
    assert blocks[0] == blocks[1]
    assert blocks[0]["FloquetWaveVector"] == pytest.approx(
        [math.pi / 2 / 100.25, 0.0, 0.0]
    )


def test_public_api_propagates_expected_period():
    sim = EigenmodeSim()
    sim.set_eigenmode(target=40e9, floquet=True, phi_target=-0.4, periodic_length=100)
    assert sim.eigenmode.compute_floquet_wave_vector(periodic_axis="x") == (
        pytest.approx([-0.004, 0.0, 0.0])
    )


def test_expected_period_must_match_geometry(tmp_path):
    with pytest.raises(ValueError, match="does not match the mesh translation"):
        _generate(
            tmp_path,
            EigenmodeConfig(target=40e9, floquet=True, periodic_length=100),
        )
    assert not (tmp_path / "config.json").exists()


def test_matching_expected_period_uses_measured_length(tmp_path):
    config = _generate(
        tmp_path,
        EigenmodeConfig(target=40e9, floquet=True, periodic_length=100.25),
        translation=(100.2500002, 0.0, 0.0),
    )
    assert config["Boundaries"]["Periodic"]["FloquetWaveVector"][0] == (
        math.pi / 2 / 100.2500002
    )


@pytest.mark.parametrize("period", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_period_rejected_by_config_and_helper(period):
    with pytest.raises(ValueError):
        EigenmodeConfig(periodic_length=period)
    with pytest.raises(ValueError, match="finite and positive"):
        EigenmodeConfig().compute_floquet_wave_vector(
            periodic_axis="x", periodic_length=period
        )


@pytest.mark.parametrize("phase", [math.nan, math.inf, -math.inf])
def test_nonfinite_phase_rejected(phase):
    with pytest.raises(ValueError):
        EigenmodeConfig(phi_target=phase)


def test_direct_helper_requires_explicit_period():
    with pytest.raises(ValueError, match="requires an actual periodic_length"):
        EigenmodeConfig(target=40e9).compute_floquet_wave_vector(periodic_axis="x")


@pytest.mark.parametrize("axis", [None, "invalid"])
def test_missing_or_invalid_axis_rejected(tmp_path, axis):
    with pytest.raises(ValueError, match="requires a periodic axis"):
        _generate(tmp_path, axis=axis)


@pytest.mark.parametrize(
    "translation",
    [(0, 0, 0), (-100, 0, 0), (100, 2, 0), (math.nan, 0, 0), (math.inf, 0, 0), (100,)],
)
def test_invalid_mesh_translation_rejected(tmp_path, translation):
    with pytest.raises(ValueError, match="finite, positive translation"):
        _generate(tmp_path, translation=translation)


@pytest.mark.parametrize("expected_period", [None, 100.25])
def test_missing_metadata_never_falls_back_to_guess(tmp_path, expected_period):
    with pytest.raises(ValueError, match="requires the mesh periodic_translation"):
        _generate(
            tmp_path,
            EigenmodeConfig(target=40e9, floquet=True, periodic_length=expected_period),
            translation=None,
        )


def test_deferred_config_uses_mesh_translation(tmp_path):
    mesh_result = MeshResult(
        mesh_path=tmp_path / "palace.msh",
        output_dir=tmp_path,
        groups=_groups(),
        periodic_axis="y",
        periodic_translation=(0.0, 120.5, 0.0),
    )
    config_path = write_config(
        mesh_result,
        LayerStack(),
        [],
        simulation_type="eigenmode",
        eigenmode_config=EigenmodeConfig(target=40e9, floquet=True, phi_target=-0.6),
    )
    periodic = json.loads(config_path.read_text())["Boundaries"]["Periodic"]
    assert periodic["FloquetWaveVector"] == pytest.approx([0, -0.6 / 120.5, 0])
    assert periodic["BoundaryPairs"][0]["Translation"] == [0.0, 120.5, 0.0]


def test_nonfloquet_config_unchanged_without_period_metadata(tmp_path):
    config = _generate(tmp_path, EigenmodeConfig(target=40e9), translation=None)
    assert "Periodic" not in config["Boundaries"]
    assert config["Solver"]["Eigenmode"]["Target"] == 40
