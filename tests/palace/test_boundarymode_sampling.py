"""Boundary-mode path configuration supports legacy and current Palace schemas."""

import json

import pytest

from gsim.palace import BoundaryModeSim
from gsim.palace.boundarymode import ModePath


@pytest.mark.parametrize("include_legacy", [True, False])
def test_sampling_interface_preserves_paths_and_saved_settings(include_legacy):
    """The schema choice survives saved models without changing path definitions."""
    simulation = BoundaryModeSim(include_legacy_nsamples=include_legacy)
    simulation.add_impedance_path(
        "loop",
        voltage=[[0.0, 0.0], [1.0, 0.0]],
        current=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        nsamples=200,
    )
    simulation.add_impedance_path("gap", voltage=[[0.0, 1.0], [1.0, 1.0]])
    restored = BoundaryModeSim.model_validate_json(simulation.model_dump_json())
    assert restored.include_legacy_nsamples is include_legacy
    assert restored.mode_paths[0].nsamples == 200
    postprocessing = restored.mode_postprocessing()
    for kind in ("Impedance", "Voltage"):
        assert [entry["Index"] for entry in postprocessing[kind]] == [1, 2]
        assert postprocessing[kind][0]["VoltagePath"] == [[0.0, 0.0], [1.0, 0.0]]
        if include_legacy:
            assert [entry["NSamples"] for entry in postprocessing[kind]] == [200, 100]
        else:
            assert all("NSamples" not in entry for entry in postprocessing[kind])
    assert postprocessing["Impedance"][0]["CurrentPath"] == [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    assert "CurrentPath" not in postprocessing["Impedance"][1]


def test_old_saved_paths_keep_the_legacy_sampling_default():
    """Saved configurations from before the flag retain explicit quadrature counts."""
    simulation = BoundaryModeSim(
        mode_paths=[
            ModePath(
                name="gap",
                voltage_path=[[0.0, 0.0], [1.0, 0.0]],
                nsamples=250,
            )
        ]
    )
    saved = json.loads(simulation.model_dump_json())
    del saved["include_legacy_nsamples"]
    restored = BoundaryModeSim.model_validate(saved)
    assert restored.mode_postprocessing()["Voltage"][0]["NSamples"] == 250


@pytest.mark.parametrize("include_legacy", [True, False])
def test_empty_paths_have_no_postprocessing(include_legacy):
    """Selecting a schema does not create empty postprocessing blocks."""
    assert (
        BoundaryModeSim(include_legacy_nsamples=include_legacy).mode_postprocessing()
        == {}
    )
