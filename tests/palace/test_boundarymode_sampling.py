"""Boundary-mode paths use Palace's element-wise integration format."""

import json

import pytest

from gsim.palace import BoundaryModeSim


def test_paths_and_indices_survive_saved_settings_without_sampling_fields():
    """Saved settings and generated inputs describe the same coordinate paths."""
    simulation = BoundaryModeSim()
    simulation.add_impedance_path(
        "loop",
        voltage=[[0.0, 0.0], [1.0, 0.0]],
        current=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    )
    simulation.add_impedance_path("gap", voltage=[[0.0, 1.0], [1.0, 1.0]])
    saved = json.loads(simulation.model_dump_json())
    assert "include_legacy_nsamples" not in saved
    assert all("nsamples" not in path for path in saved["mode_paths"])
    restored = BoundaryModeSim.model_validate(saved)
    postprocessing = restored.mode_postprocessing()
    for kind in ("Impedance", "Voltage"):
        assert [entry["Index"] for entry in postprocessing[kind]] == [1, 2]
        assert postprocessing[kind][0]["VoltagePath"] == [[0.0, 0.0], [1.0, 0.0]]
        assert all("NSamples" not in entry for entry in postprocessing[kind])
    assert postprocessing["Impedance"][0]["CurrentPath"] == [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    assert "CurrentPath" not in postprocessing["Impedance"][1]


def test_removed_sampling_argument_is_rejected():
    """An obsolete sampling option cannot silently change a path declaration."""
    with pytest.raises(TypeError, match="nsamples"):
        BoundaryModeSim().add_impedance_path(
            "gap",
            voltage=[[0.0, 0.0], [1.0, 0.0]],
            nsamples=200,  # ty: ignore[unknown-argument]
        )


def test_empty_paths_have_no_postprocessing():
    """No path declarations produce no postprocessing block."""
    assert BoundaryModeSim().mode_postprocessing() == {}
