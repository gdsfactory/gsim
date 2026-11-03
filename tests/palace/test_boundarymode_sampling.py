"""Boundary-mode paths use Palace's element-wise integration format."""

import json
import warnings

import pytest
from pydantic import ValidationError

from gsim.common.modes import Conductor
from gsim.palace import BoundaryModeSim
from gsim.palace.boundarymode import ModePath
from gsim.palace.line_impedance import ImpedancePaths, declare_impedance_paths
from gsim.palace.terminal_ports import TerminalPortSpec, declare_terminal_paths

_VOLTAGE = [[0.0, 0.0], [1.0, 0.0]]
_CURRENT = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
_DOMAIN = ((0.0, 10.0), (0.0, 5.0))
_TERMINALS = TerminalPortSpec(
    "pair",
    terminals=[
        Conductor("S1", ((2.0, 3.0), (2.0, 3.0))),
        Conductor("S2", ((4.0, 5.0), (2.0, 3.0))),
    ],
    reference=Conductor("G", ((7.0, 8.0), (2.0, 3.0))),
)


def _declare(simulation, api, **options):
    """Exercise each public path declaration with the same optional arguments."""
    if api == "add_impedance_path":
        return simulation.add_impedance_path(
            "gap", voltage=_VOLTAGE, current=_CURRENT, **options
        )
    if api == "declare_impedance_paths":
        paths = ImpedancePaths(
            voltage=((0.0, 0.0), (1.0, 0.0)),
            current=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )
        return declare_impedance_paths(simulation, paths, **options)
    return declare_terminal_paths(simulation, _TERMINALS, _DOMAIN, **options)


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


@pytest.mark.parametrize(
    "api", ["add_impedance_path", "declare_impedance_paths", "declare_terminal_paths"]
)
@pytest.mark.parametrize("nsamples", [100, 200])
def test_explicit_sampling_warns_once_without_changing_solver_inputs(api, nsamples):
    """Legacy calls keep indices and paths and never emit NSamples."""
    current = BoundaryModeSim()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        indices = _declare(current, api)
    legacy = BoundaryModeSim()
    with pytest.warns(DeprecationWarning, match="nsamples.*ignored") as captured:
        legacy_indices = _declare(legacy, api, nsamples=nsamples)

    assert len(captured) == 1
    assert captured[0].filename == __file__
    assert legacy_indices == indices
    assert legacy.mode_postprocessing() == current.mode_postprocessing()
    for entries in legacy.mode_postprocessing().values():
        assert all("NSamples" not in entry for entry in entries)
    assert all("nsamples" not in path for path in legacy.model_dump()["mode_paths"])


@pytest.mark.parametrize("nsamples", [100, 200])
def test_mode_path_accepts_explicit_sampling_with_a_warning(nsamples):
    """The deprecated model input stays readable but is excluded from settings."""
    with pytest.warns(DeprecationWarning, match="nsamples.*ignored") as captured:
        path = ModePath(name="gap", voltage_path=_VOLTAGE, nsamples=nsamples)
    assert len(captured) == 1
    assert captured[0].filename == __file__
    assert path.nsamples == nsamples
    assert "nsamples" not in path.model_dump()
    assert "nsamples" not in json.loads(path.model_dump_json())


def test_mode_path_assignment_warns_and_preserves_validation():
    """Assigning an old model field warns without affecting Palace output."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        path = ModePath(name="gap", voltage_path=_VOLTAGE)
    assert path.nsamples == 100
    with pytest.warns(DeprecationWarning, match="nsamples.*ignored") as captured:
        path.nsamples = 200
    assert len(captured) == 1
    assert captured[0].filename == __file__
    assert path.nsamples == 200
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        path.nsamples = 0
    assert path.nsamples == 200


@pytest.mark.parametrize("format_name", ["dict", "json"])
def test_saved_sampling_settings_warn_and_migrate_without_changing_paths(format_name):
    """Loading an old settings file warns; saving removes the obsolete field."""
    current = BoundaryModeSim()
    current.add_impedance_path("gap", voltage=_VOLTAGE)
    settings = current.model_dump()
    settings["mode_paths"][0]["nsamples"] = 500
    with pytest.warns(DeprecationWarning, match="nsamples.*ignored") as captured:
        restored = (
            BoundaryModeSim.model_validate(settings)
            if format_name == "dict"
            else BoundaryModeSim.model_validate_json(json.dumps(settings))
        )
    assert len(captured) == 1
    assert captured[0].filename == __file__
    assert restored.mode_postprocessing() == current.mode_postprocessing()
    migrated = json.loads(restored.model_dump_json())
    assert "nsamples" not in migrated["mode_paths"][0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        reloaded = BoundaryModeSim.model_validate(migrated)
    assert reloaded.mode_postprocessing() == current.mode_postprocessing()


@pytest.mark.parametrize(
    "api", ["add_impedance_path", "declare_impedance_paths", "declare_terminal_paths"]
)
def test_invalid_legacy_count_still_fails_before_declaring_paths(api):
    """Keeping the deprecated keyword also retains its positive-count check."""
    simulation = BoundaryModeSim()
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _declare(simulation, api, nsamples=0)
    assert simulation.mode_paths == []


def test_invalid_mode_path_sampling_still_fails_validation():
    """Old serialized counts retain their validation during the transition."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ModePath(name="gap", voltage_path=_VOLTAGE, nsamples=0)


def test_empty_paths_have_no_postprocessing():
    """No path declarations produce no postprocessing block."""
    assert BoundaryModeSim().mode_postprocessing() == {}
