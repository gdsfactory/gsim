"""The Palace config generator's typed inputs and their deprecated ``hints`` keys.

The generated ``config.json`` is what Palace solves and what the cloud cache key
is computed from, so both spellings must write the same bytes.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import BoundaryModeSim, DrivenSim
from gsim.palace.mesh import write_config
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models import BoundaryModeConfig
from gsim.palace.models.ports import ImpedanceBoundaryConfig

POSTPROCESSING = {
    "Voltage": [{"Index": 1, "VoltagePath": [[0.0, 0.0], [1.0, 0.0]], "NSamples": 9}]
}
IMPEDANCE = [ImpedanceBoundaryConfig(attributes=[20], resistance=1.0)]

# keyword, value, the deprecated hints key, and what the keyword writes
OPTIONS = [
    (
        "impedance_boundaries",
        IMPEDANCE,
        "_impedance_boundaries",
        lambda config: config["Boundaries"]["Impedance"][0]["Attributes"] == [20],
    ),
    (
        "metallic_boundaries",
        True,
        "_metallic_boundaries",
        lambda config: 8 in config["Boundaries"]["PEC"]["Attributes"],
    ),
    (
        "conductor_boundary",
        "pec",
        "_conductor_boundary",
        lambda config: (
            "Conductivity" not in config["Boundaries"]
            and {4, 5} <= set(config["Boundaries"]["PEC"]["Attributes"])
        ),
    ),
    (
        "mode_postprocessing",
        POSTPROCESSING,
        "_mode_postprocessing",
        lambda config: config["Boundaries"]["Postprocessing"] == POSTPROCESSING,
    ),
    (
        "cross_section_axis",
        "x",
        "_cross_section_axis",
        lambda config: config["Domains"]["Materials"][0]["Permittivity"][0] != 4.0,
    ),
]
OPTION_IDS = [option[0] for option in OPTIONS]


def _stack() -> LayerStack:
    stack = LayerStack()
    stack.layers["metal"] = Layer(
        name="metal",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=3.0,
        thickness=3.0,
        material="aluminum",
        layer_type="conductor",
    )
    stack.materials = {
        "aluminum": {"conductivity": 3.77e7},
        "core": {"permittivity": [4.0, 5.0, 6.0]},
    }
    return stack


def _groups() -> dict:
    return {
        "volumes": {"core": {"phys_group": 1}},
        "conductor_surfaces": {
            "metal_xy": {"phys_group": 4},
            "metal_z": {"phys_group": 5},
        },
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {"absorbing": {"phys_group": 8}},
    }


def _generate(output_dir: Path, **options) -> str:
    """Write a boundary-mode config and return its exact text."""
    output_dir.mkdir()
    path = generate_palace_config(
        groups=_groups(),
        ports=[],
        port_info=[],
        stack=_stack(),
        output_path=output_dir,
        model_name="palace",
        fmax=50e9,
        simulation_type="boundarymode",
        boundary_mode_config=BoundaryModeConfig(freq=50e9, num_modes=2, save=1),
        absorbing_boundary=False,
        **options,
    )
    return path.read_text()


@pytest.mark.parametrize(
    ("keyword", "value", "_key", "written"), OPTIONS, ids=OPTION_IDS
)
def test_keyword_changes_the_config(tmp_path, keyword, value, _key, written):
    baseline = _generate(tmp_path / "baseline")
    config = _generate(tmp_path / "option", **{keyword: value})
    assert config != baseline
    assert written(json.loads(config))


@pytest.mark.parametrize(
    ("keyword", "value", "key", "_written"), OPTIONS, ids=OPTION_IDS
)
def test_deprecated_hint_writes_the_same_config(
    tmp_path, keyword, value, key, _written
):
    expected = _generate(tmp_path / "keyword", **{keyword: value})
    with pytest.warns(DeprecationWarning, match=keyword):
        assert _generate(tmp_path / "hint", hints={key: value}) == expected


def test_keyword_wins_over_a_deprecated_hint(tmp_path):
    expected = _generate(tmp_path / "keyword", conductor_boundary="pec")
    with pytest.warns(DeprecationWarning, match="conductor_boundary"):
        config = _generate(
            tmp_path / "both",
            conductor_boundary="pec",
            hints={"_conductor_boundary": "conductivity"},
        )
    assert config == expected


# keyword, its off value, the deprecated hints key, and the hint's on value
OFF_OPTIONS = [
    ("impedance_boundaries", [], "_impedance_boundaries", IMPEDANCE),
    ("metallic_boundaries", False, "_metallic_boundaries", True),
    ("conductor_boundary", "conductivity", "_conductor_boundary", "pec"),
    ("mode_postprocessing", {}, "_mode_postprocessing", POSTPROCESSING),
]


@pytest.mark.parametrize(
    ("keyword", "off", "key", "on"),
    OFF_OPTIONS,
    ids=[option[0] for option in OFF_OPTIONS],
)
def test_explicit_off_keyword_wins_over_a_deprecated_hint(
    tmp_path, keyword, off, key, on
):
    """A falsy or default keyword passed explicitly is not "not passed"."""
    expected = _generate(tmp_path / "baseline")
    with pytest.warns(DeprecationWarning, match=keyword):
        config = _generate(tmp_path / "both", **{keyword: off}, hints={key: on})
    assert config == expected


def test_public_hints_are_merged_and_unknown_private_keys_dropped(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        config = json.loads(
            _generate(tmp_path / "out", hints={"Extra": {"a": 1}, "_unknown": True})
        )
    assert config["Extra"] == {"a": 1}
    assert "_unknown" not in config


def _mesh_result(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        groups=_groups(),
        port_info=[],
        output_dir=output_dir,
        model_name="palace",
        fmax=50e9,
        periodic_axis=None,
        periodic_translation=None,
        mesh_stats={},
        config_path=None,
        metadata=None,
    )


def _write_config(monkeypatch, output_dir: Path, **options) -> str:
    monkeypatch.setattr(
        "gsim.palace.mesh.config_generator.write_metadata", lambda *_: {}
    )
    output_dir.mkdir()
    return write_config(
        _mesh_result(output_dir),
        _stack(),
        [],
        simulation_type="boundarymode",
        boundary_mode_config=BoundaryModeConfig(freq=50e9, num_modes=2, save=1),
        absorbing_boundary=False,
        **options,
    ).read_text()


def test_public_write_config_forwards_the_keywords(tmp_path, monkeypatch):
    config = _write_config(monkeypatch, tmp_path / "out", metallic_boundaries=True)
    assert 8 in json.loads(config)["Boundaries"]["PEC"]["Attributes"]


def test_write_config_warning_names_the_caller(tmp_path, monkeypatch):
    """Python hides a DeprecationWarning unless it is attributed to user code."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write_config(
            monkeypatch, tmp_path / "out", hints={"_metallic_boundaries": True}
        )
    deprecations = [w for w in caught if w.category is DeprecationWarning]
    assert [w.filename for w in deprecations if "hints" in str(w.message)] == [__file__]


def _captured_options(sim, tmp_path, monkeypatch) -> dict:
    captured = {}

    def fake_write(**kwargs):
        captured.update(kwargs)
        return tmp_path / "config.json"

    monkeypatch.setattr("gsim.palace.mesh.generator.write_config", fake_write)
    sim.set_output_dir(tmp_path)
    sim._last_mesh_result = SimpleNamespace(groups={"volumes": {}}, mesh_stats={})
    monkeypatch.setattr(sim, "_resolve_stack", lambda: None)
    sim.write_config(photonic=True, validate_mesh=False)
    return captured


def test_boundary_mode_settings_reach_the_generator(tmp_path, monkeypatch):
    sim = BoundaryModeSim(metallic_boundaries=True, conductor_boundary="pec")
    sim.set_cross_section("x=0")
    options = _captured_options(sim, tmp_path, monkeypatch)
    assert options["metallic_boundaries"] is True
    assert options["conductor_boundary"] == "pec"
    assert options["cross_section_axis"] == "x"
    assert "hints" not in options


def test_any_problem_type_can_declare_a_metallic_wall(tmp_path, monkeypatch):
    """The wall is read from any sim class, not only the boundary-mode one."""

    class ShieldedSim(DrivenSim):
        metallic_boundaries: bool = True

    options = _captured_options(ShieldedSim(), tmp_path, monkeypatch)
    assert options["metallic_boundaries"] is True
    assert "conductor_boundary" not in options


def test_impedance_boundaries_reach_the_generator(tmp_path, monkeypatch):
    sim = DrivenSim()
    sim._impedance_boundaries.extend(IMPEDANCE)
    options = _captured_options(sim, tmp_path, monkeypatch)
    assert options["impedance_boundaries"] == IMPEDANCE
