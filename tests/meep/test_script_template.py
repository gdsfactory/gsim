"""Smoke tests that the generated MEEP runner script is valid Python."""

from __future__ import annotations

import ast
from types import SimpleNamespace

from gsim.meep.script import _MEEP_RUNNER_TEMPLATE


def test_runner_template_parses():
    """Generated runner must be valid Python at import time."""
    ast.parse(_MEEP_RUNNER_TEMPLATE)


def test_runner_template_contains_xz_branch():
    """XZ 2D branch should be present in the runner."""
    assert 'plane = config.get("plane", "xy")' in _MEEP_RUNNER_TEMPLATE
    assert 'is_xz = plane == "xz"' in _MEEP_RUNNER_TEMPLATE


def test_runner_has_xz_geometry_path():
    """Inlined cross-section cutter and XZ geometry builder exist in the runner."""
    assert "_build_geometry_xz" in _MEEP_RUNNER_TEMPLATE
    assert "extract_xz_rectangles_runner" in _MEEP_RUNNER_TEMPLATE


def test_runner_has_fiber_source_path():
    """Gaussian-beam fiber source builder is present in the runner."""
    assert "_build_fiber_source" in _MEEP_RUNNER_TEMPLATE
    assert "GaussianBeamSource" in _MEEP_RUNNER_TEMPLATE


def test_runner_builds_lorentz_and_drude_materials():
    """Serialized card terms must reach the corresponding MEEP classes."""

    class FakeObject:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeVector3:
        __slots__ = ("values",)

        def __init__(self, *values):
            self.values = values

    class FakeLorentzian(FakeObject):
        pass

    class FakeDrude(FakeObject):
        pass

    fake_meep = SimpleNamespace(
        Vector3=FakeVector3,
        Medium=FakeObject,
        FreqRange=FakeObject,
        LorentzianSusceptibility=FakeLorentzian,
        DrudeSusceptibility=FakeDrude,
    )
    build_materials = _extract_runner_func("build_materials", mp=fake_meep)
    config = {
        "materials": {
            "mixed": {
                "epsilon_diag": [2.0, 2.0, 2.0],
                "epsilon_susceptibilities": [
                    {
                        "kind": "lorentzian",
                        "frequency": 1.0,
                        "gamma": 0.1,
                        "sigma": 2.0,
                    },
                    {
                        "kind": "drude",
                        "frequency": 3.0,
                        "gamma": 0.2,
                        "sigma": 1.0,
                    },
                ],
                "valid_freq_range": [0.5, 2.0],
            }
        }
    }

    medium = build_materials(config)["mixed"]

    terms = medium.kwargs["E_susceptibilities"]
    assert medium.kwargs["epsilon_diag"].values == (2.0, 2.0, 2.0)
    assert isinstance(terms[0], FakeLorentzian)
    assert isinstance(terms[1], FakeDrude)
    assert terms[0].kwargs["sigma"] == 2.0
    assert terms[1].kwargs["frequency"] == 3.0
    assert medium.kwargs["valid_freq_range"].kwargs == {"min": 0.5, "max": 2.0}


def test_resolved_z_bounds_are_identical_in_3d_and_xz():
    """Canonical bounds must bypass the legacy XZ margin path."""
    resolve_z_cell = _extract_runner_func("resolve_z_cell")
    domain = {
        "dpml": 1.0,
        "z_bounds": [-1.0, 3.0],
        # Deliberately large legacy values: canonical bounds must win.
        "margin_z_low": 10.0,
        "margin_z_high": 20.0,
    }

    assert resolve_z_cell(domain, -0.5, 0.22, True, False) == (6.0, 1.0)
    assert resolve_z_cell(domain, -0.5, 0.22, False, True) == (6.0, 1.0)


def test_legacy_xz_margins_are_applied_once():
    """Old configs retain one margin expansion, not the new canonical path."""
    resolve_z_cell = _extract_runner_func("resolve_z_cell")
    domain = {"dpml": 1.0, "margin_z_low": 0.5, "margin_z_high": 1.0}

    # Inner bounds are [-1, 3], outer bounds are [-2, 4].
    assert resolve_z_cell(domain, -0.5, 2.0, False, True) == (6.0, 1.0)


def _extract_runner_func(name: str, **namespace):
    """Exec a single pure-Python helper from the runner template in isolation.

    The runner template can't be imported wholesale (it imports meep), so we
    pull out one FunctionDef by name and exec just that, with numpy available.
    """
    import numpy as np

    tree = ast.parse(_MEEP_RUNNER_TEMPLATE)
    func_node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )
    assert func_node is not None, f"{name} not found in runner template"
    ns: dict = {"np": np, **namespace}
    exec(compile(ast.Module([func_node], []), "<runner>", "exec"), ns)  # noqa: S102
    return ns[name]


def test_core_z_center_uses_core_layer_not_stack_midpoint():
    """Animation/diagnostic XY slices must cut the waveguide core.

    Regression: for an asymmetric SOI stack (thick BOX below, metal above)
    the full-stack midpoint lands in the oxide, well below the core, so the
    animation showed no waveguide and no guided field. The slice z must be
    the highest-permittivity (core) layer center instead.
    """
    core_z_center = _extract_runner_func("_core_z_center")

    config = {
        "layer_stack": [
            {"layer_name": "box", "zmin": -3.0, "zmax": 0.0, "material": "sio2"},
            {"layer_name": "clad", "zmin": 0.0, "zmax": 1.8, "material": "sio2"},
            {"layer_name": "core", "zmin": 0.0, "zmax": 0.22, "material": "si"},
            {"layer_name": "metal", "zmin": 1.8, "zmax": 2.5, "material": "al"},
        ],
        "materials": {
            "si": {"epsilon_diag": [12.04, 12.04, 12.04]},
            "sio2": {"epsilon_diag": [2.085, 2.085, 2.085]},
            "al": {"epsilon_diag": [1.0, 1.0, 1.0]},
        },
    }

    # Core spans [0.0, 0.22] -> center 0.11. Full-stack midpoint is -0.25.
    assert core_z_center(config) == 0.11


def test_core_z_center_falls_back_to_midpoint_without_optical_data():
    """With no high-index layer, fall back to the full-stack midpoint."""
    core_z_center = _extract_runner_func("_core_z_center")

    config = {
        "layer_stack": [
            {"layer_name": "a", "zmin": -1.0, "zmax": 0.0, "material": "m"},
            {"layer_name": "b", "zmin": 0.0, "zmax": 3.0, "material": "m"},
        ],
        "materials": {"m": {"epsilon_diag": [1.0, 1.0, 1.0]}},
    }

    assert core_z_center(config) == 1.0
