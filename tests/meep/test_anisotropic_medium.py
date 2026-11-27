"""An anisotropic material reaches Meep as a tensor.

Covers the in-process solver and the written runner. ``mp.Medium(epsilon=eps[0])``
used to be built for every material, so a uniaxial crystal such as thin-film
lithium niobate was solved with its first (ordinary) component in every
direction. A stub stands in for ``meep``, so the tests need no Meep install.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from gsim.meep import mode_solver
from gsim.meep.models.config import MaterialData
from gsim.meep.script import generate_meep_mode_solver_script


class _Vector3(tuple):
    __slots__ = ()

    def __new__(cls, x, y, z):
        return super().__new__(cls, (x, y, z))


class _Medium:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_STUB_MP = SimpleNamespace(Medium=_Medium, Vector3=_Vector3)

ORDINARY, EXTRAORDINARY = 4.9, 4.6  # uniaxial, optical axis along z


def _runner_make_medium():
    """``_make_medium`` of the generated mode-solver runner, with a stub ``mp``."""
    script = generate_meep_mode_solver_script()
    match = re.search(
        r"^def _make_medium\(.*?(?=^def )", script, flags=re.DOTALL | re.MULTILINE
    )
    assert match, "the runner has no _make_medium"
    namespace = {"mp": _STUB_MP}
    exec(match.group(0), namespace)  # noqa: S102
    return namespace["_make_medium"]


@pytest.fixture
def in_process(monkeypatch):
    monkeypatch.setattr(mode_solver, "_import_meep", lambda: _STUB_MP)
    return lambda **fields: mode_solver._meep_medium(MaterialData(**fields))


@pytest.fixture
def runner():
    make_medium = _runner_make_medium()
    return lambda **fields: make_medium(fields)


@pytest.fixture(params=["in_process", "runner"])
def build(request):
    """The same assertions on both paths: ``build(**material_fields) -> Medium``."""
    return request.getfixturevalue(request.param)


def test_uniaxial_material_keeps_all_three_components(build):
    medium = build(epsilon_diag=[ORDINARY, ORDINARY, EXTRAORDINARY])
    assert medium.kwargs == {"epsilon_diag": (ORDINARY, ORDINARY, EXTRAORDINARY)}


def test_the_extraordinary_component_is_not_dropped_when_it_comes_first(build):
    medium = build(epsilon_diag=[EXTRAORDINARY, ORDINARY, ORDINARY])
    assert medium.kwargs["epsilon_diag"] == (EXTRAORDINARY, ORDINARY, ORDINARY)


def test_rotated_axes_pass_the_off_diagonal_terms(build):
    medium = build(
        epsilon_diag=[ORDINARY, ORDINARY, EXTRAORDINARY], epsilon_offdiag=[0, 0.1, 0]
    )
    assert medium.kwargs["epsilon_diag"] == (ORDINARY, ORDINARY, EXTRAORDINARY)
    assert medium.kwargs["epsilon_offdiag"] == (0, 0.1, 0)


def test_isotropic_material_is_built_as_before(build):
    assert build(epsilon_diag=[12.0, 12.0, 12.0]).kwargs == {"epsilon": 12.0}
    assert build(epsilon_diag=[12.0, 12.0, 12.0], epsilon_offdiag=[0, 0, 0]).kwargs == {
        "epsilon": 12.0
    }


def test_a_material_without_permittivity_is_vacuum_in_the_runner():
    assert _runner_make_medium()({}).kwargs == {"epsilon": 1.0}
