"""Material names match ignoring case on the femwell path (follow-up to #543).

``epsilon_by_region`` is what assigns a permittivity to every mesh region
of a femwell solve, so these tests read its answer directly: no mode solve
and no femwell runtime are needed.
"""

from __future__ import annotations

import warnings

import meshio
import numpy as np
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.common.stack.materials import MaterialProperties
from gsim.femwell.adapter import epsilon_by_region

WL_UM = 1.55
F_RF = 50e9

STACK_EPS = 2.0
CLAD_EPS = 2.085


def _two_region_mesh(tmp_path):
    """Unit square split into two triangle groups 'core' and 'clad'."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = [
        ("triangle", np.array([[0, 1, 2]])),
        ("triangle", np.array([[0, 2, 3]])),
    ]
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data={
            "gmsh:physical": [np.array([1]), np.array([2])],
            "gmsh:geometrical": [np.array([1]), np.array([2])],
        },
        field_data={"core": np.array([1, 2]), "clad": np.array([2, 2])},
    )
    path = tmp_path / "two_region.msh"
    meshio.write(str(path), mesh, file_format="gmsh22", binary=False)
    return path


def _stack(*, layer_material="Polymer", stack_key="Polymer", extra=None):
    """A 'core' layer of ``layer_material``; the stack describes ``stack_key``.

    ``extra`` adds further stack entries (e.g. a case variant of ``stack_key``).
    """
    stack = LayerStack(pdk_name="test", units="um")
    stack.layers["core"] = Layer(
        name="core",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=0.22,
        thickness=0.22,
        material=layer_material,
        layer_type="dielectric",
        mesh_resolution="fine",
    )
    stack.materials[stack_key] = {"permittivity": STACK_EPS}
    stack.materials["clad"] = {"permittivity": CLAD_EPS}
    stack.materials.update(extra or {})
    return stack


def _eps(path, stack, **kwargs):
    return epsilon_by_region(path, stack, wavelength_um=WL_UM, **kwargs)


def test_stack_value_without_override(tmp_path):
    eps = _eps(_two_region_mesh(tmp_path), _stack())
    assert eps["core"].real == pytest.approx(STACK_EPS)


@pytest.mark.parametrize("spelling", ["Polymer", "polymer", "POLYMER", "  polymer "])
def test_override_reaches_layer_whatever_its_case(tmp_path, spelling):
    path = _two_region_mesh(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a matching override must not warn
        eps = _eps(
            path,
            _stack(),
            overrides={spelling: MaterialProperties(permittivity=9.0)},
        )
    assert eps["core"].real == pytest.approx(9.0)
    assert eps["clad"].real == pytest.approx(CLAD_EPS)


def test_override_reaches_layer_on_the_rf_path(tmp_path):
    path = _two_region_mesh(tmp_path)
    eps = epsilon_by_region(
        path,
        _stack(),
        frequency_hz=F_RF,
        overrides={"polymer": MaterialProperties(permittivity=9.0, conductivity=1e3)},
    )
    assert eps["core"].real == pytest.approx(9.0)
    assert eps["core"].imag < 0.0  # exp(+i omega t): the override's sigma acts


@pytest.mark.parametrize(
    ("layer_material", "stack_key"),
    [("Polymer", "polymer"), ("polymer", "POLYMER"), ("POLYMER", "Polymer")],
)
def test_stack_material_found_whatever_its_case(tmp_path, layer_material, stack_key):
    """A layer's material and the stack's entry for it may be spelled apart."""
    path = _two_region_mesh(tmp_path)
    stack = _stack(layer_material=layer_material, stack_key=stack_key)
    assert _eps(path, stack)["core"].real == pytest.approx(STACK_EPS)
    overrides = {"Polymer": MaterialProperties(permittivity=9.0)}
    eps = _eps(path, stack, overrides=overrides)
    assert eps["core"].real == pytest.approx(9.0)


@pytest.mark.parametrize(
    ("layer_material", "first", "second"),
    [
        ("Polymer", "Polymer", "polymer"),
        ("Polymer", "polymer", "Polymer"),
        ("polymer", "Polymer", "polymer"),
        ("polymer", "polymer", "Polymer"),
    ],
)
def test_exact_stack_key_wins_over_case_variant(
    tmp_path, layer_material, first, second
):
    """Stack keys differing only in case keep their own values, in any order."""
    path = _two_region_mesh(tmp_path)
    values = {"Polymer": 3.0, "polymer": 4.0}
    stack = _stack(
        layer_material=layer_material,
        stack_key=first,
        extra={second: {"permittivity": values[second]}},
    )
    stack.materials[first] = {"permittivity": values[first]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eps = _eps(path, stack)
    assert eps["core"].real == pytest.approx(values[layer_material])


def test_override_beats_exact_stack_key(tmp_path):
    path = _two_region_mesh(tmp_path)
    stack = _stack(extra={"polymer": {"permittivity": 4.0}})
    eps = _eps(path, stack, overrides={"POLYMER": MaterialProperties(permittivity=9.0)})
    assert eps["core"].real == pytest.approx(9.0)


def test_whitespace_in_layer_material_matches_override(tmp_path):
    """A layer material ' Polymer ' takes the override and does not warn."""
    path = _two_region_mesh(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eps = _eps(
            path,
            _stack(layer_material=" Polymer "),
            overrides={"polymer": MaterialProperties(permittivity=9.0)},
        )
    assert eps["core"].real == pytest.approx(9.0)


@pytest.mark.parametrize("order", [("polymer", "POLYMER"), ("POLYMER", "polymer")])
def test_last_of_two_overrides_differing_in_case_wins(tmp_path, order):
    path = _two_region_mesh(tmp_path)
    first, last = order
    overrides = {
        first: MaterialProperties(permittivity=5.0),
        last: MaterialProperties(permittivity=7.0),
    }
    with pytest.warns(UserWarning, match="Duplicate material entries") as record:
        eps = _eps(path, _stack(), overrides=overrides)
    assert len(record) == 1
    assert eps["core"].real == pytest.approx(7.0)


def test_unmatched_override_warns_once_with_available_names(tmp_path):
    path = _two_region_mesh(tmp_path)
    with pytest.warns(UserWarning, match="do not match") as record:
        eps = _eps(
            path,
            _stack(),
            overrides={"Alumnum": MaterialProperties(permittivity=9.0)},
        )
    assert len(record) == 1
    message = str(record[0].message)
    assert "alumnum" in message
    assert "Polymer" in message
    assert "clad" in message
    # Nothing was overridden.
    assert eps["core"].real == pytest.approx(STACK_EPS)
    assert eps["clad"].real == pytest.approx(CLAD_EPS)


def test_no_overrides_warns_nothing(tmp_path):
    path = _two_region_mesh(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _eps(path, _stack())
        _eps(path, _stack(), overrides={})
