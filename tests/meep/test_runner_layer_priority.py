"""Check overlap precedence in generated geometry without a Meep installation."""

import ast
import logging
import math
from types import SimpleNamespace

import pytest
from shapely.geometry import box

from gsim.meep.script import _MEEP_RUNNER_TEMPLATE


def _runner():
    """Load the geometry functions and replace only the external Meep objects."""
    names = {
        "order_physical_layers",
        "build_geometry",
        "_build_geometry_xz",
        "extract_xz_rectangles_runner",
        "_xz_runner_line_intervals",
    }
    tree = ast.parse(_MEEP_RUNNER_TEMPLATE)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    polygon = box(-1, -1, 1, 1)
    ns = dict(
        math=math,
        logger=logging.getLogger(__name__),
        mp=SimpleNamespace(
            Medium=object,
            Vector3=lambda *values: values,
            Prism=lambda **values: SimpleNamespace(**values),
            Block=lambda **values: SimpleNamespace(**values),
            inf=float("inf"),
        ),
        load_gds_component=lambda _: object(),
        extract_layer_polygons=lambda *args, **kwargs: [polygon],
        _xz_runner_layer_polys=lambda *args: [polygon],
    )
    exec(compile(ast.Module(definitions, []), "<runner>", "exec"), ns)  # noqa: S102
    return ns


@pytest.mark.parametrize("plane", ["3d", "xy", "xz"])
@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_material_priority_is_independent_of_declaration_order(
    plane, reverse
):
    runner = _runner()
    layers = [
        dict(
            layer_name=name,
            material=name,
            mesh_order=priority,
            gds_layer=[index, 0],
            zmin=0,
            zmax=1,
            sidewall_angle=0,
        )
        for index, (name, priority) in enumerate((("core", 1), ("cladding", 10)), 1)
    ]
    if reverse:
        layers.reverse()
    config = dict(
        gds_filename="unused.gds",
        is_3d=plane == "3d",
        plane=plane,
        accuracy={"simplify_tol": 0},
        layer_stack=layers,
        y_cut=0,
    )
    geometry, _ = runner["build_geometry"](config, {"core": 12, "cladding": 2})
    # Both objects cover the origin. Meep assigns the last object's medium.
    assert geometry[-1].material == 12
    assert geometry[0].material == 2


def test_legacy_and_equal_priority_entries_keep_relative_order():
    order = _runner()["order_physical_layers"]
    legacy = [{"layer_name": "first"}, {"layer_name": "second", "mesh_order": None}]
    assert order(legacy) == legacy
    tied = [{"layer_name": name, "mesh_order": 2} for name in ("first", "second")]
    assert order(tied) == tied
    assert order([tied[0], *legacy]) == [*legacy, tied[0]]
