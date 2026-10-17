"""Exercise native-runtime failures and polygon compatibility without native MEEP."""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import klayout.db as kdb
import numpy as np
import pytest

from gsim.meep import mode_solver


@pytest.mark.parametrize(
    ("level", "verbosity"),
    [(logging.WARNING, 0), (logging.INFO, 1), (logging.DEBUG, 2), (logging.NOTSET, 0)],
)
def test_lazy_runtime_import_respects_native_logging(monkeypatch, level, verbosity):
    native = SimpleNamespace(verbosity=Mock())
    monkeypatch.setitem(sys.modules, "meep", native)
    monkeypatch.setattr(
        logging.getLogger("meep.native"), "getEffectiveLevel", lambda: level
    )
    assert mode_solver._import_meep() is native
    native.verbosity.assert_called_once_with(verbosity)


def test_missing_native_runtime_has_installation_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "meep", None)
    with pytest.raises(ImportError, match="meep"):
        mode_solver._import_meep()


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_polygon_intervals_merge_overlaps_and_keep_disjoint_regions(axis, duplicate):
    shapes = [
        [(-3, -2), (3, -2), (3, 2), (1, 2), (1, -1), (-1, -1), (-1, 2), (-3, 2)],
        [(-3, -1), (-1, -1), (-1, 1), (1, 1), (2, 0), (3, 1), (3, 2), (-3, 2)],
        [(-3, -1), (-2, -1), (-2, 1), (-3, 1)],
        [(5, -1), (6, -1), (6, 1), (5, 1)],
    ]
    polygons = [
        np.array(shape)[:, ::-1] if axis == "y" else np.array(shape) for shape in shapes
    ]
    data = {(1, 0): [*polygons, [], object(), np.ones((3, 3))]}
    component = Mock()
    component.get_polygons_points.return_value = data
    component.dup.return_value.get_polygons_points.return_value = data
    if duplicate:
        component.get_polygons_points.side_effect = RuntimeError("stale component")
    function = (
        mode_solver._layer_x_intervals_at_y
        if axis == "x"
        else mode_solver._layer_y_intervals_at_x
    )
    assert function(component, SimpleNamespace(gds_layer=(1, 0)), 0) == [
        (-3, -1),
        (1, 3),
        (5, 6),
    ]


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("invalid", ["layer", "empty", "sequence", "unavailable"])
def test_polygon_queries_fail_closed_when_geometry_is_unavailable(axis, invalid):
    component = Mock()
    component.get_polygons_points.return_value = [] if invalid == "sequence" else {}
    if invalid == "unavailable":
        component.get_polygons_points.side_effect = RuntimeError("unavailable")
        component.dup.return_value.get_polygons_points.side_effect = RuntimeError(
            "unavailable"
        )
    layer = SimpleNamespace(gds_layer=None if invalid == "layer" else (1, 0))
    function = (
        mode_solver._layer_x_intervals_at_y
        if axis == "x"
        else mode_solver._layer_y_intervals_at_x
    )
    assert function(component, layer, 0) == []
    assert not mode_solver._layer_has_any_polygon(component, layer)


@pytest.mark.parametrize(
    "polygon",
    [
        np.array([[0, 0], [2, 0], [2, 3], [0, 3]]),
        ((0, 0), (2, 0), (2, 3), (0, 3)),
        kdb.Polygon(kdb.Box(0, 0, 2, 3)),
    ],
)
def test_polygon_representations_preserve_area(polygon):
    assert mode_solver._poly_to_shapely(polygon).area == 6


@pytest.mark.parametrize("polygon", [np.empty((0, 2)), kdb.Polygon(), object()])
def test_empty_or_unsupported_polygon_representations_are_ignored(polygon):
    assert mode_solver._poly_to_shapely(polygon) is None


@pytest.fixture
def filtered_context(index_context):
    stack, component = index_context
    for name, material, height in [
        ("air", "air", 1),
        ("missing", "unlisted", 1),
        ("sheet", "core", 0),
    ]:
        stack.layers[name] = stack.layers["core"].model_copy(
            update={
                "name": name,
                "material": material,
                "zmax": height,
                "thickness": height,
            }
        )
    return stack, component


@pytest.mark.parametrize("plane", ["xz", "yz"])
@pytest.mark.parametrize(
    "failure",
    ["none", "zero-fields", "field-read", "missing-velocity", "zero-velocity"],
)
def test_cross_section_solve_survives_missing_native_field_or_velocity_data(
    filtered_context, backend, plane, failure
):
    stack, component = filtered_context
    backend.zero_fields = failure == "zero-fields"
    backend.amplitude_error = failure == "field-read"
    backend.missing_velocity = failure == "missing-velocity"
    backend.group_velocity = 0 if failure == "zero-velocity" else 0.25
    result = mode_solver.solve_cross_section_mode(
        component,
        stack,
        wavelength=1.55,
        position=(0, 0),
        cross_section_plane=plane,
        x_span=4,
        y_span=4,
        z_margin=0,
        pml_thickness=0,
        background_material="clad",
        field_z_grid=np.array([0.5]),
        **{"field_x_grid" if plane == "xz" else "field_y_grid": np.array([-0.5, 0.5])},
    )
    assert result.n_eff == pytest.approx(1.8)
    assert result.n_group == (
        None if failure in {"missing-velocity", "zero-velocity"} else 4
    )
    assert bool(result.fields) == (failure not in {"zero-fields", "field-read"})
    assert backend.instances[0].resets == 1
    assert not hasattr(backend.instances[0], "boundary_layers")


@pytest.mark.parametrize("kind", ["single", "bands", "wavelengths"])
def test_slab_variants_ignore_air_missing_materials_and_zero_height_layers(
    filtered_context, backend, kind
):
    stack, _ = filtered_context
    if kind == "single":
        result = mode_solver.solve_slab_mode(stack, 1.55, z_margin=0)
        assert result.n_eff == pytest.approx(1.8)
    elif kind == "bands":
        result = mode_solver.solve_slab_modes(stack, 1.55, background_material="clad")
        assert list(result) == [1]
    else:
        result = mode_solver.solve_slab_wavelength_sweep(stack, [1.5, 1.6])
        assert list(result) == [1.5, 1.6]
    assert all(instance.resets == 1 for instance in backend.instances)


@pytest.mark.parametrize(
    "function",
    [
        mode_solver.solve_slab_mode,
        mode_solver.solve_slab_modes,
        mode_solver.solve_slab_wavelength_sweep,
    ],
)
@pytest.mark.usefixtures("backend")
def test_slab_variants_reject_missing_stack(function):
    with pytest.raises(ValueError, match="LayerStack"):
        function(
            None,
            [1.55] if function is mode_solver.solve_slab_wavelength_sweep else 1.55,
        )
