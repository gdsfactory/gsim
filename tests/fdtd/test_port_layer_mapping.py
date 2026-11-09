"""Explicit virtual-port mappings preserve complete physical cross-sections."""

import json

import gdsfactory as gf
import meshio
import numpy as np
import pytest
from gdsfactory.technology import LogicalLayer

from gsim import fdtd
from gsim.common.pdk import UnsupportedPortError, resolve_passive_pcell
from gsim.fdtd.api import Geometry


@pytest.fixture
def mapped_stack(fdtd_pdk_module):
    """Use three physical bands and a separate virtual routing layer."""
    module = fdtd_pdk_module
    module.PDK.activate()
    original = module.PDK.layer_stack.layers["core"]
    module.PDK.layer_stack.layers = {
        name: original.model_copy(
            update={
                "zmin": zmin,
                "thickness": thickness,
                "mesh_order": order,
                "sidewall_angle": 0,
            }
        )
        for name, zmin, thickness, order in (
            ("base", 0, 0.08, 3),
            ("middle", 0.08, 0.12, 2),
            ("cap", 0.2, 0.2, 1),
        )
    }
    component = gf.Component()
    component.add_polygon([(0, -0.3), (2, -0.3), (2, 0.3), (0, 0.3)], layer=(1, 0))
    component.add_polygon(
        [(0, -0.2), (0.1, -0.2), (0.1, 0.2), (0, 0.2)], layer=(100, 0)
    )
    for name, x, angle in (("o1", 0, 180), ("o2", 2, 0)):
        component.add_port(
            name, center=(x, 0), width=0.6, orientation=angle, layer=(100, 0)
        )
    return module, component


def _port_geometry(artifacts):
    mesh = meshio.read(artifacts.mesh_path)
    tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    vertices = mesh.points[
        mesh.cells_dict["triangle"][tags == mesh.field_data["port_o1"][0]]
    ]
    area = (
        np.linalg.norm(
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=1,
        ).sum()
        / 2
    )
    return vertices, area, set(mesh.field_data)


@pytest.mark.parametrize(
    ("selection", "expected_keys", "active_bounds"),
    [
        ("middle", ("middle",), (0.08, 0.2)),
        (("middle",), ("middle",), (0.08, 0.2)),
        (("base", "middle"), ("base", "middle"), (0, 0.2)),
        (("middle", "base"), ("middle", "base"), (0, 0.2)),
        (("cap", "base", "middle"), ("cap", "middle", "base"), (0, 0.4)),
    ],
)
@pytest.mark.parametrize("crop", [None, (0.03, 0.35)])
def test_explicit_mapping_meshes_exactly_the_selected_height(
    tmp_path, mapped_stack, selection, expected_keys, active_bounds, crop
):
    module, component = mapped_stack
    simulation = fdtd.Simulation(pdk=module)
    simulation.domain(z_bounds=crop, clip_geometry=True)
    simulation.source(port="o1")
    resolved = simulation.geometry(
        component, port_layer_map={(100, 0): selection}, mesh_size_nm=700
    )
    low, high = active_bounds
    if crop is not None:
        low, high = max(low, crop[0]), min(high, crop[1])
    port = resolved.ports["o1"]
    assert port.physical_layer_keys == expected_keys
    assert port.layer_key == expected_keys[0]
    assert port.center[2] == pytest.approx((low + high) / 2)
    assert port.z_bounds == pytest.approx((low, high))
    simulation.add_monitors(xy=False, margin_um=0)
    monitor = simulation.monitors["port_o1"]
    assert monitor.center_um[2] == pytest.approx((low + high) / 2)
    assert monitor.size_um == pytest.approx((0, 0.6, high - low))
    artifacts = simulation.write(tmp_path)
    vertices, area, groups = _port_geometry(artifacts)
    assert vertices[:, :, 2].min() == pytest.approx(low * 1000)
    assert vertices[:, :, 2].max() == pytest.approx(high * 1000)
    assert area == pytest.approx(600 * (high - low) * 1000)
    assert groups == {"background", "base", "middle", "cap", "port_o1", "port_o2"}
    config = json.loads(artifacts.config_path.read_text())
    assert config["geometry"]["ports"]["o1"]["layer"] == expected_keys[0]


def test_constructor_normalizes_band_lists_and_preserves_mapping_round_trip(
    tmp_path, mapped_stack
):
    module, component = mapped_stack
    simulation = fdtd.Simulation(
        pdk=module,
        source={"port": "o1"},
        geometry={
            "component": component,
            "port_layer_map": {(100, 0): ["cap", "base", "middle"]},
        },
    )
    expected = {(100, 0): ("cap", "base", "middle")}
    assert simulation.geometry.port_layer_map == expected
    assert (
        Geometry.model_validate(simulation.geometry.model_dump()).port_layer_map
        == expected
    )
    vertices, area, _ = _port_geometry(simulation.write(tmp_path))
    assert vertices[:, :, 2].max() == pytest.approx(400)
    assert area == pytest.approx(600 * 400)


def test_explicit_mapping_supports_reverse_extrusions(tmp_path, mapped_stack):
    module, component = mapped_stack
    for level in module.PDK.layer_stack.layers.values():
        level.zmin += level.thickness
        level.thickness = -level.thickness
    simulation = fdtd.Simulation(pdk=module)
    simulation.source(port="o1")
    simulation.geometry(component, port_layer_map={(100, 0): ("base", "middle", "cap")})
    assert simulation.resolved.ports["o1"].z_bounds == pytest.approx((0, 0.4))
    assert _port_geometry(simulation.write(tmp_path))[1] == pytest.approx(600 * 400)


def test_single_band_tuple_and_legacy_string_write_identical_artifacts(
    tmp_path, mapped_stack
):
    module, component = mapped_stack
    artifacts = []
    for index, selection in enumerate(("middle", ("middle",))):
        simulation = fdtd.Simulation(pdk=module)
        simulation.source(port="o1")
        simulation.geometry(component, port_layer_map={(100, 0): selection})
        artifacts.append(simulation.write(tmp_path / str(index)))
    assert artifacts[0].mesh_path.read_bytes() == artifacts[1].mesh_path.read_bytes()
    assert (
        artifacts[0].config_path.read_bytes() == artifacts[1].config_path.read_bytes()
    )


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ((), "nonempty"),
        (("base", "base"), "duplicate"),
        (("base", ""), "nonempty"),
        (("base", 1), "nonempty"),
        (("base", "missing"), "has no resolved geometry"),
    ],
)
def test_invalid_band_selections_fail_during_resolution(
    mapped_stack, selection, message
):
    module, component = mapped_stack
    with pytest.raises(UnsupportedPortError, match=message):
        resolve_passive_pcell(
            component, pdk=module, port_layer_map={(100, 0): selection}
        )


@pytest.mark.parametrize("invalid", ["gap", "overlap", "material", "branch"])
def test_incompatible_selected_bands_are_rejected(mapped_stack, invalid):
    module, component = mapped_stack
    middle = module.PDK.layer_stack.layers["middle"]
    selection = ("base", "middle")
    if invalid == "gap":
        middle.zmin += 0.01
    elif invalid == "overlap":
        middle.zmin -= 0.01
    elif invalid == "material":
        middle.material = "SiO2"
    else:
        module.PDK.layer_stack.layers["branch"] = middle.model_copy(
            update={"thickness": 0.2}
        )
        selection = ("base", "middle", "branch")
    with pytest.raises(
        UnsupportedPortError,
        match="abutting, non-overlapping bands of the same material",
    ):
        resolve_passive_pcell(
            component, pdk=module, port_layer_map={(100, 0): selection}
        )


def test_each_selected_band_must_cover_the_port(mapped_stack):
    module, component = mapped_stack
    component.add_polygon(
        [(0.5, -0.3), (1.5, -0.3), (1.5, 0.3), (0.5, 0.3)], layer=(2, 0)
    )
    module.PDK.layer_stack.layers["middle"].layer = LogicalLayer(layer=(2, 0))
    with pytest.raises(UnsupportedPortError, match="outside resolved geometry"):
        resolve_passive_pcell(
            component, pdk=module, port_layer_map={(100, 0): ("base", "middle")}
        )


def test_selected_band_outside_crop_is_not_silently_dropped(mapped_stack):
    module, component = mapped_stack
    with pytest.raises(UnsupportedPortError, match=r"base.*has no resolved geometry"):
        resolve_passive_pcell(
            component,
            pdk=module,
            port_layer_map={(100, 0): ("base", "middle")},
            z_bounds=(0.1, 0.35),
        )
