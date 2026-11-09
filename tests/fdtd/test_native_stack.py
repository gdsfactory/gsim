"""Native PDK bands, material cards and explicitly bounded mesh artifacts."""

import json
from types import SimpleNamespace

import gdsfactory as gf
import meshio
import numpy as np
import pytest
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer
from pdk_schema import Index, ScalarValue

from gsim import fdtd
from gsim.common.materials import GSIM_MATERIAL_CARDS
from gsim.common.pdk import LayerResolutionError, resolve_passive_pcell


def _constant_card(name, index):
    base = GSIM_MATERIAL_CARDS["SiO2"]
    assert base.optical is not None
    return base.model_copy(
        update={
            "name": name,
            "optical": base.optical.model_copy(
                update={
                    "permittivity": Index(
                        validity=None,
                        variation=None,
                        conductivity=None,
                        n=ScalarValue(value=index, unit=""),
                        k=None,
                    )
                }
            ),
        }
    )


@pytest.fixture
def native_stack():
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon([(0, -0.3), (2, -0.3), (2, 0.3), (0, 0.3)], layer=(1, 0))
    for name, x, angle in (("o1", 0, 180), ("o2", 2, 0)):
        component.add_port(
            name, center=(x, 0), width=0.6, orientation=angle, layer=(1, 0)
        )
    stack = LayerStack(
        layers={
            "lower": LayerLevel(
                layer=LogicalLayer(layer=(1, 0)), zmin=0, thickness=0.1, material="Core"
            ),
            "upper": LayerLevel(
                layer=LogicalLayer(layer=(1, 0)),
                zmin=0.1,
                thickness=0.2,
                material="Core",
            ),
            "oxide": LayerLevel(
                layer=(2, 0), zmin=-1, thickness=1, material="Oxide", background=True
            ),
            "remote": LayerLevel(
                layer=(3, 0), zmin=2, thickness=1, material="Unmodeled", background=True
            ),
        }
    )
    pdk = gf.Pdk(name="native_stack_fixture", layer_stack=stack)
    module = SimpleNamespace(
        PDK=pdk,
        MATERIAL_CARDS={
            "Core": _constant_card("Core", 2),
            "Oxide": _constant_card("Oxide", 1.5),
        },
    )
    return module, component


def _simulation(native_stack):
    module, component = native_stack
    simulation = fdtd.Simulation(pdk=module)
    simulation.domain(z_bounds=(-0.2, 0.7), clip_geometry=True)
    simulation.materials(background="Oxide")
    simulation.geometry(component, mesh_size_nm=700)
    simulation.source(port="o1", wavelength_um=1)
    return simulation


def test_native_stack_uses_cards_and_crops_without_pdk_mutation(tmp_path, native_stack):
    module, _ = native_stack
    stack = module.PDK.layer_stack
    simulation = _simulation(native_stack)
    assert set(simulation.resolved.layers) == {"lower", "upper", "oxide"}
    assert simulation.resolved.materials["Core"].source == "project"
    assert simulation.resolved.materials["Core"].refractive_index == 2
    assert simulation.resolved.ports["o1"].center[2] == pytest.approx(0.15)
    artifacts = simulation.write(tmp_path)
    config = json.loads(artifacts.config_path.read_text())
    assert config["grid"]["non_pml_bounds"]["min"][2] == -200
    assert config["grid"]["non_pml_bounds"]["max"][2] == 700
    assert config["materials"]["Core"] == {"refractive_index": 2}
    assert "Unmodeled" not in config["materials"]
    mesh = meshio.read(artifacts.mesh_path)
    assert mesh.points[:, 2].min() == pytest.approx(-200)
    assert mesh.points[:, 2].max() == pytest.approx(700)
    assert module.PDK.layer_stack is stack
    assert set(stack.layers) == {"lower", "upper", "oxide", "remote"}
    assert stack.layers["oxide"].zmin == -1
    assert stack.layers["oxide"].thickness == 1


def test_port_faces_cover_both_contiguous_material_bands(tmp_path, native_stack):
    simulation = _simulation(native_stack)
    port = simulation.resolved.ports["o1"]
    assert port.physical_layer_keys == ("lower", "upper")
    artifacts = simulation.write(tmp_path)
    config = json.loads(artifacts.config_path.read_text())
    assert config["geometry"]["ports"]["o1"]["layer"] == "lower"
    assert {
        config["geometry"]["layers"][key]["material"]
        for key in port.physical_layer_keys
    } == {config["geometry"]["layers"]["lower"]["material"]}
    mesh = meshio.read(artifacts.mesh_path)
    triangles = mesh.cells_dict["triangle"]
    tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    tag = mesh.field_data["port_o1"][0]
    vertices = mesh.points[triangles[tags == tag]]
    assert vertices[:, :, 2].min() == pytest.approx(0)
    assert vertices[:, :, 2].max() == pytest.approx(300)
    area = (
        np.linalg.norm(
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=1,
        ).sum()
        / 2
    )
    assert area == pytest.approx(600 * 300)


def test_missing_material_inside_window_still_raises(native_stack):
    module, component = native_stack
    with pytest.raises(LayerResolutionError, match="Unmodeled"):
        resolve_passive_pcell(component, pdk=module, z_bounds=(-0.2, 2.5))


def test_explicit_single_band_port_mapping_is_preserved(native_stack):
    module, component = native_stack
    resolved = resolve_passive_pcell(
        component, pdk=module, z_bounds=(-0.2, 0.7), port_layer_map={(1, 0): "lower"}
    )
    assert resolved.ports["o1"].physical_layer_keys == ("lower",)
    assert resolved.ports["o1"].center[2] == pytest.approx(0.05)


@pytest.mark.parametrize("reverse_extrusion", [False, True])
def test_films_touching_window_without_volume_are_excluded(
    native_stack, reverse_extrusion
):
    module, component = native_stack
    if reverse_extrusion:
        # Reverse extrusion has the same physical interval [2, 3].
        module.PDK.layer_stack.layers["remote"].zmin = 3
        module.PDK.layer_stack.layers["remote"].thickness = -1
    assert (
        "remote"
        not in resolve_passive_pcell(component, pdk=module, z_bounds=(-0.2, 2)).layers
    )


@pytest.mark.parametrize(("other_material", "gap"), [("Other", 0), ("Core", 0.1)])
def test_port_does_not_join_different_material_or_disconnected_band(
    native_stack, other_material, gap
):
    module, component = native_stack
    module.PDK.layer_stack.layers["upper"].material = other_material
    module.PDK.layer_stack.layers["upper"].zmin += gap
    module.MATERIAL_CARDS["Other"] = _constant_card("Other", 2.5)
    resolved = resolve_passive_pcell(component, pdk=module, z_bounds=(-0.2, 0.7))
    assert resolved.ports["o1"].physical_layer_keys == ("lower",)
    assert resolved.ports["o1"].center[2] == pytest.approx(0.05)


@pytest.mark.parametrize("bounds", [(1, 0), (1, 1), (0, float("inf"))])
def test_invalid_resolution_window_is_rejected(native_stack, bounds):
    module, component = native_stack
    with pytest.raises(LayerResolutionError, match="finite and strictly increasing"):
        resolve_passive_pcell(component, pdk=module, z_bounds=bounds)


def test_default_static_slice_stays_inside_cropped_domain(native_stack):
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    simulation = _simulation(native_stack)
    axis = simulation.plot_2d_static()
    assert axis.get_title() == "FDTD · z = 0.25 µm"


def test_cropped_full_width_films_mesh_without_overlapping_facets(
    tmp_path, native_stack
):
    module, _ = native_stack
    module.PDK.layer_stack.layers["top_fill"] = LayerLevel(
        layer=(4, 0), zmin=0.6, thickness=0.7, material="Oxide", background=True
    )
    simulation = _simulation(native_stack)
    simulation.domain(y_bounds=(-0.3, 0.3))
    artifacts = simulation.write(tmp_path)
    mesh = meshio.read(artifacts.mesh_path)
    assert mesh.points.min(axis=0) == pytest.approx([0, -300, -200])
    assert mesh.points.max(axis=0) == pytest.approx([2000, 300, 700])


def test_crop_omits_disconnected_islands_and_external_materials(tmp_path, native_stack):
    module, component = native_stack
    component.add_polygon([(0, 2), (1, 2), (1, 3), (0, 3)], layer=(1, 0))
    component.add_polygon([(3, 0), (4, 0), (4, 0.1), (3, 0.1)], layer=(5, 0))
    module.PDK.layer_stack.layers["off_axis"] = LayerLevel(
        layer=(5, 0), zmin=0.1, thickness=0.1, material="Unmodeled"
    )
    simulation = fdtd.Simulation(pdk=module)
    simulation.domain(
        x_bounds=(0, 2), y_bounds=(-0.3, 0.3), z_bounds=(-0.2, 0.7), clip_geometry=True
    )
    simulation.materials(background="Oxide")
    simulation.source(port="o1")
    simulation.geometry(component, mesh_size_nm=700)
    assert "off_axis" not in simulation.resolved.layers
    artifacts = simulation.write(tmp_path)
    mesh = meshio.read(artifacts.mesh_path)
    assert mesh.points.min(axis=0) == pytest.approx([0, -300, -200])
    assert mesh.points.max(axis=0) == pytest.approx([2000, 300, 700])


@pytest.mark.parametrize(("x", "y"), [(2.02, 0), (2.15, 0.45)])
def test_sidewall_overhang_can_enter_window_from_external_footprint(
    tmp_path, native_stack, x, y
):
    module, component = native_stack
    component.add_polygon(
        [(x, y), (x + 0.1, y), (x + 0.1, y + 0.1), (x, y + 0.1)], layer=(6, 0)
    )
    module.PDK.layer_stack.layers["overhang"] = LayerLevel(
        layer=(6, 0),
        zmin=0,
        thickness=0.2,
        material="Oxide",
        sidewall_angle=45,
        width_to_z=1,
    )
    simulation = fdtd.Simulation(pdk=module)
    simulation.domain(
        x_bounds=(0, 2), y_bounds=(-0.3, 0.3), z_bounds=(-0.2, 0.7), clip_geometry=True
    )
    simulation.materials(background="Oxide")
    simulation.source(port="o1")
    simulation.geometry(component, mesh_size_nm=700)
    assert "overhang" in simulation.resolved.layers
    artifacts = simulation.write(tmp_path)
    mesh = meshio.read(artifacts.mesh_path)
    assert "overhang" in mesh.field_data
