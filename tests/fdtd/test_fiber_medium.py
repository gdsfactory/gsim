"""Infer analytic receiver media from generic exported material volumes."""

from __future__ import annotations

import json

import pytest
from gdsfactory.technology import LayerLevel, LogicalLayer

from gsim import fdtd


def clad_component(pdk, name):
    """Draw the dielectric footprint across the full physical domain."""
    pdk.activate()
    component = pdk.get_component(name).copy()
    component.add_polygon([(0, -0.5), (2, -0.5), (2, 0.5), (0, 0.5)], layer=(100, 0))
    return component


@pytest.fixture
def embedded_simulation(fdtd_pdk_module):
    """Embed a straight core in a dielectric distinct from the ambient."""
    stack = fdtd_pdk_module.PDK.layer_stack
    stack.layers["core"].sidewall_angle = 0
    stack.layers["cladding"] = LayerLevel(
        layer=LogicalLayer(layer=(100, 0)),
        zmin=-0.5,
        thickness=1.7,
        material="cladding",
        background=True,
        mesh_order=10,
    )
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.materials(
        background="SiO2", overrides={"Si": 3.4, "cladding": 1.6, "SiO2": 1.2}
    )
    simulation.domain(x_bounds=(0, 2), y_bounds=(-0.5, 0.5), z_bounds=(-0.5, 1.2))
    simulation.source(port="o1")
    simulation.geometry(
        clad_component(fdtd_pdk_module.PDK, "straight"), mesh_size_nm=750
    )
    return simulation


def add_receiver(
    simulation, *, center=(1, 0, 0.6), size=(1, 0.5, 0), normal="+z", index=None
):
    """Add an analytic receiver without changing its physical environment."""
    return simulation.monitors.add_plane(
        "receiver",
        center_um=center,
        size_um=size,
        normal=normal,
        fiber_mode=fdtd.FiberMode(
            propagation_direction=(0, 0, 1),
            e_polarization=(0, 1, 0),
            focal_point_um=(1, 0, 0.4),
            waist_radius_um=0.2,
            refractive_index=index,
        ),
    )


def exported_document(simulation, directory):
    """Read public solver inputs after the real local exporter runs."""
    artifacts = simulation.write(directory)
    return json.loads(artifacts.config_path.read_text())


@pytest.mark.parametrize(
    ("center", "size", "normal"),
    [
        ((1, 0, 0.6), (1, 0.5, 0), "+z"),
        ((1, 0, 0.6), (1, 0.5, 0), "-z"),
        ((1, 0.4, 0.6), (0, 0.1, 0.2), "+x"),
        ((1, 0.4, 0.6), (1, 0, 0.2), "-y"),
        ((1, 0, 1.2), (1, 0.5, 0), "+z"),
    ],
)
def test_receiver_uses_its_material_instead_of_ambient(
    tmp_path, embedded_simulation, center, size, normal
):
    monitor = add_receiver(embedded_simulation, center=center, size=size, normal=normal)

    document = exported_document(embedded_simulation, tmp_path)

    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == 1.6
    assert document["background_refractive_index"] == 1.2
    assert document["materials"]["Si"]["refractive_index"] == 3.4
    assert monitor.fiber_mode.refractive_index is None


def test_receiver_explicit_index_is_preserved(tmp_path, embedded_simulation):
    add_receiver(embedded_simulation, index=1.7)

    document = exported_document(embedded_simulation, tmp_path)

    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == 1.7


def test_receiver_respects_material_priority(tmp_path, embedded_simulation):
    add_receiver(embedded_simulation, center=(1, 0, 0.11), size=(1, 0.2, 0))

    document = exported_document(embedded_simulation, tmp_path)

    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == 3.4


@pytest.mark.parametrize(
    ("center", "size"),
    [((1, 0, 0.11), (1, 0.8, 0)), ((1, 0, 0.22), (1, 0.2, 0))],
)
def test_ambiguous_receiver_warns_and_keeps_legacy_fallback(
    tmp_path, embedded_simulation, center, size
):
    add_receiver(embedded_simulation, center=center, size=size)

    with pytest.warns(UserWarning, match="receiver.*refractive_index"):
        document = exported_document(embedded_simulation, tmp_path)

    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == 1.2


def test_explicit_index_handles_a_mixed_receiver(tmp_path, embedded_simulation):
    add_receiver(embedded_simulation, center=(1, 0, 0.11), size=(1, 0.8, 0), index=1.7)

    document = exported_document(embedded_simulation, tmp_path)

    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == 1.7


def test_receiver_infers_from_the_cropped_aperture(tmp_path, embedded_simulation):
    add_receiver(embedded_simulation, size=(4, 4, 0))

    with pytest.warns(UserWarning, match="cropped to the physical"):
        document = exported_document(embedded_simulation, tmp_path)

    receiver = document["monitors"][0]
    assert receiver["region_min"] == [0, -500, 600]
    assert receiver["region_max"] == [2000, 500, 600]
    assert receiver["fiber_mode"]["refractive_index"] == 1.6


def test_vertical_port_keeps_implicit_beam_and_receiver_in_the_same_medium(
    tmp_path, embedded_simulation
):
    embedded_simulation.source(port="o2")
    embedded_simulation.geometry(
        clad_component(embedded_simulation.pdk.PDK, "vertical_coupler"),
        mesh_size_nm=750,
    )

    document = exported_document(embedded_simulation, tmp_path)

    assert document["monitors"][0]["name"] == "o2"
    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == 1.6
    assert document["excitation"]["gaussian_beam"]["refractive_index"] == 1.6


def test_vertical_port_explicit_medium_is_preserved(tmp_path, embedded_simulation):
    embedded_simulation.source(port="o2", vertical_refractive_index=1.75)
    embedded_simulation.geometry(
        clad_component(embedded_simulation.pdk.PDK, "vertical_coupler"),
        mesh_size_nm=750,
    )

    document = exported_document(embedded_simulation, tmp_path)

    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == 1.75
    assert document["excitation"]["gaussian_beam"]["refractive_index"] == 1.75


def test_uncovered_receiver_uses_the_actual_background(tmp_path, embedded_simulation):
    embedded_simulation.domain(z_bounds=(-0.5, 1.5))
    add_receiver(embedded_simulation, center=(1, 0, 1.4))

    document = exported_document(embedded_simulation, tmp_path)

    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == 1.2


@pytest.mark.parametrize(
    ("y", "z", "expected_index"), [(0.257, 0.03, 3.4), (0.245, 0.19, 1.6)]
)
def test_receiver_respects_exported_sidewalls(
    tmp_path, embedded_simulation, y, z, expected_index
):
    embedded_simulation.pdk.PDK.layer_stack.layers["core"].sidewall_angle = 10
    add_receiver(embedded_simulation, center=(1, y, z), size=(0.5, 0.004, 0))

    document = exported_document(embedded_simulation, tmp_path)

    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == expected_index


def test_receiver_inside_a_material_hole_uses_background(tmp_path, embedded_simulation):
    component = embedded_simulation.geometry.component.copy()
    component.add_polygon(
        [(0.75, -0.15), (1.25, -0.15), (1.25, 0.15), (0.75, 0.15)], layer=(50, 0)
    )
    cladding = embedded_simulation.pdk.PDK.layer_stack.layers["cladding"]
    cladding.background_exclude_layers = ((50, 0),)
    embedded_simulation.geometry(component)
    add_receiver(embedded_simulation, size=(0.2, 0.1, 0))

    document = exported_document(embedded_simulation, tmp_path)

    assert document["monitors"][0]["fiber_mode"]["refractive_index"] == 1.2
