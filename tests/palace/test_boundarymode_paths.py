"""BoundaryMode path projection, port geometry, and config serialization."""

from __future__ import annotations

import json
from typing import cast

import gdsfactory as gf
import numpy as np
import pytest

from gsim.common import Layer, LayerStack
from gsim.palace import BoundaryModeSim


@pytest.fixture
def mode_stack():
    return LayerStack(
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=1,
                zmax=2,
                thickness=1,
                material="silicon",
                layer_type="dielectric",
            )
        }
    )


@pytest.fixture
def component():
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    comp.add_polygon([(0, 0), (4, 0), (4, 6), (0, 6)], layer=(1, 0))
    comp.add_port("unused", center=(0, 0), width=1, orientation=0, layer=(1, 0))
    comp.add_port("signal", center=(2, 3), width=2, orientation=90, layer=(1, 0))
    return comp


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ("x", [[2, 3], [5, 6]]),
        ("y", [[1, 3], [4, 6]]),
        ("z", [[1, 2], [4, 5]]),
    ],
)
def test_layout_paths_project_onto_requested_plane(axis, expected):
    assert BoundaryModeSim._project_mode_path([[1, 2, 3], [4, 5, 6]], axis) == expected


@pytest.mark.parametrize("point", [[1], [1, 2, 3, 4]])
def test_path_rejects_invalid_coordinate_count(point):
    with pytest.raises(ValueError, match=r"must have 2 .* or 3"):
        BoundaryModeSim._project_mode_path([point], "x")


def test_explicit_cpw_paths_preserve_port_order_and_current_loop(mode_stack):
    sim = BoundaryModeSim()
    current_loop = [[0, -2, 0], [0, 2, 0], [0, 2, 2], [0, -2, 0]]
    sim.add_port("first", voltage_path=[[0, 0], [1, 0]], nsamples=20)
    sim.add_cpw_port(
        "cpw",
        layer="core",
        s_width=2,
        gap_width=1,
        voltage_paths=[[[1, 0], [2, 0]], [[-1, 0], [-2, 0]]],
        current_path=current_loop,
        nsamples=50,
    )
    sim.add_port("last", voltage_path=[[0, 1], [1, 1]], nsamples=30)
    post = sim._build_boundarymode_postprocessing(mode_stack, None)
    assert post["Voltage"] == [
        {"Index": 1, "VoltagePath": [[0, 0], [1, 0]], "NSamples": 20},
        {"Index": 2, "VoltagePath": [[1, 0], [2, 0]], "NSamples": 50},
        {"Index": 3, "VoltagePath": [[-1, 0], [-2, 0]], "NSamples": 50},
        {"Index": 4, "VoltagePath": [[0, 1], [1, 1]], "NSamples": 30},
    ]
    assert post["Impedance"][0] == post["Voltage"][0]
    assert post["Impedance"][3] == post["Voltage"][3]
    for index in (1, 2):
        assert post["Impedance"][index] == {
            **post["Voltage"][index],
            "CurrentPath": [[-2, 0], [2, 0], [2, 2], [-2, 0]],
        }


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, [[3, 1.5], [1, 1.5]]),
        ({"center": (5, 7), "orientation": 180, "width": 4}, [[5, 1.5], [5, 1.5]]),
    ],
)
def test_single_port_uses_component_geometry_and_explicit_overrides(
    mode_stack,
    component,
    overrides,
    expected,
):
    sim = BoundaryModeSim()
    sim.set_geometry(component)
    sim.set_cross_section("y=3")
    sim.add_port("signal", layer="core", **overrides)
    post = sim._build_boundarymode_postprocessing(mode_stack, sim.cross_section)
    np.testing.assert_allclose(
        cast("list[list[float]]", post["Voltage"][0]["VoltagePath"]), expected
    )
    # Verify the full layout path as well: projection can hide transverse motion.
    path = sim._derive_single_port_path(sim.ports[0], mode_stack)
    expected_layout = (
        [[5, 9, 1.5], [5, 5, 1.5]] if overrides else [[3, 3, 1.5], [1, 3, 1.5]]
    )
    assert path is not None
    np.testing.assert_allclose(path, expected_layout)


def test_cpw_paths_use_component_center_and_orientation(mode_stack, component):
    sim = BoundaryModeSim()
    sim.set_geometry(component)
    sim.set_cross_section("y=3")
    sim.add_cpw_port("signal", layer="core", s_width=2, gap_width=1)
    post = sim._build_boundarymode_postprocessing(mode_stack, sim.cross_section)
    np.testing.assert_allclose(
        cast("list[list[float]]", post["Voltage"][0]["VoltagePath"]),
        [[1, 1.5], [0, 1.5]],
    )
    np.testing.assert_allclose(
        cast("list[list[float]]", post["Voltage"][1]["VoltagePath"]),
        [[3, 1.5], [4, 1.5]],
    )


@pytest.mark.parametrize(
    ("kind", "settings"),
    [
        ("single", {"layer": "core", "width": 2}),
        ("single", {"layer": "core", "center": (0, 0)}),
        ("single", {"layer": "missing", "center": (0, 0), "width": 2}),
        ("cpw", {"layer": "core"}),
        ("cpw", {"layer": "missing", "center": (0, 0)}),
        ("cpw", {"layer": "", "center": (0, 0)}),
    ],
)
def test_ports_without_resolvable_geometry_omit_postprocessing(
    mode_stack, component, kind, settings
):
    sim = BoundaryModeSim()
    sim.set_geometry(component)
    if kind == "single":
        sim.add_port("unavailable", **settings)
    else:
        sim.add_cpw_port("unavailable", s_width=2, gap_width=1, **settings)
    assert sim._build_boundarymode_postprocessing(mode_stack, None) == {}


def test_explicit_voltage_path_validates_without_a_layer():
    sim = BoundaryModeSim()
    sim.set_cross_section("x=0")
    sim.add_port("signal", voltage_path=[[0, 0], [1, 0]])
    assert not any(
        "inplane ports require" in error for error in sim.validate_config().errors
    )


def test_mode_paths_are_written_to_palace_config(tmp_path, mode_stack, component):
    sim = BoundaryModeSim()
    sim.set_geometry(component)
    sim.set_stack(mode_stack)
    sim.set_output_dir(tmp_path)
    sim.set_airbox(margin_x=1, margin_y=1, z_above=1, z_below=1)
    sim.set_cross_section("x=2")
    sim.add_port("probe", voltage_path=[[2, 1, 1.5], [2, 5, 1.5]], nsamples=25)
    sim.mesh(preset="coarse", verbose=False)
    sim.write_config()
    config = json.loads((tmp_path / "config.json").read_text())
    entry = {"Index": 1, "VoltagePath": [[1, 1.5], [5, 1.5]], "NSamples": 25}
    assert config["Boundaries"]["Postprocessing"]["Voltage"] == [entry]
    assert config["Boundaries"]["Postprocessing"]["Impedance"] == [entry]
    assert "LumpedPort" not in config["Boundaries"]


def test_validation_reports_port_after_rejected_path_removal(mode_stack, component):
    """A failed port edit must remain visible to simulation validation."""
    sim = BoundaryModeSim()
    sim.set_geometry(component)
    sim.set_stack(mode_stack)
    sim.set_cross_section("x=2")
    sim.add_port("probe", voltage_path=[[1, 1.5], [5, 1.5]])
    assert sim.validate_config().valid

    # Pydantic's after-validator rejects the edit after assigning the value.
    with pytest.raises(ValueError, match="Inplane ports require 'layer'"):
        sim.ports[0].voltage_path = None
    validation = sim.validate_config()
    assert not validation.valid
    assert "Port 'probe': inplane ports require 'layer'" in validation.errors
