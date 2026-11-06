"""Fixtures for PDK-native GDSFactory FDTD artifact tests."""

from __future__ import annotations

from types import SimpleNamespace

import gdsfactory as gf
import pytest
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

from gsim.common.materials import GSIM_MATERIAL_CARDS


def straight_component(length: float = 2.0) -> gf.Component:
    """Return a minimal two-port waveguide on the test core layer."""
    component = gf.Component()
    component.add_polygon(
        [(0, -0.25), (length, -0.25), (length, 0.25), (0, 0.25)],
        layer=(1, 0),
    )
    component.add_port(
        name="o1",
        center=(0, 0),
        width=0.5,
        orientation=180,
        layer=(1, 0),
    )
    component.add_port(
        name="o2",
        center=(length, 0),
        width=0.5,
        orientation=0,
        layer=(1, 0),
    )
    return component


def vertical_coupler_component(length: float = 2.0) -> gf.Component:
    """Return a waveguide with one guided and one vertical optical port."""
    component = gf.Component()
    component.add_polygon(
        [(0, -0.25), (length, -0.25), (length, 0.25), (0, 0.25)],
        layer=(1, 0),
    )
    component.add_port(
        name="o1",
        center=(0, 0),
        width=0.5,
        orientation=180,
        layer=(1, 0),
    )
    component.add_port(
        name="o2",
        center=(length / 2, 0),
        width=1,
        orientation=0,
        layer=(1, 0),
        port_type="vertical_te",
    )
    return component


def bend_component() -> gf.Component:
    """Return an L bend with guided ports facing negative x and positive y."""
    component = gf.Component()
    component.add_polygon(
        [(0, -0.25), (2.25, -0.25), (2.25, 2), (1.75, 2), (1.75, 0.25), (0, 0.25)],
        layer=(1, 0),
    )
    component.add_port(
        name="o1", center=(0, 0), width=0.5, orientation=180, layer=(1, 0)
    )
    component.add_port(
        name="o2", center=(2, 2), width=0.5, orientation=90, layer=(1, 0)
    )
    return component


def ten_nm_feature_component(length: float = 2.0) -> gf.Component:
    """Return a waveguide plus one explicit 10 nm-wide material feature."""
    component = gf.Component()
    component.add_polygon(
        [(0, -0.25), (length, -0.25), (length, 0.25), (0, 0.25)],
        layer=(2, 0),
    )
    component.add_polygon(
        [(0.5, 0.4), (0.51, 0.4), (0.51, 0.5), (0.5, 0.5)],
        layer=(2, 0),
    )
    component.add_port(
        name="o1",
        center=(0, 0),
        width=0.5,
        orientation=180,
        layer=(2, 0),
    )
    component.add_port(
        name="o2",
        center=(length, 0),
        width=0.5,
        orientation=0,
        layer=(2, 0),
    )
    return component


@pytest.fixture
def fdtd_pdk_module() -> SimpleNamespace:
    """Return a PDK module with project Si and fallback-only SiO2."""
    layer_stack = LayerStack(
        layers={
            "core": LayerLevel(
                layer=LogicalLayer(layer=(1, 0)),
                thickness=0.22,
                zmin=0,
                sidewall_angle=10,
                width_to_z=0.5,
                mesh_order=2,
                material="Si",
            ),
            "thin_core": LayerLevel(
                layer=LogicalLayer(layer=(2, 0)),
                thickness=0.22,
                zmin=0,
                sidewall_angle=0,
                width_to_z=0.5,
                mesh_order=2,
                material="Si",
            ),
            "buried_oxide": LayerLevel(
                layer=LogicalLayer(layer=(99, 0)),
                thickness=2,
                zmin=-1,
                mesh_order=4,
                material="SiO2",
            ),
        }
    )
    pdk = gf.Pdk(
        name="fdtd_test_pdk",
        cells={
            "straight": straight_component,
            "bend": bend_component,
            "vertical_coupler": vertical_coupler_component,
            "ten_nm_feature": ten_nm_feature_component,
        },
        layer_stack=layer_stack,
    )
    project_si = GSIM_MATERIAL_CARDS["Si-Li-293K"].model_copy(update={"name": "Si"})
    return SimpleNamespace(PDK=pdk, MATERIAL_CARDS={"Si": project_si})
