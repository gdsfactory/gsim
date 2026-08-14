"""Fixtures for PDK-native FDTD artifact tests."""

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
        cells={"straight": straight_component},
        layer_stack=layer_stack,
    )
    project_si = GSIM_MATERIAL_CARDS["Si-Li-293K"].model_copy(update={"name": "Si"})
    return SimpleNamespace(PDK=pdk, MATERIAL_CARDS={"Si": project_si})
