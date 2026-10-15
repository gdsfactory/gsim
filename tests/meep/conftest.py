"""Activate the generic PDK for tests in gsim.meep that build components."""

from __future__ import annotations

import contextlib

import gdsfactory as gf
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.meep import mode_solver
from gsim.meep.models.config import MaterialData


@pytest.fixture(autouse=True)
def activate_generic_pdk():
    """Ensure the generic PDK is active before each test."""
    with contextlib.suppress(AttributeError):
        gf.gpdk.PDK.activate()


@pytest.fixture
def index_context(monkeypatch):
    # Fix material indices so expectations depend only on geometry, not on a
    # material database or dispersion fit. GDS slicing and plotting remain real.
    materials = {
        "core": MaterialData(epsilon_diag=[4, 4, 4]),
        "clad": MaterialData(epsilon_diag=[2.25, 2.25, 2.25]),
    }
    monkeypatch.setattr(
        mode_solver, "_resolve_materials_cached", lambda *_args, **_kwargs: materials
    )
    stack = LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=layer,
                zmin=zmin,
                zmax=zmax,
                thickness=zmax - zmin,
                material=name,
                layer_type="dielectric",
            )
            for name, layer, zmin, zmax in [
                ("clad", (2, 0), -1, 0),
                ("core", (1, 0), 0, 1),
            ]
        },
        dielectrics=[{"name": "clad", "zmin": -1, "zmax": 0, "material": "clad"}],
    )
    component = gf.Component()
    component.add_polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)], layer=(1, 0))
    return stack, component
