"""Activate the generic PDK for tests in gsim.meep that build components."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from types import SimpleNamespace

import gdsfactory as gf
import numpy as np
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.meep import mode_solver
from gsim.meep.models.config import MaterialData


@dataclass
class _ModeVector:
    x: float = 0
    y: float = 0
    z: float = 0


@pytest.fixture
def backend(monkeypatch):
    """Record native MEEP calls while retaining gsim's real geometry calculations."""

    class Simulation:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)
            self.calls = []
            self.initialized = self.resets = 0
            backend.instances.append(self)

        def init_sim(self):
            self.initialized += 1

        def reset_meep(self):
            self.resets += 1

        def get_eigenmode(self, frequency, _direction, _where, **kwargs):
            self.calls.append(kwargs)
            if (
                kwargs["band_num"] in backend.failed_bands
                or frequency in backend.failed_frequencies
            ):
                return None

            def amplitude(point, _component):
                if backend.amplitude_error:
                    raise RuntimeError("native field unavailable")
                return (
                    0
                    if backend.zero_fields
                    else point.x + 2 * point.y + 3 * point.z + 1j
                )

            result = SimpleNamespace(
                kdom=_ModeVector(1.8 * frequency), amplitude=amplitude
            )
            if not backend.missing_velocity:
                result.group_velocity = backend.group_velocity
            return result

    backend = SimpleNamespace(
        Vector3=_ModeVector,
        Simulation=Simulation,
        instances=[],
        failed_bands=set(),
        failed_frequencies=set(),
        inf=np.inf,
        amplitude_error=False,
        zero_fields=False,
        missing_velocity=False,
        group_velocity=0.25,
        **dict.fromkeys(["Medium", "Block", "Prism", "PML", "Volume"], SimpleNamespace),
        **{name: name for name in ["X", "Y", "Z", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]},
    )
    monkeypatch.setattr(mode_solver, "_import_meep", lambda: backend)
    return backend


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
