"""Integration tests for mesh auto-sizing gating in ``_build_mesh_config``.

These tests exercise the pure-config path only — no gmsh invocation, so
they run anywhere including macOS.
"""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common import Geometry
from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import DrivenSim


@pytest.fixture(autouse=True)
def _activate_pdk():
    """Activate the generic PDK for every test in this module."""
    gf.gpdk.PDK.activate()


def _conductor_stack(gds_layer: tuple[int, int] = (1, 0)) -> LayerStack:
    """Return a minimal stack with a single conductor layer."""
    stack = LayerStack()
    stack.layers["metal"] = Layer(
        name="metal",
        gds_layer=gds_layer,
        zmin=0.0,
        zmax=0.5,
        thickness=0.5,
        material="copper",
        layer_type="conductor",
    )
    return stack


def _small_feature_component(width: float = 2.0) -> gf.Component:
    """Component with a very narrow trace (width um wide)."""
    c = gf.Component()
    half_w = width / 2
    c.add_polygon(
        [(-50, -half_w), (50, -half_w), (50, half_w), (-50, half_w)],
        layer=(1, 0),
    )
    return c


def _make_sim(component: gf.Component, stack: LayerStack) -> DrivenSim:
    """Build a DrivenSim wired to the given component and stack without gmsh."""
    sim = DrivenSim()
    sim.geometry = Geometry(component=component)
    # Bypass lazy-stack resolution — use the pre-built stack directly.
    sim.stack = stack
    sim._stack_kwargs = {"_prebuilt": True}
    return sim


# Preset nominal refined_mesh_size values
PRESET_REFINED = {
    "coarse": 10.0,
    "default": 5.0,
    "fine": 2.0,
}


@pytest.mark.parametrize("preset", ["coarse", "default", "fine"])
def test_autosizing_fires_for_every_preset(preset):
    """With small features and no explicit override, auto-sizing must
    shrink refined_mesh_size below the preset's nominal value.

    With a 2 um trace, auto_size = 2/4 = 0.5 um, which is below every
    preset's nominal (10.0 / 5.0 / 2.0). So it must fire in all cases.
    """
    sim = _make_sim(_small_feature_component(width=2.0), _conductor_stack())
    mesh_config = sim._build_mesh_config(
        preset=preset,
        refined_mesh_size=None,
        max_mesh_size=None,
        margin=None,
        airbox_margin=None,
        fmax=None,
        planar_conductors=None,
        show_gui=False,
    )
    assert mesh_config.refined_mesh_size < PRESET_REFINED[preset], (
        f"preset={preset}: expected auto-sizer to shrink refined_mesh_size "
        f"below {PRESET_REFINED[preset]}, got {mesh_config.refined_mesh_size}"
    )
    assert mesh_config.refined_mesh_size == pytest.approx(0.5)


@pytest.mark.parametrize("preset", ["coarse", "default", "fine"])
def test_explicit_override_bypasses_autosizing(preset):
    """When user passes an explicit refined_mesh_size, auto-sizing must NOT
    overwrite it regardless of preset.
    """
    sim = _make_sim(_small_feature_component(width=2.0), _conductor_stack())
    user_size = 7.5
    mesh_config = sim._build_mesh_config(
        preset=preset,
        refined_mesh_size=user_size,
        max_mesh_size=None,
        margin=None,
        airbox_margin=None,
        fmax=None,
        planar_conductors=None,
        show_gui=False,
    )
    assert mesh_config.refined_mesh_size == user_size


def test_autosizing_noop_for_large_features():
    """Designs with large features keep the preset size."""
    # 100 um trace, default preset (5.0 um). auto = min(5.0, 100/4) = 5.0.
    c = _small_feature_component(width=100.0)
    sim = _make_sim(c, _conductor_stack())
    mesh_config = sim._build_mesh_config(
        preset="default",
        refined_mesh_size=None,
        max_mesh_size=None,
        margin=None,
        airbox_margin=None,
        fmax=None,
        planar_conductors=None,
        show_gui=False,
    )
    assert mesh_config.refined_mesh_size == PRESET_REFINED["default"]


def test_autosizing_cpw_gap_fires():
    """CPW with 15 um gap drives refinement below preset even though trace
    is 20 um wide.
    """
    from tests.palace.test_auto_size import _cpw_component

    c = _cpw_component(s_width=20.0, gap=15.0, ground_width=40.0)
    sim = _make_sim(c, _conductor_stack())
    mesh_config = sim._build_mesh_config(
        preset="default",
        refined_mesh_size=None,
        max_mesh_size=None,
        margin=None,
        airbox_margin=None,
        fmax=None,
        planar_conductors=None,
        show_gui=False,
    )
    # min_feature = 15 (gap), /4 = 3.75 < preset 5.0
    assert mesh_config.refined_mesh_size == pytest.approx(3.75)
