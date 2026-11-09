"""Integration tests for mesh auto-sizing gating in ``_build_mesh_config``.

These tests exercise the pure-config path only — no gmsh invocation, so
they run anywhere including macOS.
"""

from __future__ import annotations

import logging

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
def test_autosizing_fires_for_every_preset_when_opted_in(preset):
    """With small features and auto_size=True, auto-sizing must shrink
    refined_mesh_size below the preset's nominal value.

    With a 2 um trace, auto_size = 2/2 = 1.0 um, which is below every
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
        auto_size=True,
    )
    assert mesh_config.refined_mesh_size < PRESET_REFINED[preset], (
        f"preset={preset}: expected auto-sizer to shrink refined_mesh_size "
        f"below {PRESET_REFINED[preset]}, got {mesh_config.refined_mesh_size}"
    )
    assert mesh_config.refined_mesh_size == pytest.approx(1.0)


@pytest.mark.parametrize("preset", ["coarse", "default", "fine"])
def test_default_preserves_preset_refined_size(preset):
    """Without auto_size (default), refined_mesh_size must equal the preset's
    literal value even when the component has small features.
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
    assert mesh_config.refined_mesh_size == PRESET_REFINED[preset]


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
        auto_size=True,
    )
    assert mesh_config.refined_mesh_size == user_size


def test_autosizing_noop_for_large_features():
    """Designs with large features keep the preset size even with auto_size."""
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
        auto_size=True,
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
        auto_size=True,
        cells_per_feature=4,
    )
    # min_feature = 15 (gap), /4 = 3.75 < preset 5.0
    assert mesh_config.refined_mesh_size == pytest.approx(3.75)


def test_cells_per_feature_kwarg_respected():
    """cells_per_feature=4 halves the refined_mesh_size vs default=2."""
    c = _small_feature_component(width=2.0)
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
        auto_size=True,
        cells_per_feature=4,
    )
    # min_feature = 2, /4 = 0.5 < preset 5.0
    assert mesh_config.refined_mesh_size == pytest.approx(0.5)


def test_sim_mesh_default_uses_preset_refined_size():
    """``sim.mesh()`` with no ``auto_size`` kwarg returns the preset's
    literal ``refined_mesh_size``, not a scaled value.

    This test uses ``_build_mesh_config`` directly since ``sim.mesh()`` would
    invoke gmsh.
    """
    sim = _make_sim(_small_feature_component(width=2.0), _conductor_stack())
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


def test_sim_mesh_auto_size_scales_for_small_feature():
    """``auto_size=True`` scales the mesh down for a small-feature component."""
    sim = _make_sim(_small_feature_component(width=2.0), _conductor_stack())
    mesh_config = sim._build_mesh_config(
        preset="default",
        refined_mesh_size=None,
        max_mesh_size=None,
        margin=None,
        airbox_margin=None,
        fmax=None,
        planar_conductors=None,
        show_gui=False,
        auto_size=True,
    )
    # min_feature = 2, /2 (default cells_per_feature) = 1.0 < preset 5.0
    assert mesh_config.refined_mesh_size == pytest.approx(1.0)


def test_under_resolution_warning_fires(caplog):
    """When a small feature is present and ``auto_size=False``, the mesh
    build path emits a ``logger.warning`` suggesting ``auto_size=True``.
    """
    sim = _make_sim(_small_feature_component(width=1.0), _conductor_stack())
    # 1 um feature < default preset's 5.0/2 = 2.5, so warn should fire.
    with caplog.at_level(logging.WARNING, logger="gsim.palace.base"):
        sim._build_mesh_config(
            preset="default",
            refined_mesh_size=None,
            max_mesh_size=None,
            margin=None,
            airbox_margin=None,
            fmax=None,
            planar_conductors=None,
            show_gui=False,
        )
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Small conductor feature" in record.getMessage()
    ]
    assert warnings, f"expected under-resolution warning, got {caplog.records}"
    assert "auto_size=True" in warnings[0].getMessage()


@pytest.mark.parametrize(
    ("preset", "size", "auto_size", "cells_per_feature", "expected_size", "warn"),
    [
        ("coarse", 8.0, False, 2, 8.0, True),
        ("coarse", 1.0, False, 2, 1.0, False),
        ("fine", None, False, 2, 2.0, True),
        ("fine", 8.0, False, 2, 8.0, True),
        ("coarse", 8.0, True, 2, 8.0, True),
        ("coarse", None, True, 2, 1.0, False),
        ("coarse", 1.0, False, 4, 1.0, True),
        ("coarse", 0.5, False, 4, 0.5, False),
    ],
)
def test_resolution_diagnostics_use_effective_size(
    caplog, preset, size, auto_size, cells_per_feature, expected_size, warn
):
    sim = _make_sim(_small_feature_component(width=2.0), _conductor_stack())
    with caplog.at_level(logging.WARNING, logger="gsim.palace.base"):
        config = sim._build_mesh_config(
            preset=preset,
            refined_mesh_size=size,
            max_mesh_size=None,
            margin=None,
            airbox_margin=None,
            fmax=None,
            planar_conductors=None,
            show_gui=False,
            auto_size=auto_size,
            cells_per_feature=cells_per_feature,
        )
    assert config.refined_mesh_size == expected_size
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "Small conductor feature" in record.getMessage()
    ]
    if warn:
        assert len(warnings) == 1
        assert f"refined_mesh_size={expected_size:.3f} um" in warnings[0]
        assert f"<= {2 / cells_per_feature:.3f} um" in warnings[0]
        if size is not None:
            assert "Reduce refined_mesh_size" in warnings[0]
            assert "preserves an explicit size" in warnings[0]
    else:
        assert not warnings


def test_under_resolution_warning_silenced_by_auto_size(caplog):
    """With ``auto_size=True`` we don't emit the under-resolution warning."""
    sim = _make_sim(_small_feature_component(width=1.0), _conductor_stack())
    with caplog.at_level(logging.WARNING, logger="gsim.palace.base"):
        sim._build_mesh_config(
            preset="default",
            refined_mesh_size=None,
            max_mesh_size=None,
            margin=None,
            airbox_margin=None,
            fmax=None,
            planar_conductors=None,
            show_gui=False,
            auto_size=True,
        )
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Small conductor feature" in record.getMessage()
    ]
    assert not warnings, (
        f"did not expect under-resolution warning with auto_size, got {warnings}"
    )


def test_no_warning_when_feature_is_large_enough(caplog):
    """Two cells across a feature satisfy the default heuristic."""
    # 10 um feature vs preset 5 gives exactly two cells across the feature.
    sim = _make_sim(_small_feature_component(width=10.0), _conductor_stack())
    with caplog.at_level(logging.WARNING, logger="gsim.palace.base"):
        sim._build_mesh_config(
            preset="default",
            refined_mesh_size=None,
            max_mesh_size=None,
            margin=None,
            airbox_margin=None,
            fmax=None,
            planar_conductors=None,
            show_gui=False,
        )
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Small conductor feature" in record.getMessage()
    ]
    assert not warnings
