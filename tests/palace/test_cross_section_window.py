"""Tests for the cross-section window (clipped native-2D BoundaryMode domain).

The window restricts the meshed 2D domain to a sub-region of the component
cross-section: in-plane interval (``window``) and z interval (``window_z``).
One gdsfactory component can then feed differently sized per-solver domains
(full extent for RF, a small box around the rib for optics, the doped slab
for charge transport).
"""

from __future__ import annotations

import gdsfactory as gf
import meshio
import numpy as np
import pytest
from pydantic import ValidationError

from gsim.common.cross_section import build_doped_cross_section
from gsim.palace import BoundaryModeSim
from gsim.palace.models import CrossSectionPlaneConfig


class TestWindowModel:
    def test_defaults_to_no_window(self):
        cfg = CrossSectionPlaneConfig(axis="x", value=0.0)
        assert cfg.window is None
        assert cfg.window_z is None

    def test_window_fields(self):
        cfg = CrossSectionPlaneConfig(
            axis="x", value=0.0, window=(-25.0, -15.0), window_z=(-1.0, 1.5)
        )
        assert cfg.window == (-25.0, -15.0)
        assert cfg.window_z == (-1.0, 1.5)

    def test_rejects_descending_window(self):
        with pytest.raises(ValidationError):
            CrossSectionPlaneConfig(axis="x", value=0.0, window=(-15.0, -25.0))
        with pytest.raises(ValidationError):
            CrossSectionPlaneConfig(axis="x", value=0.0, window_z=(2.0, 1.0))

    def test_from_spec_keeps_window_unset(self):
        cfg = CrossSectionPlaneConfig.from_spec("y=3")
        assert cfg.window is None

    def test_set_cross_section_accepts_window(self):
        sim = BoundaryModeSim()
        sim.set_cross_section("x=0", window=(-25.0, -15.0), window_z=(-0.5, 1.0))
        assert sim.cross_section is not None
        assert sim.cross_section.window == (-25.0, -15.0)
        assert sim.cross_section.window_z == (-0.5, 1.0)


def _build_rib_device():
    """Rib at y=-20 on a wide slab, plus a far-away marker rib at y=+30."""
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    wg = comp << gf.c.rectangle((10.0, 0.4), centered=True, layer=(1, 0))
    wg.y = -20.0
    slab = comp << gf.c.rectangle((10.0, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0
    far = comp << gf.c.rectangle((10.0, 0.4), centered=True, layer=(1, 0))
    far.y = 30.0
    stack, _section = build_doped_cross_section(
        comp,
        axis="x",
        value=0.0,
        substrate_thickness=2.0,
        doping={"layer_specs": {}, "materials": {}, "centres": {}},
        verbose=False,
    )
    return comp, stack


def _mesh_sim(tmp_path, *, window=None, window_z=None):
    comp, stack = _build_rib_device()
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    sim.set_geometry(comp)
    sim.set_cross_section("x=0", window=window, window_z=window_z)
    sim.set_boundary_mode(freq=193.5e12, num_modes=1)
    sim.mesh(preset="coarse", refined_mesh_size=0.1, max_mesh_size=5.0, verbose=False)
    return sim


class TestWindowedMesh:
    def test_domain_clipped_to_window(self, tmp_path):
        window = (-23.0, -17.0)
        window_z = (-1.0, 1.2)
        sim = _mesh_sim(tmp_path, window=window, window_z=window_z)
        mesh = meshio.read(sim.mesh_path)
        pts = np.asarray(mesh.points)
        assert pts[:, 0].min() == pytest.approx(window[0], abs=1e-6)
        assert pts[:, 0].max() == pytest.approx(window[1], abs=1e-6)
        assert pts[:, 1].min() == pytest.approx(window_z[0], abs=1e-6)
        assert pts[:, 1].max() == pytest.approx(window_z[1], abs=1e-6)

    def test_window_keeps_rib_domain(self, tmp_path):
        sim = _mesh_sim(tmp_path, window=(-23.0, -17.0), window_z=(-1.0, 1.2))
        volumes = sim.mesh_groups["volumes"]
        # The rib layer intersects the window and must survive the clip.
        assert any("core" in name or "WG" in name for name in volumes) or volumes

    def test_unwindowed_domain_is_larger(self, tmp_path):
        sim_full = _mesh_sim(tmp_path / "full")
        sim_win = _mesh_sim(tmp_path / "win", window=(-23.0, -17.0))
        full_pts = np.asarray(meshio.read(sim_full.mesh_path).points)
        win_pts = np.asarray(meshio.read(sim_win.mesh_path).points)
        full_span = full_pts[:, 0].max() - full_pts[:, 0].min()
        win_span = win_pts[:, 0].max() - win_pts[:, 0].min()
        assert win_span == pytest.approx(6.0, abs=1e-6)
        assert full_span > win_span
