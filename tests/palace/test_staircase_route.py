"""Staircase route: carrier strips as Palace BoundaryMode domains.

Hermetic config-generation tests (no Palace binary): synthetic carrier
maps are binned into N strips, registered through the patterned-dielectric
machinery, and the generated mesh groups and Palace config material blocks
are asserted per strip.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gdsfactory as gf
import numpy as np
import pytest

from gsim.modulator.staircase import (
    RFStripMaterial,
    StaircaseDrawing,
    build_staircase_cross_section,
    strip_averages_from_nodes,
)
from gsim.palace import BoundaryModeSim
from tests._helpers import fake_coupling

RIB_CENTER_Y = -20.0
RIB_WIDTH = 0.4
N_STRIPS = 3


def _synthetic_carriers():
    """Depletion-like carrier profile across the rib (junction axis = y)."""
    y = np.linspace(RIB_CENTER_Y - RIB_WIDTH / 2, RIB_CENTER_Y + RIB_WIDTH / 2, 201)
    rel = (y - RIB_CENTER_Y) / (RIB_WIDTH / 2)
    electrons = 1e18 * np.clip(-rel, 0.0, 1.0)
    holes = 1e18 * np.clip(rel, 0.0, 1.0)
    return y, electrons, holes


def _build_staircase_device(n_strips=N_STRIPS):
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    wg = comp << gf.c.rectangle((10.0, RIB_WIDTH), centered=True, layer=(1, 0))
    wg.y = RIB_CENTER_Y
    slab = comp << gf.c.rectangle((10.0, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0

    y, electrons, holes = _synthetic_carriers()
    staircase = build_staircase_cross_section(
        SimpleNamespace(
            x_um=y, y_um=np.full(y.size, 0.11), electrons_cm3=electrons, holes_cm3=holes
        ),
        n_strips=n_strips,
        junction=(float(y[0]), float(y[-1])),
        zmin=0.0,
        zmax=0.22,
        response=fake_coupling,
        material=RFStripMaterial(),
        electrodes=None,
        drawing=StaircaseDrawing(base_layer=(40, 0), component=comp),
    )
    return comp, staircase.stack(), staircase


@pytest.fixture(scope="module")
def staircase_sim(tmp_path_factory):
    comp, stack, staircase = _build_staircase_device()
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path_factory.mktemp("staircase")))
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    sim.set_geometry(comp)
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=50e9, num_modes=1)
    sim.mesh(preset="coarse", refined_mesh_size=0.05, max_mesh_size=40.0, verbose=False)
    sim.write_config()
    config = json.loads((Path(sim.output_dir) / "config.json").read_text())
    return sim, config, staircase


class TestStaircaseDomains:
    def test_every_strip_is_a_mesh_domain(self, staircase_sim):
        sim, _config, _staircase = staircase_sim
        volumes = sim.mesh_groups["volumes"]
        for i in range(N_STRIPS):
            assert f"strip_{i}" in volumes, f"strip_{i} missing from mesh domains"
            assert volumes[f"strip_{i}"].get("is_shaped_dielectric") is True

    def test_config_has_material_block_per_strip(self, staircase_sim):
        sim, config, staircase = staircase_sim
        volumes = sim.mesh_groups["volumes"]
        materials = config["Domains"]["Materials"]
        sigma = staircase.strips.conductivity_s_per_m
        for i in range(N_STRIPS):
            attr = volumes[f"strip_{i}"]["phys_group"]
            entries = [m for m in materials if attr in m.get("Attributes", [])]
            assert len(entries) == 1, f"expected one material block for strip_{i}"
            entry = entries[0]
            if sigma[i] > 0:
                assert float(entry.get("Conductivity", 0.0)) > 0.0
            assert float(entry["Permittivity"]) > 1.0

    def test_strip_conductivity_profile_is_asymmetric(self, staircase_sim):
        # The synthetic profile is n-heavy on one side, p-heavy on the other;
        # strip averages must preserve that asymmetry end to end.
        _sim, _config, staircase = staircase_sim
        sigma = staircase.strips.conductivity_s_per_m
        assert sigma[0] > sigma[-1]  # electrons conduct better: n-side first


class TestMetallicWindow:
    """``metallic_boundaries`` puts the outer wall under ``Boundaries.PEC``.

    femwell's ``metallic_boundaries`` shields the whole domain boundary;
    without this, nothing in the Palace pipeline expressed the same
    condition and Palace defaulted the unconditioned wall to PMC — the
    opposite one — so the two solvers answered different boundary-value
    problems on the identical mesh.
    """

    @staticmethod
    def _outer_attrs(sim) -> list[int]:
        pg = sim.mesh_groups["boundary_surfaces"]["absorbing"]["phys_group"]
        return pg if isinstance(pg, list) else [pg]

    def test_the_outer_wall_is_unconditioned_by_default(self, staircase_sim):
        sim, config, _staircase = staircase_sim
        pec = config.get("Boundaries", {}).get("PEC", {}).get("Attributes", [])
        assert not set(self._outer_attrs(sim)) & set(pec)

    def test_the_outer_wall_lands_under_pec_when_asked(self, staircase_sim):
        sim, _config, _staircase = staircase_sim
        sim.metallic_boundaries = True
        try:
            sim.write_config()
            shielded = json.loads((Path(sim.output_dir) / "config.json").read_text())
        finally:
            sim.metallic_boundaries = False
        pec = shielded["Boundaries"]["PEC"]["Attributes"]
        assert set(self._outer_attrs(sim)) <= set(pec)
        assert "Absorbing" not in shielded["Boundaries"]

    def test_claiming_the_wall_twice_is_refused(self, staircase_sim):
        sim, _config, _staircase = staircase_sim
        sim.metallic_boundaries = True
        sim.absorbing_boundary = True
        try:
            with pytest.raises(ValueError, match="both claim the outer wall"):
                sim.write_config()
        finally:
            sim.metallic_boundaries = False
            sim.absorbing_boundary = False


class TestConvergenceInStripCount:
    def test_material_profile_converges_with_n(self):
        y, electrons, _holes = _synthetic_carriers()

        def error(n_strips):
            edges, means = strip_averages_from_nodes(y, electrons, n_strips=n_strips)
            idx = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, n_strips - 1)
            return float(np.sqrt(np.mean((means[idx] - electrons) ** 2)))

        errors = [error(n) for n in (1, 2, 4, 8, 16)]
        assert errors == sorted(errors, reverse=True)
        assert errors[-1] < 0.1 * errors[0]
