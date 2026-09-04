"""End-to-end tests: PN-junction capacitance vs high-res mesh representation.

Capacitance mode must produce a Palace ``Boundaries.Impedance`` entry with
``Cs = C / interface_length`` and no junction domain.  High-res mode must
produce a ``junction`` dielectric domain (pure real permittivity) and no
Impedance boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import pytest

from gsim.common.cross_section import build_doped_cross_section
from gsim.common.stack.doping import make_pn_junction_profile
from gsim.common.stack.junction import PNJunctionConfig
from gsim.palace import BoundaryModeSim

F_RF = 50e9


def _thin_junction() -> PNJunctionConfig:
    """W ~ 17 nm -> below the auto threshold -> capacitance mode."""
    return PNJunctionConfig(na_cm3=1e19, nd_cm3=1e19)


def _wide_junction() -> PNJunctionConfig:
    """W ~ 71 nm -> above the auto threshold -> high-res mode."""
    return PNJunctionConfig(na_cm3=1e18, nd_cm3=1e18, v_reverse=1.0)


def _build_device(junction: PNJunctionConfig, **profile_kwargs):
    """Build the rib+slab+doping device and return (comp, stack)."""
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    wg = comp << gf.c.rectangle((10.0, 0.4), centered=True, layer=(1, 0))
    wg.y = -20.0
    slab = comp << gf.c.rectangle((10.0, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0

    pn = make_pn_junction_profile(
        comp,
        length=10.0,
        center_y=-20.0,
        rib_width=0.4,
        junction=junction,
        p_region=("p_rib", (21, 0), 1.6e3),
        n_region=("n_rib", (20, 0), 1.6e3),
        junction_region=("junction", (22, 0)),
        zmin=0.0,
        zmax=0.22,
        **profile_kwargs,
    )
    stack, _section = build_doped_cross_section(
        comp,
        axis="x",
        value=0.0,
        substrate_thickness=2.0,
        doping=pn,
        verbose=False,
    )
    return comp, stack, pn


def _make_sim(junction: PNJunctionConfig, tmp_path: Path, apply_capacitance: bool):
    comp, stack, pn = _build_device(junction)
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path / "palace-sim-pn"))
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    sim.set_geometry(comp)
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=F_RF, num_modes=1, save=0)
    sim.mesh(preset="coarse", refined_mesh_size=0.05, max_mesh_size=40.0)
    if apply_capacitance:
        applied = sim.set_pn_junction(
            junction,
            layer_p="p_rib",
            layer_n="n_rib",
            length_um=10.0,
            height_um=0.22,
        )
        assert applied == pytest.approx(junction.capacitance(10.0, 0.22))
    sim.write_config()
    config_path = Path(sim.output_dir) / "config.json"
    return sim, json.loads(config_path.read_text()), pn


@pytest.fixture(scope="module")
def cap_mode(tmp_path_factory):
    """Thin depletion: auto-selected capacitance mode with lumped C."""
    return _make_sim(_thin_junction(), tmp_path_factory.mktemp("cap"), True)


@pytest.fixture(scope="module")
def hires_mode(tmp_path_factory):
    """Wide depletion: auto-selected high-res mode, no lumped C."""
    return _make_sim(_wide_junction(), tmp_path_factory.mktemp("hires"), False)


class TestCapacitanceMode:
    def test_no_junction_domain_on_mesh(self, cap_mode):
        sim, _config, _pn = cap_mode
        groups = sim._last_mesh_result.groups
        assert "junction" not in groups["volumes"]

    def test_impedance_boundary_in_config(self, cap_mode):
        _sim, config, _pn = cap_mode
        impedance = config.get("Boundaries", {}).get("Impedance", [])
        assert len(impedance) == 1
        assert "Cs" in impedance[0]
        assert impedance[0]["Cs"] > 0

    def test_cs_value_matches_computed_capacitance(self, cap_mode):
        _sim, config, pn = cap_mode
        # Interface p_rib|n_rib is the vertical rib edge; its curve length is
        # the 0.22 um rib height, so Cs = C / 0.22um.
        expected_cs = pn["junction"]["c_f"] / (0.22 * 1e-6)
        cs = config["Boundaries"]["Impedance"][0]["Cs"]
        assert cs == pytest.approx(expected_cs, rel=1e-9)

    def test_doped_domains_present(self, cap_mode):
        sim, _config, _pn = cap_mode
        groups = sim._last_mesh_result.groups
        assert {"p_rib", "n_rib"} <= set(groups["volumes"])


class TestHighResMode:
    def test_junction_dielectric_domain_on_mesh(self, hires_mode):
        sim, _config, _pn = hires_mode
        groups = sim._last_mesh_result.groups
        assert "junction" in groups["volumes"]
        assert groups["volumes"]["junction"].get("is_shaped_dielectric") is True

    def test_no_impedance_boundary(self, hires_mode):
        _sim, config, _pn = hires_mode
        assert not config.get("Boundaries", {}).get("Impedance")

    def test_junction_material_is_pure_dielectric(self, hires_mode):
        sim, config, _pn = hires_mode
        groups = sim._last_mesh_result.groups
        junc_attr = groups["volumes"]["junction"]["phys_group"]
        materials = config["Domains"]["Materials"]
        entries = [m for m in materials if junc_attr in m.get("Attributes", [])]
        assert len(entries) == 1, f"Expected one material for attr {junc_attr}"
        entry = entries[0]
        assert abs(float(entry["Permittivity"]) - 11.9) < 1e-6
        assert not entry.get("Conductivity"), (
            "Depleted silicon must have zero conductivity"
        )

    def test_p_n_junction_strip_contiguous(self, hires_mode):
        """All three regions survive as separate domains."""
        sim, _config, _pn = hires_mode
        volumes = sim._last_mesh_result.groups["volumes"]
        assert {"p_rib", "n_rib", "junction"} <= set(volumes)
