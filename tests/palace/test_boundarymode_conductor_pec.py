"""``BoundaryModeSim(conductor_boundary="pec")``: conductors as perfect conductors.

The 3D wave-port eigenproblem treats the finite-conductivity conductor
surfaces as PEC (``waveport_boundary="pec"``); a 2D boundary-mode solve keeps
them as Palace's Robin conductivity condition unless asked otherwise. The tests
mesh one generic coplanar-strip cross-section over a lower plane (generic PDK,
two conductor layers, no solver) and check
what the option does to the generated configuration: exactly the conductor
attributes move from ``Boundaries.Conductivity`` to ``Boundaries.PEC``, and
nothing else changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import pytest
from pydantic import ValidationError

from gsim.palace import BoundaryModeSim

LAYER = "M3"
STRIPS = {  # (y0, y1) um: ground, signal, ground
    "G1": (-60.0, -16.0),
    "S1": (-10.0, 10.0),
    "G2": (16.0, 60.0),
}
LENGTH = 100.0


def strip_component() -> gf.Component:
    """Three parallel strips on one metal of the generic PDK."""
    gf.gpdk.PDK.activate()
    layer = getattr(gf.gpdk.LAYER, LAYER)
    component = gf.Component()
    for y0, y1 in STRIPS.values():
        component.add_polygon(
            [(-LENGTH / 2, y0), (LENGTH / 2, y0), (LENGTH / 2, y1), (-LENGTH / 2, y1)],
            layer=layer,
        )
    plane = gf.gpdk.LAYER.M1  # a second conductor layer: a second attribute moves
    component.add_polygon(
        [
            (-LENGTH / 2, -60.0),
            (LENGTH / 2, -60.0),
            (LENGTH / 2, 60.0),
            (-LENGTH / 2, 60.0),
        ],
        layer=plane,
    )
    component.add_port(
        name="o1",
        center=(-LENGTH / 2, 0),
        width=20.0,
        orientation=180,
        port_type="electrical",
        layer=layer,
    )
    component.add_port(
        name="o2",
        center=(LENGTH / 2, 0),
        width=20.0,
        orientation=0,
        port_type="electrical",
        layer=layer,
    )
    return component


def meshed_sim(tmp_path: Path) -> BoundaryModeSim:
    """A meshed 2D simulation with two voltage paths declared."""
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path / "sim2d"))
    sim.set_geometry(strip_component())
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=0.0, margin_y=20.0, z_above=20.0, z_below=10.0)
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=10e9, num_modes=2, save=0)
    sim.mesh(preset="coarse", verbose=False)
    ((h0, _), (_, v1)) = sim.mesh_extent
    sim.add_impedance_path("gap", voltage=[[-14.0, 4.0], [-12.0, 4.0]])
    sim.add_impedance_path(
        "probe", voltage=[[h0 + 1.0, v1 - 1.0], [h0 + 2.0, v1 - 1.0]]
    )
    return sim


def config_of(sim: BoundaryModeSim) -> dict:
    """Write the configuration and read it back."""
    path = Path(sim.write_config())
    return json.loads(path.read_text())


def conductivity_attributes(config: dict) -> set[int]:
    """Attributes of every ``Boundaries.Conductivity`` entry."""
    return {
        a
        for entry in config["Boundaries"].get("Conductivity", [])
        for a in entry["Attributes"]
    }


def pec_attributes(config: dict) -> set[int]:
    """Attributes of ``Boundaries.PEC``."""
    return set(config["Boundaries"].get("PEC", {}).get("Attributes", []))


def without_conductor_boundaries(config: dict) -> dict:
    """The configuration with the two keys the option may touch removed."""
    stripped = json.loads(json.dumps(config))
    stripped["Boundaries"].pop("Conductivity", None)
    stripped["Boundaries"].pop("PEC", None)
    return stripped


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    """Mesh once; every test writes the configuration it wants from it."""
    return meshed_sim(tmp_path_factory.mktemp("conductor-pec"))


@pytest.fixture
def lossy(sim):
    """The configuration with the default (Robin conductivity) conductors."""
    sim.conductor_boundary = "conductivity"
    sim.metallic_boundaries = False
    return config_of(sim)


class TestTheOption:
    def test_the_default_keeps_the_conductors_as_conductivity(self, lossy):
        assert BoundaryModeSim().conductor_boundary == "conductivity"
        assert len(conductivity_attributes(lossy)) == 2, "two Robin conductor layers"
        assert not pec_attributes(lossy) & conductivity_attributes(lossy)

    def test_the_conductor_attributes_move_to_pec(self, sim, lossy):
        moved = conductivity_attributes(lossy)
        sim.conductor_boundary = "pec"
        try:
            config = config_of(sim)
        finally:
            sim.conductor_boundary = "conductivity"
        assert "Conductivity" not in config["Boundaries"]
        assert pec_attributes(config) == pec_attributes(lossy) | moved
        assert moved <= pec_attributes(config)

    def test_nothing_else_changes(self, sim, lossy):
        sim.conductor_boundary = "pec"
        try:
            config = config_of(sim)
        finally:
            sim.conductor_boundary = "conductivity"
        assert without_conductor_boundaries(config) == without_conductor_boundaries(
            lossy
        )
        # the paths (postprocessing only) are still written, and no internal hint leaks
        assert len(config["Boundaries"]["Postprocessing"]["Voltage"]) == 2
        assert not [key for key in config if key.startswith("_")]

    def test_the_option_composes_with_a_metallic_outer_wall(self, sim, lossy):
        moved = conductivity_attributes(lossy)
        sim.metallic_boundaries = True
        try:
            shielded = config_of(sim)
            sim.conductor_boundary = "pec"
            both = config_of(sim)
        finally:
            sim.conductor_boundary = "conductivity"
            sim.metallic_boundaries = False
        assert pec_attributes(shielded) - pec_attributes(lossy), "the wall is PEC"
        assert pec_attributes(both) == pec_attributes(shielded) | moved
        assert without_conductor_boundaries(both) == without_conductor_boundaries(
            shielded
        )

    def test_it_can_be_switched_back(self, sim, lossy):
        sim.conductor_boundary = "pec"
        config_of(sim)
        sim.conductor_boundary = "conductivity"
        assert config_of(sim) == lossy

    def test_the_configuration_stays_valid(self, sim):
        sim.conductor_boundary = "pec"
        try:
            assert sim.validate_config().valid
        finally:
            sim.conductor_boundary = "conductivity"

    def test_an_unknown_value_is_refused(self):
        with pytest.raises(ValidationError):
            BoundaryModeSim(conductor_boundary="robin")
        simulation = BoundaryModeSim()
        with pytest.raises(ValidationError):
            simulation.conductor_boundary = "impedance"  # ty: ignore[invalid-assignment]
