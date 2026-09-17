"""Vertical gap ports: conformal geometry and tangential excitation."""

import json

import gdsfactory as gf
import gmsh
import numpy as np
import pytest

from gsim.palace import DrivenSim


@pytest.mark.parametrize("orientation", [0, 90, 180, 270])
def test_gap_port_mesh(tmp_path, orientation):
    gf.gpdk.PDK.activate()
    c = gf.Component()
    # Two metal leads separated by 4 um, joined by the port at their ends.
    along_x = orientation % 180 == 0
    for x0, x1 in [(-8, -2), (2, 8)]:
        points = [(x0, 0), (x1, 0), (x1, 10), (x0, 10)]
        if not along_x:
            points = [(y, x) for x, y in points]
        c.add_polygon(points, layer=gf.gpdk.LAYER.M1)
    c.add_port(
        "feed",
        center=(0, 0),
        width=4,
        orientation=orientation,
        layer=gf.gpdk.LAYER.M1,
        port_type="electrical",
    )
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_geometry(c)
    sim.set_stack(substrate_thickness=2, air_above=10)
    sim.set_airbox(margin_x=10, margin_y=10, z_above=10, z_below=2)
    sim.add_port("feed", geometry="gap", layer="metal1")
    sim.set_driven(fmin=1e9, fmax=10e9, num_points=2)
    sim.mesh(preset="coarse")
    sim.write_config()
    config = json.loads((tmp_path / "config.json").read_text())
    ports = config["Boundaries"]["LumpedPort"]
    assert len(ports) == 1
    port = ports[0]
    assert port["Direction"] == {0: "X", 90: "Y", 180: "-X", 270: "-Y"}[orientation]
    assert port["R"] == 50
    assert port["Excitation"] == 1
    assert port["Attributes"]
    # Check the actual meshed port is vertical and its field lies in its plane.
    gmsh.initialize()
    try:
        gmsh.open(str(tmp_path / "palace.msh"))
        tags, coords = gmsh.model.mesh.getNodesForPhysicalGroup(
            2, port["Attributes"][0]
        )
        assert len(tags) >= 4
        xyz = np.asarray(coords).reshape(-1, 3)
        extent = np.ptp(xyz, axis=0)
        axis = 0 if along_x else 1
        assert extent[axis] > 0
        assert extent[2] > 0
        assert extent[1 - axis] == pytest.approx(0, abs=1e-10)
        # Port endpoints coincide with the two inner conductor edges.
        assert xyz[:, axis].min() == pytest.approx(-2)
        assert xyz[:, axis].max() == pytest.approx(2)
    finally:
        gmsh.finalize()


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"layer": "metal1", "length": 4}, {"layer": "metal1", "from_layer": "metal2"}],
)
def test_gap_port_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DrivenSim().add_port("feed", geometry="gap", **kwargs)


@pytest.mark.parametrize(
    ("orientation", "layer_type", "thickness", "message"),
    [
        (45, "conductor", 1, "cardinal"),
        (0, "dielectric", 1, "conductor layer"),
        (0, "conductor", 0, "positive width and layer thickness"),
    ],
)
def test_gap_port_invalid_geometry(orientation, layer_type, thickness, message):
    from types import SimpleNamespace

    from gsim.palace.ports.config import configure_gap_port, extract_ports

    gf.gpdk.PDK.activate()
    c = gf.Component()
    port = c.add_port(
        "feed",
        center=(0, 0),
        width=4,
        orientation=orientation,
        layer=gf.gpdk.LAYER.M1,
        port_type="electrical",
    )
    configure_gap_port(port, layer="metal1")
    stack = SimpleNamespace(
        layers={
            "metal1": SimpleNamespace(
                layer_type=layer_type,
                zmin=0,
                zmax=thickness,
            )
        }
    )
    with pytest.raises(ValueError, match=message):
        extract_ports(c, stack)
