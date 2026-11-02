"""Reusing a real Gmsh mesh in another simulation (gsim#389).

One coarse eigenmode mesh of a tiny two-port layout is meshed by Gmsh once
for the module. A driven simulation then takes it with ``reuse_mesh`` and
writes its config with the default mesh validation, which opens the copied
``.msh``: the Gmsh-free tests in ``test_reuse_mesh.py`` cannot show that the
copied file passes ``validate_mesh`` and agrees with the copied groups.
"""

from __future__ import annotations

import json

import gdsfactory as gf
import pytest

from gsim.palace import DrivenSim, EigenmodeSim
from tests._electrode_layouts import stack

# o1 and o2: lumped ports on m1 at either end of the m1 strip
PORT = {"layer": "m1", "length": 1.0}


def _layout() -> gf.Component:
    """A strip on m1 and one on m2, with ports o1 and o2 at the ends of m1."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    c.add_polygon([(0, 0), (10, 0), (10, 1), (0, 1)], layer=(1, 0))
    c.add_polygon([(0, 3), (10, 3), (10, 4), (0, 4)], layer=(2, 0))
    for name, x, orientation in (("o1", 0, 180), ("o2", 10, 0)):
        c.add_port(
            name=name,
            center=(x, 0.5),
            width=1.0,
            orientation=orientation,
            layer=(1, 0),
            port_type="electrical",
        )
    return c


@pytest.fixture(scope="module")
def meshed_eigen(tmp_path_factory) -> EigenmodeSim:
    """An eigenmode sim of the layout with two lumped ports, meshed by Gmsh."""
    sim = EigenmodeSim(absorbing_boundary=False)
    sim.set_output_dir(tmp_path_factory.mktemp("reuse") / "eigen")
    sim.set_geometry(_layout())
    sim.set_stack(stack())
    # The ports straddle the ends of the strip; the airbox keeps them inside.
    sim.set_airbox(margin_x=5, margin_y=5, z_above=5, z_below=0)
    sim.add_port("o1", **PORT)
    sim.add_port("o2", **PORT)
    sim.set_eigenmode(target=5e9)
    sim.mesh(preset="coarse", verbose=False)
    return sim


def _no_gmsh(**_kwargs):
    """Stand in for ``generate_mesh`` once meshing is not allowed."""
    raise AssertionError("generate_mesh was called")


def test_driven_sim_reuses_a_real_eigenmode_mesh(meshed_eigen, monkeypatch) -> None:
    """The copied mesh passes validation and drives o1 with reflecting walls."""
    monkeypatch.setattr("gsim.palace.mesh.generator.generate_mesh", _no_gmsh)
    folder = meshed_eigen.output_dir.parent / "driven"
    receiver = DrivenSim(absorbing_boundary=False)
    receiver.set_output_dir(folder)
    receiver.add_port("o1", **PORT, excited=True)
    receiver.add_port("o2", **PORT, excited=False)
    receiver.set_driven(fmin=7e9, fmax=10e9, excitation_port="o1")

    receiver.reuse_mesh(meshed_eigen)

    assert receiver.mesh_path == folder / "palace.msh"
    assert receiver.mesh_path.read_bytes() == meshed_eigen.mesh_path.read_bytes()
    assert receiver.mesh_groups == meshed_eigen.mesh_groups

    # validate_mesh=True (the default) opens the copied mesh with Gmsh
    config = json.loads(receiver.write_config().read_text())

    assert config["Problem"]["Type"] == "Driven"
    lumped = config["Boundaries"]["LumpedPort"]
    assert [port["Excitation"] for port in lumped] == [1, False]
    assert "Absorbing" not in config["Boundaries"]
