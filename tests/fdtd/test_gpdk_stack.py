"""A generic-PDK component meshes with its whole stack and no template."""

import json
import warnings

import gdsfactory as gf
import meshio
import numpy as np
import pytest

from gsim import fdtd


@pytest.fixture(autouse=True)
def gpdk_active():
    """Use the generic PDK and restore the previous process state."""
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    gf.gpdk.PDK.activate(force=True)
    yield
    if previous is not None:
        previous.activate(force=True)


def _simulation(name="mmi1x2"):
    simulation = fdtd.Simulation()
    simulation.domain()
    simulation.source(port="o1")
    simulation.geometry(name, mesh_size_nm=700)
    return simulation


def test_gpdk_stack_reaches_the_mesh_and_config(tmp_path) -> None:
    simulation = _simulation()
    resolved = simulation.resolved
    assert list(resolved.layers) == ["substrate", "box", "core", "clad"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifacts = simulation.write(tmp_path)
    mesh = meshio.read(artifacts.mesh_path)
    for name in ("substrate", "box", "core", "clad"):
        assert name in mesh.field_data

    config = json.loads(artifacts.config_path.read_text())
    assert {"Si", "SiO2"} <= set(config["materials"])

    # Ports stay on the domain faces: the blanket films end where the ports do.
    port_x = sorted({port.center[0] for port in resolved.ports.values()})
    clad_tag = mesh.field_data["clad"][0]
    tags = mesh.cell_data_dict["gmsh:physical"]["tetra"]
    clad_points = mesh.points[np.unique(mesh.cells_dict["tetra"][tags == clad_tag])]
    assert clad_points[:, 0].min() == pytest.approx(port_x[0] * 1000)
    assert clad_points[:, 0].max() == pytest.approx(port_x[-1] * 1000)
    assert clad_points[:, 2].min() == pytest.approx(0)
    assert clad_points[:, 2].max() == pytest.approx(3000)


def test_explicit_blanket_marking_gives_the_same_stack() -> None:
    plain = _simulation().resolved
    stack = gf.gpdk.PDK.layer_stack
    assert stack is not None
    for key in ("substrate", "box", "clad"):
        stack.layers[key].background = True
    try:
        marked = _simulation().resolved
    finally:
        for key in ("substrate", "box", "clad"):
            stack.layers[key].background = False
    assert list(marked.layers) == list(plain.layers)
    for key, layer in plain.layers.items():
        assert marked.layers[key].geometry.bounds == pytest.approx(
            layer.geometry.bounds
        )
        assert marked.layers[key].material == layer.material
