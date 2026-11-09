"""Check physical FDTD artifacts for inferred virtual port layers."""

import json

import gdsfactory as gf
import meshio
import numpy as np
import pytest
from gdsfactory.cross_section import Section

from gsim import fdtd


@pytest.mark.parametrize("description", ["snapshot", "registry"])
def test_inferred_ports_write_the_same_physical_input_as_explicit_maps(
    tmp_path, fdtd_pdk_module, description
):
    """A marker is never meshed, and the port area equals core width times height."""
    pdk = fdtd_pdk_module.PDK
    pdk.activate()
    pdk.layer_stack.layers["core"].sidewall_angle = 0

    def routing():
        return gf.cross_section.cross_section(
            width=0.5,
            layer=(100, 0),
            sections=(Section(width=0.5, layer=(1, 0)),),
        )

    if description == "registry":
        pdk.cross_sections["routing"] = routing
    component = gf.Component()
    component.add_polygon([(0, -0.25), (2, -0.25), (2, 0.25), (0, 0.25)], layer=(1, 0))
    component.add_polygon([(0, -0.4), (2, -0.4), (2, 0.4), (0, 0.4)], layer=(100, 0))
    for name, center, angle in (("o1", (0, 0), 180), ("o2", (2, 0), 0)):
        if description == "snapshot":
            component.add_port(
                name=name, center=center, orientation=angle, cross_section=routing()
            )
        else:
            component.add_port(
                name=name, center=center, orientation=angle, width=0.5, layer=(100, 0)
            )
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.source(port="o1", num_wavelengths=3)
    simulation.geometry(component)
    assert simulation.geometry.port_layer_map == {}
    automatic = simulation.write(tmp_path / "automatic")
    reference = fdtd.Simulation(pdk=fdtd_pdk_module)
    reference.source(port="o1", num_wavelengths=3)
    reference.geometry(component, port_layer_map={(100, 0): "core"})
    explicit = reference.write(tmp_path / "explicit")
    assert json.loads(automatic.config_path.read_text()) == json.loads(
        explicit.config_path.read_text()
    )

    mesh = meshio.read(automatic.mesh_path)
    assert set(mesh.field_data) == {"core", "background", "port_o1", "port_o2"}
    triangles = mesh.cells_dict["triangle"]
    groups = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    port_vertices = mesh.points[triangles[groups == mesh.field_data["port_o1"][0]]]
    area = (
        np.linalg.norm(
            np.cross(
                port_vertices[:, 1] - port_vertices[:, 0],
                port_vertices[:, 2] - port_vertices[:, 0],
            ),
            axis=1,
        ).sum()
        / 2
    )
    assert area == pytest.approx(500 * 220)
    assert port_vertices[:, :, 0] == pytest.approx(0)
    assert port_vertices[:, :, 2].min() == pytest.approx(0)
    assert port_vertices[:, :, 2].max() == pytest.approx(220)
