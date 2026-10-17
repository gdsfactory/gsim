"""Synthetic layouts shared by the electrostatic terminal tests (generic stack).

Stack (z in um): ``m1`` [0, 1] aluminum, ``v1`` [1, 2] tungsten via,
``m2`` [2, 3] aluminum, in an SiO2 box from z = -2 to 5.

Layout: two electrodes ``A`` (port PLUS) and ``B`` (port MINUS) on ``m1``, a
loose island on ``m1``, and an ``m2`` bar tied to each electrode by a via.
"""

from __future__ import annotations

import re
from pathlib import Path

import gdsfactory as gf
import numpy as np

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import ElectrostaticSim

LAYERS = {
    "m1": (1, 0.0, 1.0, "conductor", "aluminum"),
    "v1": (3, 1.0, 2.0, "via", "tungsten"),
    "m2": (2, 2.0, 3.0, "conductor", "aluminum"),
}

# Box = (layer, x0, y0, x1, y1) in um
A = ("m1", 0, 0, 4, 4)
B = ("m1", 6, 0, 10, 4)
ISLAND = ("m1", 0, 6, 10, 7)
BAR_A = ("m2", 1, 1, 3, 3)
VIA_A = ("v1", 1.5, 1.5, 2.5, 2.5)
BAR_B = ("m2", 7, 1, 9, 3)
VIA_B = ("v1", 7.5, 1.5, 8.5, 2.5)

# the layout of the probe: A and B with their bars, plus the island
FULL = [A, B, ISLAND, BAR_A, VIA_A, VIA_B, BAR_B]
PORTS = [("PLUS", 2, 2, "m1"), ("MINUS", 8, 2, "m1")]

MESH = {"refined_mesh_size": 0.5, "max_mesh_size": 3.0, "merge_via_distance": 0}


def stack() -> LayerStack:
    """The generic three-layer stack."""
    return LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=(number, 0),
                zmin=z0,
                zmax=z1,
                thickness=z1 - z0,
                material=material,
                layer_type=kind,
            )
            for name, (number, z0, z1, kind, material) in LAYERS.items()
        },
        materials={
            "aluminum": {"type": "conductor", "conductivity": 3.77e7},
            "tungsten": {"type": "conductor", "conductivity": 1.8e7},
            "sio2": {"type": "dielectric", "permittivity": 3.9},
            "air": {"type": "dielectric", "permittivity": 1.0},
        },
        dielectrics=[{"name": "ox", "zmin": -2.0, "zmax": 5.0, "material": "sio2"}],
    )


def component(boxes=FULL, ports=PORTS) -> gf.Component:
    """Rectangles on the stack layers plus electrical ports."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    for layer, x0, y0, x1, y1 in boxes:
        c.add_polygon(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=(LAYERS[layer][0], 0)
        )
    for name, x, y, layer in ports:
        c.add_port(
            name=name,
            center=(x, y),
            width=1.0,
            orientation=0,
            layer=(LAYERS[layer][0], 0),
            port_type="electrical",
        )
    return c


def make_sim(out_dir: Path, boxes=FULL, ports=PORTS) -> ElectrostaticSim:
    """An electrostatic simulation of a layout, with no terminals yet."""
    sim = ElectrostaticSim()
    sim.set_output_dir(out_dir)
    sim.set_geometry(component(boxes, ports))
    sim.set_stack(stack())
    sim.set_airbox(margin_x=5, margin_y=5, z_above=5, z_below=0)
    return sim


def nets_of(boxes=FULL, ports=PORTS):
    """The nets of a layout of rectangles, as ``extract_nets`` finds them."""
    from gsim.palace.mesh.geometry import extract_geometry
    from gsim.palace.mesh.nets import extract_nets

    c = component(boxes, ports)
    st = stack()
    return extract_nets(extract_geometry(c, st), st, c.ports)


def physical_names(msh_path: Path, dim: int = 2) -> dict[int, str]:
    """``{tag: name}`` of the physical groups of a text ``.msh`` (format 2.2)."""
    text = Path(msh_path).read_text()
    block = text.split("$PhysicalNames")[1].split("$EndPhysicalNames")[0]
    names: dict[int, str] = {}
    for line in block.strip().splitlines()[1:]:
        match = re.match(r'(\d+)\s+(\d+)\s+"(.*)"', line.strip())
        if match and int(match.group(1)) == dim:
            names[int(match.group(2))] = match.group(3)
    return names


def surface_nodes_by_group(msh_path: Path) -> dict[str, np.ndarray]:
    """``{physical name: (n, 2) array of x, y}`` of the triangle nodes per group.

    Reads the text ``.msh`` (format 2.2) without gmsh, so it checks the file
    that Palace gets.
    """
    names = physical_names(msh_path)
    lines = Path(msh_path).read_text().splitlines()
    start = lines.index("$Nodes")
    count = int(lines[start + 1])
    xy = {}
    for line in lines[start + 2 : start + 2 + count]:
        node, x, y, _ = line.split()
        xy[int(node)] = (float(x), float(y))
    start = lines.index("$Elements")
    count = int(lines[start + 1])
    nodes: dict[str, list[tuple[float, float]]] = {}
    for line in lines[start + 2 : start + 2 + count]:
        fields = [int(v) for v in line.split()]
        _, kind, ntags = fields[:3]
        if kind != 2:  # triangles only
            continue
        phys = fields[3]
        if phys in names:
            nodes.setdefault(names[phys], []).extend(xy[n] for n in fields[3 + ntags :])
    return {name: np.array(points) for name, points in nodes.items()}


def attribute_names(config: dict, msh_path: Path) -> dict[str, list[set[str]]]:
    """Boundary groups of a config as sets of physical names.

    Returns ``{"Terminal": [names of terminal 1, ...], "Ground": [names]}``.
    """
    names = physical_names(msh_path)
    boundaries = config["Boundaries"]
    out: dict[str, list[set[str]]] = {
        "Terminal": [
            {names[a] for a in entry["Attributes"]} for entry in boundaries["Terminal"]
        ],
        "Ground": [],
    }
    if "Ground" in boundaries:
        out["Ground"] = [{names[a] for a in boundaries["Ground"]["Attributes"]}]
    return out
