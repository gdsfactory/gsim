"""Electrical nets of a layout: which conductor shapes are connected.

Two conductor shapes are on the same net when

- metal touches metal on one layer: the shapes overlap or share an edge (a
  single shared point has no width and does not conduct), or
- a via joins them: the z-range of the via meets that of the metal and their
  footprints overlap in plan.

Only vias join layers, and a via joins only the conductor layers its z-range
reaches. Shapes that touch nothing are nets of their own.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from shapely import Point, Polygon, STRtree

from gsim.palace.mesh.geometry import get_layer_infos

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from gsim.common.stack import LayerStack
    from gsim.palace.mesh.geometry import GeometryData

# Numerical guards, far below the 1 nm grid of a layout: a real contact is at
# least 1e-3 um long and 1e-6 um^2 in area, and vias meet metal in z to within
# the float noise of the z values of the stack.
_MIN_CONTACT_LENGTH_UM = 1e-6
_MIN_CONTACT_AREA_UM2 = 1e-9
_Z_TOUCH_TOLERANCE_UM = 1e-6

_CONNECTING = ("conductor", "via")


@dataclass(frozen=True)
class Net:
    """Conductor shapes that are electrically connected.

    Attributes:
        name: The names of the ports on the net joined by ``+``, or ``net<k>``
            when it has none, with ``k`` its position in the list of nets,
            counting from 1.
        ports: Names of the ports whose centre lies on the net.
        members: ``(layer name, index)`` of each shape, where the index is the
            position of the shape in ``GeometryData.polygons``.
    """

    name: str
    ports: tuple[str, ...]
    members: tuple[tuple[str, int], ...]

    @property
    def layers(self) -> tuple[str, ...]:
        """Names of the conductor and via layers the net has shapes on, sorted."""
        return tuple(sorted({layer for layer, _ in self.members}))


@dataclass(frozen=True, eq=False)
class Nets:
    """The nets of a layout, in the order their first shape appears.

    Attributes:
        nets: The nets.
        unmatched_ports: Names of ports whose centre is on no conductor or via
            shape with the GDS layer number of the port.
    """

    nets: tuple[Net, ...]
    unmatched_ports: tuple[str, ...] = ()
    _shapes: dict[str, list[tuple[Polygon, Net]]] = field(
        default_factory=dict, repr=False
    )

    def __len__(self) -> int:
        """Number of nets."""
        return len(self.nets)

    def __iter__(self) -> Iterator[Net]:
        """Iterate over the nets."""
        return iter(self.nets)

    def __getitem__(self, index: int) -> Net:
        """The net at a position in the list."""
        return self.nets[index]

    def __str__(self) -> str:
        """One line per net with its shapes per layer, then any unmatched ports."""
        lines = [f"{len(self)} nets"]
        for net in self:
            counts = Counter(layer for layer, _ in net.members)
            shapes = ", ".join(f"{layer} x{n}" for layer, n in sorted(counts.items()))
            lines.append(f"  {net.name}: {shapes}")
        if self.unmatched_ports:
            lost = ", ".join(self.unmatched_ports)
            lines.append(f"  ports on no conductor: {lost}")
        return "\n".join(lines)

    def net_at(self, x: float, y: float, layer: str) -> Net | None:
        """The net under the point ``(x, y)`` (um) on a conductor or via layer.

        Returns None when the layer has no shape there; a point on the edge of
        a shape counts as on it.

        Raises:
            KeyError: If ``layer`` is not a conductor or via layer of the stack.
        """
        if layer not in self._shapes:
            msg = (
                f"unknown conductor or via layer {layer!r}; "
                f"the layers are {sorted(self._shapes)}"
            )
            raise KeyError(msg)
        point = Point(x, y)
        # ponytail: linear scan, an STRtree per layer if this is called in a loop
        for shape, net in self._shapes[layer]:
            if shape.covers(point):
                return net
        return None


def _connected(a, b, shape_a: Polygon, shape_b: Polygon) -> bool:
    """Whether two intersecting shapes, on stack layers ``a`` and ``b``, conduct."""
    if a.name == b.name:
        common = shape_a.intersection(shape_b)
        return common.length > _MIN_CONTACT_LENGTH_UM
    if {a.layer_type, b.layer_type} != set(_CONNECTING):
        return False
    tol = _Z_TOUCH_TOLERANCE_UM
    if a.zmin > b.zmax + tol or b.zmin > a.zmax + tol:
        return False
    return shape_a.intersection(shape_b).area > _MIN_CONTACT_AREA_UM2


def extract_nets(
    geometry: GeometryData, stack: LayerStack, ports: Iterable = ()
) -> Nets:
    """Group the conductor and via shapes of a layout into electrical nets.

    Args:
        geometry: The polygons of the layout, as the mesher reads them
            (``extract_geometry``).
        stack: The layer stack; its conductor and via layers are the ones
            considered.
        ports: Ports of the component. Each names the net of the shapes that
            cover its centre on layers with the GDS layer number of the port,
            whatever the datatype (pins are usually on another datatype than
            the metal).

    Returns:
        The nets, each shape on exactly one of them.
    """
    layers = {
        name: layer
        for name, layer in stack.layers.items()
        if layer.layer_type in _CONNECTING
    }
    names_by_number: dict[int, list[str]] = {}
    nodes: list[tuple[str, int]] = []  # (layer name, index in geometry.polygons)
    shapes: list[Polygon] = []
    for index, (number, xs, ys, holes) in enumerate(geometry.polygons):
        if number not in names_by_number:
            names_by_number[number] = [
                info["name"]
                for info in get_layer_infos(stack, number)
                if info["type"] in _CONNECTING
            ]
        if not names_by_number[number]:
            continue
        shape = Polygon(
            list(zip(xs, ys, strict=True)),
            [list(zip(hx, hy, strict=True)) for hx, hy in holes],
        )
        for name in names_by_number[number]:
            nodes.append((name, index))
            shapes.append(shape)

    if not shapes:
        # nothing to connect, and an STRtree cannot be queried with no shapes
        unmatched = tuple(port.name for port in ports)
        return Nets(
            nets=(),
            unmatched_ports=unmatched,
            _shapes={name: [] for name in layers},
        )

    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # one drawn shape is one piece of material, even when several layers of the
    # stack share its GDS layer number
    seen: dict[int, int] = {}
    for i, (_, index) in enumerate(nodes):
        if index in seen:
            parent[find(i)] = find(seen[index])
        else:
            seen[index] = i

    tree = STRtree(shapes)
    first, second = tree.query(shapes, predicate="intersects")
    for i, j in zip(first.tolist(), second.tolist(), strict=True):
        if (
            i < j
            and find(i) != find(j)
            and _connected(
                layers[nodes[i][0]], layers[nodes[j][0]], shapes[i], shapes[j]
            )
        ):
            parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)
    # nets ordered by their first shape in the layout
    ordered = sorted(groups.values(), key=lambda g: min(nodes[i][1] for i in g))

    net_of_node = {i: k for k, group in enumerate(ordered) for i in group}
    port_names: list[list[str]] = [[] for _ in ordered]
    unmatched: list[str] = []
    for port in ports:
        x, y = port.dcenter
        number = port.layer_info.layer
        hits = tree.query(Point(x, y), predicate="intersects").tolist()
        found = {
            net_of_node[i] for i in hits if layers[nodes[i][0]].gds_layer[0] == number
        }
        if not found:
            unmatched.append(port.name)
        for k in sorted(found):
            if port.name not in port_names[k]:
                port_names[k].append(port.name)

    result = tuple(
        Net(
            name="+".join(port_names[k]) or f"net{k + 1}",
            ports=tuple(port_names[k]),
            members=tuple(nodes[i] for i in group),
        )
        for k, group in enumerate(ordered)
    )
    lookup: dict[str, list[tuple[Polygon, Net]]] = {name: [] for name in layers}
    for net, group in zip(result, ordered, strict=True):
        for i in group:
            lookup[nodes[i][0]].append((shapes[i], net))
    return Nets(nets=result, unmatched_ports=tuple(unmatched), _shapes=lookup)


__all__ = ["Net", "Nets", "extract_nets"]
