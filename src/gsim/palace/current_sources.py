"""Resolve magnetostatic current sources to source sheets.

A current source names two gdsfactory ports on one conductor
(:class:`~gsim.palace.models.ports.CurrentSourceConfig`). Here each source
becomes the vertical sheet that bridges the gap between the two port faces,
through the thickness of the conductor layer, with the Palace ``Direction`` of
its current and the net that closes its loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from gsim.palace.ports.config import PalacePort, PortGeometry, PortType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapely import LineString

    from gsim.common.stack import LayerStack
    from gsim.palace.mesh.nets import Nets
    from gsim.palace.models.ports import CurrentSourceConfig

Direction = Literal["+X", "-X", "+Y", "-Y"]

# Positions closer than this are equal: the 1 nm layout grid, as a tolerance.
_TOL_UM = 1e-3

# Cardinal angle (degrees) of each Palace direction.
_ANGLE: dict[str, float] = {"+X": 0.0, "+Y": 90.0, "-X": 180.0, "-Y": 270.0}


@dataclass(frozen=True)
class ResolvedCurrentSource:
    """A current source as the sheet that carries its current.

    Attributes:
        name: Source name; it labels the row of the inductance matrix.
        layer: Conductor layer whose thickness the sheet spans.
        center: ``(x, y)`` of the middle of the sheet, in um.
        span: Length of the sheet along ``direction`` (the gap), in um.
        direction: Palace ``Direction`` of the current in the sheet, the
            cardinal direction from the minus port to the plus port.
        net: Name of the net that joins the two ports (the return path).
        plus_port: Port where the current enters the conductor.
        minus_port: Port where the current leaves the conductor.
    """

    name: str
    layer: str
    center: tuple[float, float]
    span: float
    direction: Direction
    net: str
    plus_port: str
    minus_port: str

    def to_palace_port(self, stack: LayerStack) -> PalacePort:
        """The sheet as a ``SURFACE_CURRENT`` gap port for the mesher.

        Args:
            stack: The layer stack; it gives the z-range of the sheet.

        Returns:
            A ``PalacePort`` with ``GAP`` geometry: ``width`` is the span and
            ``orientation`` the angle of ``direction``.
        """
        layer = stack.layers[self.layer]
        return PalacePort(
            name=self.name,
            port_type=PortType.SURFACE_CURRENT,
            geometry=PortGeometry.GAP,
            center=self.center,
            width=self.span,
            orientation=_ANGLE[self.direction],
            zmin=layer.zmin,
            zmax=layer.zmax,
            layer=self.layer,
        )


def _error(source: CurrentSourceConfig, rule: str, detail: str) -> ValueError:
    """A ``ValueError`` naming the source and the rule it breaks."""
    return ValueError(f"Current source '{source.name}' ({rule}): {detail}")


def _cardinal(port: Any) -> int | None:
    """Quarter turns (0-3) of a port orientation, None when not cardinal."""
    if port.orientation is None:
        return None
    turns = float(port.orientation) / 90.0
    if abs(turns - round(turns)) > 1e-6:
        return None
    return round(turns) % 4


def _layer(
    source: CurrentSourceConfig, plus: Any, minus: Any, stack: LayerStack
) -> str:
    """The conductor layer of a source, given or inferred from the ports.

    Raises:
        ValueError: If the layer is unknown, not a conductor or has no
            thickness, or cannot be inferred from the ports unambiguously.
    """
    name = source.layer
    if name is None:
        candidates = []
        for port in (plus, minus):
            number = port.layer_info.layer
            candidates.append(
                sorted(
                    layer_name
                    for layer_name, layer in stack.layers.items()
                    if layer.layer_type == "conductor" and layer.gds_layer[0] == number
                )
            )
        if candidates[0] != candidates[1] or len(candidates[0]) != 1:
            found = {
                source.plus_port: candidates[0],
                source.minus_port: candidates[1],
            }
            raise _error(
                source,
                "layer rule",
                "cannot infer the conductor layer from the GDS layer of the "
                f"ports (conductor layers per port: {found}); pass layer=",
            )
        name = candidates[0][0]
    layer = stack.layers.get(name)
    if layer is None or layer.layer_type != "conductor":
        conductors = sorted(
            n for n, lay in stack.layers.items() if lay.layer_type == "conductor"
        )
        raise _error(
            source,
            "layer rule",
            f"{name!r} is not a conductor layer of the stack; the conductor "
            f"layers are {conductors}",
        )
    if layer.zmax <= layer.zmin:
        raise _error(source, "layer rule", f"conductor layer {name!r} has no thickness")
    return name


def _bridge(
    source: CurrentSourceConfig, plus: Any, minus: Any
) -> tuple[tuple[float, float], float, Direction]:
    """Centre, span and Direction of the sheet between two ports.

    The ports either face each other across a gap along ``d = c_minus -
    c_plus`` (span ``|d|``), or are side-by-side leads with one orientation
    perpendicular to ``d`` (span ``|d| - (w_plus + w_minus) / 2``, inner edge
    to inner edge).

    Raises:
        ValueError: If the ports fit neither arrangement or the span is not
            positive.
    """
    rule = "bridge rule"
    q_plus, q_minus = _cardinal(plus), _cardinal(minus)
    if q_plus is None or q_minus is None:
        raise _error(
            source,
            rule,
            "both ports need a cardinal orientation (0/90/180/270); got "
            f"{plus.orientation} and {minus.orientation}",
        )
    (xp, yp), (xm, ym) = plus.dcenter, minus.dcenter
    dx, dy = xm - xp, ym - yp
    if min(abs(dx), abs(dy)) > _TOL_UM:
        raise _error(
            source,
            rule,
            f"the port centres are not on an x or y line (d = ({dx:.4g}, {dy:.4g}) um)",
        )
    along_x = abs(dx) >= abs(dy)
    distance = abs(dx) if along_x else abs(dy)
    # quarter turns of the unit vector u from plus to minus
    q_d = (0 if dx > 0 else 2) if along_x else (1 if dy > 0 else 3)
    if q_plus == q_d and q_minus == (q_d + 2) % 4:
        span = distance
        offset = span / 2
    elif q_plus == q_minus and (q_plus - q_d) % 2 == 1:
        span = distance - (float(plus.width) + float(minus.width)) / 2
        offset = float(plus.width) / 2 + span / 2
    else:
        raise _error(
            source,
            rule,
            "the ports must face each other across the gap, or be side-by-side "
            "leads with one orientation perpendicular to the line through "
            f"their centres; got orientations {plus.orientation} and "
            f"{minus.orientation} with d = ({dx:.4g}, {dy:.4g}) um",
        )
    if span <= _TOL_UM:
        raise _error(
            source,
            rule,
            f"the gap between the ports is {span:.4g} um; it must be positive",
        )
    if along_x:
        sign = 1.0 if dx > 0 else -1.0
        center = (xp + sign * offset, (yp + ym) / 2)
    else:
        sign = 1.0 if dy > 0 else -1.0
        center = ((xp + xm) / 2, yp + sign * offset)
    # the current runs from minus to plus inside the sheet: along -u
    direction: Direction = (
        ("-X" if sign > 0 else "+X") if along_x else ("-Y" if sign > 0 else "+Y")
    )
    return center, span, direction


def _loop_net(
    source: CurrentSourceConfig,
    plus: Any,
    minus: Any,
    layer: str,
    nets: Nets,
) -> str:
    """Name of the net that joins the two ports on ``layer``.

    Raises:
        ValueError: If a port is on no metal of the layer, or the two ports are
            on different nets.
    """
    found = {}
    for port in (plus, minus):
        x, y = port.dcenter
        net = nets.net_at(x, y, layer)
        if net is None:
            raise _error(
                source,
                "open loop",
                f"port {port.name!r} is on no metal of layer {layer!r}; both "
                "ports must sit on the conductor that closes the loop",
            )
        found[port.name] = net
    if found[plus.name] is not found[minus.name]:
        raise _error(
            source,
            "open loop",
            f"ports {plus.name!r} and {minus.name!r} are on different nets "
            f"({found[plus.name].name!r} and {found[minus.name].name!r}); "
            "the drawn metal must join them to return the current",
        )
    return found[plus.name].name


def _resolve_one(
    source: CurrentSourceConfig,
    by_name: dict[str, Any],
    stack: LayerStack,
    nets: Nets,
) -> ResolvedCurrentSource:
    """Resolve one source on its own (every rule but those between sources).

    Raises:
        ValueError: If a port is missing, or the source breaks the layer or
            bridge rule, its loop is open or its sheet crosses metal.
    """
    for port_name in (source.plus_port, source.minus_port):
        if port_name not in by_name:
            raise _error(
                source,
                "missing port",
                f"the component has no port {port_name!r}; its ports are "
                f"{sorted(by_name)}",
            )
    plus, minus = by_name[source.plus_port], by_name[source.minus_port]
    layer = _layer(source, plus, minus, stack)
    center, span, direction = _bridge(source, plus, minus)
    net = _loop_net(source, plus, minus, layer, nets)
    blocking = nets.net_at(*center, layer)
    if blocking is not None:
        raise _error(
            source,
            "sheet crosses metal",
            f"the middle of the sheet, ({center[0]:.4g}, {center[1]:.4g}) um, "
            f"is on net {blocking.name!r} of layer {layer!r}; the gap between "
            "the ports must be free of metal",
        )
    return ResolvedCurrentSource(
        name=source.name,
        layer=layer,
        center=center,
        span=span,
        direction=direction,
        net=net,
        plus_port=source.plus_port,
        minus_port=source.minus_port,
    )


def _sheet(source: ResolvedCurrentSource) -> LineString:
    """The plan view of a sheet: a segment across the gap, in um."""
    from shapely import LineString

    x, y = source.center
    half = source.span / 2
    if source.direction in ("+X", "-X"):
        return LineString([(x - half, y), (x + half, y)])
    return LineString([(x, y - half), (x, y + half)])


def _check_pairs(
    sources: Sequence[CurrentSourceConfig],
    resolved: Sequence[ResolvedCurrentSource],
    stack: LayerStack,
) -> None:
    """Check that no two sources share a port pair or a sheet.

    Sharing a single port is allowed (a centre tap).

    Raises:
        ValueError: If two sources name the same two ports, or their sheets
            meet.
    """
    for j, (source, sheet) in enumerate(zip(sources, resolved, strict=True)):
        layer = stack.layers[sheet.layer]
        for other_source, other in zip(sources[:j], resolved[:j], strict=True):
            pair = {source.plus_port, source.minus_port}
            if pair == {other_source.plus_port, other_source.minus_port}:
                raise _error(
                    source,
                    "duplicate port pair",
                    f"source '{other.name}' already joins ports "
                    f"{sorted(pair)}; one pair carries one source",
                )
            other_layer = stack.layers[other.layer]
            z_overlap = layer.zmin < other_layer.zmax and other_layer.zmin < layer.zmax
            if z_overlap and _sheet(sheet).intersects(_sheet(other)):
                raise _error(
                    source,
                    "intersecting sheets",
                    f"its sheet meets the sheet of source '{other.name}'; "
                    "sources need separate gaps",
                )


def resolve_current_sources(
    sources: Sequence[CurrentSourceConfig],
    component: Any,
    stack: LayerStack,
    nets: Nets,
) -> list[ResolvedCurrentSource]:
    """Resolve current sources to the sheets that carry their current.

    Args:
        sources: The sources, in the order they were added; this is the order
            of the rows of the inductance matrix.
        component: The gdsfactory component that carries the ports.
        stack: The layer stack.
        nets: The nets of the layout, built from **all** component ports so
            that optical ports find their net too.

    Returns:
        The resolved sources, in the order given.

    Raises:
        ValueError: Naming the source and the rule it breaks, if there is no
            source, a port is missing, the ports break the bridge or layer
            rule, the loop is open, the sheet crosses metal, or two sources
            share a name, a port pair or sheet.
    """
    if not sources:
        raise ValueError(
            "There is no current source; add one with add_current_source(name, "
            "plus_port=..., minus_port=...)."
        )
    by_name = {port.name: port for port in component.ports}
    resolved: list[ResolvedCurrentSource] = []
    for source in sources:
        if any(other.name == source.name for other in resolved):
            raise _error(
                source,
                "duplicate name",
                "two sources have this name; it labels a row of the inductance "
                "matrix, so give each source its own name",
            )
        resolved.append(_resolve_one(source, by_name, stack, nets))
    _check_pairs(sources, resolved, stack)
    return resolved


__all__ = ["ResolvedCurrentSource", "resolve_current_sources"]
