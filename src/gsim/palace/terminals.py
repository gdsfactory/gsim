"""Resolve electrostatic terminals to the nets of a layout.

A terminal selects conductors by net name, by region or by layer
(:class:`~gsim.palace.models.ports.TerminalConfig`). Here each selection becomes
a set of nets, and the nets of no terminal become Ground or extra floating
terminals. Nets are referred to by their position in
:class:`~gsim.palace.mesh.nets.Nets`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gsim.palace.mesh.nets import Nets
    from gsim.palace.models.ports import TerminalConfig


@dataclass(frozen=True)
class ResolvedTerminal:
    """A terminal and the nets it drives.

    Attributes:
        name: Terminal name; for an automatic floating terminal the net name.
        nets: Positions of the nets in ``Nets``, in net order.
        floating: The terminal is solved but not driven; its row is eliminated
            from the capacitance matrix.
        net_names: Names of those nets, to read the selection at a glance.
    """

    name: str
    nets: tuple[int, ...]
    floating: bool
    net_names: tuple[str, ...] = ()


def _select(terminal: TerminalConfig, nets: Nets) -> list[int]:
    """Positions of the nets a terminal selects, in net order.

    Raises:
        ValueError: If the selection names an unknown net or layer, or matches
            no net.
    """
    position = {id(net): k for k, net in enumerate(nets.nets)}
    try:
        if terminal.net is not None:
            hits = nets.find(terminal.net)
        elif terminal.region is not None:
            polygon = terminal.region_polygon()
            layers = None if terminal.layer is None else [terminal.layer]
            hits = () if polygon is None else nets.nets_in(polygon, layers=layers)
        else:
            hits = tuple(net for net in nets.nets if terminal.layer in net.layers)
    except KeyError as error:
        message = error.args[0] if error.args else str(error)
        raise ValueError(f"Terminal '{terminal.name}': {message}") from None
    if not hits:
        if terminal.net is not None:
            what = f"net {terminal.net!r}"
        elif terminal.region is not None:
            where = f" on layer {terminal.layer!r}" if terminal.layer else ""
            what = f"the region {terminal.region}{where}"
        else:
            what = f"layer {terminal.layer!r}"
        raise ValueError(f"Terminal '{terminal.name}': no net found at {what}")
    return sorted(position[id(net)] for net in hits)


def resolve_terminals(
    terminals: Sequence[TerminalConfig],
    nets: Nets,
    *,
    unassigned: Literal["ground", "floating"] = "ground",
) -> list[ResolvedTerminal]:
    """Resolve terminals to nets.

    Args:
        terminals: The terminals, in the order they were added.
        nets: The nets of the layout.
        unassigned: What the nets in no terminal do. ``"ground"`` leaves them
            out, so they are Ground. ``"floating"`` adds one floating terminal
            per net, named after the net, after the user's terminals.

    Returns:
        The user's terminals in order, then the automatic floating terminals
        in net order.

    Raises:
        ValueError: If a selection matches no net, a net is in two terminals,
            an automatic floating terminal has the name of another terminal,
            or no terminal is driven.
    """
    resolved: list[ResolvedTerminal] = []
    owner: dict[int, str] = {}
    for terminal in terminals:
        selected = _select(terminal, nets)
        for k in selected:
            if k in owner:
                raise ValueError(
                    f"Net '{nets[k].name}' is in two terminals, '{owner[k]}' and "
                    f"'{terminal.name}'. Select it in one terminal only; with "
                    "layer= or region= a terminal takes every net it touches."
                )
            owner[k] = terminal.name
        resolved.append(
            ResolvedTerminal(
                name=terminal.name,
                nets=tuple(selected),
                floating=terminal.floating,
                net_names=tuple(nets[k].name for k in selected),
            )
        )
    if resolved and all(t.floating for t in resolved):
        raise ValueError(
            "Every terminal is floating; at least one terminal must be driven."
        )
    if unassigned == "floating":
        taken = {t.name for t in resolved}
        for k, net in enumerate(nets.nets):
            if k in owner:
                continue
            if net.name in taken:
                raise ValueError(
                    f"The unassigned net '{net.name}' would become a floating "
                    "terminal with the name of another terminal. Rename the "
                    "terminal, or put the net in a terminal."
                )
            taken.add(net.name)
            resolved.append(
                ResolvedTerminal(
                    name=net.name,
                    nets=(k,),
                    floating=True,
                    net_names=(net.name,),
                )
            )
    return resolved


__all__ = ["ResolvedTerminal", "resolve_terminals"]
