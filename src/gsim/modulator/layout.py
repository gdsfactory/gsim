"""Deriving Contacts, Interfaces and Windows from the device description.

The device description names Regions; the drawn Cross-section says where
they are. Reading the two together gives the Contacts (metal on
semiconductor), the Interfaces between adjacent doped Regions, which of
them is the Junction, and the charge Window that spans the doped slab —
none of which a user should have to declare a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

from gsim.modulator.device import Device

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack.extractor import LayerStack

__all__ = [
    "Contact",
    "DeviceLayout",
    "Interface",
    "Span",
    "derive_layout",
]

#: Coordinates closer than this count as touching (um).
TOUCH_TOL_UM: float = 1e-3


@dataclass(frozen=True)
class Span:
    """Extent of a Region on the Cross-section.

    Attributes:
        h: ``(min, max)`` in-plane extent along the junction axis (um).
        z: ``(min, max)`` vertical extent (um).
    """

    h: tuple[float, float]
    z: tuple[float, float]


@dataclass(frozen=True)
class Contact:
    """A terminal where an electrode meets a semiconductor Region.

    Attributes:
        name: Terminal name (e.g. ``"anode"``).
        electrode: Region name of the metal.
        region: Region name of the semiconductor it lands on.
        side: ``"p"`` or ``"n"``, the side of the Junction it is on.
    """

    name: str
    electrode: str
    region: str
    side: Literal["p", "n"]


@dataclass(frozen=True)
class Interface:
    """A boundary between two adjacent semiconductor Regions.

    Attributes:
        name: Physical-group name of the boundary.
        regions: The two Region names it joins.
    """

    name: str
    regions: tuple[str, str]


@dataclass(frozen=True)
class DeviceLayout:
    """Everything the Stages need that the device description implies.

    Attributes:
        region_spans: Extent of every Region on the Cross-section.
        doped_regions: The doped Region names, p side first.
        contacts: The derived Contacts.
        interfaces: The derived Interfaces, the Junction among them.
        junction: The Interface at the metallurgical PN boundary.
        window: ``(min, max)`` charge Window along the junction axis (um).
    """

    region_spans: dict[str, Span]
    doped_regions: tuple[str, ...]
    contacts: tuple[Contact, ...]
    interfaces: tuple[Interface, ...]
    junction: Interface
    window: tuple[float, float]

    @property
    def junction_position(self) -> float:
        """Where the metallurgical PN boundary sits on the Cross-section (um).

        The two Regions the Junction separates touch, so the overlap of
        their in-plane spans is the boundary itself.
        """
        left, right = (self.region_spans[name] for name in self.junction.regions)
        return 0.5 * (max(left.h[0], right.h[0]) + min(left.h[1], right.h[1]))

    @property
    def junction_span(self) -> Span:
        """Extent of the two Regions the metallurgical Junction separates.

        The rib the Junction sits in, in other words: where the carriers
        actually move, and the band a Staircase's Strips have to resolve
        however much wider than it they tile. Measured from the device
        description rather than declared alongside it.
        """
        left, right = (self.region_spans[name] for name in self.junction.regions)
        return Span(
            h=(min(left.h[0], right.h[0]), max(left.h[1], right.h[1])),
            z=(min(left.z[0], right.z[0]), max(left.z[1], right.z[1])),
        )

    @property
    def doped_span(self) -> tuple[float, float]:
        """Extent of every doped Region along the junction axis (um).

        The doped slab, pads included: what the charge solve covers, the
        guide's core, and the band the optical Staircase tiles by
        default. The RF Staircase tiles it too wherever the preset
        configures one, to carry the pads' series resistance into the
        line. As against :attr:`junction_span`, which is the rib alone.
        """
        spans = [self.region_spans[name].h for name in self.doped_regions]
        return (min(span[0] for span in spans), max(span[1] for span in spans))

    @property
    def guide_span_z(self) -> tuple[float, float]:
        """Vertical extent of the doped Regions a Mode is guided in (um)."""
        spans = [self.region_spans[name] for name in self.doped_regions]
        return (min(s.z[0] for s in spans), max(s.z[1] for s in spans))

    def window_around_junction(self, *, margin_um: float) -> tuple[float, float]:
        """A Window centred on the Junction, sized for a Mode.

        The Window a mode solve needs is a box around the rib, not the
        doped slab the charge solve spans (ADR 0002), so it is measured
        from the Junction outwards rather than from the Contacts inwards.

        Args:
            margin_um: Half-width of the Window either side of the
                Junction (um).

        Returns:
            ``(min, max)`` along the junction axis.
        """
        centre = self.junction_position
        return (centre - margin_um, centre + margin_um)

    def window_z_around_guide(
        self, *, above_um: float, below_um: float
    ) -> tuple[float, float]:
        """A vertical Window clearing the guiding layer by a margin.

        Args:
            above_um: Margin above the doped Regions (um).
            below_um: Margin below them (um).

        Returns:
            ``(min, max)`` vertical interval.
        """
        low, high = self.guide_span_z
        return (low - below_um, high + above_um)

    def is_below_junction(self, region: str) -> bool:
        """Whether a Region sits on the low side of the Junction.

        Which side of the metallurgical boundary a Region is on, read off
        the Cross-section: the answer a Stage needs to tell one flanking
        electrode from the other.

        Args:
            region: Region name.

        Returns:
            True when the Region's centre is below the Junction along the
            junction axis.
        """
        span = self.region_spans[region].h
        return 0.5 * (span[0] + span[1]) < self.junction_position

    def contact_on(self, side: Literal["p", "n"]) -> Contact:
        """The Contact on one side of the Junction.

        Args:
            side: ``"p"`` or ``"n"``.

        Returns:
            The single Contact on that side.
        """
        matches = [contact for contact in self.contacts if contact.side == side]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {side}-side contact, found "
                f"{[c.name for c in matches]}."
            )
        return matches[0]


def _region_spans(
    component: gf.Component,
    stack: LayerStack,
    *,
    axis: Literal["x", "y", "z"],
    value: float,
) -> dict[str, Span]:
    """Merge the Cross-section rectangles of each layer into one span."""
    from gsim.common.cross_section import (
        extract_xz_rectangles,
        extract_yz_rectangles,
    )

    if axis == "z":
        raise ValueError(
            "A modulator Study needs a vertical Cross-section; use an "
            "'x=<value>' or 'y=<value>' plane."
        )
    spans: dict[str, Span] = {}
    if axis == "x":
        intervals = [
            (rect.layer_name, rect.y0, rect.y1, rect.zmin, rect.zmax)
            for rect in extract_yz_rectangles(component, stack, value)
        ]
    else:
        intervals = [
            (rect.layer_name, rect.x0, rect.x1, rect.zmin, rect.zmax)
            for rect in extract_xz_rectangles(component, stack, value)
        ]
    for layer_name, low, high, z_low, z_high in intervals:
        existing = spans.get(layer_name)
        if existing is None:
            spans[layer_name] = Span(
                h=(float(low), float(high)),
                z=(float(z_low), float(z_high)),
            )
        else:
            spans[layer_name] = Span(
                h=(min(existing.h[0], float(low)), max(existing.h[1], float(high))),
                z=(
                    min(existing.z[0], float(z_low)),
                    max(existing.z[1], float(z_high)),
                ),
            )
    return spans


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Length of the overlap between two intervals (negative when apart)."""
    return min(a[1], b[1]) - max(a[0], b[0])


def _touching(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two intervals meet or overlap within the tolerance."""
    return _overlap(a, b) >= -TOUCH_TOL_UM


def _electrode_regions(
    stack: LayerStack, device: Device, spans: dict[str, Span]
) -> list[str]:
    """Electrode Region names, from the description or the stack."""
    if device.electrodes is not None:
        missing = [name for name in device.electrodes if name not in spans]
        if missing:
            raise ValueError(
                f"Electrodes {missing} are not on the cross-section. "
                f"Available regions: {sorted(spans)}"
            )
        return list(device.electrodes)
    return [name for name in stack.get_conductor_layers() if name in spans]


def _derive_contacts(
    device: Device, spans: dict[str, Span], electrodes: list[str]
) -> tuple[Contact, ...]:
    """Bind every electrode to the doped Region it lands on."""
    contacts: list[Contact] = []
    for electrode in electrodes:
        electrode_span = spans[electrode]
        candidates = [
            (region, _overlap(electrode_span.h, spans[region].h))
            for region in device.doped_regions
            if _overlap(electrode_span.h, spans[region].h) > TOUCH_TOL_UM
            and _touching(electrode_span.z, spans[region].z)
        ]
        if not candidates:
            continue
        region = max(candidates, key=lambda item: item[1])[0]
        side: Literal["p", "n"] = "p" if region in device.p_regions else "n"
        name = device.contact_names.get(
            electrode, "anode" if side == "p" else "cathode"
        )
        contacts.append(
            Contact(name=name, electrode=electrode, region=region, side=side)
        )
    if not contacts:
        raise ValueError(
            "No contact could be derived: none of the electrode regions "
            f"{electrodes} lands on a doped region "
            f"{device.doped_regions}. Name the electrodes explicitly with "
            "Device(electrodes=[...])."
        )
    names = [contact.name for contact in contacts]
    if len(set(names)) != len(names):
        raise ValueError(
            f"Derived contact names are not unique: {names}. Disambiguate "
            "them with Device(contact_names={electrode: name})."
        )
    return tuple(contacts)


def _interface_name(a: str, b: str) -> str:
    """Physical-group name of the boundary between two Regions."""
    return f"{a}_{b}"


def _derive_interfaces(
    device: Device, spans: dict[str, Span]
) -> tuple[tuple[Interface, ...], Interface]:
    """Every adjacent doped pair, and the one that is the Junction."""
    regions = device.doped_regions
    ordered = sorted(regions, key=lambda name: spans[name].h[0])
    adjacent: list[tuple[str, str]] = []
    for left, right in pairwise(ordered):
        if (
            abs(spans[right].h[0] - spans[left].h[1]) <= TOUCH_TOL_UM
            and _overlap(spans[left].z, spans[right].z) > TOUCH_TOL_UM
        ):
            adjacent.append((left, right))
    if not adjacent:
        raise ValueError(
            f"The doped regions {regions} do not touch on the cross-section, "
            "so no interface (and no junction) can be derived."
        )

    pn_pairs = [
        pair
        for pair in adjacent
        if (pair[0] in device.p_regions) != (pair[1] in device.p_regions)
    ]
    if device.junction is not None:
        wanted = set(device.junction)
        matching = [pair for pair in adjacent if set(pair) == wanted]
        if not matching:
            raise ValueError(
                f"The junction regions {device.junction} are not adjacent on "
                f"the cross-section; adjacent doped pairs are {adjacent}."
            )
        junction_pair = matching[0]
    elif len(pn_pairs) == 1:
        junction_pair = pn_pairs[0]
    elif not pn_pairs:
        raise ValueError(
            "No junction: no p-doped region is adjacent to an n-doped one "
            f"among {adjacent}. Check the device description."
        )
    else:
        raise ValueError(
            f"Several p/n boundaries {pn_pairs} could be the junction. Name "
            "it with Device(junction=(p_region, n_region))."
        )

    junction = Interface(name="junction", regions=junction_pair)
    interfaces = tuple(
        junction
        if pair == junction_pair
        else Interface(name=_interface_name(*pair), regions=pair)
        for pair in adjacent
    )
    return interfaces, junction


def derive_layout(
    component: gf.Component,
    stack: LayerStack,
    device: Device,
    *,
    axis: Literal["x", "y", "z"] = "x",
    value: float = 0.0,
) -> DeviceLayout:
    """Derive Contacts, Interfaces, the Junction and the charge Window.

    Args:
        component: The drawn device.
        stack: The layer stack the Regions are named in.
        device: The device description.
        axis: Cross-section normal axis.
        value: Cross-section plane coordinate (um).

    Returns:
        The :class:`DeviceLayout`.

    Raises:
        ValueError: When a named Region is not on the Cross-section, when
            no Contact can be bound, or when the Junction is missing or
            ambiguous.
    """
    spans = _region_spans(component, stack, axis=axis, value=value)
    missing = [name for name in device.doped_regions if name not in spans]
    if missing:
        raise ValueError(
            f"Doped regions {missing} are not on the cross-section at "
            f"{axis}={value}. Available regions: {sorted(spans)}"
        )

    electrodes = _electrode_regions(stack, device, spans)
    contacts = _derive_contacts(device, spans, electrodes)
    interfaces, junction = _derive_interfaces(device, spans)

    low = min(spans[name].h[0] for name in device.doped_regions)
    high = max(spans[name].h[1] for name in device.doped_regions)
    window = (low - device.window_margin_um, high + device.window_margin_um)

    return DeviceLayout(
        region_spans=spans,
        doped_regions=tuple(device.doped_regions),
        contacts=contacts,
        interfaces=interfaces,
        junction=junction,
        window=window,
    )
