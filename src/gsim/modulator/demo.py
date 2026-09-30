"""A lateral PN Phase shifter, drawn from nothing, for examples and tests.

This is scaffolding, not an entry point. A real Study is built over a
device the user already drew: their own component, their own layer stack,
and the device description naming which Regions are p and which are n. The
builder here exists so that a notebook or a test which needs *some*
device to talk about does not carry sixty lines of component construction
before it gets to the point.

What it draws is the canonical lateral PN Phase shifter: a silicon rib
split into four doped Regions along the junction axis
(``n_pad | n_rib | p_rib | p_pad``), with a metal electrode landing on
each outer pad. The dimensions are all arguments, so the shape can be
stretched, but nothing about it is calibrated to any foundry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from gsim.modulator.device import Device

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack.extractor import LayerStack
    from gsim.modulator.staircase import ElectrodeSpec
    from gsim.tcad.doping import StepDoping, TableDoping

__all__ = [
    "DemoPhaseShifter",
    "RibPhaseShifter",
    "demo_phase_shifter",
    "rib_phase_shifter",
]

#: Position of the Junction on the layout's junction axis (um).
DEFAULT_CENTER_UM: float = -20.0

#: Width of each junction flank (um).
DEFAULT_HALF_WIDTH_UM: float = 0.3

#: Width of each contact pad (um).
DEFAULT_PAD_WIDTH_UM: float = 0.3

#: Height of the silicon rib (um).
DEFAULT_RIB_HEIGHT_UM: float = 0.22

#: Thickness of the metal over each pad (um).
DEFAULT_ELECTRODE_THICKNESS_UM: float = 0.5

#: Drawn length of the device (um). The solves are all Cross-section
#: physics, so this only has to be long enough to cut a plane from.
DEFAULT_LENGTH_UM: float = 10.0

#: Width of the drawn waveguide core (um).
DEFAULT_WAVEGUIDE_WIDTH_UM: float = 0.4

#: Region names the builder draws, low side of the junction axis first.
REGION_NAMES: tuple[str, str, str, str] = ("n_pad", "n_rib", "p_rib", "p_pad")

#: Electrode Region name over each outer pad, in the same order.
ELECTRODE_NAMES: dict[str, str] = {
    "n_pad": "cathode_metal",
    "p_pad": "anode_metal",
}

#: ``(layer, datatype)`` the doped Regions are drawn on, Region ``i``
#: taking ``datatype + i``.
DOPING_LAYER: tuple[int, int] = (30, 0)

#: ``(layer, datatype)`` the electrodes are drawn on. Outside the generic
#: PDK's own layers, like :data:`DOPING_LAYER`: an electrode drawn on the
#: PDK's ``metal1`` layer is read twice — once as the electrode this
#: builder declares, at the rib's own height, and once as ``metal1``, a
#: micron higher — so one drawn rectangle becomes two stacked conductors
#: and the Cross-section grows a metal ceiling nobody asked for.
ELECTRODE_LAYER: tuple[int, int] = (46, 0)


@dataclass(frozen=True)
class DemoPhaseShifter:
    """A drawn demo device and the description that interprets it.

    Attributes:
        component: The drawn device.
        stack: The layer stack its Regions are named in.
        device: The device description matching what was drawn, ready to
            hand to :func:`~gsim.modulator.preset.pn_phase_shifter`.
        center_um: Position of the Junction on the layout's junction axis (um).
        half_width_um: Width of each junction flank (um).
        pad_width_um: Width of each contact pad (um).
        rib_height_um: Height of the silicon rib (um).
        electrode_thickness_um: Thickness of the metal over each pad (um).
        waveguide_width_um: Width of the drawn waveguide core (um).
        length_um: Drawn length of the device (um); every solve is
            Cross-section physics, so this only has to be long enough to
            cut a plane from.
    """

    component: gf.Component
    stack: LayerStack
    device: Device
    center_um: float
    half_width_um: float
    pad_width_um: float
    rib_height_um: float
    electrode_thickness_um: float
    waveguide_width_um: float
    length_um: float


def _region_spans(
    *, center_um: float, half_width_um: float, pad_width_um: float
) -> dict[str, tuple[float, float]]:
    """Extent of each doped Region along the junction axis (um)."""
    return {
        "n_pad": (center_um - half_width_um - pad_width_um, center_um - half_width_um),
        "n_rib": (center_um - half_width_um, center_um),
        "p_rib": (center_um, center_um + half_width_um),
        "p_pad": (center_um + half_width_um, center_um + half_width_um + pad_width_um),
    }


def demo_phase_shifter(
    *,
    center_um: float = DEFAULT_CENTER_UM,
    half_width_um: float = DEFAULT_HALF_WIDTH_UM,
    pad_width_um: float = DEFAULT_PAD_WIDTH_UM,
    rib_height_um: float = DEFAULT_RIB_HEIGHT_UM,
    electrode_thickness_um: float = DEFAULT_ELECTRODE_THICKNESS_UM,
    length_um: float = DEFAULT_LENGTH_UM,
    waveguide_width_um: float = DEFAULT_WAVEGUIDE_WIDTH_UM,
    substrate_thickness_um: float = 2.0,
    permittivity: float = 11.9,
    sigma_s_per_m: float = 1.6e3,
    p_doping_cm3: float = 1e18,
    n_doping_cm3: float = 1e18,
) -> DemoPhaseShifter:
    """Draw a lateral PN Phase shifter and describe it.

    Scaffolding for examples and tests. A Study over a real device takes
    the user's own component and stack; this builder is only here so that
    example code has a device to point at.

    Args:
        center_um: Position of the Junction on the layout's junction axis (um).
        half_width_um: Width of each junction flank (um).
        pad_width_um: Width of each contact pad (um).
        rib_height_um: Height of the silicon rib (um).
        electrode_thickness_um: Thickness of the metal over each pad (um).
        length_um: Drawn length of the device (um).
        waveguide_width_um: Width of the drawn waveguide core (um).
        substrate_thickness_um: Substrate below ``z = 0`` (um).
        permittivity: Relative permittivity of the doped silicon.
        sigma_s_per_m: Background conductivity of the doped Regions (S/m).
        p_doping_cm3: Acceptor concentration of the p Regions (cm^-3).
        n_doping_cm3: Donor concentration of the n Regions (cm^-3).

    Returns:
        The drawn component, its stack, the matching device description,
        and the dimensions they were drawn with.
    """
    import gdsfactory as gf

    from gsim.common.cross_section import build_doped_cross_section
    from gsim.common.stack.extractor import Layer
    from gsim.common.stack.materials import make_doped_materials

    gf.gpdk.PDK.activate()
    component = gf.Component()
    waveguide = component << gf.c.rectangle(
        (length_um, waveguide_width_um), centered=True, layer=(1, 0)
    )
    waveguide.y = center_um
    slab = component << gf.c.rectangle((length_um, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0

    spans = _region_spans(
        center_um=center_um,
        half_width_um=half_width_um,
        pad_width_um=pad_width_um,
    )
    layer_specs: dict[str, Layer] = {}
    for index, name in enumerate(REGION_NAMES):
        low, high = spans[name]
        gds_layer = (DOPING_LAYER[0], DOPING_LAYER[1] + index)
        rect = component << gf.c.rectangle((length_um, high - low), layer=gds_layer)
        rect.y = 0.5 * (low + high)
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=0.0,
            zmax=rib_height_um,
            thickness=rib_height_um,
            material=name,
            layer_type="dielectric",
            mesh_resolution="fine",
        )
    for index, (pad, electrode) in enumerate(ELECTRODE_NAMES.items()):
        low, high = spans[pad]
        gds_layer = (ELECTRODE_LAYER[0], ELECTRODE_LAYER[1] + index)
        rect = component << gf.c.rectangle((length_um, high - low), layer=gds_layer)
        rect.y = 0.5 * (low + high)
        layer_specs[electrode] = Layer(
            name=electrode,
            gds_layer=gds_layer,
            zmin=rib_height_um,
            zmax=rib_height_um + electrode_thickness_um,
            thickness=electrode_thickness_um,
            material="aluminum",
            layer_type="conductor",
            mesh_resolution="fine",
        )

    materials = make_doped_materials(
        [(name, sigma_s_per_m) for name in REGION_NAMES],
        permittivity=permittivity,
    )
    stack, _section = build_doped_cross_section(
        component,
        axis="x",
        value=0.0,
        substrate_thickness=substrate_thickness_um,
        doping={"layer_specs": layer_specs, "materials": materials},
        verbose=False,
    )

    device = Device(
        p_regions=["p_rib", "p_pad"],
        n_regions=["n_rib", "n_pad"],
        p_doping_cm3=p_doping_cm3,
        n_doping_cm3=n_doping_cm3,
    )
    return DemoPhaseShifter(
        component=component,
        stack=stack,
        device=device,
        center_um=center_um,
        half_width_um=half_width_um,
        pad_width_um=pad_width_um,
        rib_height_um=rib_height_um,
        electrode_thickness_um=electrode_thickness_um,
        waveguide_width_um=waveguide_width_um,
        length_um=length_um,
    )


#: Doped Regions of :func:`rib_phase_shifter`, low side of the junction
#: axis first: ``(name, dopant type, kind)``, where the kind names the
#: dimension and doping level the Region takes.
RIB_REGIONS: tuple[tuple[str, Literal["donor", "acceptor"], str], ...] = (
    ("n_contact", "donor", "contact"),
    ("n_plus", "donor", "plus"),
    ("n_slab", "donor", "core"),
    ("n_rib", "donor", "rib"),
    ("p_rib", "acceptor", "rib"),
    ("p_slab", "acceptor", "core"),
    ("p_plus", "acceptor", "plus"),
    ("p_contact", "acceptor", "contact"),
)


#: Half-extent of the samples taken around each smeared step, in
#: straggles: past it the error function is flat to double precision.
GRADED_REACH_STRAGGLES: float = 6.0

#: Samples per straggle around each smeared step. Interpolated linearly
#: between them, the profile is within a few parts in 1e4 of its level.
GRADED_SAMPLES_PER_STRAGGLE: int = 10

#: Concentration below which a smeared dopant is left out of a Region
#: (cm^-3): ten orders under the intrinsic density, so nothing reads it.
GRADED_FLOOR_CM3: float = 1.0


def _graded_doping(
    spans: dict[str, tuple[float, float]],
    steps: list[StepDoping],
    straggle_um: float,
) -> list[TableDoping]:
    """The step doping of a row of Regions, smeared along the junction axis.

    Each dopant's piecewise-constant level is convolved with a Gaussian of
    standard deviation ``straggle_um``, which turns every step into an
    error function and carries each dopant into the Regions beside its
    own: donors and acceptors overlap across the Junction, and partly
    compensate there. The outer ends of the row are left unsmeared, as an
    implant mask opening reaching past the drawn device would leave them.

    A profile reaches the solve on the nodes of the Region it names, so
    each Region takes one sampled profile per dopant present in it, dense
    around the steps and flat between them.

    Args:
        spans: Extent of each Region along the junction axis (um).
        steps: The step profile of each Region, low side first.
        straggle_um: Lateral straggle (um), positive.

    Returns:
        The graded profiles, Region by Region in the order given.
    """
    import numpy as np
    from scipy.special import erf

    from gsim.tcad.doping import TableDoping

    first, last = steps[0].region, steps[-1].region
    edges = sorted({edge for span in spans.values() for edge in span})[1:-1]
    reach = GRADED_REACH_STRAGGLES * straggle_um
    around_a_step = np.linspace(
        -reach,
        reach,
        2 * round(GRADED_REACH_STRAGGLES * GRADED_SAMPLES_PER_STRAGGLE) + 1,
    )

    def smeared(dopant: str, x: np.ndarray) -> np.ndarray:
        """One dopant's smeared concentration (cm^-3) at positions in um."""
        total = np.zeros_like(x)
        for step in steps:
            if step.dopant_type != dopant:
                continue
            low, high = spans[step.region]
            rise = (
                1.0
                if step.region == first
                else 0.5 * (1.0 + erf((x - low) / (np.sqrt(2.0) * straggle_um)))
            )
            fall = (
                0.0
                if step.region == last
                else 0.5 * (1.0 + erf((x - high) / (np.sqrt(2.0) * straggle_um)))
            )
            total = total + step.concentration_cm3 * (rise - fall)
        return total

    graded: list[TableDoping] = []
    for step in steps:
        low, high = spans[step.region]
        near = [
            edge + around_a_step for edge in edges if low - reach < edge < high + reach
        ]
        x = np.unique(np.clip(np.concatenate([[low, high], *near]), low, high))
        for dopant in ("donor", "acceptor"):
            values = smeared(dopant, x)
            if values.max() < GRADED_FLOOR_CM3:
                continue
            graded.append(
                TableDoping(
                    region=step.region,
                    dopant_type=dopant,
                    x_um=x.tolist(),
                    values_cm3=values.tolist(),
                )
            )
    return graded


@dataclass(frozen=True)
class RibPhaseShifter:
    """A drawn rib-waveguide Phase shifter, its description and its line.

    Attributes:
        component: The drawn device.
        stack: The layer stack its Regions are named in.
        device: The device description matching what was drawn, doping
            per Region included, ready to hand to
            :func:`~gsim.modulator.preset.pn_phase_shifter`.
        electrodes: The Traveling-wave electrodes the RF Staircase
            flanks its Strips with, sized like the drawn metal.
        center_um: Position of the Junction on the junction axis (um).
        rib_width_um: Width of the rib (um).
        rib_height_um: Height of the rib (um).
        slab_height_um: Height of the slab either side of it (um).
        doping_cm3: Drawn doping concentration per Region (cm^-3): the
            level each Region reaches away from its edges, graded or not.
        length_um: Drawn length of the device (um).
        lateral_straggle_um: Lateral straggle the doping steps are smeared
            by (um); zero for the abrupt device.
    """

    component: gf.Component
    stack: LayerStack
    device: Device
    electrodes: ElectrodeSpec
    center_um: float
    rib_width_um: float
    rib_height_um: float
    slab_height_um: float
    doping_cm3: dict[str, float]
    length_um: float
    lateral_straggle_um: float = 0.0


def rib_phase_shifter(
    *,
    center_um: float = DEFAULT_CENTER_UM,
    rib_width_um: float = 0.5,
    rib_height_um: float = 0.22,
    slab_height_um: float = 0.09,
    core_width_um: float = 0.5,
    plus_width_um: float = 1.0,
    contact_width_um: float = 1.0,
    p_core_cm3: float = 5e17,
    n_core_cm3: float = 3e17,
    plus_cm3: float = 1e19,
    contact_cm3: float = 1e20,
    lateral_straggle_um: float = 0.0,
    electrode_width_um: float = 10.0,
    electrode_thickness_um: float = 1.0,
    length_um: float = DEFAULT_LENGTH_UM,
    substrate_thickness_um: float = 2.0,
    permittivity: float = 11.9,
) -> RibPhaseShifter:
    """Draw a lateral PN Phase shifter as foundries build one.

    A silicon rib on a thinner slab, split by a lateral PN Junction at
    its centre. Each side, going out from the Junction: the lightly doped
    core — the rib half and a stretch of slab beside it, where the
    carriers the light sees move — then a moderately doped ``plus``
    Region, then a heavily doped ``contact`` Region under the metal,
    which keeps the Ohmic contact and the path to it from carrying the
    line's series resistance. The defaults are generic published values
    for a 220 nm silicon-on-insulator depletion modulator, not any
    foundry's.

    Left at zero, ``lateral_straggle_um`` dopes every Region at one level
    and leaves the Junction perfectly abrupt. A positive value smears
    every doping step into an error function of that standard deviation,
    as implant straggle and diffusion do: the same drawn device, with
    donors and acceptors overlapping across the Junction. The net doping
    changes sign at the drawn Junction when the two cores are doped
    alike, and a fraction of a straggle into the lighter one when they
    are not.

    The metal is drawn over each contact Region; the Traveling-wave
    electrodes the RF Staircase flanks its Strips with are returned
    alongside, as wide and thick as a real line's, since the drawn
    Cross-section only needs the metal where it lands.

    Args:
        center_um: Position of the Junction on the junction axis (um).
        rib_width_um: Width of the rib (um).
        rib_height_um: Height of the rib (um).
        slab_height_um: Height of the slab (um).
        core_width_um: Width of the lightly doped slab beside the rib (um).
        plus_width_um: Width of each moderately doped Region (um).
        contact_width_um: Width of each heavily doped contact Region (um).
        p_core_cm3: Acceptor concentration of the p core (cm^-3).
        n_core_cm3: Donor concentration of the n core (cm^-3).
        plus_cm3: Concentration of the moderately doped Regions (cm^-3).
        contact_cm3: Concentration of the contact Regions (cm^-3).
        lateral_straggle_um: Standard deviation every doping step is
            smeared by along the junction axis (um); zero keeps the steps.
        electrode_width_um: Width of each Traveling-wave electrode (um).
        electrode_thickness_um: Thickness of the electrode metal (um).
        length_um: Drawn length of the device (um).
        substrate_thickness_um: Substrate below ``z = 0`` (um).
        permittivity: Relative permittivity of the doped silicon.

    Returns:
        The drawn component, its stack, the matching device description
        and the Traveling-wave electrodes.
    """
    import gdsfactory as gf

    from gsim.common.cross_section import build_doped_cross_section
    from gsim.common.stack.extractor import Layer
    from gsim.common.stack.materials import make_doped_materials
    from gsim.modulator.staircase import ElectrodeSpec
    from gsim.tcad.doping import StepDoping

    if lateral_straggle_um < 0.0:
        raise ValueError(
            f"lateral_straggle_um must be zero or positive, got {lateral_straggle_um}."
        )

    widths = {
        "rib": rib_width_um / 2.0,
        "core": core_width_um,
        "plus": plus_width_um,
        "contact": contact_width_um,
    }
    heights = {
        "rib": rib_height_um,
        "core": slab_height_um,
        "plus": slab_height_um,
        "contact": slab_height_um,
    }
    levels = {"plus": plus_cm3, "contact": contact_cm3}

    spans: dict[str, tuple[float, float]] = {}
    edge = center_um - sum(widths.values())
    for name, _dopant, kind in RIB_REGIONS:
        spans[name] = (edge, edge + widths[kind])
        edge += widths[kind]

    gf.gpdk.PDK.activate()
    component = gf.Component()
    layer_specs: dict[str, Layer] = {}
    doping: list[StepDoping] = []
    doping_cm3: dict[str, float] = {}
    for index, (name, dopant, kind) in enumerate(RIB_REGIONS):
        low, high = spans[name]
        gds_layer = (DOPING_LAYER[0], DOPING_LAYER[1] + index)
        rect = component << gf.c.rectangle((length_um, high - low), layer=gds_layer)
        rect.y = 0.5 * (low + high)
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=0.0,
            zmax=heights[kind],
            thickness=heights[kind],
            material=name,
            layer_type="dielectric",
            # The core is where the depletion edge moves; the plus and
            # contact Regions only conduct.
            mesh_resolution="fine" if kind in ("rib", "core") else "medium",
        )
        concentration = levels.get(
            kind, p_core_cm3 if dopant == "acceptor" else n_core_cm3
        )
        doping_cm3[name] = concentration
        doping.append(
            StepDoping(region=name, dopant_type=dopant, concentration_cm3=concentration)
        )

    for index, (pad, electrode) in enumerate(
        (("n_contact", "cathode_metal"), ("p_contact", "anode_metal"))
    ):
        low, high = spans[pad]
        gds_layer = (ELECTRODE_LAYER[0], ELECTRODE_LAYER[1] + index)
        rect = component << gf.c.rectangle((length_um, high - low), layer=gds_layer)
        rect.y = 0.5 * (low + high)
        layer_specs[electrode] = Layer(
            name=electrode,
            gds_layer=gds_layer,
            zmin=slab_height_um,
            zmax=slab_height_um + electrode_thickness_um,
            thickness=electrode_thickness_um,
            material="aluminum",
            layer_type="conductor",
            mesh_resolution="fine",
        )

    materials = make_doped_materials(
        [(name, 0.0) for name, _dopant, _kind in RIB_REGIONS],
        permittivity=permittivity,
    )
    stack, _section = build_doped_cross_section(
        component,
        axis="x",
        value=0.0,
        substrate_thickness=substrate_thickness_um,
        doping={"layer_specs": layer_specs, "materials": materials},
        verbose=False,
    )

    device = Device(
        p_regions=[name for name, dopant, _ in RIB_REGIONS if dopant == "acceptor"],
        n_regions=[name for name, dopant, _ in RIB_REGIONS if dopant == "donor"],
        # The core levels, which are the Junction's: what an abrupt-junction
        # estimate of the capacitance reads.
        p_doping_cm3=p_core_cm3,
        n_doping_cm3=n_core_cm3,
        doping=(
            _graded_doping(spans, doping, lateral_straggle_um)
            if lateral_straggle_um > 0.0
            else doping
        ),
    )
    return RibPhaseShifter(
        component=component,
        stack=stack,
        device=device,
        electrodes=ElectrodeSpec(
            width_um=electrode_width_um, thickness_um=electrode_thickness_um
        ),
        center_um=center_um,
        rib_width_um=rib_width_um,
        rib_height_um=rib_height_um,
        slab_height_um=slab_height_um,
        doping_cm3=doping_cm3,
        length_um=length_um,
        lateral_straggle_um=lateral_straggle_um,
    )
