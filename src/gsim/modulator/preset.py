"""One call from a drawn device to a Study with every Stage configured.

A Study configures five Stages, and a lateral PN Phase shifter under a
Traveling-wave electrode wants roughly the same five configurations every
time. The preset here supplies them, interpreting the user's own
component and stack through the device description naming which Regions
are p and which are n: Contacts, Interfaces and every Stage's Window are
derived from that (ADR 0002), not declared a second time.

It generates no geometry. The device is always the caller's, and
:func:`~gsim.modulator.demo.demo_phase_shifter` is the separate piece of
scaffolding that draws one when an example needs a device to point at.

Nothing it sets is locked: the preset writes defaults into each Stage's
settings, and every section stays callable afterwards::

    study = pn_phase_shifter(component=comp, stack=stack, device=device)
    study.optical(wavelength_um=1.31)  # still yours to change
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from gsim.common.carriers import PlasmaDispersionModel
from gsim.modulator.study import Study

if TYPE_CHECKING:
    from pathlib import Path

    import gdsfactory as gf

    from gsim.common.stack.extractor import LayerStack
    from gsim.modulator.device import Device
    from gsim.modulator.route import EMRoute
    from gsim.modulator.staircase import ElectrodeSpec

__all__ = ["pn_phase_shifter"]

#: Biases the charge Stage sweeps unless the caller says otherwise (V).
#: They are applied to the swept Contact, which is the cathode, so
#: positive values reverse-bias the Junction — the regime a depletion-mode
#: Phase shifter is driven in.
DEFAULT_BIASES_V: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)

#: Frequencies the RF Stage solves at unless the caller says otherwise.
DEFAULT_FREQUENCIES_HZ: tuple[float, ...] = (10e9, 20e9, 40e9, 60e9, 80e9, 100e9)

#: Strip count the RF Staircase is built with unless the caller says otherwise.
#: The Strips tile the whole doped slab, so they have to be narrow enough
#: that one of them sits inside the depletion region: five Strips across
#: the demo's slab average it away at 2 V, and the slab then shunts the
#: electrodes. Twenty-one leave a dielectric Strip at 2 V and land within
#: 4% of sixty-one on the line's index at 10 GHz and 1% at 100 GHz. The
#: optical Staircase takes the same count wherever its Route needs one.
DEFAULT_N_STRIPS: int = 21

#: Traveling-wave electrode length (um) unless the caller says otherwise.
DEFAULT_LENGTH_UM: float = 3000.0

#: The plasma-dispersion coefficient sets shipped with gsim, by the
#: wavelength each was fitted at. The preset picks the one matching the
#: optical solve rather than leaving a 1.55 um fit to answer a 1.31 um
#: question.
FITTED_DISPERSION: tuple[PlasmaDispersionModel, ...] = (
    PlasmaDispersionModel.nedeljkovic_1550(),
    PlasmaDispersionModel.nedeljkovic_1310(),
)

#: Wavelengths within this of a fit count as that fit's wavelength (um).
WAVELENGTH_TOL_UM: float = 1e-3


def _dispersion_for(wavelength_um: float) -> PlasmaDispersionModel:
    """The shipped coefficient set closest to *wavelength_um*.

    Args:
        wavelength_um: Wavelength the optical Stage solves at (um).

    Returns:
        The fit at that wavelength, or the nearest one — with a warning,
        because coefficients read off the wrong fit shift the index and
        inflate the free-carrier absorption silently.
    """
    nearest = min(
        FITTED_DISPERSION, key=lambda model: abs(model.wavelength_um - wavelength_um)
    )
    if abs(nearest.wavelength_um - wavelength_um) <= WAVELENGTH_TOL_UM:
        return nearest
    fitted = ", ".join(f"{model.wavelength_um:g}" for model in FITTED_DISPERSION)
    warnings.warn(
        f"No plasma-dispersion coefficients are fitted at "
        f"{wavelength_um:g} um; gsim ships fits at {fitted} um. The "
        f"nearest, at {nearest.wavelength_um:g} um, is standing in, so the "
        "index shift and the free-carrier absorption are extrapolated. "
        "Supply your own with pn_phase_shifter(dispersion=...) or "
        "study.carriers(dispersion=...).",
        UserWarning,
        stacklevel=3,
    )
    return nearest


def pn_phase_shifter(
    *,
    component: gf.Component,
    stack: LayerStack,
    device: Device | dict[str, Any],
    plane: str = "x=0",
    biases: Sequence[float] = DEFAULT_BIASES_V,
    wavelength_um: float = 1.55,
    frequencies_hz: Sequence[float] = DEFAULT_FREQUENCIES_HZ,
    n_strips: int = DEFAULT_N_STRIPS,
    electrodes: ElectrodeSpec | None = None,
    length_um: float = DEFAULT_LENGTH_UM,
    n_group: float | None = None,
    route: EMRoute = "femwell",
    dispersion: PlasmaDispersionModel | dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = False,
) -> Study:
    """A fully configured Study over a lateral PN Phase shifter.

    The device description is interpreted against the drawn Cross-section
    before the call returns, so one this preset cannot make sense of — a
    Region that is not drawn, an ambiguous Junction, a Junction with a
    Contact on only one side — is reported here, naming what was missing,
    rather than surfacing later as a meshing failure.

    Args:
        component: The drawn device. Never generated here.
        stack: The layer stack its Regions are named in.
        device: The device description — which Regions are p, which are
            n, and anything else
            :class:`~gsim.modulator.device.Device` takes — as a
            ``Device`` or its keyword arguments.
        plane: Cross-section plane spec, e.g. ``"x=0"``.
        biases: Biases the charge Stage sweeps (V), applied to the
            cathode, so positive values reverse-bias the Junction.
        wavelength_um: Vacuum wavelength the optical Stage solves at (um),
            which also selects the plasma-dispersion coefficients.
        frequencies_hz: Frequencies the RF Stage solves at (Hz).
        n_strips: Strips the RF Staircase is built with, and the optical
            one wherever the optical Stage needs a Staircase at all —
            which the Palace Route does and the femwell Route does not.
            On the RF Staircase these Strips cross the Junction extent,
            and every other doped Region of the slab takes Strips of its
            own.
        electrodes: The Traveling-wave electrodes flanking the RF
            Staircase; the RF Stage's default pair when omitted.
        length_um: Length of the Traveling-wave electrode (um).
        n_group: Optical group index the Velocity mismatch is measured
            against — a measured one, say; the optical Stage computes it,
            with two more solves, when left unset.
        route: Backend both EM Stages use — ``"femwell"`` or ``"palace"``.
        dispersion: Plasma-dispersion coefficients replacing the fit
            selected from ``wavelength_um``; foundry-calibrated values go
            here.
        output_dir: Directory for meshes and solver files; a temporary
            directory is used when omitted.
        verbose: Print one line per Stage on entry and exit.

    Returns:
        The Study, with every Stage configured and none of them run.

    Raises:
        ValueError: When the device description cannot be interpreted
            against the drawn Cross-section at ``plane``.
    """
    study = Study(
        component=component,
        stack=stack,
        device=device,
        plane=plane,
        output_dir=output_dir,
        verbose=verbose,
    )

    # Derive now rather than at mesh time, so a description that cannot be
    # read fails here, where the message can still name what was missing.
    # Both Contacts are asked for by name: a Traveling-wave electrode that
    # misses its pad leaves the layout one Contact short, and every Stage
    # downstream would otherwise fail on the absence rather than the cause.
    try:
        layout = study.layout
        layout.contact_on("p")
        layout.contact_on("n")
    except ValueError as error:
        raise ValueError(
            f"pn_phase_shifter cannot derive this device's layout on the "
            f"cross-section at {plane}: {error}"
        ) from error

    study.charge(biases=list(biases))
    study.carriers(
        dispersion=dispersion
        if dispersion is not None
        else _dispersion_for(wavelength_um)
    )
    # Both EM Stages state their Strip extent here, in one place. The
    # optical Staircase tiles the doped slab because that is what the
    # Carrier map covers and what the guide's core is made of; tiling the
    # rib alone leaves the pads to the surrounding Regions, which carry
    # the drawn material and no carriers.
    study.optical(
        route=route, wavelength_um=wavelength_um, strip_span=layout.doped_span
    )
    # The optical Stage keeps the continuous profile wherever its Route
    # can carry one; where it cannot, the Stage says it needs a strip
    # count, and the preset's is the one it gets.
    if study.optical.effective_n_strips() is not None:
        study.optical(n_strips=n_strips)
    # The RF Staircase tiles the doped slab rather than the rib alone:
    # the pads are part of the line the drive sees, and a Staircase that
    # stops at the rib puts the electrodes against it, dropping their
    # series resistance. The rib keeps n_strips however wide the slab.
    study.rf(
        route=route,
        frequencies_hz=list(frequencies_hz),
        n_strips=n_strips,
        strip_span=layout.doped_span,
    )
    if electrodes is not None:
        study.rf(electrodes=electrodes)
    study.line(length_um=length_um, n_group=n_group)
    return study
