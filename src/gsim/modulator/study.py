"""One device, one set of conditions, every Stage's configuration and result.

A Study holds the component, the layer stack and the device description,
derives from them everything the Stages need, and exposes each Stage as a
callable section (ADR 0001)::

    study = Study(component=comp, stack=stack, device=device)
    study.charge(biases=[0.0, -1.0, -2.0])
    sweep = study.charge.run()

Stages solve lazily and cache: running one twice costs one solve, and
re-configuring a Stage clears its result and every downstream Stage's.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gsim.common.cross_section import parse_plane_spec
from gsim.modulator.carriers import CarriersStage
from gsim.modulator.charge import ChargeStage
from gsim.modulator.device import Device
from gsim.modulator.layout import DeviceLayout, derive_layout
from gsim.modulator.line import LineStage
from gsim.modulator.optical import OpticalStage
from gsim.modulator.rf import RFStage

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.cross_section import PolygonXY2D, Rect2D, RectYZ2D
    from gsim.common.stack.extractor import LayerStack
    from gsim.modulator.report import TWMZMReport
    from gsim.modulator.stage import Stage

__all__ = ["Study"]

#: Each Stage and the Stages whose results it consumes. A Stage's result
#: is cleared by any change upstream of it, and only by those: the two EM
#: Stages both read the carriers Stage, and neither reads the other, and
#: the line Stage reads them both.
STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "charge": (),
    "carriers": ("charge",),
    "optical": ("carriers",),
    "rf": ("carriers",),
    "line": ("optical", "rf"),
}

#: Stage order, upstream first — the order a Study reports its Stages in.
STAGE_ORDER: tuple[str, ...] = tuple(STAGE_DEPENDENCIES)


def _dependents_of(name: str) -> list[str]:
    """Every Stage that reads *name*'s result, directly or through another.

    Args:
        name: Stage name.

    Returns:
        The dependent Stage names, in Stage order.
    """
    dependents = {
        stage for stage, upstream in STAGE_DEPENDENCIES.items() if name in upstream
    }
    # Grow to a fixed point rather than in one pass, so the answer does
    # not depend on the order the dependencies happen to be declared in.
    while True:
        grown = dependents | {
            stage
            for stage, upstream in STAGE_DEPENDENCIES.items()
            if dependents & set(upstream)
        }
        if grown == dependents:
            return [stage for stage in STAGE_ORDER if stage in dependents]
        dependents = grown


class Study:
    """A modulator device investigated across one set of conditions.

    Re-assigning :attr:`component`, :attr:`stack`, :attr:`device` or
    :attr:`plane` drops the derived layout and every Stage's result, the
    same way re-configuring a Stage does.

    Attributes:
        component: The drawn device.
        stack: The layer stack its Regions are named in.
        device: The device description every derivation starts from.
        plane: Cross-section plane spec (e.g. ``"x=0"``).
        verbose: Print one line per Stage on entry and exit.
        charge: The charge Stage section.
        carriers: The carrier-coupling Stage section.
        optical: The optical-Mode Stage section.
        rf: The RF line-parameter Stage section.
        line: The whole-device figures-of-merit Stage section.
    """

    def __init__(
        self,
        *,
        component: gf.Component,
        stack: LayerStack,
        device: Device | dict[str, Any],
        plane: str = "x=0",
        output_dir: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        """Build a Study over one device.

        Args:
            component: The drawn device.
            stack: The layer stack its Regions are named in.
            device: The device description (or its keyword arguments).
            plane: Cross-section plane spec, e.g. ``"x=0"``.
            output_dir: Directory for meshes and solver files; a temporary
                directory is used when omitted.
            verbose: Print one line per Stage on entry and exit.
        """
        self._component = component
        self._stack = stack
        self._device = (
            device if isinstance(device, Device) else Device.model_validate(device)
        )
        self._plane = plane
        self.verbose = verbose
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._layout: DeviceLayout | None = None
        self._section: list[Rect2D] | list[RectYZ2D] | list[PolygonXY2D] | None = None

        self.charge = ChargeStage()
        self.carriers = CarriersStage()
        self.optical = OpticalStage()
        self.rf = RFStage()
        self.line = LineStage()
        self._wire_stages()

    # ------------------------------------------------------------------
    # Stage wiring
    # ------------------------------------------------------------------

    @property
    def stages(self) -> dict[str, Stage]:
        """Every Stage of this Study, in dependency order."""
        return {name: getattr(self, name) for name in STAGE_ORDER}

    def _wire_stages(self) -> None:
        """Attach each Stage to this Study and to the Stages it feeds."""
        stages = self.stages
        for name, stage in stages.items():
            stage.wire(
                study=self,
                downstream=[stages[other] for other in _dependents_of(name)],
                is_verbose=lambda: self.verbose,
            )

    def invalidate(self) -> None:
        """Drop the derived layout and every Stage's result."""
        self._layout = None
        self._section = None
        for stage in self.stages.values():
            stage.invalidate()

    # ------------------------------------------------------------------
    # Device description
    # ------------------------------------------------------------------

    @property
    def component(self) -> gf.Component:
        """The drawn device."""
        return self._component

    @component.setter
    def component(self, value: gf.Component) -> None:
        """Redraw the device, dropping everything derived from it."""
        self._component = value
        self.invalidate()

    @property
    def stack(self) -> LayerStack:
        """The layer stack the device's Regions are named in."""
        return self._stack

    @stack.setter
    def stack(self, value: LayerStack) -> None:
        """Replace the stack, dropping everything derived from it."""
        self._stack = value
        self.invalidate()

    @property
    def device(self) -> Device:
        """The device description every derivation starts from."""
        return self._device

    @device.setter
    def device(self, value: Device | dict[str, Any]) -> None:
        """Replace the description, dropping everything derived from it."""
        self._device = (
            value if isinstance(value, Device) else Device.model_validate(value)
        )
        self.invalidate()

    @property
    def plane(self) -> str:
        """Cross-section plane spec, e.g. ``"x=0"``."""
        return self._plane

    @plane.setter
    def plane(self, value: str) -> None:
        """Move the Cross-section, dropping everything derived from it."""
        parse_plane_spec(value)
        self._plane = value
        self.invalidate()

    # ------------------------------------------------------------------
    # Derived device
    # ------------------------------------------------------------------

    @property
    def layout(self) -> DeviceLayout:
        """Contacts, Interfaces, the Junction and the charge Window.

        Derived from the device description against the drawn
        Cross-section, and cached.
        """
        if self._layout is None:
            axis, value = parse_plane_spec(self.plane)
            self._layout = derive_layout(
                self.component,
                self.stack,
                self.device,
                axis=axis,
                value=value,
            )
        return self._layout

    @property
    def section(self) -> list[Rect2D] | list[RectYZ2D] | list[PolygonXY2D]:
        """The drawn Cross-section's Regions, as rectangles.

        Every Region the drawn device puts on the plane, with its
        material and its extent — what a Stage needs to redraw the device
        around something of its own (a Staircase, say) rather than solve
        that something in isolation. Derived once and cached.
        """
        if self._section is None:
            from gsim.common.cross_section import extract_plane_section

            axis, value = parse_plane_spec(self.plane)
            self._section = extract_plane_section(
                self.component.copy(), self.stack, axis=axis, value=value
            )
        return self._section

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, *, force: bool = False) -> TWMZMReport:
        """The whole-device figures of merit, running what has not run.

        The report is the line Stage's result, so asking for it runs the
        optical and RF Stages first where they hold no result, and costs
        nothing where they do.

        Args:
            force: Re-combine even when a report is already held. The
                Stages upstream keep their results; force one of those to
                re-solve through its own ``run(force=True)``.

        Returns:
            The assembled device report.
        """
        report: TWMZMReport = self.line.run(force=force)
        return report

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        """Directory holding the Study's meshes and solver files."""
        if self._output_dir is None:
            self._output_dir = Path(tempfile.mkdtemp(prefix="gsim-modulator-"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir

    def stage_dir(self, stage_name: str) -> Path:
        """Directory a Stage writes into.

        Args:
            stage_name: Name of the Stage.

        Returns:
            The (created) per-Stage output directory.
        """
        path = self.output_dir / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path
