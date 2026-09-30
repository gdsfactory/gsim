"""The Palace Route: the second first-class Backend, run as a binary.

Palace's *text* results report Modes as effective indices rather than as
field vectors, so this Route cannot say how much of a Mode sits on the
Window wall or outside the Strips, and says so. What it can do is
integrate the line's voltage and current itself: the Route declares the
paths on the simulation before the solve, reads the characteristic
impedance off Palace's own tables under the index the simulation
assigned, and tells the wall Mode from the line Mode by the voltage the
gap carries against the Mode's power (ADR 0005). Its electrode metal
defaults to a perfect conductor (ADR 0003), because a metal Region takes
Palace's eigenvalue search over.

None of that integration is written here. Sizing the two paths, reading
the tables and falling back to the saved fields are statements about
Palace and live in :mod:`gsim.palace.line_impedance`; salvaging a
crashed run's table and diagnosing a dead binary are statements about a
local Palace run and live on the simulation. What is left in this file
is what names a Stage, a Window or a Staircase: the Route's own
vocabulary.

The Palace binary is resolved here, once per Stage run, and threaded
through nothing.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from gsim.common.modes import Conductor, LineReading
from gsim.modulator.route import EMRoute, Route

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from gsim.modulator.staircase import (
        ConductorModel,
        StaircaseCrossSection,
        SurroundingRegion,
    )
    from gsim.palace import BoundaryModeSim
    from gsim.palace.results import PalaceTextResults

__all__ = [
    "PalaceMode",
    "PalaceRoute",
    "PalaceSolve",
    "conductor_clearance",
    "containment_unmeasurable",
    "solve_palace_modes",
]


@dataclass(frozen=True)
class PalaceMode:
    """One Mode of a Palace ``BoundaryMode`` solve.

    Palace's *text* results report Modes as effective indices rather than
    as field vectors, so a Palace Mode carries its index and its
    solver-assigned number and nothing else. What else the solve
    measured is on disk under the same number: the characteristic
    impedance its postprocessing paths integrate (``mode-Z.csv``, read
    by :func:`gsim.palace.line_impedance.native_line_impedance`) and,
    when the solve was asked to save them, the fields themselves
    (ParaView, read by
    :func:`gsim.palace.line_impedance.field_line_impedance`). Neither is
    in hand at selection time,
    so anything a Stage measures while choosing a Mode (the
    Window-containment ratio) stays a femwell-Route capability rather
    than something to invent.

    Attributes:
        n_eff: Complex effective index (``exp(+i omega t)`` convention).
        mode_id: Palace's own mode number, 1-based. This is also the
            ParaView cycle its saved fields land in.
    """

    n_eff: complex
    mode_id: int


@dataclass(frozen=True)
class PalaceSolve:
    """One Palace ``BoundaryMode`` run: its Modes and the tables it wrote.

    Attributes:
        modes: The solved Modes, in Palace's own mode order.
        results: The run's text results, off which a declared impedance
            path's reading is taken.
    """

    modes: list[PalaceMode]
    results: PalaceTextResults


def _palace_hint(stage_name: str) -> str:
    """The error a user selecting the Palace Route without Palace gets.

    What the Route knows that :func:`gsim.palace.runtime.require_palace_binary`
    does not: which Stage asked, and that there is another Route to ask
    instead.
    """
    return (
        f"The {stage_name} stage is routed to Palace, but no Palace binary "
        "was found. Point PALACE_BIN at one, put 'palace' on PATH, or go "
        f"back to the default route with study.{stage_name}(route='femwell')."
    )


#: The way back off this Route, named at the end of a Palace runtime
#: failure. The simulation writes the rest of that report and knows
#: nothing about Routes, so the one modulator-shaped sentence in it is
#: handed down from here.
_FEMWELL_REMEDY: str = "re-solve on the default route with route='femwell'"


def solve_palace_modes(
    sim: BoundaryModeSim,
    *,
    freq_hz: float,
    num_modes: int,
    binary: Path,
    target: float = 0.0,
    save: int = 0,
    verbose: bool = False,
) -> PalaceSolve:
    """Solve one ``BoundaryMode`` problem with Palace on an existing mesh.

    The simulation must already be meshed. Meshing is the Route-neutral
    part of a solve, so a caller comparing the two Routes can mesh once
    and hand the same simulation to both. The simulation owns its run
    directory: it clears the previous run's tables before this one and
    hands back this run's.

    Args:
        sim: A meshed ``BoundaryModeSim``.
        freq_hz: Frequency to solve at (Hz).
        num_modes: Number of Modes to compute.
        target: Effective-index target centring the shift-and-invert
            search; ``0.0`` leaves it to Palace.
        binary: Palace executable, from
            :func:`gsim.palace.runtime.require_palace_binary`.
        save: Number of Modes whose fields are written to ParaView, in
            mode order. A Stage reading fields back — the RF Stage, when
            no impedance path could be declared — asks for every Mode it
            might select, because which one that is is not known until
            they are all solved.
        verbose: Stream Palace's output.

    Returns:
        The solved Modes and the run's results.

    Raises:
        RuntimeError: When Palace returns no mode table, or when the
            binary exits abnormally with nothing to salvage. The
            simulation owns both of those — it salvages a complete table
            a crashed run left and turns a dead binary into a runtime
            report — so all this Route adds is the way back to femwell.
    """
    sim.set_boundary_mode(
        freq=float(freq_hz),
        num_modes=int(num_modes),
        target=target,
        save=int(save),
    )
    sim.write_config(photonic=True)
    text = sim.run_local(
        palace_executable=binary, verbose=verbose, remedy=_FEMWELL_REMEDY
    )
    modes = getattr(text, "modes", {})
    if not modes:
        raise RuntimeError(
            f"Palace produced no mode table at f = {freq_hz:g} Hz; its output "
            f"is in {sim.output_dir}."
        )
    return PalaceSolve(
        modes=[
            PalaceMode(n_eff=complex(modes[mode_id]["n_eff"]), mode_id=int(mode_id))
            for mode_id in sorted(modes)
        ],
        results=text,
    )


def containment_unmeasurable(stage_name: str) -> str:
    """Why the Window-containment check does not run on the Palace Route.

    ADR 0002 makes a Stage warn when a solved Mode still carries field at
    its Window boundary, because a too-small Window is the failure mode
    per-Stage Windows create. The check is measured on every candidate
    Mode as the Stage chooses between them, and Palace's *text* results —
    which is all a Stage has at that point — carry only effective
    indices. A check that quietly does not run is worse than one that
    says so.

    Args:
        stage_name: Stage the message names.

    Returns:
        The warning text.
    """
    return (
        f"The {stage_name} stage's palace route cannot check window "
        "containment: the mode table it selects from carries only effective "
        "indices, so boundary_field_ratio is NaN and a mode clipped by its "
        "window will not announce itself (ADR 0002). Re-solve with "
        f"study.{stage_name}(route='femwell') to have the window checked."
    )


def conductor_clearance(
    surroundings: Sequence[SurroundingRegion],
    *,
    window: tuple[float, float],
    window_z: tuple[float, float],
    stage_name: str,
) -> None:
    """Refuse a Staircase whose metal is sliced by the Window.

    A drawn conductor is meshed as an outline with its interior left out
    of the domain (ADR 0003). When the Window cuts through one, that
    outline runs along the Window's own outer wall, and Palace's meshing
    does not survive it — the solver aborts rather than reporting
    anything. The femwell Route meshes it, so this is a Route limitation
    and not a modelling one, which is why it is checked here and not in
    the Staircase.

    Args:
        surroundings: The Regions redrawn around the Strips.
        window: In-plane Window the Staircase is clipped to (um).
        window_z: Vertical Window (um).
        stage_name: Stage asking, named in the error.

    Raises:
        ValueError: When a conductor crosses the Window boundary on
            either axis.
    """
    for region in surroundings:
        if region.layer_type not in ("conductor", "via"):
            continue
        for extent, bounds, axis in (
            (region.h, window, "window"),
            (region.z, window_z, "window_z"),
        ):
            inside = extent[0] >= bounds[0] and extent[1] <= bounds[1]
            outside = extent[1] <= bounds[0] or extent[0] >= bounds[1]
            if inside or outside:
                continue
            raise ValueError(
                f"The {stage_name} stage's palace route cannot solve this "
                f"staircase: the drawn conductor '{region.name}' spans "
                f"{extent[0]:.4g}..{extent[1]:.4g} um, which the {axis} "
                f"{bounds[0]:.4g}..{bounds[1]:.4g} um cuts through, so its "
                "perfect-conductor outline would run along the window's own "
                f"wall. Widen the window to contain it (study.{stage_name}"
                f"({axis}=...)) or narrow it to leave the conductor out, or "
                f"solve with study.{stage_name}(route='femwell'), which "
                "meshes it."
            )


class PalaceRoute(Route):
    """The Route interface on Palace.

    Keeps, across one Stage run, the binary it resolved, the index its
    impedance path was declared under, and the last run's results.
    """

    name: ClassVar[EMRoute] = "palace"
    conductor_model: ClassVar[ConductorModel] = "pec"
    continuous_materials: ClassVar[bool] = False

    def __init__(self) -> None:
        """A fresh route: nothing resolved, nothing declared, nothing run."""
        self._binary: Path | None = None
        self._index: int | None = None
        self._last: PalaceSolve | None = None
        self._said_unmeasurable = False

    @property
    def impedance_index(self) -> int | None:
        """Index the line's impedance path was declared under, or None."""
        return self._index

    def require(self, *, stage_name: str) -> None:
        """Locate the Palace binary, once, for this run.

        Raises:
            RuntimeError: When no binary is available.
        """
        from gsim.palace.runtime import require_palace_binary

        self._binary = require_palace_binary(hint=_palace_hint(stage_name))

    def _executable(self, stage_name: str) -> Path:
        """The resolved binary, resolving it if :meth:`require` never ran."""
        from gsim.palace.runtime import require_palace_binary

        if self._binary is None:
            self._binary = require_palace_binary(hint=_palace_hint(stage_name))
        return self._binary

    def check_staircase(
        self,
        staircase: StaircaseCrossSection,
        *,
        window: tuple[float, float],
        window_z: tuple[float, float],
        stage_name: str,
    ) -> None:
        """Refuse a Staircase whose drawn metal the Window cuts through."""
        conductor_clearance(
            staircase.surroundings,
            window=window,
            window_z=window_z,
            stage_name=stage_name,
        )

    def prepare_line(
        self,
        sim: BoundaryModeSim,
        *,
        signal: Conductor,
        return_: Conductor | None,
        stage_name: str,
    ) -> None:
        """Declare the voltage path and current loop on the simulation.

        Sized from the electrodes and the meshed domain
        (:func:`line_impedance_paths`). A line whose paths cannot be
        sized — no single return electrode, or a signal electrode the
        Window clips — is not refused: the impedance is then read off the
        saved fields, and the solve saves every Mode for it.

        Warns:
            UserWarning: When the paths cannot be declared.
        """
        from gsim.palace.line_impedance import (
            declare_impedance_paths,
            line_impedance_paths,
        )

        self._index = None
        if return_ is None:
            reason = "the staircase has no single return electrode beside the signal"
        else:
            try:
                paths = line_impedance_paths(
                    signal=signal.extent,
                    ground=return_.extent,
                    domain=sim.mesh_extent,
                )
            except ValueError as err:
                reason = str(err)
            else:
                self._index = declare_impedance_paths(sim, paths)
                return
        warnings.warn(
            f"The {stage_name} stage's palace route cannot declare its "
            f"impedance paths, so the characteristic impedance is read off "
            f"the saved fields instead: {reason}",
            stacklevel=3,
        )

    def solve(
        self,
        sim: BoundaryModeSim,
        *,
        freq_hz: float,
        num_modes: int,
        target: float | None,
        order: int,  # noqa: ARG002 - Palace's element order is its own
        verbose: bool,
        stage_name: str,
        epsilon: ArrayLike | None = None,
    ) -> Sequence[Any]:
        """Run Palace on the meshed simulation at one frequency.

        Fields are saved only when the impedance has to be read off them
        — every Mode then, because which one is the line Mode is not
        known until they are all solved.

        Raises:
            ValueError: When a continuous permittivity is asked for,
                which Palace cannot carry.

        Warns:
            UserWarning: Once per run, that the Window containment
                cannot be checked on this Route (ADR 0002).
        """
        if epsilon is not None:
            raise ValueError(
                "The palace route takes piecewise-constant materials per "
                "region and cannot carry a continuous permittivity."
            )
        if not self._said_unmeasurable:
            self._said_unmeasurable = True
            warnings.warn(containment_unmeasurable(stage_name), stacklevel=3)
        self._last = solve_palace_modes(
            sim,
            freq_hz=freq_hz,
            num_modes=num_modes,
            binary=self._executable(stage_name),
            target=target if target is not None else 0.0,
            save=0 if self._index is not None else num_modes,
            verbose=verbose,
        )
        return self._last.modes

    def boundary_ratio(self, mode: Any) -> float:  # noqa: ARG002 - no fields in hand
        """NaN: Palace's mode table carries no field to measure."""
        return math.nan

    def strip_fraction_outside(
        self,
        mode: Any,  # noqa: ARG002 - no fields in hand
        span: tuple[float, float],  # noqa: ARG002 - nothing to measure over
        *,
        stage_name: str,
    ) -> float:
        """NaN, and say so: no mode fields come back to measure."""
        warnings.warn(
            f"The {stage_name} stage's palace route cannot check how much of "
            "the mode sits outside the strip extent either, for the same "
            "reason: no mode fields come back. Re-solve with "
            f"study.{stage_name}(route='femwell') at the same strip count to "
            "have both checks run on the identical staircase.",
            stacklevel=3,
        )
        return math.nan

    def read_line(
        self,
        sim: BoundaryModeSim,
        mode: Any,
        *,
        freq_hz: float,  # noqa: ARG002 - Palace's tables are per run
        signal: Conductor,
        return_: Conductor | None,  # noqa: ARG002 - the gap voltage stands in
        stage_name: str,
    ) -> LineReading:
        """Tables first, saved fields as the fallback.

        Palace's own ``mode-Z.csv`` under the declared index gives the
        power-current impedance and, through the voltage across the gap,
        the wall-Mode diagnostic. Without the tables the impedance is
        the Marks-Williams integral on the saved fields and nothing says
        which Mode this is.
        """
        from gsim.palace.line_impedance import palace_line_impedance

        if self._last is None:
            raise RuntimeError("read_line() called before solve().")
        return palace_line_impedance(
            sim,
            self._last.results,
            index=self._index,
            mode_id=mode.mode_id,
            n_eff=mode.n_eff,
            signal=signal,
            context=f"The {stage_name} stage's palace route",
        )
