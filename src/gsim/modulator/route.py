"""Which Backend answers an EM Stage's question, and what a Stage asks of it.

A Route is a per-Stage, per-run choice between the Backends that can
answer the same question, and both EM Stages of a modulator Study accept
one. femwell is the default because it needs no external binary and can
carry a continuously varying permittivity; Palace is the second
first-class Route the originating spec asks for, and the one the
cross-Route agreement check is written against.

The two Backends differ in what they can express, and that difference is
the reason a Route is a choice rather than an implementation detail:
Palace takes piecewise-constant materials per mesh Region and nothing
else, so a carrier distribution reaches it as a Staircase. Where a Stage
can also solve a continuous profile, selecting the Palace Route moves it
onto the Staircase representation of the same physics.

A Stage holds a Route as an object — a :class:`Route` — and asks it the
handful of things a Stage needs: is the Backend here, solve Modes on a
meshed simulation, how much field sits on the Window wall, read the
selected Mode. Each Backend answers through its own Route
(:mod:`gsim.modulator.femwell_route`, :mod:`gsim.modulator.palace_route`);
a test answers through a fake one. Nothing here imports a Backend at
module scope: a Study that never selects a Route pays for neither.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from gsim.common.modes import Conductor, LineReading
    from gsim.modulator.staircase import ConductorModel, StaircaseCrossSection
    from gsim.palace import BoundaryModeSim

__all__ = [
    "DEFAULT_PALACE_STRIPS",
    "ROUTES",
    "EMRoute",
    "Route",
    "route_for",
]

#: The Backends an EM Stage can be routed to.
EMRoute = Literal["femwell", "palace"]

#: Strip count the Palace Route falls back to when a Stage that can also
#: solve a continuous profile has not been given one. Palace cannot
#: express a continuous profile at all, so the choice is a strip count or
#: an error, and a workable default beats an error.
DEFAULT_PALACE_STRIPS: int = 5


class Route(ABC):
    """What an EM Stage asks of the Backend answering it.

    One instance serves one Stage run: a route may keep what it learns
    early in the run (a resolved binary, a declared impedance path) for
    the readings later in it.

    Attributes:
        name: The Route's name, as a Stage's ``route`` setting spells it.
        conductor_model: How this Route expresses a Staircase's electrode
            metal unless the Stage says otherwise (ADR 0003).
        continuous_materials: Whether the Backend can carry a continuous
            per-element permittivity, or takes piecewise-constant
            materials per Region and nothing else.
    """

    name: ClassVar[EMRoute]
    conductor_model: ClassVar[ConductorModel]
    continuous_materials: ClassVar[bool]

    @abstractmethod
    def require(self, *, stage_name: str) -> None:
        """Check the Backend is available, before the Stage spends anything.

        Args:
            stage_name: Stage asking, named in the error.

        Raises:
            ImportError: When a Python extra is missing.
            RuntimeError: When a binary is missing.
        """

    def check_line_settings(
        self,
        *,
        conductor_model: ConductorModel,
        metallic_boundaries: bool,
        order: int,
        stage_name: str,
    ) -> None:
        """Refuse or warn about RF settings this Backend cannot honour.

        Reads settings only, so a Stage can ask before it meshes. The
        default honours everything.

        Args:
            conductor_model: How the electrodes reach the mesh.
            metallic_boundaries: Whether the Window wall is a conductor.
            order: Finite-element order of the solve.
            stage_name: Stage asking, named in the message.
        """
        del conductor_model, metallic_boundaries, order, stage_name

    def check_staircase(
        self,
        staircase: StaircaseCrossSection,
        *,
        window: tuple[float, float],
        window_z: tuple[float, float],
        stage_name: str,
    ) -> None:
        """Refuse a Staircase this Backend cannot mesh.

        The default meshes anything.

        Args:
            staircase: The Staircase about to be meshed.
            window: In-plane Window it is clipped to (um).
            window_z: Vertical Window (um).
            stage_name: Stage asking, named in the error.
        """
        del staircase, window, window_z, stage_name

    def prepare_line(
        self,
        sim: BoundaryModeSim,
        *,
        signal: Conductor,
        return_: Conductor | None,
        stage_name: str,
    ) -> None:
        """Set up what reading a line's impedance needs, before the solve.

        Called once per RF run on the meshed simulation. The default
        needs nothing.

        Args:
            sim: The meshed simulation about to be solved.
            signal: The signal electrode.
            return_: The return electrode, or ``None`` when there is not
                exactly one.
            stage_name: Stage asking, named in any warning.
        """
        del sim, signal, return_, stage_name

    @abstractmethod
    def solve(
        self,
        sim: BoundaryModeSim,
        *,
        freq_hz: float,
        num_modes: int,
        target: float | None,
        order: int,
        verbose: bool,
        stage_name: str,
        epsilon: ArrayLike | None = None,
    ) -> Sequence[Any]:
        """Solve the Modes of a meshed simulation at one frequency.

        Args:
            sim: A meshed ``BoundaryModeSim`` carrying the materials.
            freq_hz: Frequency to solve at (Hz).
            num_modes: Number of Modes to compute.
            target: Effective-index guess centering the eigenvalue
                search, or ``None`` for the Backend's own.
            order: Finite-element order, where the Backend has one.
            verbose: Stream the Backend's own output.
            stage_name: Stage asking, named in any warning.
            epsilon: A continuous per-element permittivity replacing the
                simulation's materials, for a Backend that can carry one.

        Returns:
            Every Mode the Backend solved, in its own order.
        """

    @abstractmethod
    def boundary_ratio(self, mode: Any) -> float:
        """Peak field at the Window boundary over peak field overall.

        Args:
            mode: A solved Mode of this Route.

        Returns:
            The ratio, or NaN when the Backend cannot say.
        """

    @abstractmethod
    def strip_fraction_outside(
        self, mode: Any, span: tuple[float, float], *, stage_name: str
    ) -> float:
        """How much of a Mode's power sits outside a Strip extent.

        Args:
            mode: A solved Mode of this Route.
            span: ``(min, max)`` the Strips tile (um).
            stage_name: Stage asking, named in any warning.

        Returns:
            The fraction, or NaN when the Backend cannot say.
        """

    @abstractmethod
    def read_line(
        self,
        sim: BoundaryModeSim,
        mode: Any,
        *,
        freq_hz: float,
        signal: Conductor,
        return_: Conductor | None,
        stage_name: str,
    ) -> LineReading:
        """The selected RF Mode's index, impedance and wall-Mode diagnostic.

        Args:
            sim: The simulation the Mode was solved on.
            mode: The selected Mode.
            freq_hz: The frequency it was solved at (Hz).
            signal: The signal electrode.
            return_: The return electrode, or ``None`` when there is not
                exactly one.
            stage_name: Stage asking, named in any warning.

        Returns:
            The reading.
        """


#: The implementation behind each Route name. A test registers a fake here.
ROUTES: dict[str, type[Route]] = {}


def route_for(name: EMRoute) -> Route:
    """A fresh instance of one Route, for one Stage run.

    Args:
        name: The Route, as the Stage's ``route`` setting spells it.

    Returns:
        A new Route instance.

    Raises:
        ValueError: When no route is registered under that name.
    """
    if name not in ROUTES:
        from gsim.modulator.femwell_route import FemwellRoute
        from gsim.modulator.palace_route import PalaceRoute

        ROUTES.setdefault(FemwellRoute.name, FemwellRoute)
        ROUTES.setdefault(PalaceRoute.name, PalaceRoute)
    try:
        route = ROUTES[name]
    except KeyError:
        raise ValueError(
            f"No route is registered under {name!r}; known routes are {sorted(ROUTES)}."
        ) from None
    return route()
