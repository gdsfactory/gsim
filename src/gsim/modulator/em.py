"""What the two EM Stages share: a Staircase, a Route, and a Mode to check.

The optical and the RF Stage ask different questions, but both answer them
on a Staircase — the Bias point's Carrier map reduced to Strips tiling the
Junction extent, drawn as a component of its own and meshed through the
native ``BoundaryMode`` pipeline — and both reach their Backend through
the same :class:`~gsim.modulator.route.Route`. The Strips, the extent they
tile, the substrate under them, the plane cut through the middle of them,
the simulation that meshes them, the Route that solves it and the check
that the selected Mode is contained by its Window (ADR 0002) are the same
decisions in both Stages, so they are made once here.

What differs is the material each Stage adds to a Strip — the optical
wavelength and unperturbed index, or the RF lattice permittivity and top
frequency — and that is the one typed value each Stage answers
:meth:`EMStage.strip_material` with. The coupling itself is the carriers
Stage's, handed to the Staircase whole. The settings both Stages take —
how many Modes to solve, the guided-index floor, the boundary-field
tolerance, the metallic wall, the element order and the index guess —
are declared here once, each Stage restating only the defaults its own
physics wants.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field, model_validator

from gsim.modulator.meshing import STAGE_AIRBOX, STAGE_MESH
from gsim.modulator.route import EMRoute, Route, route_for
from gsim.modulator.stage import Stage
from gsim.modulator.staircase import (
    SEGMENT_TOL_UM,
    STRIP_LENGTH_UM,
    StaircaseDrawing,
    StripSegment,
)

if TYPE_CHECKING:
    from pathlib import Path

    import gdsfactory as gf

    from gsim.common.stack.extractor import LayerStack
    from gsim.modulator.staircase import (
        ElectrodeSpec,
        StaircaseCrossSection,
        StripMaterial,
        SurroundingRegion,
    )
    from gsim.palace import BoundaryModeSim
    from gsim.tcad.results import CarrierMap

__all__ = ["EMStage"]


class EMStage(Stage):
    """A Stage that solves an EM Mode on a Staircase, through a Route.

    Subclasses say what they add to a Strip's material in
    :meth:`strip_material`, restate the defaults of the shared settings
    in :attr:`setting_defaults`, and add the settings their own physics
    needs on top of the ones here.

    Attributes:
        route: Backend answering this Stage — ``"femwell"`` (the default)
            or ``"palace"``. Both solve the same Staircase on the same
            mesh; only femwell reads its Mode's fields while it selects,
            so the Palace Route reports NaN for the Window-containment
            ratio and says so.
        strip_span: ``(min, max)`` extent the Strips tile along the
            junction axis (um); :meth:`default_strip_span` when unset,
            which each Stage answers for itself.
        substrate_thickness_um: Substrate below the Staircase (um).
        window: In-plane Window (um) the Cross-section is clipped to.
        window_z: Vertical Window (um).
        mesh: Keyword arguments forwarded to the mesh pipeline.
        airbox: Background region around what the Stage meshes.
        num_modes: Number of Modes solved at each point, out of which
            the physical one is selected.
        min_index: Lower bound on ``Re(n_eff)`` for a Mode to count as
            guided; raise it to the cladding index to reject radiation
            Modes.
        boundary_field_tol: Warn above this boundary-field ratio, the
            sign of a Window squeezing the Mode (ADR 0002).
        metallic_boundaries: Enforce a perfect conductor on the domain
            boundary. Both Routes express it identically: femwell applies
            its perfect-conductor condition to the domain boundary, and
            the Palace config puts the outer wall under ``Boundaries.PEC``
            — which is what lets the cross-Route gate compare their
            answers at all.
        order: Finite-element order of the mode solve, where the Route
            has one.
        n_guess: Effective-index guess centering the eigenvalue search;
            ``None`` leaves it to the Backend.
    """

    #: The defaults a Stage restates for the shared settings below.
    setting_defaults: ClassVar[dict[str, Any]] = {}

    route: EMRoute = "femwell"
    strip_span: tuple[float, float] | None = None
    substrate_thickness_um: float = Field(default=2.0, gt=0.0)
    window: tuple[float, float] | None = None
    window_z: tuple[float, float] | None = None
    mesh: dict[str, Any] = Field(default_factory=STAGE_MESH.copy)
    airbox: dict[str, Any] = Field(default_factory=STAGE_AIRBOX.copy)
    num_modes: int = Field(default=1, ge=1)
    min_index: float = Field(default=1.0, ge=0.0)
    boundary_field_tol: float = Field(default=0.01, gt=0.0)
    metallic_boundaries: bool = True
    order: int = Field(default=1, ge=1)
    n_guess: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _restate_defaults(cls, data: Any) -> Any:
        """Fill the shared settings a Stage left unset with its own defaults."""
        if isinstance(data, dict):
            return {**cls.setting_defaults, **data}
        return data

    # ------------------------------------------------------------------
    # Route
    # ------------------------------------------------------------------

    def resolved_route(self) -> Route:
        """A fresh instance of the selected Route, for one run.

        Returns:
            The Route, its Backend not yet checked.
        """
        return route_for(self.route)

    def check_route(self) -> Route:
        """The selected Route, its Backend checked.

        Called before the charge solve and before meshing, so a user
        whose Route cannot run pays only for the error message.

        Returns:
            The Route for this run.

        Raises:
            ImportError: When the femwell extra is not installed.
            RuntimeError: When no Palace binary is available.
        """
        route = self.resolved_route()
        route.require(stage_name=self.stage_name)
        return route

    # ------------------------------------------------------------------
    # Staircase
    # ------------------------------------------------------------------

    def default_strip_span(self) -> tuple[float, float]:
        """The extent this Stage tiles when ``strip_span`` says nothing.

        The Junction extent — the rib — which is what a Staircase
        standing on its own is a model of. A Stage whose Staircase is
        drawn inside the device's own Cross-section wants the doped slab
        instead, and says so by overriding this.

        Returns:
            ``(min, max)`` along the junction axis (um).
        """
        return self._require_study().layout.junction_span.h

    def strip_extent(self, carriers: CarrierMap | None = None) -> tuple[float, float]:
        """``(min, max)`` extent the Strips tile along the junction axis (um).

        Strips cannot outrun the Carrier map they average, so a map
        narrower than the extent asked for (a charge Window clipped
        tighter than the doped Regions) narrows it to what the map
        covers, and says so. That holds whether the extent was derived or
        chosen: the preset chooses it, and a Stage that failed only for
        the callers who said what they wanted would fail for most of
        them.

        Args:
            carriers: The Carrier map the Strips will average, to bound
                the extent by what it covers; unbounded when omitted.

        Returns:
            The extent the Strips tile.
        """
        study = self._require_study()
        if self.strip_span is not None:
            wanted = (float(self.strip_span[0]), float(self.strip_span[1]))
            source = "the extent asked for"
        else:
            wanted = self.default_strip_span()
            source = "the extent derived for this stage"
        if carriers is None:
            return wanted

        from gsim.modulator.staircase import carrier_map_extent

        covered = carrier_map_extent(carriers, study.layout.junction_span.z)
        clipped = (max(wanted[0], covered[0]), min(wanted[1], covered[1]))
        if clipped != wanted:
            warnings.warn(
                f"The {self.stage_name} stage's strips tile "
                f"{clipped[0]:.4g}..{clipped[1]:.4g} um rather than {source} "
                f"{wanted[0]:.4g}..{wanted[1]:.4g} um: the carrier map only "
                f"covers {covered[0]:.4g}..{covered[1]:.4g} um, and strips "
                "cannot reach past the map they average. The doped silicon "
                "outside them keeps its drawn material and carries no "
                "carrier response. Widen the charge window with "
                "study.charge(window=...), or choose the extent yourself "
                f"with study.{self.stage_name}(strip_span=...).",
                stacklevel=3,
            )
        return clipped

    def region_segments(
        self, extent: tuple[float, float], *, n_strips: int, strips_per_region: int
    ) -> list[StripSegment]:
        """Strips that follow the drawn device across *extent*.

        Every doped Region the extent crosses is a segment of its own,
        at the Region's own height, so no Strip straddles a doping step
        or a step in the silicon's thickness: a rib beside a thinner slab
        stays a rib beside a slab, and a lightly doped Region beside a
        heavily doped one is not averaged into a resistance neither of
        them has. The Junction extent is where the carriers move, so it
        takes *n_strips* between its two Regions — as one run when they
        stand at one height, split by width otherwise — and every other
        Region takes *strips_per_region*.

        Args:
            extent: ``(min, max)`` the Strips tile along the junction
                axis (um).
            n_strips: Strips across the Junction extent.
            strips_per_region: Strips across each other doped Region.

        Returns:
            The segments, low edge first.
        """
        layout = self._require_study().layout
        low, high = extent
        runs: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
        for name in sorted(
            layout.doped_regions, key=lambda region: layout.region_spans[region].h
        ):
            span = layout.region_spans[name]
            start, stop = max(span.h[0], low), min(span.h[1], high)
            if stop - start > SEGMENT_TOL_UM:
                runs.append((name, (start, stop), span.z))

        junction = set(layout.junction.regions)
        inside = [run for run in runs if run[0] in junction]
        merged = len(inside) == 2 and inside[0][2] == inside[1][2]
        widths = [run[1][1] - run[1][0] for run in inside]
        shares = [n_strips] if merged or len(inside) < 2 else _split(n_strips, widths)

        segments: list[StripSegment] = []
        for name, span, z in runs:
            if name not in junction:
                segments.append(
                    StripSegment(span=span, z=z, n_strips=strips_per_region)
                )
            elif merged:
                if name == inside[0][0]:
                    segments.append(
                        StripSegment(
                            span=(span[0], inside[1][1][1]), z=z, n_strips=n_strips
                        )
                    )
            else:
                segments.append(StripSegment(span=span, z=z, n_strips=shares.pop(0)))
        return segments

    def strip_material(self) -> StripMaterial:
        """What this Stage adds to a Strip's material.

        The optical wavelength and unperturbed index, or the RF lattice
        permittivity and top frequency: the one typed value the
        Staircase builder takes per Stage. Implemented by each EM Stage.
        """
        raise NotImplementedError

    def build_staircase(
        self,
        carriers: CarrierMap,
        *,
        n_strips: int,
        electrodes: ElectrodeSpec | None,
        surroundings: Sequence[SurroundingRegion] = (),
        span: tuple[float, float] | None = None,
        segments: Sequence[StripSegment] = (),
    ) -> StaircaseCrossSection:
        """Reduce a Carrier map to a meshable Staircase.

        The Strips tile :meth:`strip_extent`, sit at the Junction's own
        height, and take the carriers Stage's coupling whole — so both EM
        Stages read the one coupling, evaluated on strip averages — with
        this Stage's own :meth:`strip_material` added.

        Args:
            carriers: The Bias point's Carrier map.
            n_strips: Number of Strips to tile the extent with.
            electrodes: The drawn conductors flanking the Strips, or
                ``None`` for a Staircase carrying none.
            surroundings: The drawn device's own Regions to redraw around
                the Strips; empty leaves the Strips alone in the
                background medium.
            span: The extent to tile, when the caller has already
                resolved it; :meth:`strip_extent` decides otherwise.
            segments: Strips that follow the drawn device
                (:meth:`region_segments`), in place of *n_strips* equal
                Strips at the Junction's height across the extent.

        Returns:
            The Staircase Cross-section, drawn on its own component.
        """
        from gsim.modulator.staircase import build_staircase_cross_section

        study = self._require_study()
        junction = study.layout.junction_span
        extent: dict[str, Any] = (
            {"segments": segments}
            if segments
            else {
                "n_strips": n_strips,
                "junction": span if span is not None else self.strip_extent(carriers),
                "zmin": junction.z[0],
                "zmax": junction.z[1],
            }
        )
        return build_staircase_cross_section(
            carriers,
            **extent,
            response=study.carriers.response,
            material=self.strip_material(),
            electrodes=electrodes,
            surroundings=surroundings,
            drawing=StaircaseDrawing(substrate_thickness=self.substrate_thickness_um),
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def new_simulation(
        self,
        *,
        stack: LayerStack,
        component: gf.Component,
        plane: str,
        output_dir: str | Path,
        freq_hz: float,
        window: tuple[float, float] | None,
        window_z: tuple[float, float] | None,
    ) -> BoundaryModeSim:
        """Assemble a cross-section simulation the way every EM Stage does.

        The one place the metallic wall reaches a simulation: both Routes
        put the same condition on the Window's outer wall — femwell reads
        this setting directly, and the Palace config puts the wall under
        ``Boundaries.PEC`` — and without it Palace defaults the
        unconditioned wall to PMC, the opposite condition, so the two
        Routes would solve different boundary-value problems.

        Args:
            stack: The layer stack the Regions are named in.
            component: The drawn component the plane cuts.
            plane: Cross-section plane spec, e.g. ``"x=5"``.
            output_dir: Directory the mesh and solver files land in.
            freq_hz: Frequency recorded in the boundary-mode block.
            window: In-plane Window (um), or ``None`` for the full extent.
            window_z: Vertical Window (um), likewise.

        Returns:
            The configured (unmeshed) ``BoundaryModeSim``.
        """
        from gsim.palace import BoundaryModeSim

        sim = BoundaryModeSim()
        sim.set_output_dir(output_dir)
        sim.set_stack(stack)
        sim.set_geometry(component)
        sim.set_airbox(**self.airbox)
        sim.set_cross_section(plane, window=window, window_z=window_z)
        sim.set_boundary_mode(
            freq=freq_hz,
            num_modes=self.num_modes,
            target=self.n_guess if self.n_guess is not None else 0.0,
        )
        sim.metallic_boundaries = self.metallic_boundaries
        return sim

    def build_staircase_simulation(
        self,
        staircase: StaircaseCrossSection,
        *,
        output_dir: str | Path,
        freq_hz: float,
        window: tuple[float, float] | None = None,
        window_z: tuple[float, float] | None = None,
    ) -> BoundaryModeSim:
        """Assemble the Staircase Cross-section this Stage meshes.

        The Staircase is a component of its own — Strips and electrodes
        drawn in the Cross-section's transverse coordinates and extruded
        along the propagation direction — so the plane cuts through the
        middle of it rather than through the drawn device, and the airbox
        rather than a derived Window is what puts cladding around the
        Strips.

        Args:
            staircase: The Staircase to mesh.
            output_dir: Directory the mesh and solver files land in.
            freq_hz: Frequency recorded in the boundary-mode block.
            window: In-plane Window (um) to clip the Staircase to; the
                Stage's own ``window`` when omitted.
            window_z: Vertical Window (um); likewise.

        Returns:
            The configured (unmeshed) ``BoundaryModeSim``.
        """
        return self.new_simulation(
            stack=staircase.stack(),
            component=staircase.component,
            plane=f"x={STRIP_LENGTH_UM / 2.0}",
            output_dir=output_dir,
            freq_hz=freq_hz,
            window=window if window is not None else self.window,
            window_z=window_z if window_z is not None else self.window_z,
        )

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------

    def selection(self) -> dict[str, Any]:
        """The keyword arguments this Stage selects its Mode with.

        The guided-index floor for every Stage; a Stage with a
        transmission line's tighter bounds adds them.

        Returns:
            What :func:`gsim.common.modes.select_line_mode` takes.
        """
        return {"min_index": self.min_index}

    def window_hint(self) -> str:
        """How this Stage's Window is widened, for the containment warning."""
        return (
            f"Widen the window with study.{self.stage_name}(window=..., window_z=...)."
        )

    def select_mode(
        self, modes: Sequence[Any], route: Route, *, at: str
    ) -> tuple[Any, float]:
        """Select the physical Mode of one solve, and check its Window.

        The one ADR 0002 rule for both Stages: a solved Mode still
        carrying field at its Window boundary is a clipped Mode, and the
        Stage says so. A Route that cannot measure the ratio answers NaN
        and has already said why.

        Args:
            modes: Every Mode the Route solved.
            route: The Route that solved them.
            at: Where the solve was, for the warning (``"f = 10 GHz"``,
                ``"V = 2"``).

        Returns:
            ``(mode, ratio)``: the selected Mode and its boundary-field
            ratio, NaN when unmeasured.
        """
        from gsim.common.modes import select_line_mode

        mode = select_line_mode(modes, **self.selection())
        ratio = route.boundary_ratio(mode)
        if not math.isnan(ratio) and ratio > self.boundary_field_tol:
            warnings.warn(
                f"The {self.stage_name} stage's mode at {at} carries "
                f"{ratio:.1%} of its peak field at the window boundary "
                f"(tolerance {self.boundary_field_tol:.1%}); its effective "
                f"index is a clipped mode's. {self.window_hint()}",
                stacklevel=3,
            )
        return mode, ratio


def _split(total: int, widths: Sequence[float]) -> list[int]:
    """Share *total* Strips between Regions in proportion to their widths.

    Every Region keeps at least one Strip, and the shares add up to
    *total* unless that is fewer than the Regions.
    """
    shares = [max(1, round(total * width / sum(widths))) for width in widths]
    shares[-1] = max(1, total - sum(shares[:-1]))
    return shares
