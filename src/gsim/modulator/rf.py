"""The RF Stage: the traveling-wave line parameters versus frequency.

The Stage answers what the Traveling-wave electrode does to an RF signal
once the Phase shifter's carriers load it. Palace and femwell both take
piecewise-constant materials per Region, so the Carrier map of the chosen
Bias point is reduced to a Staircase — Strips tiling the Junction extent
(or any wider extent asked for), each carrying the Drude conductivity of
its average carrier concentrations, flanked by the electrode's two drawn
conductors — and the electrode-loaded Cross-section is solved once per
requested frequency.

The Staircase is built here rather than by the caller: its Strips, their
materials and the electrodes all follow from the device description and
the carriers Stage's coupling, so nothing outside assembles a second
component.

Each frequency gives one reading of the selected Mode, asked of the Route
(:meth:`~gsim.modulator.route.Route.read_line`): its complex effective
index, from which the RF index and the RF loss follow; its characteristic
impedance by the Marks-Williams power-current definition over the signal
conductor — the electrode on the Contact the RF drive is applied to,
identified from the device description; and whether it is the line Mode
or the wall Mode. The femwell Route reads the impedance from the Mode's
fields and the diagnostic from the balance of the two electrode currents;
the Palace Route reads both from Palace's own tables, falling back to the
saved fields when the tables are absent. The Stage has one solve loop and
one wall-Mode warning for both.

What the two Routes can express of the electrode metal differs, and that
is what ``conductor_model`` chooses between (ADR 0003). Meshed as
Regions of conducting material — the ``"volume"`` model — the electrodes
have a permittivity whose imaginary part is of order ``1e7`` at RF, and
Palace's shift-and-invert search returns those metal-dominated modes
rather than the quasi-TEM one. Meshed as perfect conductors — the
``"pec"`` model — their interior is left out of the domain and their
outline carries the boundary condition, which both Routes express
identically and neither is derailed by. Each Route therefore carries its
own default, and the Stage reads it off the route.

The Window's outer wall is metallic on both Routes, which makes it a
third conductor: beside the line Mode between the two electrodes, the
shielded line has a Mode on which both electrodes sit at one potential
and return their current through the wall. The index does not separate
them, and at an undepleted Bias the loaded line Mode can fall outside
the loss bound and leave the wall Mode as the slowest candidate. The
Route's reading says which Mode was selected, and the Stage warns when
it is the wall Mode (ADR 0005).

The selected Route's runtime is checked before the Stage meshes, so a
missing extra or a missing binary costs nothing but the error message.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr, field_validator
from scipy.constants import epsilon_0 as EPS0  # noqa: N812

from gsim.common.modes import LineModeRule
from gsim.common.sweep import BIAS_TOL_V
from gsim.modulator.em import EMStage
from gsim.modulator.staircase import (
    DEFAULT_ELECTRODES,
    ConductorModel,
    ElectrodeSpec,
    RFStripMaterial,
)

if TYPE_CHECKING:
    from pathlib import Path

    from gsim.common.modes import Conductor, LineReading
    from gsim.common.transmission_line import JunctionBranch, RFLineParams
    from gsim.modulator.carriers import CarrierResponse, CarrierResponseSweep
    from gsim.modulator.report import LoadedLineComparison
    from gsim.modulator.route import Route
    from gsim.modulator.staircase import StaircaseCrossSection
    from gsim.palace import BoundaryModeSim
    from gsim.tcad.results import BiasSweepResult, CarrierMap

__all__ = ["RFStage"]


class RFStage(EMStage):
    """The line parameters of the Traveling-wave electrode, versus frequency.

    The settings every EM Stage takes — ``route``, ``num_modes``,
    ``min_index``, ``boundary_field_tol``, ``metallic_boundaries``,
    ``order``, ``n_guess``, the Window and the mesh — are documented on
    :class:`~gsim.modulator.em.EMStage`; this Stage restates four
    defaults. Four Modes are solved per frequency because the line Mode
    is selected among them; the boundary-field tolerance is far looser
    than an optical Window's, because the outer boundary is metallic and
    carries the line's return field, so some field there is the
    structure rather than a clipped tail; the index guess is the
    slow-wave index rather than the solver's, because RF materials have
    large conductive ``|Im(eps)|``.

    Attributes:
        conductor_model: How the electrode metal is expressed in the
            mesh (ADR 0003) — ``"volume"`` as Regions of lossy metal,
            ``"pec"`` as perfect conductors whose interior is left out of
            the meshed domain. Unset, each Route takes the model it can
            express: ``"volume"`` on femwell, which carries the metal's
            own loss, and ``"pec"`` on Palace, whose eigenvalue search a
            metal Region takes over. Set it to the same value on both to
            compare them.
        frequencies_hz: RF frequencies to solve at (Hz), kept ascending.
        n_strips: Number of Strips across the Junction extent — the two
            Regions the Junction separates, where the carriers move; more
            approximate the continuous profile more closely at the cost
            of mesh size.
        strips_per_region: Number of Strips across every other doped
            Region the Strips tile. The Strips follow the drawn device
            (:meth:`~gsim.modulator.em.EMStage.region_segments`): no
            Strip straddles two Regions, and each stands at its own
            Region's height, so a slab beside the rib is drawn as thin as
            it is and a heavily doped contact Region is not averaged into
            the lighter one beside it.
        bias_v: Bias point the Staircase is built from; the last point of
            the Bias sweep when unset.
        strip_span: ``(min, max)`` extent the Strips tile along the
            junction axis (um); the Junction extent — the rib — when
            unset, which leaves the doped pads out of the Staircase and
            so leaves out the series resistance they carry, the
            electrodes standing directly against the rib.
            :func:`~gsim.modulator.preset.pn_phase_shifter` therefore
            widens it to the doped slab
            (:attr:`~gsim.modulator.layout.DeviceLayout.doped_span`).
            Widening it costs no resolution at the Junction, which keeps
            its ``n_strips`` whatever else the span takes in.
        electrodes: The drawn conductors of the Traveling-wave
            electrode, flanking the Strips.
        signal_contact: Contact the RF drive is applied to, naming the
            signal conductor; the charge Stage's swept Contact when
            unset.
        rule: Candidate rule replacing the default one of
            :func:`gsim.common.modes.select_line_mode`.
        max_loss_ratio: Upper bound on ``|Im(n_eff)| / Re(n_eff)`` for
            the default rule; one, where a Mode loses as fast as it
            advances. The loaded line of a Phase shifter is a slow wave
            carrying the slab's series resistance, and at the bottom of
            the band it loses most of a radian per radian even at a
            depleted Bias (0.74 at 10 GHz on the demo), so a tighter
            bound drops the line Mode and leaves the wall Mode
            (ADR 0005). The discretization's spurious Modes sit far
            outside one, at hundreds of radians of loss per radian.
        degeneracy_rtol: Relative spread within which two candidate Modes
            count as ambiguous, and the selection warns.
        strip_permittivity: Relative permittivity of the Strip lattice,
            which the carrier conductivity loads.
        substrate_thickness_um: Substrate below the Staircase (um).
        track_modes: Follow the line Mode across the sweep, guessing each
            frequency from the index solved at the one before it, so a
            shift-and-invert search anchored to one number cannot settle
            on a different branch as the materials move with frequency.
            Turn it off to aim every frequency at ``n_guess`` instead.
        jump_rtol: How far ``Re(n_eff)`` may step between neighbouring
            frequencies, per unit of relative frequency step, before the
            sweep is reported as having changed branch rather than
            dispersed. Scaling by the frequency step is what lets one
            number serve a sparse sweep and a dense one: a doubling in
            frequency is allowed ``jump_rtol`` of index, a 1% step a
            hundredth of it.
    """

    stage_name: ClassVar[str] = "rf"
    setting_defaults: ClassVar[dict[str, Any]] = {
        "num_modes": 4,
        "boundary_field_tol": 0.2,
        "n_guess": 3.0,
    }

    #: The unloaded solve's cached result, dropped with the loaded one.
    _unloaded_result: RFLineParams | None = PrivateAttr(default=None)

    frequencies_hz: list[float] = Field(
        default_factory=lambda: [10e9, 40e9], min_length=1
    )
    n_strips: int = Field(default=21, ge=1)
    strips_per_region: int = Field(default=2, ge=1)
    bias_v: float | None = None
    conductor_model: ConductorModel | None = None
    electrodes: ElectrodeSpec = Field(default=DEFAULT_ELECTRODES)
    signal_contact: str | None = None
    rule: LineModeRule | None = None
    max_loss_ratio: float = Field(default=1.0, gt=0.0)
    degeneracy_rtol: float = Field(default=0.03, gt=0.0)
    strip_permittivity: float = Field(default=11.9, gt=0.0)
    track_modes: bool = True
    jump_rtol: float = Field(default=0.25, gt=0.0)

    @field_validator("frequencies_hz")
    @classmethod
    def _ascending_and_positive(cls, value: list[float]) -> list[float]:
        """Frequencies are positive, and travel in ascending order."""
        if any(freq <= 0.0 for freq in value):
            raise ValueError("RF frequencies must be positive.")
        return sorted(float(freq) for freq in value)

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    def bias_point(self) -> CarrierResponse:
        """The Bias point the Staircase is built from.

        Runs the carriers Stage (and, through it, the charge Stage) if
        they have not run.

        Returns:
            The chosen Bias point's carrier response.

        Raises:
            ValueError: When the sweep is empty, or has no point at the
                configured bias.
        """
        responses: CarrierResponseSweep = self._require_study().carriers.run()
        if not responses.points:
            raise ValueError(
                "The bias sweep has no points, so there is no carrier map to "
                "staircase. Configure study.charge(biases=[...])."
            )
        if self.bias_v is None:
            return responses.points[-1]
        try:
            return responses.point_at(self.bias_v, tol=BIAS_TOL_V)
        except ValueError as err:
            raise ValueError(
                f"{err} Solve that bias with study.charge(biases=[...]) or "
                f"pick one of them with study.{self.stage_name}(bias_v=...)."
            ) from None

    def signal_contact_name(self) -> str:
        """Name of the Contact the RF drive is applied to.

        Returns:
            The configured Contact, or the one the charge sweep drives.
        """
        if self.signal_contact is not None:
            return self.signal_contact
        return str(self._require_study().charge.swept_contact())

    def signal_electrode(self) -> str:
        """Region name of the Staircase electrode carrying the signal.

        The Contact the drive is applied to sits on one side of the
        Junction; the electrode flanking that side is the signal
        conductor, and the other one is the return.

        Returns:
            The electrode Region name.

        Raises:
            ValueError: When no Contact of the device has that name.
        """
        layout = self._require_study().layout
        wanted = self.signal_contact_name()
        matches = [contact for contact in layout.contacts if contact.name == wanted]
        if not matches:
            raise ValueError(
                f"The device has no contact named '{wanted}'; its contacts are "
                f"{[contact.name for contact in layout.contacts]}. Name the "
                f"driven one with study.{self.stage_name}(signal_contact=...)."
            )
        low_side = layout.is_below_junction(matches[0].region)
        return self.electrodes.names[0 if low_side else 1]

    def effective_conductor_model(self) -> ConductorModel:
        """How this run expresses the electrode metal (ADR 0003).

        Returns:
            The configured model, or the one the selected Route
            expresses by default: ``"volume"`` on femwell, which carries
            the metal's own conductivity, and ``"pec"`` on Palace, whose
            eigenvalue search returns a metal Region's own modes rather
            than the line's.
        """
        if self.conductor_model is not None:
            return self.conductor_model
        return self.resolved_route().conductor_model

    def staircase(self) -> StaircaseCrossSection:
        """Reduce the Bias point's Carrier map to a meshable Staircase.

        The Strips tile the Junction extent — the two Regions the
        metallurgical boundary separates — unless ``strip_span`` widens
        them, and take the plasma-dispersion coefficients and mobilities
        of the carriers Stage, so the RF conductivity is the same
        coupling the optical Stage reads. Unlike the optical Staircase,
        this one draws its own flanking electrodes and none of the device
        around them (ADR 0004): the question is the Traveling-wave line,
        which the Cross-section of a Phase shifter does not draw.

        The Strip materials are valid up to the highest requested
        frequency, which is what the solve asks of them, and the
        electrodes flanking them take this run's conductor model.

        Returns:
            The Staircase Cross-section, drawn on its own component.
        """
        staircase = self._staircase_for(self.bias_point().carriers)
        self._check_depletion(staircase)
        return staircase

    def _check_depletion(self, staircase: StaircaseCrossSection) -> None:
        """Warn when no Strip is depleted, so the slab shunts the line.

        A Strip is depleted when it is a dielectric at the highest
        frequency solved — its conductivity below ``omega * eps`` there.
        With none, every Strip conducts, the electrodes are joined by a
        resistive slab, and the loaded line is an RC wave losing about as
        fast as it advances: the selection lands either on it, at the
        edge of ``max_loss_ratio``, or past it on the wall Mode
        (ADR 0005), and neither is the depleted line. Either the Bias does
        not deplete the Junction, or the Strips are wider than the
        depletion region and average it away.

        Args:
            staircase: The loaded Staircase the Stage built.
        """
        fmax = max(self.frequencies_hz)
        dielectric_below = 2.0 * np.pi * fmax * EPS0 * self.strip_permittivity
        least = float(
            np.min(np.asarray(staircase.strips.conductivity_s_per_m, dtype=float))
        )
        if least < dielectric_below:
            return
        warnings.warn(
            f"The {self.stage_name} stage's staircase at V = "
            f"{self.bias_point().bias_v:g} has no depleted strip: the least "
            f"conductive of its {self.n_strips} strips carries {least:.3g} S/m, "
            f"above the {dielectric_below:.3g} S/m below which a strip is a "
            f"dielectric at {fmax / 1e9:g} GHz. The slab shunts the electrodes, "
            "so the loaded line is an RC wave losing about as fast as it "
            "advances, and the selection lands on it or on the wall mode, not on "
            f"the depleted line. Raise study.{self.stage_name}(n_strips=...) until "
            "a strip resolves the depletion region, or solve at a depleting "
            "bias_v.",
            stacklevel=2,
        )

    def unloaded_staircase(self) -> StaircaseCrossSection:
        """The loaded solve's Staircase with the carriers switched off.

        Same Bias point's Carrier map, same Strips, same electrodes and
        the same Window — every Strip at zero electron and hole
        concentration, so it carries no Drude conductivity and the solve
        answers for the bare Traveling-wave electrode
        (:meth:`~gsim.modulator.staircase.StaircaseCrossSection.unloaded`).

        Returns:
            The Staircase Cross-section, drawn on its own component.
        """
        # Not through staircase(): the carriers are about to be switched
        # off, so whether the loaded Strips resolve the depletion is not
        # this Staircase's question.
        return self._staircase_for(self.bias_point().carriers).unloaded()

    def strip_material(self) -> RFStripMaterial:
        """The Strip lattice permittivity, valid up to the top frequency.

        The Strip materials are valid up to the highest requested
        frequency, which is what the solve asks of them.
        """
        return RFStripMaterial(
            permittivity=self.strip_permittivity, fmax_hz=max(self.frequencies_hz)
        )

    def _staircase_for(self, carriers: CarrierMap) -> StaircaseCrossSection:
        """The Staircase this Stage meshes, built from one Carrier map."""
        segments = self.region_segments(
            self.strip_extent(carriers),
            n_strips=self.n_strips,
            strips_per_region=self.strips_per_region,
        )
        return self.build_staircase(
            carriers,
            n_strips=self.n_strips,
            segments=segments,
            electrodes=replace(
                self.electrodes, conductor_model=self.effective_conductor_model()
            ),
        )

    def simulation(
        self,
        staircase: StaircaseCrossSection | None = None,
        *,
        output_dir: Path | None = None,
    ) -> BoundaryModeSim:
        """Assemble the cross-section this Stage meshes.

        The Staircase is a component of its own — Strips and electrodes,
        drawn in the Cross-section's transverse coordinates and extruded
        along the propagation direction — so the plane cuts through the
        middle of it rather than through the drawn device.

        Args:
            staircase: The Staircase to mesh; built from the Bias point
                when omitted.
            output_dir: Directory the mesh and solver files land in; the
                Stage's own when omitted.

        Returns:
            The configured (unmeshed) ``BoundaryModeSim``.
        """
        study = self._require_study()
        stair = staircase if staircase is not None else self.staircase()
        # One mesh serves the whole frequency sweep: the femwell Route
        # re-solves it per frequency without reading the boundary-mode
        # block, which records the first frequency so the sim is a
        # complete description.
        return self.build_staircase_simulation(
            stair,
            output_dir=(
                output_dir
                if output_dir is not None
                else study.stage_dir(self.stage_name)
            ),
            freq_hz=float(self.frequencies_hz[0]),
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def selection(self) -> dict[str, Any]:
        """A transmission line's bounds on the default selection rule."""
        return {
            "rule": self.rule,
            "min_index": self.min_index,
            "max_loss_ratio": self.max_loss_ratio,
            "degeneracy_rtol": self.degeneracy_rtol,
        }

    def window_hint(self) -> str:
        """A squeezed line Mode wants a wider Window or wider electrodes."""
        return (
            "The window is squeezing the line mode rather than shielding it. "
            f"Widen it with study.{self.stage_name}(window=..., window_z=...) "
            "or move the electrodes further apart."
        )

    def _wall_mode_hint(self) -> str:
        """The ways out of a selection that landed on the wall Mode.

        One hint for both Routes, so that whichever measurement caught
        it — the electrode currents or the gap voltage — the user is
        told the same thing.
        """
        return (
            "The loaded line mode is most often outside max_loss_ratio at an "
            f"undepleted bias; solve at a depleted one with study.{self.stage_name}"
            "(bias_v=...), move n_guess toward the loaded line's index, widen "
            "max_loss_ratio, or pass a rule= selecting the mode yourself."
        )

    def _check_wall_mode(self, reading: LineReading, freq_hz: float) -> None:
        """Warn when the selected Mode is the wall Mode, not the line Mode.

        A shielded two-electrode line has two propagating Modes. On the
        line Mode the signal and return electrodes carry equal and
        opposite currents; on the other both sit at one potential and
        their alike currents return through the metallic Window wall.
        The index alone does not tell them apart, and at an undepleted
        Bias the loaded line Mode can lose as fast as it advances and
        fall outside ``max_loss_ratio``, leaving the wall Mode as the
        slowest candidate. Each Route reads the diagnosis its own way —
        femwell off the two electrode currents, Palace off the gap
        voltage against the power — and the reading says which it was.

        Args:
            reading: What the Route read off the selected Mode.
            freq_hz: The frequency, named in the warning.

        Warns:
            UserWarning: When the reading says the Mode is the wall Mode.
                A reading that could not tell (``None``) does not warn.
        """
        if not reading.wall_mode:
            return
        warnings.warn(
            f"The {self.stage_name} stage's {self.route} route selected a mode "
            f"at f = {freq_hz / 1e9:g} GHz (n_eff = {reading.n_eff:.6g}) "
            f"{reading.diagnostic}. " + self._wall_mode_hint(),
            stacklevel=3,
        )

    def _guess_for(self, solved: Sequence[complex]) -> float | None:
        """Effective-index guess for the next frequency of the sweep.

        Args:
            solved: The line Modes' indices solved so far, in sweep order.

        Returns:
            The previous frequency's index when the Mode is tracked, and
            the configured guess otherwise.
        """
        if self.track_modes and solved:
            return float(solved[-1].real)
        return self.n_guess

    def _check_continuity(self, n_eff: Sequence[complex]) -> None:
        """Warn when ``n_rf(f)`` steps rather than disperses.

        A shift-and-invert search can settle on a different branch from
        one frequency to the next, and the sweep then reports an RF index
        that jumps. Tracking makes that unlikely rather than impossible,
        so the sweep is checked either way.

        Args:
            n_eff: The solved indices, in frequency order.
        """
        indices = np.asarray([value.real for value in n_eff], dtype=np.float64)
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64)
        if indices.size < 2:
            return
        steps = np.abs(np.diff(indices)) / np.maximum(np.abs(indices[:-1]), 1e-12)
        # Against the frequency step, not against a fixed number: the
        # sweep's spacing is the user's, and a line disperses by about as
        # much as its frequency moves.
        spacing = np.diff(frequencies) / frequencies[:-1]
        jumped = np.nonzero(steps > self.jump_rtol * spacing)[0]
        if jumped.size == 0:
            return
        where = ", ".join(
            f"{self.frequencies_hz[i] / 1e9:g} -> "
            f"{self.frequencies_hz[i + 1] / 1e9:g} GHz "
            f"({indices[i]:.3g} -> {indices[i + 1]:.3g})"
            for i in jumped
        )
        tracked = (
            " Every frequency after it was aimed at the index solved before "
            "it, so the branch it changed to is the one the rest of the "
            "sweep followed, smoothly and wrongly."
            if self.track_modes
            else ""
        )
        warnings.warn(
            f"The {self.stage_name} stage's rf index jumps across the sweep "
            f"at {where}, faster than the frequency step: the eigenvalue "
            "search has most likely settled on a different mode branch "
            f"rather than the line mode dispersing.{tracked} Raise "
            f"num_modes, tighten the selection with study.{self.stage_name}"
            "(min_index=...), or solve the frequencies from the jump onwards "
            "on their own.",
            stacklevel=2,
        )

    def line_conductors(
        self, staircase: StaircaseCrossSection
    ) -> tuple[Conductor, Conductor | None]:
        """The signal electrode and, when there is exactly one, the return.

        Args:
            staircase: The Staircase the electrodes were drawn on.

        Returns:
            ``(signal, return_)``; ``return_`` is ``None`` on a Staircase
            with other than one electrode beside the signal, which then
            has no pair to compare and is not checked for the wall Mode.
        """
        signal = staircase.conductor(self.signal_electrode())
        others = [name for name in staircase.electrode_names if name != signal.name]
        return_ = staircase.conductor(others[0]) if len(others) == 1 else None
        return signal, return_

    def _solve_line(
        self, sim: BoundaryModeSim, staircase: StaircaseCrossSection, route: Route
    ) -> tuple[list[complex], list[complex]]:
        """Solve the meshed Staircase at every frequency, on the Route.

        One loop for both Routes: the route solves, the shared
        selection picks the line Mode and checks its Window, the
        route reads the selected Mode, and the Stage says so when the
        reading is the wall Mode's.

        Args:
            sim: The meshed Staircase simulation.
            staircase: The Staircase it was built from.
            route: The Route answering this run.

        Returns:
            ``(n_eff, z0_ohm)``, one entry per configured frequency.
        """
        signal, return_ = self.line_conductors(staircase)
        route.prepare_line(
            sim, signal=signal, return_=return_, stage_name=self.stage_name
        )
        verbose = self._is_verbose()

        n_eff: list[complex] = []
        z0_ohm: list[complex] = []
        for freq in self.frequencies_hz:
            modes = route.solve(
                sim,
                freq_hz=freq,
                num_modes=self.num_modes,
                target=self._guess_for(n_eff),
                order=self.order,
                verbose=verbose,
                stage_name=self.stage_name,
            )
            mode, _ratio = self.select_mode(modes, route, at=f"f = {freq / 1e9:g} GHz")
            reading = route.read_line(
                sim,
                mode,
                freq_hz=freq,
                signal=signal,
                return_=return_,
                stage_name=self.stage_name,
            )
            self._check_wall_mode(reading, freq)
            n_eff.append(complex(reading.n_eff))
            z0_ohm.append(complex(reading.z0_ohm))
        return n_eff, z0_ohm

    def check_route(self) -> Route:
        """The selected Route, its Backend and this Stage's settings checked.

        Before the charge solve and before meshing: a user whose Route
        cannot run, or whose settings the Route cannot honour, should
        pay nothing to find that out. Every check reads settings only.

        Returns:
            The Route for this run.
        """
        route = super().check_route()
        route.check_line_settings(
            conductor_model=self.effective_conductor_model(),
            metallic_boundaries=self.metallic_boundaries,
            order=self.order,
            stage_name=self.stage_name,
        )
        return route

    def _solve_staircase(
        self,
        staircase: StaircaseCrossSection,
        route: Route,
        *,
        bias_v: float,
        output_dir: Path | None = None,
        unloaded: bool = False,
    ) -> RFLineParams:
        """Mesh one Staircase and solve the line Mode at every frequency.

        The result records the Bias the Staircase was built at and the
        Contact its impedance was read over, so nothing downstream has to
        ask this Stage what it solved.
        """
        from gsim.common.transmission_line import line_params_from_neff

        sim = self.simulation(staircase, output_dir=output_dir)
        sim.mesh(**self.mesh)
        n_eff, z0_ohm = self._solve_line(sim, staircase, route)
        self._check_continuity(n_eff)

        return line_params_from_neff(
            np.asarray(self.frequencies_hz, dtype=np.float64),
            n_eff,
            z0_ohm=z0_ohm,
            unloaded=unloaded,
            bias_v=bias_v,
            signal_contact=self.signal_contact_name(),
        )

    def run_unloaded(self, *, force: bool = False) -> RFLineParams:
        """The bare electrode's line parameters: same Staircase, carriers off.

        The "EM solve of the bare electrode" half of the classic
        loaded-line workflow: the Bias point's Staircase with every Strip
        at zero electron and hole concentration, geometry, electrodes,
        Window and conductor model unchanged (ADR 0003). The result is
        flagged ``unloaded`` so it cannot be mistaken for a Bias point's
        answer, and is cached alongside the loaded one — re-configuring
        the Stage drops both.

        Args:
            force: Solve again even when a cached result is available.

        Returns:
            The unloaded line parameters.
        """
        if self._unloaded_result is not None and not force:
            return self._unloaded_result
        route = self.check_route()
        output_dir = self._require_study().stage_dir(self.stage_name) / "unloaded"
        output_dir.mkdir(parents=True, exist_ok=True)
        line = self._solve_staircase(
            self.unloaded_staircase(),
            route,
            bias_v=self.bias_point().bias_v,
            output_dir=output_dir,
            unloaded=True,
        )
        self._unloaded_result = line
        return line

    def junction_branch(self) -> JunctionBranch:
        """The series-RC junction branch at this Stage's Bias point.

        Read off the charge sweep's small-signal admittance at the same
        Bias the loaded solve uses — the shunt branch per meter of
        Traveling-wave electrode the loaded-line assembly inserts.

        Returns:
            The fitted :class:`~gsim.common.transmission_line.JunctionBranch`.

        Raises:
            ValueError: When the sweep's point holds no small-signal
                admittance, or one a series RC cannot represent.
        """
        bias = self.bias_point().bias_v
        sweep: BiasSweepResult = self._require_study().charge.run()
        return sweep.point_at(bias, tol=BIAS_TOL_V).junction_branch()

    def junction_branches(self) -> tuple[NDArray[np.float64], JunctionBranch]:
        """The series-RC junction branch at every Bias point of the sweep.

        The same reading as :meth:`junction_branch`, over the whole
        sweep: what a consumer of the exported junction model compares
        its file against. This Stage is where the charge sweep's
        admittance is read into a shunt branch, so the line Stage asks
        here rather than reading the charge Stage itself.

        Returns:
            ``(bias_v, branch)`` — the biases in sweep order (V) and a
            :class:`~gsim.common.transmission_line.JunctionBranch` of arrays over
            them.

        Raises:
            ValueError: When a Bias point holds no small-signal
                admittance, or one a series RC cannot represent.
        """
        sweep: BiasSweepResult = self._require_study().charge.run()
        return sweep.voltages, sweep.junction_branch()

    def crosscheck(self, *, force: bool = False) -> LoadedLineComparison:
        """Both loaded-line routes side by side, at this Stage's Bias point.

        The demonstration that the pipeline and the standard workflow
        are the same compact model assembled two ways: the direct route
        solves the carrier-loaded Staircase (:meth:`run`); the assembled
        route loads the unloaded RLGC (:meth:`run_unloaded`) with the
        charge solve's series-RC junction branch
        (:func:`gsim.common.transmission_line.loaded_line_params`). The returned
        comparison holds both routes' n_RF, loss and Z0 with their
        relative deltas, and its ``check()`` fails loudly where they
        disagree.

        Args:
            force: Re-solve both routes even where results are cached.

        Returns:
            The side-by-side comparison.
        """
        from gsim.common.transmission_line import (
            line_params_from_gamma,
            loaded_line_params,
        )
        from gsim.modulator.report import LoadedLineComparison

        direct: RFLineParams = self.run(force=force)
        unloaded = self.run_unloaded(force=force)
        freq = np.asarray(self.frequencies_hz, dtype=np.float64)
        gamma, z0 = loaded_line_params(
            freq, rlgc=unloaded.rlgc, junction=self.junction_branch()
        )
        assembled = line_params_from_gamma(freq, gamma, z0_ohm=z0)
        return LoadedLineComparison(direct=direct, assembled=assembled)

    def invalidate(self) -> None:
        """Drop the loaded and the unloaded result, and downstream ones."""
        self._unloaded_result = None
        super().invalidate()

    def _solve(self) -> RFLineParams:
        """Mesh the Staircase and solve the line Mode at every frequency."""
        route = self.check_route()
        point = self.bias_point()
        return self._solve_staircase(self.staircase(), route, bias_v=point.bias_v)
