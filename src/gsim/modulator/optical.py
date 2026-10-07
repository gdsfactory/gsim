"""The optical Stage: the carrier-perturbed Mode at every Bias point.

The Stage answers one question — what does the Phase shifter's optical
Mode do as the bias moves — and it answers it on a Window of its own. A
Window sized to span the Contacts is not a Window that contains a 1.55 um
Mode, so the optical Stage derives a box around the rib instead, meshes
it, and carries the Carrier maps of the charge solve onto that mesh with
an explicit transfer (ADR 0002). Nothing here reuses the charge mesh.

Each Bias point becomes one complex effective index, from which the two
numbers a designer wants follow: the index shift relative to zero bias,
which sets modulation efficiency, and the bias-dependent loss. A third
is there for the asking: the group index the Velocity mismatch is
measured against, which no solve at one wavelength produces, so
:meth:`OpticalStage.group_index` solves two more either side of it.

Either Backend can answer, and the choice changes how the carriers reach
the solver. A Route that carries a continuous permittivity — femwell,
the default — solves the drawn device with ``eps(x, y)`` projected onto
its mesh elements. One that takes piecewise-constant materials per
Region and nothing else — Palace — moves the Stage onto a Staircase: the
doped silicon replaced by Strips tiling it, and the rest of the drawn
Cross-section — the slab, the metal on the pads, whatever else the plane
crosses — redrawn around them, in the same Window. Asking for a strip
count puts the femwell Route on that same Staircase too, which is what
makes the two Routes comparable at all: they differ then in their
materials and in nothing else. The one choice that is this Stage's own
is that one — continuous Carrier map or Staircase; what a Route can and
cannot check, it says for itself.

The selected Route's runtime is checked before the Stage meshes, so a
missing extra or a missing binary costs nothing but the error message.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gsim.modulator.em import EMStage
from gsim.modulator.route import DEFAULT_PALACE_STRIPS
from gsim.modulator.staircase import DEFAULT_SI_INDEX, OpticalStripMaterial

if TYPE_CHECKING:
    from pathlib import Path

    import meshio

    from gsim.modulator.carriers import CarrierResponse, CarrierResponseSweep
    from gsim.modulator.route import Route
    from gsim.modulator.staircase import (
        StaircaseCrossSection,
        SurroundingRegion,
    )
    from gsim.palace import BoundaryModeSim
    from gsim.tcad.results import CarrierMap

__all__ = ["GroupIndex", "OpticalMode", "OpticalStage", "OpticalSweep"]

#: One solved Mode: ``(bias_v, n_eff, boundary_field_ratio)``.
_Solved = tuple[float, complex, float]

#: dB per cm of propagation loss per unit of ``|Im(n_eff)| / lambda[cm]``.
_DB_PER_CM = 40.0 * np.pi / np.log(10.0)


class OpticalMode(BaseModel):
    """The Phase shifter's optical Mode at one Bias point.

    Attributes:
        bias_v: Applied bias on the swept Contact (V).
        n_eff: Complex effective index (``exp(+i omega t)``: a lossy Mode
            has ``Im(n_eff) < 0``).
        index_shift: ``Re(n_eff)`` minus its value at the reference bias.
        loss_db_cm: Propagation loss at this bias (dB/cm).
        boundary_field_ratio: Peak field at the Window boundary over peak
            field overall; large values mean the Window is too small.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bias_v: float
    n_eff: complex
    index_shift: float
    loss_db_cm: float
    boundary_field_ratio: float


class GroupIndex(BaseModel):
    """The Phase shifter's optical group index, and what it was read off.

    ``n_g = n_eff - lambda * d(n_eff)/d(lambda)``, the slope a central
    difference between two Modes solved either side of the Stage's
    wavelength, at the reference Bias point and with every material
    re-resolved at the wavelength it is solved at.

    Attributes:
        n_group: The group index at ``wavelength_um``.
        wavelength_um: The wavelength the group index is taken at (um).
        step_um: The finite-difference step either side of it (um).
        bias_v: The Bias point the Modes were solved at (V).
        wavelengths_um: The two wavelengths solved, below and above (um).
        n_eff: ``Re(n_eff)`` of the Mode at those two wavelengths.
        core_index: Index the guide's core resolved to at those two
            wavelengths, before the carriers move it.
    """

    n_group: float
    wavelength_um: float
    step_um: float
    bias_v: float
    wavelengths_um: tuple[float, float]
    n_eff: tuple[float, float]
    core_index: tuple[float, float]

    @property
    def material_dispersion(self) -> bool:
        """Whether the core's own index moved between the two wavelengths."""
        return self.core_index[0] != self.core_index[1]


class OpticalSweep(BaseModel):
    """The optical Mode across the whole Bias sweep.

    Attributes:
        contact: The Contact the charge sweep drove.
        wavelength_um: Vacuum wavelength the Modes were solved at (um).
        reference_bias_v: The bias the index shift is measured from.
        points: One solved Mode per Bias point, in sweep order.
        group_index: The group index at the reference bias, once it has
            been asked for (:meth:`OpticalStage.group_index`); it costs
            two more solves, so the sweep does not pay for it unasked.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    contact: str
    wavelength_um: float
    reference_bias_v: float
    points: list[OpticalMode] = Field(default_factory=list)
    group_index: GroupIndex | None = None

    @property
    def voltages(self) -> NDArray[np.float64]:
        """Applied biases (V) in sweep order."""
        return np.asarray([p.bias_v for p in self.points], dtype=np.float64)

    @property
    def n_eff(self) -> NDArray[np.complex128]:
        """Complex effective indices in sweep order."""
        return np.asarray([p.n_eff for p in self.points], dtype=np.complex128)

    @property
    def index_shift(self) -> NDArray[np.float64]:
        """Index shift from the reference bias, in sweep order."""
        return np.asarray([p.index_shift for p in self.points], dtype=np.float64)

    @property
    def loss_db_cm(self) -> NDArray[np.float64]:
        """Propagation loss (dB/cm) in sweep order."""
        return np.asarray([p.loss_db_cm for p in self.points], dtype=np.float64)


class OpticalStage(EMStage):
    """The carrier-perturbed optical Mode, bias point by bias point.

    The two representations are not interchangeable, and what the
    Staircase costs is a strip count. On the demo Phase shifter at
    1.55 um, against the drawn device solved with a continuous
    ``eps(x, y)``: two Strips land within 0.3% on ``Re n_eff`` but 59% on
    the index shift, sixteen Strips within 0.01% and 2%. The index shift
    is the quantity ``VpiL`` is computed from and by far the slower of
    the two to converge — the index is a whole guide's worth of material
    and the shift is a sliver's — so a Staircase chosen for ``VpiL``
    wants many more Strips than one chosen for the index. Both errors
    shrink with the strip count, which is the whole claim, and what
    ``tests/modulator/test_representation_gate.py`` holds the Route to.

    The settings every EM Stage takes — ``route``, ``num_modes``,
    ``min_index``, ``boundary_field_tol``, ``metallic_boundaries``,
    ``order``, ``n_guess``, the Window and the mesh — are documented on
    :class:`~gsim.modulator.em.EMStage`, and their defaults there are
    this Stage's. The metallic wall matters here because the drawn metal
    inside an optical Window is part of that boundary: a conductor is
    meshed as an outline with its interior left out (ADR 0003), and
    femwell applies this one condition to every facet at once, so
    turning it off leaves the electrodes as open slots while the Palace
    Route still reads them as perfect conductors — the two Routes would
    solve different boundary-value problems.

    Attributes:
        n_strips: Number of Strips the carrier response is reduced to.
            ``None`` — the default — keeps the continuous ``eps(x, y)``
            of the drawn device, which only a Route carrying continuous
            materials can solve; a Route that cannot staircases with
            :data:`~gsim.modulator.route.DEFAULT_PALACE_STRIPS` instead.
            Setting a count puts either Route on the Staircase, so the
            two solve the identical problem.
        strip_span: ``(min, max)`` extent the Strips tile along the
            junction axis (um); the doped slab — every doped Region, pads
            included — when unset, which is both the guide's core and the
            widest extent the Carrier map covers. Unused when the
            continuous profile is solved.
        strip_index: Unperturbed refractive index of the Strips, which
            the plasma dispersion perturbs. ``None`` — the default —
            reads it off the drawn Junction's own material at
            ``wavelength_um``, which is what the continuous profile
            perturbs, so the two representations start from one index.
            Unused when the continuous profile is solved.
        substrate_thickness_um: Substrate below the Staircase (um).
            Unused when the continuous profile is solved.
        wavelength_um: Vacuum wavelength of the solve (um).
        window: In-plane optical Window (um); derived as a box around the
            rib when unset. A Staircase is clipped to the same Window as
            the continuous profile, so the two Routes mesh one domain.
        window_z: Vertical Window (um); derived from the guiding layer
            when unset.
        mode_margin_um: Half-width of the derived Window either side of
            the Junction (um).
        z_above_um: Margin above the guiding layer in the derived vertical
            Window (um).
        z_below_um: Margin below it (um).
        perturbed_regions: Regions whose permittivity the Carrier maps
            perturb; defaults to the device's doped Regions.
        strip_field_tol: Warn above this fraction of the Mode's power
            falling outside the Strip extent — the sign of Strips too
            narrow to carry the carrier response where the Mode actually
            is. Measured only by a Route that reads its Mode's fields.
        group_index_step_um: Finite-difference step of the group index
            (um): :meth:`group_index` solves the Mode this far either
            side of ``wavelength_um``. The difference is central, so its
            error falls with the square of the step: a silicon slab's
            group index moves by 1e-5 between 10 nm and 5 nm. Much
            smaller, and the difference of two solved indices starts to
            read the eigensolver's tolerance instead.
    """

    stage_name: ClassVar[str] = "optical"

    n_strips: int | None = Field(default=None, ge=1)
    strip_index: float | None = Field(default=None, gt=0.0)
    wavelength_um: float = Field(default=1.55, gt=0.0)
    mode_margin_um: float = Field(default=2.0, gt=0.0)
    z_above_um: float = Field(default=1.0, ge=0.0)
    z_below_um: float = Field(default=1.0, ge=0.0)
    perturbed_regions: list[str] | None = None
    strip_field_tol: float = Field(default=0.5, gt=0.0, le=1.0)
    group_index_step_um: float = Field(default=0.01, gt=0.0)

    @model_validator(mode="after")
    def _step_inside_the_wavelength(self) -> OpticalStage:
        """Both wavelengths the group index is read off must be positive."""
        if self.group_index_step_um >= self.wavelength_um:
            raise ValueError(
                f"group_index_step_um = {self.group_index_step_um:g} reaches "
                f"past zero from wavelength_um = {self.wavelength_um:g}; the "
                "step is a small fraction of the wavelength, 0.01 um by default."
            )
        return self

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    def default_strip_span(self) -> tuple[float, float]:
        """The doped slab — every doped Region, pads included.

        This Staircase is drawn inside the device's own Cross-section
        (ADR 0004), so its Strips stand for the doped silicon and nothing
        else. Tiling the rib alone would leave the pads as unperturbed
        drawn silicon in the middle of the guide, and the doped slab is
        both the guide's core and the widest extent the Carrier map
        covers.

        Returns:
            ``(min, max)`` along the junction axis (um).
        """
        return self._require_study().layout.doped_span

    def mode_window(self) -> tuple[float, float]:
        """The in-plane Window a Mode of this Stage is solved in (um).

        Returns:
            The configured ``window``, or a box around the Junction
            ``mode_margin_um`` wide either side (ADR 0002).
        """
        if self.window is not None:
            return self.window
        return self._require_study().layout.window_around_junction(
            margin_um=self.mode_margin_um
        )

    def mode_window_z(self) -> tuple[float, float]:
        """The vertical Window a Mode of this Stage is solved in (um).

        Returns:
            The configured ``window_z``, or the guiding layer cleared by
            ``z_above_um`` and ``z_below_um``.
        """
        if self.window_z is not None:
            return self.window_z
        return self._require_study().layout.window_z_around_guide(
            above_um=self.z_above_um, below_um=self.z_below_um
        )

    def perturbed_region_names(self) -> list[str]:
        """Regions the Carrier maps perturb.

        Returns:
            The configured Regions, or the device's doped Regions.
        """
        if self.perturbed_regions is not None:
            return list(self.perturbed_regions)
        return list(self._require_study().device.doped_regions)

    def _regions_on_mesh(self, available: Iterable[str]) -> list[str]:
        """The perturbed Regions the optical mesh actually carries.

        The optical Window is a box around the rib, so a doped Region far
        enough from the Junction — a contact pad, usually — is simply not
        on this mesh, and is left unperturbed rather than treated as an
        error. A Region named explicitly is a different matter: the user
        asked for it, so its absence is reported.

        Args:
            available: Every 2D Region name on the optical mesh.

        Returns:
            The Regions to perturb, in configured order.

        Raises:
            ValueError: When a named Region is off the mesh, or when the
                Window leaves no perturbed Region on it at all.
        """
        on_mesh = set(available)
        wanted = self.perturbed_region_names()
        if self.perturbed_regions is not None:
            missing = [name for name in wanted if name not in on_mesh]
            if missing:
                raise ValueError(
                    f"Perturbed region(s) {missing} are not on the "
                    f"{self.stage_name} window. On it: {sorted(on_mesh)}."
                )
            return wanted
        present = [name for name in wanted if name in on_mesh]
        if not present:
            raise ValueError(
                f"The {self.stage_name} window contains none of the doped "
                f"regions {wanted}, so the carriers would perturb nothing. "
                f"Widen it with study.{self.stage_name}(mode_margin_um=...) "
                "or set window= explicitly."
            )
        return present

    def effective_n_strips(self) -> int | None:
        """Strip count this Stage will actually solve with.

        Returns:
            The configured count; ``None`` when the continuous
            ``eps(x, y)`` is solved instead, which only a Route carrying
            continuous materials can do, so a Route that cannot falls
            back to :data:`~gsim.modulator.route.DEFAULT_PALACE_STRIPS`.
        """
        if self.n_strips is not None:
            return int(self.n_strips)
        if self.resolved_route().continuous_materials:
            return None
        return DEFAULT_PALACE_STRIPS

    def unperturbed_index(self) -> float:
        """Refractive index the Strips carry before the carriers move it.

        The continuous profile perturbs each drawn Region's own index; a
        Staircase that starts from a textbook silicon index instead
        differs from it by a constant offset at every bias — small, but
        it is the whole of what separates the two representations once
        the geometry matches.

        Returns:
            The configured ``strip_index``, or the index of the drawn
            Junction's material at this Stage's wavelength.
            :data:`~gsim.modulator.staircase.DEFAULT_SI_INDEX` stands
            in for a material the stack cannot resolve.
        """
        if self.strip_index is not None:
            return float(self.strip_index)
        return self._drawn_core_index()

    def _drawn_core_index(self) -> float:
        """Index of the drawn Junction's material at this Stage's wavelength."""
        from gsim.common.stack.materials import (
            MaterialProperties,
            resolve_material_at_wavelength,
        )

        study = self._require_study()
        # Either side of the metallurgical boundary answers: the two are
        # the same silicon, differing in dopant and not in host index.
        region = study.layout.junction.regions[0]
        layer = study.stack.layers.get(region)
        if layer is None:
            return DEFAULT_SI_INDEX
        overrides = {
            name: (
                props
                if isinstance(props, MaterialProperties)
                else MaterialProperties.model_validate(props)
            )
            for name, props in (study.stack.materials or {}).items()
        }
        resolved = resolve_material_at_wavelength(
            layer.material, self.wavelength_um, overrides=overrides
        )
        if resolved is None or resolved.permittivity_scalar is None:
            return DEFAULT_SI_INDEX
        return float(np.sqrt(float(resolved.permittivity_scalar)))

    def surroundings(
        self, span: tuple[float, float] | None = None
    ) -> tuple[SurroundingRegion, ...]:
        """The drawn device redrawn around the Strips.

        The Strips carry the Carrier map and nothing else, so a Staircase
        made of Strips alone is a silicon wire in the background medium —
        not the drawn guide. Every other Region on the drawn
        Cross-section is cut against the Strip footprint and redrawn
        beside them: the undoped slab the rib sits on, the Traveling-wave
        metal landing on the pads, and whatever else the plane crosses.

        The drawn conductors matter most. Metal inside the optical Window
        is meshed as a perfect conductor either Route honours, and
        omitting it moved this device's index by 0.18 — six times what
        the strip count moves it.

        Args:
            span: The Strip extent the Regions are cut against;
                :meth:`strip_extent` decides when omitted.

        Returns:
            The surrounding Regions, empty when the drawn Cross-section
            has nothing on it but the doped silicon the Strips replace.
        """
        from gsim.modulator.staircase import surroundings_from_section

        study = self._require_study()
        junction = study.layout.junction_span
        return surroundings_from_section(
            study.section,
            strip_span=span if span is not None else self.strip_extent(),
            strip_z=junction.z,
            stack=study.stack,
        )

    def staircase(self, point: CarrierResponse) -> StaircaseCrossSection:
        """Reduce one Bias point's Carrier map to a meshable Staircase.

        The Strips tile :meth:`strip_extent` — the doped slab unless
        ``strip_span`` says otherwise — and take the plasma-dispersion
        coefficients of the carriers Stage: the same coupling the
        continuous profile reads, evaluated on strip averages instead of
        on mesh elements, and at this Stage's own ``wavelength_um``, so
        both paths carry the same loss. Around the Strips the Staircase
        redraws the device itself (see :meth:`surroundings`), so the two
        representations differ in their materials and not in their
        geometry. No flanking electrodes are invented: the drawn ones are
        already among the surrounding Regions.

        Args:
            point: The Bias point to staircase.

        Returns:
            The Staircase Cross-section, drawn on its own component.

        Raises:
            ValueError: When this Stage is solving the continuous
                profile, so there is no strip count to tile with.
        """
        n_strips = self.effective_n_strips()
        if n_strips is None:
            raise ValueError(
                f"The {self.stage_name} stage is solving the continuous "
                "permittivity, so it builds no staircase. Ask for one with "
                f"study.{self.stage_name}(n_strips=...)."
            )
        span = self.strip_extent(point.carriers)
        self._check_heights(span)
        return self.build_staircase(
            point.carriers,
            n_strips=n_strips,
            electrodes=None,
            span=span,
            surroundings=self.surroundings(span),
        )

    def _check_heights(self, span: tuple[float, float]) -> None:
        """Warn when the Strips stand taller than the silicon they replace.

        The optical Staircase draws every Strip at the Junction's height.
        That is the device only when every doped Region the Strips tile
        stands as tall as the rib; beside a thinner slab it thickens the
        slab to the rib's height, and the guide it answers for is not the
        drawn one. The RF Staircase follows the Regions instead
        (:meth:`~gsim.modulator.em.EMStage.region_segments`).

        Args:
            span: The extent the Strips will tile (um).
        """
        layout = self._require_study().layout
        height = layout.junction_span.z
        lower = sorted(
            name
            for name in layout.doped_regions
            if layout.region_spans[name].z != height
            and layout.region_spans[name].h[1] > span[0]
            and layout.region_spans[name].h[0] < span[1]
        )
        if not lower:
            return
        warnings.warn(
            f"The {self.stage_name} stage's staircase draws every strip at the "
            f"rib's height, {height[0]:.3g}..{height[1]:.3g} um, but the doped "
            f"regions {lower} stand lower: the strips thicken them to the "
            "rib's height, and the mode solved is not the drawn guide's. "
            f"Narrow study.{self.stage_name}(strip_span=...) to the rib, or "
            "solve the continuous profile on the femwell route.",
            stacklevel=3,
        )

    def strip_material(self) -> OpticalStripMaterial:
        """This Stage's wavelength and the Strips' unperturbed index.

        The coefficients are the carriers Stage's, but the wavelength they
        are read at is this Stage's: the model's own is where it was
        fitted, not where the Mode is solved.
        """
        return OpticalStripMaterial(
            wavelength_um=self.wavelength_um, index=self.unperturbed_index()
        )

    def staircase_simulation(
        self, staircase: StaircaseCrossSection, *, output_dir: str | Path
    ) -> BoundaryModeSim:
        """Assemble the Staircase cross-section this Stage meshes.

        The Staircase is a component of its own, so the plane cuts through
        the middle of it rather than through the drawn device. It is
        clipped to the same Window the continuous profile is solved in,
        because the Staircase now carries the same Regions: two Routes
        meshing different domains would not be comparable whatever their
        materials agreed on.

        Args:
            staircase: The Staircase to mesh.
            output_dir: Directory this Bias point's mesh and solver files
                land in; one per point, because the Strip materials move
                with the bias.

        Returns:
            The configured (unmeshed) ``BoundaryModeSim``.
        """
        from scipy.constants import speed_of_light as c0

        return self.build_staircase_simulation(
            staircase,
            output_dir=output_dir,
            freq_hz=c0 / (self.wavelength_um * 1e-6),
            window=self.mode_window(),
            window_z=self.mode_window_z(),
        )

    def simulation(self, *, output_dir: str | Path | None = None) -> BoundaryModeSim:
        """Assemble the cross-section this Stage meshes.

        The Window is the optical one — a box around the rib derived from
        the Junction, never the charge Stage's slab (ADR 0002) — and the
        mesh lands in the Stage's own output directory.

        Args:
            output_dir: Directory the mesh and solver files land in; the
                Stage's own when omitted.

        Returns:
            The configured (unmeshed) ``BoundaryModeSim``.
        """
        from scipy.constants import speed_of_light as c0

        study = self._require_study()
        return self.new_simulation(
            stack=study.stack,
            component=study.component,
            plane=study.plane,
            output_dir=(
                output_dir
                if output_dir is not None
                else study.stage_dir(self.stage_name)
            ),
            freq_hz=c0 / (self.wavelength_um * 1e-6),
            window=self.mode_window(),
            window_z=self.mode_window_z(),
        )

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _element_epsilon(
        self,
        mesh: meshio.Mesh,
        base_epsilon: dict[str, complex],
        carriers: CarrierMap,
    ) -> NDArray[np.complex128]:
        """Per-element permittivity of one Bias point on the optical mesh.

        Every element starts from its Region's stack permittivity; the
        perturbed Regions then take the index and absorption the
        transferred Carrier map implies, so the carrier loss replaces the
        material's own rather than adding to it.
        """
        from gsim.common.carrier_transfer import transfer_carriers
        from gsim.common.carriers import permittivity_perturbation

        study = self._require_study()
        regions = self._regions_on_mesh(base_epsilon)
        transferred = transfer_carriers(
            carriers, mesh, at="elements", fill=0.0, regions=regions
        )
        response = study.carriers.response(
            transferred.electrons_cm3, transferred.holes_cm3
        )

        unmapped = sorted(set(transferred.region) - set(base_epsilon))
        if unmapped:
            raise ValueError(
                f"The optical mesh has element(s) in region(s) {unmapped} with "
                "no permittivity; every 2D region must resolve to a material."
            )
        epsilon = np.asarray(
            [base_epsilon[name] for name in transferred.region],
            dtype=np.complex128,
        )
        wanted = set(regions)
        for index, name in enumerate(transferred.region):
            if name not in wanted:
                continue
            epsilon[index] = permittivity_perturbation(
                n0=float(np.sqrt(base_epsilon[name].real)),
                dn=float(response.index_shift[index]),
                dalpha_cm=float(response.absorption_cm[index]),
                wavelength_um=self.wavelength_um,
            )
        return epsilon

    def _check_strip_coverage(
        self, route: Route, mode: Any, bias_v: float, span: tuple[float, float]
    ) -> None:
        """Warn when the Mode mostly sits off the carrier-bearing Strips.

        The Strips are the only Regions of a Staircase the Carrier map
        reaches. A Mode whose power is largely outside them is answered
        by the surrounding Regions, which carry the drawn materials and
        no carriers at all — so the index shift the bias sweep reports is
        the shift of whatever fraction of the Mode the Strips do hold. A
        Route that cannot measure the fraction answers NaN and has
        already said why.
        """
        fraction = route.strip_fraction_outside(mode, span, stage_name=self.stage_name)
        if np.isnan(fraction) or fraction <= self.strip_field_tol:
            return
        warnings.warn(
            f"The {self.stage_name} stage's staircase at V = {bias_v:g} "
            f"carries {fraction:.1%} of the mode's power outside the strip "
            f"extent {span[0]:.3g}..{span[1]:.3g} um (tolerance "
            f"{self.strip_field_tol:.1%}); only the strips carry the carrier "
            "response, so the index shift is that of the fraction inside "
            f"them. Widen the strips with study.{self.stage_name}"
            "(strip_span=...) — up to what the carrier map covers — and "
            "widen the charge window with study.charge(window=...) to make "
            "a wider span legal.",
            stacklevel=2,
        )

    def window_hint(self) -> str:
        """A clipped optical Mode wants a wider derived Window."""
        return (
            f"Widen the window with study.{self.stage_name}(mode_margin_um=...) / "
            "(z_above_um=..., z_below_um=...) or set it explicitly."
        )

    def _sweep_from(
        self,
        contact: str,
        solved: list[tuple[float, complex, float]],
    ) -> OpticalSweep:
        """Assemble the sweep both Routes report.

        Args:
            contact: The Contact the charge sweep drove.
            solved: ``(bias_v, n_eff, boundary_field_ratio)`` per Bias
                point, in sweep order.

        Returns:
            The optical sweep, its index shift measured from zero bias
            when the sweep visited it and from its first point otherwise.
        """
        reference_bias, reference_index, _ = next(
            (entry for entry in solved if entry[0] == 0.0), solved[0]
        )
        wavelength_cm = self.wavelength_um * 1e-4
        return OpticalSweep(
            contact=contact,
            wavelength_um=self.wavelength_um,
            reference_bias_v=reference_bias,
            points=[
                OpticalMode(
                    bias_v=bias_v,
                    n_eff=n_eff,
                    index_shift=n_eff.real - reference_index.real,
                    loss_db_cm=_DB_PER_CM * abs(n_eff.imag) / wavelength_cm,
                    boundary_field_ratio=ratio,
                )
                for bias_v, n_eff, ratio in solved
            ],
        )

    def _bias_sweep(self) -> CarrierResponseSweep:
        """The Bias sweep to solve, running the upstream Stages if needed.

        Raises:
            ValueError: When the sweep is empty, so there is no Mode to
                solve.
        """
        responses: CarrierResponseSweep = self._require_study().carriers.run()
        if not responses.points:
            raise ValueError(
                "The bias sweep has no points, so there is no mode to solve. "
                "Configure study.charge(biases=[...])."
            )
        return responses

    def _solve_continuous(
        self, route: Route, points: Sequence[CarrierResponse], *, directory: Path
    ) -> list[_Solved]:
        """Solve the drawn device with a continuous ``eps(x, y)``.

        One mesh serves every Bias point: the geometry does not move with
        the bias, only the per-element permittivity the Carrier maps
        imply.

        Args:
            route: The Route answering this run; one that carries
                continuous materials.
            points: The Bias points to solve, in order.
            directory: Where the mesh lands.
        """
        import meshio
        from scipy.constants import speed_of_light as c0

        from gsim.femwell.adapter import epsilon_by_region

        study = self._require_study()
        sim = self.simulation(output_dir=directory)
        # The Carrier map is transferred onto this mesh, element by
        # element: where the charge mesh resolved the depletion edge this
        # one has to, or the transfer smears it (the index shift read
        # 15 % low at this Stage's own refined size).
        boxes = {"refinement_boxes": study.charge.junction_boxes()}
        sim.mesh(**(boxes | self.mesh))
        mesh = meshio.read(str(sim.mesh_path))
        base_epsilon = epsilon_by_region(
            mesh, study.stack, wavelength_um=self.wavelength_um
        )

        solved: list[_Solved] = []
        for point in points:
            modes = route.solve(
                sim,
                freq_hz=c0 / (self.wavelength_um * 1e-6),
                num_modes=self.num_modes,
                target=self.n_guess,
                order=self.order,
                verbose=self._is_verbose(),
                stage_name=self.stage_name,
                epsilon=self._element_epsilon(mesh, base_epsilon, point.carriers),
            )
            mode, ratio = self.select_mode(modes, route, at=f"V = {point.bias_v:g}")
            solved.append((point.bias_v, complex(mode.n_eff), ratio))
        return solved

    def _solve_staircase(
        self, route: Route, points: Sequence[CarrierResponse], *, directory: Path
    ) -> list[_Solved]:
        """Solve the Staircase of every Bias point, on the Route.

        Each Bias point gets its own Staircase, and its own mesh under its
        own directory: the Strip edges do not move with the bias but the
        Strip materials do, and a Backend may read its materials off the
        meshed stack rather than from an array handed in per solve.

        Args:
            route: The Route answering this run.
            points: The Bias points to solve, in order.
            directory: Where the per-point directories land.
        """
        from scipy.constants import speed_of_light as c0

        verbose = self._is_verbose()
        solved: list[_Solved] = []
        for index, point in enumerate(points):
            staircase = self.staircase(point)
            route.check_staircase(
                staircase,
                window=self.mode_window(),
                window_z=self.mode_window_z(),
                stage_name=self.stage_name,
            )
            point_dir = directory / f"bias_{index:02d}"
            point_dir.mkdir(parents=True, exist_ok=True)
            sim = self.staircase_simulation(staircase, output_dir=point_dir)
            sim.mesh(**self.mesh)

            modes = route.solve(
                sim,
                freq_hz=c0 / (self.wavelength_um * 1e-6),
                num_modes=self.num_modes,
                target=self.n_guess,
                order=self.order,
                verbose=verbose,
                stage_name=self.stage_name,
            )
            mode, ratio = self.select_mode(modes, route, at=f"V = {point.bias_v:g}")
            self._check_strip_coverage(route, mode, point.bias_v, staircase.strip_span)
            solved.append((point.bias_v, complex(mode.n_eff), ratio))
        return solved

    def _solve_points(
        self, route: Route, points: Sequence[CarrierResponse], *, directory: Path
    ) -> list[_Solved]:
        """Solve the Mode at *points*, in whichever representation applies.

        Everything that depends on the wavelength — the stack's materials,
        the Strips' unperturbed index, the carrier loss, the frequency
        handed to the Route — is read off this Stage's ``wavelength_um``
        here and below, which is what lets :meth:`group_index` solve
        another wavelength by asking a copy of the Stage set to it.
        """
        if self.effective_n_strips() is None:
            return self._solve_continuous(route, points, directory=directory)
        return self._solve_staircase(route, points, directory=directory)

    def _solve(self) -> OpticalSweep:
        """Solve the Mode at every Bias point, on the selected Route."""
        # Before the charge solve and before meshing: a user whose Route
        # cannot run should pay nothing to find that out.
        route = self.check_route()
        responses = self._bias_sweep()
        if self.n_strips is None and not route.continuous_materials:
            warnings.warn(
                f"The {self.stage_name} stage's {route.name} route cannot carry "
                "a continuous permittivity, so it is solving a staircase "
                f"of {DEFAULT_PALACE_STRIPS} strips instead of the "
                "continuous eps(x, y) a route carrying one would have used. "
                f"Choose the count with study.{self.stage_name}"
                "(n_strips=...).",
                stacklevel=2,
            )
        solved = self._solve_points(
            route,
            responses.points,
            directory=self._require_study().stage_dir(self.stage_name),
        )
        return self._sweep_from(responses.contact, solved)

    # ------------------------------------------------------------------
    # Group index
    # ------------------------------------------------------------------

    def group_index(self) -> float:
        """The Phase shifter's optical group index at ``wavelength_um``.

        ``n_g = n_eff - lambda * d(n_eff)/d(lambda)``: what the Velocity
        mismatch is measured against, and nothing a solve at one
        wavelength produces. The slope is a central difference between
        two more Modes, solved ``group_index_step_um`` either side of the
        Stage's wavelength at the reference Bias point. Each is solved
        with every material resolved again at its own wavelength, so the
        answer carries the materials' dispersion as well as the guide's.

        Runs the Stage first when it holds no result. The two extra Modes
        are solved once and kept on the sweep
        (:attr:`OpticalSweep.group_index`), so they are dropped whenever
        the sweep is.

        Returns:
            The group index.

        Warns:
            UserWarning: When the guide's core resolves to one index at
                both wavelengths, so the answer carries no material
                dispersion.
        """
        sweep: OpticalSweep = self.run()
        if sweep.group_index is None:
            sweep.group_index = self._solve_group_index(sweep)
        return sweep.group_index.n_group

    def _core_index(self) -> float:
        """Index the guide's core is solved with, before the carriers move it."""
        if self.effective_n_strips() is None:
            return self._drawn_core_index()
        return self.unperturbed_index()

    def _solve_group_index(self, sweep: OpticalSweep) -> GroupIndex:
        """Solve the two Modes either side of the wavelength, and difference them.

        Args:
            sweep: This Stage's result, which the centre index is read off.
        """
        route = self.check_route()
        reference = next(
            point
            for point in self._bias_sweep().points
            if point.bias_v == sweep.reference_bias_v
        )
        directory = self._require_study().stage_dir(self.stage_name) / "group_index"

        step = float(self.group_index_step_um)
        wavelengths = (self.wavelength_um - step, self.wavelength_um + step)
        n_eff: list[float] = []
        core: list[float] = []
        for side, wavelength_um in zip(("below", "above"), wavelengths, strict=True):
            # A copy of this Stage set to the other wavelength, attached to
            # the same Study: whatever reads wavelength_um re-resolves there.
            shifted = self.model_copy(update={"wavelength_um": wavelength_um})
            (directory / side).mkdir(parents=True, exist_ok=True)
            ((_, index, _),) = shifted._solve_points(  # noqa: SLF001 - this Stage
                route, [reference], directory=directory / side
            )
            n_eff.append(index.real)
            core.append(shifted._core_index())  # noqa: SLF001 - this Stage

        centre = next(
            point.n_eff.real
            for point in sweep.points
            if point.bias_v == sweep.reference_bias_v
        )
        slope = (n_eff[1] - n_eff[0]) / (2.0 * step)
        record = GroupIndex(
            n_group=centre - self.wavelength_um * slope,
            wavelength_um=self.wavelength_um,
            step_um=step,
            bias_v=reference.bias_v,
            wavelengths_um=wavelengths,
            n_eff=(n_eff[0], n_eff[1]),
            core_index=(core[0], core[1]),
        )
        if not record.material_dispersion:
            self._warn_no_material_dispersion(record)
        return record

    def _warn_no_material_dispersion(self, record: GroupIndex) -> None:
        """Say that the group index is the guide's dispersion alone."""
        if self.strip_index is not None and self.effective_n_strips() is not None:
            cause = (
                f"the strips carry the one strip_index = {self.strip_index:g} "
                "at every wavelength"
            )
            remedy = (
                f"Leave study.{self.stage_name}(strip_index=None) so the strips "
                "read the drawn material at each wavelength"
            )
        else:
            region = self._require_study().layout.junction.regions[0]
            cause = (
                f"the material of the guide's core ({region!r}) resolves to the "
                f"one index {record.core_index[0]:.4f}"
            )
            remedy = (
                "Give that material a dispersion model covering the optical "
                "wavelength in the stack's materials"
            )
        warnings.warn(
            f"The {self.stage_name} stage's group index n_g = "
            f"{record.n_group:.4f} carries no material dispersion: {cause} at "
            f"{record.wavelengths_um[0]:g} and {record.wavelengths_um[1]:g} um, "
            "so only the waveguide's own dispersion is in it, and a silicon "
            "guide's group index comes out low. "
            f"{remedy}, or set a measured value with study.line(n_group=...).",
            stacklevel=4,
        )
