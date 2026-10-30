"""Magnetostatic simulation class for inductance extraction.

This module provides the MagnetostaticSim class for extracting the
inductance matrix between current sources.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from gsim.palace.base import _PalaceSimBase
from gsim.palace.inductance import InductanceMatrix, load_inductance
from gsim.palace.models import (
    CurrentSourceConfig,
    MagnetostaticConfig,
    MagnetostaticSolverConfig,
    ValidationResult,
    WavePortConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

    from gsim.common.stack import LayerStack
    from gsim.palace.current_sources import ResolvedCurrentSource
    from gsim.palace.mesh.geometry import GeometryData
    from gsim.palace.mesh.nets import Net, Nets
    from gsim.palace.ports.config import PalacePort
    from gsim.palace.results import PalaceTextResults, SParams

logger = logging.getLogger(__name__)


class MagnetostaticSim(_PalaceSimBase):
    """Magnetostatic simulation for inductance matrix extraction.

    Each current source drives a unit current around one conductor loop, and
    Palace returns the inductance matrix between the sources from the stored
    magnetic energy. Conductors are perfect conductors, so this is the
    external inductance; the drawn metal that joins a source's two ports is
    its return path.

    The outer wall of the domain is a perfect electric conductor
    (``metallic_boundaries=True``), which pushes the field in and gives a
    lower bound on each self inductance. ``metallic_boundaries=False`` leaves
    it a perfect magnetic conductor, an upper bound. For mutual inductances
    the two runs are a first-order bracket, not a bound; the mean of the two
    runs cancels the first-order wall error. Keep the walls far from the coils,
    each airbox extent at least the larger side of the meshed conductors:
    ``set_airbox(margin_x=100, margin_y=100, z_above=150, z_below=150)`` suits
    coils some tens of um across.

    Mesh sizes come from ``refined_mesh_size`` at the conductors and the
    source sheets and ``max_mesh_size`` in the air: use ``auto_size=True``, or
    a ``refined_mesh_size`` of at most half the trace width, with a coarse
    ``max_mesh_size`` in the large airbox. ``fmax`` has no effect here.

    Example:
        >>> from gsim.palace import MagnetostaticSim
        >>>
        >>> sim = MagnetostaticSim()
        >>> sim.set_geometry(transformer)  # ports P1/P2, S1/S2
        >>> sim.set_stack()
        >>> sim.set_airbox(margin_x=100, margin_y=100, z_above=150, z_below=150)
        >>> sim.add_current_source("primary", plus_port="P1", minus_port="P2")
        >>> sim.add_current_source("secondary", plus_port="S1", minus_port="S2")
        >>> sim.set_output_dir("./sim")
        >>> sim.mesh(preset="default")
        >>> inductance = sim.load_inductance(sim.run())
        >>> inductance.mutual_inductance("primary", "secondary")  # in H

    Attributes:
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        current_sources: The current sources, in matrix row order
        metallic_boundaries: PEC outer wall when True, PMC when False
        materials: Material property overrides
        solver: Grouped numerical and magnetostatic solver configuration
    """

    simulation_type: Literal["magnetostatic"] = "magnetostatic"

    ports: None = None
    cpw_ports: None = None
    two_terminal_ports: None = None
    wave_ports: list[WavePortConfig] = Field(default_factory=list)

    #: The current sources, in the order they were added: the order of the
    #: rows of the inductance matrix.
    current_sources: list[CurrentSourceConfig] = Field(default_factory=list)

    #: Palace ignores an absorbing wall in a magnetostatic solve; refused.
    absorbing_boundary: bool = False
    #: Put a perfect electric conductor on the outer wall of the domain.
    #: Off, Palace's default for an unconditioned outer boundary applies,
    #: which is a perfect magnetic conductor. The two bracket the open-space
    #: inductance.
    metallic_boundaries: bool = True

    solver: MagnetostaticSolverConfig = Field(default_factory=MagnetostaticSolverConfig)

    # -------------------------------------------------------------------------
    # Current sources
    # -------------------------------------------------------------------------

    def add_current_source(
        self,
        name: str,
        *,
        plus_port: str,
        minus_port: str,
        layer: str | None = None,
    ) -> None:
        """Add a current source between two ports on one conductor.

        gsim bridges the gap between the two port faces with a vertical sheet
        through the thickness of the conductor layer. The ports either face
        each other across the gap, or are side-by-side leads that point the
        same way. A unit current enters the conductor at ``plus_port`` and
        leaves it at ``minus_port`` (passive sign convention), so swapping the
        two flips the sign of the source's row and column of the inductance
        matrix. The drawn metal that joins the two ports is the return path.

        Re-adding a name replaces that source and moves it to the end.

        Args:
            name: Source name; it labels the row of the inductance matrix.
            plus_port: Name of the component port where the current enters.
            minus_port: Name of the component port where the current leaves.
            layer: Conductor layer of the sheet; by default inferred from the
                GDS layer of the ports.

        Example:
            >>> sim.add_current_source("L1", plus_port="P1", minus_port="P2")
        """
        self.current_sources = [
            source for source in self.current_sources if source.name != name
        ]
        self.current_sources.append(
            CurrentSourceConfig(
                name=name, plus_port=plus_port, minus_port=minus_port, layer=layer
            )
        )

    def _resolve_with_nets(
        self, stack: LayerStack, geometry: GeometryData | None = None
    ) -> tuple[list[ResolvedCurrentSource], Nets]:
        """The resolved sources, and the nets they were resolved on.

        The nets are built from all component ports, whatever their type, so
        that a source on optical ports finds its net too.

        Args:
            stack: The resolved layer stack.
            geometry: The component's geometry extracted on ``stack``, when
                the caller already has it; extracted here otherwise.

        Raises:
            ValueError: If no geometry has been set, or a source breaks a
                rule of :func:`~gsim.palace.current_sources.resolve_current_sources`.
        """
        from gsim.palace.current_sources import resolve_current_sources
        from gsim.palace.mesh.geometry import extract_geometry
        from gsim.palace.mesh.nets import extract_nets

        component = self.component
        if component is None:
            raise ValueError("No component set. Call set_geometry(component) first.")
        if geometry is None:
            geometry = extract_geometry(component, stack)
        nets = extract_nets(geometry, stack, list(component.ports))
        resolved = resolve_current_sources(self.current_sources, component, stack, nets)
        return resolved, nets

    def resolve_current_sources(self) -> list[ResolvedCurrentSource]:
        """Show the sheet, Direction and net of each source, without meshing.

        ``validate_config()`` and ``mesh()`` run the same check.

        Returns:
            The sources in the order they were added, each with the centre,
            span and Palace ``Direction`` of its sheet and the net that closes
            its loop.

        Raises:
            ValueError: If no geometry has been set, there is no source, a
                port is missing, the ports break the bridge or layer rule, a
                loop is open, a sheet crosses metal, or two sources share a
                name, a port pair or a sheet.

        Example:
            >>> for source in sim.resolve_current_sources():
            ...     print(source.name, source.direction, source.net)
        """
        resolved, _ = self._resolve_with_nets(self._resolve_stack())
        return resolved

    def _palace_ports(self, stack: LayerStack) -> list[PalacePort]:
        """The source sheets, in source order.

        Built from the current sources alone: the ``info`` of the gdsfactory
        ports is neither read nor written, so ports that another simulation
        configured on the same component do not enter this one.
        """
        resolved, _ = self._resolve_with_nets(stack)
        return [source.to_palace_port(stack) for source in resolved]

    # -------------------------------------------------------------------------
    # Ports, terminals and impedances have no meaning here
    # -------------------------------------------------------------------------

    @staticmethod
    def _no_ports(kind: str) -> ValueError:
        """The refusal a port-like declaration gets, naming the way in."""
        return ValueError(
            f"A magnetostatic simulation takes no {kind}: it is driven by "
            "current sources. Declare one per conductor loop with "
            "add_current_source(name, plus_port=..., minus_port=...)."
        )

    def add_port(self, *_args: Any, **_kwargs: Any) -> None:
        """A lumped port belongs to a driven or eigenmode solve; refused here."""
        raise self._no_ports("lumped ports")

    def add_cpw_port(self, *_args: Any, **_kwargs: Any) -> None:
        """A CPW port belongs to a driven or eigenmode solve; refused here."""
        raise self._no_ports("CPW ports")

    def add_wave_port(self, *_args: Any, **_kwargs: Any) -> None:
        """A wave port belongs to a driven or eigenmode solve; refused here."""
        raise self._no_ports("wave ports")

    def add_two_terminal_port(self, *_args: Any, **_kwargs: Any) -> None:
        """A two-terminal port belongs to a driven solve; refused here."""
        raise self._no_ports("two-terminal ports")

    def add_terminal(self, *_args: Any, **_kwargs: Any) -> None:
        """A terminal belongs to an electrostatic solve; refused here."""
        raise self._no_ports("terminals")

    def add_impedance_boundary(self, *_args: Any, **_kwargs: Any) -> None:
        """Palace ignores impedance boundaries in a magnetostatic solve; refused."""
        raise ValueError(
            "Impedance boundaries do not apply to a magnetostatic simulation: "
            "Palace ignores them in a magnetostatic solve."
        )

    # -------------------------------------------------------------------------
    # Magnetostatic configuration
    # -------------------------------------------------------------------------

    def set_magnetostatic(self, *, save_fields: int = 0) -> None:
        """Configure the magnetostatic solve.

        Args:
            save_fields: Number of source solutions whose fields are saved.

        Example:
            >>> sim.set_magnetostatic(save_fields=1)
        """
        self.solver.magnetostatic = MagnetostaticConfig(save_fields=save_fields)

    def validate_config(self) -> ValidationResult:
        """Validate the magnetostatic simulation configuration.

        Errors: no current source, a source that does not resolve (see
        :meth:`resolve_current_sources`), an absorbing outer wall, a direct
        linear solver (``solver_type`` other than ``"Default"``), or the
        BoomerAMG preconditioner.

        Warnings: no airbox, or an airbox extent smaller than the larger side
        of the meshed conductors (the layout on the stack's layers, which
        ``mesh()`` pads); and conductors that carry no source, which screen
        any loop they close.

        Returns:
            ValidationResult with validation status and messages
        """
        from gsim.palace.mesh.geometry import extract_geometry

        base = super().validate_config()
        errors = list(base.errors)
        warnings = list(base.warnings)

        # The geometry is extracted once, for the sources, the screening
        # warning and the airbox warning.
        component = self.component
        extracted: tuple[LayerStack, GeometryData] | ValueError | OSError | None = None
        if component is not None:
            try:
                stack = self._resolve_stack()
                extracted = (stack, extract_geometry(component, stack))
            except (ValueError, OSError) as error:
                extracted = error

        if not self.current_sources:
            errors.append(
                "No current source. Add one per conductor loop with "
                "add_current_source(name, plus_port=..., minus_port=...)."
            )
        elif isinstance(extracted, OSError):
            raise extracted
        elif isinstance(extracted, ValueError):
            errors.append(str(extracted))
        elif extracted is not None:
            stack, geometry = extracted
            try:
                resolved, nets = self._resolve_with_nets(stack, geometry)
            except ValueError as error:
                errors.append(str(error))
            else:
                driven = {source.net for source in resolved}
                screening = self._screening_names(
                    [net for net in nets if net.name not in driven], geometry, stack
                )
                if screening:
                    warnings.append(
                        f"Conductors {screening} carry no current source. They "
                        "are perfect conductors that screen any loop they "
                        "close (ground rings, shields); remove them from the "
                        "component if they should not screen."
                    )

        if self.absorbing_boundary:
            errors.append(
                "absorbing_boundary is not supported: Palace ignores an "
                "Absorbing wall in a magnetostatic solve and leaves it PMC. "
                "Choose the outer wall with metallic_boundaries (True: PEC, "
                "False: PMC)."
            )
        linear = self.solver.linear
        if linear.solver_type != "Default":
            errors.append(
                f"solver.linear.solver_type={linear.solver_type!r} is not "
                "supported: the magnetostatic curl-curl operator is singular, "
                "so Palace solves it iteratively. Use 'Default' (AMS + CG)."
            )
        if linear.preconditioner == "BoomerAMG":
            errors.append(
                "solver.linear.preconditioner='BoomerAMG' is not supported for "
                "the magnetostatic curl-curl operator. Use 'AMS' (or 'Default')."
            )

        geometry = extracted[1] if isinstance(extracted, tuple) else None
        warnings.extend(self._airbox_warnings(geometry))
        return ValidationResult(valid=not errors, errors=errors, warnings=warnings)

    def _screening_names(
        self, screening: list[Net], geometry: GeometryData, stack: LayerStack
    ) -> list[str]:
        """Name the nets that carry no source as :meth:`nets` names them.

        The nets the sources resolve on also take non-electrical ports, which
        only changes their names; a net is matched by its shapes, and keeps
        its all-port name when :meth:`nets` has no net with the same shapes.
        """
        named = {
            net.members: net.name for net in self._electrical_nets(geometry, stack)
        }
        return [named.get(net.members, net.name) for net in screening]

    def _airbox_warnings(self, geometry: GeometryData | None) -> list[str]:
        """Warn when the outer wall is close enough to bias the inductance."""
        recipe = "set_airbox(margin_x=100, margin_y=100, z_above=150, z_below=150)"
        airbox = self._airbox_config
        if not airbox:
            return [
                "No airbox: the outer wall sits on the stack and biases the "
                f"inductance. Add room around the coils, e.g. {recipe}."
            ]
        size = self._conductor_size(geometry)
        if size is None:
            return []
        extents = {
            key: float(airbox.get(key, 0.0))
            for key in ("margin_x", "margin_y", "z_above", "z_below")
        }
        small = {key: value for key, value in extents.items() if value < size}
        if not small:
            return []
        return [
            f"Airbox extents {small} um are smaller than the meshed conductors "
            f"({size:.4g} um across): the outer wall biases the inductance. "
            f"Make them at least that large, e.g. {recipe} for coils some tens "
            "of um across."
        ]

    def _conductor_size(self, geometry: GeometryData | None) -> float | None:
        """The larger lateral side of the meshed conductors, in um.

        This is the box that ``mesh()`` pads with the airbox margins, so
        marker layers outside the stack do not count. Falls back on the
        component bounding box when the geometry could not be extracted
        (``geometry`` is None) or holds none of the stack's layers; ``None``
        without a component.
        """
        component = self.component
        if component is None:
            return None
        if geometry is not None:
            xmin, ymin, xmax, ymax = geometry.bbox
        else:
            xmin, ymin, xmax, ymax = (math.inf, math.inf, -math.inf, -math.inf)
        if xmax >= xmin and ymax >= ymin:
            return max(xmax - xmin, ymax - ymin)
        return max(float(component.xsize), float(component.ysize))

    # -------------------------------------------------------------------------
    # Run and results
    # -------------------------------------------------------------------------

    def run_local(
        self,
        *,
        palace_sif_path: str | Path | None = None,
        palace_executable: str | Path | None = None,
        use_apptainer: bool = False,
        num_processes: int | None = None,
        num_threads: int | None = None,
        verbose: bool = True,
        remedy: str | None = None,
        log_path: str | Path | None = None,
    ) -> SParams | PalaceTextResults | dict[str, Path]:
        """Run Palace locally and keep its log with the results.

        A magnetostatic solve that does not converge still exits 0 and
        writes a plausible inductance matrix; only Palace's log says so. The
        log is therefore kept as ``palace.log`` next to the result tables by
        default, where :meth:`load_inductance` checks it.

        Args:
            palace_sif_path: Palace Apptainer SIF file; see
                :meth:`gsim.palace.base.PalaceSimMixin.run_local`.
            palace_executable: Palace executable; overrides ``use_apptainer``.
            use_apptainer: Run Palace through Apptainer.
            num_processes: Number of MPI processes; a size-aware default when
                None.
            num_threads: Number of OpenMP threads.
            verbose: Print progress and stream Palace's output.
            remedy: A second way out of a dead Palace runtime, named in the
                abort report.
            log_path: File for Palace's output. Defaults to ``palace.log`` in
                the Palace output directory. A file elsewhere is not among the
                returned results, so :meth:`load_inductance` then warns that
                convergence was not checked.

        Returns:
            The parsed text results, as ``PalaceSimMixin.run_local`` returns
            them; pass them to :meth:`load_inductance`.

        Example:
            >>> inductance = sim.load_inductance(sim.run_local())
        """
        if log_path is None:
            log_path = self._postpro_dir() / "palace.log"
        return super().run_local(
            palace_sif_path=palace_sif_path,
            palace_executable=palace_executable,
            use_apptainer=use_apptainer,
            num_processes=num_processes,
            num_threads=num_threads,
            verbose=verbose,
            remedy=remedy,
            log_path=log_path,
        )

    def load_inductance(
        self,
        results: dict[str, Path] | PalaceTextResults | str | Path,
        *,
        check_convergence: bool = True,
    ) -> InductanceMatrix:
        """Load the inductance matrix of a run, labelled with the source names.

        The rows follow the order the sources were added in, which is
        Palace's source order. A unit current enters at each source's
        ``plus_port`` and leaves at its ``minus_port``.

        Args:
            results: The results dict that ``run()`` returns, the
                ``PalaceTextResults`` that ``run_local()`` returns, or the
                output directory of the simulation.
            check_convergence: Read the run's ``palace.log`` and refuse the
                matrix if a solve did not converge; see
                :func:`~gsim.palace.inductance.load_inductance`.

        Returns:
            The inductance matrix, with checks; see
            :class:`~gsim.palace.inductance.InductanceMatrix`.

        Raises:
            RuntimeError: If ``palace.log`` shows that a solve did not
                converge.

        Example:
            >>> inductance = sim.load_inductance(sim.run())
            >>> inductance.self_inductance("primary")  # in H
            >>> inductance.coupling("primary", "secondary")
            >>> assert not inductance.problems()
        """
        return load_inductance(
            results,
            source_names=[source.name for source in self.current_sources],
            check_convergence=check_convergence,
        )


__all__ = ["MagnetostaticSim"]
