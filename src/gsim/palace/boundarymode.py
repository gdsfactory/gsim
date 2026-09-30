"""Boundary mode simulation class for 2D propagation mode analysis.

A ``BoundaryModeSim`` meshes one cross-section plane of a component and
asks Palace for the Modes that propagate along its normal. It owns the
two things a caller should never have to spell for it: where Palace's
own tables of a run land (:attr:`BoundaryModeSim.run_dir`, cleared
before every local run and read back by :meth:`BoundaryModeSim.read_results`),
and the postprocessing index Palace reports an impedance path under
(:meth:`BoundaryModeSim.add_impedance_path`).

It also owns what a local run that ends badly means. Palace 0.17 corrupts
its heap while shutting a boundary-mode solve down, *after* writing the
answer, so :meth:`BoundaryModeSim.run_local` reads a complete table back
rather than throwing the run away; a binary that died before writing
anything is reported as the runtime failure it is
(:func:`gsim.palace.runtime.local_abort_report`).
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.models import (
    BoundaryModeConfig,
    ContactSpec,
    CrossSectionPlaneConfig,
    InterfaceSpec,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    RefinementConfig,
)
from gsim.palace.models.results import ValidationResult

if TYPE_CHECKING:
    from gsim.common.modes import Extent
    from gsim.palace.mode_fields import BoundaryModeField
    from gsim.palace.results import PalaceTextResults

__all__ = ["BoundaryModeSim", "ModePath"]

#: Where a local Palace run writes its tables, under the simulation's
#: output directory.
RUN_SUBDIR = Path("output") / "palace"


class ModePath(BaseModel):
    """The line integrals Palace evaluates on a solved Mode.

    Postprocessing only: the paths load nothing and leave the 2D
    eigenproblem as it was. Palace integrates ``E`` along the open
    voltage path for the mode voltage (``mode-V.csv``) and, with a
    current loop declared, reports the characteristic impedance
    (``mode-Z.csv``) under this path's index.

    Attributes:
        name: What the caller calls this path.
        voltage_path: Two or more ``(h, v)`` points in the cross-section's
            own coordinates (um), from one conductor's face to the other's.
        current_path: Corners of a closed loop around one conductor, in
            the same coordinates; Palace joins the last corner back to the
            first. ``None`` asks for the voltage alone.
        nsamples: Quadrature order of each line integral.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(min_length=1)
    voltage_path: list[list[float]] = Field(min_length=2)
    current_path: list[list[float]] | None = Field(default=None, min_length=3)
    nsamples: int = Field(default=100, ge=1)

    @field_validator("voltage_path", "current_path")
    @classmethod
    def _points_are_planar(
        cls, value: list[list[float]] | None
    ) -> list[list[float]] | None:
        """Every point is ``(h, v)`` on the cross-section, nothing else."""
        if value is None:
            return None
        cleaned = []
        for point in value:
            if len(point) != 2:
                raise ValueError(
                    "Mode path points are (h, v) cross-section coordinates; "
                    f"got a point with {len(point)} coordinates."
                )
            cleaned.append([float(point[0]), float(point[1])])
        return cleaned


class BoundaryModeSim(PalaceSimMixin, BaseModel):
    """Boundary mode simulation for 2D waveguide cross-section analysis.

    This class configures Palace ``BoundaryMode`` simulations used to compute
    propagation constants and mode profiles on a cross-section plane. It
    takes no 3D excitation ports: what a boundary-mode solve postprocesses
    is declared as a :class:`ModePath` through :meth:`add_impedance_path`.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    simulation_type: Literal["boundarymode"] = "boundarymode"

    # Composed objects
    geometry: Geometry | None = None
    stack: LayerStack | None = None

    # Boundary mode config
    boundary_mode: BoundaryModeConfig = Field(default_factory=BoundaryModeConfig)
    cross_section: CrossSectionPlaneConfig | None = None
    contact_specs: list[ContactSpec] = Field(default_factory=list)
    interface_specs: list[InterfaceSpec] = Field(default_factory=list)
    #: Postprocessing paths, in declaration order: Palace reports the
    #: first under index 1, the second under 2, and so on.
    mode_paths: list[ModePath] = Field(default_factory=list)

    # Unused in boundary mode (kept for mixin compatibility)
    driven: None = None
    eigenmode: None = None

    # Mesh and solver config
    mesh_config: MeshConfig = Field(default_factory=MeshConfig.default)
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    absorbing_boundary: bool = False
    #: Put a perfect-conductor condition on the outer wall of the meshed
    #: domain, making the mode solve a shielded one. Off, Palace's own
    #: default for an unconditioned outer boundary applies, which is PMC.
    metallic_boundaries: bool = False

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _pec_blocks: list = PrivateAttr(default_factory=list)
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)
    _impedance_boundaries: list = PrivateAttr(default_factory=list)
    _airbox_config: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)

    # Cloud job state
    _job_id: str | None = PrivateAttr(default=None)

    def set_boundary_mode(
        self,
        *,
        freq: float = 5e9,
        num_modes: int = 1,
        save: int = 0,
        target: float = 0.0,
        tolerance: float = 1e-6,
        max_size: int = 0,
        solver_type: str = "Default",
    ) -> None:
        """Configure boundary mode solver parameters.

        Args:
            freq: Operating frequency in Hz.
            num_modes: Number of propagation modes to compute.
            save: Number of modes to save to disk.
            target: Target effective index for shift-and-invert.
            tolerance: Relative eigensolver tolerance.
            max_size: Eigensolver max subspace size.
            solver_type: Palace eigensolver type.
        """
        self.boundary_mode = BoundaryModeConfig(
            freq=freq,
            num_modes=num_modes,
            save=save,
            target=target,
            tolerance=tolerance,
            max_size=max_size,
            solver_type=solver_type,
        )

    def set_cross_section(
        self,
        plane: str | CrossSectionPlaneConfig,
        *,
        window: tuple[float, float] | None = None,
        window_z: tuple[float, float] | None = None,
    ) -> None:
        """Set the explicit cross-section plane for 2D mode extraction.

        Args:
            plane: Plane spec as ``"x=<value>"`` or ``"y=<value>"``, or
                a prebuilt CrossSectionPlaneConfig.
            window: Optional in-plane ``(min, max)`` interval in um clipping
                the meshed domain (y for an x-plane, x for a y-plane).
            window_z: Optional vertical ``(min, max)`` interval in um
                clipping the meshed domain.
        """
        if isinstance(plane, CrossSectionPlaneConfig):
            config = plane
        else:
            config = CrossSectionPlaneConfig.from_spec(plane)
        if window is not None or window_z is not None:
            config = CrossSectionPlaneConfig(
                axis=config.axis,
                value=config.value,
                window=window if window is not None else config.window,
                window_z=window_z if window_z is not None else config.window_z,
            )
        self.cross_section = config

    def add_contact(self, *, name: str, layer_a: str, layer_b: str) -> None:
        """Declare a named contact at the interface between two layers.

        The shared curves between the two layers' meshed 2D regions are
        tagged as a dim-1 physical group named *name* during ``mesh()``, so
        charge-transport solvers (DEVSIM ``add_gmsh_contact``) can bind to
        the contact by name.

        Args:
            name: Contact / physical-group name (e.g. ``"anode"``).
            layer_a: First layer of the interface (e.g. the electrode).
            layer_b: Second layer (e.g. the doped semiconductor region).
        """
        self.contact_specs = [
            *self.contact_specs,
            ContactSpec(name=name, layer_a=layer_a, layer_b=layer_b),
        ]

    def add_interface(self, *, name: str, layer_a: str, layer_b: str) -> None:
        """Declare a named interface between two semiconductor layers.

        Tagged on the mesh the way a contact is — the shared curves become
        a dim-1 physical group named *name* — but recorded apart from the
        contacts, because an interface carries continuity and no terminal
        voltage.

        Args:
            name: Interface / physical-group name (e.g. ``"junction"``).
            layer_a: First semiconductor layer.
            layer_b: Second semiconductor layer.
        """
        self.interface_specs = [
            *self.interface_specs,
            InterfaceSpec(name=name, layer_a=layer_a, layer_b=layer_b),
        ]

    # -------------------------------------------------------------------------
    # Postprocessing paths
    # -------------------------------------------------------------------------

    def add_impedance_path(
        self,
        name: str,
        *,
        voltage: list[list[float]],
        current: list[list[float]] | None = None,
        nsamples: int = 100,
    ) -> int:
        """Declare a voltage path and current loop Palace evaluates a Mode on.

        The paths take no part in meshing and load nothing; Palace reports
        the impedance it integrates along them under the index returned
        here, which is their position in declaration order. Declaring a
        name twice replaces the earlier path and keeps its index.

        Args:
            name: What to call the path.
            voltage: ``(h, v)`` points (um) from one conductor's face
                across the gap to the other's.
            current: Corners of a closed loop around the signal conductor
                (um), or ``None`` for the voltage alone.
            nsamples: Quadrature order of each line integral.

        Returns:
            The 1-based index Palace reports this path under.
        """
        path = ModePath(
            name=name, voltage_path=voltage, current_path=current, nsamples=nsamples
        )
        paths = list(self.mode_paths)
        for index, existing in enumerate(paths):
            if existing.name == name:
                paths[index] = path
                self.mode_paths = paths
                return index + 1
        paths.append(path)
        self.mode_paths = paths
        return len(paths)

    def mode_postprocessing(self) -> dict[str, list[dict[str, object]]]:
        """Palace's ``Boundaries.Postprocessing`` block for the declared paths.

        Each path yields an ``Impedance`` entry (``mode-Z.csv``) and a
        ``Voltage`` entry (``mode-V.csv``) under the same index.

        Returns:
            The block, empty when no path is declared.
        """
        impedance_entries: list[dict[str, object]] = []
        voltage_entries: list[dict[str, object]] = []
        for index, path in enumerate(self.mode_paths, start=1):
            entry: dict[str, object] = {
                "Index": index,
                "VoltagePath": [list(point) for point in path.voltage_path],
                "NSamples": path.nsamples,
            }
            if path.current_path is not None:
                entry["CurrentPath"] = [list(point) for point in path.current_path]
            impedance_entries.append(entry)
            voltage_entries.append(
                {
                    "Index": index,
                    "VoltagePath": [list(point) for point in path.voltage_path],
                    "NSamples": path.nsamples,
                }
            )
        result: dict[str, list[dict[str, object]]] = {}
        if impedance_entries:
            result["Impedance"] = impedance_entries
            result["Voltage"] = voltage_entries
        return result

    # -------------------------------------------------------------------------
    # 3D ports have no meaning on a cross-section
    # -------------------------------------------------------------------------

    @staticmethod
    def _no_ports(kind: str) -> ValueError:
        """The refusal a 3D port declaration gets, naming the way in."""
        return ValueError(
            f"A boundary-mode simulation takes no {kind}: it meshes a "
            "cross_section-only native 2D domain and solves the Modes on it. "
            "Declare what Palace should postprocess with "
            "add_impedance_path(name, voltage=..., current=...)."
        )

    def add_port(self, *_args: Any, **_kwargs: Any) -> None:
        """A lumped port is a 3D excitation; refused here."""
        raise self._no_ports("lumped ports")

    def add_cpw_port(self, *_args: Any, **_kwargs: Any) -> None:
        """A CPW port is a 3D excitation; refused here."""
        raise self._no_ports("CPW ports")

    def add_wave_port(self, *_args: Any, **_kwargs: Any) -> None:
        """A wave port is a 3D excitation; refused here."""
        raise self._no_ports("wave ports")

    def add_terminal(self, *_args: Any, **_kwargs: Any) -> None:
        """A terminal belongs to an electrostatic solve; refused here."""
        raise self._no_ports("terminals")

    # -------------------------------------------------------------------------
    # This run's results
    # -------------------------------------------------------------------------

    @property
    def mesh_extent(self) -> Extent:
        """The rectangle this simulation's 2D mesh covers.

        Read off the mesh itself rather than off the geometry that asked
        for it, so it is what the solver will actually see — which is
        what a postprocessing path sampled on the domain has to stay
        inside.

        Returns:
            ``((h_min, h_max), (v_min, v_max))`` in the mesh's own
            cross-section coordinates (um).

        Raises:
            ValueError: When the simulation has not been meshed.
        """
        import meshio

        points = meshio.read(str(self.mesh_path)).points
        h = points[:, 0]
        v = points[:, 1]
        return ((float(h.min()), float(h.max())), (float(v.min()), float(v.max())))

    @property
    def run_dir(self) -> Path:
        """Where a local run's Palace tables land.

        Raises:
            ValueError: When no output directory is set.
        """
        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")
        return Path(self._output_dir) / RUN_SUBDIR

    @property
    def last_run_files(self) -> dict[str, Path]:
        """Every file the last local run left, by name; empty when none."""
        run_dir = self.run_dir
        if not run_dir.is_dir():
            return {}
        return {
            path.name: path
            for path in run_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        }

    def read_results(self) -> PalaceTextResults | None:
        """The last local run's text results, off disk.

        Only this run's directory is read — it is cleared before every
        run — so a table left anywhere else cannot stand in for it.

        Returns:
            The parsed results, or ``None`` when the run left nothing
            parseable.
        """
        from gsim.palace.results import load_text_results

        files = self.last_run_files
        if not files:
            return None
        try:
            return load_text_results(files)
        except FileNotFoundError:
            return None

    def read_mode_field(self, mode_id: int) -> BoundaryModeField:
        """The fields the last local run saved for one Mode.

        Args:
            mode_id: Palace's own mode number, 1-based.

        Returns:
            The Mode's fields on the meshed cross-section.

        Raises:
            FileNotFoundError: When the run saved no fields.
            ValueError: When that Mode was not among the saved ones.
        """
        from gsim.palace.mode_fields import load_boundary_mode_field

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")
        return load_boundary_mode_field(Path(self._output_dir), mode_id=mode_id)

    def _salvage_mode_table(self, err: Exception) -> PalaceTextResults | None:
        """Read a crashed run's mode table back, if it is complete.

        Palace 0.17 intermittently corrupts its heap while shutting down
        a ``BoundaryMode`` solve (``free(): corrupted unsorted chunks``),
        after the solve itself has finished and its results are on disk.
        The crash is non-deterministic on an identical mesh and config,
        so a run that exits abnormally may still have answered the
        question it was asked — and the answer on disk is used rather
        than thrown away, which is what lets a gate depend on a live
        solve at all. The run directory is cleared before each run, so
        what is read back is this run's.

        Args:
            err: What the run raised, named in the warning.

        Returns:
            The parsed text results when the table is complete, else
            ``None`` so the caller re-raises.
        """
        try:
            text = self.read_results()
        except Exception:
            return None
        num_modes = self.boundary_mode.num_modes
        if text is None or len(getattr(text, "modes", {})) < num_modes:
            return None
        warnings.warn(
            f"Palace exited abnormally at f = {self.boundary_mode.freq:g} Hz but "
            f"its complete mode table was on disk, so the run's answer is used "
            f"({err}). Palace 0.17 is known to corrupt its heap on shutdown of "
            "a boundary-mode solve.",
            stacklevel=3,
        )
        return text

    def _run_context(self) -> str:
        """A boundary-mode solve is named by the frequency it ran at."""
        return f"at f = {self.boundary_mode.freq:g} Hz"

    def run_local(
        self, *, salvage: bool = True, remedy: str | None = None, **kwargs: Any
    ) -> PalaceTextResults:
        """Run Palace locally and hand back this run's text results.

        The run directory is cleared first, so a table an earlier run left
        cannot shadow this one's, and what :meth:`read_results` returns
        afterwards — on success or after a crash — is this run's.

        A run that ends abnormally is asked whether it answered anyway
        before it is reported as a failure, because on Palace 0.17 it
        often did (:meth:`_salvage_mode_table`). One that did not, and
        whose binary died rather than exited, is reported as the runtime
        failure it is rather than as a raw exit status.

        Args:
            salvage: Take a complete mode table left by an abnormal exit
                as the run's answer, warning that it happened. ``False``
                re-raises instead, which is also what a caller running
                with warnings as errors gets.
            remedy: A second way out, named at the end of the abort
                report beyond pointing ``PALACE_BIN`` somewhere else.
            **kwargs: What :meth:`gsim.palace.base.PalaceSimMixin.run_local`
                takes (``palace_executable``, ``verbose``, ...).

        Returns:
            The parsed mode tables.

        Raises:
            RuntimeError: When Palace ran but left no parseable results,
                or when its binary aborted with nothing to salvage.

        Warns:
            UserWarning: When an abnormal exit's mode table is salvaged.
        """
        shutil.rmtree(self.run_dir, ignore_errors=True)
        try:
            super().run_local(remedy=remedy, **kwargs)
        except (RuntimeError, subprocess.CalledProcessError) as err:
            # The mixin reports an aborted binary; what is asked here
            # first is whether this solve answered before it died.
            salvaged = self._salvage_mode_table(err) if salvage else None
            if salvaged is None:
                raise
            return salvaged
        results = self.read_results()
        if results is None:
            raise RuntimeError(f"Palace produced no text results under {self.run_dir}.")
        return results

    def validate_config(self) -> ValidationResult:
        """Validate boundary mode simulation configuration."""
        base_result = super().validate_config()
        errors = list(base_result.errors)
        warnings = list(base_result.warnings)

        if self.cross_section is None:
            errors.append(
                "Boundary mode requires an explicit cross-section plane. "
                "Call set_cross_section('x=<value>') or set_cross_section('y=<value>')."
            )
        elif self.cross_section.axis == "z":
            errors.append(
                "Boundary mode native 2D currently supports only x/y cross sections. "
                "Use set_cross_section('x=<value>') or set_cross_section('y=<value>')."
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
