"""What every Palace simulation does the same way.

:class:`PalaceSimMixin` provides the methods shared across the simulation
types — DrivenSim, EigenmodeSim, ElectrostaticSim and BoundaryModeSim:
geometry and stack configuration, meshing, config generation, and the
local run, including what a run that dies on a signal means.
:class:`_PalaceSimBase` declares, once, the state those methods share.

:class:`MeshSourceMixin` is the same surface for a simulation that does
not mesh for itself but delegates to one that does.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.common.cloud import CloudJobMixin
from gsim.palace.models import (
    CPWPortConfig,
    ImpedanceBoundaryConfig,
    MaterialConfig,
    MeshConfig,
    PortConfig,
    RefinementConfig,
    SymmetryPlaneConfig,
    TerminalConfig,
    TwoTerminalPortConfig,
    WavePortConfig,
)
from gsim.palace.models.results import (
    SimulationResult,
    ValidationResult,
    format_field_dofs,
    format_mesh_distortion,
    mesh_identity_lines,
)
from gsim.palace.solver_settings import SolverSettingsMixin

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gdsfactory.component import Component

    from gsim.palace.results import PalaceTextResults, SParams

logger = logging.getLogger(__name__)


def _count_physical_cpus() -> int:
    """Return the number of physical CPU cores (not logical threads)."""
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
        # Each physical core has a unique (physical_id, core_id) pair.
        cores: set[tuple[str, str]] = set()
        phys_id = None
        for line in content.splitlines():
            if line.startswith("physical id"):
                phys_id = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":")[1].strip()
                if phys_id is not None:
                    cores.add((phys_id, core_id))
        if cores:
            return len(cores)
    except Exception:
        pass
    return os.cpu_count() or 1


# Minimum estimated unknowns per MPI rank for SuperLU_DIST to factor
# efficiently. Below this, the 2D block-cyclic processor grid is dominated by
# communication/fill overhead and the factorization degrades dramatically
# (an effective hang on small 2D meshes at 8-16 ranks).
_MIN_DOFS_PER_RANK = 50_000
# Absolute ceiling on MPI ranks for 3D problems that use a sparse direct
# solver. 2D (BoundaryMode) problems never use MPI: see _recommend_parallel.
_MAX_DIRECT_SOLVE_RANKS = 4


def _estimate_dofs(mesh_stats: dict) -> int | None:
    """Estimate total unknowns from mesh statistics.

    Uses the same element-count * polynomial-order heuristic as
    :meth:`PalaceSimMixin.print_mesh_stats` (order 2 for 2D, 1 for 3D).
    """
    if not mesh_stats:
        return None
    elements = mesh_stats.get("elements") or 0
    if not elements:
        return None
    is_3d = bool(mesh_stats.get("tetrahedra"))
    p = 1 if is_3d else 2
    nd_dofs = elements * p * (p + 1)
    h1_dofs = elements * (p + 1) * (p + 2) // 2
    return nd_dofs + h1_dofs


def _recommend_parallel(
    mesh_stats: dict,
    simulation_type: str,
    num_processes: int | None,
    num_threads: int | None,
    physical_cpus: int | None = None,
) -> tuple[int, int | None]:
    """Recommend MPI/OpenMP settings that avoid pathological direct solves.

    SuperLU_DIST factors the MPI rank count into a 2D processor grid. For
    small problems (especially 2D mode analysis), using every physical core
    produces a grid (e.g. 4x4 at 16 ranks) whose factorization is
    communication-bound and effectively hangs. This returns a safe default:

    - 2D (``boundarymode`` or a planar mesh): a single MPI rank with OpenMP
      threads. MPI for a 2D problem is overkill and the sparse-direct grid
      does not scale.
    - 3D: at most ``_MAX_DIRECT_SOLVE_RANKS`` ranks, and never more than
      ``est_dofs // _MIN_DOFS_PER_RANK`` so each rank has enough unknowns.

    Explicit caller-supplied values are always respected; when they exceed
    the recommendation the caller should log a warning.

    Returns:
        ``(num_processes, num_threads)``. ``num_threads`` stays ``None``
        unless it should be auto-defaulted to the physical core count.
    """
    if physical_cpus is None:
        physical_cpus = _count_physical_cpus()

    if num_processes is not None:
        # Explicit request: keep it, default threads only for serial runs.
        if num_processes == 1 and num_threads is None:
            return num_processes, physical_cpus
        return num_processes, num_threads

    is_2d = simulation_type == "boundarymode"
    if not is_2d and mesh_stats:
        bbox = mesh_stats.get("bbox") or {}
        dz = bbox.get("zmax", bbox.get("zmin", 0.0)) - bbox.get(
            "zmin", bbox.get("zmax", 0.0)
        )
        if not mesh_stats.get("tetrahedra") and dz <= 1e-6:
            is_2d = True

    if is_2d:
        return 1, (num_threads if num_threads is not None else physical_cpus)

    # 3D: size-aware cap for sparse-direct solves.
    est_dofs = _estimate_dofs(mesh_stats)
    size_cap = physical_cpus
    if est_dofs is not None:
        size_cap = max(1, est_dofs // _MIN_DOFS_PER_RANK)
    recommended = min(physical_cpus, _MAX_DIRECT_SOLVE_RANKS, size_cap)
    return max(1, recommended), num_threads


def _wrote_solver_output(postpro_dir: Path) -> bool:
    """Whether a local run left any solver output in Palace's output dir.

    What separates a dead Palace runtime from a dead solve: a runtime
    that cannot start kills the binary before the solver writes
    anything.
    """
    if not postpro_dir.is_dir():
        return False
    return any(
        file.is_file() and not file.name.startswith(".")
        for file in postpro_dir.iterdir()
    )


class MeshSourceMixin:
    """The configuration and mesh reads of whoever holds the mesh.

    A simulation that does not mesh for itself still has to be
    configured like one and asked the same questions afterwards: give it
    a geometry, a stack, a Cross-section and its contacts, then ask
    where the mesh landed and what groups it carries. A charge-transport
    sim is that case — it delegates every one of them to a
    :class:`~gsim.palace.BoundaryModeSim` it owns — and hand-mirroring
    the surface is how the two drift apart.

    Subclasses name the simulation that actually holds the mesh through
    :meth:`_mesh_source`, and say whether it exists yet through
    :meth:`_mesh_source_or_none`: the setters build it, the readers
    answer for an unconfigured simulation rather than building one to
    ask. The guard names the concrete class, so a user reads the sim
    they called, not the delegate they never saw.
    """

    def _mesh_source(self) -> Any:
        """The simulation that owns the mesh, built if it does not exist."""
        raise NotImplementedError

    def _mesh_source_or_none(self) -> Any | None:
        """The simulation that owns the mesh, or ``None`` if never built."""
        raise NotImplementedError

    def _meshed_mesh_source(self) -> Any:
        """The mesh source, or the report that nothing has meshed yet."""
        source = self._mesh_source_or_none()
        if source is None or not source.has_mesh:
            raise ValueError(
                f"No mesh generated for this {type(self).__name__}. Call mesh() first."
            )
        return source

    # -------------------------------------------------------------------------
    # Delegated configuration
    # -------------------------------------------------------------------------

    def set_geometry(self, component: Any) -> None:
        """Set the gdsfactory component (delegates to the mesh pipeline)."""
        self._mesh_source().set_geometry(component)

    def set_stack(self, *args: Any, **kwargs: Any) -> None:
        """Configure the layer stack (same arguments as the palace backends)."""
        self._mesh_source().set_stack(*args, **kwargs)

    def set_airbox(self, **kwargs: Any) -> None:
        """Configure the background region around the clipped domain."""
        self._mesh_source().set_airbox(**kwargs)

    def set_output_dir(self, path: str | Path) -> None:
        """Set the output directory for mesh files."""
        self._mesh_source().set_output_dir(path)

    def set_cross_section(
        self,
        plane: Any,
        *,
        window: tuple[float, float] | None = None,
        window_z: tuple[float, float] | None = None,
    ) -> None:
        """Set the cross-section plane and the window clipping the mesh.

        Args:
            plane: Plane spec (``"x=<value>"`` / ``"y=<value>"``) or a
                prebuilt ``CrossSectionPlaneConfig``.
            window: In-plane ``(min, max)`` clip in um.
            window_z: Vertical ``(min, max)`` clip in um.
        """
        self._mesh_source().set_cross_section(plane, window=window, window_z=window_z)

    def add_contact(self, *, name: str, layer_a: str, layer_b: str) -> None:
        """Declare a named contact at the interface between two layers."""
        self._mesh_source().add_contact(name=name, layer_a=layer_a, layer_b=layer_b)

    def add_interface(self, *, name: str, layer_a: str, layer_b: str) -> None:
        """Declare a named interface between two semiconductor layers.

        An interface is never a contact: it carries continuity across the
        junction and no terminal voltage, and the mesh keeps the two
        apart.

        Args:
            name: Interface / physical-group name (e.g. ``"junction"``).
            layer_a: First doped region layer name.
            layer_b: Second doped region layer name.
        """
        self._mesh_source().add_interface(name=name, layer_a=layer_a, layer_b=layer_b)

    # -------------------------------------------------------------------------
    # Delegated reads
    # -------------------------------------------------------------------------

    @property
    def contact_specs(self) -> list[Any]:
        """Declared contacts — the ohmic terminals, and nothing else."""
        source = self._mesh_source_or_none()
        return list(source.contact_specs) if source is not None else []

    @property
    def interface_specs(self) -> list[Any]:
        """Declared semiconductor-semiconductor interfaces."""
        source = self._mesh_source_or_none()
        return list(source.interface_specs) if source is not None else []

    @property
    def output_dir(self) -> Path | None:
        """Output directory (or None)."""
        source = self._mesh_source_or_none()
        return source.output_dir if source is not None else None

    @property
    def has_mesh(self) -> bool:
        """Whether ``mesh()`` has run and left a mesh to read."""
        source = self._mesh_source_or_none()
        return source is not None and bool(source.has_mesh)

    @property
    def mesh_path(self) -> Path:
        """Path of the generated mesh (um), once meshed.

        Raises:
            ValueError: When read before ``mesh()`` has run.
        """
        return Path(self._meshed_mesh_source().mesh_path)

    @property
    def mesh_groups(self) -> dict[str, Any]:
        """Physical groups of the generated mesh, once meshed.

        Raises:
            ValueError: When read before ``mesh()`` has run.
        """
        return dict(self._meshed_mesh_source().mesh_groups)


class PalaceSimMixin(CloudJobMixin, SolverSettingsMixin):
    """Mixin providing common methods for all Palace simulation classes.

    The state these methods use is declared once, by :class:`_PalaceSimBase`;
    the hints below only tell the type checker about it.
    """

    if TYPE_CHECKING:
        geometry: Geometry | None
        stack: LayerStack | None
        materials: dict[str, MaterialConfig]
        refinement: RefinementConfig
        mesh_config: MeshConfig
        simulation_type: Literal["driven", "eigenmode", "electrostatic", "boundarymode"]
        absorbing_boundary: bool | None
        absorbing_order: Literal[1, 2]
        _pec_default_noted: bool
        _output_dir: Path | None
        _stack_kwargs: dict[str, Any]
        _airbox_config: dict[str, Any]
        _pec_blocks: list
        _symmetry_planes: list[SymmetryPlaneConfig]
        _hints: dict[str, Any]
        _impedance_boundaries: list[ImpedanceBoundaryConfig]
        _configured_ports: bool
        _last_mesh_result: Any
        _last_ports: list
        # Declared by the sim classes that take 3D ports or terminals; a
        # boundary-mode sim has none, so these must not become fields.
        ports: list[PortConfig]
        cpw_ports: list[CPWPortConfig]
        wave_ports: list[WavePortConfig]
        two_terminal_ports: list[TwoTerminalPortConfig]
        terminals: list[TerminalConfig]
    _cloud_job_type = "palace"

    def _mesh_by_net(self) -> bool:
        """Whether the mesh needs one physical group per net; only some sims do."""
        return False

    def _config_terminals(
        self,
        groups: dict[str, Any],  # noqa: ARG002
    ) -> list[Any] | None:
        """Terminals resolved to nets for the config; only net-mode sims have them."""
        return None

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    def set_output_dir(self, path: str | Path) -> None:
        """Set the output directory for mesh and config files.

        Args:
            path: Directory path for output files

        Example:
            >>> sim.set_output_dir("./palace-sim")
        """
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path | None:
        """Get the current output directory."""
        return self._output_dir

    @property
    def has_mesh(self) -> bool:
        """Whether ``mesh()`` has run and left a mesh to read."""
        return self._last_mesh_result is not None

    def _model_name(self) -> str:
        """The base name mesh() gave the mesh file and Palace's output dir."""
        result = self._last_mesh_result
        return result.model_name if result is not None else "palace"

    def _postpro_dir(self) -> Path:
        """Where a local run's Palace tables land.

        Raises:
            ValueError: When no output directory is set.
        """
        from gsim.palace.mesh.config_generator import output_subdir

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")
        return Path(self._output_dir) / output_subdir(self._model_name())

    def _run_context(self) -> str:
        """What was being run, named in the first line of an abort report."""
        return f"during the {self.simulation_type} solve"

    def _require_mesh_result(self) -> Any:
        """Return the last mesh result, or explain that meshing is missing."""
        result = self._last_mesh_result
        if result is None:
            raise ValueError(
                f"No mesh generated for this {type(self).__name__}. Call mesh() first."
            )
        return result

    @property
    def mesh_path(self) -> Path:
        """Path of the generated mesh (um), once meshed.

        Raises:
            ValueError: When read before ``mesh()`` has run.
        """
        return Path(self._require_mesh_result().mesh_path)

    @property
    def mesh_groups(self) -> dict[str, Any]:
        """Physical groups of the generated mesh, once meshed.

        Raises:
            ValueError: When read before ``mesh()`` has run.
        """
        return dict(self._require_mesh_result().groups or {})

    # -------------------------------------------------------------------------
    # Geometry methods
    # -------------------------------------------------------------------------

    def set_geometry(self, component: Component) -> None:
        """Set the gdsfactory component for simulation.

        Args:
            component: gdsfactory Component to simulate

        Example:
            >>> sim.set_geometry(my_component)
        """
        from gsim.common import Geometry

        self.geometry = Geometry(component=component)

    @property
    def component(self) -> Component | None:
        """Get the current component (for backward compatibility)."""
        return self.geometry.component if self.geometry else None

    # Backward compatibility alias
    @property
    def _component(self) -> Component | None:
        """Internal component access (backward compatibility)."""
        return self.component

    # -------------------------------------------------------------------------
    # Stack methods
    # -------------------------------------------------------------------------

    def set_stack(
        self,
        stack: LayerStack | None = None,
        *,
        yaml_path: str | Path | None = None,
        air_above: float | None = None,
        air_below: float | None = None,
        substrate_thickness: float | None = None,
        include_substrate: bool = False,
        add_oxide_dielectric: bool = True,
        pdk_authoritative: bool = False,
        dielectrics: list[dict[str, Any]] | None = None,
        role_overrides: dict[str, str] | None = None,
        use_pdk_overlay: bool = True,
        **kwargs,
    ) -> None:
        """Configure the layer stack.

        Three modes of use:

         1. **Active PDK** (default — auto-detects IHP, QPDK, etc.)::

             sim.set_stack(substrate_thickness=2.0)

        2. **YAML file**::

               sim.set_stack(yaml_path="custom_stack.yaml")

        3. **Custom stack** (advanced — pass a hand-built LayerStack)::

               sim.set_stack(my_layer_stack)

        Args:
            stack: Custom gsim LayerStack (bypasses PDK extraction).
            yaml_path: Path to custom YAML stack file.
            air_above: Deprecated and ignored. Use set_airbox().
            air_below: Deprecated and ignored. Use set_airbox().
            substrate_thickness: Depth below ``z=0`` in um (2.0 when omitted).
                It only matters where something is built from it: the
                synthetic oxide fill (``include_substrate=False`` with
                ``add_oxide_dielectric=True``) or the silicon substrate
                (``include_substrate=True``; a stack with shaped (photonic)
                dielectrics keeps the PDK's own substrate depth if that is
                deeper). With ``include_substrate=False`` and
                ``add_oxide_dielectric=False`` nothing is built from it, it is
                ignored, and a warning is logged if it was given.
            include_substrate: Include lossy silicon substrate (default
                ``False``). With ``False`` the PDK's substrate layer is left
                out of the simulation, so changing its thickness in the PDK
                stack changes neither the mesh nor the config. A warning is
                logged when the PDK substrate and these settings disagree.
            add_oxide_dielectric: Add the synthetic oxide background
                dielectric (default ``True``). When the substrate is omitted
                this is the region whose bottom is ``-substrate_thickness``.
                With ``False`` the stack bottom comes from the PDK's own
                dielectric regions, or from the lowest layer when there are
                none, not from ``substrate_thickness``.
            pdk_authoritative: Build directly from the PDK LayerStack without
                synthetic oxide or substrate; unknown materials raise. Defaults
                to ``False`` to preserve legacy behavior.
            dielectrics: Explicit bulk regions with
                ``name``/``material``/``zmin``/``zmax`` in authoritative mode.
            role_overrides: Explicit ``{layer_key: role}`` declarations in
                authoritative mode.
            use_pdk_overlay: Apply the reviewed per-PDK overlay in authoritative
                mode. Defaults to ``True``.
            **kwargs: Additional args passed to extract_layer_stack.

        The bottom of the Palace domain is the lowest non-air dielectric
        region of the resolved stack (the lowest layer when there is no such
        region) minus ``set_airbox(z_below=...)``. What sets that region:

        * ``include_substrate=True``: the silicon, whose bottom is
          ``-substrate_thickness`` (or the PDK substrate bottom, if deeper, for
          stacks with shaped dielectrics).
        * ``include_substrate=False`` and ``add_oxide_dielectric=True`` (the
          default): the oxide fill, whose bottom is ``-substrate_thickness``.
        * ``include_substrate=False`` and ``add_oxide_dielectric=False``: not
          ``substrate_thickness``; the PDK's dielectric layers and layer
          extents. Move the bottom with ``z_below`` or the PDK layers.

        With a custom ``LayerStack`` the bottom is whatever its dielectrics
        say.

        Example:
            >>> sim.set_stack(substrate_thickness=2.0)
            >>> sim.set_stack(
            ...     pdk_authoritative=True,
            ...     dielectrics=[
            ...         {"name": "ild", "material": "sio2", "zmin": -2.0, "zmax": 14.03}
            ...     ],
            ...     role_overrides={"via1": "via"},
            ... )  # PDK-authoritative stack with explicit background/roles
        """
        if stack is not None:
            # Directly use a pre-built LayerStack — skip lazy resolution
            self.stack = stack
            self._stack_kwargs = {"_prebuilt": True}
            return

        if air_above is not None or air_below is not None:
            logger.warning(
                "set_stack(air_above/air_below) is deprecated and ignored. "
                "Use set_airbox(margin_x=..., margin_y=..., "
                "z_above=..., z_below=...)."
            )

        if pdk_authoritative:
            self._warn_on_ignored_legacy_stack_kwargs(
                yaml_path=yaml_path,
                substrate_thickness=substrate_thickness,
                include_substrate=include_substrate,
                add_oxide_dielectric=add_oxide_dielectric,
                extra_kwargs=kwargs,
            )
        elif not use_pdk_overlay:
            raise ValueError("use_pdk_overlay=False requires pdk_authoritative=True")

        self._stack_kwargs = {
            "yaml_path": yaml_path,
            "include_substrate": include_substrate,
            "add_oxide_dielectric": add_oxide_dielectric,
            "pdk_authoritative": pdk_authoritative,
            "dielectrics": dielectrics,
            "role_overrides": role_overrides,
            "use_pdk_overlay": use_pdk_overlay,
            **kwargs,
        }
        # Left out when unset so the stack builder can tell the 2.0 um default
        # from a value the user chose, and warn only about the former.
        if substrate_thickness is not None:
            self._stack_kwargs["substrate_thickness"] = substrate_thickness
        # Stack will be resolved lazily during mesh() or simulate()
        self.stack = None

    def _warn_on_ignored_legacy_stack_kwargs(
        self,
        *,
        yaml_path: str | Path | None,
        substrate_thickness: float | None,
        include_substrate: bool,
        add_oxide_dielectric: bool,
        extra_kwargs: dict[str, Any],
    ) -> None:
        """Raise/warn when legacy (inferred-stack) kwargs cannot apply.

        With ``pdk_authoritative=True`` the PDK LayerStack is used verbatim, so
        the legacy extractor tuning is meaningless.  Passing it silently would
        be a trap (gsim#230 review), so reject the mutually-exclusive options
        and warn about the rest.
        """
        if yaml_path is not None:
            raise ValueError(
                "set_stack(pdk_authoritative=True) resolves the active PDK "
                "LayerStack directly and cannot be combined with yaml_path."
            )

        ignored: list[str] = []
        if substrate_thickness is not None:
            ignored.append(f"substrate_thickness={substrate_thickness!r}")
        if include_substrate:
            ignored.append("include_substrate=True")
        if not add_oxide_dielectric:
            ignored.append("add_oxide_dielectric=False")
        ignored.extend(f"{key}={value!r}" for key, value in extra_kwargs.items())
        if ignored:
            raise ValueError(
                "set_stack(pdk_authoritative=True) ignores legacy stack "
                f"options ({', '.join(sorted(ignored))}); they only apply to "
                "the inferred legacy extractor. Remove them or use explicit "
                "dielectrics=[...]/role_overrides={...} instead."
            )

    def set_airbox(
        self,
        *,
        margin_x: float | None = None,
        margin_y: float | None = None,
        z_above: float | None = None,
        z_below: float | None = None,
        material: str = "air",
    ) -> None:
        """Configure an explicit weak-priority airbox for meshing.

        This helper centralizes airbox controls that were previously split
        across mesh margin arguments and stack air thickness parameters.
        The resulting airbox is created as a dedicated dielectric volume
        with the weakest boolean-priority in mesh construction.

        ``material`` sets the material name of the padded background region.
        It defaults to ``"air"`` (eps=1). For a photonic cross-section with a
        uniform cladding, pass e.g. ``"sio2"`` so the region around the device
        is filled with SiO2 instead of air. The material must be resolvable in
        the stack's materials (or the built-in database); its permittivity is
        evaluated at the simulation frequency (e.g. Sellmeier SiO2 at 1550 nm).

        Note: the background material currently applies to the native-2D
        BoundaryMode domain (the region outside the stack/cladding); the 3D
        airbox region remains air.

        Args:
            margin_x: Airbox x-margin around the design (um).
                Defaults to ``0.0`` when omitted.
            margin_y: Airbox y-margin around the design (um).
                Defaults to ``0.0`` when omitted.
            z_above: Airbox extension above the stack top (um).
                Defaults to ``0.0`` when omitted.
            z_below: Airbox extension below the stack bottom (um).
                Defaults to ``0.0`` when omitted. The stack bottom is the
                lowest dielectric region: the silicon substrate with
                ``set_stack(include_substrate=True)``, otherwise the oxide
                bottom at ``-substrate_thickness``. With
                ``add_oxide_dielectric=False`` there is no such oxide, and the
                bottom comes from the PDK's remaining dielectric regions or
                layer extents (``substrate_thickness`` is ignored). A
                substrate layer left out of the stack does not move it.
            material: Background material name for the padded region
                (default ``"air"``).
        """
        mx = 0.0 if margin_x is None else margin_x
        my = 0.0 if margin_y is None else margin_y
        za = 0.0 if z_above is None else z_above
        zb = 0.0 if z_below is None else z_below

        if mx < 0:
            raise ValueError("margin_x must be >= 0")
        if my < 0:
            raise ValueError("margin_y must be >= 0")
        if za < 0 or zb < 0:
            raise ValueError("z_above and z_below must be >= 0")
        if not material or not isinstance(material, str):
            raise ValueError("material must be a non-empty string")

        # Deliberately not synced into ``mesh_config.margin_x``/``margin_y``:
        # the airbox config below is the single source of the lateral margin, and
        # writing it to both made it possible to apply it twice. See
        # ``_resolve_domain_margins``.

        # Store explicit airbox expansion for generator plumbing.
        self._airbox_config = {
            "margin_x": mx,
            "margin_y": my,
            "z_above": za,
            "z_below": zb,
            "material": material,
        }

    def _warn_if_substrate_omitted(self, stack: LayerStack) -> None:
        """Warn when a substrate layer of the stack cannot move the domain bottom.

        With ``include_substrate=False`` (the default) the PDK's substrate layer
        is left out of the stack, so a different silicon thickness there gives
        the same mesh and config. This matters once ``set_airbox(z_below=...)``
        is used to place the bottom boundary: it is measured from the stack
        bottom, not from the silicon. That is the oxide bottom
        (``-substrate_thickness``), or, with ``add_oxide_dielectric=False``,
        the lowest PDK dielectric region or layer extent.
        """
        omitted = getattr(stack, "omitted_substrate_layers", None)
        z_below = (self._airbox_config or {}).get("z_below") or 0.0
        if not omitted or z_below <= 0:
            return
        from gsim.palace.mesh.geometry import _resolve_stack_z_extent

        z_bottom, _ = _resolve_stack_z_extent(stack, True)
        if (stack.simulation or {}).get("add_oxide_dielectric", True):
            move_hint = "substrate_thickness / z_below"
        else:
            move_hint = (
                "z_below or the PDK dielectric layers (substrate_thickness "
                "has no effect with add_oxide_dielectric=False)"
            )
        layers = ", ".join(
            f"'{d['name']}' ({d['thickness']:g} um, z={d['zmin']:g} to {d['zmax']:g})"
            for d in omitted
        )
        logger.warning(
            "The stack's substrate layer %s is not part of the domain because "
            "include_substrate=False, so its thickness does not move the domain "
            "bottom. The bottom is z=%g um (stack bottom) minus "
            "set_airbox(z_below=%g). Use set_stack(include_substrate=True) to "
            "mesh the substrate, or set %s to move the bottom.",
            layers,
            z_bottom,
            z_below,
            move_hint,
        )

    def _resolve_domain_margins(self, mesh_config) -> tuple[float, float]:
        """Lateral margins for the mesh domain, excluding the airbox expansion.

        ``set_airbox()`` owns the lateral margin whenever it has been called: the
        requested value travels to the mesher as ``airbox_margin_x`` /
        ``airbox_margin_y`` and is applied there, so it must not be applied a
        second time as the domain margin. Passing it in both places padded the
        air volume twice, so ``set_airbox(margin_x=N)`` produced ``2 * N`` um of
        air.

        ``mesh_config`` is only the fallback, for a simulation that never called
        ``set_airbox()`` and sizes its domain through the mesh config alone.
        """
        airbox_cfg = self._airbox_config or {}
        margin_x = 0.0 if "margin_x" in airbox_cfg else mesh_config.effective_margin_x
        margin_y = 0.0 if "margin_y" in airbox_cfg else mesh_config.effective_margin_y
        return margin_x, margin_y

    def _apply_airbox_overrides(
        self,
        *,
        margin_x: float | None = None,
        margin_y: float | None = None,
        z_above: float | None = None,
        z_below: float | None = None,
    ) -> None:
        """Route mesh-time airbox kwargs through set_airbox()."""
        if (
            margin_x is None
            and margin_y is None
            and z_above is None
            and z_below is None
        ):
            return

        current = self._airbox_config or {}
        self.set_airbox(
            margin_x=margin_x if margin_x is not None else current.get("margin_x"),
            margin_y=margin_y if margin_y is not None else current.get("margin_y"),
            z_above=z_above if z_above is not None else current.get("z_above"),
            z_below=z_below if z_below is not None else current.get("z_below"),
            material=str(current.get("material", "air")),
        )

    # -------------------------------------------------------------------------
    # Impedance boundaries
    # -------------------------------------------------------------------------

    def add_impedance_boundary(
        self,
        layer_a: str,
        layer_b: str,
        *,
        capacitance: float | None = None,
        resistance: float | None = None,
        inductance: float | None = None,
        name: str | None = None,
    ) -> None:
        """Add an Impedance boundary on the interface between two dielectric layers.

        The capacitance, resistance, and inductance values are absolute and
        are divided by the interface curve length internally to produce the
        per-unit-length ``Cs``, ``Rs``, ``Ls`` values expected by Palace.

        Args:
            layer_a: First dielectric layer name.
            layer_b: Second dielectric layer name.
            capacitance: Absolute capacitance [F].
            resistance: Absolute resistance [Ohm].
            inductance: Absolute inductance [H].
            name: Optional display name.
        """
        self._impedance_boundaries.append(
            ImpedanceBoundaryConfig(
                layer_a=layer_a,
                layer_b=layer_b,
                capacitance=capacitance,
                resistance=resistance,
                inductance=inductance,
                name=name,
            )
        )

    def set_pn_junction(
        self,
        junction: Any,
        *,
        layer_p: str,
        layer_n: str,
        length_um: float,
        height_um: float,
        name: str | None = None,
    ) -> float:
        """Apply the depletion capacitance of a PN junction between two layers.

        Capacitance-mode modelling of a PN junction: the depletion width
        ``W`` is computed from the doping concentrations and bias point via
        :class:`gsim.common.stack.pn_junction.PNJunctionConfig` (Sze,
        *Physics of Semiconductor Devices*, ch. 2), converted to an absolute
        parallel-plate capacitance ``C = eps_s * A / W``, and applied as a
        lumped Impedance boundary on the shared P/N interface.

        Use this when the depletion strip is too thin to resolve on the mesh;
        for a well-resolved depletion region prefer drawing it as dielectric
        geometry of its own instead.

        Args:
            junction: ``PNJunctionConfig`` or its dict form (doping
                concentrations, bias, temperature, permittivity).
            layer_p: Name of the P-doped layer.
            layer_n: Name of the N-doped layer.
            length_um: Device length along the propagation direction (um).
            height_um: Junction z-extent (um), e.g. the rib height.
            name: Optional display name for the boundary.

        Returns:
            The absolute capacitance applied [F].

        Example:
            >>> sim.set_pn_junction(
            ...     {"na_cm3": 1e19, "nd_cm3": 1e19},
            ...     layer_p="p_rib",
            ...     layer_n="n_rib",
            ...     length_um=10.0,
            ...     height_um=0.22,
            ... )
        """
        from gsim.common.stack.pn_junction import PNJunctionConfig

        cfg = (
            junction
            if isinstance(junction, PNJunctionConfig)
            else PNJunctionConfig.model_validate(junction)
        )
        capacitance = cfg.capacitance(length_um=length_um, height_um=height_um)
        self.add_impedance_boundary(
            layer_p,
            layer_n,
            capacitance=capacitance,
            name=name,
        )
        return capacitance

    # -------------------------------------------------------------------------
    # Material methods
    # -------------------------------------------------------------------------

    def set_material(
        self,
        name: str,
        *,
        material_type: (
            Literal["conductor", "dielectric", "semiconductor"] | None
        ) = None,
        conductivity: float | None = None,
        permittivity: float | None = None,
        loss_tangent: float | None = None,
    ) -> None:
        """Override or add material properties.

        Args:
            name: Material name
            material_type: Material type (conductor, dielectric, semiconductor)
            conductivity: Conductivity in S/m (for conductors)
            permittivity: Relative permittivity (for dielectrics)
            loss_tangent: Dielectric loss tangent

        Example:
            >>> sim.set_material(
            ...     "aluminum", material_type="conductor", conductivity=3.8e7
            ... )
            >>> sim.set_material("sio2", material_type="dielectric", permittivity=3.9)
        """
        # Determine type if not provided
        resolved_type = material_type
        if resolved_type is None:
            if conductivity is not None and conductivity > 1e4:
                resolved_type = "conductor"
            elif permittivity is not None:
                resolved_type = "dielectric"
            else:
                resolved_type = "dielectric"

        self.materials[name] = MaterialConfig(
            type=resolved_type,
            conductivity=conductivity,
            permittivity=permittivity,
            loss_tangent=loss_tangent,
        )

    def add_pec(
        self,
        *,
        gds_layer: tuple[int, int],
        from_layer: str,
        to_layer: str,
    ) -> None:
        """Add a PEC block between two stack layers.

        The PEC block is a user-drawn polygon on a GDS layer that gets
        extruded between ``from_layer.zmin`` and ``to_layer.zmax`` and
        treated as a PEC boundary. This is a common practice in commercial
        EM solvers for connecting ground planes across metal layers at
        port boundaries.

        Args:
            gds_layer: GDS layer tuple where the PEC polygon is drawn.
            from_layer: Stack layer name — extrusion starts at this layer's zmin.
            to_layer: Stack layer name — extrusion ends at this layer's zmax.

        Example:
            >>> sim.add_pec(
            ...     gds_layer=(65000, 0), from_layer="metal1", to_layer="topmetal2"
            ... )
        """
        from gsim.palace.models.pec import PECBlockConfig

        self._pec_blocks.append(
            PECBlockConfig(
                gds_layer=gds_layer,
                from_layer=from_layer,
                to_layer=to_layer,
            )
        )

    def add_symmetry_plane(
        self,
        axis: Literal["x", "y"] = "y",
        position: float = 0.0,
        kind: Literal["pec", "pmc"] = "pmc",
        keep: Literal["positive", "negative"] = "positive",
        *,
        verify_symmetry: bool = True,
    ) -> None:
        """Simulate half of a mirror-symmetric structure.

        A PMC plane selects the even (common) mode, a PEC plane the odd
        (differential) mode. Only one plane is supported, and only for
        driven and eigenmode simulations.

        Args:
            axis: Axis normal to the plane.
            position: Plane position along ``axis`` in um, aligned to the
                owning layout's database grid. Checked during geometry extraction.
            kind: ``"pmc"`` (even mode) or ``"pec"`` (odd mode).
            keep: Side of the plane that is simulated.
            verify_symmetry: Check that the layout is mirror-symmetric.

        Raises:
            NotImplementedError: For electrostatic and boundarymode simulations.
            ValueError: If a plane was already added.

        Example:
            >>> sim.add_symmetry_plane(axis="y", position=0.0, kind="pec")
        """
        if self.simulation_type not in ("driven", "eigenmode"):
            raise NotImplementedError(
                "Symmetry planes are only supported for driven and eigenmode "
                f"simulations, not {self.simulation_type!r}."
            )
        if self._symmetry_planes:
            raise ValueError(
                "Only one symmetry plane is supported; one was already added."
            )
        self._symmetry_planes.append(
            SymmetryPlaneConfig(
                axis=axis,
                position=position,
                kind=kind,
                keep=keep,
                verify_symmetry=verify_symmetry,
            )
        )

    def set_refinement(
        self,
        *,
        max_its: int = 0,
        tol: float = 1e-2,
        uniform_levels: int = 0,
        max_dofs: int | None = None,
        update_fraction: float | None = None,
        nonconformal: bool | None = None,
        max_nc_levels: int | None = None,
        save_adapt_iterations: bool | None = None,
        save_adapt_mesh: bool | None = None,
    ) -> None:
        """Configure Palace's adaptive mesh refinement (AMR).

        Palace refines the elements that carry most of its estimated error and
        re-solves, until the error norm falls below ``tol``, ``max_its``
        passes have run, or the problem reaches ``max_dofs`` degrees of
        freedom. See :class:`~gsim.palace.models.RefinementConfig`.

        Args:
            max_its: Maximum number of AMR passes. 0 disables AMR.
            tol: Stop refining when the estimated error norm falls below this.
            uniform_levels: Uniform refinement levels applied to the input
                mesh before solving.
            max_dofs: Maximum degrees of freedom. None means no limit.
            update_fraction: Dörfler marking fraction, between 0 and 1.
            nonconformal: Refine with hanging nodes instead of conformally.
            max_nc_levels: Maximum nonconformal refinement levels; 0 means
                no limit.
            save_adapt_iterations: Keep the output of every pass in an
                ``iterationX`` subdirectory.
            save_adapt_mesh: Save the final adapted mesh.

        Example:
            >>> sim.set_refinement(max_its=5, tol=1e-3, max_dofs=2_000_000)
        """
        self.refinement = RefinementConfig(
            max_its=max_its,
            tol=tol,
            uniform_levels=uniform_levels,
            max_dofs=max_dofs,
            update_fraction=update_fraction,
            nonconformal=nonconformal,
            max_nc_levels=max_nc_levels,
            save_adapt_iterations=save_adapt_iterations,
            save_adapt_mesh=save_adapt_mesh,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _resolve_stack(self) -> LayerStack:
        """Resolve the layer stack from PDK, YAML, or custom object.

        Returns:
            LayerStack object for mesh generation
        """
        # If a custom stack was given via set_stack(layer_stack), use it
        if self.stack is not None and self._stack_kwargs.get("_prebuilt"):
            # Apply material overrides
            for name, props in self.materials.items():
                self.stack.materials[name] = props.to_dict()
            return self.stack

        # PDK-authoritative path (gsim#230): build directly from the PDK
        # LayerStack, with no synthetic geometry or silent material fallbacks.
        # Read (do not pop) these options: if resolution raises, a caller that
        # follows the error's advice (e.g. calls set_airbox() and retries) must
        # still resolve through the authoritative path.
        dielectrics = self._stack_kwargs.get("dielectrics")
        role_overrides = self._stack_kwargs.get("role_overrides")
        pdk_authoritative = bool(self._stack_kwargs.get("pdk_authoritative", False))
        if pdk_authoritative:
            from gsim.common.pdk import resolve_layer_stack
            from gsim.palace.stack import build_palace_layer_stack

            resolved = resolve_layer_stack(
                self.component,
                dielectrics=dielectrics,
                role_overrides=role_overrides,
                use_pdk_overlay=bool(self._stack_kwargs.get("use_pdk_overlay", True)),
            )
            canonical = build_palace_layer_stack(resolved)

            # Apply material overrides to both canonical names and PDK tokens
            # that resolve to them. Layers retain their original material token.
            from gsim.common.stack.materials import MATERIAL_ALIASES

            for name, props in self.materials.items():
                target = MATERIAL_ALIASES.get(name.lower(), name.lower())
                for token in canonical.materials:
                    if MATERIAL_ALIASES.get(token.lower(), token.lower()) == target:
                        canonical.materials[token] = props.to_dict()
                canonical.materials[name] = props.to_dict()

            # Keep stack free of synthetic air regions; Palace airboxes are
            # generated explicitly from set_airbox().
            canonical.dielectrics = [
                d
                for d in canonical.dielectrics
                if str(d.get("material", "")).lower() != "air"
            ]

            has_airbox = bool(getattr(self, "_airbox_config", None))
            if not canonical.dielectrics and not has_airbox:
                raise ValueError(
                    "PDK-authoritative stack has no bulk dielectric region and "
                    "no airbox is configured, so Palace would fail with "
                    "'No dielectric volumes in mesh'. Pass "
                    "dielectrics=[...] to set_stack() or call set_airbox()."
                )
            if not canonical.dielectrics:
                logger.warning(
                    "PDK-authoritative stack has no bulk dielectric region; the "
                    "domain background will be air only. Pass dielectrics=[...] "
                    "to set_stack() to model the BEOL/cladding."
                )

            self._stack_kwargs["pdk_authoritative"] = True
            self._stack_kwargs["dielectrics"] = dielectrics
            self._stack_kwargs["role_overrides"] = role_overrides
            self._stack_kwargs["use_pdk_overlay"] = bool(
                self._stack_kwargs.get("use_pdk_overlay", True)
            )
            self.stack = canonical
            return canonical

        # Legacy path: the authoritative-only options must not reach the
        # inferred extractor.
        dielectrics = self._stack_kwargs.pop("dielectrics", None)
        role_overrides = self._stack_kwargs.pop("role_overrides", None)
        self._stack_kwargs.pop("pdk_authoritative", None)
        self._stack_kwargs.pop("use_pdk_overlay", None)
        if dielectrics or role_overrides:
            logger.warning(
                "set_stack(dielectrics=/role_overrides=) only applies with "
                "pdk_authoritative=True; ignoring them."
            )

        from gsim.common.stack import get_stack

        yaml_path = self._stack_kwargs.pop("yaml_path", None)
        legacy_stack = get_stack(yaml_path=yaml_path, **self._stack_kwargs)

        # Restore yaml_path for potential re-resolution
        self._stack_kwargs["yaml_path"] = yaml_path

        # Apply material overrides
        for name, props in self.materials.items():
            legacy_stack.materials[name] = props.to_dict()

        # Keep stack free of synthetic air regions; Palace airboxes are
        # generated explicitly from set_airbox().
        legacy_stack.dielectrics = [
            d
            for d in legacy_stack.dielectrics
            if str(d.get("material", "")).lower() != "air"
        ]

        # Store the LayerStack
        self.stack = legacy_stack

        return legacy_stack

    def _build_mesh_config(
        self,
        preset: Literal["coarse", "default", "fine"] | None,
        refined_mesh_size: float | None,
        max_mesh_size: float | None,
        margin: float | None,
        airbox_margin: float | None,
        fmax: float | None,
        planar_conductors: bool | None,
        show_gui: bool,
        margin_x: float | None = None,
        margin_y: float | None = None,
        auto_size: bool = False,
        cells_per_feature: int = 2,
        curve_fit_mode: Literal["line", "spline", "bspline"] | None = None,
        curve_fit_layers: list[str] | None = None,
        curve_fit_tolerance_um: float | None = None,
        curve_fit_min_points: int | None = None,
        curve_fit_corner_angle_deg: float | None = None,
        high_order_elements: bool | None = None,
        high_order_order: int | None = None,
        high_order_optimize: bool | None = None,
        refinement_boxes: (
            list[tuple[float, float, float, float, float]] | None
        ) = None,
        algorithm_3d: Literal["delaunay", "hxt"] | None = None,
        threads: int | None = None,
        surface_threads: int | None = None,
    ) -> MeshConfig:
        """Build mesh config from preset with optional overrides.

        When ``auto_size`` is True and ``refined_mesh_size`` is not explicitly
        provided, ``refined_mesh_size`` is scaled down to
        ``min(preset_size, min_conductor_feature / cells_per_feature)``.

        Explicit sizes always override the preset and bypass auto-sizing.
        Warn when the effective size exceeds the detected feature size divided
        by ``cells_per_feature``; this is a heuristic, not an accuracy guarantee.
        """
        if preset is None or preset == "default":
            mesh_config = MeshConfig.default()
        elif preset == "coarse":
            mesh_config = MeshConfig.coarse()
        elif preset == "fine":
            mesh_config = MeshConfig.fine()
        else:
            raise ValueError(
                f"Unknown preset {preset!r}; "
                "expected one of 'coarse', 'default', 'fine', or None"
            )

        # Resolve the explicit size before any resolution diagnostics.
        if refined_mesh_size is not None:
            mesh_config.refined_mesh_size = refined_mesh_size

        # Auto-size is explicit opt-in. Presets are "honest" speed tiers by
        # default; when auto_size=True the refined_mesh_size shrinks to
        # min(preset, min_conductor_feature / cells_per_feature). When
        # auto_size=False we just warn if a small feature risks being
        # under-resolved.
        if auto_size and refined_mesh_size is None:
            component = self.geometry.component if self.geometry else None
            if component is not None:
                from gsim.palace.mesh.auto_size import auto_refined_mesh_size

                stack = self._resolve_stack()
                scaled = auto_refined_mesh_size(
                    component,
                    stack,
                    preset_size=mesh_config.refined_mesh_size,
                    cells_per_feature=cells_per_feature,
                )
                if scaled < mesh_config.refined_mesh_size:
                    logger.info(
                        "Auto-sizing refined_mesh_size: %.3f -> %.3f um "
                        "(min conductor feature / %d)",
                        mesh_config.refined_mesh_size,
                        scaled,
                        cells_per_feature,
                    )
                    mesh_config.refined_mesh_size = scaled
        if (not auto_size or refined_mesh_size is not None) and cells_per_feature > 0:
            component = self.geometry.component if self.geometry else None
            if component is not None:
                from gsim.palace.mesh.auto_size import min_conductor_feature_size

                stack = self._resolve_stack()
                feature = min_conductor_feature_size(component, stack)
                if feature is not None and math.isfinite(feature) and feature > 0:
                    suggested_size = feature / cells_per_feature
                    if (
                        mesh_config.refined_mesh_size > suggested_size
                        and not math.isclose(
                            mesh_config.refined_mesh_size, suggested_size, rel_tol=1e-9
                        )
                    ):
                        advice = (
                            "Reduce refined_mesh_size; auto_size=True preserves an "
                            "explicit size."
                            if refined_mesh_size is not None
                            else "Pass auto_size=True to scale the mesh down."
                        )
                        logger.warning(
                            "Small conductor feature detected (%.3f um) may be "
                            "under-resolved by refined_mesh_size=%.3f um. "
                            "The heuristic target is <= %.3f um (%d cells/feature). %s",
                            feature,
                            mesh_config.refined_mesh_size,
                            suggested_size,
                            cells_per_feature,
                            advice,
                        )

        existing_config = self.mesh_config

        # Preserve curve-fit settings from sim.mesh_config unless overridden.
        mesh_config.curve_fit_mode = existing_config.curve_fit_mode
        mesh_config.curve_fit_layers = list(existing_config.curve_fit_layers)
        mesh_config.curve_fit_tolerance_um = existing_config.curve_fit_tolerance_um
        mesh_config.curve_fit_min_points = existing_config.curve_fit_min_points
        mesh_config.curve_fit_corner_angle_deg = (
            existing_config.curve_fit_corner_angle_deg
        )
        mesh_config.high_order_elements = existing_config.high_order_elements
        mesh_config.high_order_order = existing_config.high_order_order
        mesh_config.high_order_optimize = existing_config.high_order_optimize
        mesh_config.refinement_boxes = list(existing_config.refinement_boxes)
        mesh_config.algorithm_3d = existing_config.algorithm_3d
        mesh_config.threads = existing_config.threads
        mesh_config.surface_threads = existing_config.surface_threads
        mesh_config.clip_merged_vias = existing_config.clip_merged_vias

        # Preserve planar_conductors from sim.mesh_config if not
        # explicitly provided via sim.mesh(planar_conductors=...)
        if planar_conductors is None:
            mesh_config.planar_conductors = existing_config.planar_conductors
        else:
            mesh_config.planar_conductors = planar_conductors

        # Apply remaining overrides
        if max_mesh_size is not None:
            mesh_config.max_mesh_size = max_mesh_size
        if margin is not None:
            mesh_config.margin = margin
        if margin_x is not None:
            mesh_config.margin_x = margin_x
        if margin_y is not None:
            mesh_config.margin_y = margin_y
        if airbox_margin is not None:
            raise ValueError(
                "airbox_margin has been removed from mesh()/preview(). "
                "Use set_airbox(margin_x=..., margin_y=..., "
                "z_above=..., z_below=...)."
            )
        if fmax is not None:
            mesh_config.fmax = fmax
        if curve_fit_mode is not None:
            mesh_config.curve_fit_mode = curve_fit_mode
        if curve_fit_layers is not None:
            mesh_config.curve_fit_layers = curve_fit_layers
        if curve_fit_tolerance_um is not None:
            mesh_config.curve_fit_tolerance_um = curve_fit_tolerance_um
        if curve_fit_min_points is not None:
            mesh_config.curve_fit_min_points = curve_fit_min_points
        if curve_fit_corner_angle_deg is not None:
            mesh_config.curve_fit_corner_angle_deg = curve_fit_corner_angle_deg
        if high_order_elements is not None:
            mesh_config.high_order_elements = high_order_elements
        if high_order_order is not None:
            mesh_config.high_order_order = high_order_order
        if high_order_optimize is not None:
            mesh_config.high_order_optimize = high_order_optimize
        if refinement_boxes is not None:
            mesh_config.refinement_boxes = refinement_boxes
        if algorithm_3d is not None:
            mesh_config.algorithm_3d = algorithm_3d
        if threads is not None:
            mesh_config.threads = threads
        if surface_threads is not None:
            mesh_config.surface_threads = surface_threads
        mesh_config.show_gui = show_gui

        return mesh_config

    # -------------------------------------------------------------------------
    # Post-mesh validation
    # -------------------------------------------------------------------------

    def validate_mesh(self) -> ValidationResult:
        """Validate the generated mesh and config before cloud submission.

        Checks that physical groups are correctly assigned after meshing:
        conductor surfaces, dielectric volumes, ports, and absorbing boundary.
        Also verifies the generated config.json structure.

        Call after mesh() and before run().

        Returns:
            ValidationResult with validation status and messages when valid.

        Raises:
            RuntimeError: If mesh/config validation fails.

        Example:
            >>> sim.mesh(preset="coarse")
            >>> result = sim.validate_mesh()
            >>> print(result)
        """
        from gsim.palace.mesh.validation import validate_mesh as _validate_mesh

        result = _validate_mesh(self)
        if not result.valid:
            raise RuntimeError(f"Mesh validation failed:\n{result}")
        return result

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def show_stack(self) -> None:
        """Print the layer stack table.

        Example:
            >>> sim.show_stack()
        """
        from gsim.common.stack import print_stack_table

        if self.stack is None:
            self._resolve_stack()

        if self.stack is not None:
            print_stack_table(self.stack)

    def plot_stack(self) -> None:
        """Plot the layer stack visualization.

        Example:
            >>> sim.plot_stack()
        """
        from gsim.common.stack import plot_stack

        if self.stack is None:
            self._resolve_stack()

        if self.stack is not None:
            plot_stack(self.stack)

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_mesh(
        self,
        output: str | Path | None = None,
        show_groups: list[str] | None = None,
        interactive: bool | None = None,
        style: Literal["wireframe", "solid"] = "wireframe",
        transparent_groups: list[str] | None = None,
    ) -> None:
        """Plot the mesh using PyVista.

        Requires mesh() to be called first.

        Args:
            output: Output PNG path (only used for static rendering).
            show_groups: List of group name patterns to show (None = all).
                Example: ["metal", "P"] to show metal layers and ports.
            interactive: If True, force an interactive 3D viewer (a widget on
                the shared trame server in a notebook, or a blocking window
                otherwise).  If False, save a static PNG.  If None (default),
                follow the session flag from ``gsim.viz.set_interactive_mode``
                — off by default, so plots render statically until enabled.
            style: ``"wireframe"`` (edges only) or ``"solid"`` (coloured
                surfaces per physical group).
            transparent_groups: Group names rendered at low opacity in
                *solid* mode.  Ignored in *wireframe* mode.

        Raises:
            ValueError: If output_dir not set or mesh file doesn't exist

        Example:
            >>> sim.mesh(preset="default")
            >>> sim.plot_mesh(show_groups=["metal", "P"])
            >>> sim.plot_mesh(style="solid", transparent_groups=["Absorbing_boundary"])
        """
        from gsim.viz import plot_mesh as _plot_mesh

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        mesh_path = self._output_dir / f"{self._model_name()}.msh"
        if not mesh_path.exists():
            raise ValueError(f"Mesh file not found: {mesh_path}. Call mesh() first.")

        # Default output path for the static rendering path.
        if output is None and interactive is not True:
            output = self._output_dir / "mesh.png"

        _plot_mesh(
            msh_path=mesh_path,
            output=output,
            show_groups=show_groups,
            interactive=interactive,
            style=style,
            transparent_groups=transparent_groups,
        )

    # -------------------------------------------------------------------------
    # Port lists, for the sim classes that carry them
    # -------------------------------------------------------------------------

    @property
    def _ports(self) -> list[PortConfig]:
        """The lumped ports declared, or none on a sim without any."""
        return list(getattr(self, "ports", None) or [])

    @property
    def _cpw_ports(self) -> list[CPWPortConfig]:
        """The CPW ports declared, or none on a sim without any."""
        return list(getattr(self, "cpw_ports", None) or [])

    @property
    def _wave_ports(self) -> list[Any]:
        """The wave ports declared, or none on a sim without any."""
        return list(getattr(self, "wave_ports", None) or [])

    @property
    def _two_terminal_ports(self) -> list[Any]:
        """The two-terminal ports declared, or none on a sim without any."""
        return list(getattr(self, "two_terminal_ports", None) or [])

    @property
    def _terminals(self) -> list[Any]:
        """The terminals declared, or none on a sim without any."""
        return list(getattr(self, "terminals", None) or [])

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_config(self) -> ValidationResult:
        """Validate the simulation configuration.

        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings_list = []

        # Check geometry
        if self.geometry is None:
            errors.append("No component set. Call set_geometry(component) first.")

        # Check stack
        if self.stack is None and not self._stack_kwargs:
            warnings_list.append(
                "No stack configured. Will use active PDK with defaults."
            )

        # Check ports

        ports, cpw_ports, wave_ports = self._ports, self._cpw_ports, self._wave_ports
        has_ports = (
            bool(ports)
            or bool(cpw_ports)
            or bool(wave_ports)
            or bool(self._two_terminal_ports)
        )
        if not has_ports:
            if self.simulation_type == "driven":
                warnings_list.append(
                    "No ports configured. Call add_port(), add_cpw_port(),"
                    " or add_wave_port()."
                )
            elif self.simulation_type == "eigenmode":
                warnings_list.append(
                    "No ports configured. Eigenmode findsallmodes without port loading."
                )
        else:
            # Validate port configurations
            for port in ports:
                if port.geometry == "inplane" and port.layer is None:
                    errors.append(f"Port '{port.name}': inplane ports require 'layer'")
                if port.geometry == "interlayer" and (
                    port.from_layer is None or port.to_layer is None
                ):
                    errors.append(
                        f"Port '{port.name}': interlayer ports require "
                        "'from_layer' and 'to_layer'"
                    )

            # Validate CPW ports
            errors.extend(
                f"CPW port '{cpw.name}': 'layer' is required"
                for cpw in cpw_ports
                if not cpw.layer
            )
            # Validate wave ports
            errors.extend(
                f"Wave port '{wp.name}': 'layer' is required"
                for wp in wave_ports
                if not wp.layer
            )
            errors.extend(self._multimode_wave_port_errors(wave_ports))
            errors.extend(self._wave_port_face_errors(wave_ports))

        eigenmode = self._get_problem_settings("eigenmode")
        if eigenmode is not None and eigenmode.target is None:
            errors.append(
                "A positive eigenmode target frequency is required. "
                "Call set_eigenmode(target=...) with the frequency in Hz."
            )

        # Validate excitation port if specified
        driven = self._get_problem_settings("driven")
        if driven is not None and driven.excitation_port is not None:
            port_names = [p.name for p in self.ports]
            cpw_names = [cpw.name for cpw in self.cpw_ports]
            all_port_names = port_names + cpw_names
            if driven.excitation_port not in all_port_names:
                errors.append(
                    f"Excitation port '{driven.excitation_port}' not found. "
                    f"Available: {all_port_names}"
                )

        if self.simulation_type == "electrostatic" and len(self._terminals) < 2:
            # Electrostatic requires at least 2 terminals
            errors.append(
                "Electrostatic simulation requires at least 2 terminals. "
                "Call add_terminal() to add terminals."
            )
        if self.simulation_type == "electrostatic":
            # Validate terminal configurations
            terminal_errors = [
                f"Terminal '{terminal.name}': needs layer, net or region"
                for terminal in self._terminals
                if not (terminal.layer or terminal.net or terminal.region)
            ]
            errors.extend(terminal_errors)
            # Two layer-only terminals on one layer select the same conductors
            by_layer: dict[str, list[str]] = {}
            for terminal in self._terminals:
                if terminal.layer and terminal.net is None and terminal.region is None:
                    by_layer.setdefault(terminal.layer, []).append(terminal.name)
            warnings_list.extend(
                f"Terminals {', '.join(repr(n) for n in names)} all select every "
                f"shape on layer '{layer}', so they get the same conductors. Use "
                "net= or region= to separate electrodes on one layer."
                for layer, names in by_layer.items()
                if len(names) > 1
            )
            # Unknown nets, empty regions and shared nets fail before meshing
            resolve = getattr(self, "resolve_terminals", None)
            if (
                resolve is not None
                and not terminal_errors
                and self.geometry is not None
                and len(self._terminals) >= 2
                and self._mesh_by_net()
            ):
                try:
                    resolve()
                except Exception as error:
                    message = error.args[0] if error.args else error
                    errors.append(f"Terminals: {message}")

        if self._symmetry_planes and self.simulation_type not in (
            "driven",
            "eigenmode",
        ):
            errors.append(
                "Symmetry planes are only supported for driven and eigenmode "
                f"simulations, not {self.simulation_type!r}."
            )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _multimode_wave_port_errors(self, wave_ports: list[Any]) -> list[str]:
        """Errors that only multimode wave ports can have.

        Two ports on one face would be two surfaces in one place, and Palace
        refuses two active ports on one attribute; a multimode port puts all
        its modes on one surface instead, so the second port is refused here.
        """
        from gsim.palace.ports.modes import same_face_pairs

        if not any(wp.modes for wp in wave_ports):
            return []
        errors: list[str] = []
        if self.simulation_type != "driven":
            errors.append(
                "Multimode wave ports are only supported for driven simulations, "
                f"not {self.simulation_type!r}."
            )
        component = self.geometry.component if self.geometry else None
        if component is None:
            return errors
        by_name = {port.name: port for port in component.ports}
        faces = [
            (
                wp.name,
                (float(by_name[wp.name].center[0]), float(by_name[wp.name].center[1])),
                float(by_name[wp.name].orientation or 0.0),
            )
            for wp in wave_ports
            if wp.name in by_name
        ]
        errors.extend(
            f"Wave ports {first!r} and {second!r} are on the same face; give a "
            "face one wave port and list its modes with modes=..."
            for first, second in same_face_pairs(faces)
        )
        return errors

    def _wave_port_face_errors(self, wave_ports: list[Any]) -> list[str]:
        """Errors for wave ports whose generated faces overlap.

        The mesher builds one rectangle per wave port: a ``max_size`` port
        spans the whole side of the domain whatever its centre, margins widen
        a face. Where two rectangles overlap the boolean pass merges them, one
        port keeps no surface and ``write_config()`` emits no ``WavePort``
        entry for it. The rectangles are built here with the mesher's own
        function. The domain margins are only known at ``mesh()``, so the
        domain is left unbounded, which does not change which faces overlap.
        The mesher repeats the check with the real domain.
        """
        component = self.geometry.component if self.geometry else None
        if component is None or len(wave_ports) < 2:
            return []
        from gsim.palace.mesh.geometry import (
            UNBOUNDED_DOMAIN,
            overlapping_wave_ports,
            wave_port_overlap_message,
        )
        from gsim.palace.ports.config import PalacePort, PortType
        from gsim.palace.ports.modes import same_face_pairs

        by_name = {port.name: port for port in component.ports}
        ports = [
            PalacePort(
                name=wp.name,
                port_type=PortType.WAVEPORT,
                center=(
                    float(by_name[wp.name].center[0]),
                    float(by_name[wp.name].center[1]),
                ),
                width=float(by_name[wp.name].width),
                orientation=float(by_name[wp.name].orientation or 0.0),
                layer=wp.layer,
                z_margin=wp.z_margin,
                lateral_margin=wp.lateral_margin,
                max_size=wp.max_size,
            )
            for wp in wave_ports
            if wp.layer and wp.name in by_name
        ]
        if len(ports) < 2:
            return []
        # Resolving the stack must not leave a stack behind: set_stack() may
        # still follow, and mesh() resolves it again.
        previous = self.stack
        try:
            stack = self._resolve_stack()
        except Exception:
            return []  # the mesher reports a stack that does not resolve
        finally:
            self.stack = previous
        # A pair on the same face is already reported by the multimode check
        reported = (
            {
                frozenset(pair)
                for pair in same_face_pairs(
                    (p.name, p.center, p.orientation) for p in ports
                )
            }
            if any(wp.modes for wp in wave_ports)
            else set()
        )
        return [
            wave_port_overlap_message(first, second)
            for first, second in overlapping_wave_ports(
                ports, stack, domain_bounds=UNBOUNDED_DOMAIN
            )
            if frozenset((first, second)) not in reported
        ]

    def _find_gf_port(self, port_name: str):
        """Find a gdsfactory port by name."""
        component = self.geometry.component if self.geometry else None
        if component is None:
            raise ValueError("No component set")

        for p in component.ports:
            if p.name == port_name:
                return p

        raise ValueError(
            f"Port '{port_name}' not found on component. "
            f"Available ports: {[p.name for p in component.ports]}"
        )

    def _configure_ports_on_component(self, stack: LayerStack) -> None:  # noqa: ARG002
        """Configure ports on the component using legacy functions."""
        from gsim.palace.ports import (
            configure_cpw_port,
            configure_inplane_port,
            configure_interlayer_port,
            configure_two_terminal_port,
            configure_wave_port,
        )

        component = self.geometry.component if self.geometry else None
        if component is None:
            raise ValueError("No component set")

        # Configure regular ports
        for port_config in self._ports:
            if self.simulation_type != "driven":
                port_config.excited = False
            if port_config.name is None:
                continue

            # Find matching gdsfactory port
            gf_port = self._find_gf_port(port_config.name)

            if gf_port is None:
                raise ValueError(
                    f"Port '{port_config.name}' not found on component. "
                    f"Available ports: {[p.name for p in component.ports]}"
                )

            if port_config.geometry == "gap":
                from gsim.palace.ports.config import configure_gap_port

                if port_config.layer is None:
                    raise ValueError(
                        f"Port '{port_config.name}': gap port requires a layer."
                    )
                configure_gap_port(
                    gf_port,
                    layer=port_config.layer,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                    offset=port_config.offset,
                )
            elif port_config.geometry == "inplane" and port_config.layer is not None:
                configure_inplane_port(
                    gf_port,
                    layer=port_config.layer,
                    length=port_config.length or gf_port.width,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                    offset=port_config.offset,
                )
            elif port_config.geometry == "interlayer" and (
                port_config.from_layer is not None and port_config.to_layer is not None
            ):
                configure_interlayer_port(
                    gf_port,
                    from_layer=port_config.from_layer,
                    to_layer=port_config.to_layer,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                    offset=port_config.offset,
                )

            # Clear any stale reactive metadata from previous simulations.
            # kfactory Info is mapping-like (get/setitem) but has no pop/delitem.
            gf_port.info["resistance"] = None
            gf_port.info["inductance"] = None
            gf_port.info["capacitance"] = None

            # Attach RLC values to port info for downstream consumers.
            if port_config.resistance is not None:
                gf_port.info["resistance"] = port_config.resistance
            if port_config.inductance is not None:
                gf_port.info["inductance"] = port_config.inductance
            if port_config.capacitance is not None:
                gf_port.info["capacitance"] = port_config.capacitance

        # Configure CPW ports
        for cpw_config in self._cpw_ports:
            # Find the single gdsfactory port at the signal center
            gf_port = self._find_gf_port(cpw_config.name)

            if gf_port is None:
                raise ValueError(
                    f"CPW port '{cpw_config.name}' not found on component. "
                    f"Available: {[p.name for p in component.ports]}"
                )

            configure_cpw_port(
                gf_port,
                layer=cpw_config.layer,
                s_width=cpw_config.s_width,
                gap_width=cpw_config.gap_width,
                length=cpw_config.length,
                impedance=cpw_config.impedance,
                excited=cpw_config.excited,
                offset=cpw_config.offset,
            )
        # Configure wave ports
        for port_config in self._wave_ports:
            if port_config.name is None:
                continue

            # Find matching gdsfactory port
            gf_port = self._find_gf_port(port_config.name)

            if port_config.layer is not None:
                configure_wave_port(
                    gf_port,
                    layer=port_config.layer,
                    z_margin=port_config.z_margin,
                    lateral_margin=port_config.lateral_margin,
                    max_size=port_config.max_size,
                    excited=port_config.excited,
                    mode=port_config.mode,
                    offset=port_config.offset,
                    eigensolver_type=port_config.eigensolver_type,
                    eigensolver_tol=port_config.eigensolver_tol,
                    eigensolver_ksp_tol=port_config.eigensolver_ksp_tol,
                    eigensolver_max_size=port_config.eigensolver_max_size,
                    eigensolver_verbose=port_config.eigensolver_verbose,
                    modes=port_config.modes,
                    excite_modes=port_config.excite_modes,
                    pilot_paths=port_config.pilot_paths,
                )

        # Configure two-terminal ports
        for tt_config in self._two_terminal_ports:
            plus_gf_port = self._find_gf_port(tt_config.plus_port)
            minus_gf_port = self._find_gf_port(tt_config.minus_port)

            configure_two_terminal_port(
                plus_gf_port,
                minus_gf_port,
                layer=tt_config.layer,
                impedance=tt_config.impedance,
                excited=tt_config.excited,
            )

        self._configured_ports = True

    def _generate_mesh_internal(
        self,
        output_dir: Path,
        mesh_config: MeshConfig,
        ports: list,
        driven_config: Any,
        model_name: str,
        verbose: bool,
        write_config: bool = True,
        periodic_axis: str | None = None,
        decimate_tolerance: float | None = None,
        gmsh_verbosity: int = 0,
    ) -> SimulationResult:
        """Internal mesh generation."""
        from gsim.palace.mesh.generator import generate_mesh

        component = self.geometry.component if self.geometry else None

        # Get effective fmax from driven config if mesh doesn't specify
        effective_fmax = mesh_config.fmax
        if driven_config is not None and mesh_config.fmax == 100e9:
            effective_fmax = driven_config.fmax

        # Resolve stack
        stack = self._resolve_stack()
        self._warn_if_substrate_omitted(stack)
        airbox_cfg = self._airbox_config or {}
        domain_margin_x, domain_margin_y = self._resolve_domain_margins(mesh_config)

        if verbose:
            logger.info("Generating mesh in %s", output_dir)

        mesh_result = generate_mesh(
            component=component,
            stack=stack,
            ports=ports,
            output_dir=output_dir,
            model_name=model_name,
            refined_mesh_size=mesh_config.refined_mesh_size,
            max_mesh_size=mesh_config.max_mesh_size,
            margin_x=domain_margin_x,
            margin_y=domain_margin_y,
            air_margin=mesh_config.airbox_margin,
            airbox_margin_x=airbox_cfg.get("margin_x"),
            airbox_margin_y=airbox_cfg.get("margin_y"),
            airbox_z_above=airbox_cfg.get("z_above"),
            airbox_z_below=airbox_cfg.get("z_below"),
            airbox_material=airbox_cfg.get("material", "air"),
            fmax=effective_fmax,
            show_gui=mesh_config.show_gui,
            simulation_type=self.simulation_type,
            driven_config=driven_config,
            eigenmode_config=self._get_problem_settings("eigenmode"),
            numerical_config=self.solver,
            boundary_mode_config=self._get_problem_settings("boundary_mode"),
            cross_section=getattr(self, "cross_section", None),
            contact_specs=getattr(self, "contact_specs", None),
            interface_specs=getattr(self, "interface_specs", None),
            write_config=write_config,
            planar_conductors=mesh_config.planar_conductors,
            pec_blocks=self._pec_blocks or None,
            absorbing_boundary=self._outer_wall()[0],
            absorbing_order=self.absorbing_order,
            periodic_axis=periodic_axis,
            symmetry_plane=self._symmetry_planes[0] if self._symmetry_planes else None,
            merge_via_distance=mesh_config.merge_via_distance,
            clip_merged_vias=mesh_config.clip_merged_vias,
            curve_fit_mode=mesh_config.curve_fit_mode,
            curve_fit_layers=mesh_config.curve_fit_layers,
            curve_fit_tolerance_um=mesh_config.curve_fit_tolerance_um,
            curve_fit_min_points=mesh_config.curve_fit_min_points,
            curve_fit_corner_angle_deg=mesh_config.curve_fit_corner_angle_deg,
            high_order_elements=mesh_config.high_order_elements,
            high_order_order=mesh_config.high_order_order,
            high_order_optimize=mesh_config.high_order_optimize,
            refinement_boxes=mesh_config.refinement_boxes,
            algorithm_3d=mesh_config.algorithm_3d,
            threads=mesh_config.threads,
            surface_threads=mesh_config.surface_threads,
            verbosity=gmsh_verbosity,
            decimate_tolerance=decimate_tolerance,
            split_conductors_by_net=self._mesh_by_net(),
        )

        # Store mesh_result for deferred config generation
        self._last_mesh_result = mesh_result
        self._last_ports = ports

        return SimulationResult(
            mesh_path=mesh_result.mesh_path,
            output_dir=output_dir,
            config_path=mesh_result.config_path,
            port_info=mesh_result.port_info,
            mesh_stats=mesh_result.mesh_stats,
            metadata=mesh_result.metadata,
        )

    def print_mesh_stats(self) -> None:
        """Print mesh statistics from the last mesh generation.

        Reports node/element counts and input-mesh Field DOF estimates at the
        configured field order, before Palace preprocessing and refinement.
        """
        mr = self._last_mesh_result
        if mr is None:
            print("No mesh result. Call mesh() first.")  # noqa: T201
            return

        stats = mr.mesh_stats or {}
        groups = mr.groups or {}

        nodes = stats.get("nodes", 0)
        elements = stats.get("elements", 0)
        tets = stats.get("tetrahedra", 0)

        bbox = stats.get("bbox", {})
        if bbox:
            dx = bbox.get("xmax", 0) - bbox.get("xmin", 0)
            dy = bbox.get("ymax", 0) - bbox.get("ymin", 0)
            dz = bbox.get("zmax", 0) - bbox.get("zmin", 0)
            print(f"  Bounding box: {dx:.3f} x {dy:.3f} x {dz:.3f} um")  # noqa: T201

        print(f"  Nodes:     {nodes:,}")  # noqa: T201
        print(f"  Elements:  {elements:,}")  # noqa: T201
        if tets:
            print(f"  Tetrahedra: {tets:,}")  # noqa: T201
        if distortion := format_mesh_distortion(stats):
            print(f"  {distortion}")  # noqa: T201
        if field_dofs := format_field_dofs(stats):
            print(f"  {field_dofs}")  # noqa: T201

        dom_volumes = groups.get("volumes", {})
        bdr_conductors = groups.get("conductor_surfaces", {})
        bdr_interfaces = groups.get("interface_surfaces", {})
        print(f"  Domain groups:     {len(dom_volumes)}")  # noqa: T201
        print(f"  Conductor surfaces:{len(bdr_conductors)}")  # noqa: T201
        print(f"  Interface surfaces:{len(bdr_interfaces)}")  # noqa: T201

        # The domain regions hold the volume elements in 3D and the surface
        # elements in 2D meshes, where there are no volume groups.
        physical = stats.get("groups", {})
        regions = [g for g in physical.get("volumes", []) if g.get("elements")] or [
            g for g in physical.get("surfaces", []) if g.get("elements")
        ]
        region_total = sum(g["elements"] for g in regions)
        if region_total:
            print("  Elements by region:")  # noqa: T201
            for g in sorted(regions, key=lambda g: g["elements"], reverse=True):
                share = 100 * g["elements"] / region_total
                line = f"    {g['name']:<24}{g['elements']:>12,}  {share:5.1f}%"
                if edges := g.get("edge_length"):
                    line += f"  edges {edges['min']:.3g}-{edges['max']:.3g} um"
                print(line)  # noqa: T201

        if elements and not tets:
            p = 2
            nd_dofs_est = elements * p * (p + 1)
            h1_dofs_est = elements * (p + 1) * (p + 2) // 2
            print(f"  Est. ND-space DOFs (order {p}):  ~{nd_dofs_est:,}")  # noqa: T201
            print(f"  Est. H1-space DOFs (order {p}):  ~{h1_dofs_est:,}")  # noqa: T201
            print(f"  Est. total DOFs:                  ~{nd_dofs_est + h1_dofs_est:,}")  # noqa: T201

        for line in mesh_identity_lines(stats):
            print(f"  {line}")  # noqa: T201

    def _get_ports_for_preview(self, stack: LayerStack) -> list:
        """Get ports for preview."""
        from gsim.palace.ports import extract_ports

        component = self.geometry.component if self.geometry else None
        self._configure_ports_on_component(stack)
        return extract_ports(component, stack)

    # -------------------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------------------

    def preview(
        self,
        *,
        preset: Literal["coarse", "default", "fine"] | None = None,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        margin: float | None = None,
        margin_x: float | None = None,
        margin_y: float | None = None,
        z_above: float | None = None,
        z_below: float | None = None,
        airbox_margin: float | None = None,
        fmax: float | None = None,
        planar_conductors: bool | None = None,
        show_gui: bool = True,
        auto_size: bool = False,
        cells_per_feature: int = 2,
        curve_fit_mode: Literal["line", "spline", "bspline"] | None = None,
        curve_fit_layers: list[str] | None = None,
        curve_fit_tolerance_um: float | None = None,
        curve_fit_min_points: int | None = None,
        curve_fit_corner_angle_deg: float | None = None,
        high_order_elements: bool | None = None,
        high_order_order: int | None = None,
        high_order_optimize: bool | None = None,
        algorithm_3d: Literal["delaunay", "hxt"] | None = None,
        threads: int | None = None,
        surface_threads: int | None = None,
        decimate_tolerance: float | None = None,
    ) -> None:
        """Preview the mesh without running simulation.

        Opens the gmsh GUI to visualize the mesh interactively.

        Args:
            preset: Mesh quality preset ("coarse", "default", "fine")
            refined_mesh_size: Mesh size near conductors (um)
            max_mesh_size: Max mesh size in air/dielectric (um)
            margin: XY margin around design (um)
            margin_x: X-axis margin (um). Overrides margin for X.
            margin_y: Y-axis margin (um). Overrides margin for Y.
            z_above: Airbox extension above stack top (um).
            z_below: Airbox extension below stack bottom (um).
            airbox_margin: Deprecated. Use set_airbox().
            fmax: Max frequency for mesh sizing (Hz)
            planar_conductors: Treat conductors as 2D PEC surfaces
            show_gui: Show gmsh GUI for interactive preview
            auto_size: If True, scale refined_mesh_size down to the smallest
                conductor feature / cells_per_feature. Off by default.
                Explicit refined_mesh_size overrides bypass auto-sizing.
            cells_per_feature: Target cells across the smallest conductor
                feature for auto-sizing and resolution warnings. Default 2.
            curve_fit_mode: Patterned dielectric boundary mode (line/spline/bspline).
            curve_fit_layers: Layer names where spline fitting is applied.
            curve_fit_tolerance_um: Point merge tolerance before fitting.
            curve_fit_min_points: Min contour points required to fit curves.
            curve_fit_corner_angle_deg: Turn-angle threshold used to split
                sharp corners from smooth curve-fit segments.
            high_order_elements: Enable high-order geometric mesh elements.
            high_order_order: Polynomial order for high-order elements.
            high_order_optimize: Run gmsh high-order optimization after meshing.
            algorithm_3d: Gmsh 3D meshing algorithm, "delaunay" or "hxt".
            threads: Threads for 3D meshing (see ``MeshConfig``).
            surface_threads: Threads for 1D and 2D meshing. Above 1 the mesh
                differs from run to run.
            decimate_tolerance: Relative tolerance for polygon decimation
                (None = no decimation; typical 0.001-0.01).

        Example:
            >>> sim.preview(preset="fine", planar_conductors=True, show_gui=True)
        """
        from gsim.palace.mesh.generator import generate_mesh

        component = self.geometry.component if self.geometry else None

        # Validate configuration
        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        self._apply_airbox_overrides(
            margin_x=margin_x,
            margin_y=margin_y,
            z_above=z_above,
            z_below=z_below,
        )

        # Build mesh config
        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            margin_x=margin_x,
            margin_y=margin_y,
            airbox_margin=airbox_margin,
            fmax=fmax,
            planar_conductors=planar_conductors,
            show_gui=show_gui,
            auto_size=auto_size,
            cells_per_feature=cells_per_feature,
            curve_fit_mode=curve_fit_mode,
            curve_fit_layers=curve_fit_layers,
            curve_fit_tolerance_um=curve_fit_tolerance_um,
            curve_fit_min_points=curve_fit_min_points,
            curve_fit_corner_angle_deg=curve_fit_corner_angle_deg,
            high_order_elements=high_order_elements,
            high_order_order=high_order_order,
            high_order_optimize=high_order_optimize,
            algorithm_3d=algorithm_3d,
            threads=threads,
            surface_threads=surface_threads,
        )

        # Resolve stack
        stack = self._resolve_stack()
        self._warn_if_substrate_omitted(stack)

        # Get ports
        ports = self._get_ports_for_preview(stack)

        airbox_cfg = self._airbox_config or {}
        domain_margin_x, domain_margin_y = self._resolve_domain_margins(mesh_config)

        # Generate mesh in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_mesh(
                component=component,
                stack=stack,
                ports=ports,
                output_dir=tmpdir,
                refined_mesh_size=mesh_config.refined_mesh_size,
                max_mesh_size=mesh_config.max_mesh_size,
                margin_x=domain_margin_x,
                margin_y=domain_margin_y,
                air_margin=mesh_config.airbox_margin,
                airbox_margin_x=airbox_cfg.get("margin_x"),
                airbox_margin_y=airbox_cfg.get("margin_y"),
                airbox_z_above=airbox_cfg.get("z_above"),
                airbox_z_below=airbox_cfg.get("z_below"),
                airbox_material=airbox_cfg.get("material", "air"),
                fmax=mesh_config.fmax,
                show_gui=True,
                simulation_type=self.simulation_type,
                driven_config=self._get_problem_settings("driven"),
                eigenmode_config=self._get_problem_settings("eigenmode"),
                numerical_config=self.solver,
                boundary_mode_config=self._get_problem_settings("boundary_mode"),
                planar_conductors=mesh_config.planar_conductors,
                pec_blocks=self._pec_blocks or None,
                absorbing_boundary=self._outer_wall()[0],
                absorbing_order=self.absorbing_order,
                symmetry_plane=(
                    self._symmetry_planes[0] if self._symmetry_planes else None
                ),
                merge_via_distance=mesh_config.merge_via_distance,
                clip_merged_vias=mesh_config.clip_merged_vias,
                curve_fit_mode=mesh_config.curve_fit_mode,
                curve_fit_layers=mesh_config.curve_fit_layers,
                curve_fit_tolerance_um=mesh_config.curve_fit_tolerance_um,
                curve_fit_min_points=mesh_config.curve_fit_min_points,
                curve_fit_corner_angle_deg=mesh_config.curve_fit_corner_angle_deg,
                high_order_elements=mesh_config.high_order_elements,
                high_order_order=mesh_config.high_order_order,
                high_order_optimize=mesh_config.high_order_optimize,
                refinement_boxes=mesh_config.refinement_boxes,
                algorithm_3d=mesh_config.algorithm_3d,
                threads=mesh_config.threads,
                surface_threads=mesh_config.surface_threads,
                decimate_tolerance=decimate_tolerance,
            )

    # -------------------------------------------------------------------------
    # Mesh generation
    # -------------------------------------------------------------------------

    def mesh(
        self,
        *,
        preset: Literal["coarse", "default", "fine"] | None = None,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        margin: float | None = None,
        margin_x: float | None = None,
        margin_y: float | None = None,
        z_above: float | None = None,
        z_below: float | None = None,
        airbox_margin: float | None = None,
        fmax: float | None = None,
        planar_conductors: bool | None = None,
        show_gui: bool = False,
        model_name: str = "palace",
        verbose: bool = True,
        verbosity: int = 0,
        auto_size: bool = False,
        cells_per_feature: int = 2,
        periodic_axis: str | None = None,
        decimate_tolerance: float | None = None,
        merge_via_distance: float | None = None,
        clip_merged_vias: bool | None = None,
        curve_fit_mode: Literal["line", "spline", "bspline"] | None = None,
        curve_fit_layers: list[str] | None = None,
        curve_fit_tolerance_um: float | None = None,
        curve_fit_min_points: int | None = None,
        curve_fit_corner_angle_deg: float | None = None,
        high_order_elements: bool | None = None,
        high_order_order: int | None = None,
        high_order_optimize: bool | None = None,
        refinement_boxes: (
            list[tuple[float, float, float, float, float]] | None
        ) = None,
        algorithm_3d: Literal["delaunay", "hxt"] | None = None,
        threads: int | None = None,
        surface_threads: int | None = None,
    ) -> SimulationResult:
        """Generate the mesh for Palace simulation.

        Only generates the mesh file (palace.msh). Config is generated
        separately with write_config().

        Requires set_output_dir() to be called first.

        Args:
            preset: Mesh quality preset ("coarse", "default", "fine")
            refined_mesh_size: Mesh size near conductors (um), overrides preset
            max_mesh_size: Max mesh size in air/dielectric (um), overrides preset
            margin: XY margin around design (um), overrides preset
            margin_x: X-axis margin (um). Overrides margin for X.
            margin_y: Y-axis margin (um). Overrides margin for Y.
            z_above: Airbox extension above stack top (um).
            z_below: Airbox extension below stack bottom (um).
            airbox_margin: Deprecated. Use set_airbox().
            fmax: Max frequency for mesh sizing (Hz), overrides preset
            planar_conductors: Treat conductors as 2D PEC surfaces
            show_gui: Show gmsh GUI during meshing
            model_name: Base name for output files: ``<model_name>.msh`` and
                ``output/<model_name>``. A plain name, without path separators
            verbose: Print progress messages
            verbosity: Gmsh OCC verbosity level (0=silent, 1-99=increasing)
            auto_size: If True, scale refined_mesh_size down to the smallest
                conductor feature / cells_per_feature. Off by default so presets
                use their literal refined_mesh_size. Explicit refined_mesh_size
                overrides always bypass auto-sizing.
            cells_per_feature: Target cells across the smallest conductor
                feature for auto-sizing and resolution warnings. Default 2.
            periodic_axis: Optional periodic axis ("x" or "y") for periodic
                meshing constraints on opposite domain sides.
            decimate_tolerance: Relative tolerance for polygon decimation
                (None = no decimation; typical 0.001-0.01). Reduces vertex
                count on curved geometry before meshing.
            merge_via_distance: Max gap (um) between nearby via polygons to
                merge before extrusion. Pass 0 to disable merging (keep
                each via as a separate volume). Defaults to the preset's
                value (typically 2.0 um). Merging fills the gaps between
                the vias, so a dense via array can grow to several times
                its drawn area; a warning is logged when the area grows
                by more than 5%.
            clip_merged_vias: Clip each merged via polygon to the overlap of
                the conductor layers directly below and above its layer, so
                merging adds no via area outside the metal. The drawn cuts are
                always kept, and contacts on substrate or dielectric stay as
                drawn. Defaults to the value in ``sim.mesh_config`` (False). A
                slab that lies entirely between two continuous metals is left
                unchanged; ``merge_via_distance=0`` keeps every cut there.
            curve_fit_mode: Patterned dielectric boundary mode (line/spline/bspline).
            curve_fit_layers: Layer names where spline fitting is applied.
            curve_fit_tolerance_um: Point merge tolerance before fitting.
            curve_fit_min_points: Min contour points required to fit curves.
            curve_fit_corner_angle_deg: Turn-angle threshold used to split
                sharp corners from smooth curve-fit segments.
            high_order_elements: Enable high-order geometric mesh elements.
            high_order_order: Polynomial order for high-order elements.
            high_order_optimize: Run gmsh high-order optimization after meshing.
            refinement_boxes: Native-2D cross-section meshes only: boxes held
                to an element size, each ``(h_min, h_max, z_min, z_max, size)``
                in um. Refinement lines size the elements on them only; a box
                holds the size over an area.
            algorithm_3d: Gmsh 3D meshing algorithm, "delaunay" or "hxt".
            threads: Threads for 3D meshing (see ``MeshConfig``).
            surface_threads: Threads for 1D and 2D meshing. Above 1 the mesh
                differs from run to run.

        Returns:
            SimulationResult with mesh path

        Raises:
            ValueError: If output_dir not set, configuration is invalid, or
                ``model_name`` is not a plain file name

        Example:
            >>> sim.set_output_dir("./sim")
            >>> result = sim.mesh(preset="fine", planar_conductors=True)
            >>> print(f"Mesh saved to: {result.mesh_path}")
        """
        from gsim.palace.mesh.config_generator import check_model_name
        from gsim.palace.ports import extract_ports

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")
        check_model_name(model_name)

        component = self.geometry.component if self.geometry else None

        self._apply_airbox_overrides(
            margin_x=margin_x,
            margin_y=margin_y,
            z_above=z_above,
            z_below=z_below,
        )

        # Build mesh config
        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            margin_x=margin_x,
            margin_y=margin_y,
            airbox_margin=airbox_margin,
            fmax=fmax,
            planar_conductors=planar_conductors,
            show_gui=show_gui,
            auto_size=auto_size,
            cells_per_feature=cells_per_feature,
            curve_fit_mode=curve_fit_mode,
            curve_fit_layers=curve_fit_layers,
            curve_fit_tolerance_um=curve_fit_tolerance_um,
            curve_fit_min_points=curve_fit_min_points,
            curve_fit_corner_angle_deg=curve_fit_corner_angle_deg,
            high_order_elements=high_order_elements,
            high_order_order=high_order_order,
            high_order_optimize=high_order_optimize,
            refinement_boxes=refinement_boxes,
            algorithm_3d=algorithm_3d,
            threads=threads,
            surface_threads=surface_threads,
        )

        if merge_via_distance is not None:
            mesh_config.merge_via_distance = merge_via_distance
        if clip_merged_vias is not None:
            mesh_config.clip_merged_vias = clip_merged_vias

        # Validate configuration
        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        output_dir = self._output_dir

        # Resolve stack and configure ports
        stack = self._resolve_stack()
        if self.simulation_type == "boundarymode":
            palace_ports = []
        else:
            self._configure_ports_on_component(stack)

            # Extract ports
            palace_ports = extract_ports(component, stack)

        # Generate mesh (config is written separately by simulate() or write_config())
        result = self._generate_mesh_internal(
            output_dir=output_dir,
            mesh_config=mesh_config,
            ports=palace_ports,
            driven_config=self._get_problem_settings("driven"),
            model_name=model_name,
            verbose=verbose,
            write_config=False,
            periodic_axis=periodic_axis,
            decimate_tolerance=decimate_tolerance,
            gmsh_verbosity=verbosity,
        )

        # Without an airbox the absorbing boundary is applied straight to the
        # outer dielectric surface, often microns from the metal, where it acts
        # as a lossy sheet and fakes large insertion loss and reflection. Warn
        # here so it surfaces before any run is submitted.
        from gsim.palace.mesh.geometry import is_air_like_material

        groups = getattr(self._last_mesh_result, "groups", None) or {}
        volumes = groups.get("volumes", {})
        if (
            self._outer_wall()[0]
            and groups.get("boundary_surfaces", {}).get("absorbing")
        ) and not any(is_air_like_material(stack, name) for name in volumes):
            logger.warning(
                "Absorbing boundary is applied to solid dielectric surfaces "
                "(%s) because the model has no air region, which produces "
                "large spurious insertion loss and reflection. Add an airbox "
                "via set_airbox(margin_x=..., margin_y=..., z_above=..., "
                "z_below=...). Note that get_stack(air_above=...) is "
                "deprecated and ignored.",
                ", ".join(sorted(volumes)),
            )

        # Post-mesh summary: nodes, tets, refined / max sizes (in um).
        stats = result.mesh_stats or {}
        node_count = stats.get("nodes")
        tet_count = stats.get("tetrahedra")
        if distortion := format_mesh_distortion(stats):
            logger.info("%s", distortion)
        if field_dofs := format_field_dofs(stats):
            logger.info("%s", field_dofs)
        if node_count is not None and tet_count is not None:
            logger.info(
                "Mesh: %s nodes \u00b7 %s tets \u00b7 refined=%.3g \u00b5m \u00b7 "
                "max=%.3g \u00b5m",
                f"{node_count:,}",
                f"{tet_count:,}",
                mesh_config.refined_mesh_size,
                mesh_config.max_mesh_size,
            )

        return result

    def _outer_wall(self) -> tuple[bool, bool, bool]:
        """Resolve the outer wall as (absorbing, metallic, pec_by_default).

        An unset ``absorbing_boundary`` (``None``) is absorbing, except with
        wave ports, where the wall is PEC like the port eigenproblem's.
        """
        absorbing = self.absorbing_boundary
        metallic = bool(getattr(self, "metallic_boundaries", False))
        if absorbing is not None:
            return absorbing, metallic, False
        if metallic:
            return False, True, False
        if getattr(self, "wave_ports", None):
            return False, True, True
        return True, False, False

    def write_config(
        self,
        *,
        validate_mesh: bool = True,
        photonic: bool = False,
        photononic: bool | None = None,
    ) -> Path:
        """Write Palace config.json after mesh generation.

        By default this validates that the generated mesh/config include
        conductor boundaries required for EM conductor solves. For photonic
        simulations, pass ``photonic=True`` to skip this validation.

        Args:
            photonic: Skip conductor-oriented mesh validation when ``True``.
            photononic: Deprecated alias for ``photonic``.

        Returns:
            Path to the generated config.json

        Raises:
            ValueError: If mesh() hasn't been called yet

        Example:
            >>> result = sim.mesh("./sim")
            >>> config_path = sim.write_config()
            >>> # Photonic workflow without conductor boundaries
            >>> config_path = sim.write_config(photonic=True)
        """
        from gsim.palace.mesh.generator import write_config as gen_write_config

        if photononic is not None:
            logger.warning(
                "write_config(photononic=...) is deprecated; use photonic=..."
            )
            photonic = photonic or photononic

        if self._last_mesh_result is None:
            raise ValueError("No mesh result. Call mesh() first.")

        if not self._last_mesh_result.groups:
            raise ValueError(
                "Mesh result has no groups data. "
                "Was mesh() called with write_config=True already?"
            )

        stack = self._resolve_stack()
        electrostatic_config = self._get_problem_settings("electrostatic")
        terminals = getattr(self, "terminals", None)

        # Thread impedance boundary configs through hints
        hints = dict(self._hints)
        if self._impedance_boundaries:
            hints["_impedance_boundaries"] = self._impedance_boundaries
        absorbing, metallic, pec_default = self._outer_wall()
        if metallic:
            hints["_metallic_boundaries"] = True
        if pec_default and not self._pec_default_noted:
            self._pec_default_noted = True
            logger.info(
                "Wave ports: the outer wall is PEC, matching the port "
                "eigenproblem. Set absorbing_boundary=True to keep an "
                "absorbing wall."
            )
        if getattr(self, "conductor_boundary", "conductivity") == "pec":
            hints["_conductor_boundary"] = "pec"

        # BoundaryMode voltage/impedance postprocessing paths. They do not
        # affect the 2D eigenproblem; the boundary-mode sim declares them.
        mode_postprocessing = getattr(self, "mode_postprocessing", None)
        if mode_postprocessing is not None:
            postprocessing = mode_postprocessing()
            if postprocessing:
                hints["_mode_postprocessing"] = postprocessing

        # The 2D mesh of a cross-section has its own axes; the config
        # generator writes material tensors in them (_to_section_frame).
        cross_section = getattr(self, "cross_section", None)
        if self.simulation_type == "boundarymode" and cross_section is not None:
            hints["_cross_section_axis"] = cross_section.axis

        config_path = gen_write_config(
            mesh_result=self._last_mesh_result,
            stack=stack,
            ports=self._last_ports,
            simulation_type=self.simulation_type,
            eigenmode_config=self._get_problem_settings("eigenmode"),
            driven_config=self._get_problem_settings("driven"),
            numerical_config=self.solver,
            refinement_config=self.refinement,
            boundary_mode_config=self._get_problem_settings("boundary_mode"),
            absorbing_boundary=absorbing,
            absorbing_order=self.absorbing_order,
            hints=hints,
            electrostatic_config=electrostatic_config,
            terminals=terminals or [],
            resolved_terminals=self._config_terminals(self._last_mesh_result.groups),
        )

        # Validate mesh and config unless this is a photonic workflow.
        if not photonic and validate_mesh:
            validation = self.validate_mesh()
            if not validation.valid:
                raise ValueError(f"Mesh validation failed:\n{validation}")

        return config_path

    # -------------------------------------------------------------------------
    # Cloud: fine-grained control
    # -------------------------------------------------------------------------

    def _prepare_upload_dir(self) -> Path:
        """Prepare a temp directory with all config/mesh files for upload.

        Ensures ``_output_dir`` is set, ``config.json`` exists, and copies
        everything to a fresh temp directory, so neither an upload nor its
        failure touches the user's directory.

        Returns:
            Path to temp directory ready for upload.
        """
        import shutil

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        # Always (re)generate config.json to reflect current driven settings
        self.write_config()

        # Copy input files to a temp dir so we don't destroy the user's directory
        tmp = Path(tempfile.mkdtemp(prefix="palace_"))
        for item in self._output_dir.iterdir():
            dest = tmp / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        return tmp

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
    ) -> SParams | PalaceTextResults | dict[str, Path]:
        """Run simulation locally using Palace.

        Requires mesh() and write_config() to be called first.
        Supports both Apptainer and direct Palace installation.

        Args:
            palace_sif_path: Path to Palace Apptainer SIF file.
                Only used when ``use_apptainer=True`` and no executable is
                selected.
                If None, tries PALACE_SIF, then local defaults:
                ``./Palace.sif``, ``./palace.sif``,
                ``./bin/Palace.sif``, ``./bin/palace.sif``.
            palace_executable: Path to Palace executable.
                If provided, runs Palace directly and overrides
                ``use_apptainer``.
                If None, tries PALACE_EXECUTABLE, then ``./bin/palace``,
                then ``palace`` on PATH.
            use_apptainer: If True (default), run via Apptainer using SIF file.
                If False, run Palace executable directly.
                Ignored when ``palace_executable`` is explicitly provided.
            num_processes: Number of MPI processes. If None (default), a
                problem-size-aware default is chosen: 2D mode-analysis runs
                use a single MPI rank (SuperLU_DIST's 2D processor grid does
                not scale on small 2D problems and can effectively hang at
                8-16 ranks), while 3D runs are capped to a few ranks by the
                estimated unknown count. An explicit value is always respected.
            num_threads: Number of OpenMP threads to use for OpenMP builds.
                When running with a single MPI rank and ``num_threads`` is
                omitted, defaults to the number of physical CPU cores so
                shared-memory parallelism is used.
            verbose: Print progress messages and stream Palace output in real time
            remedy: A second way out of a dead Palace runtime, named at the
                end of the abort report beyond pointing ``PALACE_BIN``
                somewhere else. A caller with no other way out gives none.

        Returns:
            Parsed Palace result object (``SParams``) when ``port-S.csv`` is
            present, matching :meth:`run` behavior. Falls back to parsed
            ``PalaceTextResults`` for text-only outputs (for example,
            BoundaryMode), then to a ``dict[str, Path]`` mapping filenames to
            local paths when no parser applies.

        Raises:
            ValueError: If output_dir not set or Palace not configured
            FileNotFoundError: If mesh, config, or Palace not found
            RuntimeError: If the simulation fails. A binary that died on a
                signal is reported through
                :func:`gsim.palace.runtime.local_abort_report`, which says
                whether the runtime or the model is at fault.

        Example:
            >>> # Using Apptainer (default)
            >>> import os
            >>> os.environ["PALACE_SIF"] = "/path/to/Palace.sif"
            >>> results = sim.run_local()
            >>>
            >>> # Using Apptainer with explicit path
            >>> results = sim.run_local(palace_sif_path="/path/to/Palace.sif")
            >>>
            >>> # Auto-discovery with no options:
            >>> # prefers local ./bin/palace if present,
            >>> # otherwise falls back to local SIF or PATH executable
            >>> results = sim.run_local()
            >>>
            >>> # Using direct Palace with custom executable path
            >>> results = sim.run_local(palace_executable="/usr/local/bin/palace")
            >>> # Using a locally built Palace binary
            >>> results = sim.run_local(palace_executable="./bin/palace")
            >>> # For DrivenSim: `results` is SParams -> results.s21.db
            >>> # For eigen / electrostatic: `results` is dict[str, Path]
        """
        import subprocess

        from gsim.palace.runtime import local_abort_report, resolve_launch, run_palace

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        output_dir = Path(self._output_dir)
        config_path = output_dir / "config.json"
        mesh_path = output_dir / f"{self._model_name()}.msh"
        postpro_dir = self._postpro_dir()

        # Choose safe parallel defaults. SuperLU_DIST's 2D processor grid does
        # not scale on small problems (2D mode analysis at 8-16 ranks hangs),
        # so 2D runs use a single MPI rank + OpenMP threads, and 3D direct
        # solves are capped by problem size.
        mesh_stats = {}
        if self._last_mesh_result is not None:
            mesh_stats = self._last_mesh_result.mesh_stats or {}

        if num_processes is None:
            num_processes, num_threads = _recommend_parallel(
                mesh_stats,
                self.simulation_type,
                None,
                num_threads,
            )
            if verbose:
                logger.info(
                    "Parallel config: %d MPI rank(s)%s",
                    num_processes,
                    f", {num_threads} OpenMP thread(s)" if num_threads else "",
                )
        else:
            # Explicit request: keep it, but warn when it exceeds what the
            # sparse-direct solver can use efficiently on this problem.
            rec_processes, _rec_threads = _recommend_parallel(
                mesh_stats,
                self.simulation_type,
                None,
                None,
            )
            if num_processes > rec_processes:
                logger.warning(
                    "Requested %d MPI ranks but SuperLU_DIST direct solve "
                    "does not scale beyond %d rank(s) on this problem size; "
                    "the factorization may be pathologically slow. Pass "
                    "num_processes=%d or use OpenMP threads instead.",
                    num_processes,
                    rec_processes,
                    rec_processes,
                )
            if num_processes == 1 and num_threads is None:
                num_threads = _count_physical_cpus()

        # Check required files exist
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. Call write_config() first."
            )

        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Mesh file not found: {mesh_path}. Call mesh() first."
            )

        launch = resolve_launch(
            executable=palace_executable,
            sif=palace_sif_path,
            use_apptainer=use_apptainer,
            search_roots=[Path.cwd(), output_dir, output_dir.parent],
            verbose=verbose,
        )

        def _emit_info(msg: str) -> None:
            if logger.isEnabledFor(logging.INFO):
                logger.info("%s", msg)
            else:
                logger.warning("%s", msg)

        try:
            run_palace(
                launch,
                cwd=output_dir,
                num_processes=num_processes,
                num_threads=num_threads,
                verbose=verbose,
                emit=_emit_info,
            )
        except subprocess.CalledProcessError as failure:
            raise RuntimeError(
                local_abort_report(
                    failure,
                    binary=launch.binary,
                    output_dir=output_dir,
                    wrote_output=_wrote_solver_output(postpro_dir),
                    during=self._run_context(),
                    remedy=remedy,
                )
            ) from failure

        if verbose:
            _emit_info("Simulation completed successfully")
            _emit_info(f"Results saved to {postpro_dir}")

        files = {
            file.name: file
            for file in postpro_dir.iterdir()
            if file.is_file() and not file.name.startswith(".")
        }

        # Match cloud run() behavior for Palace: return parsed SParams when
        # possible, else fall back to raw files dict.
        from gsim.palace.results import load_sparams, load_text_results

        try:
            return load_sparams(files)
        except FileNotFoundError:
            try:
                return load_text_results(files)
            except FileNotFoundError:
                return files

    # -------------------------------------------------------------------------
    # Port methods
    # -------------------------------------------------------------------------

    def add_port(
        self,
        name: str,
        *,
        layer: str | None = None,
        from_layer: str | None = None,
        to_layer: str | None = None,
        length: float | None = None,
        offset: float = 0.0,
        impedance: float = 50.0,
        resistance: float | None = None,
        inductance: float | None = None,
        capacitance: float | None = None,
        excited: bool = True,
        geometry: Literal["inplane", "gap", "interlayer", "via"] = "inplane",
    ) -> None:
        """Add a single-element lumped port.

        Args:
            name: Port name (must match component port name)
            layer: Target layer for inplane ports
            from_layer: First conductor layer for interlayer ports
            to_layer: Second conductor layer for interlayer ports
            length: Port extent along direction (um)
            offset: Shift the port inward along the waveguide (um).
                Positive moves away from the boundary, into the conductor.
            impedance: Port impedance (Ohms)
            resistance: Series resistance (Ohms)
            inductance: Series inductance (H)
            capacitance: Shunt capacitance (F)
            excited: Whether this port is excited
            geometry: ``"inplane"``, ``"gap"``, or ``"interlayer"``. Gap ports
                are vertical sheets
                with GDS width spanning the gap along orientation, centered in
                the gap and extending through the conductor layer thickness.
                Gap ports currently require cardinal orientations; omit length.
                ``"interlayer"`` creates a Z-directed sheet between two layers.
                ``"via"`` is a deprecated alias for ``"interlayer"``. Use
                :meth:`add_cpw_port` for the ``"cpw"`` lumped-port geometry.

        Example:
            >>> sim.add_port("o1", layer="topmetal2", length=5.0)
            >>> sim.add_port(
            ...     "feed",
            ...     from_layer="metal1",
            ...     to_layer="topmetal2",
            ...     geometry="interlayer",
            ... )
        """
        # Remove existing config for this port if any
        self.ports = [p for p in self.ports if p.name != name]

        self.ports.append(
            PortConfig(
                name=name,
                layer=layer,
                from_layer=from_layer,
                to_layer=to_layer,
                length=length,
                offset=offset,
                impedance=impedance,
                resistance=resistance,
                inductance=inductance,
                capacitance=capacitance,
                excited=excited,
                geometry=geometry,
            )
        )

    def add_cpw_port(
        self,
        name: str,
        *,
        layer: str,
        s_width: float,
        gap_width: float,
        length: float = 2.0,
        offset: float | None = None,
        impedance: float = 50.0,
        excited: bool = True,
    ) -> None:
        """Add a coplanar waveguide (CPW) port.

        CPW ports consist of two elements (upper and lower gaps) that are
        excited with opposite E-field directions to create the CPW mode.

        Place a single gdsfactory port at the center of the signal conductor.
        The two gap element surfaces are computed from s_width and gap_width.

        Args:
            name: Port name (must match a component port at the signal center)
            layer: Target conductor layer (e.g., "topmetal2")
            s_width: Width of the signal (center) conductor (um)
            gap_width: Width of each gap between signal and ground (um)
            length: Port extent along direction (um)
            offset: Shift the port inward along the waveguide (um).
                Positive moves away from the boundary, into the conductor.
                Defaults to length/2 (port flush with conductor edge).
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited

        Example:
            >>> sim.add_cpw_port(
            ...     "left", layer="topmetal2", s_width=20, gap_width=15, length=5.0
            ... )
        """
        # Remove existing CPW port with same name if any
        self.cpw_ports = [p for p in self.cpw_ports if p.name != name]

        self.cpw_ports.append(
            CPWPortConfig(
                name=name,
                layer=layer,
                s_width=s_width,
                gap_width=gap_width,
                length=length,
                offset=offset,
                impedance=impedance,
                excited=excited,
            )
        )

    def add_two_terminal_port(
        self,
        plus_port: str,
        minus_port: str,
        *,
        layer: str,
        impedance: float = 50.0,
        excited: bool = True,
    ) -> None:
        """Add a two-terminal lumped port for true 1-port S11 simulation.

        Combines two GDS ports into a single Palace LumpedPort with two EDGE-geometry
        elements (vertical surfaces spanning the conductor thickness). No artificial
        reference layer is needed — P1 is the excitation and P2 is the reference,
        both on the same metal layer.

        Args:
            plus_port: GDS port name for the + (excitation) terminal
            minus_port: GDS port name for the - (reference) terminal
            layer: Conductor layer containing both terminals (e.g., "metal1")
            impedance: Port impedance in Ohms (default: 50)
            excited: Whether this port is excited (default: True)

        Example:
            >>> sim.add_two_terminal_port("P1", "P2", layer="metal1", excited=True)
        """
        self.two_terminal_ports = [
            p
            for p in self.two_terminal_ports
            if p.plus_port != plus_port and p.minus_port != minus_port
        ]
        self.two_terminal_ports.append(
            TwoTerminalPortConfig(
                plus_port=plus_port,
                minus_port=minus_port,
                layer=layer,
                impedance=impedance,
                excited=excited,
            )
        )

    def add_wave_port(
        self,
        name: str,
        *,
        layer: str | None = None,
        z_margin: float = 0.0,
        lateral_margin: float = 0.0,
        max_size: bool = False,
        mode: int = 1,
        excited: bool = True,
        offset: float = 0.0,
        eigensolver_type: Literal["Default", "SLEPc", "ARPACK"] | None = None,
        eigensolver_tol: float | None = None,
        eigensolver_ksp_tol: float | None = None,
        eigensolver_max_size: int | None = None,
        eigensolver_verbose: int | None = None,
    ) -> None:
        """Add a single element wave port.

        Args:
            name: Port name (must match a component port at the signal center)
            layer: Target conductor layer (e.g., "topmetal2")
            z_margin: Margin in z direction
            lateral_margin: Margin in x/y directions
              Ignores lateral margin and port_width.
            max_size: When True, automatically set z_margin and lateral_margin
                to fill the full simulation domain boundary on that side.
                Overrides z_margin and lateral_margin values.
            mode: Mode number to excite.
            excited: Whether this port is excited
            offset: Offset distance used for scattering parameter de-embedding.
            eigensolver_type: Palace SolverType for this port's 2D mode
                eigenproblem ("Default", "SLEPc" or "ARPACK"). None uses
                Palace's own default.
            eigensolver_tol: Palace EigenTol for this port's mode solve.
                None uses Palace's own default.
            eigensolver_ksp_tol: Palace KSPTol for this port's mode solve.
                None uses Palace's own default.
            eigensolver_max_size: Palace MaxSize (eigensolver subspace
                dimension) for this port's mode solve - unrelated to
                max_size above, which sizes geometry. None lets Palace
                pick its own default.
            eigensolver_verbose: Palace Verbose level for this port's
                mode solve. None uses Palace's own default.

        Example:
            >>> sim.add_wave_port(
            ...     "w1",
            ...     layer="topmetal2",
            ...     max_size=True,
            ...     mode=1,
            ...     excited=True,
            ...     offset=0.0,
            ... )
        """
        self.wave_ports = [p for p in self.wave_ports if p.name != name]

        self.wave_ports.append(
            WavePortConfig(
                name=name,
                layer=layer,
                z_margin=z_margin,
                lateral_margin=lateral_margin,
                max_size=max_size,
                mode=mode,
                excited=excited,
                offset=offset,
                eigensolver_type=eigensolver_type,
                eigensolver_tol=eigensolver_tol,
                eigensolver_ksp_tol=eigensolver_ksp_tol,
                eigensolver_max_size=eigensolver_max_size,
                eigensolver_verbose=eigensolver_verbose,
            )
        )

    def add_multimode_wave_port(
        self,
        name: str,
        *,
        modes: Sequence[int],
        layer: str | None = None,
        active_mode: int = 1,
        excite: Literal["active", "all"] = "active",
        pilot_paths: Mapping[int, Sequence[Sequence[float]]] | None = None,
        z_margin: float = 0.0,
        lateral_margin: float = 0.0,
        max_size: bool = False,
        excited: bool = True,
        offset: float = 0.0,
        eigensolver_type: Literal["Default", "SLEPc", "ARPACK"] | None = None,
        eigensolver_tol: float | None = None,
        eigensolver_ksp_tol: float | None = None,
        eigensolver_max_size: int | None = None,
        eigensolver_verbose: int | None = None,
    ) -> None:
        """Add a wave port that follows several modes on one port face.

        The port keeps one boundary surface. Palace gets one ``WavePort``
        entry per mode on that surface: the entry of ``active_mode`` keeps the
        port's own index and terminates the face, and every other mode is an
        observation-only entry (``"Active": false``) with an index after all
        the ports, so the active entry always has the lowest index of its
        face, which Palace requires. Every entry solves its own 2D port
        mode, and every entry is measured, so ``port-S.csv`` gets a column per
        entry and excitation. The map from Palace index to (port, mode,
        active) is written next to ``config.json`` as ``port_modes.json``.

        Only the active entry loads the face. A wave that arrives in another
        mode is not matched there, so the S-parameters of the observed modes
        carry a reflection error of the order of the relative difference of
        the wave numbers. The error is zero for the active mode. To get every
        mode with the active-mode accuracy, run once per mode with that mode
        as ``active_mode`` and use only the columns of the active entries.

        Args:
            name: Port name (must match a component port at the signal center).
            modes: The Palace mode numbers (1-based) to follow on this face,
                for example ``(1, 2, 3)``. Must contain ``active_mode``.
            layer: Target conductor layer.
            active_mode: The mode that terminates the face.
            excite: ``"active"`` excites only the active entry (one excitation
                per port); ``"all"`` excites every entry, which gives a column
                per mode but excites the unmatched modes with an amplitude
                that is off by the same mismatch.
            pilot_paths: Optional ``{mode: points}``. Each path is an open
                polyline of ``(x, y, z)`` points in um across the face, from
                the signal (high-potential) conductor to the reference, and
                becomes that entry's ``VoltagePath``: Palace flips the mode so
                the line integral of ``E`` along it is positive, which pins the
                sign of the mode, and writes the voltage to ``port-V.csv``.
            z_margin: Margin in z direction.
            lateral_margin: Margin in x/y directions.
            max_size: Fill the full simulation domain boundary on that side.
            excited: Whether the port is excited at all.
            offset: Offset distance used for scattering parameter de-embedding.
            eigensolver_type: Palace SolverType of the 2D mode eigenproblem.
            eigensolver_tol: Palace EigenTol of the mode solve.
            eigensolver_ksp_tol: Palace KSPTol of the mode solve.
            eigensolver_max_size: Palace MaxSize of the mode solve.
            eigensolver_verbose: Palace Verbose level of the mode solve.

        Raises:
            ValueError: If this is not a driven simulation, or the modes,
                active mode or pilot paths are inconsistent.

        Example:
            >>> sim.add_multimode_wave_port(
            ...     "o1",
            ...     layer="metal3",
            ...     modes=(1, 2, 3),
            ...     active_mode=1,
            ...     max_size=True,
            ... )
        """
        if self.simulation_type != "driven":
            raise ValueError(
                "A multimode wave port is a driven-simulation feature, not "
                f"{self.simulation_type!r}."
            )
        paths = (
            {
                int(mode): [[float(c) for c in point] for point in path]
                for mode, path in pilot_paths.items()
            }
            if pilot_paths
            else None
        )
        config = WavePortConfig(
            name=name,
            layer=layer,
            z_margin=z_margin,
            lateral_margin=lateral_margin,
            max_size=max_size,
            mode=active_mode,
            modes=tuple(int(m) for m in modes),
            active_mode=active_mode,
            excite_modes=excite,
            pilot_paths=paths,
            excited=excited,
            offset=offset,
            eigensolver_type=eigensolver_type,
            eigensolver_tol=eigensolver_tol,
            eigensolver_ksp_tol=eigensolver_ksp_tol,
            eigensolver_max_size=eigensolver_max_size,
            eigensolver_verbose=eigensolver_verbose,
        )
        self.wave_ports = [p for p in self.wave_ports if p.name != name]
        self.wave_ports.append(config)


class _PalaceSimBase(PalaceSimMixin, BaseModel):
    """The state every Palace simulation holds, declared once.

    Each problem class adds only what varies: its ``simulation_type`` and
    ``absorbing_boundary`` defaults, its ``solver`` type, and its ports or
    terminals.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Composed objects (from common)
    geometry: Geometry | None = None
    stack: LayerStack | None = None
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    # Defaults set by each problem class
    simulation_type: Literal["driven", "eigenmode", "electrostatic", "boundarymode"]
    absorbing_boundary: bool | None
    absorbing_order: Literal[1, 2] = Field(
        default=2,
        description=(
            "Order of the absorbing boundary condition on the outer walls "
            "(Palace ``Boundaries.Absorbing.Order``). Palace releases without "
            "awslabs/palace#1070 apply the second-order term with the wrong "
            "sign, which makes it reflect more than first order for oblique "
            "outgoing waves. Prefer 1 for radiating structures until the "
            "Palace in use includes that fix."
        ),
    )
    mesh_config: MeshConfig = Field(default_factory=MeshConfig.default)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _airbox_config: dict[str, Any] = PrivateAttr(default_factory=dict)
    _pec_blocks: list = PrivateAttr(default_factory=list)
    _symmetry_planes: list[SymmetryPlaneConfig] = PrivateAttr(default_factory=list)
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)
    _impedance_boundaries: list[ImpedanceBoundaryConfig] = PrivateAttr(
        default_factory=list
    )

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_ports: bool = PrivateAttr(default=False)
    _pec_default_noted: bool = PrivateAttr(default=False)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)
