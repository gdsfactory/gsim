"""Electrostatic simulation class for capacitance extraction.

This module provides the ElectrostaticSim class for extracting
capacitance matrices between terminals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.capacitance import CapacitanceMatrices, load_capacitance
from gsim.palace.models import (
    ElectrostaticConfig,
    ElectrostaticSolverConfig,
    MaterialConfig,
    RefinementConfig,
    TerminalConfig,
    WavePortConfig,
)
from gsim.palace.models.solver import warn_legacy_solver_setting

if TYPE_CHECKING:
    from gsim.palace.mesh.nets import Nets
    from gsim.palace.terminals import ResolvedTerminal

logger = logging.getLogger(__name__)


class ElectrostaticSim(PalaceSimMixin, BaseModel):
    """Electrostatic simulation for capacitance matrix extraction.

    This class configures and runs electrostatic simulations to extract
    the capacitance matrix between conductor terminals. Unlike driven
    and eigenmode simulations, this does not use ports.

    Example:
        >>> from gsim.palace import ElectrostaticSim
        >>>
        >>> sim = ElectrostaticSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack()
        >>> sim.set_airbox(margin_x=120.0, margin_above=120.0, margin_below=20.0)
        >>> sim.add_terminal("T1", net="PLUS")  # two electrodes, even on one layer
        >>> sim.add_terminal("T2", net="MINUS")
        >>> sim.set_output_dir("./sim")
        >>> sim.mesh(preset="default")
        >>> results = sim.run()  # dict[str, Path]
        >>> print(results["terminal-C.csv"])

    Attributes:
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        terminals: List of terminal configurations
        electrostatic: Electrostatic simulation configuration
        materials: Material property overrides
        solver: Grouped numerical and problem-specific solver configuration
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    simulation_type: Literal["electrostatic"] = "electrostatic"

    ports: None = None
    cpw_ports: None = None
    two_terminal_ports: None = None
    wave_ports: list[WavePortConfig] = Field(default_factory=list)
    # Composed objects (from common)
    geometry: Geometry | None = None
    stack: LayerStack | None = None

    # Terminal configurations (no ports in electrostatic)
    terminals: list[TerminalConfig] = Field(default_factory=list)
    # What conductors that are in no terminal do: ground, or float
    unassigned_conductors: Literal["ground", "floating"] = "ground"

    absorbing_boundary: bool = False

    # Material overrides and solver config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    solver: ElectrostaticSolverConfig = Field(default_factory=ElectrostaticSolverConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _airbox_config: dict[str, Any] = PrivateAttr(default_factory=dict)
    _pec_blocks: list = PrivateAttr(default_factory=list)
    _symmetry_planes: list = PrivateAttr(default_factory=list)
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)
    _impedance_boundaries: list = PrivateAttr(default_factory=list)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_terminals: bool = PrivateAttr(default=False)
    # Terminals resolved to nets by the last write_config (net mode only)
    _resolved_terminals: list[Any] | None = PrivateAttr(default=None)

    # Legacy solver settings

    @property
    def electrostatic(self) -> ElectrostaticConfig:
        """Deprecated alias for solver.electrostatic, retaining the original type."""
        warn_legacy_solver_setting("sim.electrostatic", "sim.solver.electrostatic")
        return self.solver.electrostatic

    @electrostatic.setter
    def electrostatic(self, value: ElectrostaticConfig | dict[str, Any]) -> None:
        """Replace electrostatic settings through their deprecated top-level name."""
        warn_legacy_solver_setting("sim.electrostatic", "sim.solver.electrostatic")
        self._set_problem_settings("electrostatic", value)

    # -------------------------------------------------------------------------
    # Terminal methods
    # -------------------------------------------------------------------------

    def add_terminal(
        self,
        name: str,
        *,
        layer: str | None = None,
        net: str | None = None,
        region: (
            tuple[float, float, float, float] | list[tuple[float, float]] | None
        ) = None,
        floating: bool = False,
    ) -> None:
        """Add a terminal for capacitance extraction.

        A terminal selects conductors in one of three ways; give at least one
        of ``layer``, ``net`` and ``region``.

        - ``layer`` alone selects every conductor shape on that layer. When no
          terminal uses ``net``, ``region`` or ``floating``, this is the whole
          behaviour (layer mode), and two electrodes on one layer are one
          terminal.
        - ``net`` selects a connected conductor, metal and vias on all the
          layers it spans, by the name in :meth:`nets`: the names of the ports
          on it joined by ``+``, or ``net<k>``, or the name of one port on it.
        - ``region`` selects every connected conductor with a shape in a box
          or polygon, optionally only on ``layer``. It always selects whole
          nets.

        Once any terminal uses nets, ``layer`` alone means every net with a
        shape on that layer, including the metal that vias join to it on other
        layers. Check the selection with :meth:`resolve_terminals`.

        Args:
            name: Terminal name
            layer: Target conductor layer. With ``region`` it limits the shapes
                tested against the region.
            net: Name of a net or of a port on it; not combined with
                ``region`` or ``layer``.
            region: Box ``(xmin, ymin, xmax, ymax)`` or list of at least three
                ``(x, y)`` points of a polygon, in um.
            floating: Solve the conductor but leave it undriven: it is
                eliminated from the capacitance matrix by
                :meth:`load_capacitance`.

        Example:
            >>> sim.add_terminal("TOP", net="PLUS")
            >>> sim.add_terminal("LEFT", region=(0, 0, 20, 20), layer="m1")
            >>> sim.add_terminal("SHIELD", net="net3", floating=True)
            >>> sim.add_terminal("T1", layer="metal5")  # the whole layer
        """
        # Remove existing terminal with same name
        self.terminals = [t for t in self.terminals if t.name != name]
        self.terminals.append(
            TerminalConfig(
                name=name,
                layer=layer,
                net=net,
                region=region,
                floating=floating,
            )
        )

    def set_unassigned_conductors(self, mode: Literal["ground", "floating"]) -> None:
        """Choose what happens to conductors that are in no terminal.

        Args:
            mode: ``"ground"`` (default) puts them at zero potential. With
                ``"floating"`` each one becomes an extra terminal named after
                its net, solved and eliminated by :meth:`load_capacitance`.
                PEC blocks always stay grounded.

        Example:
            >>> sim.set_unassigned_conductors("floating")
        """
        self.unassigned_conductors = mode

    def _mesh_by_net(self) -> bool:
        """Whether the mesh must carry one physical group per net.

        True when any terminal uses ``net``, ``region`` or ``floating``, or when
        the conductors in no terminal float. Layer-only simulations return
        False and take the layer path.
        """
        return self.unassigned_conductors == "floating" or any(
            t.net is not None or t.region is not None or t.floating
            for t in self.terminals
        )

    def _config_terminals(
        self, groups: dict[str, Any]
    ) -> list[ResolvedTerminal] | None:
        """The terminals resolved to nets for the config, when the mesh has nets.

        Checks that the nets of the geometry are still the ones of the mesh,
        keeps the resolved terminals for :meth:`load_capacitance` and writes
        them to ``terminals.json`` in the output directory. A layer-only
        simulation gets None and no ``terminals.json``.

        Raises:
            ValueError: If the mesh and the terminals do not fit each other:
                the mode (layers or nets) changed since ``mesh()``, or the nets
                differ from those of the mesh.
        """
        from gsim.palace.terminals import resolve_terminals

        mesh_nets = groups.get("nets")
        by_net = self._mesh_by_net()
        out_dir = Path(self._require_mesh_result().output_dir)
        path = out_dir / "terminals.json"
        if mesh_nets is None:
            if by_net:
                raise ValueError(
                    "The terminals select nets (net=, region=, floating or "
                    "floating unassigned conductors), but the mesh has one "
                    "physical group per layer. Run mesh() again."
                )
            path.unlink(missing_ok=True)  # left by an earlier net-mode mesh
            self._resolved_terminals = None
            return None
        if not by_net:
            raise ValueError(
                "The mesh has one physical group per net, but the terminals now "
                "select whole layers. Run mesh() again."
            )

        nets = self.nets()
        same = len(nets) == len(mesh_nets) and all(
            net.name == entry["name"] and list(net.layers) == entry["layers"]
            for net, entry in zip(nets, mesh_nets, strict=True)
        )
        if not same:
            raise ValueError(
                "The nets of the geometry differ from the nets of the mesh "
                f"({len(nets)} against {len(mesh_nets)}, or other names or "
                "layers). Run mesh() again (decimate_tolerance can change nets)."
            )
        resolved = resolve_terminals(
            self.terminals, nets, unassigned=self.unassigned_conductors
        )
        self._resolved_terminals = resolved
        path.write_text(
            json.dumps(
                [
                    {
                        "index": index,
                        "name": terminal.name,
                        "nets": list(terminal.nets),
                        "net_names": list(terminal.net_names),
                        "floating": terminal.floating,
                    }
                    for index, terminal in enumerate(resolved, start=1)
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        return resolved

    def nets(self) -> Nets:
        """Find the electrically connected conductor shapes of the geometry.

        Metal that touches on one layer, and metal on different layers that a
        via joins, is one net; disconnected electrodes stay separate even when
        they share a layer. Electrical ports name the nets they sit on.

        Terminals select nets by these names: ``add_terminal(name, net=...)``
        takes the name above or the name of one of its ports. A terminal given
        only ``layer=`` still takes every shape on its layer when no terminal
        uses nets, so two disconnected electrodes on one layer are then a single
        terminal. This shows how many electrodes each layer holds.

        Returns:
            The nets, in the order their first shape appears in the layout.

        Raises:
            ValueError: If no geometry has been set.

        Example:
            >>> print(sim.nets())
        """
        from gsim.palace.mesh.geometry import extract_geometry
        from gsim.palace.mesh.nets import extract_nets

        component = self.component
        if component is None:
            msg = "No component set. Call set_geometry(component) first."
            raise ValueError(msg)
        stack = self._resolve_stack()
        ports = [port for port in component.ports if port.port_type == "electrical"]
        return extract_nets(extract_geometry(component, stack), stack, ports)

    def resolve_terminals(self) -> list[ResolvedTerminal]:
        """Show which nets each terminal drives, without meshing.

        Applies the terminal selections to :meth:`nets`. Use it to check a
        configuration before meshing; ``validate_config()`` and ``mesh()`` run
        the same check when a terminal uses nets.

        Returns:
            The terminals in the order they were added, then one floating
            terminal per unassigned net when ``unassigned_conductors`` is
            ``"floating"``. Each lists the names and positions of its nets.

        Raises:
            ValueError: If no geometry has been set, a net or layer is
                unknown, a selection matches no net, a net is in two terminals,
                or every terminal is floating.

        Example:
            >>> for terminal in sim.resolve_terminals():
            ...     print(terminal.name, terminal.net_names, terminal.floating)
        """
        from gsim.palace.terminals import resolve_terminals

        return resolve_terminals(
            self.terminals, self.nets(), unassigned=self.unassigned_conductors
        )

    # -------------------------------------------------------------------------
    # Electrostatic configuration
    # -------------------------------------------------------------------------

    def set_electrostatic(
        self,
        *,
        save_fields: int = 0,
    ) -> None:
        """Configure electrostatic simulation.

        Args:
            save_fields: Number of field solutions to save

        Example:
            >>> sim.set_electrostatic(save_fields=1)
        """
        self.solver.electrostatic = ElectrostaticConfig(
            save_fields=save_fields,
        )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    def _terminal_labels(self) -> tuple[list[str], list[str]]:
        """Names of the Palace terminals in index order, and the floating ones.

        In net mode these are the resolved terminals of the last
        ``write_config()``, or ``terminals.json`` in the output directory when
        this object has not written the config; they include the automatic
        floating terminals. Otherwise they are the layer terminals as added.

        Raises:
            ValueError: If the terminals select nets and neither the resolved
                terminals nor ``terminals.json`` are available.
        """
        if not self._mesh_by_net():
            return [terminal.name for terminal in self.terminals], []
        if self._resolved_terminals is not None:
            return (
                [t.name for t in self._resolved_terminals],
                [t.name for t in self._resolved_terminals if t.floating],
            )
        path = self._output_dir / "terminals.json" if self._output_dir else None
        if path is None or not path.exists():
            msg = (
                "The terminals select nets, so their order in Palace follows "
                "the nets; call write_config() (or run()) first, or set the "
                "output directory that holds terminals.json."
            )
            raise ValueError(msg)
        entries = json.loads(path.read_text(encoding="utf-8"))
        return (
            [entry["name"] for entry in entries],
            [entry["name"] for entry in entries if entry["floating"]],
        )

    def load_capacitance(
        self,
        results: dict[str, Path] | str | Path,
        *,
        eliminate_floating: bool = True,
    ) -> CapacitanceMatrices:
        """Load the capacitance matrices of a run, labelled with the terminals.

        The terminal names are the ones given to :meth:`add_terminal`, in the
        order they were added, which is Palace's terminal order. When
        ``unassigned_conductors`` is ``"floating"`` the unassigned nets follow,
        named after the net.

        Palace has no floating terminal, so floating conductors are solved as
        terminals and removed here with a Schur complement (see
        :meth:`~gsim.palace.capacitance.CapacitanceMatrices.eliminate`).

        Args:
            results: The results dict that ``run()`` returns, or the output
                directory of the simulation.
            eliminate_floating: Remove the floating terminals from the
                matrices. Pass False to see every solved terminal.

        Returns:
            The Maxwell and mutual capacitance matrices, with checks; see
            :class:`~gsim.palace.capacitance.CapacitanceMatrices`.

        Example:
            >>> cap = sim.load_capacitance(sim.run())
            >>> cap.between("T1", "T2")  # plate to plate, in F
            >>> cap.to_ground("T1")  # to the substrate, in F
            >>> cap.pairs()  # every capacitance as a table
            >>> assert not cap.problems()
        """
        names, floating = self._terminal_labels()
        matrices = load_capacitance(results, terminal_names=names)
        if eliminate_floating and floating:
            matrices = matrices.eliminate(floating)
        return matrices


__all__ = ["ElectrostaticSim"]
