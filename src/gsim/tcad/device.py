"""The DEVSIM device behind a :class:`gsim.tcad.Simulation`.

Builds the device from the shared native-2D mesh and the validated
concerns, and solves it bias point by bias point. The solve uses DEVSIM's
prebuilt Scharfetter-Gummel drift-diffusion physics
(``devsim.python_packages.simple_physics``). Doping enters as ``Donors`` /
``Acceptors`` node solutions evaluated from the analytic profiles in
:mod:`gsim.tcad.doping`, combined into the ``NetDoping`` node model.

DEVSIM keeps meshes, devices and its circuit in one process-wide
namespace; this module owns gsim's share of it.
"""

from __future__ import annotations

import itertools
import math
import sys
import warnings
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.constants import epsilon_0 as EPS0  # noqa: N812

from gsim.tcad.doping import DopingProfile, acceptor_donor_concentrations
from gsim.tcad.mesh import UM_TO_CM, line_group_points
from gsim.tcad.results import BiasPoint, CarrierMap
from gsim.tcad.runtime import import_simple_physics, require_devsim
from gsim.tcad.silicon import intrinsic_density_cm3

if TYPE_CHECKING:
    from gsim.tcad.api import Contact, Interface, Materials, Solver, TCADMesh

# DEVSIM keeps meshes and devices in one process-wide namespace, so every
# sim (and every re-mesh) claims its own names: two Studies, or one Stage
# re-run after a configuration change, would otherwise collide on the
# second setup with "a mesh already exists with name ...".
_DEVSIM_NAME_IDS = itertools.count()


#: Vacuum permittivity in DEVSIM's units (F/cm).
VACUUM_PERMITTIVITY_F_PER_CM: float = EPS0 * 1e-2

#: Rounding of node coordinates (cm) when matching a mesh node to a DEVSIM
#: region node: 1e-10 cm is a picometre, far below any element.
_NODE_KEY_CM: float = 1e-10

#: DEVSIM material name given to the insulating regions.
INSULATOR_MATERIAL: str = "Oxide"

#: Mesh groups that name a Region an electrode can be: a meshed volume, or
#: a conductor the mesh pipeline turned into a boundary.
_ELECTRODE_GROUPS: tuple[str, ...] = ("volumes", "conductor_surfaces", "pec_surfaces")


def _unique_devsim_name(kind: str) -> str:
    """Return a process-unique DEVSIM mesh or device name."""
    return f"gsim_tcad_{kind}_{next(_DEVSIM_NAME_IDS)}"


#: DEVSIM devices this process created and has not deleted. DEVSIM solves
#: every registered device in one Newton, so a device left behind by a
#: finished sim would take part in the next sim's solve.
_LIVE_DEVICES: set[str] = set()


def _release_gsim_devices(devsim: Any) -> None:
    """Delete the DEVSIM devices gsim created and still holds."""
    for name in list(_LIVE_DEVICES):
        with suppress(Exception):
            devsim.delete_device(device=name)
        _LIVE_DEVICES.discard(name)


@dataclass(frozen=True)
class DeviceLayout:
    """The device a mesh and the declarations describe, checked.

    Attributes:
        regions: The doped Regions, in the order the profiles name them.
        insulators: Insulating Region to relative permittivity.
        contact_regions: Contact name to the doped Region it injects into.
        interface_regions: Interface name to its two doped Regions.
        insulator_interfaces: Shared-curve group name to the (doped or
            insulating, insulating) Region pair it joins.
    """

    regions: list[str]
    insulators: dict[str, float]
    contact_regions: dict[str, str]
    interface_regions: dict[str, tuple[str, str]]
    insulator_interfaces: dict[str, tuple[str, str]]


def check_layout(
    groups: dict[str, Any],
    *,
    doping: list[DopingProfile],
    insulators: dict[str, float],
    contacts: list[Contact],
    interfaces: list[Interface],
) -> DeviceLayout:
    """Check the declarations against the mesh and against each other.

    A Region is doped or insulating, never both. A Contact joins a doped
    Region (its ``semiconductor``) to a Region that is neither (its
    ``electrode``); an Interface joins two doped Regions.

    Args:
        groups: Physical groups of the mesh.
        doping: The doping profiles.
        insulators: Insulating Region to relative permittivity.
        contacts: The declared contacts.
        interfaces: The declared interfaces.

    Returns:
        The checked layout.

    Raises:
        ValueError: On the first declaration that the mesh or the rules
            refuse, naming it.
    """
    if not doping:
        raise ValueError(
            "No doping profiles declared. Call sim.doping.add() with at least "
            "one profile."
        )
    if not contacts:
        raise ValueError(
            "Charge transport requires at least one contact. Call "
            "sim.contacts.add(name=..., semiconductor=..., electrode=...)."
        )

    volumes = set(groups.get("volumes", {}))
    electrodes = {name for key in _ELECTRODE_GROUPS for name in groups.get(key, {})}
    contact_lines = set(groups.get("contact_lines", {}))
    interface_lines = set(groups.get("interface_lines", {}))

    regions = list(dict.fromkeys(profile.region for profile in doping))
    unknown = [r for r in regions if r not in volumes]
    if unknown:
        raise ValueError(
            f"Doping regions {unknown} are not volume groups on the mesh. "
            f"Available regions: {sorted(volumes)}"
        )

    insulating = list(insulators)
    missing = [r for r in insulating if r not in volumes]
    if missing:
        raise ValueError(
            f"Insulating regions {missing} are not volume groups on the mesh. "
            f"Available regions: {sorted(volumes)}"
        )
    both = [r for r in insulating if r in regions]
    if both:
        raise ValueError(
            f"Regions {both} are declared insulating and carry a doping "
            "profile; a Region is one or the other."
        )

    contact_regions: dict[str, str] = {}
    for contact in contacts:
        if contact.name not in contact_lines:
            raise ValueError(
                f"Contact '{contact.name}' has no line group on the mesh: "
                f"'{contact.semiconductor}' and '{contact.electrode}' share no "
                "boundary."
            )
        if contact.semiconductor in insulating:
            raise ValueError(
                f"Contact '{contact.name}': semiconductor "
                f"'{contact.semiconductor}' is declared insulating."
            )
        if contact.semiconductor not in regions:
            raise ValueError(
                f"Contact '{contact.name}': semiconductor "
                f"'{contact.semiconductor}' carries no doping profile. Doped "
                f"regions: {regions}"
            )
        if contact.electrode in regions:
            raise ValueError(
                f"Contact '{contact.name}': electrode '{contact.electrode}' "
                "carries a doping profile. The electrode is the metal side of "
                "the contact."
            )
        if contact.electrode in insulating:
            raise ValueError(
                f"Contact '{contact.name}' lies on the boundary between "
                f"'{contact.semiconductor}' and '{contact.electrode}', and "
                f"'{contact.electrode}' is declared insulating: one curve "
                "cannot be both a Contact and a semiconductor-insulator "
                "Interface. Declare the Contact against its electrode, or "
                "leave the insulator out."
            )
        if contact.electrode not in electrodes:
            raise ValueError(
                f"Contact '{contact.name}': electrode '{contact.electrode}' is "
                f"not a region of the mesh. Available: {sorted(electrodes)}"
            )
        contact_regions[contact.name] = contact.semiconductor

    interface_regions: dict[str, tuple[str, str]] = {}
    for interface in interfaces:
        if interface.name not in interface_lines:
            raise ValueError(
                f"Interface '{interface.name}' has no line group on the mesh: "
                f"'{interface.region_a}' and '{interface.region_b}' share no "
                "boundary."
            )
        if interface.region_a not in regions or interface.region_b not in regions:
            raise ValueError(
                f"Interface '{interface.name}' must join two doped regions; got "
                f"'{interface.region_a}' and '{interface.region_b}' with doped "
                f"regions {regions}."
            )
        interface_regions[interface.name] = (interface.region_a, interface.region_b)

    # The mesh pipeline tags the curve every two touching Regions share
    # ("interface_<a>_<b>"). Two touching doped Regions with no declared
    # Interface would be two devices that never exchange a carrier, and
    # the capacitance between the contacts collapses to nothing.
    shared_curves = set(groups.get("interface_surfaces", {}))
    declared = {frozenset(pair) for pair in interface_regions.values()}
    for index, region_a in enumerate(regions):
        for region_b in regions[index + 1 :]:
            touching = {
                f"interface_{region_a}_{region_b}",
                f"interface_{region_b}_{region_a}",
            } & shared_curves
            if touching and frozenset((region_a, region_b)) not in declared:
                raise ValueError(
                    f"Doped regions '{region_a}' and '{region_b}' touch but no "
                    "Interface joins them, so no carrier crosses between them. "
                    f"Declare it: sim.interfaces.add(name=..., region_a="
                    f"'{region_a}', region_b='{region_b}')."
                )

    # The Interfaces an insulator shares with a doped Region or with
    # another insulator are the pairwise shared-curve groups the mesh
    # pipeline tags on its own.
    insulator_interfaces: dict[str, tuple[str, str]] = {}
    for index, insulator in enumerate(insulating):
        for other in [*regions, *insulating[:index]]:
            for name in (
                f"interface_{other}_{insulator}",
                f"interface_{insulator}_{other}",
            ):
                if name in shared_curves:
                    insulator_interfaces[name] = (other, insulator)

    return DeviceLayout(
        regions=regions,
        insulators=dict(insulators),
        contact_regions=contact_regions,
        interface_regions=interface_regions,
        insulator_interfaces=insulator_interfaces,
    )


@dataclass
class Device:
    """One DEVSIM device: its setup, its bias state and its solve.

    Built on the first solve and dropped whenever the mesh or a device
    input changes; never shared between simulations.
    """

    mesh: TCADMesh
    materials: Materials
    doping: list[DopingProfile]
    contacts: list[Contact]
    interfaces: list[Interface]
    solver: Solver
    layout: DeviceLayout
    name: str | None = None
    mesh_name: str | None = None
    dd_initialized: bool = False
    current_bias: dict[str, float] = field(default_factory=dict)

    _CONTINUITY_EQUATIONS: ClassVar[dict[str, str]] = {
        "Potential": "PotentialEquation",
        "Electrons": "ElectronContinuityEquation",
        "Holes": "HoleContinuityEquation",
    }

    # ------------------------------------------------------------------
    # Setup and release
    # ------------------------------------------------------------------

    @property
    def deleted(self) -> bool:
        """Whether the device was set up and has since left DEVSIM.

        Another simulation claiming DEVSIM's namespace deletes it.
        """
        return self.name is not None and self.name not in _LIVE_DEVICES

    def release(self) -> None:
        """Best-effort removal of this device and its DEVSIM mesh."""
        if self.name is None:
            return
        # Read the module already in sys.modules rather than importing it:
        # a device only exists because DEVSIM was imported to create it,
        # and a second import in a process that has dropped the module
        # re-runs DEVSIM's one-shot initialisation, which then raises.
        devsim: Any = sys.modules.get("devsim")
        if devsim is not None:
            with suppress(Exception):
                devsim.delete_device(device=self.name)
            if self.mesh_name is not None:
                with suppress(Exception):
                    devsim.delete_mesh(mesh=self.mesh_name)
        _LIVE_DEVICES.discard(self.name)
        self.name = None
        self.mesh_name = None
        self.dd_initialized = False
        self.current_bias = {}

    def setup(self, name: str | None = None) -> str:
        """Create the DEVSIM device from the shared mesh.

        Loads the cm-scaled mesh, registers one DEVSIM region per doped
        and insulating mesh region and one contact per declared contact,
        applies the doping node models, and sets up the potential-only
        physics.

        DEVSIM's meshes, devices and circuits live in one process-global
        namespace, and it refuses new circuit nodes once an analysis has
        run. The device therefore claims the circuit by default
        (``solver.claim_devsim_namespace``), which drops the contact
        sources of any DEVSIM device built earlier in the process. One
        charge-transport device is live at a time; meshes and devices
        themselves get process-unique names.

        Args:
            name: DEVSIM device name; a process-unique one is generated
                when omitted.

        Returns:
            The device name.
        """
        layout = self.layout
        device = name if name is not None else _unique_devsim_name("device")
        devsim = require_devsim()
        sp = import_simple_physics()

        if self.solver.claim_devsim_namespace:
            # The circuit is process-global and refuses new nodes after an
            # analysis, and every registered device joins the next solve;
            # a previous charge device's circuit and device have to go.
            with suppress(Exception):
                devsim.delete_circuit()
            _release_gsim_devices(devsim)

        # 128-bit float assembly (see Solver.extended_precision). The
        # parameters are process-wide, so they are set either way: an
        # earlier device's choice must not carry over.
        extended = self.solver.extended_precision
        for name in ("extended_solver", "extended_model", "extended_equation"):
            devsim.set_parameter(name=name, value=extended)

        mesh_name = _unique_devsim_name("mesh")
        self.mesh_name = mesh_name
        devsim.create_gmsh_mesh(mesh=mesh_name, file=str(self.mesh.devsim_mesh_path))
        for region in layout.regions:
            devsim.add_gmsh_region(
                mesh=mesh_name,
                gmsh_name=region,
                region=region,
                material=self.materials.semiconductor,
            )
        for region in layout.insulators:
            devsim.add_gmsh_region(
                mesh=mesh_name,
                gmsh_name=region,
                region=region,
                material=INSULATOR_MATERIAL,
            )
        for contact in self.contacts:
            devsim.add_gmsh_contact(
                mesh=mesh_name,
                gmsh_name=contact.name,
                region=layout.contact_regions[contact.name],
                material="metal",
                name=contact.name,
            )
        for interface, (region_a, region_b) in layout.interface_regions.items():
            devsim.add_gmsh_interface(
                mesh=mesh_name,
                gmsh_name=interface,
                region0=region_a,
                region1=region_b,
                name=interface,
            )
        devsim.finalize_mesh(mesh=mesh_name)
        devsim.create_device(mesh=mesh_name, device=device)
        self.name = device
        _LIVE_DEVICES.add(device)
        try:
            bound = self._bind_insulator_interfaces(devsim, device)
            for region in layout.regions:
                self._apply_doping(devsim, device, region)
                sp.SetSiliconParameters(device, region, self.materials.temperature_k)
                # simple_physics keeps n_i at its 300 K value whatever the
                # temperature; the SRH trap densities n1 = p1 = n_i follow.
                n_i = intrinsic_density_cm3(self.materials.temperature_k)
                for name in ("n_i", "n1", "p1"):
                    devsim.set_parameter(
                        device=device, region=region, name=name, value=n_i
                    )
                sp.CreateSiliconPotentialOnly(device, region)
            for region, permittivity in layout.insulators.items():
                # Potential only, for good: the drift-diffusion stage never
                # reaches these Regions, so they hold no carriers.
                devsim.set_parameter(
                    device=device,
                    region=region,
                    name="Permittivity",
                    value=permittivity * VACUUM_PERMITTIVITY_F_PER_CM,
                )
                sp.CreateOxidePotentialOnly(device, region)
            for contact in self.contacts:
                # Each contact is driven through a circuit voltage source so the
                # small-signal AC solve can read the terminal admittance (the
                # C(V) extraction) and the DC current from the circuit node.
                devsim.circuit_element(
                    name=self._source_name(contact.name),
                    n1=sp.GetContactBiasName(contact.name),
                    n2=0,
                    value=0.0,
                    acreal=0.0,
                    acimag=0.0,
                )
                sp.CreateSiliconPotentialOnlyContact(
                    device, layout.contact_regions[contact.name], contact.name, True
                )
                self.current_bias[contact.name] = 0.0
            for interface in (*layout.interface_regions, *bound):
                self._interface_continuity(devsim, sp, device, interface, "Potential")
        except BaseException:
            # A device set up halfway would be taken for a finished one by
            # the next solve.
            self.release()
            raise
        self.dd_initialized = False
        return device

    def _apply_doping(self, devsim: Any, device: str, region: str) -> None:
        """Evaluate the region's profiles onto DEVSIM node solutions."""
        x_cm = np.asarray(
            devsim.get_node_model_values(device=device, region=region, name="x"),
            dtype=np.float64,
        )
        y_cm = np.asarray(
            devsim.get_node_model_values(device=device, region=region, name="y"),
            dtype=np.float64,
        )
        profiles = [p for p in self.doping if p.region == region]
        acceptors, donors = acceptor_donor_concentrations(
            profiles, x_cm / UM_TO_CM, y_cm / UM_TO_CM
        )
        for name, values in (("Acceptors", acceptors), ("Donors", donors)):
            devsim.node_solution(device=device, region=region, name=name)
            devsim.set_node_values(
                device=device, region=region, name=name, values=list(values)
            )
        devsim.node_model(
            device=device,
            region=region,
            name="NetDoping",
            equation="Donors - Acceptors",
        )
        # The mobilities are node values rather than a DEVSIM expression so
        # that the transport solve and the RF conductivity evaluate one
        # model, not two transcriptions of it. They depend on the doping
        # alone, so the current's derivatives are unchanged.
        impurity = acceptors + donors
        mobility = self.materials.mobility
        for name, values in (
            ("ElectronMobility", mobility.electrons_cm2(impurity)),
            ("HoleMobility", mobility.holes_cm2(impurity)),
        ):
            devsim.node_solution(device=device, region=region, name=name)
            devsim.set_node_values(
                device=device, region=region, name=name, values=list(values)
            )
            devsim.edge_average_model(
                device=device,
                region=region,
                node_model=name,
                edge_model=f"{name}Edge",
                average_type="arithmetic",
            )

    def _bind_insulator_interfaces(self, devsim: Any, device: str) -> list[str]:
        """Create the insulator Interfaces node by node; return their names.

        The mesh already tags the curves an insulator shares with each
        neighbour, but DEVSIM cannot take them whole: a node that carries
        two Interfaces — the end of every Interface between doped Regions,
        where the insulator meets both — assembles neither correctly, and
        the device then drives a current through itself at zero bias. So
        each Interface is built from its node pairs
        (``create_interface_from_nodes``), leaving out the nodes that
        already carry a Contact, a declared Interface or an earlier
        insulator Interface. The insulator's own node at such a point
        stays an ordinary interior node of the insulator, tied to the
        silicon through its neighbours one element away.
        """
        layout = self.layout
        if not layout.insulator_interfaces:
            return []
        mesh_path = self.mesh.devsim_mesh_path

        def keys(points_cm: NDArray[np.float64]) -> list[tuple[int, int]]:
            """Coordinates as hashable keys, rounded far below any element."""
            return [
                (round(float(x) / _NODE_KEY_CM), round(float(y) / _NODE_KEY_CM))
                for x, y in points_cm
            ]

        def region_nodes(region: str) -> dict[tuple[int, int], int]:
            """DEVSIM's node index of *region* at each coordinate key."""
            coords = np.column_stack(
                [
                    np.asarray(
                        devsim.get_node_model_values(
                            device=device, region=region, name=axis
                        ),
                        dtype=np.float64,
                    )
                    for axis in ("x", "y")
                ]
            )
            return {key: i for i, key in enumerate(keys(coords))}

        nodes = {
            region: region_nodes(region)
            for pair in layout.insulator_interfaces.values()
            for region in pair
        }
        taken = {
            key
            for name in (
                *(contact.name for contact in self.contacts),
                *layout.interface_regions,
            )
            for key in keys(line_group_points(mesh_path, name))
        }

        bound: list[str] = []
        unmatched: dict[str, int] = {}
        for name, (other, insulator) in layout.insulator_interfaces.items():
            free = [
                key
                for key in keys(line_group_points(mesh_path, name))
                if key not in taken
            ]
            shared = [
                key for key in free if key in nodes[other] and key in nodes[insulator]
            ]
            if len(shared) < len(free):
                unmatched[name] = len(free) - len(shared)
            if not shared:
                continue
            taken.update(shared)
            devsim.create_interface_from_nodes(
                device=device,
                name=name,
                region0=other,
                region1=insulator,
                nodes0=[nodes[other][key] for key in shared],
                nodes1=[nodes[insulator][key] for key in shared],
            )
            bound.append(name)
        if unmatched:
            # The pairs are found by coordinate; a node found on one side
            # only is a point where the potential is not continuous.
            warnings.warn(
                "Insulator Interface nodes of the mesh match no node of their "
                "two Regions and are left untied, so the potential is not "
                "continuous there: "
                + ", ".join(f"{count} on {name!r}" for name, count in unmatched.items())
                + ".",
                stacklevel=2,
            )
        return bound

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _solve_dc(self, devsim: Any) -> None:
        """Run one DC Newton solve with the configured tolerances."""
        devsim.solve(
            type="dc",
            absolute_error=self.solver.absolute_error,
            relative_error=self.solver.relative_error,
            maximum_iterations=self.solver.max_iterations,
        )

    @staticmethod
    def _create_solution(sp: Any, device: str, region: str, name: str) -> None:
        """simple_physics re-exports CreateSolution in most DEVSIM versions."""
        create = getattr(sp, "CreateSolution", None)
        if create is None:  # pragma: no cover - version-dependent fallback
            import importlib

            create = importlib.import_module(
                "devsim.python_packages.model_create"
            ).CreateSolution
        create(device, region, name)

    def _interface_continuity(
        self, devsim: Any, sp: Any, device: str, interface: str, variable: str
    ) -> None:
        """Enforce continuity of *variable* across a region-region interface."""
        equation = self._CONTINUITY_EQUATIONS[variable]
        create = getattr(sp, "CreateContinuousInterfaceModel", None)
        if create is not None:
            model_name = create(device, interface, variable)
        else:  # pragma: no cover - version-dependent fallback
            model_name = f"continuous{variable}"
            devsim.interface_model(
                device=device,
                interface=interface,
                name=model_name,
                equation=f"{variable}@r0 - {variable}@r1",
            )
        devsim.interface_equation(
            device=device,
            interface=interface,
            name=equation,
            interface_model=model_name,
            type="continuous",
        )

    def _initialize_drift_diffusion(self) -> None:
        """Initial potential-only solve, then switch on drift-diffusion."""
        if self.dd_initialized:
            return
        if self.name is None:
            self.setup()
        devsim = require_devsim()
        sp = import_simple_physics()
        device = self.name
        if device is None:  # pragma: no cover - setup() above set it
            raise RuntimeError("DEVSIM device setup did not complete.")

        self._solve_dc(devsim)

        for region in self.layout.regions:
            for carrier, intrinsic in (
                ("Electrons", "IntrinsicElectrons"),
                ("Holes", "IntrinsicHoles"),
            ):
                self._create_solution(sp, device, region, carrier)
                devsim.set_node_values(
                    device=device,
                    region=region,
                    name=carrier,
                    init_from=intrinsic,
                )
            sp.CreateSiliconDriftDiffusion(
                device, region, "ElectronMobilityEdge", "HoleMobilityEdge"
            )
        for contact in self.contacts:
            sp.CreateSiliconDriftDiffusionAtContact(
                device, self.layout.contact_regions[contact.name], contact.name, True
            )
        for interface in self.layout.interface_regions:
            self._interface_continuity(devsim, sp, device, interface, "Electrons")
            self._interface_continuity(devsim, sp, device, interface, "Holes")
        self._solve_dc(devsim)
        self.dd_initialized = True

    @staticmethod
    def _source_name(contact: str) -> str:
        """Circuit voltage-source name attached to a contact."""
        return f"V_{contact}"

    def _set_bias(self, contact: str, bias: float) -> None:
        """Ramp the contact bias to the target in ``bias_step_v`` steps."""
        devsim = require_devsim()
        start = self.current_bias.get(contact, 0.0)
        delta = bias - start
        n_steps = max(1, math.ceil(abs(delta) / self.solver.bias_step_v))
        for i in range(1, n_steps + 1):
            value = start + delta * i / n_steps
            devsim.circuit_alter(name=self._source_name(contact), value=value)
            self._solve_dc(devsim)
        self.current_bias[contact] = bias

    def _contact_current(self, devsim: Any, contact: str) -> float:
        """DC terminal current from the contact's circuit source (A/cm)."""
        return float(
            devsim.get_circuit_node_value(
                node=f"{self._source_name(contact)}.I", solution="dcop"
            )
        )

    def _contact_charge(self, devsim: Any, contact: str) -> float:
        """Contact charge from the potential equation in C per cm of depth."""
        return float(
            devsim.get_contact_charge(
                device=self.name, contact=contact, equation="PotentialEquation"
            )
        )

    def _collect_carriers(self, devsim: Any) -> CarrierMap:
        """Concatenate node fields across the device regions (coords in um)."""
        columns: dict[str, list[NDArray[np.float64]]] = {
            "x": [],
            "y": [],
            "Electrons": [],
            "Holes": [],
            "Potential": [],
            "NetDoping": [],
        }
        region_names: list[str] = []
        for region in self.layout.regions:
            n_nodes = 0
            for name, column in columns.items():
                values = np.asarray(
                    devsim.get_node_model_values(
                        device=self.name, region=region, name=name
                    ),
                    dtype=np.float64,
                )
                n_nodes = values.size
                column.append(values)
            region_names.extend([region] * n_nodes)
        return CarrierMap(
            x_um=np.asarray(np.concatenate(columns["x"]) / UM_TO_CM, dtype=np.float64),
            y_um=np.asarray(np.concatenate(columns["y"]) / UM_TO_CM, dtype=np.float64),
            region=region_names,
            electrons_cm3=np.concatenate(columns["Electrons"]),
            holes_cm3=np.concatenate(columns["Holes"]),
            potential_v=np.concatenate(columns["Potential"]),
            net_doping_cm3=np.concatenate(columns["NetDoping"]),
        )

    def solve_point(self, bias: float, contact: str) -> BiasPoint:
        """Solve one bias point on the swept contact.

        Returns:
            The solved :class:`BiasPoint`, with carrier maps, terminal
            currents, and the small-signal capacitance ``Im(I)/omega`` from
            the quasi-static AC solve.
        """
        self._initialize_drift_diffusion()
        devsim = require_devsim()

        # The other contacts go back to 0 V first: one swept earlier would
        # otherwise keep its bias.
        for other in self.contacts:
            if other.name != contact and self.current_bias.get(other.name, 0.0):
                self._set_bias(other.name, 0.0)
        self._set_bias(contact, bias)
        carriers = self._collect_carriers(devsim)
        currents = {
            other.name: self._contact_current(devsim, other.name)
            for other in self.contacts
        }
        charge = self._contact_charge(devsim, contact)

        # Small-signal AC solve at a quasi-static frequency: the terminal
        # admittance of the swept contact's unit-amplitude source gives
        # C = |Im(I)| / omega (the small-signal charge per volt). Only the
        # swept source carries AC amplitude; the others stay AC-grounded.
        for other in self.contacts:
            devsim.circuit_alter(
                name=self._source_name(other.name),
                param="acreal",
                value=1.0 if other.name == contact else 0.0,
            )
        freq = self.solver.small_signal_freq_hz
        node = f"{self._source_name(contact)}.I"
        devsim.solve(type="ac", frequency=freq)
        current_imag = float(
            devsim.get_circuit_node_value(node=node, solution="ssac_imag")
        )
        capacitance = abs(current_imag) / (2.0 * math.pi * freq)

        # A second AC solve, high enough that the capacitive current
        # dominates the junction's DC leakage (which owns Re(Y) at the
        # quasi-static frequency) yet far below the branch's RC rolloff:
        # this is the admittance the series-RC junction fit inverts. The
        # source drives the swept contact with unit AC amplitude, so its
        # circuit-node current is the terminal admittance Y = I / V — up
        # to sign: the circuit node reports the current through the
        # source, which flows out of the contact it drives.
        devsim.solve(type="ac", frequency=self.solver.junction_freq_hz)
        admittance = -complex(
            float(devsim.get_circuit_node_value(node=node, solution="ssac_real")),
            float(devsim.get_circuit_node_value(node=node, solution="ssac_imag")),
        )

        return BiasPoint(
            bias_v=bias,
            carriers=carriers,
            currents_a_per_cm=currents,
            charge_c_per_cm=charge,
            capacitance_f_per_cm=capacitance,
            admittance_s_per_cm=admittance,
            admittance_freq_hz=self.solver.junction_freq_hz,
        )


__all__ = [
    "INSULATOR_MATERIAL",
    "VACUUM_PERMITTIVITY_F_PER_CM",
    "Device",
    "DeviceLayout",
    "check_layout",
]
