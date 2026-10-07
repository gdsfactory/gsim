"""DEVSIM charge-transport simulation on the shared native-2D mesh.

``ChargeTransportSim`` follows the established gsim backend idiom
(``set_geometry`` / ``set_stack`` / ``set_cross_section``, then solve or
sweep). Meshing is delegated to the existing ``BoundaryModeSim`` native-2D
pipeline — the exact mesh Palace BoundaryMode and the femwell adapter see —
so there is no second meshing path. Contacts declared with ``add_contact``
become named dim-1 physical groups that DEVSIM binds boundary conditions to
via ``add_gmsh_contact``.

The solve uses DEVSIM's prebuilt Scharfetter-Gummel drift-diffusion physics
(``devsim.python_packages.simple_physics``). Doping enters as ``Donors`` /
``Acceptors`` node solutions evaluated from the analytic profiles in
:mod:`gsim.tcad.doping`, combined into the ``NetDoping`` node model.
"""

from __future__ import annotations

import itertools
import logging
import math
import sys
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from scipy.constants import epsilon_0 as EPS0  # noqa: N812

from gsim.common.carriers import MobilityModel
from gsim.palace.base import MeshSourceMixin
from gsim.tcad.doping import (
    DopingProfile,
    acceptor_donor_concentrations,
)
from gsim.tcad.mesh import UM_TO_CM, line_group_points, write_scaled_msh
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap
from gsim.tcad.runtime import devsim_output, import_simple_physics, require_devsim

logger = logging.getLogger(__name__)

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


class Insulator(BaseModel):
    """An insulating Region taking part in the electrostatic solve only.

    Attributes:
        region: Mesh region (volume group) name, e.g. ``"sio2"``.
        relative_permittivity: Static relative permittivity of the Region.
    """

    model_config = ConfigDict(frozen=True)

    region: str
    relative_permittivity: float = Field(gt=0.0)


class ChargeTransportSim(MeshSourceMixin, BaseModel):
    """Poisson + drift-diffusion simulation of a waveguide cross-section.

    Example:
        >>> sim = ChargeTransportSim()
        >>> sim.set_output_dir("./tcad-sim")
        >>> sim.set_geometry(component)
        >>> sim.set_stack(stack)
        >>> sim.set_cross_section("x=0", window=(-25.0, -15.0))
        >>> sim.add_contact(name="anode", layer_a="metal1", layer_b="p_rib")
        >>> sim.add_contact(name="cathode", layer_a="metal1", layer_b="n_rib")
        >>> sim.add_doping(
        ...     StepDoping(
        ...         region="p_rib", dopant_type="acceptor", concentration_cm3=1e18
        ...     )
        ... )
        >>> sim.mesh(preset="coarse")
        >>> result = sim.sweep([0.0, -0.5, -1.0], contact="cathode")
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    simulation_type: Literal["charge"] = "charge"

    #: Lattice temperature (K) for the silicon parameter set.
    temperature: float = Field(default=300.0, gt=0.0)
    #: DEVSIM material name assigned to the semiconductor regions.
    material: str = "Silicon"
    #: Analytic doping profiles (each names the mesh region it applies to).
    doping: list[DopingProfile] = Field(default_factory=list)
    #: Insulating Regions solved for the potential alone (see
    #: :meth:`add_insulator`); none by default, which keeps the
    #: electrostatics inside the doped semiconductor.
    insulators: list[Insulator] = Field(default_factory=list)
    #: Low-field mobility against the total doping, evaluated node by node.
    #: DEVSIM's own silicon defaults are two constants (400 / 200 cm^2/Vs)
    #: whatever the doping, which misstates the series resistance of a slab
    #: doped from 1e17 to 1e20 cm^-3.
    mobility: MobilityModel = Field(default_factory=MobilityModel.masetti_silicon)
    #: Voltage step used when ramping the swept contact between biases (V).
    bias_step_v: float = Field(default=0.1, gt=0.0)
    #: Frequency (Hz) of the small-signal AC solve extracting C(V); low
    #: enough to be quasi-static.
    small_signal_freq_hz: float = Field(default=1.0, gt=0.0)
    #: Frequency (Hz) of the second small-signal AC solve, the one whose
    #: complex admittance the series-RC junction fit reads. At the
    #: quasi-static frequency the real part of the branch impedance is
    #: the junction's DC leakage rather than the slab's series
    #: resistance; here the capacitive current dominates the leakage
    #: while still sitting far below the branch's own RC rolloff, so
    #: ``Re(1/Y)`` is the series resistance.
    junction_freq_hz: float = Field(default=1e9, gt=0.0)
    #: Newton absolute error target passed to ``devsim.solve``.
    absolute_error: float = Field(default=1e10, gt=0.0)
    #: Newton relative error target passed to ``devsim.solve``. The 2D
    #: drift-diffusion Newton typically floors around 1e-8 on coarse
    #: meshes; tighten per solve when the mesh supports it.
    relative_error: float = Field(default=1e-6, gt=0.0)
    #: Newton iteration cap passed to ``devsim.solve``.
    max_iterations: int = Field(default=100, gt=0)
    #: Use DEVSIM's 128-bit extended-precision assembly (recommended).
    extended_precision: bool = True
    #: Claim DEVSIM's process-global namespace when this sim sets up its
    #: device: drop the circuit (DEVSIM refuses new circuit nodes once an
    #: analysis has run) and delete the devices gsim created earlier
    #: (DEVSIM solves every registered device in one Newton). Foreign
    #: DEVSIM devices are never touched. Turn this off only when an
    #: earlier charge-transport device in the process must stay alive.
    claim_devsim_namespace: bool = True

    # Meshing is delegated to the shared native-2D BoundaryMode pipeline.
    _bsim: Any = PrivateAttr(default=None)
    _devsim_mesh_path: Path | None = PrivateAttr(default=None)
    _devsim_mesh_name: str | None = PrivateAttr(default=None)
    _device: str | None = PrivateAttr(default=None)
    _contact_regions: dict[str, str] = PrivateAttr(default_factory=dict)
    _interface_regions: dict[str, tuple[str, str]] = PrivateAttr(default_factory=dict)
    _insulator_interfaces: dict[str, tuple[str, str]] = PrivateAttr(
        default_factory=dict
    )
    _dd_initialized: bool = PrivateAttr(default=False)
    _current_bias: dict[str, float] = PrivateAttr(default_factory=dict)

    # ------------------------------------------------------------------
    # Delegated configuration (shared geometry seam)
    # ------------------------------------------------------------------

    def _mesh_source(self) -> Any:
        """Lazily create the internal BoundaryModeSim used for meshing."""
        if self._bsim is None:
            from gsim.palace import BoundaryModeSim

            self._bsim = BoundaryModeSim()
        return self._bsim

    def _mesh_source_or_none(self) -> Any | None:
        """The internal BoundaryModeSim, or None while nothing configured it."""
        return self._bsim

    def add_doping(self, profile: DopingProfile) -> None:
        """Add an analytic doping profile (see :mod:`gsim.tcad.doping`)."""
        self.doping = [*self.doping, profile]

    def add_insulator(self, *, region: str, relative_permittivity: float) -> None:
        """Include an insulating Region in the electrostatic solve.

        By default Poisson is solved in the doped semiconductor alone, its
        boundary with the surrounding insulator a zero-normal-field wall:
        every field line between the two sides of the Junction is forced
        through the depleted silicon. An insulator declared here joins the
        solve as a Region of its own carrying the potential and nothing
        else — no carriers, no doping, no mobility — with the potential
        continuous across every Interface it shares with a doped Region
        or another insulator, so the field may fringe around the Junction
        through it and the small-signal capacitance and admittance count
        that path. Those Interfaces are the shared curves the mesh
        pipeline already tags; nothing further is declared. The Carrier
        map is unchanged: it holds the doped Regions only.

        The insulator's outer boundary, the electrode faces it touches
        included, stays a zero-normal-field wall: a Contact is where
        metal meets a semiconductor, and the capacitance between the
        electrodes themselves belongs to the RF solve, not to this one.

        Args:
            region: Mesh region (volume group) name, e.g. ``"sio2"``.
            relative_permittivity: Static relative permittivity.
        """
        self.insulators = [
            *self.insulators,
            Insulator(region=region, relative_permittivity=relative_permittivity),
        ]

    @property
    def geometry(self) -> Any:
        """The delegated geometry object (or None)."""
        return self._bsim.geometry if self._bsim is not None else None

    @property
    def stack(self) -> Any:
        """The delegated layer stack (or None)."""
        return self._bsim.stack if self._bsim is not None else None

    @property
    def cross_section(self) -> Any:
        """The delegated cross-section plane and window (or None)."""
        return self._bsim.cross_section if self._bsim is not None else None

    @property
    def devsim_mesh_path(self) -> Path:
        """Path of the cm-scaled mesh copy loaded by DEVSIM, once meshed.

        Raises:
            ValueError: When read before ``mesh()`` has run.
        """
        if self._devsim_mesh_path is None:
            raise ValueError(
                "No DEVSIM mesh generated for this ChargeTransportSim. "
                "Call mesh() first."
            )
        return self._devsim_mesh_path

    # ------------------------------------------------------------------
    # Meshing
    # ------------------------------------------------------------------

    def mesh(self, **kwargs: Any) -> Any:
        """Generate the shared native-2D mesh and its cm-scaled DEVSIM copy.

        All keyword arguments are forwarded to ``BoundaryModeSim.mesh``
        (presets, mesh sizes, ...). Requires geometry, stack, cross-section
        and at least one contact to be configured.

        Returns:
            The mesh-generation result of the shared pipeline.
        """
        bsim = self._mesh_source()
        if not bsim.contact_specs:
            raise ValueError(
                "Charge transport requires at least one contact. "
                "Call add_contact(name=..., layer_a=..., layer_b=...) first."
            )
        result = bsim.mesh(**kwargs)

        output_dir = bsim.output_dir
        if output_dir is None:  # pragma: no cover - mesh() enforces this
            raise ValueError("Output directory not set.")
        self._devsim_mesh_path = write_scaled_msh(
            result.mesh_path, Path(output_dir) / "devsim.msh", scale=UM_TO_CM
        )
        # A new mesh invalidates any existing DEVSIM device.
        self.reset_device()
        self._devsim_mesh_name = _unique_devsim_name("mesh")
        return result

    # ------------------------------------------------------------------
    # DEVSIM device setup
    # ------------------------------------------------------------------

    def _release_devsim_device(self) -> None:
        """Best-effort removal of this sim's DEVSIM device and mesh."""
        if self._device is None:
            return
        # Read the module already in sys.modules rather than importing it:
        # a device only exists because DEVSIM was imported to create it,
        # and a second import in a process that has dropped the module
        # re-runs DEVSIM's one-shot initialisation, which then raises.
        devsim: Any = sys.modules.get("devsim")
        if devsim is not None:
            with suppress(Exception):
                devsim.delete_device(device=self._device)
            if self._devsim_mesh_name is not None:
                with suppress(Exception):
                    devsim.delete_mesh(mesh=self._devsim_mesh_name)
        _LIVE_DEVICES.discard(self._device)

    def reset_device(self) -> None:
        """Forget the DEVSIM device; setup runs again on the next solve."""
        self._release_devsim_device()
        self._devsim_mesh_name = None
        self._device = None
        self._dd_initialized = False
        self._contact_regions = {}
        self._interface_regions = {}
        self._insulator_interfaces = {}
        self._current_bias = {}

    def _device_regions(self) -> list[str]:
        """Ordered unique mesh regions named by the doping profiles."""
        regions: list[str] = []
        for profile in self.doping:
            if profile.region not in regions:
                regions.append(profile.region)
        return regions

    def _validate_setup(self) -> list[str]:
        """Check mesh/doping/contact consistency; return the device regions."""
        if self._devsim_mesh_path is None:
            raise ValueError("No mesh generated. Call mesh() first.")
        if not self.doping:
            raise ValueError(
                "No doping profiles declared. Call add_doping() with at "
                "least one profile."
            )

        groups = self.mesh_groups
        volumes = set(groups.get("volumes", {}))
        contact_lines = set(groups.get("contact_lines", {}))
        interface_lines = set(groups.get("interface_lines", {}))

        regions = self._device_regions()
        unknown = [r for r in regions if r not in volumes]
        if unknown:
            raise ValueError(
                f"Doping regions {unknown} are not volume groups on the "
                f"mesh. Available regions: {sorted(volumes)}"
            )

        insulating = [insulator.region for insulator in self.insulators]
        missing = [r for r in insulating if r not in volumes]
        if missing:
            raise ValueError(
                f"Insulating regions {missing} are not volume groups on the "
                f"mesh. Available regions: {sorted(volumes)}"
            )
        doped = [r for r in insulating if r in regions]
        if doped:
            raise ValueError(
                f"Regions {doped} are declared insulating and carry a doping "
                "profile; a Region is one or the other."
            )

        self._contact_regions = {}
        for spec in self.contact_specs:
            if spec.name not in contact_lines:
                raise ValueError(
                    f"Contact '{spec.name}' has no line group on the mesh. "
                    "Re-run mesh() after add_contact()."
                )
            sides = [s for s in (spec.layer_a, spec.layer_b) if s in regions]
            if not sides:
                raise ValueError(
                    f"Contact '{spec.name}' touches neither of the doped "
                    f"device regions {regions}: it connects "
                    f"'{spec.layer_a}' and '{spec.layer_b}'. Add a doping "
                    "profile for the semiconductor side of the contact."
                )
            walled = [s for s in (spec.layer_a, spec.layer_b) if s in insulating]
            if walled:
                raise ValueError(
                    f"Contact '{spec.name}' lies on the boundary between "
                    f"'{spec.layer_a}' and '{spec.layer_b}', and "
                    f"'{walled[0]}' is declared insulating: one curve cannot "
                    "be both a Contact and a semiconductor-insulator "
                    "Interface. Declare the Contact against its electrode, "
                    "or leave the insulator out."
                )
            self._contact_regions[spec.name] = sides[0]
        if not self._contact_regions:
            raise ValueError(
                "Charge transport requires at least one contact. "
                "Call add_contact() before mesh()."
            )

        self._interface_regions = {}
        for spec in self.interface_specs:
            if spec.name not in interface_lines:
                raise ValueError(
                    f"Interface '{spec.name}' has no line group on the mesh. "
                    "Re-run mesh() after add_interface()."
                )
            if spec.layer_a not in regions or spec.layer_b not in regions:
                raise ValueError(
                    f"Interface '{spec.name}' must join two doped device "
                    f"regions; got '{spec.layer_a}' and '{spec.layer_b}' "
                    f"with device regions {regions}."
                )
            self._interface_regions[spec.name] = (spec.layer_a, spec.layer_b)

        # The Interfaces an insulator shares with a doped Region or with
        # another insulator are the pairwise shared-curve groups the mesh
        # pipeline tags on its own ("interface_<a>_<b>").
        shared_curves = set(groups.get("interface_surfaces", {}))
        self._insulator_interfaces = {}
        for index, insulator in enumerate(insulating):
            for other in [*regions, *insulating[:index]]:
                for name in (
                    f"interface_{other}_{insulator}",
                    f"interface_{insulator}_{other}",
                ):
                    if name in shared_curves:
                        self._insulator_interfaces[name] = (other, insulator)
        return regions

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
        for name, values in (
            ("ElectronMobility", self.mobility.electrons_cm2(impurity)),
            ("HoleMobility", self.mobility.holes_cm2(impurity)),
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
        if not self._insulator_interfaces:
            return []
        mesh_path = self.devsim_mesh_path

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
            for pair in self._insulator_interfaces.values()
            for region in pair
        }
        taken = {
            key
            for name in (
                *(spec.name for spec in self.contact_specs),
                *self._interface_regions,
            )
            for key in keys(line_group_points(mesh_path, name))
        }

        bound: list[str] = []
        unmatched: dict[str, int] = {}
        for name, (other, insulator) in self._insulator_interfaces.items():
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

    def setup_device(self, device: str | None = None, *, verbose: bool = False) -> str:
        """Create the DEVSIM device from the shared mesh.

        Loads the cm-scaled mesh, registers one DEVSIM region per doped
        mesh region and one contact per declared contact, applies the
        doping node models, and sets up the potential-only physics.

        DEVSIM's meshes, devices and circuits live in one process-global
        namespace, and it refuses new circuit nodes once an analysis has
        run. This device therefore claims the circuit by default
        (``claim_devsim_circuit``), which drops the contact sources of any
        DEVSIM device built earlier in the process — including this sim's
        own previous device. One charge-transport device is live at a
        time; meshes and devices themselves get process-unique names.

        Args:
            device: DEVSIM device name; a process-unique one is generated
                when omitted.
            verbose: Stream DEVSIM's own output (silent by default).

        Returns:
            The device name.
        """
        with devsim_output(verbose):
            return self._setup_device(device)

    def _setup_device(self, device: str | None) -> str:
        """Build the DEVSIM device; see :meth:`setup_device`."""
        regions = self._validate_setup()
        device = device if device is not None else _unique_devsim_name("device")
        devsim = require_devsim()
        sp = import_simple_physics()

        if self.claim_devsim_namespace:
            # The circuit is process-global and refuses new nodes after an
            # analysis, and every registered device joins the next solve;
            # a previous charge device's circuit and device have to go.
            with suppress(Exception):
                devsim.delete_circuit()
            _release_gsim_devices(devsim)

        if self.extended_precision:
            # 128-bit float assembly: the 2D drift-diffusion Newton floors
            # around 1e-5 relative error in double precision on these
            # meshes; extended precision restores full convergence.
            devsim.set_parameter(name="extended_solver", value=True)
            devsim.set_parameter(name="extended_model", value=True)
            devsim.set_parameter(name="extended_equation", value=True)

        mesh_name = self._devsim_mesh_name or _unique_devsim_name("mesh")
        self._devsim_mesh_name = mesh_name
        devsim.create_gmsh_mesh(mesh=mesh_name, file=str(self._devsim_mesh_path))
        for region in regions:
            devsim.add_gmsh_region(
                mesh=mesh_name,
                gmsh_name=region,
                region=region,
                material=self.material,
            )
        for insulator in self.insulators:
            devsim.add_gmsh_region(
                mesh=mesh_name,
                gmsh_name=insulator.region,
                region=insulator.region,
                material=INSULATOR_MATERIAL,
            )
        for spec in self.contact_specs:
            devsim.add_gmsh_contact(
                mesh=mesh_name,
                gmsh_name=spec.name,
                region=self._contact_regions[spec.name],
                material="metal",
                name=spec.name,
            )
        for name, (region_a, region_b) in self._interface_regions.items():
            devsim.add_gmsh_interface(
                mesh=mesh_name,
                gmsh_name=name,
                region0=region_a,
                region1=region_b,
                name=name,
            )
        devsim.finalize_mesh(mesh=mesh_name)
        devsim.create_device(mesh=mesh_name, device=device)

        bound = self._bind_insulator_interfaces(devsim, device)
        for region in regions:
            self._apply_doping(devsim, device, region)
            sp.SetSiliconParameters(device, region, self.temperature)
            sp.CreateSiliconPotentialOnly(device, region)
        for insulator in self.insulators:
            # Potential only, for good: the drift-diffusion stage never
            # reaches these Regions, so they hold no carriers.
            devsim.set_parameter(
                device=device,
                region=insulator.region,
                name="Permittivity",
                value=insulator.relative_permittivity * VACUUM_PERMITTIVITY_F_PER_CM,
            )
            sp.CreateOxidePotentialOnly(device, insulator.region)
        for spec in self.contact_specs:
            # Each contact is driven through a circuit voltage source so the
            # small-signal AC solve can read the terminal admittance (the
            # C(V) extraction) and the DC current from the circuit node.
            devsim.circuit_element(
                name=self._source_name(spec.name),
                n1=sp.GetContactBiasName(spec.name),
                n2=0,
                value=0.0,
                acreal=0.0,
                acimag=0.0,
            )
            sp.CreateSiliconPotentialOnlyContact(
                device, self._contact_regions[spec.name], spec.name, True
            )
            self._current_bias[spec.name] = 0.0
        for name in (*self._interface_regions, *bound):
            self._interface_continuity(devsim, sp, device, name, "Potential")

        self._device = device
        _LIVE_DEVICES.add(device)
        self._dd_initialized = False
        return device

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _solve_dc(self, devsim: Any) -> None:
        """Run one DC Newton solve with the configured tolerances."""
        devsim.solve(
            type="dc",
            absolute_error=self.absolute_error,
            relative_error=self.relative_error,
            maximum_iterations=self.max_iterations,
        )

    def _create_solution(self, sp: Any, device: str, region: str, name: str) -> None:
        """simple_physics re-exports CreateSolution in most DEVSIM versions."""
        create = getattr(sp, "CreateSolution", None)
        if create is None:  # pragma: no cover - version-dependent fallback
            import importlib

            create = importlib.import_module(
                "devsim.python_packages.model_create"
            ).CreateSolution
        create(device, region, name)

    _CONTINUITY_EQUATIONS: ClassVar[dict[str, str]] = {
        "Potential": "PotentialEquation",
        "Electrons": "ElectronContinuityEquation",
        "Holes": "HoleContinuityEquation",
    }

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
        if self._dd_initialized:
            return
        if self._device is None:
            self._setup_device(None)
        devsim = require_devsim()
        sp = import_simple_physics()
        device = self._device
        if device is None:  # pragma: no cover - _setup_device() above set it
            raise RuntimeError("DEVSIM device setup did not complete.")
        regions = self._device_regions()

        self._solve_dc(devsim)

        for region in regions:
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
        for spec in self.contact_specs:
            sp.CreateSiliconDriftDiffusionAtContact(
                device, self._contact_regions[spec.name], spec.name, True
            )
        for name in self._interface_regions:
            self._interface_continuity(devsim, sp, device, name, "Electrons")
            self._interface_continuity(devsim, sp, device, name, "Holes")
        self._solve_dc(devsim)
        self._dd_initialized = True

    @staticmethod
    def _source_name(contact: str) -> str:
        """Circuit voltage-source name attached to a contact."""
        return f"V_{contact}"

    def _set_bias(self, contact: str, bias: float) -> None:
        """Ramp the contact bias to the target in ``bias_step_v`` steps."""
        devsim = require_devsim()
        start = self._current_bias.get(contact, 0.0)
        delta = bias - start
        n_steps = max(1, math.ceil(abs(delta) / self.bias_step_v))
        for i in range(1, n_steps + 1):
            value = start + delta * i / n_steps
            devsim.circuit_alter(name=self._source_name(contact), value=value)
            self._solve_dc(devsim)
        self._current_bias[contact] = bias

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
                device=self._device, contact=contact, equation="PotentialEquation"
            )
        )

    def _collect_carriers(self, devsim: Any) -> CarrierMap:
        """Concatenate node fields across the device regions (coords in um)."""
        device = self._device
        columns: dict[str, list[NDArray[np.float64]]] = {
            "x": [],
            "y": [],
            "Electrons": [],
            "Holes": [],
            "Potential": [],
            "NetDoping": [],
        }
        region_names: list[str] = []
        for region in self._device_regions():
            n_nodes = 0
            for name, column in columns.items():
                values = np.asarray(
                    devsim.get_node_model_values(
                        device=device, region=region, name=name
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

    def _resolve_sweep_contact(self, contact: str | None) -> str:
        """Default to the first declared contact; reject unknown names."""
        specs = self.contact_specs
        if not specs:
            raise ValueError("No contacts declared. Call add_contact() first.")
        if contact is None:
            return str(specs[0].name)
        names = [s.name for s in specs]
        if contact not in names:
            raise ValueError(f"Unknown contact '{contact}'. Declared contacts: {names}")
        return contact

    def solve(
        self,
        bias: float = 0.0,
        *,
        contact: str | None = None,
        verbose: bool = False,
    ) -> BiasPoint:
        """Solve the drift-diffusion system at one bias point.

        Args:
            bias: Bias voltage applied to the swept contact (V); the other
                contacts stay at 0 V.
            contact: Swept contact name (defaults to the first declared).
            verbose: Stream DEVSIM's own output, Newton iterations
                included (silent by default).

        Returns:
            The solved :class:`BiasPoint` including carrier maps, terminal
            currents, and the small-signal capacitance ``Im(I)/omega`` from
            the quasi-static AC solve.
        """
        contact = self._resolve_sweep_contact(contact)
        with devsim_output(verbose):
            return self._solve_point(bias, contact)

    def _solve_point(self, bias: float, contact: str) -> BiasPoint:
        """Solve one bias point on a resolved contact; see :meth:`solve`."""
        self._initialize_drift_diffusion()
        devsim = require_devsim()

        self._set_bias(contact, bias)
        carriers = self._collect_carriers(devsim)
        currents = {
            spec.name: self._contact_current(devsim, spec.name)
            for spec in self.contact_specs
        }
        charge = self._contact_charge(devsim, contact)

        # Small-signal AC solve at a quasi-static frequency: the terminal
        # admittance of the swept contact's unit-amplitude source gives
        # C = |Im(I)| / omega (the small-signal charge per volt). Only the
        # swept source carries AC amplitude; the others stay AC-grounded.
        for spec in self.contact_specs:
            devsim.circuit_alter(
                name=self._source_name(spec.name),
                param="acreal",
                value=1.0 if spec.name == contact else 0.0,
            )
        freq = self.small_signal_freq_hz
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
        devsim.solve(type="ac", frequency=self.junction_freq_hz)
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
            admittance_freq_hz=self.junction_freq_hz,
        )

    def sweep(
        self,
        biases: list[float] | Any,
        *,
        contact: str | None = None,
        verbose: bool = False,
    ) -> BiasSweepResult:
        """Solve a bias sweep and return carrier maps and C(V) per point.

        Args:
            biases: Bias voltages (V) applied in order to the swept contact.
            contact: Swept contact name (defaults to the first declared).
            verbose: Stream DEVSIM's own output, Newton iterations
                included (silent by default).

        Returns:
            :class:`BiasSweepResult` with one :class:`BiasPoint` per bias.
        """
        contact = self._resolve_sweep_contact(contact)
        with devsim_output(verbose):
            points = [self._solve_point(float(bias), contact) for bias in biases]
        return BiasSweepResult(contact=contact, points=points)


__all__ = ["ChargeTransportSim", "Insulator"]
