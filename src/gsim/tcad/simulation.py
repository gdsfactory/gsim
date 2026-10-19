"""DEVSIM charge-transport simulation of a waveguide cross-section.

``Simulation`` is configured by physical concern, like
:class:`gsim.fdtd.Simulation`. Meshing goes through the shared native-2D
pipeline of :class:`gsim.palace.BoundaryModeSim`, the exact mesh Palace
BoundaryMode sees, so there is no second meshing path. Contacts and
Interfaces become named line groups on that mesh, which DEVSIM binds
boundary conditions and continuity to.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gsim.common.config import Concern, Same, component_digest, fingerprint
from gsim.tcad.api import (
    MESH_FIELDS,
    Contact,
    Contacts,
    Domain,
    Doping,
    Geometry,
    Interface,
    Interfaces,
    Materials,
    Solver,
    TCADMesh,
)
from gsim.tcad.device import Device, check_layout
from gsim.tcad.doping import DopingProfile
from gsim.tcad.mesh import UM_TO_CM, write_scaled_msh
from gsim.tcad.results import BiasPoint, BiasSweepResult
from gsim.tcad.runtime import devsim_output

_CONCERNS = (
    "geometry",
    "domain",
    "materials",
    "doping",
    "contacts",
    "interfaces",
    "solver",
)


class Simulation:
    """Poisson and drift-diffusion on a waveguide cross-section.

    Example:
        >>> from gsim import tcad
        >>> sim = tcad.Simulation()
        >>> sim.geometry(component, stack=stack)
        >>> sim.domain(plane="x=0", window=(-25.0, -15.0))
        >>> sim.doping.add(
        ...     tcad.StepDoping(
        ...         region="p_rib", dopant_type="acceptor", concentration_cm3=1e18
        ...     )
        ... )
        >>> sim.contacts.add(name="anode", semiconductor="p_rib", electrode="metal_p")
        >>> sim.solver(mesh_preset="coarse", refined_mesh_size_um=0.05)
        >>> sim.mesh(output_dir="./tcad")
        >>> sweep = sim.sweep([0.0, 1.0, 2.0], contact="cathode")

    The mesh inputs (``geometry``, ``domain``, ``contacts``,
    ``interfaces`` and the mesh fields of ``solver``) are recorded by
    :meth:`mesh`; a solve raises once any of them has changed. A change to
    a device input (``materials``, ``doping``, the other ``solver`` fields)
    rebuilds the DEVSIM device on the existing mesh at the next solve.
    """

    geometry = Concern(Geometry)
    domain = Concern(Domain)
    materials = Concern(Materials)
    doping = Concern(Doping)
    contacts = Concern(Contacts)
    interfaces = Concern(Interfaces)
    solver = Concern(Solver)

    def __init__(
        self,
        *,
        geometry: Geometry | Mapping[str, Any] | None = None,
        domain: Domain | Mapping[str, Any] | None = None,
        materials: Materials | Mapping[str, Any] | None = None,
        doping: Doping | Iterable[DopingProfile] | None = None,
        contacts: Contacts | Iterable[Contact | Mapping[str, Any]] | None = None,
        interfaces: Interfaces | Iterable[Interface | Mapping[str, Any]] | None = None,
        solver: Solver | Mapping[str, Any] | None = None,
    ) -> None:
        """Create a simulation from its concerns, as models or mappings."""
        self.geometry = geometry
        self.domain = domain
        self.materials = materials
        self.doping = doping
        self.contacts = contacts
        self.interfaces = interfaces
        self.solver = solver
        self._output_dir: Path | None = None
        self._mesh: TCADMesh | None = None
        self._mesh_fingerprint: Any = None
        self._device: Device | None = None
        self._device_fingerprint: Any = None

    def __repr__(self) -> str:
        """Show the concerns."""
        concerns = ", ".join(f"{name}={getattr(self, name)!r}" for name in _CONCERNS)
        return f"Simulation({concerns})"

    # ------------------------------------------------------------------
    # Fingerprints
    # ------------------------------------------------------------------

    def _mesh_inputs(self) -> Any:
        """Snapshot of everything the mesh is built from."""
        component = self.geometry.component
        return fingerprint(
            (
                Same(component),
                component_digest(component),
                self.geometry.stack,
                self.domain,
                list(self.contacts),
                list(self.interfaces),
                {name: getattr(self.solver, name) for name in MESH_FIELDS},
            )
        )

    def _device_inputs(self) -> Any:
        """Snapshot of everything the DEVSIM device is built from."""
        return fingerprint(
            (
                self.materials,
                list(self.doping),
                {
                    name: value
                    for name, value in dict(self.solver).items()
                    if name not in MESH_FIELDS
                },
            )
        )

    # ------------------------------------------------------------------
    # Meshing
    # ------------------------------------------------------------------

    def _check_mesh_inputs(self) -> tuple[Any, str]:
        """Refuse a configuration the mesh cannot be built from.

        Returns:
            The component and the cross-section plane.
        """
        component = self.geometry.component
        plane = self.domain.plane
        if component is None:
            raise ValueError("No geometry. Call sim.geometry(component) first.")
        if plane is None:
            raise ValueError(
                "No cross-section plane. Call sim.domain(plane='x=<um>') first."
            )
        if not self.contacts:
            raise ValueError(
                "Charge transport requires at least one contact. Call "
                "sim.contacts.add(name=..., semiconductor=..., electrode=...)."
            )
        names = [item.name for item in (*self.contacts, *self.interfaces)]
        clashes = sorted({name for name in names if names.count(name) > 1})
        if clashes:
            raise ValueError(
                f"Names {clashes} are used by both a contact and an interface; "
                "each becomes a mesh group of that name."
            )
        return component, plane

    def _boundary_mode_sim(self, component: Any, plane: str) -> Any:
        """Translate the mesh concerns into the shared meshing pipeline."""
        from gsim.palace import BoundaryModeSim

        bsim = BoundaryModeSim()
        bsim.set_output_dir(str(self._output_dir))
        stack = self.geometry.stack
        if stack is None:
            bsim.set_stack()
        elif isinstance(stack, (str, Path)):
            bsim.set_stack(yaml_path=stack)
        else:
            bsim.set_stack(stack)
        domain = self.domain
        margins = (domain.margin_x, domain.margin_y, domain.z_above, domain.z_below)
        if any(margin is not None for margin in margins):
            bsim.set_airbox(
                margin_x=domain.margin_x,
                margin_y=domain.margin_y,
                z_above=domain.z_above,
                z_below=domain.z_below,
            )
        bsim.set_geometry(component)
        bsim.set_cross_section(plane, window=domain.window, window_z=domain.window_z)
        for contact in self.contacts:
            bsim.add_contact(
                name=contact.name,
                layer_a=contact.semiconductor,
                layer_b=contact.electrode,
            )
        for interface in self.interfaces:
            bsim.add_interface(
                name=interface.name,
                layer_a=interface.region_a,
                layer_b=interface.region_b,
            )
        return bsim

    def mesh(
        self, *, output_dir: str | Path | None = None, verbose: bool = False
    ) -> TCADMesh:
        """Generate the shared native-2D mesh and its cm-scaled DEVSIM copy.

        Resolution comes from the ``solver`` concern. A new mesh drops the
        DEVSIM device; the next solve builds it again.

        Args:
            output_dir: Directory the meshes are written to. Required on
                the first call, remembered afterwards.
            verbose: Print the meshing pipeline's progress.

        Returns:
            The mesh paths and physical groups.
        """
        if output_dir is not None:
            self._output_dir = Path(output_dir)
        if self._output_dir is None:
            raise ValueError("No output directory. Call sim.mesh(output_dir=...).")
        component, plane = self._check_mesh_inputs()
        self._output_dir.mkdir(parents=True, exist_ok=True)

        bsim = self._boundary_mode_sim(component, plane)
        solver = self.solver
        result = bsim.mesh(
            preset=solver.mesh_preset,
            refined_mesh_size=solver.refined_mesh_size_um,
            max_mesh_size=solver.max_mesh_size_um,
            refinement_boxes=solver.refinement_boxes,
            mesh_grading=solver.mesh_grading,
            verbose=verbose,
        )
        devsim_mesh_path = write_scaled_msh(
            result.mesh_path, self._output_dir / "devsim.msh", scale=UM_TO_CM
        )
        self._reset_device()
        self._mesh = TCADMesh(
            mesh_path=Path(result.mesh_path),
            devsim_mesh_path=devsim_mesh_path,
            groups=dict(bsim.mesh_groups),
        )
        self._mesh_fingerprint = self._mesh_inputs()
        return self._mesh

    def _current_mesh(self) -> TCADMesh:
        """The mesh, refusing one the configuration has moved away from."""
        if self._mesh is None:
            raise ValueError("No mesh generated. Call mesh() first.")
        if self._mesh_inputs() != self._mesh_fingerprint:
            raise ValueError(
                "The mesh inputs (geometry, domain, contacts, interfaces or the "
                "solver's mesh settings) changed since mesh(); call mesh() again."
            )
        return self._mesh

    # ------------------------------------------------------------------
    # DEVSIM device
    # ------------------------------------------------------------------

    def _reset_device(self) -> None:
        """Forget the DEVSIM device; setup runs again on the next solve."""
        if self._device is not None:
            self._device.release()
        self._device = None
        self._device_fingerprint = None

    def _current_device(self) -> Device:
        """The device for the current configuration, built when needed."""
        mesh = self._current_mesh()
        fingerprint = self._device_inputs()
        if (
            self._device is not None
            and not self._device.deleted
            and fingerprint == self._device_fingerprint
        ):
            return self._device
        self._reset_device()
        doping = list(self.doping)
        contacts = list(self.contacts)
        interfaces = list(self.interfaces)
        materials = self.materials.model_copy(deep=True)
        layout = check_layout(
            mesh.groups,
            doping=doping,
            insulators=materials.insulators,
            contacts=contacts,
            interfaces=interfaces,
        )
        self._device = Device(
            mesh=mesh,
            materials=materials,
            doping=[profile.model_copy(deep=True) for profile in doping],
            contacts=contacts,
            interfaces=interfaces,
            solver=self.solver.model_copy(deep=True),
            layout=layout,
        )
        self._device_fingerprint = fingerprint
        return self._device

    def _setup_device(self, device: str | None = None, *, verbose: bool = False) -> str:
        """Build the DEVSIM device now rather than at the first solve.

        Args:
            device: DEVSIM device name; a process-unique one when omitted.
            verbose: Stream DEVSIM's own output.

        Returns:
            The device name.
        """
        current = self._current_device()
        if current.name is not None:
            current.release()
        with devsim_output(verbose):
            return current.setup(device)

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    @staticmethod
    def _finite_bias(bias: float) -> float:
        """The bias as a float, refusing NaN and infinities."""
        value = float(bias)
        if not math.isfinite(value):
            raise ValueError(f"A bias must be a finite voltage; got {bias!r}.")
        return value

    def _resolve_contact(self, contact: str | None) -> str:
        """Default to the first declared contact; reject unknown names."""
        names = [c.name for c in self.contacts]
        if not names:
            raise ValueError("No contacts declared. Call sim.contacts.add() first.")
        if contact is None:
            return names[0]
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
        value = self._finite_bias(bias)
        device = self._current_device()
        contact = self._resolve_contact(contact)
        with devsim_output(verbose):
            return device.solve_point(value, contact)

    def sweep(
        self,
        biases: Iterable[float],
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
        values = [self._finite_bias(bias) for bias in biases]
        device = self._current_device()
        contact = self._resolve_contact(contact)
        with devsim_output(verbose):
            points = [device.solve_point(bias, contact) for bias in values]
        return BiasSweepResult(
            contact=contact,
            points=points,
            semiconductor_permittivity=self.materials.semiconductor_permittivity,
        )


__all__ = ["Simulation"]
