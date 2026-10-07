"""The charge Stage: Poisson + drift-diffusion across a Bias sweep.

The Stage assembles a :class:`gsim.tcad.ChargeTransportSim` from the
device description — its Window, its Contacts, its Interfaces and its
doping profiles all derived — meshes it once, and sweeps the bias. The
result is the backend's own Bias sweep, so nothing downstream has to learn
a second result type.

DEVSIM is optional: the Stage checks for it before it meshes, so a missing
extra costs nothing but the error message.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field, PrivateAttr

from gsim.common.carriers import MobilityModel
from gsim.modulator.meshing import STAGE_AIRBOX, STAGE_MESH
from gsim.modulator.stage import Stage
from gsim.tcad.runtime import require_devsim

if TYPE_CHECKING:
    from gsim.tcad.results import BiasSweepResult
    from gsim.tcad.sim import ChargeTransportSim

__all__ = ["ChargeStage"]


class ChargeStage(Stage):
    """Charge transport through the Phase shifter, bias point by bias point.

    Attributes:
        biases: Bias voltages applied to the swept Contact (V).
        contact: Swept Contact; defaults to the Contact on the n side, so
            positive voltages reverse-bias the Junction.
        window: In-plane charge Window (um); derived from the device
            description when unset.
        window_z: Vertical Window (um); unclipped when unset.
        mesh: Keyword arguments forwarded to the mesh pipeline.
        airbox: Background region around the Window.
        temperature: Lattice temperature (K).
        oxide: Solve Poisson in the oxide around the doped slab as well as
            in the silicon — the potential alone, no carriers — so the
            capacitance and the series-RC Junction branch count the field
            fringing around the Junction. The Carrier map holds the doped
            Regions only either way, but with the oxide it varies in
            depth: the fringing field depletes the silicon's surfaces
            first. A Staircase's Strips are uniform in depth and spread
            that surface depletion over their whole height, where the
            optical Mode is strongest: an optical Staircase then reads
            the index shift about 7 % high, however many Strips it has
            (the continuous optical solve is not affected). Turn it off
            for the Poisson solve in the silicon alone that a
            one-dimensional depletion formula describes.
        mobility: Low-field mobility model the transport solve runs on,
            and — unless the carriers Stage is given its own — the one
            the RF conductivity is evaluated with.
        settings: Extra settings applied to the charge-transport sim.
    """

    stage_name: ClassVar[str] = "charge"

    #: The charge-transport sim of the last run, kept so its DEVSIM state
    #: is released before the next one is built.
    _sim: Any = PrivateAttr(default=None)

    biases: list[float] = Field(default_factory=lambda: [0.0])
    contact: str | None = None
    window: tuple[float, float] | None = None
    window_z: tuple[float, float] | None = None
    # Finer elements than the shared default: the charge solve has to
    # resolve the depletion edge, not a mode.
    mesh: dict[str, Any] = Field(
        default_factory=lambda: STAGE_MESH | {"refined_mesh_size": 0.02}
    )
    airbox: dict[str, Any] = Field(default_factory=STAGE_AIRBOX.copy)
    temperature: float = Field(default=300.0, gt=0.0)
    oxide: bool = True
    mobility: MobilityModel = Field(default_factory=MobilityModel.masetti_silicon)
    settings: dict[str, Any] = Field(default_factory=dict)

    def simulation(self) -> ChargeTransportSim:
        """Assemble the charge-transport sim this Stage would run.

        Contacts, Interfaces, doping profiles and the Window all come from
        the device description; nothing is declared twice.

        Returns:
            The configured (unmeshed) :class:`ChargeTransportSim`.
        """
        from gsim.tcad.doping import StepDoping
        from gsim.tcad.sim import ChargeTransportSim

        study = self._require_study()
        layout = study.layout
        device = study.device

        sim = ChargeTransportSim(
            temperature=self.temperature, mobility=self.mobility, **self.settings
        )
        sim.set_output_dir(study.stage_dir(self.stage_name))
        sim.set_stack(study.stack)
        sim.set_geometry(study.component)
        sim.set_airbox(**self.airbox)
        sim.set_cross_section(
            study.plane,
            window=self.window if self.window is not None else layout.window,
            window_z=self.window_z,
        )

        for contact in layout.contacts:
            sim.add_contact(
                name=contact.name,
                layer_a=contact.region,
                layer_b=contact.electrode,
            )
        for interface in layout.interfaces:
            sim.add_interface(
                name=interface.name,
                layer_a=interface.regions[0],
                layer_b=interface.regions[1],
            )
        if device.doping is not None:
            for profile in device.doping:
                sim.add_doping(profile)
        else:
            for region in device.doped_regions:
                sim.add_doping(
                    StepDoping(
                        region=region,
                        dopant_type=device.dopant_type(region),
                        concentration_cm3=device.concentration_cm3(region),
                    )
                )
        if self.oxide:
            for region, permittivity in self._oxide_regions().items():
                sim.add_insulator(region=region, relative_permittivity=permittivity)
        return sim

    def junction_boxes(self) -> list[tuple[float, float, float, float, float]]:
        """The boxes this Stage's mesh is held to a size in, around the Junction.

        The mesh pipeline sizes elements on its refinement lines — the
        Junction, the silicon's surfaces — and lets the size grow at once
        with the distance, so mid-slab and 50 nm from the Junction, where
        the depletion edge sits under bias, an element is several times
        the refined size. Carriers fall by decades across that edge: the
        capacitance is then off by ten per cent and more, and nothing
        downstream can read the edge's position off the Carrier map. The
        two Regions the Junction separates are therefore held to the size
        the lines get. ``refinement_boxes`` in :attr:`mesh` replaces them.

        A Stage that carries the Carrier map onto a mesh of its own, not
        onto Strips, needs the same boxes: a transfer onto coarser
        elements smears the edge again.

        Returns:
            ``(h_min, h_max, z_min, z_max, size)`` boxes (um).
        """
        if "refinement_boxes" in self.mesh:
            return [tuple(box) for box in self.mesh["refinement_boxes"]]
        span = self._require_study().layout.junction_span
        size = 0.5 * float(self.mesh["refined_mesh_size"])
        return [(*span.h, *span.z, size)]

    def _oxide_regions(self) -> dict[str, float]:
        """The stack's insulating background dielectrics: Region to permittivity.

        The mesh names a background dielectric's Region after its
        material, so two dielectric slabs of one material are one Region.
        A slab drawn in a doped Region's material, or in one that
        conducts (a silicon substrate), is no insulator and is left out.
        """
        stack = self._require_study().stack
        doped = set(self._require_study().device.doped_regions)
        regions: dict[str, float] = {}
        for dielectric in stack.dielectrics:
            material = str(dielectric["material"])
            properties = stack.materials.get(material, {})
            permittivity = properties.get("permittivity")
            conducts = float(properties.get("conductivity") or 0.0) > 0.0
            if material in doped or conducts or permittivity is None:
                continue
            regions[material] = float(permittivity)
        return regions

    def swept_contact(self) -> str:
        """Name of the Contact the sweep drives.

        Returns:
            The configured Contact, or the n-side one by default.
        """
        if self.contact is not None:
            return self.contact
        return str(self._require_study().layout.contact_on("n").name)

    def export_junction_model(self, path: str | Path | None = None) -> Path:
        """Write the junction's series-RC branch per Bias point to a file.

        The devsim-derived compact model leaves gsim as one tabular JSON
        file — C_j(V) and R_s(V) per meter of Traveling-wave electrode,
        with units, the swept Contact and the solve settings recorded —
        so circulax or any circuit tool reads it back without importing
        gsim. The format lives in
        :func:`gsim.common.circuit.write_junction_model`; the matching
        reader is :func:`gsim.common.circuit.read_junction_model`. Runs
        the charge Stage first when it holds no result.

        Args:
            path: Output file; ``junction.json`` in the charge Stage's
                output directory when omitted.

        Returns:
            The written path.

        Raises:
            ValueError: When a Bias point holds no small-signal
                admittance, or the sweep's points were fitted at
                different frequencies.
        """
        from gsim import __version__
        from gsim.common.circuit import write_junction_model

        sweep: BiasSweepResult = self.run()
        if not sweep.points:
            raise ValueError(
                "The bias sweep holds no points, so there is no junction to "
                "export; sweep at least one bias with study.charge(biases=[...])."
            )
        branch = sweep.junction_branch()
        frequencies = {point.admittance_freq_hz for point in sweep.points}
        if len(frequencies) != 1:
            listed = ", ".join(f"{freq:g}" for freq in sorted(frequencies))
            raise ValueError(
                f"The bias sweep's admittances were fitted at {listed} Hz; a "
                "junction model records one fit frequency, so re-solve the "
                "sweep with a single junction_freq_hz."
            )
        target = (
            Path(path)
            if path is not None
            else self._require_study().stage_dir(self.stage_name) / "junction.json"
        )
        return write_junction_model(
            target,
            bias_v=sweep.voltages,
            r_s_ohm_m=np.asarray(branch.r_s_ohm_m, dtype=np.float64),
            c_j_f_per_m=np.asarray(branch.c_j_f_per_m, dtype=np.float64),
            contact=sweep.contact,
            freq_hz=frequencies.pop(),
            provenance={
                "generator": f"gsim {__version__}",
                "temperature_k": self.temperature,
            },
        )

    def _solve(self) -> BiasSweepResult:
        """Mesh the charge Window and sweep the bias."""
        require_devsim()
        if self._sim is not None:
            # DEVSIM's device, mesh and circuit namespaces are global: the
            # previous run's device has to go before this one is built.
            self._sim.reset_device()
        sim = self.simulation()
        self._sim = sim
        sim.mesh(**(self.mesh | {"refinement_boxes": self.junction_boxes()}))
        return sim.sweep(
            list(self.biases),
            contact=self.swept_contact(),
            verbose=self._is_verbose(),
        )
