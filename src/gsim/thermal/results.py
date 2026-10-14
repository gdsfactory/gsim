"""Results of a thermal phase-shifter solve, in gsim units.

What :class:`~gsim.thermal.ThermalSim` hands back:

- :class:`ThermalResult` -- the steady answer: P_pi, efficiency, heater
  resistance and V_pi, crosstalk, thermal resistance at each waveguide,
  and the temperature-rise field at any power (the model is linear in
  power, so one solve answers every power);
- :class:`TemperatureMap` -- the temperature rise on the thermal mesh at
  one heater power;
- :class:`StepResponse` -- the switch-on transient: temperature at each
  waveguide core and the phase over time, with the switching time tau;
- :class:`HeaterReport` -- one table of the figures above.

Units: lengths in um, power in mW (total heater power over the heater
length), temperature **rise** above ambient in K, time in us,
resistance in ohm, voltage in V, phase in rad. Every result is a 2D
cross-section of an infinitely long heater: no end losses, meanders or
via spreading.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gsim.thermal.optical import sample_temperature

if TYPE_CHECKING:
    from gsim.thermal.mesh import CrossSectionMesh

__all__ = [
    "HeaterReport",
    "StepResponse",
    "TemperatureMap",
    "ThermalResult",
]

_UM_TO_M = 1e-6
_MW_TO_W = 1e-3


def power_per_length(power_mW: float, length_um: float) -> float:
    """Heater power per unit length (W/m) of a total power over a length.

    Args:
        power_mW: Total heater power (mW).
        length_um: Heater length (um).

    Returns:
        P / L in W/m.
    """
    return power_mW * _MW_TO_W / (length_um * _UM_TO_M)


def _in_plane_axis(plane: str) -> str:
    """Name of the in-plane axis of a vertical cut, ``y`` for ``x=...``."""
    return "y" if plane.strip().lower().startswith("x") else "x"


@dataclass(frozen=True)
class TemperatureMap:
    """Temperature rise on the thermal mesh at one heater power.

    A generic scalar field: one value per mesh node, read anywhere in the
    thermal Window by its P1 interpolant.

    Attributes:
        mesh: The meshed thermal Cross-section (coordinates in um).
        values: Temperature rise above ambient at every mesh node (K).
        power: Total heater power the field belongs to (mW).
    """

    mesh: CrossSectionMesh = field(repr=False)
    values: NDArray[np.float64] = field(repr=False)
    power: float

    def at(self, position: ArrayLike, z: ArrayLike) -> Any:
        """Temperature rise at in-plane position(s) and height(s).

        Args:
            position: In-plane coordinate(s) in um -- ``y`` for an
                ``"x=..."`` cut, ``x`` for a ``"y=..."`` cut, as in
                :meth:`ThermalSim.add_waveguide`.
            z: Height(s) in um, broadcast against ``position``.

        Returns:
            The rise in K: a float for scalar inputs, else an array of
            the broadcast shape.

        Raises:
            ValueError: When a point lies outside the thermal Window.
        """
        h, v = np.broadcast_arrays(
            np.asarray(position, dtype=float), np.asarray(z, dtype=float)
        )
        values = sample_temperature(
            self.mesh, self.values, np.vstack([h.ravel(), v.ravel()])
        )
        if h.ndim == 0:
            return float(values[0])
        return values.reshape(h.shape)

    def peak(self, layer: str | None = None) -> float:
        """Peak temperature rise, over the whole Window or one Region.

        Args:
            layer: A Region of the thermal mesh -- a drawn stack layer
                such as the heater, or a slab by its material name.
                ``None`` takes the whole Window.

        Returns:
            The largest nodal rise in K.

        Raises:
            ValueError: When the mesh has no Region of that name.
        """
        if layer is None:
            return float(self.values.max())
        nodes = np.unique(self.mesh.mesh.t[:, self.mesh.region_mask(layer)])
        return float(self.values[nodes].max())

    def plot(self, ax: Any = None, *, cmap: str = "inferno") -> Any:
        """Draw the temperature rise over the Cross-section.

        Args:
            ax: Matplotlib axes to draw on; a new figure when ``None``.
            cmap: Matplotlib colormap name.

        Returns:
            The matplotlib axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation

        if ax is None:
            _, ax = plt.subplots()
        m = self.mesh.mesh
        triangulation = Triangulation(m.p[0], m.p[1], m.t.T)
        image = ax.tripcolor(triangulation, self.values, shading="gouraud", cmap=cmap)
        ax.figure.colorbar(image, ax=ax, label="Temperature rise (K)")
        ax.set_xlabel(f"{_in_plane_axis(self.mesh.plane)} (µm)")
        ax.set_ylabel("z (µm)")
        ax.set_title(f"Temperature rise at {self.power:.3g} mW")
        ax.set_aspect("equal")
        return ax


@dataclass(frozen=True)
class StepResponse:
    """Response of the heated waveguides to a heater power switched on at t = 0.

    In this linear model the switch-off from steady state is the exact
    complement of the switch-on, so the fall time equals the rise time
    and one ``tau`` describes both.

    Attributes:
        time: Sample times from 0 to the duration (us).
        power: Total heater power stepped on at t = 0 (mW).
        phase_shifter: Name of the waveguide whose phase is traced.
        phase: Phase shift of the phase-shifter waveguide over time
            (rad), from first-order perturbation scaled so that its
            steady value matches the steady mode re-solve.
        steady_phase: Steady-state phase shift at ``power`` (rad).
        temperature: Temperature rise at each waveguide's core centre
            over time (K), by waveguide name.
        tau: 10-90 % rise time of the phase (us), the usual measured
            switching time.
        tau_1e: Time to reach 1 - 1/e (63 %) of the steady phase (us),
            for comparison with papers that quote a 1/e time constant.
    """

    time: NDArray[np.float64] = field(repr=False)
    power: float
    phase_shifter: str
    phase: NDArray[np.float64] = field(repr=False)
    steady_phase: float
    temperature: dict[str, NDArray[np.float64]] = field(repr=False)
    tau: float
    tau_1e: float

    def plot(self, ax: Any = None) -> Any:
        """Draw the phase over time, marking tau.

        Args:
            ax: Matplotlib axes to draw on; a new figure when ``None``.

        Returns:
            The matplotlib axes.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.time, self.phase, label=self.phase_shifter)
        ax.axhline(self.steady_phase, color="0.6", linestyle=":", label="steady")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Phase shift (rad)")
        ax.set_title(f"Step to {self.power:.3g} mW: tau(10-90 %) = {self.tau:.3g} µs")
        ax.legend()
        return ax


@dataclass(frozen=True)
class HeaterReport:
    """One table of a heater's figures of merit.

    Attributes:
        p_pi: Total heater power for a pi phase shift (mW).
        efficiency: Phase per power, pi / P_pi (rad/mW).
        resistance: Heater resistance over its length (ohm).
        v_pi: Voltage for a pi phase shift (V).
        peak_temperature: Peak temperature rise in the heater at P_pi (K).
        crosstalk: Phase on each victim waveguide at P_pi (rad).
        thermal_resistance: Core temperature rise per mW, by waveguide
            (K/mW).
        length: Heater length (um).
        wavelength: Vacuum wavelength (um).
        tau: 10-90 % switching time (us), when a step response was given.
        tau_1e: 1/e time constant (us), when a step response was given.
    """

    p_pi: float
    efficiency: float
    resistance: float
    v_pi: float
    peak_temperature: float
    crosstalk: dict[str, float]
    thermal_resistance: dict[str, float]
    length: float
    wavelength: float
    tau: float | None = None
    tau_1e: float | None = None

    def rows(self) -> list[tuple[str, str]]:
        """The table rows, ``(quantity with unit, formatted value)``.

        Returns:
            One row per figure, in reading order.
        """
        rows = [
            ("P_pi (mW)", f"{self.p_pi:.4g}"),
            ("Efficiency (rad/mW)", f"{self.efficiency:.4g}"),
            ("Resistance (ohm)", f"{self.resistance:.4g}"),
            ("V_pi (V)", f"{self.v_pi:.4g}"),
            ("Peak heater rise at P_pi (K)", f"{self.peak_temperature:.4g}"),
        ]
        rows += [
            (f"Thermal resistance, {name} (K/mW)", f"{value:.4g}")
            for name, value in self.thermal_resistance.items()
        ]
        rows += [
            (f"Crosstalk at P_pi, {name} (rad)", f"{value:.4g}")
            for name, value in self.crosstalk.items()
        ]
        if self.tau is not None:
            rows.append(("tau 10-90 % (us)", f"{self.tau:.4g}"))
        if self.tau_1e is not None:
            rows.append(("tau 1/e (us)", f"{self.tau_1e:.4g}"))
        rows += [
            ("Heater length (um)", f"{self.length:.4g}"),
            ("Wavelength (um)", f"{self.wavelength:.4g}"),
        ]
        return rows

    def _title(self) -> str:
        """Heading line stating the model the numbers come from."""
        return "Heater report (2D cross-section, linear in power)"

    def __str__(self) -> str:
        """The table as aligned plain text, units in the row labels."""
        rows = self.rows()
        width = max(len(name) for name, _ in rows)
        lines = [self._title()]
        lines += [f"  {name:<{width}}  {value}" for name, value in rows]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """HTML table for notebooks."""
        body = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{html.escape(value)}</td></tr>"
            for name, value in self.rows()
        )
        return (
            f"<table><caption>{html.escape(self._title())}</caption>"
            "<tr><th>Quantity</th><th>Value</th></tr>"
            f"{body}</table>"
        )


@dataclass(frozen=True)
class ThermalResult:
    """Steady heater solve: P_pi, crosstalk, resistance and the temperature field.

    The model is linear in power (constant k and dn/dT), so every power
    follows from one solve: the temperature rise and the effective-index
    shift scale with the heater power, and P_pi does not depend on the
    heater length in 2D.

    Attributes:
        phase_shifter: Name of the phase-shifter waveguide (the first
            added); the other waveguides are crosstalk victims.
        wavelength: Vacuum wavelength (um).
        length: Heater length (um).
        p_pi: Total heater power for a pi phase shift of the phase
            shifter (mW).
        efficiency: pi / P_pi (rad/mW).
        resistance: Heater resistance, length over the summed
            conductance of its conductors (ohm).
        v_pi: Voltage across the heater at P_pi (V).
        crosstalk: Phase on each victim waveguide when the heater
            drives P_pi (rad).
        thermal_resistance: Temperature rise at each waveguide's core
            centre per mW of heater power (K/mW).
        dneff_per_power: Effective-index shift per mW of heater power,
            from the mode re-solve, by waveguide (1/mW).
        dneff_first_order: The same from first-order perturbation, the
            cross-check of the re-solve (1/mW).
    """

    phase_shifter: str
    wavelength: float
    length: float
    p_pi: float
    efficiency: float
    resistance: float
    v_pi: float
    crosstalk: dict[str, float]
    thermal_resistance: dict[str, float]
    dneff_per_power: dict[str, float]
    dneff_first_order: dict[str, float]
    heater_layers: tuple[str, ...] = field(repr=False)
    mesh: CrossSectionMesh = field(repr=False)
    rise_per_power_per_length: NDArray[np.float64] = field(repr=False)
    """Nodal rise per heater power per length (K per W/m)."""

    def temperature(self, power: float) -> TemperatureMap:
        """Temperature rise over the Cross-section at one heater power.

        Args:
            power: Total heater power (mW).

        Returns:
            The rise on the thermal mesh (K).
        """
        scale = power_per_length(power, self.length)
        return TemperatureMap(
            mesh=self.mesh,
            values=np.asarray(self.rise_per_power_per_length * scale, dtype=np.float64),
            power=float(power),
        )

    def phase(self, power: float) -> dict[str, float]:
        """Phase shift of every waveguide at one heater power.

        Args:
            power: Total heater power (mW).

        Returns:
            ``2 pi dn_eff L / lambda`` by waveguide name (rad).
        """
        return {
            name: 2.0 * math.pi * slope * power * self.length / self.wavelength
            for name, slope in self.dneff_per_power.items()
        }

    @property
    def peak_temperature(self) -> float:
        """Peak temperature rise in the heater at P_pi (K)."""
        tmap = self.temperature(self.p_pi)
        return max(tmap.peak(layer) for layer in self.heater_layers)

    def report(self, step: StepResponse | None = None) -> HeaterReport:
        """Figures of merit in one table.

        Args:
            step: A step response to take tau from, if any.

        Returns:
            The :class:`HeaterReport`.
        """
        return HeaterReport(
            p_pi=self.p_pi,
            efficiency=self.efficiency,
            resistance=self.resistance,
            v_pi=self.v_pi,
            peak_temperature=self.peak_temperature,
            crosstalk=dict(self.crosstalk),
            thermal_resistance=dict(self.thermal_resistance),
            length=self.length,
            wavelength=self.wavelength,
            tau=step.tau if step is not None else None,
            tau_1e=step.tau_1e if step is not None else None,
        )
