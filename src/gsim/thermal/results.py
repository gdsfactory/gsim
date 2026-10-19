"""Results of a thermal phase-shifter solve, in gsim units.

What :class:`~gsim.thermal.Simulation` hands back:

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
length), temperature **rise** above ambient in K (absolute temperatures
only where an ``ambient_k`` is given), time in us,
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
from gsim.thermal.solver import (
    _crossing_time,
    _normalised_step,
    drive_power_per_length,
    element_dofs,
    lagrange_degree,
    piecewise_linear,
)

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


def _absolute(ambient_k: float | None) -> float:
    """The offset an ambient adds to a rise: 0 for none, else the ambient."""
    if ambient_k is None:
        return 0.0
    if not ambient_k > 0:
        raise ValueError(
            "ambient_k is an absolute temperature in kelvin (e.g. 293.15 for "
            f"20 C), got {ambient_k}."
        )
    return float(ambient_k)


def _in_plane_axis(plane: str) -> str:
    """Name of the in-plane axis of a vertical cut, ``y`` for ``x=...``."""
    return "y" if plane.strip().lower().startswith("x") else "x"


@dataclass(frozen=True)
class TemperatureMap:
    """Temperature rise on the thermal mesh at one heater power.

    A generic scalar field: one value per degree of freedom (DOF) of the
    heat solve, read anywhere in the thermal Window by its Lagrange
    interpolant.

    Attributes:
        mesh: The meshed thermal Cross-section (coordinates in um).
        values: Temperature rise above ambient at every DOF (K): the
            mesh vertices first, in ``mesh.mesh.p`` order, then the
            edge and interior points of a degree 2 or 3 solve.
        power_mw: Total heater power the field belongs to (mW).
    """

    mesh: CrossSectionMesh = field(repr=False)
    values: NDArray[np.float64] = field(repr=False)
    power_mw: float

    def at(
        self,
        position_um: ArrayLike,
        z_um: ArrayLike,
        *,
        ambient_k: float | None = None,
    ) -> Any:
        """Temperature rise (or temperature) at in-plane position(s) and height(s).

        Args:
            position_um: In-plane coordinate(s) in um -- ``y`` for an
                ``"x=..."`` cut, ``x`` for a ``"y=..."`` cut, as in
                :meth:`Waveguides.add <gsim.thermal.Waveguides.add>`.
            z_um: Height(s) in um, broadcast against ``position_um``.
            ambient_k: Ambient (heat-sink) temperature in K, usually 293.15.
                ``None`` returns the rise above ambient; a value returns
                the absolute temperature, rise plus ambient.

        Returns:
            The rise (or temperature) in K: a float for scalar inputs,
            else an array of the broadcast shape.

        Raises:
            ValueError: When a point lies outside the thermal Window, or
                ``ambient_k`` is not a positive temperature in K.
        """
        offset = _absolute(ambient_k)
        h, v = np.broadcast_arrays(
            np.asarray(position_um, dtype=float), np.asarray(z_um, dtype=float)
        )
        values = sample_temperature(
            self.mesh, self.values, np.vstack([h.ravel(), v.ravel()])
        )
        if h.ndim == 0:
            return float(values[0]) + offset
        return values.reshape(h.shape) + offset

    def peak(
        self, layer: str | None = None, *, ambient_k: float | None = None
    ) -> float:
        """Peak temperature rise (or temperature), over the Window or one Region.

        Args:
            layer: A Region of the thermal mesh -- a drawn stack layer
                such as the heater, or a slab by its material name.
                ``None`` takes the whole Window.
            ambient_k: Ambient (heat-sink) temperature in K, usually 293.15.
                ``None`` returns the rise above ambient; a value returns
                the absolute peak temperature, rise plus ambient.

        Returns:
            The largest DOF value of the rise (or temperature) in K.

        Raises:
            ValueError: When the mesh has no Region of that name, or
                ``ambient_k`` is not a positive temperature in K.
        """
        offset = _absolute(ambient_k)
        if layer is None:
            return float(self.values.max()) + offset
        m = self.mesh.mesh
        degree = lagrange_degree(m, self.values.size)
        dofs = np.unique(element_dofs(m, degree)[:, self.mesh.region_mask(layer)])
        return float(self.values[dofs].max()) + offset

    def plot(
        self,
        ax: Any = None,
        *,
        cmap: str = "inferno",
        xlim: tuple[float, float] | None = None,
        zlim: tuple[float, float] | None = None,
    ) -> Any:
        """Draw the temperature rise over the Cross-section.

        Args:
            ax: Matplotlib axes to draw on; a new figure when ``None``.
            cmap: Matplotlib colormap name.
            xlim: In-plane range ``(min, max)`` to show (um), e.g. to zoom
                on the heater; the whole Window when ``None``.
            zlim: Height range ``(min, max)`` to show (um); the whole
                Window when ``None``.

        Returns:
            The matplotlib axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation

        if ax is None:
            _, ax = plt.subplots()
        m, values = piecewise_linear(self.mesh.mesh, self.values)
        triangulation = Triangulation(m.p[0], m.p[1], m.t.T)
        image = ax.tripcolor(triangulation, values, shading="gouraud", cmap=cmap)
        ax.figure.colorbar(image, ax=ax, label="Temperature rise (K)")
        ax.set_xlabel(f"{_in_plane_axis(self.mesh.plane)} (µm)")
        ax.set_ylabel("z (µm)")
        ax.set_title(f"Temperature rise at {self.power_mw:.3g} mW")
        ax.set_aspect("equal")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if zlim is not None:
            ax.set_ylim(*zlim)
        return ax


@dataclass(frozen=True)
class StepResponse:
    """Response of the heated waveguides to a heater power switched on at t = 0.

    In this linear model the switch-off from steady state is the exact
    complement of the switch-on, so the fall time equals the rise time
    and one ``tau`` describes both.

    Attributes:
        time_us: Sample times from 0 to the duration (us).
        power_mw: Total heater power stepped on at t = 0 (mW).
        phase_shifter: Name of the waveguide whose phase is traced.
        phase_rad: Phase shift of the phase-shifter waveguide over time
            (rad), from first-order perturbation scaled so that its
            steady value matches the steady mode re-solve.
        steady_phase_rad: Steady-state phase shift at ``power_mw`` (rad).
        temperature_k: Temperature rise at each waveguide's core centre
            over time (K), by waveguide name.
        tau_us: 10-90 % rise time of the phase (us), the usual measured
            switching time.
        tau_1e_us: Time to reach 1 - 1/e (63 %) of the steady phase (us),
            for comparison with papers that quote a 1/e time constant.
    """

    time_us: NDArray[np.float64] = field(repr=False)
    power_mw: float
    phase_shifter: str
    phase_rad: NDArray[np.float64] = field(repr=False)
    steady_phase_rad: float
    temperature_k: dict[str, NDArray[np.float64]] = field(repr=False)
    tau_us: float
    tau_1e_us: float

    def plot(self, ax: Any = None) -> Any:
        """Draw the phase over time, marking the 10 % and 90 % points and tau_1e.

        Args:
            ax: Matplotlib axes to draw on; a new figure when ``None``.

        Returns:
            The matplotlib axes.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        t, s = _normalised_step(self.time_us, self.phase_rad)
        start, span = self.phase_rad[0], self.phase_rad[-1] - self.phase_rad[0]
        levels = np.array([0.1, 0.9])
        ax.plot(self.time_us, self.phase_rad, label=self.phase_shifter)
        ax.axhline(self.steady_phase_rad, color="0.6", linestyle=":", label="steady")
        ax.plot(
            [_crossing_time(t, s, level) for level in levels],
            start + levels * span,
            "o",
            color="C1",
            label="10 % and 90 %",
        )
        ax.plot(
            [self.time_us[0] + self.tau_1e_us],
            [start + (1.0 - math.exp(-1.0)) * span],
            "s",
            color="C2",
            label="1/e",
        )
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Phase shift (rad)")
        ax.set_title(
            f"Step to {self.power_mw:.3g} mW: tau(10-90 %) = {self.tau_us:.3g} µs, "
            f"tau(1/e) = {self.tau_1e_us:.3g} µs"
        )
        ax.legend()
        return ax


@dataclass(frozen=True)
class HeaterReport:
    """One table of a heater's figures of merit.

    Attributes:
        p_pi_mw: Total heater power for a pi phase shift (mW).
        efficiency_rad_per_mw: Phase per power, pi / P_pi (rad/mW).
        resistance_ohm: Heater resistance over its length (ohm).
        v_pi_v: Voltage for a pi phase shift (V).
        peak_temperature_k: Peak temperature rise in the heater at P_pi (K).
        crosstalk_rad: Phase on each victim waveguide at P_pi (rad).
        crosstalk_ratio: Victim phase over phase-shifter phase, by
            victim waveguide.
        thermal_resistance_k_per_mw: Core temperature rise per mW, by
            waveguide (K/mW).
        length_um: Heater length (um).
        wavelength_um: Vacuum wavelength (um).
        tau_us: 10-90 % switching time (us), when a step response was given.
        tau_1e_us: 1/e time constant (us), when a step response was given.
    """

    p_pi_mw: float
    efficiency_rad_per_mw: float
    resistance_ohm: float
    v_pi_v: float
    peak_temperature_k: float
    crosstalk_rad: dict[str, float]
    crosstalk_ratio: dict[str, float]
    thermal_resistance_k_per_mw: dict[str, float]
    length_um: float
    wavelength_um: float
    tau_us: float | None = None
    tau_1e_us: float | None = None

    def rows(self) -> list[tuple[str, str]]:
        """The table rows, ``(quantity with unit, formatted value)``.

        Returns:
            One row per figure, in reading order.
        """
        rows = [
            ("P_pi (mW)", f"{self.p_pi_mw:.4g}"),
            ("Efficiency (rad/mW)", f"{self.efficiency_rad_per_mw:.4g}"),
            ("Resistance (ohm)", f"{self.resistance_ohm:.4g}"),
            ("V_pi (V)", f"{self.v_pi_v:.4g}"),
            ("Peak heater rise at P_pi (K)", f"{self.peak_temperature_k:.4g}"),
        ]
        rows += [
            (f"Thermal resistance, {name} (K/mW)", f"{value:.4g}")
            for name, value in self.thermal_resistance_k_per_mw.items()
        ]
        rows += [
            (f"Crosstalk at P_pi, {name} (rad)", f"{value:.4g}")
            for name, value in self.crosstalk_rad.items()
        ]
        rows += [
            (f"Crosstalk ratio, {name}", f"{value:.4g}")
            for name, value in self.crosstalk_ratio.items()
        ]
        if self.tau_us is not None:
            rows.append(("tau 10-90 % (us)", f"{self.tau_us:.4g}"))
        if self.tau_1e_us is not None:
            rows.append(("tau 1/e (us)", f"{self.tau_1e_us:.4g}"))
        rows += [
            ("Heater length (um)", f"{self.length_um:.4g}"),
            ("Wavelength (um)", f"{self.wavelength_um:.4g}"),
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
        wavelength_um: Vacuum wavelength (um).
        length_um: Heater length (um).
        p_pi_mw: Total heater power for a pi phase shift of the phase
            shifter (mW).
        efficiency_rad_per_mw: pi / P_pi (rad/mW).
        resistance_ohm: Heater resistance, length over the summed
            conductance of its conductors (ohm).
        v_pi_v: Voltage across the heater at P_pi (V).
        crosstalk_rad: Phase on each victim waveguide when the heater
            drives P_pi (rad).
        crosstalk_ratio: Phase on each victim waveguide over the phase
            shifter's phase, at any power (linear model).
        thermal_resistance_k_per_mw: Temperature rise at each waveguide's
            core centre per mW of heater power (K/mW).
        dneff_per_mw: Effective-index shift per mW of heater power, from
            the mode re-solve, by waveguide (1/mW).
        dneff_first_order_per_mw: The same from first-order
            perturbation, the cross-check of the re-solve (1/mW).
    """

    phase_shifter: str
    wavelength_um: float
    length_um: float
    p_pi_mw: float
    efficiency_rad_per_mw: float
    resistance_ohm: float
    v_pi_v: float
    crosstalk_rad: dict[str, float]
    crosstalk_ratio: dict[str, float]
    thermal_resistance_k_per_mw: dict[str, float]
    dneff_per_mw: dict[str, float]
    dneff_first_order_per_mw: dict[str, float]
    heater_layers: tuple[str, ...] = field(repr=False)
    mesh: CrossSectionMesh = field(repr=False)
    rise_per_power_per_length: NDArray[np.float64] = field(repr=False)
    """DOF vector of the rise per heater power per length (K per W/m)."""

    def temperature(self, power_mw: float) -> TemperatureMap:
        """Temperature rise over the Cross-section at one heater power.

        Args:
            power_mw: Total heater power (mW).

        Returns:
            The rise on the thermal mesh (K).
        """
        scale = power_per_length(power_mw, self.length_um)
        return TemperatureMap(
            mesh=self.mesh,
            values=np.asarray(self.rise_per_power_per_length * scale, dtype=np.float64),
            power_mw=float(power_mw),
        )

    def phase(self, power_mw: float) -> dict[str, float]:
        """Phase shift of every waveguide at one heater power.

        Args:
            power_mw: Total heater power (mW).

        Returns:
            ``2 pi dn_eff L / lambda`` by waveguide name (rad).
        """
        return {
            name: 2.0 * math.pi * slope * power_mw * self.length_um / self.wavelength_um
            for name, slope in self.dneff_per_mw.items()
        }

    def drive_power(
        self, *, voltage_v: float | None = None, current_a: float | None = None
    ) -> float:
        """Total heater power of a voltage or current drive.

        The heater is the resistance over its length: ``V^2 / R`` or
        ``I^2 R``. Give exactly one drive.

        Args:
            voltage_v: Voltage across the heater (V).
            current_a: Current through the heater (A).

        Returns:
            The dissipated power (mW), for :meth:`phase` and
            :meth:`temperature`.

        Raises:
            ValueError: Unless exactly one drive is given.
        """
        given = [v for v in (voltage_v, current_a) if v is not None]
        if len(given) != 1:
            raise ValueError(
                "drive_power needs exactly one of voltage_v= (V) or current_a= (A)."
            )
        length_m = self.length_um * _UM_TO_M
        per_length = drive_power_per_length(
            self.resistance_ohm / length_m,
            voltage_V=voltage_v,
            current_A=current_a,
            length_um=self.length_um,
        )
        return per_length * length_m / _MW_TO_W

    @property
    def peak_temperature_k(self) -> float:
        """Peak temperature rise in the heater at P_pi (K).

        Taken over the heater layers drawn in the cut; a heater layer the
        cut misses carries no current there.
        """
        tmap = self.temperature(self.p_pi_mw)
        return max(
            tmap.peak(layer)
            for layer in self.heater_layers
            if layer in self.mesh.region_material
        )

    def report(self, step: StepResponse | None = None) -> HeaterReport:
        """Figures of merit in one table.

        Args:
            step: A step response to take tau from, if any.

        Returns:
            The :class:`HeaterReport`.
        """
        return HeaterReport(
            p_pi_mw=self.p_pi_mw,
            efficiency_rad_per_mw=self.efficiency_rad_per_mw,
            resistance_ohm=self.resistance_ohm,
            v_pi_v=self.v_pi_v,
            peak_temperature_k=self.peak_temperature_k,
            crosstalk_rad=dict(self.crosstalk_rad),
            crosstalk_ratio=dict(self.crosstalk_ratio),
            thermal_resistance_k_per_mw=dict(self.thermal_resistance_k_per_mw),
            length_um=self.length_um,
            wavelength_um=self.wavelength_um,
            tau_us=step.tau_us if step is not None else None,
            tau_1e_us=step.tau_1e_us if step is not None else None,
        )
