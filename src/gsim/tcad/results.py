"""Result containers for the charge-transport solve.

A 2D DEVSIM device is one cm deep, so extensive quantities (currents,
charges, capacitances) are per cm of device depth. Coordinates are
converted back to the gsim mesh unit (um).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from gsim.common.sweep import ScalarSweep

if TYPE_CHECKING:
    from gsim.common.transmission_line import JunctionBranch

__all__ = ["BiasPoint", "BiasSweepResult", "CarrierMap"]

#: A DEVSIM 2D device is one cm deep, so its extensive quantities come
#: back per cm of that depth; this is the conversion to per meter of
#: device length the rest of gsim works in.
PER_CM_TO_PER_M: float = 1e2


class CarrierMap(BaseModel):
    """Node-wise solution fields on the cross-section mesh.

    The concentrations are what every downstream consumer reads. The
    potential and the net doping are diagnostics the charge solve reports
    alongside them; a map built anywhere else — a canned sweep, a
    synthetic profile — leaves them out.

    Attributes:
        x_um: In-plane node coordinates (um).
        y_um: Vertical node coordinates (um).
        region: Per-node mesh region name.
        electrons_cm3: Electron concentration n(x, y) (cm^-3).
        holes_cm3: Hole concentration p(x, y) (cm^-3).
        potential_v: Electrostatic potential (V), when the solve
            reported it.
        net_doping_cm3: Net doping (donors - acceptors, cm^-3), when the
            solve reported it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x_um: NDArray[np.float64]
    y_um: NDArray[np.float64]
    region: list[str]
    electrons_cm3: NDArray[np.float64]
    holes_cm3: NDArray[np.float64]
    potential_v: NDArray[np.float64] | None = None
    net_doping_cm3: NDArray[np.float64] | None = None


class BiasPoint(BaseModel):
    """Solved state of the device at one bias voltage.

    Attributes:
        bias_v: Applied bias on the swept contact (V).
        carriers: Node-wise carrier maps.
        currents_a_per_cm: Total terminal current per contact
            (electron + hole, A per cm of depth).
        charge_c_per_cm: Contact charge on the swept contact
            (C per cm of depth).
        capacitance_f_per_cm: Small-signal capacitance |dQ/dV| at this
            bias (F per cm of depth).
        admittance_s_per_cm: Complex small-signal terminal admittance of
            the swept contact at :attr:`admittance_freq_hz`
            (S per cm of depth).
        admittance_freq_hz: Frequency the admittance's AC solve ran at
            (Hz) — above the quasi-static one, so the capacitive current
            dominates the junction leakage in ``Re(Y)``; zero when no AC
            solve produced this point.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bias_v: float
    carriers: CarrierMap
    currents_a_per_cm: dict[str, float] = Field(default_factory=dict)
    charge_c_per_cm: float = 0.0
    capacitance_f_per_cm: float = 0.0
    admittance_s_per_cm: complex = 0.0j
    admittance_freq_hz: float = 0.0

    @property
    def capacitance_f_per_m(self) -> float:
        """Small-signal capacitance per meter of device length (F/m)."""
        return self.capacitance_f_per_cm * PER_CM_TO_PER_M

    @property
    def admittance_s_per_m(self) -> complex:
        """Small-signal admittance per meter of device length (S/m)."""
        return self.admittance_s_per_cm * PER_CM_TO_PER_M

    def junction_branch(self) -> JunctionBranch:
        """Fit the series-RC junction branch to this point's admittance.

        The lumped shunt model the standard loaded-line workflow inserts
        per unit length of the Traveling-wave electrode: the junction
        capacitance behind the series resistance of the doped slab.

        Returns:
            The fitted :class:`~gsim.common.transmission_line.JunctionBranch`, both
            fields floats.

        Raises:
            ValueError: When this point holds no small-signal admittance,
                or one a series RC cannot represent.
        """
        from gsim.common.transmission_line import (
            JunctionBranch,
            series_rc_from_admittance,
        )

        if self.admittance_freq_hz <= 0 or self.admittance_s_per_cm == 0:
            raise ValueError(
                f"The bias point at {self.bias_v:g} V holds no small-signal "
                "admittance; re-solve it with a charge backend recent enough "
                "to report one."
            )
        r_s, c_j = series_rc_from_admittance(
            self.admittance_s_per_m, freq_hz=self.admittance_freq_hz
        )
        return JunctionBranch(r_s_ohm_m=float(r_s), c_j_f_per_m=float(c_j))


class BiasSweepResult(ScalarSweep[BiasPoint]):
    """Ordered collection of solved bias points from a voltage sweep."""

    sweep_noun = "bias sweep"
    key_unit = "V"

    contact: str

    def _key(self, point: BiasPoint) -> float:
        """A Bias point is keyed on the bias it was solved at."""
        return point.bias_v

    @property
    def voltages(self) -> NDArray[np.float64]:
        """Applied biases (V) in sweep order."""
        return self.keys

    @property
    def capacitance_f_per_cm(self) -> NDArray[np.float64]:
        """Small-signal C(V) per cm of depth in sweep order."""
        return np.asarray(
            [p.capacitance_f_per_cm for p in self.points], dtype=np.float64
        )

    @property
    def capacitance_f_per_m(self) -> NDArray[np.float64]:
        """Small-signal C(V) per meter of device length in sweep order."""
        return np.asarray(self.capacitance_f_per_cm * PER_CM_TO_PER_M, dtype=np.float64)

    @property
    def admittance_s_per_cm(self) -> NDArray[np.complex128]:
        """Small-signal terminal admittance per cm of depth in sweep order."""
        return np.asarray(
            [p.admittance_s_per_cm for p in self.points], dtype=np.complex128
        )

    def junction_branch(self) -> JunctionBranch:
        """The series-RC junction branch fitted at every Bias point.

        Returns:
            A :class:`~gsim.common.transmission_line.JunctionBranch` of arrays in
            sweep order.
        """
        from gsim.common.transmission_line import JunctionBranch

        fitted = [p.junction_branch() for p in self.points]
        return JunctionBranch(
            r_s_ohm_m=np.asarray([r for r, _ in fitted], dtype=np.float64),
            c_j_f_per_m=np.asarray([c for _, c in fitted], dtype=np.float64),
        )
