"""The carriers Stage: Carrier maps turned into material response.

Both EM Stages read the same physics from the charge solve — the optical
Stage needs the plasma-dispersion index shift and free-carrier absorption,
the RF Stage needs the Drude conductivity — so the coupling is one
configurable Stage of its own rather than a setting duplicated in each.

Running it evaluates :mod:`gsim.common.carriers` at every node of every
Bias point, which a user can inspect and plot before paying for a mode
solve. The same coupling is reachable through :meth:`CarriersStage.response`
for carriers already transferred onto a downstream Stage's own mesh
(ADR 0002).

A Carrier map only carries the doped semiconductor Regions the charge
solve owns, so the coupling applies to every node it is given.

The published silicon fits are the defaults; foundry-calibrated
coefficients are substituted by configuring the Stage, never by
subclassing it. Nothing here needs a solver runtime beyond the charge
Stage's own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gsim.common.carriers import (
    MobilityModel,
    PlasmaDispersionModel,
    carrier_absorption_cm,
    carrier_conductivity,
    carrier_index_shift,
)
from gsim.common.sweep import ScalarSweep
from gsim.modulator.stage import Stage
from gsim.tcad.results import CarrierMap

if TYPE_CHECKING:
    from gsim.tcad.results import BiasSweepResult

__all__ = [
    "CarrierResponse",
    "CarrierResponseSweep",
    "CarriersStage",
    "MaterialResponse",
]


class MaterialResponse(BaseModel):
    """What free carriers do to a material, sample by sample.

    Attributes:
        index_shift: Refractive-index shift ``Delta n`` (negative for
            positive carrier densities).
        absorption_cm: Free-carrier absorption ``Delta alpha`` (cm^-1).
        conductivity_s_per_m: Drude conductivity (S/m).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index_shift: NDArray[np.float64]
    absorption_cm: NDArray[np.float64]
    conductivity_s_per_m: NDArray[np.float64]

    @model_validator(mode="after")
    def _same_length(self) -> MaterialResponse:
        """Reject a response whose three quantities disagree in length."""
        sizes = {
            "index_shift": self.index_shift.size,
            "absorption_cm": self.absorption_cm.size,
            "conductivity_s_per_m": self.conductivity_s_per_m.size,
        }
        if len(set(sizes.values())) > 1:
            raise ValueError(f"Response quantities disagree in length: {sizes}.")
        return self


class CarrierResponse(MaterialResponse):
    """The material response of one Bias point, across the Cross-section.

    The Carrier map it was evaluated on travels with it, so the sample
    coordinates and concentrations behind a response are one attribute
    away (``point.carriers.x_um``).

    Attributes:
        bias_v: Applied bias on the swept Contact (V).
        carriers: The Carrier map the response was evaluated on.
    """

    bias_v: float
    carriers: CarrierMap

    @model_validator(mode="after")
    def _matches_the_carrier_map(self) -> CarrierResponse:
        """Reject a response not sampled at the Carrier map's nodes."""
        nodes = self.carriers.electrons_cm3.size
        if self.index_shift.size != nodes:
            raise ValueError(
                f"Response has {self.index_shift.size} samples but the carrier "
                f"map has {nodes} nodes."
            )
        return self


class CarrierResponseSweep(ScalarSweep[CarrierResponse]):
    """The material response of every Bias point of a sweep.

    Attributes:
        contact: The Contact the charge sweep drove.
        points: One response per Bias point, in sweep order.
    """

    sweep_noun = "bias sweep"
    key_unit = "V"

    contact: str

    def _key(self, point: CarrierResponse) -> float:
        """A response is keyed on the bias its Bias point was solved at."""
        return point.bias_v

    @property
    def voltages(self) -> NDArray[np.float64]:
        """Applied biases (V) in sweep order."""
        return self.keys


class CarriersStage(Stage):
    """Plasma dispersion and mobilities, applied to the Bias sweep.

    Attributes:
        dispersion: Plasma-dispersion coefficients; defaults to the
            published Nedeljkovic fit at 1.55 um. Substitute a
            foundry-calibrated model by configuring this Stage.
        mobility: Mobility model for the Drude conductivity. Left unset,
            it is the charge Stage's, so the conductivity the RF Stage
            meshes and the series resistance the charge solve reports
            come from one model; set it here to move the RF conductivity
            alone, without paying for a charge solve.
    """

    stage_name: ClassVar[str] = "carriers"

    dispersion: PlasmaDispersionModel = Field(
        default_factory=PlasmaDispersionModel.nedeljkovic_1550
    )
    mobility: MobilityModel | None = None

    def effective_mobility(self) -> MobilityModel:
        """The mobility model the conductivity is evaluated with.

        Returns:
            This Stage's own model, else the charge Stage's, else — built
            outside a Study — the published silicon fit.
        """
        if self.mobility is not None:
            return self.mobility
        if self._study is not None:
            shared: MobilityModel = self._study.charge.mobility
            return shared
        return MobilityModel.masetti_silicon()

    def response(self, n_cm3: ArrayLike, p_cm3: ArrayLike) -> MaterialResponse:
        """Couple carrier concentrations to material response.

        The seam the EM Stages use on carriers transferred onto their own
        mesh: same coefficients, same mobilities, any sample points.

        Args:
            n_cm3: Electron concentrations (cm^-3).
            p_cm3: Hole concentrations (cm^-3).

        Returns:
            Index shift, absorption and conductivity at those samples.
        """
        n = np.asarray(n_cm3, dtype=np.float64)
        p = np.asarray(p_cm3, dtype=np.float64)
        return MaterialResponse(
            index_shift=np.asarray(
                carrier_index_shift(n, p, model=self.dispersion), dtype=np.float64
            ),
            absorption_cm=np.asarray(
                carrier_absorption_cm(n, p, model=self.dispersion), dtype=np.float64
            ),
            conductivity_s_per_m=np.asarray(
                carrier_conductivity(n, p, mobility=self.effective_mobility()),
                dtype=np.float64,
            ),
        )

    def _solve(self) -> CarrierResponseSweep:
        """Run the charge Stage if needed, then couple every Bias point."""
        sweep: BiasSweepResult = self._require_study().charge.run()
        points = []
        for point in sweep.points:
            response = self.response(
                point.carriers.electrons_cm3, point.carriers.holes_cm3
            )
            points.append(
                CarrierResponse(
                    bias_v=point.bias_v,
                    carriers=point.carriers,
                    index_shift=response.index_shift,
                    absorption_cm=response.absorption_cm,
                    conductivity_s_per_m=response.conductivity_s_per_m,
                )
            )
        return CarrierResponseSweep(contact=sweep.contact, points=points)
