"""Guided-port mode selection for GDSFactory FDTD sources."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from gsim.common.config import MutableModel


class ModeSelection(MutableModel):
    """Select a guided mode by order or by TE/TM-like polarization."""

    kind: Literal["fundamental", "mode_order", "te_like", "tm_like"]
    mode_order: int | None = Field(default=None, ge=1)
    te_axis: Literal["x", "y", "z"] | None = None
    num_candidate_modes: int | None = Field(default=None, ge=1)
    min_polarization_fraction: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        """Require the settings used by the chosen selection method."""
        if self.kind == "mode_order":
            if self.mode_order is None:
                raise ValueError("mode_order is required for mode_order selection")
            if any(
                value is not None
                for value in (
                    self.te_axis,
                    self.num_candidate_modes,
                    self.min_polarization_fraction,
                )
            ):
                raise ValueError(
                    "mode_order selection cannot use polarization settings"
                )
        elif self.kind in {"te_like", "tm_like"}:
            if self.te_axis is None:
                raise ValueError("te_axis is required for TE/TM-like selection")
            if self.mode_order is not None:
                raise ValueError("TE/TM-like selection cannot use mode_order")
        elif any(
            value is not None
            for value in (
                self.mode_order,
                self.te_axis,
                self.num_candidate_modes,
                self.min_polarization_fraction,
            )
        ):
            raise ValueError("fundamental selection takes no additional settings")
        return self


__all__ = ["ModeSelection"]
