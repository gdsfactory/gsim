"""Numerical solver configuration models for Palace simulations.

This module contains Pydantic models for numerical solver settings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NumericalConfig(BaseModel):
    """Numerical solver configuration.

    Attributes:
        order: Finite element order (1-4)
        tolerance: Linear solver tolerance
        max_iterations: Maximum solver iterations
        solver_type: Linear solver type
        preconditioner: Preconditioner type
        device: Compute device (CPU or GPU)
        num_processors: Number of processors (None = auto)
    """

    model_config = ConfigDict(validate_assignment=True)

    # Element order
    order: int = Field(default=2, ge=1, le=4)

    # Linear solver
    tolerance: float = Field(default=1e-6, gt=0)
    max_iterations: int = Field(default=400, ge=1)
    solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = "Default"

    # Preconditioner
    preconditioner: Literal["Default", "AMS", "BoomerAMG"] = "Default"

    # Device
    device: Literal["CPU", "GPU"] = "CPU"

    # Partitioning
    num_processors: int | None = None  # None = auto

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        config = {
            "Order": self.order,
            "Solver": {
                "Tolerance": self.tolerance,
                "MaxIterations": self.max_iterations,
            },
        }

        if self.solver_type != "Default":
            config["Solver"]["Type"] = self.solver_type

        if self.preconditioner != "Default":
            config["Solver"]["Preconditioner"] = self.preconditioner

        return config


__all__ = [
    "NumericalConfig",
]
