"""Numerical solver configuration models for Palace simulations.

This module contains Pydantic models for numerical solver settings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NumericalConfig(BaseModel):
    """Numerical solver configuration for Palace FEM simulations.

    Attributes:
        order: Finite element polynomial order (1-4). Higher order means more
            accurate field approximation per mesh element at higher cost.
            order=1: fast, low accuracy. order=2: good balance (default).
            order=3-4: high accuracy, significantly more DOFs. Increasing order
            can reduce lumped port reflection artifacts and is often more
            cost-effective than mesh refinement for smooth geometries.
        tolerance: Linear solver relative residual convergence tolerance.
            Tighter tolerance (e.g. 1e-8) gives more accurate solves at the
            cost of more iterations. Default 1e-6 is suitable for most cases.
        max_iterations: Maximum Krylov solver iterations before giving up.
            Increase if you see "solver did not converge" warnings.
        solver_type: Linear solver / preconditioner backend.
            "Default" auto-selects (AMS for curl-curl, sparse direct for
            frequency domain). "SuperLU", "STRUMPACK", "MUMPS" are sparse
            direct solvers — more robust but use more memory.
        preconditioner: Preconditioner for iterative solves.
            "AMS" (Auxiliary-space Maxwell Solver) is best for EM problems.
            "BoomerAMG" is an algebraic multigrid alternative.
        device: Compute device. "GPU" enables GPU-accelerated assembly and
            solves if Palace was built with GPU support.
        num_processors: Number of MPI processes for parallel execution.
            None = auto-detect based on cloud instance.
    """

    model_config = ConfigDict(validate_assignment=True)

    order: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Finite element polynomial order. Higher order = more accurate "
        "fields per element but more expensive. order=1: fast/low accuracy, "
        "order=2: good balance (default), order=3-4: high accuracy. "
        "Increasing order can reduce lumped port reflection artifacts.",
    )

    tolerance: float = Field(
        default=1e-6,
        gt=0,
        description="Linear solver relative residual convergence tolerance. "
        "Tighter (e.g. 1e-8) gives more accurate solves at higher cost.",
    )
    max_iterations: int = Field(
        default=400,
        ge=1,
        description="Maximum Krylov solver iterations. Increase if solver "
        "does not converge.",
    )
    solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = Field(
        default="Default",
        description="Linear solver backend. 'Default' auto-selects. "
        "Direct solvers (SuperLU, STRUMPACK, MUMPS) are more robust "
        "but use more memory.",
    )

    preconditioner: Literal["Default", "AMS", "BoomerAMG"] = Field(
        default="Default",
        description="Preconditioner type. 'AMS' is best for EM curl-curl "
        "problems. 'BoomerAMG' is an algebraic multigrid alternative.",
    )

    device: Literal["CPU", "GPU"] = Field(
        default="CPU",
        description="Compute device. 'GPU' enables GPU-accelerated assembly "
        "and solves if Palace was built with GPU support.",
    )

    num_processors: int | None = Field(
        default=None,
        description="Number of MPI processes. None = auto-detect.",
    )

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        solver_config: dict[str, str | int | float] = {
            "Tolerance": self.tolerance,
            "MaxIterations": self.max_iterations,
        }

        if self.solver_type != "Default":
            solver_config["Type"] = self.solver_type

        if self.preconditioner != "Default":
            solver_config["Preconditioner"] = self.preconditioner

        return {
            "Order": self.order,
            "Solver": solver_config,
        }


__all__ = [
    "NumericalConfig",
]
