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

    def to_linear_solver_config(self) -> dict[str, object]:
        """Convert to Palace ``Solver.Linear`` config.

        Notes:
            - For ``solver_type='MUMPS'``, direct-solver defaults follow the
              existing Palace TODO template in this codebase.
            - ``preconditioner`` is only applied for ``solver_type='Default'``.
        """
        linear_conf: dict[str, object] = {
            "Type": self.solver_type,
            "KSPType": "GMRES",
            "Tol": self.tolerance,
            "MaxIts": self.max_iterations,
        }

        if self.solver_type == "MUMPS":
            linear_conf.update(
                {
                    "MaxIts": 1,
                    "MGMaxLevels": 1,
                    "EstimatorMaxIts": 0,
                    "EstimatorTol": 1e-6,
                    "DivFreeTol": 1e-6,
                    "DivFreeMaxIts": 0,
                    "PCMatReal": False,
                    "ComplexCoarseSolve": True,
                }
            )

        if self.solver_type == "Default" and self.preconditioner != "Default":
            linear_conf["Type"] = self.preconditioner

        return linear_conf

    def to_solver_config(self) -> dict[str, object]:
        """Convert to Palace ``Solver`` section config."""
        return {
            "Linear": self.to_linear_solver_config(),
            "Order": self.order,
            "Device": self.device,
        }

    def to_palace_config(self) -> dict:
        """Convert to the Palace ``Solver`` block, like ``to_solver_config``."""
        return self.to_solver_config()


class RefinementConfig(BaseModel):
    """Adaptive mesh refinement (AMR) controls for Palace.

    Maps onto ``config["Model"]["Refinement"]``. Palace refines the elements
    that carry most of its estimated error and re-solves, until the error norm
    falls below ``tol``, ``max_its`` passes have run, or the problem reaches
    ``max_dofs`` degrees of freedom. The defaults leave AMR off, as before.

    In a frequency sweep, Palace refines on the error indicator averaged over
    the sampled frequencies, so a narrow resonance can end up under-refined.

    Attributes:
        tol: Stop refining when the norm of the estimated error falls below
            this value.
        max_its: Maximum number of AMR passes. 0 disables AMR.
        uniform_levels: Levels of uniform refinement applied to the input mesh
            before solving. Each level splits every tetrahedron into eight, so
            a coarse mesh file can stand in for a much finer mesh.
        max_dofs: Maximum number of degrees of freedom AMR may reach
            (Palace's ``MaxSize``). None leaves Palace's default, no limit.
        update_fraction: Dörfler marking fraction, the share of the estimated
            error carried by the elements refined in each pass.
        nonconformal: Refine with hanging nodes instead of conformally.
        max_nc_levels: Maximum number of nonconformal refinement levels.
            0 means no limit.
        save_adapt_iterations: Keep the postprocessing output of every pass
            in an ``iterationX`` subdirectory.
        save_adapt_mesh: Save the final adapted mesh.

    Fields left as None are not written, so Palace applies its own defaults.
    """

    model_config = ConfigDict(validate_assignment=True)

    tol: float = Field(default=1e-2, gt=0)
    max_its: int = Field(default=0, ge=0)
    uniform_levels: int = Field(default=0, ge=0)
    max_dofs: int | None = Field(default=None, ge=0)
    update_fraction: float | None = Field(default=None, gt=0, lt=1)
    nonconformal: bool | None = None
    max_nc_levels: int | None = Field(default=None, ge=0)
    save_adapt_iterations: bool | None = None
    save_adapt_mesh: bool | None = None

    def to_palace_config(self) -> dict:
        """Convert to the Palace ``Model.Refinement`` block."""
        block: dict[str, int | float | bool] = {
            "UniformLevels": self.uniform_levels,
            "Tol": self.tol,
            "MaxIts": self.max_its,
        }
        optional = {
            "MaxSize": self.max_dofs,
            "UpdateFraction": self.update_fraction,
            "Nonconformal": self.nonconformal,
            "MaxNCLevels": self.max_nc_levels,
            "SaveAdaptIterations": self.save_adapt_iterations,
            "SaveAdaptMesh": self.save_adapt_mesh,
        }
        block.update({k: v for k, v in optional.items() if v is not None})
        return block


__all__ = [
    "NumericalConfig",
    "RefinementConfig",
]
