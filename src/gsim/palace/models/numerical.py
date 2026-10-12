"""Numerical solver configuration models for Palace simulations.

This module contains Pydantic models for numerical solver settings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gsim.palace.models.solver import (
    LinearSolverConfig,
    SolverConfig,
    warn_legacy_solver_setting,
)


class NumericalConfig(BaseModel):
    """Deprecated numerical settings; use SolverConfig with a linear group.

    Legacy flat constructor arguments, properties and serialization are retained.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Reuse the canonical fields, including their defaults and validation.
    order: int = deepcopy(SolverConfig.model_fields["order"])
    tolerance: float = deepcopy(LinearSolverConfig.model_fields["tolerance"])
    max_iterations: int = deepcopy(LinearSolverConfig.model_fields["max_iterations"])
    solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = deepcopy(
        LinearSolverConfig.model_fields["solver_type"]
    )
    preconditioner: Literal["Default", "AMS", "BoomerAMG"] = deepcopy(
        LinearSolverConfig.model_fields["preconditioner"]
    )
    device: Literal["CPU", "GPU"] = deepcopy(SolverConfig.model_fields["device"])

    def __init__(self, **data: Any) -> None:
        """Construct legacy settings and warn about their replacement."""
        warn_legacy_solver_setting("NumericalConfig", "SolverConfig")
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def flatten_grouped_linear_settings(cls, values: Any) -> Any:
        """Retain numerical values when copying a grouped solver dump."""
        if not isinstance(values, dict) or "linear" not in values:
            return values
        values = values.copy()
        linear = LinearSolverConfig.model_validate(values.pop("linear"))
        linear_values = linear.model_dump(exclude_unset=True)
        if overlap := values.keys() & linear_values.keys():
            raise ValueError(f"Linear settings supplied twice: {sorted(overlap)}")
        return {**values, **linear_values}

    def to_linear_solver_config(self) -> dict[str, object]:
        """Convert to the legacy Palace ``Solver.Linear`` block."""
        return LinearSolverConfig(
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            solver_type=self.solver_type,
            preconditioner=self.preconditioner,
        ).to_palace_config()

    def to_solver_config(self) -> dict[str, object]:
        """Convert to the legacy Palace ``Solver`` block."""
        return {
            "Linear": self.to_linear_solver_config(),
            "Order": self.order,
            "Device": self.device,
        }

    def to_palace_config(self) -> dict[str, object]:
        """Convert to the legacy Palace ``Solver`` block."""
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
