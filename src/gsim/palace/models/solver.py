"""Grouped numerical and problem-specific Palace solver settings."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gsim.palace.models.problems import (
    BoundaryModeConfig,
    DrivenConfig,
    EigenmodeConfig,
    ElectrostaticConfig,
)

_WARNING_SKIP_PREFIXES = (
    str(Path(__file__).resolve().parents[2]) + os.sep,
    str(Path(pydantic.__file__).resolve().parent) + os.sep,
)


def warn_legacy_solver_setting(
    previous: str,
    replacement: str,
    *,
    detail: str = "",
    category: type[Warning] = DeprecationWarning,
) -> None:
    """Warn at the caller of a legacy solver accessor or setter."""
    warnings.warn(
        f"{previous} is deprecated; use {replacement} instead. {detail}".rstrip(),
        category,
        skip_file_prefixes=_WARNING_SKIP_PREFIXES,
    )


class LinearSolverConfig(BaseModel):
    """Linear solve controls, shared by all Palace simulation types.

    ``tolerance`` is the relative residual tolerance of each linear solve.
    Eigenvalue convergence is configured separately under ``solver.eigenmode``.
    Direct backends are more robust but can use more memory than iterative solves.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    tolerance: float = Field(
        default=1e-6, gt=0, description="Linear solver relative residual tolerance."
    )
    max_iterations: int = Field(
        default=400, ge=1, description="Maximum Krylov solver iterations."
    )
    solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = Field(
        default="Default", description="Linear solver backend; Default auto-selects."
    )
    preconditioner: Literal["Default", "AMS", "BoomerAMG"] = Field(
        default="Default", description="Preconditioner used with the Default backend."
    )

    def to_palace_config(self) -> dict[str, Any]:
        """Convert to Palace's ``Solver.Linear`` block."""
        config: dict[str, object] = {
            "Type": self.solver_type,
            "KSPType": "GMRES",
            "Tol": self.tolerance,
            "MaxIts": self.max_iterations,
        }
        if self.solver_type == "MUMPS":
            config.update(
                MaxIts=1,
                MGMaxLevels=1,
                EstimatorMaxIts=0,
                EstimatorTol=1e-6,
                DivFreeTol=1e-6,
                DivFreeMaxIts=0,
                PCMatReal=False,
                ComplexCoarseSolve=True,
            )
        if self.solver_type == "Default" and self.preconditioner != "Default":
            config["Type"] = self.preconditioner
        return config


class SolverConfig(BaseModel):
    """Common Palace solver settings.

    ``order`` is the finite element field order, independent of mesh geometry
    order. Order 2 is the default; higher orders increase the solve size.
    Simulation-specific subclasses add the applicable problem configuration.
    Legacy flat linear keywords and properties remain supported.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    order: int = Field(
        default=2, ge=1, le=4, description="Finite element field order (1-4)."
    )
    device: Literal["CPU", "GPU"] = Field(
        default="CPU", description="Compute device; GPU requires a GPU-enabled Palace."
    )
    linear: LinearSolverConfig = Field(default_factory=LinearSolverConfig)

    @model_validator(mode="before")
    @classmethod
    def group_legacy_linear_settings(cls, values: Any) -> Any:
        """Accept the former NumericalConfig flat linear settings."""
        if isinstance(values, BaseModel) and not isinstance(values, cls):
            values = values.model_dump()
        if not isinstance(values, dict):
            return values
        values = values.copy()
        flat = {
            name: values.pop(name)
            for name in LinearSolverConfig.model_fields
            if name in values
        }
        if flat:
            warn_legacy_solver_setting("Flat linear solver settings", "solver.linear")
            linear = values.get("linear", {})
            if isinstance(linear, LinearSolverConfig):
                linear = linear.model_dump()
            if not isinstance(linear, dict):
                raise ValueError("linear must be a LinearSolverConfig or dictionary")
            if overlap := flat.keys() & linear.keys():
                raise ValueError(f"Linear settings supplied twice: {sorted(overlap)}")
            values["linear"] = {**linear, **flat}
        return values

    @property
    def tolerance(self) -> float:
        """Compatibility alias for ``linear.tolerance``."""
        warn_legacy_solver_setting("solver.tolerance", "solver.linear.tolerance")
        return self.linear.tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        """Set the linear tolerance through its deprecated flat name."""
        warn_legacy_solver_setting("solver.tolerance", "solver.linear.tolerance")
        self.linear.tolerance = value

    @property
    def max_iterations(self) -> int:
        """Compatibility alias for ``linear.max_iterations``."""
        warn_legacy_solver_setting(
            "solver.max_iterations", "solver.linear.max_iterations"
        )
        return self.linear.max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        """Set the iteration limit through its deprecated flat name."""
        warn_legacy_solver_setting(
            "solver.max_iterations", "solver.linear.max_iterations"
        )
        self.linear.max_iterations = value

    @property
    def solver_type(self) -> Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"]:
        """Compatibility alias for ``linear.solver_type``."""
        warn_legacy_solver_setting("solver.solver_type", "solver.linear.solver_type")
        return self.linear.solver_type

    @solver_type.setter
    def solver_type(
        self, value: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"]
    ) -> None:
        """Set the backend through its deprecated flat name."""
        warn_legacy_solver_setting("solver.solver_type", "solver.linear.solver_type")
        self.linear.solver_type = value

    @property
    def preconditioner(self) -> Literal["Default", "AMS", "BoomerAMG"]:
        """Compatibility alias for ``linear.preconditioner``."""
        warn_legacy_solver_setting(
            "solver.preconditioner", "solver.linear.preconditioner"
        )
        return self.linear.preconditioner

    @preconditioner.setter
    def preconditioner(self, value: Literal["Default", "AMS", "BoomerAMG"]) -> None:
        """Set the preconditioner through its deprecated flat name."""
        warn_legacy_solver_setting(
            "solver.preconditioner", "solver.linear.preconditioner"
        )
        self.linear.preconditioner = value

    def to_linear_solver_config(self) -> dict[str, object]:
        """Convert common settings to Palace's ``Solver.Linear`` block."""
        return self.linear.to_palace_config()

    def to_solver_config(self) -> dict[str, object]:
        """Convert common settings to Palace's ``Solver`` block."""
        return {
            "Linear": self.to_linear_solver_config(),
            "Order": self.order,
            "Device": self.device,
        }

    def to_palace_config(self) -> dict[str, Any]:
        """Convert all configured solver groups to Palace's ``Solver`` block."""
        config = self.to_solver_config()
        for name, palace_name in (
            ("driven", "Driven"),
            ("eigenmode", "Eigenmode"),
            ("electrostatic", "Electrostatic"),
            ("boundary_mode", "BoundaryMode"),
        ):
            if problem := getattr(self, name, None):
                config[palace_name] = problem.to_palace_config()
        return config


class DrivenSolverConfig(SolverConfig):
    """Common and frequency sweep settings for DrivenSim."""

    driven: DrivenConfig = Field(default_factory=DrivenConfig)


class EigenmodeSolverConfig(SolverConfig):
    """Common and eigenvalue settings for EigenmodeSim."""

    eigenmode: EigenmodeConfig = Field(default_factory=EigenmodeConfig)


class ElectrostaticSolverConfig(SolverConfig):
    """Common and electrostatic settings for ElectrostaticSim."""

    electrostatic: ElectrostaticConfig = Field(default_factory=ElectrostaticConfig)


class BoundaryModeSolverConfig(SolverConfig):
    """Common and cross-section mode settings for BoundaryModeSim."""

    boundary_mode: BoundaryModeConfig = Field(default_factory=BoundaryModeConfig)
