"""Grouped numerical and problem-specific Palace solver settings."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, ClassVar, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gsim.palace.models.problems import (
    BoundaryModeConfig,
    DrivenConfig,
    EigenmodeConfig,
    ElectrostaticConfig,
    MagnetostaticConfig,
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


# Palace's error estimator keys in its read order, written last in Solver.Linear.
_ESTIMATOR_KEYS = ("EstimatorTol", "EstimatorMaxIts", "EstimatorMG")


class LinearSolverConfig(BaseModel):
    """Linear solve controls, shared by all Palace simulation types.

    ``tolerance`` is the relative residual tolerance of each linear solve.
    Eigenvalue convergence is configured separately under ``solver.eigenmode``.
    Direct backends are more robust but can use more memory than iterative solves.

    The ``estimator_*`` fields control the error estimator, Palace's AMR
    indicator, not the adaptive frequency sweep tolerance. None, the default,
    lets gsim choose by AMR state: a cheap setting when AMR is off and Palace's
    default when it is on. Assign None to return to automatic. Settings passed
    through the deprecated ``NumericalConfig`` always use automatic.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    tolerance: float = Field(
        default=1e-6, gt=0, description="Linear solver relative residual tolerance."
    )
    max_iterations: int = Field(
        default=400, ge=1, description="Maximum Krylov solver iterations."
    )
    solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = Field(
        default="MUMPS", description="Linear solver backend; Default auto-selects."
    )
    preconditioner: Literal["Default", "AMS", "BoomerAMG"] = Field(
        default="Default", description="Preconditioner used with the Default backend."
    )
    mg_max_levels: int = Field(
        default=100,
        ge=1,
        description="Maximum multigrid levels; 1 uses a full-system solve.",
    )
    pc_mat_real: bool = Field(
        default=False,
        description="Use a real approximation for the preconditioner matrix.",
    )
    complex_coarse_solve: bool = Field(
        default=True,
        description="Use a complex coarse solve for complex-valued problems.",
    )
    estimator_tolerance: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Relative tolerance of the error estimator (the AMR indicator), not "
            "the adaptive sweep tolerance. None chooses by AMR state; assign "
            "None to return to automatic."
        ),
    )
    estimator_max_iterations: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Error estimator (AMR indicator) iterations, not an adaptive sweep "
            "setting; 0 skips its solve. None chooses by AMR state; assign None "
            "to return to automatic."
        ),
    )
    estimator_mg: bool | None = Field(
        default=None,
        description=(
            "Use multigrid for the error estimator (the AMR indicator), not the "
            "adaptive sweep. None chooses by AMR state; assign None to return "
            "to automatic."
        ),
    )

    def to_palace_config(self) -> dict[str, Any]:
        """Convert to Palace's ``Solver.Linear`` block."""
        config: dict[str, object] = {
            "Type": self.solver_type,
            "KSPType": "GMRES",
            "Tol": self.tolerance,
            "MaxIts": self.max_iterations,
            "MGMaxLevels": self.mg_max_levels,
            "PCMatReal": self.pc_mat_real,
            "ComplexCoarseSolve": self.complex_coarse_solve,
        }
        if self.solver_type == "Default" and self.preconditioner != "Default":
            config["Type"] = self.preconditioner
        # Unset keys are resolved when the config is written.
        estimator = (
            self.estimator_tolerance,
            self.estimator_max_iterations,
            self.estimator_mg,
        )
        config.update(
            {
                key: value
                for key, value in zip(_ESTIMATOR_KEYS, estimator, strict=True)
                if value is not None
            }
        )
        return config


# Settings once accepted flat on NumericalConfig; newer linear fields are not.
_LEGACY_FLAT_LINEAR_FIELDS = (
    "tolerance",
    "max_iterations",
    "solver_type",
    "preconditioner",
    "mg_max_levels",
    "pc_mat_real",
    "complex_coarse_solve",
)


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
    # Linear settings this solver class uses unless the caller sets them.
    _linear_defaults: ClassVar[dict[str, Any]] = {}

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
            for name in _LEGACY_FLAT_LINEAR_FIELDS
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

    @model_validator(mode="before")
    @classmethod
    def apply_class_linear_defaults(cls, values: Any) -> Any:
        """Use the class linear defaults for settings the caller leaves unset."""
        if not cls._linear_defaults:
            return values
        if isinstance(values, SolverConfig) and not isinstance(values, cls):
            # Preserve nested assignments and which linear fields were explicit.
            values = {**values.model_dump(), "linear": values.linear}
        elif isinstance(values, BaseModel) and not isinstance(values, cls):
            values = values.model_dump(exclude_unset=True)
        if not isinstance(values, dict):
            return values
        # Legacy flat keys are explicit; they are grouped by the next validator.
        defaults = {
            name: value
            for name, value in cls._linear_defaults.items()
            if name not in values
        }
        linear = values.get("linear", {})
        if isinstance(linear, LinearSolverConfig):
            unset = {
                name: value
                for name, value in defaults.items()
                if name not in linear.model_fields_set
            }
            if unset:
                return {**values, "linear": linear.model_copy(update=unset)}
            return values
        if isinstance(linear, dict):
            return {**values, "linear": {**defaults, **linear}}
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
            ("magnetostatic", "Magnetostatic"),
        ):
            if problem := getattr(self, name, None):
                config[palace_name] = problem.to_palace_config()
        return config


class DrivenSolverConfig(SolverConfig):
    """Common and frequency sweep settings for DrivenSim."""

    driven: DrivenConfig = Field(default_factory=DrivenConfig)


_EIGENMODE_LINEAR_DEFAULTS: dict[str, Any] = {"tolerance": 1e-9}
# Palace resolves magnetostatic Default to AMS + CG on the singular curl-curl
# operator; direct backends are not documented for it.
_MAGNETOSTATIC_LINEAR_DEFAULTS: dict[str, Any] = {
    "solver_type": "Default",
    "preconditioner": "AMS",
    "tolerance": 1e-8,
}


class EigenmodeSolverConfig(SolverConfig):
    """Common and eigenvalue settings for EigenmodeSim."""

    _linear_defaults: ClassVar[dict[str, Any]] = _EIGENMODE_LINEAR_DEFAULTS

    linear: LinearSolverConfig = Field(
        default_factory=lambda: LinearSolverConfig(**_EIGENMODE_LINEAR_DEFAULTS)
    )
    eigenmode: EigenmodeConfig = Field(default_factory=EigenmodeConfig)


class ElectrostaticSolverConfig(SolverConfig):
    """Common and electrostatic settings for ElectrostaticSim."""

    electrostatic: ElectrostaticConfig = Field(default_factory=ElectrostaticConfig)


class BoundaryModeSolverConfig(SolverConfig):
    """Common and cross-section mode settings for BoundaryModeSim."""

    boundary_mode: BoundaryModeConfig = Field(default_factory=BoundaryModeConfig)


class MagnetostaticSolverConfig(SolverConfig):
    """Common and magnetostatic settings for MagnetostaticSim.

    The linear solve defaults to Palace's magnetostatic choice: the AMS
    preconditioner with conjugate gradients at a 1e-8 relative tolerance.
    """

    _linear_defaults: ClassVar[dict[str, Any]] = _MAGNETOSTATIC_LINEAR_DEFAULTS

    linear: LinearSolverConfig = Field(
        default_factory=lambda: LinearSolverConfig(**_MAGNETOSTATIC_LINEAR_DEFAULTS)
    )
    magnetostatic: MagnetostaticConfig = Field(default_factory=MagnetostaticConfig)

    def to_linear_solver_config(self) -> dict[str, object]:
        """Convert to Palace's ``Solver.Linear`` block with a CG Krylov solve."""
        return {**super().to_linear_solver_config(), "KSPType": "CG"}
