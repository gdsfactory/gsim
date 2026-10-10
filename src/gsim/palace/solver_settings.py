"""Solver configuration helpers and compatibility aliases for Palace sims."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from gsim.palace.models.numerical import NumericalConfig
from gsim.palace.models.problems import (
    BoundaryModeConfig,
    DrivenConfig,
    EigenmodeConfig,
    ElectrostaticConfig,
)
from gsim.palace.models.solver import (
    LinearSolverConfig,
    SolverConfig,
    warn_legacy_solver_setting,
)

_DEFAULT_SOLVER = SolverConfig()
_PROBLEM_GROUPS = ("driven", "eigenmode", "electrostatic", "boundary_mode")


class _UnsetOrder(Enum):
    """Distinguish an omitted legacy order from an explicit choice."""

    DEFAULT = "default"


class SolverSettingsMixin:
    """Provide common solver setters and legacy configuration accessors."""

    solver: SolverConfig

    @model_validator(mode="before")
    @classmethod
    def group_legacy_solver_settings(cls, values: Any) -> Any:
        """Move legacy constructor arguments into their solver groups."""
        if not isinstance(values, dict):
            return values
        values = values.copy()
        if "numerical" in values:
            warn_legacy_solver_setting("numerical constructor argument", "solver")
            if "solver" in values:
                raise ValueError("Use either solver or numerical, not both")
            values["solver"] = values.pop("numerical")
        legacy = {name: values.pop(name) for name in _PROBLEM_GROUPS if name in values}
        if legacy:
            solver = values.get("solver", {})
            if isinstance(solver, BaseModel):
                solver = solver.model_dump()
            if not isinstance(solver, dict):
                raise ValueError("solver must be a SolverConfig or dictionary")
            solver = solver.copy()
            for name, problem in legacy.items():
                if problem is None:
                    continue
                warn_legacy_solver_setting(
                    f"{name} constructor argument", f"solver.{name}"
                )
                if name in solver:
                    raise ValueError(f"{name} settings supplied twice")
                solver[name] = problem
            values["solver"] = solver
        return values

    def set_solver(
        self,
        *,
        order: int = _DEFAULT_SOLVER.order,
        tolerance: float = _DEFAULT_SOLVER.linear.tolerance,
        max_iterations: int = _DEFAULT_SOLVER.linear.max_iterations,
        initial_guess: bool | None = _DEFAULT_SOLVER.linear.initial_guess,
        solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = (
            _DEFAULT_SOLVER.linear.solver_type
        ),
        preconditioner: Literal["Default", "AMS", "BoomerAMG"] = (
            _DEFAULT_SOLVER.linear.preconditioner
        ),
        device: Literal["CPU", "GPU"] = _DEFAULT_SOLVER.device,
    ) -> None:
        """Set common solver controls, preserving problem-specific settings.

        Defaults match ``SolverConfig()``: order 2, linear tolerance 1e-6,
        400 iterations, default backend/preconditioner, and CPU. Set individual
        controls directly through ``sim.solver`` to retain other common values.
        Initial guesses follow Palace's default unless set explicitly.

        Example:
            >>> sim.set_solver(order=1, tolerance=1e-6)
            >>> sim.solver.eigenmode.target = 4e9
        """
        self._replace_common_solver(
            SolverConfig(
                order=order,
                device=device,
                linear=LinearSolverConfig(
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    initial_guess=initial_guess,
                    solver_type=solver_type,
                    preconditioner=preconditioner,
                ),
            )
        )

    def set_numerical(
        self,
        *,
        order: int | _UnsetOrder = _UnsetOrder.DEFAULT,
        tolerance: float = _DEFAULT_SOLVER.linear.tolerance,
        max_iterations: int = _DEFAULT_SOLVER.linear.max_iterations,
        initial_guess: bool | None = _DEFAULT_SOLVER.linear.initial_guess,
        solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = (
            _DEFAULT_SOLVER.linear.solver_type
        ),
        preconditioner: Literal["Default", "AMS", "BoomerAMG"] = (
            _DEFAULT_SOLVER.linear.preconditioner
        ),
        device: Literal["CPU", "GPU"] = _DEFAULT_SOLVER.device,
    ) -> None:
        """Deprecated alias for set_solver, with the same order-2 defaults.

        Omitting order emits a visible migration warning because this setter
        previously defaulted to order 1. Pass order=1 to retain that behavior.
        """
        if isinstance(order, _UnsetOrder):
            # Temporary notice for the changed default in the legacy setter.
            warn_legacy_solver_setting(
                "set_numerical()",
                "set_solver()",
                detail=(
                    "Omitting order now uses order=2; previously it used order=1. "
                    "Pass order=1 to preserve the previous field order."
                ),
                category=FutureWarning,
            )
            order = _DEFAULT_SOLVER.order
        else:
            warn_legacy_solver_setting("set_numerical()", "set_solver()")
        self.set_solver(
            order=order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_guess=initial_guess,
            solver_type=solver_type,
            preconditioner=preconditioner,
            device=device,
        )

    @property
    def numerical(self) -> SolverConfig:
        """Compatibility alias for the common controls in ``solver``."""
        warn_legacy_solver_setting("sim.numerical", "sim.solver")
        return self.solver

    @numerical.setter
    def numerical(self, value: SolverConfig | NumericalConfig | dict[str, Any]) -> None:
        """Replace common controls through the deprecated numerical name."""
        warn_legacy_solver_setting("sim.numerical", "sim.solver")
        self._replace_common_solver(value)

    def _replace_common_solver(
        self, value: SolverConfig | NumericalConfig | dict[str, Any]
    ) -> None:
        """Replace common controls atomically, retaining problem settings."""
        if isinstance(value, SolverConfig):
            value = {name: getattr(value, name) for name in SolverConfig.model_fields}
        elif isinstance(value, NumericalConfig):
            value = value.model_dump()
        common = SolverConfig.model_validate(value)
        # Replace atomically so callers retaining the old object can restore it.
        # Keep problem model objects so references to those settings stay live.
        settings = {
            name: getattr(self.solver, name) for name in type(self.solver).model_fields
        }
        settings.update(
            {name: getattr(common, name) for name in SolverConfig.model_fields}
        )
        self.solver = type(self.solver).model_validate(settings)

    def _get_problem_settings(self, name: str) -> Any:
        """Return a problem group or None for an inapplicable simulation type."""
        return getattr(self.solver, name, None)

    def _set_problem_settings(
        self, name: str, value: BaseModel | dict[str, Any] | None
    ) -> None:
        """Validate and replace an applicable problem group."""
        if name not in type(self.solver).model_fields:
            if value is not None:
                raise ValueError(f"{name} settings do not apply to this simulation")
            return
        setattr(self.solver, name, value)

    @property
    def driven(self) -> DrivenConfig | None:
        """Compatibility alias for ``solver.driven`` (None for other modes)."""
        warn_legacy_solver_setting("sim.driven", "sim.solver.driven")
        return self._get_problem_settings("driven")

    @driven.setter
    def driven(self, value: DrivenConfig | dict[str, Any]) -> None:
        """Replace driven settings through their deprecated top-level name."""
        warn_legacy_solver_setting("sim.driven", "sim.solver.driven")
        self._set_problem_settings("driven", value)

    @property
    def eigenmode(self) -> EigenmodeConfig | None:
        """Compatibility alias for ``solver.eigenmode`` (None for other modes)."""
        warn_legacy_solver_setting("sim.eigenmode", "sim.solver.eigenmode")
        return self._get_problem_settings("eigenmode")

    @eigenmode.setter
    def eigenmode(self, value: EigenmodeConfig | dict[str, Any]) -> None:
        """Replace eigenmode settings through their deprecated top-level name."""
        warn_legacy_solver_setting("sim.eigenmode", "sim.solver.eigenmode")
        self._set_problem_settings("eigenmode", value)

    @property
    def electrostatic(self) -> ElectrostaticConfig | None:
        """Compatibility alias for ``solver.electrostatic`` (None for other modes)."""
        warn_legacy_solver_setting("sim.electrostatic", "sim.solver.electrostatic")
        return self._get_problem_settings("electrostatic")

    @electrostatic.setter
    def electrostatic(self, value: ElectrostaticConfig | dict[str, Any]) -> None:
        """Replace electrostatic settings through their deprecated top-level name."""
        warn_legacy_solver_setting("sim.electrostatic", "sim.solver.electrostatic")
        self._set_problem_settings("electrostatic", value)

    @property
    def boundary_mode(self) -> BoundaryModeConfig | None:
        """Compatibility alias for ``solver.boundary_mode`` (None for other modes)."""
        warn_legacy_solver_setting("sim.boundary_mode", "sim.solver.boundary_mode")
        return self._get_problem_settings("boundary_mode")

    @boundary_mode.setter
    def boundary_mode(self, value: BoundaryModeConfig | dict[str, Any]) -> None:
        """Replace boundary-mode settings through their deprecated top-level name."""
        warn_legacy_solver_setting("sim.boundary_mode", "sim.solver.boundary_mode")
        self._set_problem_settings("boundary_mode", value)
