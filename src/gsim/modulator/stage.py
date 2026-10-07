"""The Stage lifecycle shared by every question a Study answers.

A Stage owns its configuration, its result, and nothing else. It is
configured by calling it (ADR 0001), solves lazily, caches what it solved,
and clears that cache — and every downstream Stage's — as soon as its
configuration changes, because Stages depend on each other in one
direction only.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, PrivateAttr

__all__ = ["Stage", "StageNotRunError"]


def _never_verbose() -> bool:
    """Default verbosity source: a Stage prints nothing on its own."""
    return False


class StageNotRunError(RuntimeError):
    """A Stage's result was read before the Stage ran."""


class Stage(BaseModel):
    """One question in a Study: its settings, its result, its lifecycle.

    Subclasses declare their settings as model fields and implement
    :meth:`_solve`.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    #: Name this Stage reports itself under.
    stage_name: ClassVar[str] = "stage"

    _result: Any = PrivateAttr(default=None)
    _has_run: bool = PrivateAttr(default=False)
    _elapsed_s: float | None = PrivateAttr(default=None)
    _downstream: list[Stage] = PrivateAttr(default_factory=list)
    _study: Any = PrivateAttr(default=None)
    _is_verbose: Callable[[], bool] = PrivateAttr(
        default_factory=lambda: _never_verbose
    )

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def wire(
        self,
        *,
        study: Any | None = None,
        downstream: Sequence[Stage] = (),
        is_verbose: Callable[[], bool] | None = None,
    ) -> None:
        """Attach this Stage to its Study and to the Stages after it.

        Args:
            study: The Study this Stage belongs to.
            downstream: Stages whose results this Stage's results feed,
                cleared whenever this Stage is re-configured.
            is_verbose: Source of the Study's verbosity, read per run.
        """
        if study is not None:
            self._study = study
        self._downstream = list(downstream)
        if is_verbose is not None:
            self._is_verbose = is_verbose

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def __call__(self, **updates: Any) -> Self:
        """Apply several settings at once, then invalidate what they change.

        Args:
            **updates: Settings to change, validated together.

        Returns:
            The Stage, so configuration can be chained.
        """
        unknown = [name for name in updates if name not in type(self).model_fields]
        if unknown:
            raise ValueError(
                f"Unknown setting(s) {unknown} for the {self.stage_name} stage. "
                f"Available: {sorted(type(self).model_fields)}"
            )
        validated = type(self).model_validate({**self.model_dump(), **updates})
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, getattr(validated, field_name))
        self.invalidate()
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Assigning a setting invalidates this Stage and its downstream."""
        super().__setattr__(name, value)
        if name in type(self).model_fields:
            self.invalidate()

    def _require_study(self) -> Any:
        """The Study this Stage belongs to.

        Raises:
            RuntimeError: When the Stage was built outside a Study, so
                nothing can be derived from a device description.
        """
        if self._study is None:
            raise RuntimeError(
                f"The {self.stage_name} stage is not attached to a Study; "
                f"build it with gsim.modulator.Study(...)."
            )
        return self._study

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def has_run(self) -> bool:
        """Whether this Stage holds a result for its current settings."""
        return self._has_run

    @property
    def result(self) -> Any:
        """The cached result.

        Raises:
            StageNotRunError: When the Stage has not run since it was last
                configured.
        """
        if not self._has_run:
            raise StageNotRunError(
                f"The {self.stage_name} stage has not run. Call "
                f"study.{self.stage_name}.run() first."
            )
        return self._result

    @property
    def elapsed_s(self) -> float | None:
        """Wall-clock seconds the last solve took, or None."""
        return self._elapsed_s

    def invalidate(self) -> None:
        """Drop this Stage's result and every downstream Stage's."""
        self._result = None
        self._has_run = False
        self._elapsed_s = None
        for stage in self._downstream:
            stage.invalidate()

    def seed(self, result: Any) -> None:
        """Hold *result* as if this Stage had just solved it.

        The one way a result reaches a Stage without a solve: a test
        canning an upstream answer, or a caller replaying one it has
        already paid for. Every downstream Stage's result is dropped
        first, exactly as a real run drops it, because they read what is
        being replaced.

        Args:
            result: The result to hold for the current settings.
        """
        for stage in self._downstream:
            stage.invalidate()
        self._result = result
        self._has_run = True
        self._elapsed_s = None

    def run(self, *, force: bool = False) -> Any:
        """Solve this Stage, or return what it already solved.

        Args:
            force: Solve again even when a cached result is available.
                Every downstream Stage's result is dropped first: they
                read what this Stage is about to replace.

        Returns:
            The Stage's result.
        """
        if self._has_run and not force:
            return self._result
        for stage in self._downstream:
            stage.invalidate()
        verbose = self._is_verbose()
        if verbose:
            print(f"[{self.stage_name}] start")  # noqa: T201
        started = time.perf_counter()
        result = self._solve()
        elapsed = time.perf_counter() - started
        self._result = result
        self._has_run = True
        self._elapsed_s = elapsed
        if verbose:
            print(f"[{self.stage_name}] done in {elapsed:.2f} s")  # noqa: T201
        return result

    def _solve(self) -> Any:
        """Answer this Stage's question. Implemented by each Stage."""
        raise NotImplementedError
