"""Base model for the concern-grouped configuration of a simulation.

``gsim.fdtd`` and ``gsim.tcad`` configure a simulation through a few
concern objects (geometry, domain, materials, solver, ...). Each is a
validated pydantic model that can be set one attribute at a time or
several at once by calling it.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict

__all__ = ["MutableModel"]


class MutableModel(BaseModel):
    """Validated model that supports concise batch updates."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    def __call__(self, **updates: Any) -> Self:
        """Validate and apply several settings atomically.

        The current values are passed back as they are, not dumped, so a
        field holding an arbitrary object (a component, a stack) survives.
        """
        updated = type(self).model_validate({**dict(self), **updates})
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, getattr(updated, field_name))
        return self
