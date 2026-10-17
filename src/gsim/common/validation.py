"""Spec validation shared by the Backends' pydantic models.

A Backend's configuration models each validate the same handful of
shapes, and the interesting one is the closed interval a window is given
as: ``(min, max)``, finite and ascending. Written as a plain check it is
a helper every model has to remember to call, once per field; written as
an annotated type it is part of the field's declaration, so a model
cannot carry the field and forget the rule.

The messages are the ones the hand-written checks raised, minus the
field name, which pydantic puts in the error's ``loc`` instead. Assert
on the message body rather than on a whole rendered error.

This module is imported by module path; nothing joins
``gsim.common.__all__``.
"""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import AfterValidator

__all__ = ["AscendingInterval"]


def _ascending(interval: tuple[float, float]) -> tuple[float, float]:
    """Reject a non-finite or descending ``(min, max)`` interval."""
    lo, hi = interval
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("bounds must be finite")
    if hi <= lo:
        raise ValueError("must be an ascending (min, max) interval")
    return interval


#: A closed ``(min, max)`` interval in the model's own units: finite,
#: and ascending with room in it — an empty interval selects nothing, so
#: a caller that meant one is asking for something else.
AscendingInterval = Annotated[tuple[float, float], AfterValidator(_ascending)]
