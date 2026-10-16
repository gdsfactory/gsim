"""A sweep of solved points over one scalar key.

A Backend that walks a parameter — a charge solve stepping the bias on a
Contact, the material response derived from it — comes back with an
ordered list of points and is asked two things about them: the axis it
swept, and the point it solved at one value of that axis. Neither is
interesting on its own, and both are easy to write slightly differently
twice, so :class:`ScalarSweep` holds them: the key array in sweep order,
and the tolerant lookup that says which values were visited when the one
asked for was not.

What a sweep carries besides its points is its own: a charge sweep's
capacitance and admittance arrays have nothing to do with a response
sweep's, and neither belongs here.

The lookup is by tolerance rather than by equality because a bias
travels from a caller's request through a solver's settings file and
back out of its results, and a float that survives that trip is not the
one that started it.

This module is imported by module path; nothing joins
``gsim.common.__all__``.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["BIAS_TOL_V", "ScalarSweep"]

#: Biases this far apart (V) count as the same Bias point.
BIAS_TOL_V: float = 1e-9


class ScalarSweep[PointT](BaseModel):
    """Points solved along one scalar axis, looked up by their key.

    A subclass names the axis by implementing :meth:`_key`, and says how
    to talk about it through :attr:`sweep_noun` and :attr:`key_unit`,
    which appear in the lookup error.

    Attributes:
        points: The solved points, in sweep order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: What the sweep is called in the lookup error.
    sweep_noun: ClassVar[str] = "sweep"
    #: Unit of the swept key, which also names it in the lookup error.
    key_unit: ClassVar[str] = ""

    points: list[PointT] = Field(default_factory=list)

    def _key(self, point: PointT) -> float:
        """The scalar this sweep walks, read off one of its points."""
        raise NotImplementedError

    @property
    def keys(self) -> NDArray[np.float64]:
        """The swept key of every point, in sweep order."""
        return np.asarray([self._key(point) for point in self.points], dtype=np.float64)

    def point_at(self, key: float, *, tol: float = BIAS_TOL_V) -> PointT:
        """The point solved at one value of the swept key.

        Args:
            key: Value to look up, in :attr:`key_unit`.
            tol: How far apart two keys may sit and still count as the
                same point.

        Returns:
            The matching point.

        Raises:
            ValueError: When the sweep visited no such key, naming the
                keys it did visit.
        """
        for point in self.points:
            if abs(self._key(point) - key) <= tol:
                return point
        visited = ", ".join(f"{self._key(point):g}" for point in self.points)
        raise ValueError(
            f"The {self.sweep_noun} has no point at {self.key_unit} = {key:g}; "
            f"it visited {visited} {self.key_unit}."
        )
