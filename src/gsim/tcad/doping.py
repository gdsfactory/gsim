"""Analytic doping profiles for the charge-transport solve.

Doping is specified per mesh region — no process simulation. Three shapes
are analytic, and two carry a profile the caller supplies:

- :class:`StepDoping`: uniform concentration inside an optional box.
- :class:`GaussianDoping`: separable Gaussian around a center point.
- :class:`ImplantDoping`: implant-like Gaussian in depth (projected range
  and straggle below a surface) with a hard lateral window.
- :class:`TableDoping`: samples on an ascending grid — a measured SIMS
  depth profile, or a process simulator's output — interpolated linearly.
- :class:`CallableDoping`: any Python function of ``(x, y)``, for a shape
  none of the others expresses.

All profiles are pure functions of the cross-section coordinates: ``x`` is
the in-plane (transverse) coordinate and ``y`` the vertical coordinate of
the native-2D mesh, both in um. Concentrations are in cm^-3. The evaluated
node values are what enters DEVSIM as the ``Donors`` / ``Acceptors`` node
solutions behind the ``NetDoping`` node model.

Every shape is a member of the :data:`DopingProfile` union, so a profile
reaches the solve through the same seam whatever its shape:
``sim.add_doping(...)`` on a :class:`~gsim.tcad.sim.ChargeTransportSim`,
or ``Device(doping=[...])`` on a
:class:`~gsim.modulator.study.Study`. Profiles naming the same region
superpose.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, Self, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from gsim.common.validation import AscendingInterval

__all__ = [
    "CallableDoping",
    "DopingProfile",
    "GaussianDoping",
    "ImplantDoping",
    "StepDoping",
    "TableDoping",
    "acceptor_donor_concentrations",
    "net_doping_cm3",
]


def _window_mask(
    values: NDArray[np.float64], interval: tuple[float, float] | None
) -> NDArray[np.float64]:
    """Return a 0/1 mask selecting values inside the closed interval."""
    if interval is None:
        return np.ones_like(values)
    lo, hi = interval
    return ((values >= lo) & (values <= hi)).astype(np.float64)


def _validate_grid(name: str, samples: list[float] | None) -> None:
    """Reject a coordinate axis that cannot be interpolated along."""
    if samples is None:
        return
    values = np.asarray(samples, dtype=np.float64)
    if values.size < 2:
        raise ValueError(f"{name} needs at least two samples to interpolate between")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} samples must be finite")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} samples must be strictly ascending")


def _cell_weights(
    values: NDArray[np.float64], grid: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Locate values in an ascending grid, clamped to its end cells.

    Args:
        values: Coordinates to locate.
        grid: Ascending sample coordinates, at least two of them.

    Returns:
        The index of each value's lower grid point, and its weight toward
        the upper one, in ``[0, 1]``. Values outside the grid clamp to its
        end samples; :class:`TableDoping` handles ``fill="zero"`` by
        masking them separately.
    """
    lower = np.asarray(
        np.clip(np.searchsorted(grid, values) - 1, 0, grid.size - 2), dtype=np.intp
    )
    span = grid[lower + 1] - grid[lower]
    weight = np.clip((values - grid[lower]) / span, 0.0, 1.0)
    return lower, np.asarray(weight, dtype=np.float64)


def _outside(
    values: NDArray[np.float64], grid: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Mark the values falling beyond either end of an ascending grid."""
    return np.asarray((values < grid[0]) | (values > grid[-1]), dtype=bool)


class _DopingBase(BaseModel):
    """Common fields shared by all doping profile shapes.

    Attributes:
        region: Mesh volume physical-group (layer) name the profile applies
            to. The region must exist on the generated cross-section mesh.
        dopant_type: ``"donor"`` (n-type) or ``"acceptor"`` (p-type).
    """

    model_config = ConfigDict(validate_assignment=True)

    region: str = Field(min_length=1, description="Mesh region (layer) name")
    dopant_type: Literal["donor", "acceptor"]

    def concentration(
        self, x: ArrayLike, y: ArrayLike
    ) -> NDArray[np.float64]:  # pragma: no cover - abstract
        """Evaluate the dopant concentration (cm^-3) at coordinates in um."""
        raise NotImplementedError


class StepDoping(_DopingBase):
    """Uniform doping inside an optional (x, y) box.

    Attributes:
        concentration_cm3: Dopant concentration inside the box (cm^-3).
        x_range: Optional in-plane ``(min, max)`` window in um; unbounded
            when omitted.
        y_range: Optional vertical ``(min, max)`` window in um; unbounded
            when omitted.
    """

    kind: Literal["step"] = "step"
    concentration_cm3: float = Field(gt=0.0)
    x_range: AscendingInterval | None = None
    y_range: AscendingInterval | None = None

    def concentration(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Uniform ``concentration_cm3`` inside the box, zero outside."""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        return np.asarray(
            self.concentration_cm3
            * _window_mask(xa, self.x_range)
            * _window_mask(ya, self.y_range),
            dtype=np.float64,
        )


class GaussianDoping(_DopingBase):
    """Separable Gaussian doping around a center point.

    Each axis with a finite ``sigma`` contributes a Gaussian factor
    ``exp(-(u - u0)^2 / (2 sigma^2))``; an omitted sigma leaves the profile
    uniform along that axis. Optional hard windows clip the tails.

    Attributes:
        peak_cm3: Peak concentration at the center (cm^-3).
        center: ``(x0, y0)`` center in um.
        sigma_x: Optional in-plane standard deviation in um.
        sigma_y: Optional vertical standard deviation in um.
        x_range: Optional in-plane hard window in um.
        y_range: Optional vertical hard window in um.
    """

    kind: Literal["gaussian"] = "gaussian"
    peak_cm3: float = Field(gt=0.0)
    center: tuple[float, float]
    sigma_x: float | None = Field(default=None, gt=0.0)
    sigma_y: float | None = Field(default=None, gt=0.0)
    x_range: AscendingInterval | None = None
    y_range: AscendingInterval | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """At least one sigma must be given."""
        if self.sigma_x is None and self.sigma_y is None:
            raise ValueError("GaussianDoping needs sigma_x and/or sigma_y")
        return self

    def concentration(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Gaussian falloff from the peak, clipped to the hard windows."""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        x0, y0 = self.center
        result = np.full(np.broadcast(xa, ya).shape, self.peak_cm3)
        if self.sigma_x is not None:
            result = result * np.exp(-((xa - x0) ** 2) / (2.0 * self.sigma_x**2))
        if self.sigma_y is not None:
            result = result * np.exp(-((ya - y0) ** 2) / (2.0 * self.sigma_y**2))
        return np.asarray(
            result * _window_mask(xa, self.x_range) * _window_mask(ya, self.y_range),
            dtype=np.float64,
        )


class ImplantDoping(_DopingBase):
    """Implant-like Gaussian-in-depth profile below a surface.

    The concentration peaks at the projected range ``range_um`` below
    ``surface_y`` and falls off as a Gaussian with standard deviation
    ``straggle_um``: ``N(d) = peak * exp(-(d - Rp)^2 / (2 dRp^2))`` with
    depth ``d = surface_y - y``. Above the surface (``d < 0``) the
    concentration is zero. The lateral extent is a hard window.

    Attributes:
        peak_cm3: Peak concentration at the projected range (cm^-3).
        surface_y: Implant surface y coordinate in um (implant goes to -y).
        range_um: Projected range Rp below the surface in um (>= 0).
        straggle_um: Straggle (depth standard deviation) in um.
        x_range: Optional in-plane hard window in um.
    """

    kind: Literal["implant"] = "implant"
    peak_cm3: float = Field(gt=0.0)
    surface_y: float
    range_um: float = Field(ge=0.0)
    straggle_um: float = Field(gt=0.0)
    x_range: AscendingInterval | None = None

    def concentration(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Gaussian in depth below the surface, zero above it."""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        depth = self.surface_y - ya
        gauss = self.peak_cm3 * np.exp(
            -((depth - self.range_um) ** 2) / (2.0 * self.straggle_um**2)
        )
        inside = (depth >= 0.0).astype(np.float64)
        return np.asarray(
            gauss * inside * _window_mask(xa, self.x_range), dtype=np.float64
        )


class TableDoping(_DopingBase):
    """Sampled doping on an ascending grid, interpolated linearly.

    The shape for a profile that was measured or computed elsewhere: a
    SIMS depth profile, or a process simulator's output read off its own
    grid. Give one axis for a profile uniform along the other — a depth
    profile gives ``y_um`` alone — or both for a full 2D map, where
    ``values_cm3`` is indexed ``[x][y]``.

    Outside the sampled grid the profile either holds its edge value or
    drops to zero, per ``fill``; the optional hard windows clip it further
    the way they do on every other shape.

    Attributes:
        values_cm3: Concentrations at the sample points (cm^-3),
            non-negative. One list per given axis, or a nested list
            indexed ``[x][y]`` when both are given.
        x_um: Optional ascending in-plane sample coordinates in um.
        y_um: Optional ascending vertical sample coordinates in um.
        fill: ``"edge"`` holds the boundary sample outside the grid,
            ``"zero"`` reads zero there. A measured profile that stops at
            the substrate usually wants ``"edge"``; one that stops where
            the dopant does wants ``"zero"``.
        x_range: Optional in-plane hard window in um.
        y_range: Optional vertical hard window in um.
    """

    kind: Literal["table"] = "table"
    values_cm3: list[float] | list[list[float]]
    x_um: list[float] | None = None
    y_um: list[float] | None = None
    fill: Literal["edge", "zero"] = "edge"
    x_range: AscendingInterval | None = None
    y_range: AscendingInterval | None = None

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        """Axes must be ascending, and the values must match their shape."""
        if self.x_um is None and self.y_um is None:
            raise ValueError("TableDoping needs x_um and/or y_um to interpolate along")
        _validate_grid("x_um", self.x_um)
        _validate_grid("y_um", self.y_um)

        values = np.asarray(self.values_cm3, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("values_cm3 must be finite")
        if np.any(values < 0.0):
            raise ValueError("values_cm3 must be non-negative concentrations")

        if self.x_um is not None and self.y_um is not None:
            expected: tuple[int, ...] = (len(self.x_um), len(self.y_um))
        elif self.x_um is not None:
            expected = (len(self.x_um),)
        elif self.y_um is not None:
            expected = (len(self.y_um),)
        else:  # pragma: no cover - the guard above already rejected this
            raise ValueError("TableDoping needs x_um and/or y_um to interpolate along")
        if values.shape != expected:
            axes = "x_um and y_um" if len(expected) == 2 else "the given axis"
            raise ValueError(
                f"values_cm3 has shape {values.shape} but {axes} imply "
                f"{expected}; index a 2D table as values_cm3[x][y]"
            )
        return self

    def concentration(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Linearly interpolate the samples at the given coordinates."""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(xa, ya).shape
        xb = np.broadcast_to(xa, shape)
        yb = np.broadcast_to(ya, shape)
        values = np.asarray(self.values_cm3, dtype=np.float64)

        if self.x_um is not None and self.y_um is not None:
            x_grid = np.asarray(self.x_um, dtype=np.float64)
            y_grid = np.asarray(self.y_um, dtype=np.float64)
            i, wx = _cell_weights(xb, x_grid)
            j, wy = _cell_weights(yb, y_grid)
            result = (
                values[i, j] * (1.0 - wx) * (1.0 - wy)
                + values[i + 1, j] * wx * (1.0 - wy)
                + values[i, j + 1] * (1.0 - wx) * wy
                + values[i + 1, j + 1] * wx * wy
            )
            outside = _outside(xb, x_grid) | _outside(yb, y_grid)
        else:
            along_x = self.x_um is not None
            grid = np.asarray(self.x_um if along_x else self.y_um, dtype=np.float64)
            coords = xb if along_x else yb
            k, w = _cell_weights(coords, grid)
            result = values[k] * (1.0 - w) + values[k + 1] * w
            outside = _outside(coords, grid)

        if self.fill == "zero":
            result = np.where(outside, 0.0, result)
        return np.asarray(
            result * _window_mask(xb, self.x_range) * _window_mask(yb, self.y_range),
            dtype=np.float64,
        )


class CallableDoping(_DopingBase):
    """Any Python function of ``(x, y)`` as a doping profile.

    The escape hatch for a shape none of the other members expresses — a
    closed form the caller already has, or an interpolator built over data
    that is not on a grid. The function is called with the region's node
    coordinates as arrays in um and must return concentrations in cm^-3,
    broadcastable to their shape.

    Unlike the other shapes this one holds a live Python object, so it
    does not survive ``model_dump(mode="json")`` and cannot be written to
    a settings file. Reach for :class:`TableDoping` when the profile has
    to be serialized.

    Attributes:
        function: ``f(x_um, y_um) -> concentrations_cm3``. Called with
            arrays; a scalar return broadcasts.
        x_range: Optional in-plane hard window in um.
        y_range: Optional vertical hard window in um.
    """

    kind: Literal["callable"] = "callable"
    function: SkipJsonSchema[Callable[[Any, Any], Any]]
    x_range: AscendingInterval | None = None
    y_range: AscendingInterval | None = None

    def concentration(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the function, then apply the hard windows.

        Raises:
            ValueError: When the function returns something that is not
                broadcastable to the coordinate shape, or returns
                negative or non-finite concentrations — either would reach
                DEVSIM as a silently wrong ``NetDoping``.
        """
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(xa, ya).shape
        returned = np.asarray(self.function(xa, ya), dtype=np.float64)
        try:
            values = np.broadcast_to(returned, shape)
        except ValueError as error:
            raise ValueError(
                f"The doping function for region {self.region!r} returned shape "
                f"{returned.shape}, which does not broadcast to the "
                f"{shape} coordinates it was called with."
            ) from error
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"The doping function for region {self.region!r} returned "
                "non-finite concentrations."
            )
        if np.any(values < 0.0):
            raise ValueError(
                f"The doping function for region {self.region!r} returned "
                "negative concentrations; a profile is a dopant density, and "
                "the p/n sense is carried by dopant_type."
            )
        return np.asarray(
            values
            * _window_mask(np.broadcast_to(xa, shape), self.x_range)
            * _window_mask(np.broadcast_to(ya, shape), self.y_range),
            dtype=np.float64,
        )


DopingProfile = Annotated[
    Union[  # noqa: UP007
        StepDoping,
        GaussianDoping,
        ImplantDoping,
        TableDoping,
        CallableDoping,
    ],
    Field(discriminator="kind"),
]


def acceptor_donor_concentrations(
    profiles: list[DopingProfile],
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sum profiles into total (acceptors, donors) concentrations in cm^-3.

    Args:
        profiles: Doping profiles to superpose (any mix of regions —
            filter by region before calling when needed).
        x: In-plane coordinates in um.
        y: Vertical coordinates in um.

    Returns:
        Tuple ``(acceptors, donors)`` of arrays broadcast to the coordinate
        shape.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    shape = np.broadcast(xa, ya).shape
    acceptors = np.zeros(shape)
    donors = np.zeros(shape)
    for profile in profiles:
        values = profile.concentration(xa, ya)
        if profile.dopant_type == "acceptor":
            acceptors = np.asarray(acceptors + values, dtype=np.float64)
        else:
            donors = np.asarray(donors + values, dtype=np.float64)
    return (
        np.asarray(acceptors, dtype=np.float64),
        np.asarray(donors, dtype=np.float64),
    )


def net_doping_cm3(
    profiles: list[DopingProfile],
    x: ArrayLike,
    y: ArrayLike,
) -> NDArray[np.float64]:
    """Net doping ``donors - acceptors`` in cm^-3 at coordinates in um."""
    acceptors, donors = acceptor_donor_concentrations(profiles, x, y)
    return np.asarray(donors - acceptors, dtype=np.float64)
