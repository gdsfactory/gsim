"""Estimate the memory and time of a Palace run before it is submitted.

The estimate comes from power-law fits to the logs of finished Palace runs, one
fit per (solver, linear solver) group. The coefficients, their errors and the
date of the data live in ``data/resource_fits.json`` so that other tools can
read the same file::

    memory_gib = a * dofs**p
    time_s = b * dofs**q * work**w

Memory is the figure to rely on: its leave-one-out error is 4 to 16 % for the
driven and eigenmode groups. Time is a range only, off by a factor of 1.1 to
2.2 typically and far more for single runs, and nothing here decides on it.
Groups without a fit (other solvers, other linear solvers) get no estimate.

The inputs are the ``config.json`` that ``write_config()`` wrote and the mesh
statistics of ``mesh()``, so an estimate is only available after both.
"""

from __future__ import annotations

import json
import logging
import math
import textwrap
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

FITS_FILENAME = "resource_fits.json"

#: Peak memory per node of the largest successful run in the fitted data (GiB).
#: Used when the caller names no machine; ``default_machine`` in the fits file
#: carries the same number.
DEFAULT_MACHINE_MEMORY_GIB = 194.55

_BAND_SIGMAS = 2.0
_EXTRAPOLATION_FACTOR = 1.5
_TIME_LOO_WARNING = 0.5
# Relative difference between the DOFs from the mesh topology and from the
# calibrated DOFs per tetrahedron above which a note is added.
_DOFS_CROSS_CHECK = 0.10
# Palace's own default of AdaptiveMaxSamples.
_PALACE_ADAPTIVE_MAX_SAMPLES = 20

# Palace's Solver.Linear.Type mapped to the linear-solver key of the fits.
# ``Default`` resolved to SuperLU in every driven and mode log of the data; that
# may change with the Palace build.
_LINEAR_SOLVER_KEYS = {
    "MUMPS": "MUMPS",
    "SuperLU": "SuperLU",
    "Default": "SuperLU",
    "AMS": "iterative",
    "BoomerAMG": "iterative",
}


_SOURCE_TEXT = {
    "mesh_topology": "counted from the mesh edges, faces and tetrahedra",
    "tets_ratio": "tetrahedra times the calibrated DOFs per tetrahedron",
    "given": "given by the caller",
}


@dataclass(frozen=True)
class ResourceEstimate:
    """Estimated peak memory and solve time of one Palace run.

    ``available`` is ``False`` when no estimate could be made (``reason`` says
    why): the fields of the estimate are then ``None``. Memory is in GiB
    (1024-based) of peak memory per node, as Palace reports it; time is the
    solve only, without meshing, upload and queueing.

    Attributes:
        available: Whether an estimate was made.
        reason: Why none was made, when ``available`` is ``False``.
        solver: Fit group solver: ``driven_adaptive``, ``driven_uniform``,
            ``eigen`` or ``boundary_mode``.
        linear_solver: Fit group linear solver: ``MUMPS``, ``SuperLU`` or
            ``iterative``.
        dofs: Estimated number of unknowns of the field space.
        dofs_source: ``mesh_topology``, ``tets_ratio`` or ``given``.
        memory_gib: Estimated peak memory per node.
        memory_range_gib: Range of the memory estimate (plus and minus two
            leave-one-out errors).
        memory_error: Leave-one-out rms of ln(predicted / measured) of the
            memory fit; 0.15 means a typical error of about 16 %.
        time_s: Estimated solve time, for a typical adaptive sample count.
        time_range_s: Range of the time estimate. For an adaptive sweep its
            upper end uses the configured cap on samples.
        time_error: Leave-one-out rms of ln(predicted / measured) of the time
            fit; exp of it is the typical factor.
        machine_memory_gib: Memory per node the estimate is compared with.
        machine_assumption: Where that figure comes from.
        warnings: Things the user should act on, such as a memory estimate
            above the machine or beyond the fitted data.
        notes: Assumptions and limits that do not need action.
        fits_version: Version of the fits file used.
        data_date: Date of the data behind the fits, if recorded.
    """

    available: bool
    reason: str | None = None
    solver: str | None = None
    linear_solver: str | None = None
    dofs: float | None = None
    dofs_source: str | None = None
    memory_gib: float | None = None
    memory_range_gib: tuple[float, float] | None = None
    memory_error: float | None = None
    time_s: float | None = None
    time_range_s: tuple[float, float] | None = None
    time_error: float | None = None
    machine_memory_gib: float = DEFAULT_MACHINE_MEMORY_GIB
    machine_assumption: str = ""
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    fits_version: int | None = None
    data_date: str | None = None

    @property
    def exceeds_machine(self) -> bool:
        """Whether the central memory estimate is above the machine memory."""
        return self.memory_gib is not None and self.memory_gib > self.machine_memory_gib

    def summary(self) -> str:
        """Return the estimate as a few lines of text, with ranges and warnings."""
        if (
            not self.available
            or self.dofs is None
            or self.memory_gib is None
            or self.memory_range_gib is None
            or self.memory_error is None
        ):
            return f"Palace resource estimate: not available ({self.reason})"
        low, high = self.memory_range_gib
        source = _SOURCE_TEXT.get(self.dofs_source or "", "")
        typical = 100 * math.expm1(self.memory_error)
        lines = [
            f"Palace resource estimate ({self.solver}, {self.linear_solver})",
            f"  DOFs:    {self.dofs:,.0f} ({source})",
            (
                f"  Memory:  {self.memory_gib:.1f} GiB per node, range "
                f"{low:.1f} to {high:.1f} GiB (typical error {typical:.0f} %)"
            ),
        ]
        if self.time_s is not None and self.time_range_s is not None:
            factor = math.exp(self.time_error or 0.0)
            lines.append(
                f"  Time:    {_format_seconds(self.time_s)}, range "
                f"{_format_seconds(self.time_range_s[0])} to "
                f"{_format_seconds(self.time_range_s[1])} "
                f"(solve only; typical factor x{factor:.1f})"
            )
        lines.append(
            _wrap(
                "Machine",
                f"{self.machine_memory_gib:g} GiB per node, {self.machine_assumption}",
            )
        )
        lines.extend(_wrap("Warning", text) for text in self.warnings)
        lines.extend(_wrap("Note", text) for text in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return :meth:`summary`."""
        return self.summary()


def _wrap(label: str, text: str) -> str:
    """Wrap one labelled line of the summary, aligned with the lines above it."""
    return textwrap.fill(
        text,
        width=100,
        initial_indent=f"  {label}:".ljust(11),
        subsequent_indent=" " * 11,
    )


def _format_seconds(seconds: float) -> str:
    """Format a duration as seconds, minutes or hours."""
    if seconds >= 7200:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 120:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.0f} s"


@cache
def _packaged_fits() -> dict[str, Any] | None:
    """Read the fits file shipped with gsim; ``None`` if it cannot be read."""
    try:
        text = (files("gsim.palace") / "data" / FITS_FILENAME).read_text(
            encoding="utf-8"
        )
        data = json.loads(text)
    except (OSError, ValueError):
        logger.debug("Palace resource fits could not be read", exc_info=True)
        return None
    return data if isinstance(data, dict) and "groups" in data else None


def load_fits() -> dict[str, Any] | None:
    """Return the fits shipped with gsim, or ``None`` if they cannot be read.

    The returned dictionary is shared; do not modify it.
    """
    return _packaged_fits()


def nedelec_dofs(order: int, edges: int, faces: int, tets: int) -> int:
    """Count the Nedelec (ND) unknowns of an all-tetrahedron mesh.

    The order-p space has p unknowns per edge, p(p-1) per triangular face and
    p(p-1)(p-2)/2 per tetrahedron interior. This is the count before Palace's
    boundary splitting and mesh refinement.

    Args:
        order: Field order, a positive integer.
        edges: Number of unique edges.
        faces: Number of unique triangular faces.
        tets: Number of tetrahedra.

    Returns:
        The number of unknowns.

    Raises:
        ValueError: If ``order`` is not a positive integer.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("Field order must be a positive integer")
    return (
        order * edges
        + order * (order - 1) * faces
        + order * (order - 1) * (order - 2) * tets // 2
    )


def count_sweep_points(samples: list[dict[str, Any]] | None) -> int | None:
    """Count the frequency points of a Palace ``Driven.Samples`` list.

    Returns ``None`` when a sample entry has a type this does not know.
    """
    if not samples:
        return None
    total = 0
    for sample in samples:
        kind = sample.get("Type")
        if kind in ("Point", "List"):
            total += len(sample.get("Freq", []))
        elif kind == "Log" and sample.get("NSample"):
            total += int(sample["NSample"])
        elif kind == "Linear" and sample.get("FreqStep"):
            span = float(sample["MaxFreq"]) - float(sample["MinFreq"])
            total += math.floor(span / float(sample["FreqStep"]) + 1e-6) + 1
        else:
            return None
    return total


def count_excitations(config: Mapping[str, Any]) -> int:
    """Count the excited ports of a Palace config (at least 1)."""
    boundaries = config.get("Boundaries") or {}
    count = 0
    for key in ("LumpedPort", "WavePort", "SurfaceCurrent"):
        entries = boundaries.get(key) or []
        if isinstance(entries, list):
            count += sum(
                1
                for entry in entries
                if isinstance(entry, dict) and entry.get("Excitation")
            )
    return max(1, count)


def _machine(
    fits: Mapping[str, Any] | None, machine_memory_gib: float | None
) -> tuple[float, str]:
    """Return the machine memory to compare with, and where it comes from."""
    if machine_memory_gib is not None:
        if not machine_memory_gib > 0:
            raise ValueError("machine_memory_gib must be positive")
        return float(machine_memory_gib), "set by the caller"
    default = (fits or {}).get("default_machine") or {}
    value = float(default.get("memory_gib", DEFAULT_MACHINE_MEMORY_GIB))
    procs = default.get("mpi_procs", 95)
    return value, (
        f"the largest successful run in the fitted data ({procs} MPI processes); "
        "not a limit of the cloud service"
    )


def _unavailable(
    reason: str,
    fits: Mapping[str, Any] | None,
    machine_memory_gib: float | None,
) -> ResourceEstimate:
    """Build an estimate that says why there is none."""
    machine, assumption = _machine(fits, machine_memory_gib)
    return ResourceEstimate(
        available=False,
        reason=reason,
        machine_memory_gib=machine,
        machine_assumption=assumption,
    )


def unavailable_estimate(
    reason: str, *, machine_memory_gib: float | None = None
) -> ResourceEstimate:
    """Return an estimate that is not available, with the reason why.

    Raises:
        ValueError: If ``machine_memory_gib`` is not positive.
    """
    return _unavailable(reason, load_fits(), machine_memory_gib)


def _find_group(
    fits: Mapping[str, Any], solver: str, linear_solver: str
) -> dict[str, Any] | None:
    """Return the fit of one (solver, linear solver) group, if there is one."""
    for group in fits["groups"]:
        if group["solver"] == solver and group["linear_solver"] == linear_solver:
            return group
    return None


def estimate_from_inputs(
    config: Mapping[str, Any],
    mesh_stats: Mapping[str, Any] | None,
    *,
    machine_memory_gib: float | None = None,
    dofs: float | None = None,
    fits: Mapping[str, Any] | None = None,
) -> ResourceEstimate:
    """Estimate memory and time from a Palace config and the mesh statistics.

    Args:
        config: Parsed ``config.json`` of the run.
        mesh_stats: Mesh statistics of ``mesh()`` (``tetrahedra`` and
            ``topology``); may be ``None`` when ``dofs`` is given.
        machine_memory_gib: Memory per node to compare the estimate with.
            ``None`` takes the largest successful run in the fitted data,
            194.55 GiB on 95 MPI processes. The cloud service chooses its own
            resources; this is not its limit.
        dofs: Number of unknowns, to use instead of counting them in the mesh.
        fits: Fit data to use instead of the file shipped with gsim.

    Returns:
        The estimate. ``available`` is ``False`` when the solver or linear
        solver has no fit, or when the inputs do not give the number of
        unknowns; nothing is raised for that.

    Raises:
        ValueError: If ``machine_memory_gib`` is not positive.
    """
    if fits is None:
        fits = load_fits()
    if fits is None:
        return _unavailable("the fitted data file is missing", None, machine_memory_gib)
    stats = mesh_stats or {}
    solver_config = config.get("Solver") or {}
    problem = (config.get("Problem") or {}).get("Type")
    driven = solver_config.get("Driven") or {}

    if problem == "Driven":
        solver = (
            "driven_adaptive"
            if (driven.get("AdaptiveTol") or 0) > 0
            else "driven_uniform"
        )
    elif problem == "Eigenmode":
        solver = "eigen"
    elif problem == "BoundaryMode":
        solver = "boundary_mode"
    else:
        return _unavailable(
            f"no fit for the {problem} problem type", fits, machine_memory_gib
        )

    configured = (solver_config.get("Linear") or {}).get("Type", "Default")
    linear_solver = _LINEAR_SOLVER_KEYS.get(configured)
    if linear_solver is None:
        return _unavailable(
            f"no fit for the {configured} linear solver", fits, machine_memory_gib
        )
    group = _find_group(fits, solver, linear_solver)
    if group is None:
        return _unavailable(
            f"no fit for {solver} with the {configured} linear solver "
            f"({linear_solver} group)",
            fits,
            machine_memory_gib,
        )

    machine, assumption = _machine(fits, machine_memory_gib)
    warnings: list[str] = []
    notes: list[str] = []
    if configured == "Default":
        notes.append(
            "linear solver Default is read as SuperLU, as it resolved in the "
            "logged runs"
        )

    # Number of unknowns
    order = solver_config.get("Order", 1)
    tets = stats.get("tetrahedra")
    topology = stats.get("topology")
    ratio = (group.get("dofs_per_tet") or {}).get(str(order)) or {}
    per_tet = ratio.get("median")
    from_topology = None
    if tets and topology and solver != "boundary_mode":
        from_topology = nedelec_dofs(
            order, topology["edges"], topology["triangular_faces"], tets
        )
    from_ratio = tets * per_tet if tets and per_tet else None
    spread_factor = 1.0
    if dofs is not None:
        n_dofs, source = float(dofs), "given"
    elif from_topology is not None:
        n_dofs, source = float(from_topology), "mesh_topology"
        if from_ratio and abs(from_topology / from_ratio - 1) > _DOFS_CROSS_CHECK:
            notes.append(
                f"counted DOFs ({from_topology:,}) differ from the calibrated "
                f"{per_tet} DOFs per tetrahedron ({from_ratio:,.0f}) "
                "by more than 10 %"
            )
    elif from_ratio is not None:
        n_dofs, source = float(from_ratio), "tets_ratio"
        spread_factor = math.exp(_BAND_SIGMAS * ratio["spread"])
    else:
        return _unavailable(
            "no DOF count: the mesh has no tetrahedron count and edge/face "
            f"topology, and no calibrated DOFs per tetrahedron for order {order}",
            fits,
            machine_memory_gib,
        )

    # Memory
    memory_fit, time_fit = group["memory"], group["time"]
    memory = memory_fit["a"] * n_dofs ** memory_fit["p"]
    memory_error = memory_fit["loo"]
    memory_band = math.exp(_BAND_SIGMAS * memory_error)
    memory_band *= spread_factor ** memory_fit["p"]
    memory_range = (memory / memory_band, memory * memory_band)

    if n_dofs > _EXTRAPOLATION_FACTOR * group["dofs_max"]:
        warnings.append(
            f"{n_dofs:,.0f} DOFs is beyond the largest run behind the fit "
            f"({group['dofs_max']:,}): this is an extrapolation and the range "
            "is optimistic"
        )
    if n_dofs < group["dofs_min"] / _EXTRAPOLATION_FACTOR:
        notes.append(
            f"{n_dofs:,.0f} DOFs is below the smallest run behind the fit "
            f"({group['dofs_min']:,}): a power law has no fixed overhead, so "
            "memory and time may be underestimated for such a small mesh"
        )
    if memory > machine:
        warnings.append(
            f"estimated memory {memory:.1f} GiB per node is above the machine "
            f"memory of {machine:g} GiB"
        )
    elif memory_range[1] > machine:
        warnings.append(
            f"the upper end of the memory range ({memory_range[1]:.1f} GiB per "
            f"node) is above the machine memory of {machine:g} GiB"
        )
    refinement = (config.get("Model") or {}).get("Refinement") or {}
    if (refinement.get("MaxIts") or 0) > 0:
        warnings.append(
            "adaptive mesh refinement is on: the estimate is for the input mesh "
            "and refinement increases memory (no refinement runs are in the data)"
        )
    if (refinement.get("UniformLevels") or 0) > 0:
        warnings.append(
            "uniform mesh refinement levels are set: they multiply the number "
            "of DOFs and are not included in the estimate"
        )

    # Time
    time_s = time_range = None
    time_error = time_fit["loo"]
    work_typical, work_cap = _work(solver, config, group, notes)
    if work_typical is not None and work_cap is not None:
        time_band = math.exp(_BAND_SIGMAS * time_error)
        time_band *= spread_factor ** time_fit["q"]
        base = time_fit["b"] * n_dofs ** time_fit["q"]
        time_s = base * work_typical ** time_fit["w"]
        time_range = (
            time_s / time_band,
            base * work_cap ** time_fit["w"] * time_band,
        )
        if time_error > _TIME_LOO_WARNING:
            notes.append(
                "the time fit of this group is loose (typical factor "
                f"x{math.exp(time_error):.1f}): read the time as an order of magnitude"
            )
    else:
        notes.append("time not estimated: the sweep size could not be read")

    return ResourceEstimate(
        available=True,
        solver=solver,
        linear_solver=linear_solver,
        dofs=n_dofs,
        dofs_source=source,
        memory_gib=memory,
        memory_range_gib=memory_range,
        memory_error=memory_error,
        time_s=time_s,
        time_range_s=time_range,
        time_error=time_error,
        machine_memory_gib=machine,
        machine_assumption=assumption,
        warnings=tuple(warnings),
        notes=tuple(notes),
        fits_version=fits.get("fits_version"),
        data_date=fits.get("data_date"),
    )


def _work(
    solver: str,
    config: Mapping[str, Any],
    group: Mapping[str, Any],
    notes: list[str],
) -> tuple[float | None, float | None]:
    """Return the work units of the time model: typical and worst case.

    The two differ only for an adaptive sweep, whose sample count is not known
    before the run: typical is the median count in the data (at most the cap),
    worst case the configured cap.
    """
    solver_config = config.get("Solver") or {}
    if solver == "boundary_mode":
        return 1.0, 1.0
    if solver == "eigen":
        modes = float((solver_config.get("Eigenmode") or {}).get("N", 1))
        return modes, modes
    driven = solver_config.get("Driven") or {}
    excitations = count_excitations(config)
    if solver == "driven_uniform":
        points = count_sweep_points(driven.get("Samples"))
        if points is None:
            return None, None
        work = float(excitations * points)
        return work, work
    seen = group["samples_seen"]
    cap = driven.get("AdaptiveMaxSamples") or _PALACE_ADAPTIVE_MAX_SAMPLES
    typical = min(seen["median"], cap)
    basis = "the median of the data" if typical < cap else "the cap"
    notes.append(
        "adaptive sweep: the sample count is not known before the run; time "
        f"is for {typical} samples ({basis}) and its upper end for the cap of "
        f"{cap}, with {excitations} excitation(s); the data has {seen['min']} "
        f"to {seen['max']} samples"
    )
    return float(typical * excitations), float(cap * excitations)


def estimate_resources(
    sim: Any, *, machine_memory_gib: float | None = None
) -> ResourceEstimate:
    """Estimate the memory and time of a configured Palace simulation.

    Call it after ``sim.mesh()`` and ``sim.write_config()``. It reads what they
    wrote and changes nothing. See :meth:`PalaceSimMixin.estimate_resources`.

    Args:
        sim: A Palace simulation, such as :class:`gsim.palace.DrivenSim`.
        machine_memory_gib: Memory per node to compare the estimate with;
            ``None`` takes the largest successful run in the fitted data.

    Returns:
        The estimate; ``available`` is ``False`` with a reason when there is
        none.
    """
    return sim.estimate_resources(machine_memory_gib=machine_memory_gib)


def log_estimate(
    estimate: ResourceEstimate, *, log: logging.Logger | None = None
) -> None:
    """Log an estimate: the summary at INFO and each warning at WARNING.

    An estimate that is not available is only logged at DEBUG.
    """
    log = log or logger
    if not estimate.available:
        log.debug("%s", estimate.summary())
        return
    log.info("%s", estimate.summary())
    for text in estimate.warnings:
        log.warning("Palace resource estimate: %s", text)


__all__ = [
    "DEFAULT_MACHINE_MEMORY_GIB",
    "ResourceEstimate",
    "count_excitations",
    "count_sweep_points",
    "estimate_from_inputs",
    "estimate_resources",
    "load_fits",
    "log_estimate",
    "nedelec_dofs",
    "unavailable_estimate",
]
