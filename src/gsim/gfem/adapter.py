"""gfem adapter on the shared native-2D cross-section mesh.

Loads the msh the ``BoundaryModeSim`` pipeline writes into gfem and solves
the 2D port mode of its cross-section. Two things are mapped:

- every named 2D physical group becomes a :class:`Medium`
  (:func:`region_media`), resolved from the stack exactly as
  :func:`gsim.femwell.adapter.epsilon_by_region` resolves it, so both
  solvers see one permittivity per region;
- the 1D groups the mesher writes become the boundary conditions of the
  port-mode problem (:func:`edge_conditions`): ``<layer>_conductivity`` a
  finite-conductivity conductor, ``<layer>_pec`` a perfect conductor, the
  outer ``absorbing`` wall a perfect conductor, an absorbing boundary or
  nothing (gfem's natural PMC, which is Palace's default too).

:func:`solve_line` runs the solve, :func:`line_impedances` reads the three
characteristic impedances off the result, and :func:`line_reading` returns
the one reading a route asks for, the
:class:`~gsim.common.modes.LineReading` femwell and Palace return as well.

The mapping functions are pure meshio/numpy and testable without gfem;
only :func:`build_model`, :func:`solve_line`, :func:`line_impedances` and
:func:`line_reading` need it (``pip install 'gsim[gfem]'``).

Conventions are gfem's: ``exp(+i omega t)``, ``n_eff = n' - i n''`` with
``n'' >= 0`` for a lossy line, peak phasors. ``z0_ohm`` of the reading is the
power-current impedance ``Z_PI = 2 P / |I|^2``, the definition
:class:`~gsim.common.modes.LineReading` documents. The power-voltage
``Z_PV = |V|^2 / (2 conj(P))`` and the voltage-current ``Z_VI = V / I`` come
with it, each under its own name.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import meshio
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import epsilon_0 as EPS0  # noqa: N812
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.mesh_regions import group_tags
from gsim.common.modes import Conductor, Extent, LineReading
from gsim.common.stack.materials import (
    MaterialProperties,
    region_material_map,
    resolve_stack_material,
)
from gsim.gfem.runtime import require_gfem
from gsim.palace.line_impedance import (
    MIN_VOLTAGE_POWER_RATIO,
    line_impedance_paths,
)

if TYPE_CHECKING:
    from gsim.common.stack.extractor import LayerStack

__all__ = [
    "DEFAULT_CONDUCTIVITY_S_PER_M",
    "LOOP_CLEARANCE_FRACTION",
    "PORT_NAME",
    "WALL_GROUP",
    "EdgeCondition",
    "Medium",
    "build_model",
    "edge_conditions",
    "impedance_paths_m",
    "line_impedances",
    "line_reading",
    "region_media",
    "solve_line",
    "wall_mode_from_impedances",
]

#: Name of the wave port the line solve puts on the slab's ``front`` face.
PORT_NAME = "line"

#: Name the native-2D mesher gives the outer wall of the meshed domain.
WALL_GROUP = "absorbing"

#: Conductivity (S/m) of a conductor layer whose material names none: the
#: value the Palace 2D conductor boundary falls back to.
DEFAULT_CONDUCTIVITY_S_PER_M = 5.8e7

#: Fraction of the smallest clearance around the signal conductor that the
#: current loop keeps from its outline. The Palace loop keeps 1e-3 of it, a hair,
#: which is right for Palace's own integral; gfem's ``Z_PI`` is a line integral of
#: the finite-element H field and reads the wrong current in the first layer of
#: elements beside the metal. Measured (um clearance, CPW with a 2 um thick
#: signal, so 0.5 is 1 um): on a homogeneous-oxide CPW whose metal is a PEC hole
#: (1 um elements) ``Z_PI`` is 19 % above ``Z_PV`` at 0.002 um and within 0.5 %
#: of it from 1 to 4 um; on the lossy CPW with meshed metal (12.5k triangles,
#: 0.25 um in the metal) the error against femwell's ``Z_PI`` is 3.0e-3 / 5.7e-3
#: (10 / 50 GHz) at 0.002 um and at most 6.5e-4 from 0.25 to 1 um. Two meshes,
#: not a convergence study: the loop wants to be about one element away from
#: the metal and clear of every other conductor.
LOOP_CLEARANCE_FRACTION = 0.5

_CONDUCTIVITY_SUFFIX = "_conductivity"
_PEC_SUFFIX = "_pec"
_UM = 1e-6
_FRONT_FACE = "front"


@dataclass(frozen=True)
class Medium:
    """The electromagnetic properties of one 2D mesh region.

    Attributes:
        eps_r: Real relative permittivity, above zero.
        tan_delta: Dielectric loss tangent.
        conductivity: Conductivity (S/m).
    """

    eps_r: float
    tan_delta: float = 0.0
    conductivity: float = 0.0

    def permittivity(self, frequency_hz: float) -> complex:
        """Complex relative permittivity in the ``exp(+i omega t)`` convention.

        ``eps_r (1 - i tan_delta) - i sigma / (omega eps_0)``: the formula
        both gfem's ``Material`` and the femwell adapter evaluate.

        Args:
            frequency_hz: Frequency in Hz.

        Returns:
            The complex relative permittivity (``Im < 0`` when lossy).
        """
        omega = 2.0 * math.pi * frequency_hz
        return complex(
            self.eps_r * (1.0 - 1j * self.tan_delta)
            - 1j * self.conductivity / (omega * EPS0)
        )


@dataclass(frozen=True)
class EdgeCondition:
    """One boundary condition of the port-mode problem, on a 1D mesh group.

    Attributes:
        kind: ``"pec"`` (perfect conductor), ``"conductor"`` (finite
            conductivity) or ``"absorbing"``.
        group: Name of the 1D physical group it applies to.
        conductivity: Conductivity (S/m) of a ``"conductor"``, else ``None``.
        thickness_m: Metal thickness (m) of a ``"conductor"``, or ``None``
            for a thick conductor (skin effect only).
    """

    kind: Literal["pec", "conductor", "absorbing"]
    group: str
    conductivity: float | None = None
    thickness_m: float | None = None


def _read_mesh(mesh: meshio.Mesh | str | Path) -> meshio.Mesh:
    """Load the msh unless it is already a meshio mesh."""
    if isinstance(mesh, meshio.Mesh):
        return mesh
    return meshio.read(str(mesh), file_format="gmsh")


def _check_frequency(frequency_hz: float) -> float:
    """A finite frequency above zero, as a float."""
    frequency = float(frequency_hz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError(
            f"frequency_hz must be finite and above 0, got {frequency_hz!r}."
        )
    return frequency


def _as_frequencies(value: float | Sequence[float]) -> tuple[float, ...]:
    """One frequency or a sequence of them, each checked, as a tuple."""
    values = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    if values.size == 0:
        raise ValueError("frequency_hz is empty.")
    return tuple(_check_frequency(float(v)) for v in values)


def region_media(
    mesh: meshio.Mesh | str | Path,
    stack: LayerStack,
    *,
    frequency_hz: float,
    overrides: dict[str, MaterialProperties] | None = None,
) -> dict[str, Medium]:
    """Resolve the :class:`Medium` of every 2D mesh region from the stack.

    The same lookup as :func:`gsim.femwell.adapter.epsilon_by_region`: the
    region-to-material map of the stack, then the material evaluated at the
    wavelength of ``frequency_hz``. A missing permittivity reads 1, missing
    losses read 0.

    Args:
        mesh: The shared msh v2.2 mesh (path or loaded meshio mesh).
        stack: Layer stack the mesh was generated from.
        frequency_hz: Frequency in Hz.
        overrides: Optional material-property overrides by material name.

    Returns:
        ``{region_name: Medium}`` for every dim-2 physical group.

    Raises:
        ValueError: When the mesh has no 2D group, a region's material does
            not resolve, or a region's permittivity is not above zero
            (gfem solves passive media with ``Re(eps) > 0`` only, so an
            optical metal cannot be handed to it).
    """
    frequency = _check_frequency(frequency_hz)
    wavelength_um = C0 / frequency * 1e6
    loaded = _read_mesh(mesh)
    regions = list(group_tags(loaded, dim=2))
    if not regions:
        raise ValueError("Mesh has no 2D physical groups.")

    materials = region_material_map(stack, regions)
    stack_materials = dict(stack.materials or {})

    media: dict[str, Medium] = {}
    for region in regions:
        material = materials[region]
        entry = (overrides or {}).get(material, stack_materials.get(material))
        resolved = resolve_stack_material(material, entry, wavelength_um)
        if resolved is None:
            raise ValueError(
                f"Region '{region}' maps to material '{material}' which is "
                "not resolvable from the stack materials or the built-in "
                "database."
            )
        eps_r = resolved.permittivity_scalar
        eps_r = 1.0 if eps_r is None else float(eps_r)
        if eps_r <= 0.0:
            raise ValueError(
                f"Region '{region}' (material '{material}') has relative "
                f"permittivity {eps_r:g}; gfem solves media with Re(eps) > 0 "
                "only, so a metal with a negative permittivity is out of scope."
            )
        loss = resolved.loss_tangent_scalar
        sigma = resolved.conductivity_scalar
        media[region] = Medium(
            eps_r=eps_r,
            tan_delta=0.0 if loss is None else float(loss),
            conductivity=0.0 if sigma is None else float(sigma),
        )
    return media


def _layer_conductivity(
    stack: LayerStack,
    layer_name: str,
    *,
    frequency_hz: float,
    overrides: dict[str, MaterialProperties] | None = None,
) -> float:
    """Conductivity (S/m) of the material of a conductor layer.

    The material entry is looked up as in :func:`region_media`: an override
    by material name wins over the stack's own entry.
    """
    layer = stack.layers[layer_name]
    stack_materials = dict(stack.materials or {})
    entry = (overrides or {}).get(layer.material, stack_materials.get(layer.material))
    resolved = resolve_stack_material(layer.material, entry, C0 / frequency_hz * 1e6)
    sigma = None if resolved is None else resolved.conductivity_scalar
    if sigma is None or sigma <= 0.0:
        return DEFAULT_CONDUCTIVITY_S_PER_M
    return float(sigma)


def edge_conditions(
    mesh: meshio.Mesh | str | Path,
    stack: LayerStack,
    *,
    frequency_hz: float,
    metallic_boundaries: bool = False,
    absorbing_boundary: bool = False,
    conductor_boundary: Literal["conductivity", "pec"] = "conductivity",
    conductor_thickness: Literal["thick", "stack"] = "thick",
    overrides: dict[str, MaterialProperties] | None = None,
) -> list[EdgeCondition]:
    """Map the 1D groups of the mesh to port-mode boundary conditions.

    Groups the mesher writes, and what each becomes:

    - ``<layer>_conductivity``: a finite-conductivity ``"conductor"``, or a
      ``"pec"`` when ``conductor_boundary="pec"``;
    - ``<layer>_pec``: a ``"pec"``;
    - ``absorbing`` (the outer wall): a ``"pec"`` with
      ``metallic_boundaries``, else an ``"absorbing"`` boundary with
      ``absorbing_boundary``, else nothing, which leaves gfem's natural PMC
      (Palace's default for an unconditioned outer wall). The perfect
      conductor wins when both are set;
    - any other group (contacts, interfaces): ignored.

    Args:
        mesh: The shared msh v2.2 mesh (path or loaded meshio mesh).
        stack: Layer stack the mesh was generated from; it gives the
            conductivity and thickness of each conductor layer.
        frequency_hz: Frequency in Hz, where a conductor's material is
            evaluated.
        metallic_boundaries: Put a perfect conductor on the outer wall.
        absorbing_boundary: Put an absorbing boundary on the outer wall.
        conductor_boundary: What finite-conductivity conductor groups become.
        conductor_thickness: ``"thick"`` models a conductor by its skin
            effect alone; ``"stack"`` gives it the layer thickness. gfem
            counts the thickness twice on a one-sided face, which may not be
            what Palace does with the same number (unverified).
        overrides: Optional material-property overrides by material name,
            applied to the conductivity of a conductor layer as
            :func:`region_media` applies them to a region.

    Returns:
        The conditions, in mesh tag order.

    Raises:
        ValueError: When a ``<layer>_conductivity`` group names no layer of
            the stack, or ``conductor_thickness`` is not a known choice.
    """
    if conductor_thickness not in ("thick", "stack"):
        raise ValueError(
            "conductor_thickness must be 'thick' or 'stack', "
            f"got {conductor_thickness!r}."
        )
    frequency = _check_frequency(frequency_hz)
    loaded = _read_mesh(mesh)
    conditions: list[EdgeCondition] = []
    for name, _tag in sorted(group_tags(loaded, dim=1).items(), key=lambda kv: kv[1]):
        if name == WALL_GROUP:
            if metallic_boundaries:
                conditions.append(EdgeCondition("pec", name))
            elif absorbing_boundary:
                conditions.append(EdgeCondition("absorbing", name))
        elif name.endswith(_CONDUCTIVITY_SUFFIX):
            layer_name = name[: -len(_CONDUCTIVITY_SUFFIX)]
            if layer_name not in stack.layers:
                raise ValueError(
                    f"Mesh group '{name}' names layer '{layer_name}', which is "
                    f"not in the stack (layers: {sorted(stack.layers)})."
                )
            if conductor_boundary == "pec":
                conditions.append(EdgeCondition("pec", name))
                continue
            thickness = (
                stack.layers[layer_name].thickness * _UM
                if conductor_thickness == "stack"
                else None
            )
            conditions.append(
                EdgeCondition(
                    "conductor",
                    name,
                    conductivity=_layer_conductivity(
                        stack, layer_name, frequency_hz=frequency, overrides=overrides
                    ),
                    thickness_m=thickness,
                )
            )
        elif name.endswith(_PEC_SUFFIX):
            conditions.append(EdgeCondition("pec", name))
    return conditions


def _to_metres(points: Sequence[tuple[float, float]]) -> NDArray[np.float64]:
    """``(h, v)`` points in um as ``(n, 3)`` points in m on the plane z = 0."""
    array = np.asarray(points, dtype=np.float64) * _UM
    return np.column_stack([array, np.zeros(len(array))])


def impedance_paths_m(
    signal: Extent,
    ground: Extent,
    domain: Extent,
    *,
    loop_clearance_fraction: float = LOOP_CLEARANCE_FRACTION,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The voltage path and current loop of a line, in gfem's coordinates.

    Sized by :func:`gsim.palace.line_impedance.line_impedance_paths`. The
    voltage path keeps the same hair of clearance from each conductor face as
    the Palace postprocessing path, so ``Z_PV`` integrates the same gap. The
    current loop is wider than Palace's (see :data:`LOOP_CLEARANCE_FRACTION`).
    The coordinates become metres on the plane z = 0, and the loop is closed
    by repeating its first corner as ``gfem.characteristic_impedance``
    requires (Palace's own list leaves it open).

    Args:
        signal: ``((h_min, h_max), (v_min, v_max))`` of the signal conductor (um).
        ground: The same for the return conductor.
        domain: The same for the meshed domain.
        loop_clearance_fraction: Fraction of the smallest clearance around
            the signal conductor that the current loop keeps from its
            outline. The voltage path is not affected.

    Returns:
        ``(voltage, loop)``: the ``(2, 3)`` voltage path and the ``(5, 3)``
        closed loop around the signal conductor, in metres.
    """
    paths = line_impedance_paths(signal=signal, ground=ground, domain=domain)
    loop_paths = line_impedance_paths(
        signal=signal,
        ground=ground,
        domain=domain,
        clearance_fraction=loop_clearance_fraction,
    )
    voltage = _to_metres(paths.voltage)
    loop = _to_metres((*loop_paths.current, loop_paths.current[0]))
    return voltage, loop


def _current_loop_m(
    signal: Extent,
    domain: Extent,
    *,
    clearance_fraction: float = LOOP_CLEARANCE_FRACTION,
) -> NDArray[np.float64]:
    """Closed loop around the signal conductor alone, in metres on z = 0.

    The clearance rule of :func:`line_impedance_paths` without a return
    conductor: a fraction of the smallest clearance around the signal
    conductor.
    """
    (s_lo, s_hi), (t_lo, t_hi) = signal
    (d_lo, d_hi), (e_lo, e_hi) = domain
    if not (d_lo < s_lo < s_hi < d_hi and e_lo < t_lo < t_hi < e_hi):
        raise ValueError(
            f"The signal electrode spans {signal} but is not inside the meshed "
            f"domain {domain}, so no path around it can be sampled."
        )
    clearance = clearance_fraction * min(
        t_hi - t_lo, s_lo - d_lo, d_hi - s_hi, t_lo - e_lo, e_hi - t_hi
    )
    corners = (
        (s_lo - clearance, t_lo - clearance),
        (s_hi + clearance, t_lo - clearance),
        (s_hi + clearance, t_hi + clearance),
        (s_lo - clearance, t_hi + clearance),
    )
    return _to_metres((*corners, corners[0]))


def wall_mode_from_impedances(
    z_pv: complex | None, z_pi: complex
) -> tuple[bool | None, str]:
    """Tell the line Mode from the wall Mode by how much power the gap voltage carries.

    ``|Z_PV / Z_PI| = (|V| |I| / |2P|)^2``: a quasi-TEM line Mode has
    ``|V| |I| ~ 2P``, a Mode running between both electrodes together and
    the outer wall has them at one potential and sits many decades under.
    Below :data:`~gsim.palace.line_impedance.MIN_VOLTAGE_POWER_RATIO` the
    Mode is the wall Mode. This is the rule Palace's reading applies; it
    replaces the electrode-current pairing the femwell reading uses, because
    gfem 0.2 exposes no signed electrode current.

    Args:
        z_pv: Power-voltage impedance, or ``None`` when no voltage path
            was declared.
        z_pi: Power-current impedance.

    Returns:
        ``(wall_mode, diagnostic)``: whether the Mode is the wall Mode, or
        ``None`` when the ratio cannot be formed, and the clause saying what
        was measured.
    """
    if z_pv is None:
        return None, "no voltage path, so nothing says which mode this is"
    magnitude_pi = abs(z_pi)
    if (
        not math.isfinite(magnitude_pi)
        or magnitude_pi == 0.0
        or not math.isfinite(abs(z_pv))
    ):
        return None, (
            "the impedances give no finite ratio, so nothing says which mode this is"
        )
    ratio = abs(z_pv) / magnitude_pi
    if ratio < MIN_VOLTAGE_POWER_RATIO:
        return True, (
            f"|Z_PV/Z_PI| = {ratio:.2g} is below {MIN_VOLTAGE_POWER_RATIO:g}: the "
            "gap voltage carries none of the mode's power, so this is the mode "
            "between both electrodes and the outer wall rather than the line mode"
        )
    return False, (
        f"|Z_PV/Z_PI| = {ratio:.2g}: the gap voltage carries the mode's power, "
        "the line mode between the electrodes"
    )


def build_model(
    mesh_path: str | Path,
    stack: LayerStack,
    *,
    frequency_hz: float,
    voltage_path_m: ArrayLike | None = None,
    mode: int = 1,
    metallic_boundaries: bool = False,
    absorbing_boundary: bool = False,
    conductor_boundary: Literal["conductivity", "pec"] = "conductivity",
    conductor_thickness: Literal["thick", "stack"] = "thick",
    overrides: dict[str, MaterialProperties] | None = None,
) -> Any:
    """Build the gfem model of a cross-section mesh with a wave port on its slab.

    The mesh is read as a one-element-thick slab by
    ``gfem.Mesh.from_gmsh_cross_section``; the line's wave port sits on the
    slab's ``front`` face. Media come from :func:`region_media` at
    ``frequency_hz`` and boundaries from :func:`edge_conditions`.

    Args:
        mesh_path: The msh the ``BoundaryModeSim`` pipeline wrote (um).
        stack: Layer stack the mesh was generated from.
        frequency_hz: Frequency the media are evaluated at.
        voltage_path_m: Optional voltage path of the port, ``(n, 3)`` in
            metres, which gives the result its ``Z_PV``.
        mode: Which propagating mode the port carries (1 is the most confined).
        metallic_boundaries: Put a perfect conductor on the outer wall.
        absorbing_boundary: Put an absorbing boundary on the outer wall.
        conductor_boundary: What finite-conductivity conductor groups become.
        conductor_thickness: ``"thick"`` or ``"stack"``, see
            :func:`edge_conditions`.
        overrides: Optional material-property overrides by material name.

    Returns:
        The ``gfem.Model``.

    Raises:
        ImportError: When gfem is not installed.
    """
    gfem = require_gfem()
    loaded = _read_mesh(mesh_path)
    media = region_media(loaded, stack, frequency_hz=frequency_hz, overrides=overrides)
    conditions = edge_conditions(
        loaded,
        stack,
        frequency_hz=frequency_hz,
        metallic_boundaries=metallic_boundaries,
        absorbing_boundary=absorbing_boundary,
        conductor_boundary=conductor_boundary,
        conductor_thickness=conductor_thickness,
        overrides=overrides,
    )
    boundaries: list[Any] = []
    for condition in conditions:
        if condition.kind == "pec":
            boundaries.append(gfem.PEC(condition.group))
        elif condition.kind == "absorbing":
            boundaries.append(gfem.Absorbing(condition.group))
        else:
            boundaries.append(
                gfem.Conductor(
                    condition.group,
                    condition.conductivity,
                    thickness=condition.thickness_m,
                )
            )
    materials = {
        name: gfem.Material(
            eps_r=medium.eps_r,
            tan_delta=medium.tan_delta,
            conductivity=medium.conductivity,
        )
        for name, medium in media.items()
    }
    port = gfem.WavePort(
        PORT_NAME,
        _FRONT_FACE,
        mode=mode,
        voltage_path=None if voltage_path_m is None else np.asarray(voltage_path_m),
    )
    section = gfem.Mesh.from_gmsh_cross_section(Path(mesh_path), "um")
    return gfem.Model(section, materials=materials, boundaries=boundaries, ports=[port])


def solve_line(
    mesh_path: str | Path,
    stack: LayerStack,
    *,
    frequency_hz: float | Sequence[float],
    voltage_path_m: ArrayLike | None = None,
    mode: int = 1,
    metallic_boundaries: bool = False,
    absorbing_boundary: bool = False,
    conductor_boundary: Literal["conductivity", "pec"] = "conductivity",
    conductor_thickness: Literal["thick", "stack"] = "thick",
    overrides: dict[str, MaterialProperties] | None = None,
    order: int = 2,
    options: Any = None,
) -> Any:
    """Solve the port mode of a cross-section mesh with gfem.

    Args:
        mesh_path: The msh the ``BoundaryModeSim`` pipeline wrote (um).
        stack: Layer stack the mesh was generated from.
        frequency_hz: One frequency or a sequence of distinct frequencies (Hz).
            gfem picks the mode at the lowest frequency and follows it.
        voltage_path_m: Optional voltage path of the port, ``(n, 3)`` in
            metres, which gives the result its ``Z_PV``.
        mode: Which propagating mode at the lowest frequency (1 is the most confined).
        metallic_boundaries: Put a perfect conductor on the outer wall.
        absorbing_boundary: Put an absorbing boundary on the outer wall.
        conductor_boundary: What finite-conductivity conductor groups become.
        conductor_thickness: ``"thick"`` or ``"stack"``, see
            :func:`edge_conditions`.
        overrides: Optional material-property overrides by material name.
        order: Element order of the port-mode problem.
        options: A ``gfem.SolverOptions`` to use instead of
            ``SolverOptions(order=order, linear_solver="direct")``.

    Returns:
        The ``gfem.PortModeResult``: ``gamma`` (1/m, ``alpha >= 0``),
        ``n_eff`` (``n' - i n''``), ``Z_PV`` and the field.

    Raises:
        ImportError: When gfem is not installed.
        ValueError: When the media differ between the frequencies (a
            dispersive stack entry): one model carries one set of media, so
            solve each frequency in its own call.
    """
    gfem = require_gfem()
    frequencies = _as_frequencies(frequency_hz)
    loaded = _read_mesh(mesh_path)
    first = region_media(
        loaded, stack, frequency_hz=frequencies[0], overrides=overrides
    )
    for frequency in frequencies[1:]:
        media = region_media(loaded, stack, frequency_hz=frequency, overrides=overrides)
        if media != first:
            raise ValueError(
                "The stack's media differ between the requested frequencies "
                "(a dispersive entry), and one gfem model carries one set of "
                "media: call solve_line once per frequency."
            )

    model = build_model(
        mesh_path,
        stack,
        frequency_hz=frequencies[0],
        voltage_path_m=voltage_path_m,
        mode=mode,
        metallic_boundaries=metallic_boundaries,
        absorbing_boundary=absorbing_boundary,
        conductor_boundary=conductor_boundary,
        conductor_thickness=conductor_thickness,
        overrides=overrides,
    )
    solver_options = options or gfem.SolverOptions(order=order, linear_solver="direct")
    return gfem.solve(
        model, gfem.PortMode(PORT_NAME, frequencies), options=solver_options
    )


def line_impedances(
    result: Any,
    *,
    signal: Extent,
    ground: Extent,
    domain: Extent,
    loop_clearance_fraction: float = LOOP_CLEARANCE_FRACTION,
) -> dict[str, NDArray[np.complex128]]:
    """The three characteristic impedances of a solved line, per frequency.

    Args:
        result: The ``gfem.PortModeResult`` of :func:`solve_line`.
        signal: ``((h_min, h_max), (v_min, v_max))`` of the signal conductor (um).
        ground: The same for the return conductor.
        domain: The same for the meshed domain.
        loop_clearance_fraction: How far the current loop stays from the
            signal conductor, see :func:`impedance_paths_m`.

    Returns:
        ``{"PV": Z_PV, "PI": Z_PI, "VI": Z_VI}``, each a complex array over
        the result's frequencies (ohm). ``Z_PV = |V|^2 / (2 conj(P))`` along
        the gap voltage path, ``Z_PI = 2 P / |I|^2`` around the signal
        conductor, ``Z_VI = V / I`` with the sign that makes ``Re >= 0``.

    Raises:
        ImportError: When gfem is not installed.
    """
    gfem = require_gfem()
    voltage, loop = impedance_paths_m(
        signal, ground, domain, loop_clearance_fraction=loop_clearance_fraction
    )
    measured = {
        "PV": gfem.characteristic_impedance(result, voltage),
        "PI": gfem.characteristic_impedance(result, current_path=loop),
        "VI": gfem.characteristic_impedance(result, voltage, current_path=loop),
    }
    return {
        key: np.asarray(value, dtype=np.complex128) for key, value in measured.items()
    }


def _ohms(z: complex) -> str:
    """An impedance for a message, ``48.714+0.012j ohm``."""
    return f"{z.real:.5g}{z.imag:+.3g}j ohm"


def line_reading(
    result: Any,
    *,
    signal: Conductor,
    return_: Conductor | None = None,
    domain: Extent,
    index: int = 0,
    loop_clearance_fraction: float = LOOP_CLEARANCE_FRACTION,
) -> LineReading:
    """Read one frequency of a solved line as a :class:`~gsim.common.modes.LineReading`.

    ``z0_ohm`` is the power-current impedance ``Z_PI = 2 P / |I|^2`` around
    the signal conductor (negated when its real part is negative, the rule
    :func:`gsim.common.modes.z0_power_current` applies). With a return
    conductor the power-voltage and voltage-current impedances are read
    too, quoted in the diagnostic, and ``wall_mode`` follows from
    :func:`wall_mode_from_impedances`. Without one only ``Z_PI`` is read and
    ``wall_mode`` is ``None``.

    Args:
        result: The ``gfem.PortModeResult`` of :func:`solve_line`.
        signal: The signal electrode; only its ``extent`` is used.
        return_: The return electrode, or ``None``.
        domain: ``((h_min, h_max), (v_min, v_max))`` of the meshed domain (um).
        index: Which frequency of the result.
        loop_clearance_fraction: How far the current loop stays from the
            signal conductor, see :func:`impedance_paths_m`.

    Returns:
        The reading.

    Raises:
        ImportError: When gfem is not installed.
        IndexError: When ``index`` is outside the result's frequencies.
    """
    gfem = require_gfem()
    n_eff = complex(np.asarray(result.n_eff)[index])
    if return_ is None:
        loop = _current_loop_m(
            signal.extent, domain, clearance_fraction=loop_clearance_fraction
        )
        measured = gfem.characteristic_impedance(result, current_path=loop)
        z_pi = complex(np.asarray(measured)[index])
        z_pv: complex | None = None
        z_vi: complex | None = None
    else:
        impedances = line_impedances(
            result,
            signal=signal.extent,
            ground=return_.extent,
            domain=domain,
            loop_clearance_fraction=loop_clearance_fraction,
        )
        z_pi = complex(impedances["PI"][index])
        z_pv = complex(impedances["PV"][index])
        z_vi = complex(impedances["VI"][index])
    if z_pi.real < 0.0:
        z_pi = -z_pi
    wall_mode, clause = wall_mode_from_impedances(z_pv, z_pi)
    quoted = (
        ""
        if z_pv is None or z_vi is None
        else f"; Z_PV = {_ohms(z_pv)}, Z_VI = {_ohms(z_vi)}"
    )
    diagnostic = (
        f"Z0 is Z_PI = 2P/|I|^2 = {_ohms(z_pi)} around the signal conductor "
        f"(Marks-Williams){quoted}; {clause}"
    )
    return LineReading(
        n_eff=n_eff, z0_ohm=z_pi, wall_mode=wall_mode, diagnostic=diagnostic
    )
