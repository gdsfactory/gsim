"""Superconducting layers: the London surface inductance written as ``Impedance``.

A superconducting conductor layer is written as a lossless
``Boundaries.Impedance`` entry with a surface inductance ``Ls`` [H per square]
in place of the layer's ``Conductivity`` or ``PEC`` entry. This module holds the
formulas, the settings model and the matching of the settings to the mesh groups.

Derivation (lossless London limit, T << Tc). A film occupies ``0 < z < t`` and
carries current along ``x``. With ``E = i w mu0 lambda^2 J`` and ``J = -dH/dz``
the field obeys ``H'' = H / lambda^2``. Let ``u = t / lambda``, ``K_b = H(0)``,
``K_t = -H(t)``; then the tangential electric field on the two faces is::

    E_bottom = i w mu0 lambda [coth(u) K_b + csch(u) K_t]
    E_top    = i w mu0 lambda [csch(u) K_b + coth(u) K_t]

* A field on one side only (``K_t = 0``): ``Ls = mu0 lambda coth(t / lambda)``.
* A field that is antisymmetric across the film (``K_b = K_t = K / 2``, the
  case of a coplanar line without a nearby ground plane): the electric field is
  equal on both faces, ``E = i w mu0 lambda coth(t / 2 lambda) K / 2`` for each
  face carrying ``K / 2``. Per face that is ``mu0 lambda coth(t / 2 lambda)``;
  for one sheet that carries the whole current ``K`` it is
  ``mu0 (lambda / 2) coth(t / 2 lambda)``.
* ``t << lambda``: ``E = i w mu0 (lambda^2 / t) K`` whatever the split.
* ``t >> lambda``: the faces decouple and each has ``mu0 lambda``.

What gsim writes follows from how the conductor is meshed: one two-sided sheet
(planar conductor) gets :func:`sheet_inductance`, each face of a shell or of a
native 2D conductor gets :func:`face_inductance`. See
``docs/palace_conductors.md`` for the comparison with a full London solve.
What Palace does with these entries has not been checked.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)
from scipy.constants import mu_0

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gsim.common.stack import LayerStack

logger = logging.getLogger(__name__)

# Lengths are in um. Anything below this is almost certainly meters.
_MIN_LENGTH_UM = 1e-4
# A London depth above this is not a superconducting film of a quantum chip.
_WARN_LONDON_UM = 10.0


def _lengths_m(london_depth_um: float, thickness_um: float) -> tuple[float, float]:
    """Both lengths in meters, after checking that they are positive.

    Raises:
        ValueError: If a length is zero or negative.
    """
    if not (london_depth_um > 0 and thickness_um > 0):
        raise ValueError(
            "london_depth and thickness must be positive (um), got "
            f"london_depth={london_depth_um!r}, thickness={thickness_um!r}"
        )
    return london_depth_um * 1e-6, thickness_um * 1e-6


def sheet_inductance(london_depth_um: float, thickness_um: float) -> float:
    """Surface inductance of one sheet that carries the whole current.

    ``Ls = mu0 (lambda / 2) coth(t / 2 lambda)`` [H per square]: exact for any
    thickness when the field is antisymmetric across the film, and for
    ``t << lambda`` whatever the field. It tends to ``mu0 lambda^2 / t`` for a
    thin film and to ``mu0 lambda / 2`` for a thick one.

    Args:
        london_depth_um: London penetration depth lambda [um].
        thickness_um: Film thickness t [um].

    Returns:
        The surface inductance [H per square].
    """
    lam, t = _lengths_m(london_depth_um, thickness_um)
    return mu_0 * (lam / 2.0) / math.tanh(t / (2.0 * lam))


def face_inductance(london_depth_um: float, thickness_um: float) -> float:
    """Surface inductance of each of the two faces of a film.

    ``Ls = mu0 lambda coth(t / 2 lambda)`` [H per square]: twice
    :func:`sheet_inductance`, because the current of a face is half of the
    total. It tends to ``2 mu0 lambda^2 / t`` for a thin film and to
    ``mu0 lambda`` for a thick one.

    Args:
        london_depth_um: London penetration depth lambda [um].
        thickness_um: Film thickness t [um].

    Returns:
        The surface inductance of one face [H per square].
    """
    lam, t = _lengths_m(london_depth_um, thickness_um)
    return mu_0 * lam / math.tanh(t / (2.0 * lam))


def one_sided_inductance(london_depth_um: float, thickness_um: float) -> float:
    """Textbook ``Ls = mu0 lambda coth(t / lambda)`` for a field on one side only.

    A documented reference: gsim never writes this value, because the field of
    a coplanar line acts on both faces of the film.

    Args:
        london_depth_um: London penetration depth lambda [um].
        thickness_um: Film thickness t [um].

    Returns:
        The surface inductance [H per square].
    """
    lam, t = _lengths_m(london_depth_um, thickness_um)
    return mu_0 * lam / math.tanh(t / lam)


class SuperconductorConfig(BaseModel):
    """A superconducting conductor layer, by its London penetration depth.

    The layer's surfaces get a lossless ``Boundaries.Impedance`` entry with the
    surface inductance ``Ls`` instead of a ``Conductivity`` or ``PEC`` entry.
    Lengths are in um.

    Attributes:
        london_depth: London penetration depth lambda [um]. Use a value from a
            cited source for the material, film and temperature.
        thickness: Film thickness [um]. ``None`` takes the thickness of the
            layer in the stack.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    london_depth: float = Field(gt=0)
    thickness: float | None = Field(default=None, gt=0)

    @field_validator("london_depth", "thickness")
    @classmethod
    def _in_micrometers(cls, value: float | None, info: ValidationInfo) -> float | None:
        """Refuse a value so small that it must have been given in meters."""
        if value is None:
            return value
        if value < _MIN_LENGTH_UM:
            raise ValueError(
                f"{info.field_name}={value!r} um is below {_MIN_LENGTH_UM:g} um: "
                "values are in um, not m"
            )
        if info.field_name == "london_depth" and value > _WARN_LONDON_UM:
            logger.warning(
                "london_depth=%g um is above %g um; a London depth is "
                "typically tens to a few hundred nm (values are in um).",
                value,
                _WARN_LONDON_UM,
            )
        return value


@dataclass
class SuperconductorUse:
    """How one superconducting layer meets the mesh, for the config and the logs.

    Attributes:
        layer: Stack layer name.
        kind: ``"sheet"`` (3D planar conductor), ``"shell"`` (3D conductor
            faces) or ``"2d"`` (native 2D conductor boundary).
        london_depth: London depth [um].
        thickness: Thickness used for ``Ls`` [um].
        meshed_thickness: Thickness of the layer in the stack [um].
        explicit_thickness: Whether the thickness came from the settings.
        groups: ``(group name, [physical groups])`` pairs the layer owns.
        ls: The ``Ls`` written [H per square].
    """

    layer: str
    kind: str
    london_depth: float
    thickness: float
    meshed_thickness: float
    explicit_thickness: bool
    ls: float
    groups: list[tuple[str, list[int]]] = field(default_factory=list)


def _layer_of_group(name: str) -> str:
    """The layer a conductor surface group belongs to (as the config reads it)."""
    return name.rsplit("_", 1)[0] if name.endswith(("_xy", "_z")) else name


def _as_list(group: int | list[int] | tuple[int, ...]) -> list[int]:
    """Physical groups as a list; a group entry holds one or several."""
    if isinstance(group, (list, tuple)):
        return [int(g) for g in group]
    return [int(group)]


def resolve_superconductors(
    superconductors: Mapping[str, SuperconductorConfig] | None,
    groups: dict,
    stack: LayerStack,
    simulation_type: str,
) -> list[SuperconductorUse]:
    """Match the superconducting layers to the mesh groups and compute ``Ls``.

    Args:
        superconductors: Settings by stack layer name.
        groups: The mesh groups (``conductor_surfaces``, ``pec_surfaces``).
        stack: The layer stack.
        simulation_type: The problem type; a boundary-mode mesh holds native 2D
            conductor boundaries, the others shells and planar sheets.

    Returns:
        One entry per layer, in layer-name order. A layer with no surface in
        the mesh has an empty ``groups`` list.

    Raises:
        ValueError: If a layer is not in the stack or is not a conductor, or its
            stack thickness is zero and no thickness was given.
    """
    uses: list[SuperconductorUse] = []
    for name, cfg in sorted((superconductors or {}).items()):
        layer = stack.layers.get(name)
        if layer is None:
            conductors = sorted(
                n for n, lyr in stack.layers.items() if lyr.layer_type == "conductor"
            )
            raise ValueError(
                f"set_superconductor: layer {name!r} is not in the stack. "
                f"Conductor layers: {conductors}"
            )
        if layer.layer_type != "conductor":
            raise ValueError(
                f"set_superconductor: layer {name!r} is a {layer.layer_type!r} "
                "layer, not a conductor. Vias are meshed as volumes and take no "
                "surface inductance."
            )
        meshed = float(layer.zmax - layer.zmin)
        thickness = cfg.thickness if cfg.thickness is not None else meshed
        if thickness <= 1e-12:
            raise ValueError(
                f"set_superconductor: layer {name!r} has zero thickness in the "
                "stack, so its London sheet inductance needs the film thickness: "
                f"call set_superconductor({name!r}, london_depth=..., "
                "thickness=<um>)."
            )

        found: list[tuple[str, list[int]]] = []
        if simulation_type == "boundarymode":
            kind = "2d"
            for key in ("conductor_surfaces", "pec_surfaces"):
                info = groups.get(key, {}).get(name)
                if info is not None:
                    found.append((name, _as_list(info["phys_group"])))
            ls = face_inductance(cfg.london_depth, thickness)
        else:
            planar = groups.get("pec_surfaces", {}).get(name)
            shell = [
                (key, _as_list(info["phys_group"]))
                for key, info in groups.get("conductor_surfaces", {}).items()
                if key.endswith(("_xy", "_z")) and _layer_of_group(key) == name
            ]
            if shell:
                kind = "shell"
                found.extend(shell)
                ls = face_inductance(cfg.london_depth, thickness)
            else:
                kind = "sheet"
                if planar is not None:
                    found.append((name, _as_list(planar["phys_group"])))
                ls = sheet_inductance(cfg.london_depth, thickness)
            if shell and planar is not None:
                # One layer with both: each group keeps its own model's value.
                found.append((name, _as_list(planar["phys_group"])))
        uses.append(
            SuperconductorUse(
                layer=name,
                kind=kind,
                london_depth=cfg.london_depth,
                thickness=float(thickness),
                meshed_thickness=meshed,
                explicit_thickness=cfg.thickness is not None,
                ls=ls,
                groups=sorted(found),
            )
        )
    return uses


def superconductor_messages(
    uses: list[SuperconductorUse],
    *,
    simulation_type: str,
    wave_ports: bool,
    waveport_boundary: str,
) -> dict[str, str]:
    """The warnings for a set of superconducting layers, by a key that names them.

    The key holds the layer and its settings, so a caller that remembers the
    keys it has logged repeats nothing, and logs again when a setting changes.
    The numbers come from the quasi-static coplanar reference described in
    ``docs/palace_conductors.md``; none of it is a check against Palace.

    Args:
        uses: The result of :func:`resolve_superconductors`.
        simulation_type: The problem type.
        wave_ports: Whether the mesh has wave ports.
        waveport_boundary: The wave-port policy, ``"pec"`` or ``"inherit"``.

    Returns:
        The messages by key, in layer order.
    """
    messages: dict[str, str] = {}
    for use in uses:
        ratio = use.thickness / use.london_depth
        tag = f"{use.layer}|{use.kind}|{use.london_depth:g}|{use.thickness:g}"
        label = f"Superconducting layer {use.layer!r}"
        if not use.groups:
            messages[f"{tag}|nosurface"] = (
                f"{label} has no surface in the mesh (no polygon on the layer, "
                "or a zero-thickness layer of a 2D cross-section), so its "
                "London inductance is not written."
            )
            continue
        if (
            use.explicit_thickness
            and use.kind in ("shell", "2d")
            and not math.isclose(use.thickness, use.meshed_thickness, rel_tol=1e-6)
        ):
            messages[f"{tag}|thickness"] = (
                f"{label}: thickness={use.thickness:g} um differs from the "
                f"thickness {use.meshed_thickness:g} um the layer is meshed with. "
                "Ls uses the first; the geometry keeps the second. Change the "
                "stack layer to make them agree."
            )
        if use.kind == "sheet" and ratio > 2.0:
            messages[f"{tag}|thicksheet"] = (
                f"{label}: planar sheet with t/lambda = {ratio:.3g} "
                f"(t = {use.thickness:g} um, lambda = {use.london_depth:g} um). "
                "The sheet has no side walls and carries the current in one "
                "plane; it is validated for t <= lambda. In the quasi-static "
                "coplanar reference it ran too high above t = 2 lambda "
                "(+3% to +30% at t/lambda = 2, +30% to +51% at 4, about 3 times "
                "at 10; not converged there). Use the 3D shell "
                "(planar_conductors=False) for t >= 4 lambda."
            )
        if use.kind == "shell" and ratio < 1.0:
            messages[f"{tag}|thinshell"] = (
                f"{label}: 3D shell with t/lambda = {ratio:.3g} (t = "
                f"{use.thickness:g} um, lambda = {use.london_depth:g} um). The "
                "two faces share the film's inductance, exact for a field "
                "antisymmetric across the film; in the quasi-static coplanar "
                "reference the hollow shell ran 4% low at t/lambda = 0.5 and 8% "
                "low at 1, where the planar sheet was within about 1%. Prefer "
                "planar_conductors=True for t <= lambda."
            )
        if use.kind == "2d" and ratio < 1.0:
            messages[f"{tag}|thin2d"] = (
                f"{label}: boundary-mode conductor with t/lambda = {ratio:.3g}. "
                "Each face of the conductor hole takes the per-face value, "
                "exact for a field antisymmetric across the film (a coplanar "
                "line without a nearby ground plane). With the field on one "
                "side only (a film over a ground plane) it is up to 2 times "
                "too high for a thin film."
            )
        if use.kind == "shell" and ratio < 1.0 and simulation_type == "eigenmode":
            messages[f"{tag}|spurious"] = (
                f"{label}: an eigenmode run with a thin shell (t/lambda = "
                f"{ratio:.3g}) leaves a hollow interior between its faces, "
                "which may hold spurious slow-wave modes in large ground "
                "planes (not verified). Check the mode fields."
            )
        if wave_ports and simulation_type in ("driven", "eigenmode"):
            if waveport_boundary == "pec":
                messages[f"{tag}|portpec"] = (
                    f"{label}: with waveport_boundary='pec' the port mode "
                    "solve treats the superconducting surfaces as PEC, so the "
                    "port mode's propagation constant misses the kinetic "
                    "inductance that the line carries; port and line differ "
                    "by about sqrt(1 + alpha), alpha = L_kin / L_geo."
                )
            else:
                messages[f"{tag}|portinherit"] = (
                    f"{label}: with waveport_boundary='inherit' the port mode "
                    "solve sees the Impedance entries; whether Palace's port "
                    "eigenproblem includes the Ls term was not verified."
                )
    return messages


def version_warning(version: str | None) -> str | None:
    """The warning for a Palace version that meets superconducting entries.

    Args:
        version: ``"major.minor"`` as ``gsim.palace.runtime.launch_version``
            gives it, or ``None`` when unknown.

    Returns:
        The message, or ``None`` for a Palace of 0.19 or later.
    """
    release: tuple[int, ...] | None
    try:
        release = None if version is None else tuple(int(p) for p in version.split("."))
    except ValueError:
        release = None
    if release is not None and release >= (0, 19):
        return None
    which = "an unknown Palace version" if release is None else f"Palace {version}"
    return (
        f"The configuration has superconducting Impedance entries and meets "
        f"{which}. gsim assumes Palace applies an interior Impedance once, as "
        "a sheet, and without cracking it; this is not verified for any "
        "Palace version, and was not checked for versions before 0.19."
    )


def superconductor_boundaries(
    superconductors: Mapping[str, SuperconductorConfig] | None,
    groups: dict,
    stack: LayerStack,
    simulation_type: str,
) -> tuple[list[dict[str, object]], set[int]]:
    """The ``Boundaries.Impedance`` entries of the superconducting layers.

    One entry ``{"Attributes": [group], "Ls": value}`` per physical group,
    sorted by group name. There is no ``Crack`` key and the values are per
    square already: they do not go through the curve-length division of the
    interface impedance boundaries.

    Args:
        superconductors: Settings by stack layer name.
        groups: The mesh groups.
        stack: The layer stack.
        simulation_type: The problem type.

    Returns:
        The entries, and the physical groups they take over from the
        ``Conductivity`` and ``PEC`` entries.

    Raises:
        ValueError: As :func:`resolve_superconductors`.
    """
    entries: list[tuple[str, dict[str, object]]] = []
    consumed: set[int] = set()
    for use in resolve_superconductors(superconductors, groups, stack, simulation_type):
        for group_name, attributes in use.groups:
            value = use.ls
            if use.kind == "shell" and group_name == use.layer:
                # The planar sheet of a layer that also has shell faces.
                value = sheet_inductance(use.london_depth, use.thickness)
            entries.append(
                (group_name, {"Attributes": sorted(attributes), "Ls": value})
            )
            consumed.update(attributes)
    entries.sort(key=lambda item: (item[0], item[1]["Attributes"]))
    return [entry for _, entry in entries], consumed
