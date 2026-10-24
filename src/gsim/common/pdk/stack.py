"""Resolve a PDK-authoritative simulation layer stack.

Unlike :func:`gsim.common.pdk.resolve_passive_pcell`, which is component- and
geometry-centric for the FDTD backend, this module builds a backend-neutral
description of the *whole* PDK layer stack: every declared process layer with
its authoritative ``zmin``/``thickness``/``material``/``mesh_order``, resolved
RF material properties, and the vertical extent actually drawn by an optional
component.

Design rules (see gsim#230):

* The PDK ``LayerStack`` is authoritative.  Nothing is synthesized: no blanket
  oxide, no 2 um substrate, no collapsed BOX/cladding slab.
* Layer metadata is read from the PDK.  ``LayerLevel.info['mesh_order']`` and
  ``LayerLevel.info['layer_type']`` win over the class defaults when present
  (IHP keeps the real mesh order in ``info``).
* Layer role is *physics-based*: a layer is conductive when its resolved
  material conductivity is significant, dielectric otherwise.  A PDK-declared
  ``info['layer_type']`` overrides that.
* Materials are resolved with the project/gsim registry and a *hard error*
  when an unknown token appears -- no silent ``unknown`` fallback.
* The vertical extent actually drawn by a component is exposed as
  ``drawn_z_range`` for backends to use; it is not applied to the domain by
  this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from gsim.common.pdk.models import LayerResolutionError
from gsim.common.pdk.overlays import StackOverlay, get_stack_overlay
from gsim.common.pdk.resolve import _resolve_layer_stack, _resolve_pdk
from gsim.common.stack._layer_utils import get_gds_layer_tuple
from gsim.common.stack.materials import (
    MaterialProperties,
    get_material_properties,
)

LayerRole = Literal["conductor", "via", "dielectric", "substrate"]

#: Conductivity (S/m) at or above which a material is treated as conductive.
CONDUCTIVITY_THRESHOLD = 1e4

_VALID_ROLES = frozenset({"conductor", "via", "dielectric", "substrate"})


def resolve_material_properties(material: str) -> MaterialProperties:
    """Return PDK material properties, or raise for an unknown token.

    Uses the shared material registry, which normalizes common aliases
    (``si`` -> ``silicon``, ``sio2`` -> ``SiO2``, ``sin`` -> ``Si3N4``, ...).
    Raises rather than returning a silent placeholder.
    """
    props = get_material_properties(material)
    if props is None:
        raise LayerResolutionError(
            f"No material properties found for PDK material token {material!r}. "
            "Add the token to the gsim RF material registry/aliases "
            "(gsim.common.stack.materials.MATERIALS_DB / MATERIAL_ALIASES)."
        )
    return props


def _scalar(value: float | list[float] | None) -> float | None:
    """Return the isotropic scalar of a possibly tensor-valued property."""
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return float(value)


def classify_layer_role(material_properties: MaterialProperties) -> LayerRole:
    """Return ``conductor`` or ``dielectric`` from material properties alone."""
    conductivity = _scalar(material_properties.conductivity)
    if conductivity is not None and conductivity >= CONDUCTIVITY_THRESHOLD:
        return "conductor"
    return "dielectric"


@dataclass(frozen=True)
class ResolvedProcessLayer:
    """One authoritative PDK LayerStack entry."""

    key: str
    declared_name: str | None
    gds_layer: tuple[int, int]
    material: str
    material_properties: Mapping[str, Any]
    zmin: float
    zmax: float
    thickness: float
    sidewall_angle: float
    mesh_order: int
    layer_type: LayerRole
    drawn: bool

    @property
    def z_bounds(self) -> tuple[float, float]:
        """Return ascending vertical bounds, including negative thicknesses."""
        return min(self.zmin, self.zmax), max(self.zmin, self.zmax)


@dataclass(frozen=True)
class ResolvedDielectric:
    """An explicit bulk dielectric region (not synthesized by gsim)."""

    name: str
    material: str
    material_properties: Mapping[str, Any]
    zmin: float
    zmax: float
    source: str = "pdk"


@dataclass(frozen=True)
class ResolvedStack:
    """Backend-neutral, PDK-authoritative simulation stack."""

    pdk_name: str
    layer_stack: Any
    layers: Mapping[str, ResolvedProcessLayer]
    dielectrics: tuple[ResolvedDielectric, ...]
    materials: Mapping[str, Mapping[str, Any]]
    component: Any = None
    wavelength_um: float | None = None
    drawn_z_range: tuple[float, float] | None = None
    undrawn_z_range: tuple[float, float] | None = None
    overlay: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def provenance(self) -> dict[str, Any]:
        """Return an auditable, JSON-serialisable description of this stack."""
        return {
            "pdk_name": self.pdk_name,
            "overlay": self.overlay,
            "wavelength_um": self.wavelength_um,
            "drawn_z_range": list(self.drawn_z_range)
            if self.drawn_z_range is not None
            else None,
            "undrawn_z_range": list(self.undrawn_z_range)
            if self.undrawn_z_range is not None
            else None,
            "layers": {
                key: {
                    "gds_layer": list(layer.gds_layer),
                    "material": layer.material,
                    "zmin": layer.zmin,
                    "zmax": layer.zmax,
                    "thickness": layer.thickness,
                    "sidewall_angle": layer.sidewall_angle,
                    "mesh_order": layer.mesh_order,
                    "layer_type": layer.layer_type,
                    "drawn": layer.drawn,
                }
                for key, layer in self.layers.items()
            },
            "dielectrics": [
                {
                    "name": dielectric.name,
                    "material": dielectric.material,
                    "zmin": dielectric.zmin,
                    "zmax": dielectric.zmax,
                    "source": dielectric.source,
                }
                for dielectric in self.dielectrics
            ],
            "notes": list(self.notes),
        }

    @property
    def stack_hash(self) -> str:
        """Return a deterministic SHA-256 digest of the resolved stack."""
        payload = json.dumps(
            self.provenance(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf8")).hexdigest()


def _drawn_layers(component: Any) -> set[tuple[int, int]] | None:
    """Return the concrete GDS layers a component draws, or ``None``."""
    if component is None:
        return None
    try:
        polygons = component.get_polygons(by="tuple")
    except Exception:
        return None
    drawn: set[tuple[int, int]] = set()
    for key in polygons:
        if isinstance(key, tuple) and len(key) == 2:
            drawn.add((int(key[0]), int(key[1])))
    return drawn


def _mesh_order(layer_level: Any) -> int:
    """Return the authoritative mesh order, preferring ``info['mesh_order']``."""
    info = getattr(layer_level, "info", None) or {}
    if "mesh_order" in info:
        return int(info["mesh_order"])
    return int(getattr(layer_level, "mesh_order", 0) or 0)


def _layer_role_hint(layer_level: Any) -> LayerRole | None:
    """Return a PDK-declared ``info['layer_type']`` role, if any."""
    info = getattr(layer_level, "info", None) or {}
    hint = info.get("layer_type")
    if hint is None:
        return None
    hint = str(hint).lower()
    if hint not in _VALID_ROLES:
        raise LayerResolutionError(
            f"Layer {getattr(layer_level, 'name', '?')!r} declares unsupported "
            f"layer_type {hint!r}; expected one of {sorted(_VALID_ROLES)}."
        )
    return cast(LayerRole, hint)


def _resolve_dielectrics(
    explicit: Iterable[Mapping[str, Any]] | None,
) -> tuple[ResolvedDielectric, ...]:
    """Build explicit dielectric regions; gsim never adds its own."""
    if not explicit:
        return ()
    resolved: list[ResolvedDielectric] = []
    for entry in explicit:
        material = str(entry["material"])
        props = resolve_material_properties(material)
        resolved.append(
            ResolvedDielectric(
                name=str(entry["name"]),
                material=material,
                material_properties=props.to_dict(),
                zmin=float(entry["zmin"]),
                zmax=float(entry["zmax"]),
                source=str(entry.get("source", "explicit")),
            )
        )
    return tuple(resolved)


def _z_range(
    layers: Iterable[ResolvedProcessLayer],
) -> tuple[float, float] | None:
    """Return the combined vertical extent of *layers*, or ``None`` if empty."""
    zmin = min((layer.z_bounds[0] for layer in layers), default=None)
    zmax = max((layer.z_bounds[1] for layer in layers), default=None)
    if zmin is None or zmax is None:
        return None
    return (zmin, zmax)


def resolve_layer_stack(
    component: Any = None,
    *,
    pdk: Any | None = None,
    wavelength_um: float | None = None,
    dielectrics: Iterable[Mapping[str, Any]] | None = None,
    role_overrides: Mapping[str, LayerRole] | None = None,
    use_pdk_overlay: bool = True,
) -> ResolvedStack:
    """Resolve the authoritative PDK layer stack.

    Args:
        component: Optional gdsfactory component.  When given, per-layer
            ``drawn`` flags and the drawn vertical extent are computed from the
            component's polygons; when ``None`` every declared layer counts as
            drawn and the extent spans the whole stack.
        pdk: PDK object or module.  Defaults to the active PDK.
        wavelength_um: Optional wavelength used by callers evaluating
            dispersive optical properties.
        dielectrics: Optional explicit bulk regions.  gsim never inserts any
            implicitly, so the default is an empty stack of dielectrics.
        role_overrides: Optional explicit ``{layer_key: role}`` declarations
            (``"conductor"``/``"via"``/``"dielectric"``/``"substrate"``).  Role
            resolution order is: ``role_overrides`` > explicit PDK overlay >
            PDK ``info['layer_type']`` > property-based classification.  This is
            the supported way to mark interconnect layers (e.g. IHP
            ``via*``/``cont``) without falling back to name heuristics.
        use_pdk_overlay: Apply the reviewed per-PDK overlay (see
            :mod:`gsim.common.pdk.overlays`) for implicit BEOL/cladding and
            interconnect roles.  Caller-supplied ``dielectrics``/
            ``role_overrides`` always win.  Defaults to ``True``.

    Raises:
        LayerResolutionError: When the PDK has no LayerStack, a layer has no
            material, a material token cannot be resolved, or an override names
            an unknown layer or role.
    """
    pdk_object, _ = _resolve_pdk(pdk)
    layer_stack = _resolve_layer_stack(pdk_object)
    pdk_name = str(getattr(pdk_object, "name", "unknown"))

    overrides = dict(role_overrides or {})
    unknown = set(overrides) - set(layer_stack.layers)
    if unknown:
        raise LayerResolutionError(
            f"role_overrides reference unknown layer(s): {sorted(unknown)}."
        )
    for key, override_role in overrides.items():
        if override_role not in _VALID_ROLES:
            raise LayerResolutionError(
                f"role_overrides[{key!r}] = {override_role!r} is not one of "
                f"{sorted(_VALID_ROLES)}."
            )

    drawn = _drawn_layers(component)

    layers: dict[str, ResolvedProcessLayer] = {}
    materials: dict[str, Mapping[str, Any]] = {}
    notes: list[str] = []

    for key, layer_level in layer_stack.layers.items():
        material = getattr(layer_level, "material", None)
        if not material:
            raise LayerResolutionError(
                f"LayerStack entry {key!r} has no material; the PDK stack must "
                "declare a material for every process layer."
            )
        props = resolve_material_properties(material)
        materials.setdefault(material, props.to_dict())

        zmin = float(layer_level.zmin or 0.0)
        thickness = float(layer_level.thickness or 0.0)
        zmax = zmin + thickness

        gds_layer = get_gds_layer_tuple(layer_level) or (0, 0)

        hint = _layer_role_hint(layer_level)
        is_drawn = True if drawn is None else gds_layer in drawn
        role: LayerRole = overrides.get(key) or hint or classify_layer_role(props)

        layers[key] = ResolvedProcessLayer(
            key=key,
            declared_name=getattr(layer_level, "name", None),
            gds_layer=gds_layer,
            material=material,
            material_properties=props.to_dict(),
            zmin=zmin,
            zmax=zmax,
            thickness=thickness,
            sidewall_angle=float(getattr(layer_level, "sidewall_angle", 0.0) or 0.0),
            mesh_order=_mesh_order(layer_level),
            layer_type=role,
            drawn=is_drawn,
        )

    if not layers:
        raise LayerResolutionError(f"PDK {pdk_name!r} declares an empty LayerStack.")

    # Apply the reviewed per-PDK overlay (explicit caller input wins).
    overlay: StackOverlay | None = None
    if use_pdk_overlay:
        overlay = get_stack_overlay(pdk_name, layers)
    if overlay is not None:
        for key, overlay_role in overlay.role_overrides.items():
            if overlay_role not in _VALID_ROLES:
                raise LayerResolutionError(
                    f"Overlay {overlay.name!r} role for {key!r} is invalid: "
                    f"{overlay_role!r}."
                )
            if key in layers and key not in overrides:
                layers[key] = replace(layers[key], layer_type=overlay_role)
        notes.append(f"Applied PDK stack overlay {overlay.name!r}: {overlay.note}")

    merged_dielectrics: dict[str, Mapping[str, Any]] = {
        d["name"]: d for d in (overlay.dielectrics if overlay else ())
    }
    for dielectric in dielectrics or ():
        merged_dielectrics[dielectric["name"]] = dielectric

    drawn_layers = [layer for layer in layers.values() if layer.drawn]
    undrawn_layers = [layer for layer in layers.values() if not layer.drawn]

    if component is not None and not drawn_layers:
        notes.append(
            "The component draws no layer declared in the PDK LayerStack; the "
            "domain will fall back to the full declared stack."
        )

    resolved_dielectrics = _resolve_dielectrics(list(merged_dielectrics.values()))
    # Palace (and other backends) look material properties up by the material
    # token written on each dielectric region.  A caller can legitimately use a
    # token that no process layer uses verbatim (IHP names silicon ``si`` while
    # a notebook may declare ``"silicon"``), so keep those resolved properties
    # in the shared material map instead of dropping them.
    for dielectric in resolved_dielectrics:
        if dielectric.material not in materials:
            materials[dielectric.material] = dielectric.material_properties

    return ResolvedStack(
        pdk_name=pdk_name,
        layer_stack=layer_stack,
        layers=layers,
        dielectrics=resolved_dielectrics,
        materials=materials,
        component=component,
        wavelength_um=wavelength_um,
        drawn_z_range=_z_range(drawn_layers) or _z_range(layers.values()),
        undrawn_z_range=_z_range(undrawn_layers),
        overlay=overlay.name if overlay is not None else None,
        notes=tuple(notes),
    )


__all__ = [
    "CONDUCTIVITY_THRESHOLD",
    "LayerRole",
    "ResolvedDielectric",
    "ResolvedProcessLayer",
    "ResolvedStack",
    "classify_layer_role",
    "resolve_layer_stack",
    "resolve_material_properties",
]
