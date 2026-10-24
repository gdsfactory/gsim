"""Explicit, reviewed per-PDK stack overlays.

The PDK ``LayerStack`` is authoritative for the layers it *declares*.  Foundry
PDKs frequently omit implicit BEOL/cladding material and, in IHP's case, do not
tag interconnect roles.  gsim#230 forbids inferring those silently, so instead
gsim ships a small, explicit, version-controlled overlay per PDK.

Overlays are:

* **explicit** -- a named record in this file, not a runtime heuristic;
* **non-destructive** -- caller-supplied ``dielectrics``/``role_overrides``
  always win over the overlay;
* **recorded** -- the applied overlay name is surfaced on
  :class:`~gsim.common.pdk.stack.ResolvedStack` so configs can be audited.

The overlays are a bridge.  The goal is for PDKs to declare this themselves;
when they do, delete the overlay here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: Depth (um) of the implicit substrate/background below the lowest layer.
IHP_BACKGROUND_DEPTH_UM = 2.0

#: IHP BEOL interconnects are tungsten; metals are aluminium.
#:
#: IHP < 2.0 exposes the bare material names; 2.0 prefixes them with the PDK
#: name (``tungsten_sg13g2_via1``) and uses dedicated contact/MIM-via tokens.
_IHP_VIA_MATERIALS = frozenset({"tungsten", "sg13g2_cont", "sg13g2_vmim"})
_IHP_VIA_MATERIAL_PREFIX = "tungsten_sg13g2_"
_IHP_ILD_MATERIAL = "sio2"

#: Generic SOI photonic background used by Cornerstone Si220.  The PDK
#: LayerStack only declares the device levels, so the implicit BOX/cladding is
#: an explicit, reviewed overlay (with source note) rather than an inference.
_CSPDK_BOX_THICKNESS_UM = 2.0
_CSPDK_CLAD_THICKNESS_UM = 2.0
_CSPDK_BACKGROUND_MATERIAL = "SiO2"


def _region_from_layer(layer: Any, name: str) -> Mapping[str, Any]:
    """Return an explicit dielectric region matching one resolved PDK layer."""
    return {
        "name": name,
        "material": layer.material,
        "zmin": layer.zmin,
        "zmax": layer.zmax,
    }


def _is_ihp_via_material(material: str) -> bool:
    """Return whether *material* is an IHP tungsten interconnect token."""
    token = material.lower().strip()
    return token in _IHP_VIA_MATERIALS or token.startswith(_IHP_VIA_MATERIAL_PREFIX)


@dataclass(frozen=True)
class StackOverlay:
    """Explicit additions to a PDK-resolved stack."""

    name: str
    dielectrics: tuple[Mapping[str, Any], ...] = ()
    role_overrides: Mapping[str, str] = field(default_factory=dict)
    note: str = ""


def _ihp_overlay(layers: Mapping[str, Any]) -> StackOverlay:
    """Build the IHP SG13G2 overlay from the resolved PDK layers.

    * Interconnects (``cont``, ``via*``, ``vmim``, ``topvia*``) are the
      tungsten layers -> ``via``.
    * Passivation is the PDK passivation layer, preserved verbatim.
    * The implicit BEOL/cladding below passivation is modelled as a single
      ``sio2`` region down to ``-IHP_BACKGROUND_DEPTH_UM`` (the equivalent of
      the legacy inferred slab, made explicit).
    """
    passivation_level = layers.get("passiv")  # codespell:ignore passiv
    dielectrics: list[Mapping[str, Any]] = []
    if passivation_level is not None:
        dielectrics.append(
            {
                "name": "ild",
                "material": _IHP_ILD_MATERIAL,
                "zmin": -IHP_BACKGROUND_DEPTH_UM,
                "zmax": passivation_level.zmin,
            }
        )
        dielectrics.append(
            {
                "name": "passivation",
                "material": passivation_level.material,
                "zmin": passivation_level.zmin,
                "zmax": passivation_level.zmax,
            }
        )
    role_overrides = {
        name: "via"
        for name, layer in layers.items()
        if _is_ihp_via_material(layer.material)
    }
    return StackOverlay(
        name="IHP",
        dielectrics=tuple(dielectrics),
        role_overrides=role_overrides,
        note=(
            "IHP SG13G2 does not declare bulk BEOL or interconnect roles; "
            "gsim supplies this reviewed overlay (gsim#230). Upstream it into "
            "the PDK when available."
        ),
    )


def _ubc_overlay(layers: Mapping[str, Any]) -> StackOverlay:
    """Build the UBC SiEPIC-EBeam overlay from the resolved PDK layers.

    UBC declares ``substrate``/``box``/``clad`` on the same WAFER GDS tuple but
    does not tag their roles or expose them as bulk regions.  Preserve their
    authoritative z-extents and make the roles/regions explicit.
    """
    dielectrics = tuple(
        _region_from_layer(layers[name], name)
        for name in ("substrate", "box", "clad")
        if name in layers
    )
    return StackOverlay(
        name="UBC",
        dielectrics=dielectrics,
        role_overrides={"substrate": "substrate"},
        note=(
            "UBC declares substrate/BOX/cladding on the WAFER layer but does "
            "not tag the substrate role or expose bulk dielectric regions; "
            "gsim supplies this reviewed overlay (gsim#230)."
        ),
    )


def _cspdk_overlay(layers: Mapping[str, Any]) -> StackOverlay:
    """Build the Cornerstone Si220 overlay from the resolved PDK layers.

    The Cornerstone LayerStack only declares the device/heater/metal levels; the
    implicit SOI BOX and cladding are not in the PDK object.  Add them as
    explicit regions with a documented source, and mark the interconnect
    levels as conductors.
    """
    dielectrics: list[Mapping[str, Any]] = [
        {
            "name": "box",
            "material": _CSPDK_BACKGROUND_MATERIAL,
            "zmin": -_CSPDK_BOX_THICKNESS_UM,
            "zmax": 0.0,
        },
        {
            "name": "clad",
            "material": _CSPDK_BACKGROUND_MATERIAL,
            "zmin": 0.0,
            "zmax": _CSPDK_CLAD_THICKNESS_UM,
        },
    ]
    role_overrides = {
        name: "conductor" for name in ("heater", "metal") if name in layers
    }
    return StackOverlay(
        name="CSPDK",
        dielectrics=tuple(dielectrics),
        role_overrides=role_overrides,
        note=(
            "Cornerstone Si220 declares only device levels; the implicit "
            f"{_CSPDK_BOX_THICKNESS_UM:g} um BOX and "
            f"{_CSPDK_CLAD_THICKNESS_UM:g} um SiO2 cladding are a reviewed "
            "gsim overlay (gsim#230), not inferred from layer names."
        ),
    )


def get_stack_overlay(pdk_name: str, layers: Mapping[str, Any]) -> StackOverlay | None:
    """Return the explicit overlay for *pdk_name*, or ``None``.

    Args:
        pdk_name: Resolved PDK name (e.g. ``"IHP"`` or ``"cspdk.si220"``).
        layers: Resolved process layers, keyed by layer name.  Each entry must
            expose ``material``, ``zmin`` and ``zmax``.
    """
    normalized = (pdk_name or "").strip().lower()
    if normalized == "ihp":
        return _ihp_overlay(layers)
    if normalized in {"ubc", "ubcpdk"}:
        return _ubc_overlay(layers)
    if (
        normalized == "cspdk"
        or normalized.startswith("cspdk.si220")
        or normalized == "cornerstone"
    ):
        return _cspdk_overlay(layers)
    return None


__all__ = [
    "IHP_BACKGROUND_DEPTH_UM",
    "StackOverlay",
    "get_stack_overlay",
]
