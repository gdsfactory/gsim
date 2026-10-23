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


def get_stack_overlay(pdk_name: str, layers: Mapping[str, Any]) -> StackOverlay | None:
    """Return the explicit overlay for *pdk_name*, or ``None``.

    Args:
        pdk_name: Resolved PDK name (e.g. ``"IHP"``).
        layers: Resolved process layers, keyed by layer name.  Each entry must
            expose ``material``, ``zmin`` and ``zmax``.
    """
    if (pdk_name or "").upper() == "IHP":
        return _ihp_overlay(layers)
    return None


__all__ = [
    "IHP_BACKGROUND_DEPTH_UM",
    "StackOverlay",
    "get_stack_overlay",
]
