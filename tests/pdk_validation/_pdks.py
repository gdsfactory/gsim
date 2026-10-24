"""PDK targets used by the open-PDK Meep validation cases.

The three targets intentionally differ in provenance:

* UBC and Cornerstone are authoritative PDK LayerStacks.
* SiEPIC measured examples use the generic photonic PDK (``gf.gpdk``) with its
  documented 220 nm Si stack; no SiEPIC KLayout PDK is required.

Targets are optional.  Import failures are reported by ``available_targets`` so
the validation suite can skip a target cleanly on a machine without that PDK.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import gdsfactory as gf


@dataclass(frozen=True)
class PdkTarget:
    """One PDK target for validation cases."""

    key: str
    name: str
    pdk: Any
    cells: Any
    dielectrics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def activate(self) -> Any:
        """Activate this target's PDK and invalidate cross-PDK cell caches."""
        # gdsfactory caches cells by name; UBC and Cornerstone both define
        # ``coupler``/``coupler_ring`` with different layer maps.  Clear the
        # cache so a case always builds with its own target's geometry.
        gf.clear_cache()
        self.pdk.activate()
        return self.pdk


def _cspdk_module() -> tuple[Any, Any]:
    """Return ``(module, PDK)`` for an importable Cornerstone Si220 PDK."""
    errors: list[str] = []
    for module_name in ("cspdk.si220", "cspdk.si220.cband"):
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # pragma: no cover - optional dependency
            errors.append(f"{module_name}: {error}")
            continue
        pdk = getattr(module, "PDK", None)
        if pdk is not None:
            return module, pdk
    raise ImportError("; ".join(errors) or "Cornerstone cspdk.si220 PDK unavailable")


def ubc_target() -> PdkTarget:
    """Return the UBC SiEPIC-EBeam target."""
    import ubcpdk

    ubcpdk.PDK.activate()
    return PdkTarget(
        key="ubc",
        name=ubcpdk.PDK.name,
        pdk=ubcpdk.PDK,
        cells=ubcpdk.cells,
    )


def cspdk_target() -> PdkTarget:
    """Return the Cornerstone Si220 target."""
    module, pdk = _cspdk_module()
    pdk.activate()
    return PdkTarget(
        key="cspdk",
        name=pdk.name,
        pdk=pdk,
        cells=module.cells,
    )


def _bulk_dielectrics(pdk: Any, names: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    """Return explicit bulk regions for PDK layers that are not patterned."""
    layer_stack = pdk.get_layer_stack()
    regions: list[dict[str, Any]] = []
    for name in names:
        layer = layer_stack.layers.get(name)
        if layer is None:
            continue
        regions.append(
            {
                "name": name,
                "material": layer.material,
                "zmin": float(layer.zmin),
                "zmax": float(layer.zmin + layer.thickness),
            }
        )
    return tuple(regions)


def generic_target() -> PdkTarget:
    """Return the generic photonic PDK used for SiEPIC measured examples.

    The generic PDK declares a 220 nm Si core, BOX and cladding in its
    LayerStack, but those bulk levels are not patterned by cells.  Pass them as
    explicit dielectric regions so the Meep background is the documented
    photonic stack rather than air.
    """
    gf.gpdk.PDK.activate()
    return PdkTarget(
        key="generic",
        name=gf.gpdk.PDK.name,
        pdk=gf.gpdk.PDK,
        cells=gf.components,
        dielectrics=_bulk_dielectrics(gf.gpdk.PDK, ("substrate", "box", "clad")),
    )


def available_targets() -> dict[str, PdkTarget]:
    """Return every importable target, keyed by its short name."""
    targets: dict[str, PdkTarget] = {}
    for key, factory in (
        ("ubc", ubc_target),
        ("cspdk", cspdk_target),
        ("generic", generic_target),
    ):
        try:
            targets[key] = factory()
        except Exception:
            continue
    return targets


__all__ = [
    "PdkTarget",
    "available_targets",
    "cspdk_target",
    "generic_target",
    "ubc_target",
]
