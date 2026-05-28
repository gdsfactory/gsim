"""PDK overlay system for augmenting the built-in materials database.

PDK overlays are small YAML/JSON files that add foundry-specific measurements
to the built-in MATERIALS_DB. For example, ``gsim/pdk_overlays/ihp.yaml``
would contain IHP's measured eps=4.1 for SiO2 at RF, tagged with the process name.

When a PDK overlay is active, its values are merged into MATERIALS_DB at
resolution time, taking priority over the built-in values (but lower
priority than user overrides).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from gsim.common.stack.materials import (
    DispersionModel,
    MaterialProperties,
    ValidityRange,
)


def _parse_tensor_or_scalar(entry: dict, key: str) -> float | list[float] | None:
    """Parse a key that can be scalar or list-of-3 from an overlay dict."""
    val = entry.get(key)
    if val is None:
        return None
    if isinstance(val, list):
        return val
    return float(val)


def load_overlay(path: str | Path) -> dict[str, MaterialProperties]:
    """Load a PDK overlay from a YAML file.

    The YAML file should have the format:

    ```yaml
    materials:
      SiO2:
        permittivity: 4.1
        loss_tangent: 0.0
        validity_frequency: [0, 10e9]
        source: "IHP SG13G2 PDK"
    ```

    Args:
        path: Path to the overlay YAML file.

    Returns:
        Dict of material name -> MaterialProperties from the overlay.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Overlay file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data or "materials" not in data:
        return {}

    overlay_materials: dict[str, MaterialProperties] = {}
    for name, entry in data["materials"].items():
        if not isinstance(entry, dict):
            continue

        mat_type = entry.get("type", "dielectric")
        props_kwargs: dict = {"type": mat_type}

        for key in ("permittivity", "conductivity", "loss_tangent", "permeability"):
            val = _parse_tensor_or_scalar(entry, key)
            if val is not None:
                props_kwargs[key] = val

        if "material_axes" in entry:
            props_kwargs["material_axes"] = entry["material_axes"]

        dispersion_models = []
        if "dispersion_models" in entry:
            for model_data in entry["dispersion_models"]:
                if not isinstance(model_data, dict):
                    continue
                validity = ValidityRange()
                if "validity_frequency" in model_data:
                    vf = model_data["validity_frequency"]
                    validity = ValidityRange(valid_frequency=(vf[0], vf[1]))
                elif "validity_wavelength" in model_data:
                    vw = model_data["validity_wavelength"]
                    validity = ValidityRange(valid_wavelength=(vw[0], vw[1]))

                dm = DispersionModel(
                    type=model_data.get("type", "constant"),
                    permittivity=model_data.get("permittivity"),
                    validity=validity,
                    source=model_data.get("source", ""),
                )
                dispersion_models.append(dm)

        if dispersion_models:
            props_kwargs["dispersion_models"] = dispersion_models

        overlay_materials[name] = MaterialProperties(**props_kwargs)

    return overlay_materials


def merge_overlay(
    overlay: dict[str, MaterialProperties],
    base: dict[str, MaterialProperties] | None = None,
) -> dict[str, MaterialProperties]:
    """Merge a PDK overlay into the materials database.

    Overlay values take priority over base values for matching material names.
    Materials not in the overlay are kept from the base.

    Args:
        overlay: Material properties from the PDK overlay.
        base: Base materials dict. Defaults to MATERIALS_DB.

    Returns:
        Merged materials dict (new dict, does not mutate inputs).
    """
    from gsim.common.stack.materials import MATERIALS_DB

    if base is None:
        base = dict(MATERIALS_DB)

    merged = dict(base)
    for name, props in overlay.items():
        if name in merged:
            existing = merged[name]
            merged[name] = _merge_material(existing, props)
        else:
            merged[name] = props

    return merged


def _merge_material(
    base: MaterialProperties, overlay: MaterialProperties
) -> MaterialProperties:
    """Merge overlay properties into base, keeping unset fields from base."""
    kwargs: dict = {}

    for field_name in (
        "conductivity",
        "permittivity",
        "loss_tangent",
        "permeability",
        "material_axes",
    ):
        overlay_val = getattr(overlay, field_name)
        base_val = getattr(base, field_name)
        kwargs[field_name] = overlay_val if overlay_val is not None else base_val

    if overlay.dispersion_models:
        kwargs["dispersion_models"] = overlay.dispersion_models
    else:
        kwargs["dispersion_models"] = base.dispersion_models

    return MaterialProperties(**kwargs)
