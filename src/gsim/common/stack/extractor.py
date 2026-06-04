"""Extract layer stack from gdsfactory PDK and convert to YAML format.

This module reads a PDK's LayerStack and generates a YAML stack file
that can be used for Palace EM simulation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from gdsfactory.technology import LayerStack as GfLayerStack
from pydantic import BaseModel, ConfigDict, Field

from gsim.common.stack._layer_utils import classify_layer_type, get_gds_layer_tuple
from gsim.common.stack.materials import (
    MATERIALS_DB,
    get_material_properties,
)

logger = logging.getLogger(__name__)


class Layer(BaseModel):
    """Layer information for Palace simulation."""

    model_config = ConfigDict(validate_assignment=True)

    name: str
    gds_layer: tuple[int, int]  # (layer, datatype)
    zmin: float  # um
    zmax: float  # um
    thickness: float  # um
    material: str
    layer_type: Literal["conductor", "via", "dielectric", "substrate"]
    sidewall_angle: float = 0.0  # degrees
    mesh_resolution: str | float = "medium"

    def get_mesh_size(self, base_size: float = 1.0) -> float:
        """Get mesh size in um for this layer.

        Args:
            base_size: Base mesh size for "medium" resolution

        Returns:
            Mesh size in um
        """
        if isinstance(self.mesh_resolution, int | float):
            return float(self.mesh_resolution)

        resolution_map = {
            "fine": base_size * 0.5,
            "medium": base_size,
            "coarse": base_size * 2.0,
        }
        return resolution_map.get(str(self.mesh_resolution), base_size)

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML output."""
        d = {
            "gds_layer": list(self.gds_layer),
            "zmin": round(self.zmin, 4),
            "zmax": round(self.zmax, 4),
            "thickness": round(self.thickness, 4),
            "material": self.material,
            "type": self.layer_type,
            "mesh_resolution": self.mesh_resolution,
        }
        if self.sidewall_angle != 0.0:
            d["sidewall_angle"] = self.sidewall_angle
        return d


class ValidationResult(BaseModel):
    """Result of stack validation."""

    model_config = ConfigDict(validate_assignment=True)

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def __bool__(self) -> bool:
        """Returns True if the validation passed, False otherwise."""
        return self.valid

    def __str__(self) -> str:
        """Returns a formatted string representation of the validation results."""
        lines = []
        if self.valid:
            lines.append("Stack validation: PASSED")
        else:
            lines.append("Stack validation: FAILED")
        if self.errors:
            lines.append("Errors:")
            lines.extend([f"  - {e}" for e in self.errors])
        if self.warnings:
            lines.append("Warnings:")
            lines.extend([f"  - {w}" for w in self.warnings])
        return "\n".join(lines)


class LayerStack(BaseModel):
    """Complete layer stack for Palace simulation."""

    model_config = ConfigDict(validate_assignment=True)

    pdk_name: str = "unknown"
    units: str = "um"
    layers: dict[str, Layer] = Field(default_factory=dict)
    materials: dict[str, dict] = Field(default_factory=dict)
    dielectrics: list[dict] = Field(default_factory=list)
    simulation: dict = Field(default_factory=dict)

    @classmethod
    def from_layer_list(cls, layerList: list[Layer]) -> LayerStack:
        """Build a LayerStack from a list of Layer objects.

        Args:
            layerList: List of Layer definitions to include in the stack.

        Returns:
            A LayerStack with layers/materials/dielectrics assembled from `layerList`.

        Raises:
            ValueError: If `layerList` is None or empty.
        """
        if not layerList:
            raise ValueError("None or empty layer list")
        # Build Layer dict
        layer_dict = {}
        for layer in layerList:
            if layer.name in layer_dict:
                raise ValueError(f"Duplicate layer name: {layer.name}")
            layer_dict[layer.name] = layer
        # Build materials dict
        material_dict = {}
        for layer in layerList:
            material_dict[layer.material] = MATERIALS_DB[layer.material].to_dict()
        # Build dielectric list
        dielectric_list = []
        for layer in layerList:
            if layer.layer_type == "dielectric":
                dielectric = {
                    "name": layer.name,
                    "zmin": layer.zmin,
                    "zmax": layer.zmax,
                    "material": layer.material,
                }
                dielectric_list.append(dielectric)
        # Create layer stack and export to YAML
        layer_stack = LayerStack(
            layers=layer_dict, materials=material_dict, dielectrics=dielectric_list
        )
        return layer_stack

    def validate_stack(self, tolerance: float = 0.001) -> ValidationResult:
        """Validate the layer stack for simulation readiness.

        Checks:
        1. Z-axis continuity: no gaps in dielectric regions
        2. Material coverage: all materials have properties defined
        3. Layer coverage: all conductor/via layers are within dielectric envelope
        4. No negative thicknesses

        Args:
            tolerance: Tolerance for z-coordinate comparisons (um)

        Returns:
            ValidationResult with valid flag, errors, and warnings
        """
        errors = []
        warnings = []

        # 1. Check all materials have required properties
        materials_used = set()

        # Collect materials from layers
        for name, layer in self.layers.items():
            materials_used.add(layer.material)
            if layer.thickness < 0:
                errors.append(
                    f"Layer '{name}' has negative thickness: {layer.thickness}"
                )
            if layer.thickness == 0:
                warnings.append(f"Layer '{name}' has zero thickness")

        # Collect materials from dielectrics
        for d in self.dielectrics:
            materials_used.add(d["material"])

        # Check each material has properties
        for mat in materials_used:
            if mat not in self.materials:
                errors.append(
                    f"Material '{mat}' used but not defined in materials dict"
                )
            else:
                props = self.materials[mat]
                mat_type = props.get("type", "unknown")
                if mat_type == "unknown":
                    warnings.append(f"Material '{mat}' has unknown type")
                elif mat_type == "conductor":
                    if "conductivity" not in props:
                        errors.append(
                            f"Conductor material '{mat}' missing conductivity"
                        )
                elif mat_type == "dielectric" and "permittivity" not in props:
                    errors.append(f"Dielectric material '{mat}' missing permittivity")

        # 2. Check z-axis continuity of dielectrics
        if self.dielectrics:
            sorted_dielectrics = sorted(self.dielectrics, key=lambda d: d["zmin"])

            for i in range(len(sorted_dielectrics) - 1):
                current = sorted_dielectrics[i]
                next_d = sorted_dielectrics[i + 1]

                gap = next_d["zmin"] - current["zmax"]
                if gap > tolerance:
                    errors.append(
                        f"Z-axis gap between '{current['name']}' "
                        f"(zmax={current['zmax']:.4f}) and '{next_d['name']}' "
                        f"(zmin={next_d['zmin']:.4f}): gap={gap:.4f} um"
                    )
                elif gap < -tolerance:
                    warnings.append(
                        f"Z-axis overlap between '{current['name']}' and "
                        f"'{next_d['name']}': overlap={-gap:.4f} um"
                    )

            z_min_dielectric = sorted_dielectrics[0]["zmin"]
            z_max_dielectric = sorted_dielectrics[-1]["zmax"]
        else:
            errors.append("No dielectric regions defined")
            z_min_dielectric = 0
            z_max_dielectric = 0

        # 3. Check all conductor/via layers are within dielectric envelope
        for name, layer in self.layers.items():
            if layer.layer_type in ("conductor", "via"):
                if layer.zmin < z_min_dielectric - tolerance:
                    errors.append(
                        f"Layer '{name}' extends below dielectric envelope: "
                        f"layer zmin={layer.zmin:.4f}, dielectric "
                        f"zmin={z_min_dielectric:.4f}"
                    )
                if layer.zmax > z_max_dielectric + tolerance:
                    errors.append(
                        f"Layer '{name}' extends above dielectric envelope: "
                        f"layer zmax={layer.zmax:.4f}, dielectric "
                        f"zmax={z_max_dielectric:.4f}"
                    )

        # 4. Check commonly expected dielectric regions
        dielectric_names = {d["name"] for d in self.dielectrics}
        if "substrate" not in dielectric_names:
            warnings.append("No 'substrate' dielectric region defined")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    def get_z_range(self) -> tuple[float, float]:
        """Get the full z-range of the stack (substrate bottom to air top)."""
        if not self.dielectrics:
            return (0.0, 0.0)
        z_min = min(d["zmin"] for d in self.dielectrics)
        z_max = max(d["zmax"] for d in self.dielectrics)
        return (z_min, z_max)

    def get_conductor_layers(self) -> dict[str, Layer]:
        """Get all conductor layers."""
        return {
            n: layer
            for n, layer in self.layers.items()
            if layer.layer_type == "conductor"
        }

    def get_via_layers(self) -> dict[str, Layer]:
        """Get all via layers."""
        return {
            n: layer for n, layer in self.layers.items() if layer.layer_type == "via"
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML output."""
        return {
            "version": "1.0",
            "pdk": self.pdk_name,
            "units": self.units,
            "materials": self.materials,
            "layers": {name: layer.to_dict() for name, layer in self.layers.items()},
            "dielectrics": self.dielectrics,
            "simulation": self.simulation,
        }

    def to_yaml(self, path: Path | None = None) -> str:
        """Convert to YAML string and optionally write to file.

        Args:
            path: Optional path to write YAML file

        Returns:
            YAML string
        """
        yaml_str = yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml_str, encoding="utf-8")
            logger.info("Stack written to: %s", path)

        return yaml_str


def extract_layer_stack(
    gf_layer_stack: GfLayerStack,
    pdk_name: str = "unknown",
    substrate_thickness: float = 2.0,
    air_above: float = 0.0,
    air_below: float = 0.0,
    boundary_margin: float = 30.0,
    include_substrate: bool = False,
    add_oxide_dielectric: bool = True,
    add_passivation_dielectric: bool = True,
) -> LayerStack:
    """Extract layer stack from a gdsfactory LayerStack.

    Args:
        gf_layer_stack: gdsfactory LayerStack object
        pdk_name: Name of the PDK (for documentation)
        substrate_thickness: Thickness of substrate in um (default: 2.0)
        air_above: Height of air box above top dielectric in um (default: 0.0)
        air_below: Height of air box below substrate/oxide in um (default: 0.0)
        boundary_margin: Lateral margin from GDS bbox in um (default: 30)
        include_substrate: Whether to include lossy substrate (default: False)
        add_oxide_dielectric: Add synthetic oxide background dielectric region.
            Set False to rely on dielectric regions/layers provided by the PDK.
        add_passivation_dielectric: Add synthetic passivation dielectric above the
            highest stack layer. Set False to rely on PDK-provided dielectrics.

    Returns:
        LayerStack object for Palace simulation
    """
    # Auto-detect: if the PDK has dielectric layers on real GDS layers
    # (not substrate layer 999), it's a photonic stack and the BOX
    # should be preserved as SiO2 rather than merged into the substrate.
    has_patterned_dielectrics = False
    stack = LayerStack(pdk_name=pdk_name)

    z_min_overall = float("inf")
    z_max_overall = float("-inf")
    z_max_conductor = float("-inf")
    passivation_ranges: list[tuple[float, float, str]] = []
    dielectric_layers: list[tuple[float, float, str, str]] = []
    materials_used: set[str] = set()
    # Track dielectric layers on real GDS layers (not placeholder 999).
    # If any exist, the PDK has patterned dielectric geometry (photonic
    # stack) and the BOX should be preserved as SiO2.
    _substrate_gds_layers: set[int] = set()

    def _looks_like_passivation(layer_name: str, material_name: str) -> bool:
        """Heuristic to detect top passivation layers from PDK stacks."""
        lname = layer_name.strip().lower()
        mname = material_name.strip().lower()
        if "passiv" in lname or "nitride" in lname:  # codespell:ignore passiv
            return True

        if mname in {"passive", "sin", "si3n4", "silicon_nitride"}:
            return True

        return "nitride" in mname

    # First pass: discover substrate GDS layers independent of ordering.
    for layer_name, layer_level in gf_layer_stack.layers.items():
        material = layer_level.material or "unknown"
        gds_layer = get_gds_layer_tuple(layer_level) or (0, 0)
        info = getattr(layer_level, "info", None) or {}
        layer_type = info.get("layer_type") or classify_layer_type(layer_name, material)
        if layer_type == "substrate":
            _substrate_gds_layers.add(gds_layer[0])

    for layer_name, layer_level in gf_layer_stack.layers.items():
        zmin = layer_level.zmin if layer_level.zmin is not None else 0.0
        thickness = layer_level.thickness if layer_level.thickness is not None else 0.0
        zmax = zmin + thickness
        material = layer_level.material or "unknown"
        gds_layer = get_gds_layer_tuple(layer_level) or (0, 0)
        # Check for explicit layer_type in LayerLevel.info (PDK override)
        info = getattr(layer_level, "info", None) or {}
        layer_type = info.get("layer_type") or classify_layer_type(layer_name, material)
        sidewall_angle = getattr(layer_level, "sidewall_angle", 0.0) or 0.0

        if layer_type == "substrate" and not include_substrate:
            continue

        layer = Layer(
            name=layer_name,
            gds_layer=gds_layer,
            zmin=zmin,
            zmax=zmax,
            thickness=thickness,
            material=material,
            layer_type=layer_type,
            sidewall_angle=float(sidewall_angle),
        )

        stack.layers[layer_name] = layer
        materials_used.add(material)

        if layer_type != "substrate":
            z_min_overall = min(z_min_overall, zmin)
            z_max_overall = max(z_max_overall, zmax)

        if layer_type in {"conductor", "via"}:
            z_max_conductor = max(z_max_conductor, zmax)
        elif layer_type == "dielectric":
            dielectric_layers.append((zmin, zmax, material, layer_name))
            if _looks_like_passivation(layer_name, material):
                passivation_ranges.append((zmin, zmax, material))

            # Detect patterned dielectric layers: dielectric layers whose GDS
            # layer is NOT a substrate placeholder (e.g., 999).
            if gds_layer[0] not in _substrate_gds_layers:
                has_patterned_dielectrics = True

    if z_min_overall == float("inf"):
        z_min_overall = 0.0
    if z_max_overall == float("-inf"):
        z_max_overall = 0.0
    if z_max_conductor == float("-inf"):
        z_max_conductor = z_max_overall

    z_max_active = z_max_overall
    z_max_conductor_active = z_max_conductor

    for material in materials_used:
        props = get_material_properties(material)
        if props:
            stack.materials[material] = props.to_dict()
        else:
            stack.materials[material] = {
                "type": "unknown",
                "note": "Material not in database, please add properties manually",
            }

    if "air" not in stack.materials:
        stack.materials["air"] = MATERIALS_DB["air"].to_dict()

    sorted_dielectric_layers = sorted(dielectric_layers, key=lambda d: d[0])

    def _is_dielectric_material(material_name: str) -> bool:
        props = get_material_properties(material_name)
        if props is None:
            return True
        conductivity = props.conductivity
        if isinstance(conductivity, list):
            conductivity = conductivity[0] if conductivity else None

        if conductivity is not None and conductivity >= 1e4:
            return False

        if props.permittivity is not None:
            return True

        return bool(props.dispersion_models)

    # Choose a PDK-defined dielectric material for the bulk background slab.
    # Prefer non-passivation layers (e.g. oxide / IMD materials).
    bulk_material = None
    for _zmin, _zmax, material, layer_name in sorted_dielectric_layers:
        if not _is_dielectric_material(material):
            continue
        if not _looks_like_passivation(layer_name, material):
            bulk_material = material
            break
    if bulk_material is None and sorted_dielectric_layers:
        for _zmin, _zmax, material, _layer_name in sorted_dielectric_layers:
            if _is_dielectric_material(material):
                bulk_material = material
                break
    if bulk_material is None and sorted_dielectric_layers:
        bulk_material = sorted_dielectric_layers[0][2]
    if bulk_material is None:
        # Rare fallback when a PDK stack defines no dielectric layers.
        bulk_material = "SiO2"

    if include_substrate:
        if has_patterned_dielectrics:
            # With shaped dielectrics (photonic mode): respect the PDK's
            # actual layer z-ranges so the BOX stays as SiO2 rather than
            # being merged into the substrate. Build dielectrics from the
            # PDK's substrate + box layer z-ranges.
            substrate_zmin = None
            substrate_zmax = None
            box_zmin = None
            box_zmax = None
            for layer in stack.layers.values():
                if layer.layer_type == "substrate":
                    substrate_zmin = min(
                        layer.zmin,
                        substrate_zmin if substrate_zmin is not None else layer.zmin,
                    )
                    substrate_zmax = max(
                        layer.zmax,
                        substrate_zmax if substrate_zmax is not None else layer.zmax,
                    )
                # BOX detection: dielectric layers below z=0 on the WAFER GDS
                # layer that are SiO2-like (covers the buried oxide in SOI)
                if (
                    layer.layer_type == "dielectric"
                    and layer.zmax <= 0.0 + 1e-6
                    and _is_oxide_like(layer.material)
                ):
                    box_zmin = min(
                        layer.zmin, box_zmin if box_zmin is not None else layer.zmin
                    )
                    box_zmax = max(
                        layer.zmax, box_zmax if box_zmax is not None else layer.zmax
                    )

            # Extend substrate below the PDK's substrate z-range
            if substrate_zmin is not None:
                sub_zmin = min(substrate_zmin, -substrate_thickness)
            else:
                sub_zmin = -substrate_thickness

            # Substrate silicon: from extended bottom to BOX bottom (or 0)
            if box_zmin is not None:
                box_zmax_value = box_zmax if box_zmax is not None else 0.0
                stack.dielectrics.append(
                    {
                        "name": "substrate",
                        "zmin": sub_zmin,
                        "zmax": box_zmin,
                        "material": "silicon",
                    }
                )
                # BOX (buried oxide): from BOX bottom to BOX top (or 0)
                stack.dielectrics.append(
                    {
                        "name": "box",
                        "zmin": box_zmin,
                        "zmax": box_zmax_value,
                        "material": "SiO2",
                    }
                )
                oxide_zmin = box_zmax_value
            else:
                # No BOX layer detected: use legacy behavior
                stack.dielectrics.append(
                    {
                        "name": "substrate",
                        "zmin": sub_zmin,
                        "zmax": 0.0,
                        "material": "silicon",
                    }
                )
                oxide_zmin = 0.0
        else:
            # RF mode: single silicon substrate box (legacy)
            stack.dielectrics.append(
                {
                    "name": "substrate",
                    "zmin": -substrate_thickness,
                    "zmax": 0.0,
                    "material": "silicon",
                }
            )
            oxide_zmin = 0.0
        if "silicon" not in stack.materials:
            stack.materials["silicon"] = MATERIALS_DB["silicon"].to_dict()
    else:
        oxide_zmin = -substrate_thickness

    # Prefer explicit PDK passivation placement (e.g. IHP passivation layer).
    # No synthetic passivation is added when the PDK does not define one.
    passivation_candidates: list[tuple[float, float, str]] = []
    for zmin, zmax, material in passivation_ranges:
        if zmax <= z_max_conductor_active + 1e-9:
            continue
        clipped_zmin = max(zmin, z_max_conductor_active)
        clipped_zmax = min(zmax, z_max_active)
        if clipped_zmax > clipped_zmin + 1e-9:
            passivation_candidates.append((clipped_zmin, clipped_zmax, material))

    if passivation_candidates:
        passive_zmin = min(zmin for zmin, _, _ in passivation_candidates)
        passive_zmax = max(zmax for _, zmax, _ in passivation_candidates)
        # Use the material from the lowest passivation segment.
        passivation_material = min(passivation_candidates, key=lambda item: item[0])[2]
    else:
        passive_zmin = None
        passive_zmax = None
        passivation_material = None

    def _ensure_material_defined(material_name: str) -> None:
        if material_name in stack.materials:
            return
        props = get_material_properties(material_name)
        if props:
            stack.materials[material_name] = props.to_dict()
        else:
            stack.materials[material_name] = {
                "type": "unknown",
                "note": "Material not in database, please add properties manually",
            }

    _ensure_material_defined(bulk_material)

    if passivation_material is not None:
        _ensure_material_defined(passivation_material)

    oxide_zmax = (
        z_max_active if passive_zmin is None else min(passive_zmin, z_max_active)
    )

    top_of_stack = z_max_active

    if add_oxide_dielectric and oxide_zmax > oxide_zmin + 1e-9:
        stack.dielectrics.append(
            {
                "name": "oxide",
                "zmin": oxide_zmin,
                "zmax": oxide_zmax,
                "material": bulk_material,
            }
        )
        top_of_stack = oxide_zmax

    added_passivation = False
    if (
        passivation_material is not None
        and passive_zmin is not None
        and passive_zmax is not None
        and passive_zmax > passive_zmin + 1e-9
    ):
        stack.dielectrics.append(
            {
                "name": "passive",
                "zmin": passive_zmin,
                "zmax": passive_zmax,
                "material": passivation_material,
            }
        )
        top_of_stack = passive_zmax
        added_passivation = True

    # Preserve default synthetic cap for patterned dielectric stacks only.
    if (
        not added_passivation
        and add_passivation_dielectric
        and has_patterned_dielectrics
    ):
        passive_thickness = 0.4
        stack.dielectrics.append(
            {
                "name": "passive",
                "zmin": z_max_active,
                "zmax": z_max_active + passive_thickness,
                "material": "passive",
            }
        )
        if "passive" not in stack.materials:
            stack.materials["passive"] = MATERIALS_DB["passive"].to_dict()
        top_of_stack = z_max_active + passive_thickness

    if air_above > 0:
        stack.dielectrics.append(
            {
                "name": "air_box",
                "zmin": top_of_stack,
                "zmax": top_of_stack + air_above,
                "material": "air",
            }
        )

    if air_below > 0:
        stack.dielectrics.append(
            {
                "name": "air_box_bottom",
                "zmin": -substrate_thickness - air_below,
                "zmax": -substrate_thickness,
                "material": "air",
            }
        )

    stack.simulation = {
        "boundary_margin": boundary_margin,
        "air_above": air_above,
        "air_below": air_below,
        "substrate_thickness": substrate_thickness,
        "include_substrate": include_substrate,
        "add_oxide_dielectric": add_oxide_dielectric,
        "add_passivation_dielectric": add_passivation_dielectric,
    }

    return stack


_OXIDE_NAMES = {"sio2", "oxide", "box", "buried_oxide", "si3n4", "nitride"}


def _is_oxide_like(material: str) -> bool:
    """Return True if *material* looks like an oxide/nitride dielectric."""
    m = material.strip().lower()
    if m in _OXIDE_NAMES:
        return True
    # Also check via MATERIAL_ALIASES resolution
    from gsim.common.stack.materials import MATERIAL_ALIASES

    canonical = MATERIAL_ALIASES.get(m, m)
    return canonical.lower() in {"sio2", "si3n4"}


def extract_from_pdk(
    pdk_module,
    output_path: Path | None = None,
    **kwargs,
) -> LayerStack:
    """Extract layer stack from a PDK module or PDK object.

    Args:
        pdk_module: PDK module (e.g., ihp, sky130) or gdsfactory Pdk object
        output_path: Optional path to write YAML file
        **kwargs: Additional arguments passed to extract_layer_stack

    Returns:
        LayerStack object for Palace simulation
    """
    pdk_name = "unknown"

    if hasattr(pdk_module, "name") and isinstance(pdk_module.name, str):
        pdk_name = pdk_module.name
    elif hasattr(pdk_module, "PDK") and hasattr(pdk_module.PDK, "name"):
        pdk_name = pdk_module.PDK.name
    elif hasattr(pdk_module, "__name__"):
        pdk_name = pdk_module.__name__

    gf_layer_stack = None

    if hasattr(pdk_module, "layer_stack") and pdk_module.layer_stack is not None:
        gf_layer_stack = pdk_module.layer_stack
    elif hasattr(pdk_module, "LAYER_STACK"):
        gf_layer_stack = pdk_module.LAYER_STACK
    elif hasattr(pdk_module, "get_layer_stack"):
        gf_layer_stack = pdk_module.get_layer_stack()
    elif hasattr(pdk_module, "PDK") and hasattr(pdk_module.PDK, "layer_stack"):
        gf_layer_stack = pdk_module.PDK.layer_stack

    if gf_layer_stack is None:
        raise ValueError(f"Could not find layer stack in PDK: {pdk_module}")

    stack = extract_layer_stack(gf_layer_stack, pdk_name=pdk_name, **kwargs)

    if output_path:
        stack.to_yaml(output_path)

    return stack
