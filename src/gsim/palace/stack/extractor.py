"""Extract layer stack from gdsfactory PDK and convert to YAML format.

This module reads a PDK's LayerStack and generates a YAML stack file
that can be used for Palace EM simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from gdsfactory.technology import LayerLevel
from gdsfactory.technology import LayerStack as GfLayerStack

from gsim.palace.stack.materials import (
    MATERIALS_DB,
    get_material_properties,
)

logger = logging.getLogger(__name__)


@dataclass
class Layer:
    """Layer information for Palace simulation."""

    name: str
    gds_layer: tuple[int, int]  # (layer, datatype)
    zmin: float  # um
    zmax: float  # um
    thickness: float  # um
    material: str
    layer_type: str  # "conductor", "via", "dielectric", "substrate"

    # Mesh resolution control
    # Options: "fine", "medium", "coarse" or a float in um
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
        return resolution_map.get(self.mesh_resolution, base_size)

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML output."""
        return {
            "gds_layer": list(self.gds_layer),
            "zmin": round(self.zmin, 4),
            "zmax": round(self.zmax, 4),
            "thickness": round(self.thickness, 4),
            "material": self.material,
            "type": self.layer_type,
            "mesh_resolution": self.mesh_resolution,
        }


@dataclass
class ValidationResult:
    """Result of stack validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def __str__(self) -> str:
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


@dataclass
class LayerStack:
    """Complete layer stack for Palace simulation."""

    pdk_name: str
    units: str = "um"
    layers: dict[str, Layer] = field(default_factory=dict)
    materials: dict[str, dict] = field(default_factory=dict)
    dielectrics: list[dict] = field(default_factory=list)
    simulation: dict = field(default_factory=dict)

    def validate(self, tolerance: float = 0.001) -> ValidationResult:
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
            # Check for invalid thickness
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
            # Sort dielectrics by zmin
            sorted_dielectrics = sorted(self.dielectrics, key=lambda d: d["zmin"])

            # Check for gaps between dielectric regions
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
                    # Overlap - this might be intentional
                    warnings.append(
                        f"Z-axis overlap between '{current['name']}' and "
                        f"'{next_d['name']}': overlap={-gap:.4f} um"
                    )

            # Get overall dielectric envelope
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

        # 4. Check we have at least substrate, oxide, and air
        dielectric_names = {d["name"] for d in self.dielectrics}
        if "substrate" not in dielectric_names:
            warnings.append("No 'substrate' dielectric region defined")
        if "air_box" not in dielectric_names:
            warnings.append("No 'air_box' dielectric region defined")

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
            path.write_text(yaml_str)
            logger.info("Stack written to: %s", path)

        return yaml_str


def _get_gds_layer_tuple(layer_level: LayerLevel) -> tuple[int, int]:
    """Extract GDS layer tuple from LayerLevel.

    The layer can be specified in various ways in gdsfactory:
    - tuple: (8, 0)
    - int: 8
    - LogicalLayer with nested layer enum
    - kfactory LayerEnum with .layer and .datatype
    """
    layer: Any = layer_level.layer

    # Handle different layer specifications
    if isinstance(layer, tuple):
        return (int(layer[0]), int(layer[1]))

    if isinstance(layer, int):
        return (int(layer), 0)

    # Handle LogicalLayer (gdsfactory.technology.layer_stack.LogicalLayer)
    # Structure: layer_level.layer (LogicalLayer) -> .layer (Enum) -> .layer/.datatype
    # (int)
    if hasattr(layer, "layer"):
        inner = layer.layer
        # Check if inner has layer/datatype (kfactory/gdsfactory enum)
        if hasattr(inner, "layer") and hasattr(inner, "datatype"):
            return (int(inner.layer), int(inner.datatype))
        # Check if inner itself has the values
        if isinstance(inner, int):
            datatype = getattr(layer, "datatype", 0)
            return (int(inner), int(datatype) if datatype else 0)
        # Recurse one more level if needed
        if hasattr(inner, "layer"):
            innermost = inner.layer
            if isinstance(innermost, int):
                datatype = getattr(inner, "datatype", 0)
                return (int(innermost), int(datatype) if datatype else 0)

    # Direct enum with layer/datatype
    if hasattr(layer, "layer") and hasattr(layer, "datatype"):
        return (int(layer.layer), int(layer.datatype))

    # Enum with tuple value
    if hasattr(layer, "value"):
        if isinstance(layer.value, tuple):
            return (int(layer.value[0]), int(layer.value[1]))
        if isinstance(layer.value, int):
            return (int(layer.value), 0)

    # String format "8/0"
    if isinstance(layer, str):
        if "/" in layer:
            parts = layer.split("/")
            return (int(parts[0]), int(parts[1]))
        return (0, 0)

    # Fallback
    try:
        return (int(layer), 0)
    except (TypeError, ValueError):
        logger.warning("Could not parse layer %s, using (0, 0)", layer)
        return (0, 0)


def _classify_layer_type(layer_name: str, material: str) -> str:
    """Classify a layer as conductor, via, dielectric, or substrate.

    Args:
        layer_name: Name of the layer (e.g., "metal1", "via1", "substrate")
        material: Material name

    Returns:
        Layer type string
    """
    name_lower = layer_name.lower()
    material_lower = material.lower()

    # Check for via layers
    if "via" in name_lower:
        return "via"

    # Check for metal layers
    if any(
        m in name_lower for m in ["metal", "topmetal", "m1", "m2", "m3", "m4", "m5"]
    ):
        return "conductor"

    # Check for substrate
    if "substrate" in name_lower or name_lower == "sub":
        return "substrate"

    # Check by material
    props = get_material_properties(material)
    if props:
        if props.type == "conductor":
            return "conductor"
        if props.type == "semiconductor" and "substrate" in name_lower:
            return "substrate"
        if props.type == "dielectric":
            return "dielectric"

    # Check for common conductor materials
    if material_lower in ["aluminum", "copper", "tungsten", "gold", "al", "cu", "w"]:
        return "conductor"

    # Check for poly
    if "poly" in name_lower:
        return "conductor"

    # Default to dielectric
    return "dielectric"


def extract_layer_stack(
    gf_layer_stack: GfLayerStack,
    pdk_name: str = "unknown",
    substrate_thickness: float = 2.0,
    air_above: float = 200.0,
    boundary_margin: float = 30.0,
    include_substrate: bool = False,
) -> LayerStack:
    """Extract layer stack from a gdsfactory LayerStack.

    Args:
        gf_layer_stack: gdsfactory LayerStack object
        pdk_name: Name of the PDK (for documentation)
        substrate_thickness: Thickness of substrate in um (default: 2.0)
        air_above: Height of air box above top metal in um (default: 200)
        boundary_margin: Lateral margin from GDS bbox in um (default: 30)
        include_substrate: Whether to include lossy substrate (default: False).
            When False, matches gds2palace "nosub" behavior for RF simulation.

    Returns:
        LayerStack object for Palace simulation
    """
    stack = LayerStack(pdk_name=pdk_name)

    # Track z-range for dielectric regions
    z_min_overall = float("inf")
    z_max_overall = float("-inf")

    # Collect materials used
    materials_used: set[str] = set()

    # Extract each layer
    for layer_name, layer_level in gf_layer_stack.layers.items():
        # Get layer properties
        zmin = layer_level.zmin if layer_level.zmin is not None else 0.0
        thickness = layer_level.thickness if layer_level.thickness is not None else 0.0
        zmax = zmin + thickness
        material = layer_level.material if layer_level.material else "unknown"

        # Get GDS layer tuple
        gds_layer = _get_gds_layer_tuple(layer_level)

        # Classify layer type
        layer_type = _classify_layer_type(layer_name, material)

        # Skip substrate layers when include_substrate=False (matches gds2palace
        # "nosub").
        if layer_type == "substrate" and not include_substrate:
            continue

        # Create layer
        layer = Layer(
            name=layer_name,
            gds_layer=gds_layer,
            zmin=zmin,
            zmax=zmax,
            thickness=thickness,
            material=material,
            layer_type=layer_type,
        )

        stack.layers[layer_name] = layer
        materials_used.add(material)

        # Update z-range (only for non-substrate layers to get metal stack extent)
        if layer_type != "substrate":
            z_min_overall = min(z_min_overall, zmin)
            z_max_overall = max(z_max_overall, zmax)

    # Add material properties
    for material in materials_used:
        props = get_material_properties(material)
        if props:
            stack.materials[material] = props.to_dict()
        else:
            # Unknown material - add placeholder
            stack.materials[material] = {
                "type": "unknown",
                "note": "Material not in database, please add properties manually",
            }

    # Always add air for the air box
    if "air" not in stack.materials:
        stack.materials["air"] = MATERIALS_DB["air"].to_dict()

    # Add SiO2 for inter-layer dielectric if not present
    if "SiO2" not in stack.materials:
        stack.materials["SiO2"] = MATERIALS_DB["SiO2"].to_dict()

    # Generate dielectric regions
    if include_substrate:
        # 1. Substrate below z=0 (lossy silicon)
        stack.dielectrics.append(
            {
                "name": "substrate",
                "zmin": -substrate_thickness,
                "zmax": 0.0,
                "material": "silicon",
            }
        )
        if "silicon" not in stack.materials:
            stack.materials["silicon"] = MATERIALS_DB["silicon"].to_dict()
        oxide_zmin = 0.0
    else:
        # No substrate - extend oxide down slightly (matches gds2palace "nosub")
        # This provides a dielectric spacing below the bottom metal
        oxide_zmin = -substrate_thickness

    # 2. Inter-layer dielectric (simplified: one big oxide region)
    # In reality, should fill gaps between metals
    stack.dielectrics.append(
        {
            "name": "oxide",
            "zmin": oxide_zmin,
            "zmax": z_max_overall,
            "material": "SiO2",
        }
    )

    # 3. Passivation layer on top of oxide (matches gds2palace IHP SG13G2)
    passive_thickness = 0.4  # um, from gds2palace XML
    stack.dielectrics.append(
        {
            "name": "passive",
            "zmin": z_max_overall,
            "zmax": z_max_overall + passive_thickness,
            "material": "passive",
        }
    )
    if "passive" not in stack.materials:
        stack.materials["passive"] = MATERIALS_DB["passive"].to_dict()

    # 4. Air above passivation
    stack.dielectrics.append(
        {
            "name": "air_box",
            "zmin": z_max_overall + passive_thickness,
            "zmax": z_max_overall + passive_thickness + air_above,
            "material": "air",
        }
    )

    # Simulation settings
    stack.simulation = {
        "boundary_margin": boundary_margin,
        "air_above": air_above,
        "substrate_thickness": substrate_thickness,
        "include_substrate": include_substrate,
    }

    return stack


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
    # Get PDK name - handle both module and Pdk object
    pdk_name = "unknown"

    # Direct .name attribute (gdsfactory Pdk object)
    if hasattr(pdk_module, "name") and isinstance(pdk_module.name, str):
        pdk_name = pdk_module.name
    # Module with PDK.name
    elif hasattr(pdk_module, "PDK") and hasattr(pdk_module.PDK, "name"):
        pdk_name = pdk_module.PDK.name
    # Module __name__
    elif hasattr(pdk_module, "__name__"):
        pdk_name = pdk_module.__name__

    # Get layer stack from PDK - handle both module and Pdk object
    gf_layer_stack = None

    # Direct layer_stack attribute (gdsfactory Pdk object)
    if hasattr(pdk_module, "layer_stack") and pdk_module.layer_stack is not None:
        gf_layer_stack = pdk_module.layer_stack
    # Module with LAYER_STACK
    elif hasattr(pdk_module, "LAYER_STACK"):
        gf_layer_stack = pdk_module.LAYER_STACK
    # Module with get_layer_stack()
    elif hasattr(pdk_module, "get_layer_stack"):
        gf_layer_stack = pdk_module.get_layer_stack()
    # Module with PDK.layer_stack
    elif hasattr(pdk_module, "PDK") and hasattr(pdk_module.PDK, "layer_stack"):
        gf_layer_stack = pdk_module.PDK.layer_stack

    if gf_layer_stack is None:
        raise ValueError(f"Could not find layer stack in PDK: {pdk_module}")

    # Extract
    stack = extract_layer_stack(gf_layer_stack, pdk_name=pdk_name, **kwargs)

    # Write to file if path provided
    if output_path:
        stack.to_yaml(output_path)

    return stack
