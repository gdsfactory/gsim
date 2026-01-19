"""Layer stack extraction and parsing for Palace EM simulation.

Usage:
    # From PDK module (preferred, no file needed)
    from gsim.palace.stack import get_stack
    stack = get_stack(pdk=ihp)

    # From YAML file (for custom/tweaked stacks)
    stack = get_stack(yaml_path="my_stack.yaml")

    # Export to YAML for manual editing
    stack.to_yaml("my_stack.yaml")
"""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
import yaml

from gsim.palace.stack.extractor import (
    Layer,
    LayerStack,
    ValidationResult,
    extract_from_pdk,
    extract_layer_stack,
)
from gsim.palace.stack.materials import (
    MATERIALS_DB,
    MaterialProperties,
    get_material_properties,
    material_is_conductor,
    material_is_dielectric,
)
from gsim.palace.stack.visualization import (
    StackLayer,
    parse_layer_stack,
    plot_stack,
    print_stack,
    print_stack_table,
)


def get_stack(
    yaml_path: str | Path | None = None,
    **kwargs,
) -> LayerStack:
    """Get layer stack from active PDK or YAML file.

    Args:
        yaml_path: Path to custom YAML stack file. If None, uses active PDK.
        **kwargs: Additional args passed to extract_layer_stack:
            - substrate_thickness: Thickness below z=0 in um (default: 2.0)
            - air_above: Air box height above top metal in um (default: 200)
            - include_substrate: Include lossy silicon substrate (default: False).
              When False, matches gds2palace "nosub" behavior for RF simulation.

    Returns:
        LayerStack object

    Examples:
        # From active PDK (after PDK.activate()) - no substrate (recommended for RF)
        stack = get_stack()

        # With lossy substrate (for substrate coupling studies)
        stack = get_stack(include_substrate=True)

        # From YAML file
        stack = get_stack(yaml_path="custom_stack.yaml")

        # With custom settings
        stack = get_stack(air_above=300, substrate_thickness=5.0)
    """
    if yaml_path is not None:
        return load_stack_yaml(yaml_path)

    pdk = gf.get_active_pdk()
    if pdk is None:
        raise ValueError("No active PDK found. Call PDK.activate() first.")

    return extract_from_pdk(pdk, **kwargs)


def load_stack_yaml(yaml_path: str | Path) -> LayerStack:
    """Load layer stack from YAML file.

    Args:
        yaml_path: Path to YAML file

    Returns:
        LayerStack object
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Reconstruct LayerStack from dict
    stack = LayerStack(
        pdk_name=data.get("pdk", "unknown"),
        units=data.get("units", "um"),
    )

    # Load materials
    stack.materials = data.get("materials", {})

    # Load layers
    for name, layer_data in data.get("layers", {}).items():
        stack.layers[name] = Layer(
            name=name,
            gds_layer=tuple(layer_data["gds_layer"]),
            zmin=layer_data["zmin"],
            zmax=layer_data["zmax"],
            thickness=layer_data.get(
                "thickness", layer_data["zmax"] - layer_data["zmin"]
            ),
            material=layer_data["material"],
            layer_type=layer_data["type"],
            mesh_resolution=layer_data.get("mesh_resolution", "medium"),
        )

    # Load dielectrics
    stack.dielectrics = data.get("dielectrics", [])

    # Load simulation settings
    stack.simulation = data.get("simulation", {})

    return stack


__all__ = [
    "MATERIALS_DB",
    "Layer",
    "LayerStack",
    "MaterialProperties",
    "StackLayer",
    "ValidationResult",
    "extract_from_pdk",
    "extract_layer_stack",
    "get_material_properties",
    "get_stack",
    "load_stack_yaml",
    "material_is_conductor",
    "material_is_dielectric",
    "parse_layer_stack",
    "plot_stack",
    "print_stack",
    "print_stack_table",
]
