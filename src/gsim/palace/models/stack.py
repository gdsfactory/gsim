"""Layer stack configuration models for Palace simulations.

This module contains Pydantic models for layer stack definitions:
- MaterialConfig: EM material properties
- LayerConfig: Single layer in the stack
- StackConfig: Complete layer stack configuration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from gsim.palace.stack.extractor import Layer, LayerStack
    from gsim.palace.stack.materials import MaterialProperties


class MaterialConfig(BaseModel):
    """EM properties for a material.

    Attributes:
        type: Material type (conductor, dielectric, or semiconductor)
        conductivity: Conductivity in S/m (for conductors)
        permittivity: Relative permittivity (for dielectrics)
        loss_tangent: Dielectric loss tangent
    """

    model_config = ConfigDict(validate_assignment=True)

    type: Literal["conductor", "dielectric", "semiconductor"]
    conductivity: float | None = Field(default=None, ge=0)
    permittivity: float | None = Field(default=None, ge=1.0)
    loss_tangent: float | None = Field(default=None, ge=0, le=1)

    @classmethod
    def from_legacy(cls, props: MaterialProperties) -> Self:
        """Create from legacy MaterialProperties dataclass."""
        return cls(
            type=props.type,
            conductivity=props.conductivity,
            permittivity=props.permittivity,
            loss_tangent=props.loss_tangent,
        )

    @classmethod
    def conductor(cls, conductivity: float = 5.8e7) -> Self:
        """Create a conductor material."""
        return cls(type="conductor", conductivity=conductivity)

    @classmethod
    def dielectric(cls, permittivity: float, loss_tangent: float = 0.0) -> Self:
        """Create a dielectric material."""
        return cls(
            type="dielectric", permittivity=permittivity, loss_tangent=loss_tangent
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for YAML output."""
        d: dict[str, object] = {"type": self.type}
        if self.conductivity is not None:
            d["conductivity"] = self.conductivity
        if self.permittivity is not None:
            d["permittivity"] = self.permittivity
        if self.loss_tangent is not None:
            d["loss_tangent"] = self.loss_tangent
        return d


class LayerConfig(BaseModel):
    """Layer information for Palace simulation.

    Attributes:
        name: Layer name
        gds_layer: GDS layer tuple (layer, datatype)
        zmin: Bottom z-coordinate in um
        zmax: Top z-coordinate in um
        material: Material name
        layer_type: Type of layer (conductor, via, dielectric, substrate)
        mesh_resolution: Mesh resolution ("fine", "medium", "coarse", or um value)
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    gds_layer: tuple[int, int]
    zmin: float
    zmax: float
    material: str
    layer_type: Literal["conductor", "via", "dielectric", "substrate"]
    mesh_resolution: str | float = "medium"

    @property
    def thickness(self) -> float:
        """Layer thickness in um."""
        return self.zmax - self.zmin

    @classmethod
    def from_legacy(cls, layer: Layer) -> Self:
        """Create from legacy Layer dataclass."""
        return cls(
            name=layer.name,
            gds_layer=layer.gds_layer,
            zmin=layer.zmin,
            zmax=layer.zmax,
            material=layer.material,
            layer_type=layer.layer_type,
            mesh_resolution=layer.mesh_resolution,
        )

    def get_mesh_size(self, base_size: float = 1.0) -> float:
        """Get mesh size in um for this layer."""
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


class StackConfig(BaseModel):
    """Complete layer stack for Palace simulation.

    Attributes:
        pdk_name: Name of the PDK
        units: Length units (default: "um")
        layers: Dictionary of layer configurations
        materials: Dictionary of material properties
        dielectrics: List of dielectric region definitions
        simulation: Additional simulation settings
    """

    model_config = ConfigDict(validate_assignment=True)

    pdk_name: str = "unknown"
    units: str = "um"
    layers: dict[str, LayerConfig] = Field(default_factory=dict)
    materials: dict[str, dict] = Field(default_factory=dict)
    dielectrics: list[dict] = Field(default_factory=list)
    simulation: dict = Field(default_factory=dict)

    @classmethod
    def from_legacy(cls, stack: LayerStack) -> Self:
        """Create from legacy LayerStack dataclass."""
        layers = {
            name: LayerConfig.from_legacy(layer) for name, layer in stack.layers.items()
        }
        return cls(
            pdk_name=stack.pdk_name,
            units=stack.units,
            layers=layers,
            materials=stack.materials,
            dielectrics=stack.dielectrics,
            simulation=stack.simulation,
        )

    def get_z_range(self) -> tuple[float, float]:
        """Get the full z-range of the stack."""
        if not self.dielectrics:
            return (0.0, 0.0)
        z_min = min(d["zmin"] for d in self.dielectrics)
        z_max = max(d["zmax"] for d in self.dielectrics)
        return (z_min, z_max)

    def get_conductor_layers(self) -> dict[str, LayerConfig]:
        """Get all conductor layers."""
        return {
            n: layer
            for n, layer in self.layers.items()
            if layer.layer_type == "conductor"
        }

    def get_via_layers(self) -> dict[str, LayerConfig]:
        """Get all via layers."""
        return {
            n: layer for n, layer in self.layers.items() if layer.layer_type == "via"
        }


__all__ = [
    "LayerConfig",
    "MaterialConfig",
    "StackConfig",
]
