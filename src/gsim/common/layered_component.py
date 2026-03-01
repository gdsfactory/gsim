"""Layered component base class for 3D geometry from GDS + PDK.

Ported from gplugins.common.base_models.component.LayeredComponentBase
to eliminate the gplugins dependency. Combines a gdsfactory Component
with a LayerStack and provides:
  - polygon extraction with DerivedLayer support (via `fuse_polygons`)
  - z-coordinate computation per layer
  - bounding box, center, size properties
  - port extension and padding
"""

from __future__ import annotations

from functools import cached_property
from hashlib import md5

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.pdk import get_layer_name
from gdsfactory.technology import LayerLevel, LayerStack
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, computed_field

from gsim.common.polygon import cleanup_component
from gsim.common.types import AnyShapelyPolygon, GFComponent


class LayeredComponentBase(BaseModel):
    """Base class combining a gdsfactory Component with a LayerStack.

    Provides polygon extraction, z-coordinate management, bounding box,
    port handling, and layer ordering — everything needed for 3D geometry
    construction without any solver dependency.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        allow_inf_nan=False,
        validate_return=True,
    )

    component: GFComponent
    layer_stack: LayerStack
    extend_ports: NonNegativeFloat = 0.0
    port_offset: float = 0.0
    pad_xy_inner: float = 0.0
    pad_xy_outer: NonNegativeFloat = 0.0
    pad_z_inner: float = 0.0
    pad_z_outer: NonNegativeFloat = 0.0
    wafer_layer: tuple[int, int] = (999, 0)
    slice_stack: tuple[int, int | None] = (0, None)

    def __hash__(self) -> int:
        """Returns a stable hash for the model using its JSON representation."""
        if not hasattr(self, "_hash"):
            dump = str.encode(self.model_dump_json())
            object.__setattr__(self, "_hash", int(md5(dump).hexdigest()[:15], 16))
        return self._hash  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Padding helpers
    # ------------------------------------------------------------------

    @property
    def pad_xy(self) -> float:
        """Total XY padding."""
        return self.pad_xy_inner + self.pad_xy_outer

    @property
    def pad_z(self) -> float:
        """Total Z padding."""
        return self.pad_z_inner + self.pad_z_outer

    # ------------------------------------------------------------------
    # GDS component with extended ports + wafer layer
    # ------------------------------------------------------------------

    @cached_property
    def gds_component(self) -> GFComponent:
        """Build GDS component with extended ports and wafer layer."""
        c = Component()
        c << gf.components.extend_ports(
            self.component, length=self.extend_ports + self.pad_xy
        )
        (xmin, ymin), (xmax, ymax) = self._gds_bbox
        delta = self.pad_xy_outer
        points = np.array(
            [
                [xmin - delta, ymin - delta],
                [xmax + delta, ymin - delta],
                [xmax + delta, ymax + delta],
                [xmin - delta, ymax + delta],
            ]
        )
        c.add_polygon(points, layer=self.wafer_layer)
        c.add_ports(self.ports)
        c.copy_child_info(self.component)
        return c

    @cached_property
    def _gds_bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Returns the 2D bounding box of the GDS component including padding."""
        c = gf.components.extend_ports(
            self.component, length=self.extend_ports + self.pad_xy_inner
        )
        unchanged = np.isclose(
            np.abs(np.round(c.bbox_np() - self.component.bbox_np(), 3)), 0
        )
        bbox = (
            c.bbox_np() + unchanged * np.array([[-1, -1], [1, 1]]) * self.pad_xy_inner
        )
        return tuple(map(tuple, bbox))

    # ------------------------------------------------------------------
    # Ports
    # ------------------------------------------------------------------

    @cached_property
    def ports(self) -> tuple[gf.Port, ...]:
        """Component ports offset by extension and padding."""
        p = tuple(
            p.copy_polar(
                self.extend_ports + self.pad_xy_inner - self.port_offset,
                orientation=p.orientation,
            )
            for p in self.component.ports
        )
        for pi, po in zip(self.component.ports, p, strict=True):
            po.angle = pi.angle
        return p

    @cached_property
    def port_names(self) -> tuple[str, ...]:
        """Names of all ports."""
        return tuple(p.name for p in self.ports if p.name is not None)

    @cached_property
    def port_centers(self) -> tuple[tuple[float, float, float], ...]:
        """3D center coordinates of all ports."""
        return tuple(self.get_port_center(p) for p in self.ports)

    def get_port_center(self, port: gf.Port) -> tuple[float, float, float]:
        """Return 3D center of a port averaged over its layers."""
        layers = self.get_port_layers(port)
        return (
            *port.dcenter,
            np.mean([self.get_layer_center(layer)[2] for layer in layers]),
        )

    def get_port_layers(self, port: gf.Port) -> list[str]:
        """Return layer names associated with a port."""
        layer_name = get_layer_name(port.layer)
        if "_intent" in layer_name:
            layer_name = layer_name.replace("_intent", "")

        derived_layers = []
        for l_name, level in self.layer_stack.layers.items():
            if layer_name in str(level.layer):
                derived_layers.append(l_name)
        return derived_layers

    # ------------------------------------------------------------------
    # Polygon extraction (uses gsim.common.polygon)
    # ------------------------------------------------------------------

    @computed_field
    @cached_property
    def polygons(self) -> dict[str, AnyShapelyPolygon]:
        """Cleaned-up Shapely polygons keyed by layer name."""
        return cleanup_component(
            self.gds_component,
            self.layer_stack,
            round_tol=3,
            simplify_tol=1e-3,
        )

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    @cached_property
    def geometry_layers(self) -> dict[str, LayerLevel]:
        """Non-empty layers with valid z-coordinates, sorted by height."""
        layers = {
            k: v
            for k, v in self.layer_stack.layers.items()
            if not self.polygons[k].is_empty
            and v.zmin is not None
            and v.thickness is not None
        }
        layers = dict(sorted(layers.items(), key=lambda x: x[1].zmin + x[1].thickness))
        return dict(tuple(layers.items())[slice(*self.slice_stack)])

    @cached_property
    def bottom_layer(self) -> str:
        """Name of the lowest layer in the stack."""
        return min(
            self.geometry_layers.items(),
            key=lambda item: min(item[1].zmin, item[1].zmin + item[1].thickness),
        )[0]

    @cached_property
    def top_layer(self) -> str:
        """Name of the highest layer in the stack."""
        return max(
            self.geometry_layers.items(),
            key=lambda item: max(item[1].zmin, item[1].zmin + item[1].thickness),
        )[0]

    @cached_property
    def device_layers(self) -> tuple[str, ...]:
        """Layer names present in the component's GDS layers."""
        return tuple(
            k
            for k, v in self.layer_stack.layers.items()
            if v.layer in self.component.layers
        )

    # ------------------------------------------------------------------
    # Bounding box / geometry
    # ------------------------------------------------------------------

    @property
    def xmin(self) -> float:
        """Minimum x coordinate."""
        return self._gds_bbox[0][0]

    @property
    def xmax(self) -> float:
        """Maximum x coordinate."""
        return self._gds_bbox[1][0]

    @property
    def ymin(self) -> float:
        """Minimum y coordinate."""
        return self._gds_bbox[0][1]

    @property
    def ymax(self) -> float:
        """Maximum y coordinate."""
        return self._gds_bbox[1][1]

    @cached_property
    def zmin(self) -> float:
        """Minimum z coordinate including inner padding."""
        return (
            min(
                min(layer.zmin, layer.zmin + layer.thickness)
                for layer in self.geometry_layers.values()
            )
            - self.pad_z_inner
        )

    @cached_property
    def zmax(self) -> float:
        """Maximum z coordinate including inner padding."""
        return (
            max(
                max(layer.zmin, layer.zmin + layer.thickness)
                for layer in self.geometry_layers.values()
            )
            + self.pad_z_inner
        )

    @property
    def bbox(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """3D bounding box as ((xmin, ymin, zmin), (xmax, ymax, zmax))."""
        return (*self._gds_bbox[0], self.zmin), (*self._gds_bbox[1], self.zmax)

    @property
    def center(self) -> tuple[float, float, float]:
        """3D center of the bounding box."""
        return tuple(np.mean(self.bbox, axis=0))

    @property
    def size(self) -> tuple[float, float, float]:
        """3D size of the bounding box (dx, dy, dz)."""
        return tuple(np.squeeze(np.diff(self.bbox, axis=0)))  # ty: ignore[invalid-return-type]

    def get_layer_bbox(
        self, layername: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return 3D bounding box of a single layer."""
        layer = self.layer_stack[layername]
        bounds_xy = self.polygons[layername].bounds
        zmin, zmax = sorted([layer.zmin, layer.zmin + layer.thickness])

        if layername == self.bottom_layer:
            zmin -= self.pad_z
        if layername == self.top_layer:
            zmax += self.pad_z

        return (*bounds_xy[:2], zmin), (*bounds_xy[2:], zmax)

    def get_layer_center(self, layername: str) -> tuple[float, float, float]:
        """Return 3D center of a single layer."""
        bbox = self.get_layer_bbox(layername)
        return tuple(np.mean(bbox, axis=0))
