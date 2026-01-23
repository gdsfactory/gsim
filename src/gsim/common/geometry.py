"""Geometry wrapper for gdsfactory components.

This module provides a Geometry class that wraps gdsfactory Components
with computed properties useful for simulation setup.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict


class Geometry(BaseModel):
    """Shared geometry wrapper for gdsfactory Component.

    This class wraps a gdsfactory Component and provides computed properties
    that are useful for simulation setup (bounds, ports, etc.).

    Attributes:
        component: The wrapped gdsfactory Component

    Example:
        >>> from gdsfactory.components import straight
        >>> c = straight(length=100)
        >>> geom = Geometry(component=c)
        >>> print(geom.bounds)
        (0.0, -0.25, 100.0, 0.25)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    component: Any  # gdsfactory Component (Any to avoid import issues)

    @cached_property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get bounding box (xmin, ymin, xmax, ymax) in um."""
        bbox = self.component.dbbox()
        return (bbox.left, bbox.bottom, bbox.right, bbox.top)

    @cached_property
    def ports(self) -> list:
        """Get list of ports on the component."""
        return list(self.component.ports)

    @cached_property
    def port_names(self) -> list[str]:
        """Get list of port names."""
        return [p.name for p in self.component.ports]

    @property
    def width(self) -> float:
        """Get width (x-extent) of geometry in um."""
        xmin, _, xmax, _ = self.bounds
        return xmax - xmin

    @property
    def height(self) -> float:
        """Get height (y-extent) of geometry in um."""
        _, ymin, _, ymax = self.bounds
        return ymax - ymin

    def get_port(self, name: str) -> Any:
        """Get a port by name.

        Args:
            name: Port name to find

        Returns:
            Port object if found, None otherwise
        """
        for port in self.component.ports:
            if port.name == name:
                return port
        return None
