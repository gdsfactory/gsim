"""Type aliases shared across gsim modules."""

from __future__ import annotations

from typing import Any

from shapely import MultiPolygon, Polygon

type AnyShapelyPolygon = Polygon | MultiPolygon
type GFComponent = Any  # gdsfactory Component (avoid hard import)
