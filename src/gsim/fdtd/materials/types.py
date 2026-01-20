"""Type definitions for FDTD materials and simulations.

This module provides type annotations and validators for Tidy3D types.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
import tidy3d as td
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import AfterValidator


def validate_medium(v: Any) -> td.AbstractMedium:
    """Validate that input is a Tidy3D medium."""
    assert isinstance(v, td.AbstractMedium), (
        f"Input should be a tidy3d medium, but got {type(v)} instead"
    )
    return v


# Type alias for S-parameters dictionary
Sparameters = dict[str, np.ndarray[Any, Any]]

# Annotated type for Tidy3D medium with validation and serialization
Tidy3DMedium = Annotated[
    Any,
    AfterValidator(validate_medium),
    PlainSerializer(lambda x: dict(x), when_used="json"),
]

# Type for Tidy3D element mapping
Tidy3DElementMapping = tuple[
    tuple[
        tuple[tuple[str, int], tuple[str, int]], tuple[tuple[str, int, tuple[str, int]]]
    ],
    ...,
]
