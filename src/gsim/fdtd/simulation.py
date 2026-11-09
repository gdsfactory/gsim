"""Historical import path for the FDTD ``Simulation`` workflow."""

from __future__ import annotations

import warnings
from typing import Any

from gsim.fdtd.workflow import Simulation


def __getattr__(name: str) -> Any:
    """Resolve the deprecated ``ArtifactSimulation`` name to ``Simulation``."""
    if name == "ArtifactSimulation":
        warnings.warn(
            "gsim.fdtd.simulation.ArtifactSimulation is deprecated; use "
            "gsim.fdtd.Simulation. It accepts the old flat settings as keyword "
            "arguments only; after construction, change them through the "
            "concern objects (for example simulation.source.port, not "
            "simulation.default_port).",
            DeprecationWarning,
            stacklevel=2,
        )
        return Simulation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Simulation"]
