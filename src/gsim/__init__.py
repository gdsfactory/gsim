"""GSIM - GDSFactory+ Simulation Tools.

This package provides APIs and client SDKs for accessing simulation tools
of gdsfactory+.

Currently includes:
    - palace: Palace EM simulation API
    - meep: MEEP photonic FDTD simulation API
    - fdtd: PDK-native ZapFDTD artifact generation
"""

from __future__ import annotations

from gsim import fdtd as fdtd
from gsim.gcloud import get_status, wait_for_results

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "fdtd",
    "get_status",
    "wait_for_results",
]
