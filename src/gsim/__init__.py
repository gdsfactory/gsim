"""GSIM - GDSFactory+ Simulation Tools.

This package provides APIs and client SDKs for accessing simulation tools
of gdsfactory+.

Currently includes:
    - palace: Palace EM simulation API
    - meep: MEEP photonic FDTD simulation API
"""

from __future__ import annotations

from gsim.gcloud import get_status, wait_for_results

__version__ = "0.0.6"

__all__ = [
    "__version__",
    "get_status",
    "wait_for_results",
]
