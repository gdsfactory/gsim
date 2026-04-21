"""Activate the generic PDK for tests in gsim.meep that build components."""

from __future__ import annotations

import gdsfactory as gf

gf.gpdk.PDK.activate()
