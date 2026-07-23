"""Activate the generic PDK for tests in gsim.meep that build components."""

from __future__ import annotations

import contextlib

import gdsfactory as gf
import pytest


@pytest.fixture(autouse=True)
def activate_generic_pdk():
    """Ensure the generic PDK is active before each test."""
    with contextlib.suppress(AttributeError):
        gf.gpdk.PDK.activate()
