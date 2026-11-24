"""Pytest configuration for gsim tests."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

# Add src directory to path so tests can import gsim
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import the IHP PDK before any test adds a layer to gdsfactory's shared
# layout. Its layer enum fixes layout indices at import, so a raw layer made
# first (e.g. under the generic PDK) shifts every IHP layer by one, and a
# later PDK switch puts IHP marker shapes on Passivdrawing (9/0).
with contextlib.suppress(ImportError):
    import ihp  # noqa: F401

from tests._helpers import restore_generic_pdk  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _generic_pdk_layout():
    """Start and leave every test module on a clean generic PDK layer table.

    Tests switch the active PDK (IHP, the FDTD test PDK) and register raw GDS
    layers in gdsfactory's shared layout, and nothing switches back. With
    pytest-randomly, a later module can then run with duplicated or hijacked
    generic layer slots (see ``restore_generic_pdk``). Restoring per module
    keeps module-scoped fixtures that activate a PDK working for their module.
    """
    restore_generic_pdk()
    yield
    restore_generic_pdk()
