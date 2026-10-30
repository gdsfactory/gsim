"""Pytest configuration for gsim tests."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

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
