"""Pytest configuration for gsim tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path so tests can import gsim
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
