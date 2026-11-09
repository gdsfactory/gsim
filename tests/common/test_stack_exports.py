"""Every name gsim.common.stack exports is importable."""

from __future__ import annotations

import gsim.common.stack as stack


def test_every_exported_name_exists():
    missing = [name for name in stack.__all__ if not hasattr(stack, name)]
    assert missing == []
